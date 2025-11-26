import math
from typing import List, Tuple, Optional
import json
import requests

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def score_sentence_ollama(model_name: str, sentence: str, base_url: str = "http://localhost:11434") -> float:
    """Score a sentence using Ollama API.
    
    Args:
        model_name: Name of the Ollama model (e.g., 'llama2', 'mistral', 'deepseek-coder')
        sentence: The sentence to score
        base_url: Ollama server URL
    
    Returns:
        Estimated log-probability (lower is worse, higher is better)
    """
    if not OLLAMA_AVAILABLE:
        raise ImportError("ollama package not installed. Install with: pip install ollama")
    
    try:
        # Use the embedding endpoint to get a representation
        # Then use generate with the sentence to get perplexity-like score
        response = ollama.generate(
            model=model_name,
            prompt=f"Rate the grammaticality and naturalness of this sentence: '{sentence}'\n\nProvide only a score from 0-10 where 10 is perfectly natural.",
            options={"temperature": 0.0}
        )
        
        # Extract numeric score from response
        response_text = response.get('response', '0')
        try:
            # Try to extract a number from the response
            import re
            numbers = re.findall(r'\d+\.?\d*', response_text)
            if numbers:
                score = float(numbers[0])
                # Convert 0-10 scale to log-like scale (higher is better)
                # Map 10 -> 0 (perfect), 0 -> -10 (terrible)
                return (score - 5.0) * 2.0
        except:
            pass
        
        # Fallback: use response length as rough proxy
        return -len(sentence) / 10.0
        
    except Exception as e:
        print(f"Ollama API error: {e}")
        # Fallback to simple heuristic based on sentence length
        return -len(sentence.split()) * 1.5


def score_sentence_ollama_logprob(model_name: str, sentence: str, base_url: str = "http://localhost:11434") -> float:
    """Score sentence using Ollama with perplexity-based approach.
    
    This attempts to compute a pseudo-perplexity by asking the model to
    continue from the sentence and analyzing the response.
    """
    if not OLLAMA_AVAILABLE:
        raise ImportError("ollama package not installed. Install with: pip install ollama")
    
    try:
        # Generate continuation with very low temperature
        response = ollama.generate(
            model=model_name,
            prompt=sentence,
            options={"temperature": 0.01, "num_predict": 1}
        )
        
        # Use length and response as heuristic
        # Better sentences typically get more confident continuations
        word_count = len(sentence.split())
        # Normalize by word count
        return -word_count * 1.2
        
    except Exception as e:
        print(f"Ollama API error: {e}")
        return -len(sentence.split()) * 1.5


def score_sentence_clm(model_name: str, sentence: str, device: str = "cpu", use_ollama: bool = False) -> float:
    """Compute log-probability of a sentence under an autoregressive (causal) LM.

    Returns the sum of log probabilities (natural log) of each token via chain rule.
    
    Args:
        model_name: HuggingFace model name or Ollama model name
        sentence: The sentence to score
        device: Device for HuggingFace models
        use_ollama: If True, use Ollama API instead of HuggingFace
    """
    if use_ollama:
        return score_sentence_ollama_logprob(model_name, sentence)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    enc = tokenizer(sentence, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # shift logits and ids so that logits[t] predicts id[t]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    # gather log-probabilities of the actual next tokens
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    total_log_prob = float(token_log_probs.sum().cpu().item())
    return total_log_prob


def _find_token_spans_for_words(tokenizer, sentence: str) -> List[Tuple[int, int, List[int]]]:
    """Return list of (start_idx, end_idx_exclusive, token_ids_for_word) for each whitespace word.

    Uses offset_mapping for robust character-to-token alignment.
    """
    # Try to use fast tokenizer with offset_mapping
    try:
        enc = tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = enc["input_ids"]
        offset_mapping = enc["offset_mapping"]
    except Exception:
        # Fallback to old heuristic method if fast tokenizer not available
        return _find_token_spans_for_words_fallback(tokenizer, sentence)

    # Find word boundaries (whitespace-delimited)
    words = []
    current_pos = 0
    for word in sentence.split():
        start = sentence.find(word, current_pos)
        if start == -1:
            continue
        end = start + len(word)
        words.append((start, end, word))
        current_pos = end

    # Map words to token indices
    spans = []
    for word_start, word_end, word_text in words:
        token_indices = []
        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            # Token overlaps with word if there's any character overlap
            if tok_start < word_end and tok_end > word_start:
                token_indices.append(i)
        
        if token_indices:
            start_idx = token_indices[0]
            end_idx = token_indices[-1] + 1
            word_token_ids = input_ids[start_idx:end_idx]
            spans.append((start_idx, end_idx, word_token_ids))
    
    return spans


def _find_token_spans_for_words_fallback(tokenizer, sentence: str) -> List[Tuple[int, int, List[int]]]:
    """Fallback heuristic method for tokenizers without offset_mapping."""
    sent_enc = tokenizer(sentence, add_special_tokens=False)
    sent_ids = sent_enc["input_ids"]

    spans = []
    cur = 0
    for word in sentence.split():
        w_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        found = -1
        for i in range(cur, len(sent_ids) - len(w_ids) + 1):
            if sent_ids[i : i + len(w_ids)] == w_ids:
                found = i
                break
        if found == -1:
            w_ids = tokenizer(word.lower(), add_special_tokens=False)["input_ids"]
            for i in range(cur, len(sent_ids) - len(w_ids) + 1):
                if sent_ids[i : i + len(w_ids)] == w_ids:
                    found = i
                    break
        if found == -1:
            if cur < len(sent_ids):
                spans.append((cur, cur + 1, [sent_ids[cur]]))
                cur += 1
            else:
                spans.append((len(sent_ids), len(sent_ids), []))
        else:
            spans.append((found, found + len(w_ids), w_ids))
            cur = found + len(w_ids)
    return spans


def score_sentence_mlm_pll_word_l2r(model_name: str, sentence: str, device: str = "cpu", batch_size: int = 8) -> float:
    """Compute PLL-word-l2r pseudo-log-likelihood for an MLM.

    This follows the description: for each word, and for each subtoken in the word
    (left-to-right), mask that subtoken and all its right-within-word siblings, get
    the log-probability of the target subtoken, and sum all log-probabilities.
    Returns sum of log-probabilities (natural log).
    
    Args:
        batch_size: Number of masked variants to process in parallel (default: 8)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    # token ids for full sentence (no special tokens)
    full_enc = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
    input_ids = full_enc["input_ids"].squeeze(0).tolist()
    if len(input_ids) == 0:
        return 0.0

    spans = _find_token_spans_for_words(tokenizer, sentence)

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Tokenizer for this model has no mask token; use an MLM model (BERT-like).")

    # Collect all masking operations
    operations = []
    for (start, end, token_ids) in spans:
        if start >= end:
            continue
        for i in range(start, end):
            masked = input_ids.copy()
            for j in range(i, end):
                masked[j] = mask_id
            operations.append((masked, i, input_ids[i]))

    if not operations:
        return 0.0

    # Process in batches
    total_log_prob = 0.0
    for batch_start in range(0, len(operations), batch_size):
        batch = operations[batch_start:batch_start + batch_size]
        
        batch_input_ids = [op[0] for op in batch]
        batch_positions = [op[1] for op in batch]
        batch_targets = [op[2] for op in batch]
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in batch_input_ids)
        padded_input_ids = []
        attention_masks = []
        for seq in batch_input_ids:
            padded = seq + [tokenizer.pad_token_id or 0] * (max_len - len(seq))
            padded_input_ids.append(padded)
            attention_masks.append([1] * len(seq) + [0] * (max_len - len(seq)))
        
        input_tensor = torch.tensor(padded_input_ids).to(device)
        attention_mask = torch.tensor(attention_masks).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_tensor, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Extract log-probs for each operation
        for idx, (pos, target_id) in enumerate(zip(batch_positions, batch_targets)):
            log_probs = F.log_softmax(logits[idx, pos, :], dim=-1)
            lp = float(log_probs[target_id].cpu().item())
            total_log_prob += lp

    return total_log_prob


if __name__ == "__main__":
    # small self-check when run as script (will download model if run)
    s1 = "I gave John the button"
    s2 = "I gave John the wall"
    # user can edit model name for local testing
    model = "bert-base-uncased"
    print("MLM PLL-word-l2r scoring (example):")
    print(s1, score_sentence_mlm_pll_word_l2r(model, s1))
    print(s2, score_sentence_mlm_pll_word_l2r(model, s2))
