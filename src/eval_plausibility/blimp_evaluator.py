"""BLIMP-format evaluation harness with aggregated metrics and confusion analysis."""
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from tqdm import tqdm

from .eval import score_sentence_clm, score_sentence_mlm_pll_word_l2r


class BLIMPEvaluator:
    """Evaluator for BLIMP-format minimal pairs with category-wise reporting."""
    
    def __init__(self, model_name: str, model_type: str, device: str = "cpu", batch_size: int = 8):
        """
        Args:
            model_name: HuggingFace model identifier or Ollama model tag
            model_type: Either 'clm' or 'mlm'
            device: 'cpu' or 'cuda'
            batch_size: Batch size for MLM scoring (ignored for CLM)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.batch_size = batch_size
        self.use_ollama = False
        
        # Check if it's likely an Ollama model
        is_ollama_pattern = ":" in model_name or "llama" in model_name.lower() or "mistral" in model_name.lower() or "qwen" in model_name.lower() or "deepseek" in model_name.lower()
        
        if is_ollama_pattern:
            try:
                import ollama
                models = ollama.list()
                if isinstance(models, dict):
                    available_models = [m.get('name', '') for m in models.get('models', [])]
                else:
                    available_models = [getattr(m, 'name', getattr(m, 'model', '')) for m in getattr(models, 'models', [])]
                
                if any(model_name == m or m.startswith(model_name + ':') for m in available_models):
                    self.use_ollama = True
                    print(f"Detected Ollama model: {model_name}")
            except ImportError:
                pass
            except Exception as e:
                print(f"Warning: Failed to check Ollama availability: {e}")

        if not self.use_ollama:
            # Cache model and tokenizer for HuggingFace models
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if model_type == "clm":
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            else:
                self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
            self.model.eval()
            self.torch = torch
        else:
            self.model = None
            self.tokenizer = None
            self.torch = None
        
    def score_pair(self, good: str, bad: str) -> Tuple[float, float, bool, float]:
        """Score a minimal pair and return (good_score, bad_score, is_correct, time_taken)."""
        import time
        start_time = time.time()
        
        if self.use_ollama:
            good_score = self._score_ollama(good)
            bad_score = self._score_ollama(bad)
        elif self.model_type == "clm":
            good_score = self._score_clm(good)
            bad_score = self._score_clm(bad)
        else:
            good_score = self._score_mlm(good)
            bad_score = self._score_mlm(bad)
        
        end_time = time.time()
        time_taken = end_time - start_time
        
        is_correct = good_score > bad_score
        return good_score, bad_score, is_correct, time_taken

    def _score_ollama(self, sentence: str, max_retries: int = 3) -> float:
        """Score sentence using Ollama API with prompt-based rating."""
        import ollama
        import re
        import time
        
        for attempt in range(max_retries):
            try:
                prompt = f"""Rate the grammatical correctness of this sentence from 0-10 (0=ungrammatical, 10=perfect).

Sentence: "{sentence}"

Answer with ONLY the number, no explanation:"""

                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": 0.0,
                        "num_predict": 2048,
                        "num_ctx": 4096
                    }
                )
                
                response_text = response.get('response', '').strip()
                
                # Remove <think>...</think> blocks for reasoning models (e.g. DeepSeek R1)
                response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
                
                # Extract first number from response
                numbers = re.findall(r'\d+\.?\d*', response_text)
                if numbers:
                    score = float(numbers[0])
                    # Clamp to 0-10 range
                    score = max(0, min(10, score))
                    # Return raw 0-10 score
                    return score
                else:
                    # Fallback: if no number found, return 0
                    return 0.0
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    print(f"Error scoring with Ollama: {e}")
                    return 0.0
    
    def _score_clm(self, sentence: str) -> float:
        """Score sentence using autoregressive LM."""
        import torch.nn.functional as F
        
        enc = self.tokenizer(sentence, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        
        with self.torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
        
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        total_log_prob = float(token_log_probs.sum().cpu().item())
        num_tokens = shift_labels.shape[1]
        
        # Normalize to 0-10 scale
        # Avg log-prob is typically -10 to 0. Map to 0-10.
        if num_tokens > 0:
            avg_log_prob = total_log_prob / num_tokens
            score = avg_log_prob + 10.0
            return max(0.0, min(10.0, score))
        return 0.0
    
    def _score_mlm(self, sentence: str) -> float:
        """Score sentence using MLM with PLL-word-l2r."""
        import torch.nn.functional as F
        from .eval import _find_token_spans_for_words
        
        full_enc = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
        input_ids = full_enc["input_ids"].squeeze(0).tolist()
        if len(input_ids) == 0:
            return 0.0
        
        spans = _find_token_spans_for_words(self.tokenizer, sentence)
        mask_id = self.tokenizer.mask_token_id
        
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
        
        total_log_prob = 0.0
        for batch_start in range(0, len(operations), self.batch_size):
            batch = operations[batch_start:batch_start + self.batch_size]
            
            batch_input_ids = [op[0] for op in batch]
            batch_positions = [op[1] for op in batch]
            batch_targets = [op[2] for op in batch]
            
            max_len = max(len(seq) for seq in batch_input_ids)
            padded_input_ids = []
            attention_masks = []
            for seq in batch_input_ids:
                padded = seq + [self.tokenizer.pad_token_id or 0] * (max_len - len(seq))
                padded_input_ids.append(padded)
                attention_masks.append([1] * len(seq) + [0] * (max_len - len(seq)))
            
            input_tensor = self.torch.tensor(padded_input_ids).to(self.device)
            attention_mask = self.torch.tensor(attention_masks).to(self.device)
            
            with self.torch.no_grad():
                outputs = self.model(input_ids=input_tensor, attention_mask=attention_mask)
                logits = outputs.logits
            
            for idx, (pos, target_id) in enumerate(zip(batch_positions, batch_targets)):
                log_probs = F.log_softmax(logits[idx, pos, :], dim=-1)
                lp = float(log_probs[target_id].cpu().item())
                total_log_prob += lp
        
        # Normalize to 0-10 scale
        num_tokens = len(input_ids)
        if num_tokens > 0:
            avg_log_prob = total_log_prob / num_tokens
            score = avg_log_prob + 10.0
            return max(0.0, min(10.0, score))
        return 0.0
    
    def evaluate(self, pairs: List[Dict], show_progress: bool = True) -> Dict:
        """
        Evaluate list of minimal pairs.
        
        Args:
            pairs: List of dicts with 'good', 'bad', and optionally 'category'
            show_progress: Show tqdm progress bar
            
        Returns:
            Dict with overall and per-category accuracy stats
        """
        results = {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "categories": defaultdict(lambda: {"total": 0, "correct": 0, "accuracy": 0.0}),
            "errors": [],
            "pair_details": []  # Store details for each pair
        }
        
        iterator = tqdm(pairs, desc="Evaluating") if show_progress else pairs
        
        for idx, pair in enumerate(iterator, 1):
            good = pair["good"]
            bad = pair["bad"]
            category = pair.get("category", "default")
            phenomenon = pair.get("phenomenon", "N/A")
            
            good_score, bad_score, is_correct, time_taken = self.score_pair(good, bad)
            
            results["total"] += 1
            results["categories"][category]["total"] += 1
            
            if is_correct:
                results["correct"] += 1
                results["categories"][category]["correct"] += 1
            else:
                results["errors"].append({
                    "category": category,
                    "good": good,
                    "bad": bad,
                    "good_score": good_score,
                    "bad_score": bad_score
                })
            
            # Store pair-wise details
            results["pair_details"].append({
                "pair_id": idx,
                "category": category,
                "phenomenon": phenomenon,
                "good_sentence": good,
                "bad_sentence": bad,
                "good_score": good_score,
                "bad_score": bad_score,
                "score_difference": good_score - bad_score,
                "correct": is_correct,
                "time_taken": time_taken
            })
        
        # Calculate accuracies
        if results["total"] > 0:
            results["accuracy"] = results["correct"] / results["total"]
        
        for cat_data in results["categories"].values():
            if cat_data["total"] > 0:
                cat_data["accuracy"] = cat_data["correct"] / cat_data["total"]
        
        return results
    
    def print_report(self, results: Dict):
        """Print formatted evaluation report."""
        print("\n" + "="*60)
        print(f"EVALUATION REPORT: {self.model_name}")
        print(f"Model Type: {self.model_type.upper()}")
        print("="*60)
        
        print(f"\nOverall Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
        
        if len(results["categories"]) > 1:
            print("\nPer-Category Accuracy:")
            print("-" * 60)
            for category, stats in sorted(results["categories"].items()):
                print(f"  {category:30s}: {stats['accuracy']:6.2%} ({stats['correct']}/{stats['total']})")
        
        if results["errors"]:
            print(f"\nErrors: {len(results['errors'])} total")
            print("-" * 60)
            print("First 5 errors:")
            for i, error in enumerate(results["errors"][:5], 1):
                print(f"\n  Error {i} [{error['category']}]:")
                print(f"    Good: {error['good']}")
                print(f"    Bad:  {error['bad']}")
                print(f"    Scores: good={error['good_score']:.4f}, bad={error['bad_score']:.4f}")
        
        print("\n" + "="*60)


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file with minimal pairs."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


def save_results(results: Dict, output_path: str):
    """Save evaluation results to JSON file."""
    # Convert defaultdict to regular dict for JSON serialization
    serializable_results = {
        "total": results["total"],
        "correct": results["correct"],
        "accuracy": results["accuracy"],
        "categories": dict(results["categories"]),
        "errors": results["errors"],
        "pair_details": results.get("pair_details", [])
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)


def save_results_csv(results: Dict, output_path: str, model_name: str):
    """Save evaluation results to CSV file with pair-wise details and summary."""
    output_path = Path(output_path)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write header section
        writer.writerow(["BLIMP Evaluation Results"])
        writer.writerow(["Model", model_name])
        writer.writerow(["Overall Accuracy", f"{results['accuracy']:.2%}"])
        writer.writerow(["Correct", results["correct"]])
        writer.writerow(["Total", results["total"]])
        writer.writerow([])  # Empty row
        
        # Write category summary
        writer.writerow(["Category Summary"])
        writer.writerow(["Category", "Accuracy", "Correct", "Total"])
        for category, stats in sorted(results["categories"].items()):
            writer.writerow([
                category,
                f"{stats['accuracy']:.2%}",
                stats["correct"],
                stats["total"]
            ])
        writer.writerow([])  # Empty row
        
        # Write pair-wise details
        writer.writerow(["Pair-wise Results"])
        writer.writerow([
            "Pair ID",
            "Category",
            "Phenomenon",
            "Good Sentence",
            "Bad Sentence",
            "Good Score",
            "Bad Score",
            "Score Difference",
            "Correct"
        ])
        
        for detail in results.get("pair_details", []):
            writer.writerow([
                detail["pair_id"],
                detail["category"],
                detail["phenomenon"],
                detail["good_sentence"],
                detail["bad_sentence"],
                f"{detail['good_score']:.4f}",
                f"{detail['bad_score']:.4f}",
                f"{detail['score_difference']:.4f}",
                "✓" if detail["correct"] else "✗"
            ])
