"""BLIMP-format evaluation harness with aggregated metrics and confusion analysis."""
import json
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
            model_name: HuggingFace model identifier
            model_type: Either 'clm' or 'mlm'
            device: 'cpu' or 'cuda'
            batch_size: Batch size for MLM scoring (ignored for CLM)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.batch_size = batch_size
        
        # Cache model and tokenizer
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_type == "clm":
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.torch = torch
        
    def score_pair(self, good: str, bad: str) -> Tuple[float, float, bool]:
        """Score a minimal pair and return (good_score, bad_score, is_correct)."""
        import torch.nn.functional as F
        
        if self.model_type == "clm":
            good_score = self._score_clm(good)
            bad_score = self._score_clm(bad)
        else:
            good_score = self._score_mlm(good)
            bad_score = self._score_mlm(bad)
        
        is_correct = good_score > bad_score
        return good_score, bad_score, is_correct
    
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
        return float(token_log_probs.sum().cpu().item())
    
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
        
        return total_log_prob
    
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
            "errors": []
        }
        
        iterator = tqdm(pairs, desc="Evaluating") if show_progress else pairs
        
        for pair in iterator:
            good = pair["good"]
            bad = pair["bad"]
            category = pair.get("category", "default")
            
            good_score, bad_score, is_correct = self.score_pair(good, bad)
            
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
        "errors": results["errors"]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)
