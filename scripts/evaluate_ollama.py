#!/usr/bin/env python3
"""Evaluate minimal pairs using Ollama models (DeepSeek, Qwen, Llama, etc.).

This script uses Ollama's local API to evaluate sentence pairs.
Make sure Ollama is running: `ollama serve`

Available models (pull first with `ollama pull <model>`):
- deepseek-r1:1.5b, deepseek-r1:7b, deepseek-r1:14b, deepseek-r1:32b, deepseek-r1:70b
- qwen2.5:0.5b, qwen2.5:1.5b, qwen2.5:3b, qwen2.5:7b, qwen2.5:14b, qwen2.5:32b, qwen2.5:72b
- llama3.2:1b, llama3.2:3b, llama3.1:8b, llama3.1:70b
- mistral:7b, mixtral:8x7b
- phi4:14b

Example:
  python scripts/evaluate_ollama.py --model deepseek-r1:7b --data data/extensive_test_pairs.jsonl
  python scripts/evaluate_ollama.py --model qwen2.5:3b --data data/minimal_pairs.jsonl --output results.json
"""

import argparse
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm

try:
    import ollama
    from ollama import Client
    import httpx
except ImportError:
    print("Error: ollama package not installed.")
    print("Install with: pip install ollama")
    sys.exit(1)
    print("Install with: pip install ollama")
    sys.exit(1)


class OllamaEvaluator:
    """Evaluator for BLIMP-format minimal pairs using Ollama models."""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self._check_model_available()
    
    def _check_model_available(self):
        """Check if the model is available in Ollama."""
        try:
            models = ollama.list()
            # Handle both dict and object response types
            if isinstance(models, dict):
                available_models = [m.get('name', '') for m in models.get('models', [])]
            else:
                available_models = [getattr(m, 'name', getattr(m, 'model', '')) for m in getattr(models, 'models', [])]
            
            # Check exact match or prefix match
            model_available = any(
                self.model_name == m or m.startswith(self.model_name + ':') or m.startswith(self.model_name)
                for m in available_models if m
            )
            
            if not model_available:
                print(f"\n⚠️  Model '{self.model_name}' not found in Ollama.")
                print(f"Available models: {', '.join(available_models) if available_models else 'none'}")
                print(f"\nTo pull this model, run: ollama pull {self.model_name}")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
            print(f"Debug info: {type(e).__name__}")
            sys.exit(1)
    
    def score_sentence(self, sentence: str, max_retries: int = 3) -> float:
        """Score a single sentence using Ollama.
        
        Uses a grammaticality rating prompt to get a score.
        Returns a normalized score where higher is better.
        """
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
                        "num_predict": 10,
                        "num_ctx": 512
                    }
                )
                
                response_text = response.get('response', '').strip()
                
                # Extract first number from response
                import re
                numbers = re.findall(r'\d+\.?\d*', response_text)
                if numbers:
                    score = float(numbers[0])
                    # Clamp to 0-10 range
                    score = max(0, min(10, score))
                    # Convert to log-like scale: map [0,10] to roughly [-5, 0]
                    # Higher original score -> higher (less negative) result
                    return (score - 5.0) / 2.0
                else:
                    # Fallback: if no number found, use sentence length heuristic
                    return -len(sentence.split()) * 0.5
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\nRetry {attempt + 1}/{max_retries} for sentence: {sentence[:50]}...")
                    time.sleep(2)
                    continue
                else:
                    print(f"\nFailed after {max_retries} attempts: {e}")
                    # Return neutral score on failure
                    return -2.5
    
    def score_pair(self, good: str, bad: str) -> tuple[float, float, bool]:
        """Score a minimal pair and determine if model is correct."""
        good_score = self.score_sentence(good)
        bad_score = self.score_sentence(bad)
        is_correct = good_score > bad_score
        return good_score, bad_score, is_correct
    
    def evaluate(self, pairs: List[Dict], show_progress: bool = True) -> Dict:
        """Evaluate all minimal pairs."""
        results = {
            "model": self.model_name,
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "categories": defaultdict(lambda: {"total": 0, "correct": 0, "accuracy": 0.0}),
            "pair_details": []
        }
        
        iterator = tqdm(pairs, desc="Evaluating") if show_progress else pairs
        
        for idx, pair in enumerate(iterator, 1):
            good = pair["good"]
            bad = pair["bad"]
            category = pair.get("category", "default")
            phenomenon = pair.get("phenomenon", "N/A")
            
            good_score, bad_score, is_correct = self.score_pair(good, bad)
            
            results["total"] += 1
            results["categories"][category]["total"] += 1
            
            if is_correct:
                results["correct"] += 1
                results["categories"][category]["correct"] += 1
            
            results["pair_details"].append({
                "pair_id": idx,
                "category": category,
                "phenomenon": phenomenon,
                "good_sentence": good,
                "bad_sentence": bad,
                "good_score": good_score,
                "bad_score": bad_score,
                "score_difference": good_score - bad_score,
                "correct": is_correct
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
        print(f"OLLAMA EVALUATION REPORT: {self.model_name}")
        print("="*60)
        
        print(f"\nOverall Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
        
        if len(results["categories"]) > 1:
            print("\nPer-Category Accuracy:")
            print("-" * 60)
            for category, stats in sorted(results["categories"].items()):
                print(f"  {category:30s}: {stats['accuracy']:6.2%} ({stats['correct']}/{stats['total']})")
        
        # Show errors
        errors = [p for p in results["pair_details"] if not p["correct"]]
        if errors:
            print(f"\nErrors: {len(errors)} total")
            print("-" * 60)
            print("First 5 errors:")
            for i, error in enumerate(errors[:5], 1):
                print(f"\n  Error {i} [{error['category']}]:")
                print(f"    Good: {error['good_sentence']}")
                print(f"    Bad:  {error['bad_sentence']}")
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


def save_results_json(results: Dict, output_path: str):
    """Save results to JSON file."""
    serializable_results = {
        "model": results["model"],
        "total": results["total"],
        "correct": results["correct"],
        "accuracy": results["accuracy"],
        "categories": dict(results["categories"]),
        "pair_details": results["pair_details"]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)


def save_results_csv(results: Dict, output_path: str):
    """Save results to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header section
        writer.writerow(["BLIMP Evaluation Results (Ollama)"])
        writer.writerow(["Model", results["model"]])
        writer.writerow(["Overall Accuracy", f"{results['accuracy']:.2%}"])
        writer.writerow(["Correct", results["correct"]])
        writer.writerow(["Total", results["total"]])
        writer.writerow([])
        
        # Category summary
        writer.writerow(["Category Summary"])
        writer.writerow(["Category", "Accuracy", "Correct", "Total"])
        for category, stats in sorted(results["categories"].items()):
            writer.writerow([
                category,
                f"{stats['accuracy']:.2%}",
                stats["correct"],
                stats["total"]
            ])
        writer.writerow([])
        
        # Pair-wise details
        writer.writerow(["Pair-wise Results"])
        writer.writerow([
            "Pair ID", "Category", "Phenomenon",
            "Good Sentence", "Bad Sentence",
            "Good Score", "Bad Score", "Score Difference", "Correct"
        ])
        
        for detail in results["pair_details"]:
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate minimal pairs using Ollama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_ollama.py --model deepseek-r1:7b --data data/extensive_test_pairs.jsonl
  python scripts/evaluate_ollama.py --model qwen2.5:3b --data data/minimal_pairs.jsonl --output results.json
  
Available model sizes:
  DeepSeek-R1: 1.5b, 7b, 14b, 32b, 70b
  Qwen2.5: 0.5b, 1.5b, 3b, 7b, 14b, 32b, 72b
  Llama3: 1b, 3b, 8b, 70b
        """
    )
    parser.add_argument("--model", required=True, help="Ollama model name (e.g., deepseek-r1:7b, qwen2.5:3b)")
    parser.add_argument("--data", required=True, help="JSONL file with minimal pairs")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--csv", help="Output CSV file")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    args = parser.parse_args()
    
    print(f"Loading pairs from {args.data}...")
    pairs = load_jsonl(args.data)
    print(f"Loaded {len(pairs)} minimal pairs")
    
    print(f"\nInitializing Ollama evaluator with model: {args.model}")
    evaluator = OllamaEvaluator(model_name=args.model)
    
    results = evaluator.evaluate(pairs, show_progress=not args.no_progress)
    evaluator.print_report(results)
    
    if args.output:
        save_results_json(results, args.output)
        print(f"\nResults saved to: {args.output}")
        
        # Auto-generate CSV
        csv_path = Path(args.output).with_suffix('.csv')
        save_results_csv(results, str(csv_path))
        print(f"CSV results saved to: {csv_path}")
    
    if args.csv:
        save_results_csv(results, args.csv)
        print(f"\nCSV results saved to: {args.csv}")


if __name__ == "__main__":
    main()
