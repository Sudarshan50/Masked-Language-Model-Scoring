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
  python scripts/evaluate_ollama.py --models deepseek-r1:7b qwen2.5:3b --data data/minimal_pairs.jsonl --output results.csv
"""

import argparse
import sys
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

try:
    import ollama
    from ollama import Client
    import httpx
    import torch
except ImportError:
    print("Error: ollama package not installed.")
    print("Install with: pip install ollama")
    sys.exit(1)

# Add src to path to import BLIMPEvaluator
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from eval_plausibility.blimp_evaluator import BLIMPEvaluator
except ImportError:
    print("Warning: Could not import BLIMPEvaluator. HF model support will be disabled.")
    BLIMPEvaluator = None


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
        for attempt in range(max_retries):
            try:
                prompt = f"""Evaluate the following sentence on two dimensions:

1. Grammaticality (syntax, morphology, word order)
2. Semantic plausibility (real-world sense, meaningfulness)

Give a combined score from 0 to 10:

0 = ungrammatical AND semantically impossible  
5 = grammatical OR meaningful but not both  
10 = both fully grammatical AND semantically normal

Sentence: "{sentence}"

Output ONLY a single number from 0 to 10."""

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
                import re
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
                    return 0.0
    
    def score_pair(self, good: str, bad: str) -> Tuple[float, float, bool, float]:
        """Score a minimal pair and determine if model is correct.
        Returns: (good_score, bad_score, is_correct, time_taken)
        """
        start_time = time.time()
        good_score = self.score_sentence(good)
        bad_score = self.score_sentence(bad)
        end_time = time.time()
        
        is_correct = good_score > bad_score
        return good_score, bad_score, is_correct, end_time - start_time
    
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
        
        iterator = tqdm(pairs, desc=f"Evaluating {self.model_name}") if show_progress else pairs
        
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
                "time_taken": time_taken,
                "model": self.model_name
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


def save_results_csv(all_results: List[Dict], output_path: str):
    """Save combined results from multiple models to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header section
        writer.writerow(["BLIMP Evaluation Results (Ollama)"])
        writer.writerow([])
        
        # Comparative Summary
        writer.writerow(["Comparative Summary"])
        writer.writerow(["Model", "Accuracy", "Correct", "Total", "Avg Time/Pair (s)"])
        
        for res in all_results:
            avg_time = sum(d["time_taken"] for d in res["pair_details"]) / res["total"] if res["total"] > 0 else 0
            writer.writerow([
                res["model"],
                f"{res['accuracy']:.2%}",
                res["correct"],
                res["total"],
                f"{avg_time:.2f}"
            ])
        writer.writerow([])
        
        # Category summary (for the first model, or aggregated? Let's do per-model per-category if needed, 
        # but for simplicity let's just dump the pair-wise details which is most important for analysis)
        
        # Pair-wise details
        writer.writerow(["Pair-wise Results"])
        writer.writerow([
            "Model", "Pair ID", "Category", "Phenomenon",
            "Good Sentence", "Bad Sentence",
            "Good Score", "Bad Score", "Score Difference", "Correct", "Time Taken"
        ])
        
        for res in all_results:
            for detail in res["pair_details"]:
                writer.writerow([
                    detail["model"],
                    detail["pair_id"],
                    detail["category"],
                    detail["phenomenon"],
                    detail["good_sentence"],
                    detail["bad_sentence"],
                    f"{detail['good_score']:.4f}",
                    f"{detail['bad_score']:.4f}",
                    f"{detail['score_difference']:.4f}",
                    "✓" if detail["correct"] else "✗",
                    f"{detail['time_taken']:.4f}"
                ])


def parse_model_arg(model_arg: str) -> Tuple[str, str]:
    """Parse model argument in format 'name:type' or just 'name'.
    
    Returns (model_name, model_type).
    model_type is 'ollama' by default, or 'mlm'/'clm' if specified or inferred.
    """
    if ":" in model_arg:
        parts = model_arg.rsplit(":", 1)
        # Check if suffix is explicitly mlm or clm
        if len(parts) == 2 and parts[1].lower() in ["mlm", "clm"]:
            return parts[0], parts[1].lower()
    
    return model_arg, "unknown"


def is_ollama_model(model_name: str) -> bool:
    """Check if a model is available in Ollama."""
    try:
        models = ollama.list()
        if isinstance(models, dict):
            available_models = [m.get('name', '') for m in models.get('models', [])]
        else:
            available_models = [getattr(m, 'name', getattr(m, 'model', '')) for m in getattr(models, 'models', [])]
        
        # Check exact match or prefix match
        return any(
            model_name == m or m.startswith(model_name + ':') or m.startswith(model_name)
            for m in available_models if m
        )
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate minimal pairs using Ollama or HuggingFace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_ollama.py --models deepseek-r1:7b qwen2.5:3b --data data/extensive_test_pairs.jsonl
  python scripts/evaluate_ollama.py --model gpt2 --data data/minimal_pairs.jsonl
  python scripts/evaluate_ollama.py --model roberta-base:mlm --data data/minimal_pairs.jsonl
  
Available model sizes:
  DeepSeek-R1: 1.5b, 7b, 14b, 32b, 70b
  Qwen2.5: 0.5b, 1.5b, 3b, 7b, 14b, 32b, 72b
  Llama3: 1b, 3b, 8b, 70b
        """
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", nargs="+", help="List of model names (Ollama or HF)")
    group.add_argument("--model", help="Single model name (legacy support)")
    
    parser.add_argument("--data", required=True, help="JSONL file with minimal pairs")
    parser.add_argument("--output", help="Output CSV or JSON file")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    parser.add_argument("--device", default=None, help="Device for HF models (cpu, cuda, or mps). If not specified, auto-detects.")
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        # Auto-detect: prefer MPS for Apple Silicon, then CUDA, then CPU
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    
    print(f"Using device: {args.device}")
    
    # Handle legacy --model argument
    models = args.models if args.models else [args.model]
    
    print(f"Loading pairs from {args.data}...")
    pairs = load_jsonl(args.data)
    print(f"Loaded {len(pairs)} minimal pairs")
    
    all_results = []
    
    for model_arg in models:
        print(f"\nProcessing model argument: {model_arg}")
        
        # Determine if it's Ollama or HF
        model_name, explicit_type = parse_model_arg(model_arg)
        
        # Strategy:
        # 1. If explicit type is mlm/clm, use BLIMPEvaluator (HF)
        # 2. If it looks like an Ollama model (exists in ollama list), use OllamaEvaluator
        # 3. Fallback to BLIMPEvaluator (HF) - default to 'mlm' if not specified, or let BLIMPEvaluator handle it
        
        use_ollama = False
        
        if explicit_type in ["mlm", "clm"]:
            use_ollama = False
        elif is_ollama_model(model_name):
            use_ollama = True
        else:
            # Not explicitly mlm/clm, and not found in Ollama -> Assume HF
            use_ollama = False
            if explicit_type == "unknown":
                # Default to mlm for HF if unknown, similar to evaluate_blimp_hf.py logic
                # But actually, let's try to be smart. If it's gpt2, it's clm.
                if "gpt" in model_name.lower():
                    explicit_type = "clm"
                else:
                    explicit_type = "mlm"
        
        try:
            if use_ollama:
                print(f"Initializing Ollama evaluator with model: {model_name}")
                evaluator = OllamaEvaluator(model_name=model_name)
                results = evaluator.evaluate(pairs, show_progress=not args.no_progress)
            else:
                if BLIMPEvaluator is None:
                    print(f"Skipping {model_name}: BLIMPEvaluator not available (missing dependencies?)")
                    continue
                    
                print(f"Initializing HF evaluator with model: {model_name} (type: {explicit_type})")
                # BLIMPEvaluator expects model_type to be 'mlm' or 'clm'
                evaluator = BLIMPEvaluator(
                    model_name=model_name,
                    model_type=explicit_type,
                    device=args.device  # Use auto-detected or specified device
                )
                
                # We need to adapt BLIMPEvaluator results to match OllamaEvaluator results structure if they differ
                # BLIMPEvaluator.evaluate returns a dict that is very similar.
                # Let's check keys.
                # Ollama: model, total, correct, accuracy, categories, pair_details
                # BLIMP: total, correct, accuracy, categories, errors, pair_details
                # We just need to add 'model' key to results
                
                results = evaluator.evaluate(pairs, show_progress=not args.no_progress)
                results["model"] = model_name
                
                # Ensure pair_details has model name
                for p in results["pair_details"]:
                    p["model"] = model_name
            
            # Print report (OllamaEvaluator has it as a method, BLIMPEvaluator has it as a method)
            # We can just call the method on the evaluator instance?
            # Both classes have print_report(results)
            evaluator.print_report(results)
            
            all_results.append(results)
            
        except Exception as e:
            print(f"Failed to evaluate model {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print Comparative Summary
    print("\n" + "="*80)
    print("FINAL COMPARATIVE SUMMARY")
    print("="*80)
    print(f"{'Model':<30} {'Accuracy':<10} {'Correct':<10} {'Total':<10} {'Avg Time (s)':<15}")
    print("-" * 80)
    for res in all_results:
        avg_time = sum(d["time_taken"] for d in res["pair_details"]) / res["total"] if res["total"] > 0 else 0
        print(f"{res['model']:<30} {res['accuracy']:<10.2%} {res['correct']:<10} {res['total']:<10} {avg_time:<15.2f}")
    print("="*80)
    
    if args.output:
        # If output ends in .json, save the last result as JSON (legacy behavior)
        if args.output.endswith('.json') and len(all_results) == 1:
            save_results_json(all_results[0], args.output)
            print(f"\nResults saved to: {args.output}")
            
            # Also save CSV
            csv_path = Path(args.output).with_suffix('.csv')
            save_results_csv(all_results, str(csv_path))
            print(f"CSV results saved to: {csv_path}")
        else:
            # Default to CSV for multiple models or if extension is not .json
            output_path = args.output
            if not output_path.endswith('.csv') and not output_path.endswith('.json'):
                output_path += '.csv'
            
            save_results_csv(all_results, output_path)
            print(f"\nComparative results saved to: {output_path}")


if __name__ == "__main__":
    main()
