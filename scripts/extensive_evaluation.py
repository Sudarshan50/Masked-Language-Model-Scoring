#!/usr/bin/env python3
"""Comprehensive testing script with multiple models and detailed analysis.

This script evaluates multiple models on an extensive dataset and generates
comparative reports, statistical analysis, and per-phenomenon breakdowns.
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_plausibility.blimp_evaluator import BLIMPEvaluator, load_jsonl


def extensive_evaluation(models_config, data_path, output_dir, device="cpu", batch_size=16):
    """Run extensive evaluation across multiple models."""
    
    pairs = load_jsonl(data_path)
    print(f"Loaded {len(pairs)} minimal pairs from {data_path}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_name, model_type in models_config:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name} ({model_type.upper()})")
        print(f"{'='*70}")
        
        evaluator = BLIMPEvaluator(
            model_name=model_name,
            model_type=model_type,
            device=device,
            batch_size=batch_size
        )
        
        results = evaluator.evaluate(pairs, show_progress=True)
        all_results[model_name] = results
        
        evaluator.print_report(results)
        
        # Save individual results
        result_file = output_dir / f"{model_name.replace('/', '_')}_results.json"
        with open(result_file, "w") as f:
            json.dump({
                "model": model_name,
                "type": model_type,
                "total": results["total"],
                "correct": results["correct"],
                "accuracy": results["accuracy"],
                "categories": dict(results["categories"]),
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        print(f"Saved results to: {result_file}")
    
    # Generate comparative report
    print(f"\n{'='*70}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"{'Model':<30} {'Type':<8} {'Accuracy':<12} {'Correct/Total'}")
    print("-" * 70)
    for model_name in all_results:
        res = all_results[model_name]
        model_type = [t for m, t in models_config if m == model_name][0]
        print(f"{model_name:<30} {model_type.upper():<8} {res['accuracy']:>10.2%}  {res['correct']}/{res['total']}")
    
    # Phenomenon-level analysis
    if pairs and "phenomenon" in pairs[0]:
        print(f"\n{'='*70}")
        print("PER-PHENOMENON ANALYSIS")
        print(f"{'='*70}\n")
        
        phenomena_stats = defaultdict(lambda: {"total": 0, "models_correct": defaultdict(int)})
        
        for pair in pairs:
            phenomenon = pair.get("phenomenon", "unknown")
            phenomena_stats[phenomenon]["total"] += 1
        
        for model_name, results in all_results.items():
            correct_pairs = set()
            for i, pair in enumerate(pairs):
                # Check if this pair was correct
                idx = i
                is_error = any(err for err in results["errors"] 
                             if err["good"] == pair["good"] and err["bad"] == pair["bad"])
                if not is_error:
                    correct_pairs.add(i)
            
            for i, pair in enumerate(pairs):
                if i in correct_pairs:
                    phenomenon = pair.get("phenomenon", "unknown")
                    phenomena_stats[phenomenon]["models_correct"][model_name] += 1
        
        print(f"{'Phenomenon':<30} {'Total':<8} {'Model Accuracies'}")
        print("-" * 70)
        for phenomenon in sorted(phenomena_stats.keys()):
            stats = phenomena_stats[phenomenon]
            total = stats["total"]
            accuracies = " | ".join([
                f"{m.split('/')[-1][:15]}: {stats['models_correct'][m]}/{total}"
                for m in all_results.keys()
            ])
            print(f"{phenomenon:<30} {total:<8} {accuracies}")
    
    # Save comparative summary
    summary_file = output_dir / "comparative_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(data_path),
        "total_pairs": len(pairs),
        "models": {
            model_name: {
                "accuracy": results["accuracy"],
                "correct": results["correct"],
                "total": results["total"],
                "categories": dict(results["categories"])
            }
            for model_name, results in all_results.items()
        }
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nComparative summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Extensive semantic plausibility evaluation")
    parser.add_argument("--data", required=True, help="Path to JSONL test data")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for MLM")
    parser.add_argument("--models", nargs="+", help="Models to test (format: name:type)")
    args = parser.parse_args()
    
    # Default models if none specified
    if args.models:
        models_config = []
        for spec in args.models:
            name, mtype = spec.split(":")
            models_config.append((name, mtype))
    else:
        models_config = [
            ("bert-base-uncased", "mlm"),
            ("roberta-base", "mlm"),
            ("gpt2", "clm"),
        ]
    
    extensive_evaluation(
        models_config=models_config,
        data_path=args.data,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
