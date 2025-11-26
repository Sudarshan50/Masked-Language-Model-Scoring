#!/usr/bin/env python3
"""Enhanced CLI for BLIMP-format evaluation with detailed reporting.

Example:
  python scripts/evaluate_blimp.py --model bert-base-uncased --type mlm --data data/minimal_pairs.jsonl
  python scripts/evaluate_blimp.py --model gpt2 --type clm --data data/minimal_pairs.jsonl --output results.json
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_plausibility.blimp_evaluator import BLIMPEvaluator, load_jsonl, save_results, save_results_csv


def main():
    parser = argparse.ArgumentParser(description="Evaluate minimal pairs with BLIMP-format reporting")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--type", choices=["mlm", "clm"], required=True, help="Model type")
    parser.add_argument("--data", required=True, help="JSONL file with minimal pairs")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for MLM scoring")
    parser.add_argument("--output", help="Output JSON file for results (optional)")
    parser.add_argument("--csv", help="Output CSV file for results (optional)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    args = parser.parse_args()

    print(f"Loading pairs from {args.data}...")
    pairs = load_jsonl(args.data)
    print(f"Loaded {len(pairs)} minimal pairs")

    print(f"\nInitializing evaluator with model: {args.model}")
    evaluator = BLIMPEvaluator(
        model_name=args.model,
        model_type=args.type,
        device=args.device,
        batch_size=args.batch_size
    )

    results = evaluator.evaluate(pairs, show_progress=not args.no_progress)
    evaluator.print_report(results)

    if args.output:
        save_results(results, args.output)
        print(f"\nResults saved to: {args.output}")
    
    if args.csv:
        save_results_csv(results, args.csv, args.model)
        print(f"CSV results saved to: {args.csv}")
    
    # Auto-generate CSV if output is specified but csv is not
    if args.output and not args.csv:
        csv_path = Path(args.output).with_suffix('.csv')
        save_results_csv(results, str(csv_path), args.model)
        print(f"CSV results saved to: {csv_path}")


if __name__ == "__main__":
    main()
