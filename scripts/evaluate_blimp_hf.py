#!/usr/bin/env python3
"""
Evaluate BLiMP dataset from Hugging Face using PLL scoring across multiple models.

Example:
  python scripts/evaluate_blimp_hf.py --models bert-base-uncased:mlm gpt2:clm --output results.csv
"""
import argparse
import sys
import gc
import torch
from pathlib import Path
import pandas as pd
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_plausibility.blimp_evaluator import BLIMPEvaluator

def parse_model_arg(model_arg: str) -> tuple[str, str]:
    """Parse model argument in format 'name:type' or just 'name'.
    
    Handles Ollama models with tags (e.g. deepseek-r1:7b) correctly by only 
    interpreting :mlm or :clm suffixes as types.
    """
    if ":" in model_arg:
        # Check if the last part is a valid type
        parts = model_arg.rsplit(":", 1)
        if len(parts) == 2 and parts[1].lower() in ["mlm", "clm"]:
            return parts[0], parts[1].lower()
    
    # User requested PLL for ALL models by default
    return model_arg, "mlm"

def main():
    parser = argparse.ArgumentParser(description="Evaluate BLiMP dataset across multiple models (HuggingFace or Ollama)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", nargs="+", 
                      help="List of models to evaluate. Format: model_name:type (e.g. bert-base-uncased:mlm, deepseek-r1:7b). "
                           "Supports HuggingFace models and Ollama models.")
    group.add_argument("--model", help="Single model to evaluate (legacy support).")
    
    parser.add_argument("--dataset", default="hf://datasets/nyu-mll/blimp/adjunct_island/train-00000-of-00001.parquet", 
                      help="Dataset path/URL")
    parser.add_argument("--device", default=None, help="Device (cpu, cuda, or mps). If not specified, auto-detects.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for MLM scoring")
    parser.add_argument("--output", default="blimp_results.csv", help="Output CSV path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs for testing")
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        # User requested to force MPS only if available
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    
    print(f"Using device: {args.device}")
    
    # Handle legacy --model argument
    if args.model:
        args.models = [args.model]

    print(f"Loading dataset from {args.dataset}...")
    try:
        df = pd.read_parquet(args.dataset)
        if args.limit:
            df = df.head(args.limit)
            print(f"Limiting to first {args.limit} rows")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
        
    print(f"Loaded {len(df)} rows")

    # Convert to format expected by BLIMPEvaluator
    pairs = []
    for _, row in df.iterrows():
        pairs.append({
            "good": row["sentence_good"],
            "bad": row["sentence_bad"],
            "category": row.get("linguistics_term", "default"),
            "phenomenon": row.get("UID", "N/A")
        })

    all_pair_results = []
    model_metrics = []

    for model_arg in args.models:
        model_name, model_type = parse_model_arg(model_arg)
        
        # Initial attempt configuration
        current_model_type = model_type
        
        print(f"\n" + "="*60)
        print(f"Evaluating model: {model_name}")
        
        try:
            # Try to initialize evaluator
            try:
                evaluator = BLIMPEvaluator(
                    model_name=model_name,
                    model_type=current_model_type,
                    device=args.device,
                    batch_size=args.batch_size
                )
                scoring_method = "PLL (Pseudo-Log-Likelihood)" if current_model_type == "mlm" else "Causal Log-Probability"
                print(f"Model Type:       {current_model_type.upper()}")
                print(f"Scoring Method:   {scoring_method}")
                
            except Exception as e:
                # If MLM failed, try CLM fallback
                if current_model_type == "mlm" and ("Unrecognized configuration class" in str(e) or "architectures" in str(e)):
                    print(f"Warning: Could not load {model_name} as MLM ({e}).")
                    print("Falling back to CLM (Causal Language Model) mode.")
                    print("Note: For Causal LMs, the Causal Log-Probability is the mathematical equivalent of PLL.")
                    
                    current_model_type = "clm"
                    evaluator = BLIMPEvaluator(
                        model_name=model_name,
                        model_type=current_model_type,
                        device=args.device,
                        batch_size=args.batch_size
                    )
                    print(f"Model Type:       {current_model_type.upper()}")
                    print(f"Scoring Method:   Causal Log-Probability (PLL-equivalent)")
                else:
                    raise e

            print("="*60)
            results = evaluator.evaluate(pairs)
            
            # Collect metrics
            model_metrics.append({
                "Model": model_name,
                "Type": current_model_type,
                "Accuracy": results["accuracy"],
                "Correct": results["correct"],
                "Total": results["total"]
            })
            
            # Collect pair details
            for detail in results["pair_details"]:
                detail["model"] = model_name
                detail["model_type"] = current_model_type
                all_pair_results.append(detail)
                
            # Cleanup to free memory
            del evaluator
            del results
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            continue

    # Save detailed results to CSV
    if all_pair_results:
        print(f"\nSaving detailed results to {args.output}...")
        results_df = pd.DataFrame(all_pair_results)
        
        # Reorder columns for better readability
        cols = ["model", "model_type", "pair_id", "category", "phenomenon", 
                "good_sentence", "bad_sentence", "good_score", "bad_score", 
                "score_difference", "correct", "time_taken"]
        # Ensure all columns exist (in case some are missing)
        cols = [c for c in cols if c in results_df.columns] + [c for c in results_df.columns if c not in cols]
        
        results_df = results_df[cols]
        results_df.to_csv(args.output, index=False)
        print("Done.")

    # Print Comparative Summary
    print("\n" + "="*80)
    print("FINAL COMPARATIVE SUMMARY")
    print("="*80)
    if model_metrics:
        metrics_df = pd.DataFrame(model_metrics)
        # Format accuracy as percentage
        metrics_df["Accuracy"] = metrics_df["Accuracy"].apply(lambda x: f"{x:.2%}")
        print(metrics_df.to_string(index=False))
    else:
        print("No models were successfully evaluated.")
    print("="*80) 

if __name__ == "__main__":
    main()
