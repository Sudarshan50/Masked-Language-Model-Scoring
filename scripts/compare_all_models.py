#!/usr/bin/env python3
"""
Compare multiple models (HuggingFace and Ollama) on the same dataset.

This script runs evaluations across multiple models and generates a comparison CSV.

Example:
  python scripts/compare_all_models.py --data data/extensive_test_pairs.jsonl
"""

import argparse
import json
import csv
from pathlib import Path
import subprocess
import sys

def run_hf_model(model_name, model_type, data_path, output_dir):
    """Run HuggingFace model evaluation."""
    output_file = output_dir / f"{model_name.replace('/', '_')}_results.json"
    
    cmd = [
        sys.executable,
        "scripts/evaluate_blimp.py",
        "--model", model_name,
        "--type", model_type,
        "--data", data_path,
        "--output", str(output_file)
    ]
    
    print(f"\n{'='*60}")
    print(f"Evaluating HuggingFace model: {model_name}")
    print(f"{'='*60}")
    
    try:
        subprocess.run(cmd, check=True)
        return str(output_file)
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {model_name}: {e}")
        return None

def run_ollama_model(model_name, data_path, output_dir):
    """Run Ollama model evaluation."""
    output_file = output_dir / f"{model_name.replace(':', '_')}_results.json"
    
    cmd = [
        sys.executable,
        "scripts/evaluate_ollama.py",
        "--model", model_name,
        "--data", data_path,
        "--output", str(output_file)
    ]
    
    print(f"\n{'='*60}")
    print(f"Evaluating Ollama model: {model_name}")
    print(f"{'='*60}")
    
    try:
        subprocess.run(cmd, check=True)
        return str(output_file)
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {model_name}: {e}")
        return None

def load_results(json_path):
    """Load results from JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None

def create_comparison_csv(results_files, output_dir):
    """Create comprehensive comparison CSVs from multiple result files."""
    all_results = []
    
    for result_file in results_files:
        if result_file and Path(result_file).exists():
            data = load_results(result_file)
            if data:
                all_results.append(data)
    
    if not all_results:
        print("No results to compare!")
        return
    
    # Sort by accuracy for consistent ordering
    all_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
    
    # 1. Overall Summary CSV
    summary_path = output_dir / "comparison_summary.csv"
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy (%)", "Correct", "Total", "Errors"])
        
        for result in all_results:
            model_name = result.get('model', 'Unknown')
            writer.writerow([
                model_name,
                f"{result['accuracy'] * 100:.1f}",
                result['correct'],
                result['total'],
                result['total'] - result['correct']
            ])
    
    print(f"✓ Summary: {summary_path}")
    
    # 2. Category Breakdown CSV
    category_path = output_dir / "comparison_categories.csv"
    with open(category_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Collect all categories
        categories = set()
        for result in all_results:
            categories.update(result.get('categories', {}).keys())
        categories = sorted(categories)
        
        # Header
        header = ["Model"]
        for cat in categories:
            header.extend([f"{cat} Correct", f"{cat} Total", f"{cat} Accuracy (%)"])
        writer.writerow(header)
        
        # Data rows
        for result in all_results:
            row = [result.get('model', 'Unknown')]
            for cat in categories:
                cat_data = result.get('categories', {}).get(cat, {})
                row.extend([
                    cat_data.get('correct', 0),
                    cat_data.get('total', 0),
                    f"{cat_data.get('accuracy', 0) * 100:.1f}"
                ])
            writer.writerow(row)
    
    print(f"✓ Categories: {category_path}")
    
    # 3. Pair-by-Pair Comparison CSV
    pairwise_path = output_dir / "comparison_pairwise.csv"
    with open(pairwise_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        header = ["Pair ID", "Category", "Phenomenon", "Good Sentence", "Bad Sentence"]
        for result in all_results:
            model = result.get('model', 'Unknown')
            header.extend([f"{model} Good", f"{model} Bad", f"{model} Result"])
        writer.writerow(header)
        
        # Get number of pairs
        num_pairs = all_results[0]['total']
        
        # Data rows
        for pair_idx in range(num_pairs):
            first_pair = all_results[0]['pair_details'][pair_idx]
            row = [
                first_pair['pair_id'],
                first_pair['category'],
                first_pair['phenomenon'],
                first_pair['good_sentence'],
                first_pair['bad_sentence']
            ]
            
            for result in all_results:
                pair = result['pair_details'][pair_idx]
                row.extend([
                    f"{pair['good_score']:.2f}",
                    f"{pair['bad_score']:.2f}",
                    "✓" if pair['correct'] else "✗"
                ])
            
            writer.writerow(row)
    
    print(f"✓ Pair-wise: {pairwise_path}")
    
    # 4. Errors Only CSV
    errors_path = output_dir / "comparison_errors.csv"
    with open(errors_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Pair ID", "Category", "Phenomenon", 
                        "Good Sentence", "Bad Sentence", 
                        "Good Score", "Bad Score", "Difference"])
        
        for result in all_results:
            model = result.get('model', 'Unknown')
            for pair in result['pair_details']:
                if not pair['correct']:
                    writer.writerow([
                        model,
                        pair['pair_id'],
                        pair['category'],
                        pair['phenomenon'],
                        pair['good_sentence'],
                        pair['bad_sentence'],
                        f"{pair['good_score']:.2f}",
                        f"{pair['bad_score']:.2f}",
                        f"{pair['score_difference']:.2f}"
                    ])
    
    print(f"✓ Errors: {errors_path}")
    
    # 5. Phenomenon Analysis CSV
    phenomena_path = output_dir / "comparison_phenomena.csv"
    with open(phenomena_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Collect all phenomena
        phenomena_map = {}
        for result in all_results:
            for pair in result['pair_details']:
                key = (pair['category'], pair['phenomenon'])
                if key not in phenomena_map:
                    phenomena_map[key] = {}
                model = result.get('model', 'Unknown')
                if model not in phenomena_map[key]:
                    phenomena_map[key][model] = {'correct': 0, 'total': 0}
                phenomena_map[key][model]['total'] += 1
                if pair['correct']:
                    phenomena_map[key][model]['correct'] += 1
        
        # Header
        header = ["Category", "Phenomenon"]
        for result in all_results:
            header.append(f"{result.get('model', 'Unknown')} Correct")
        writer.writerow(header)
        
        # Data rows
        for (category, phenomenon), models_data in sorted(phenomena_map.items()):
            row = [category, phenomenon]
            for result in all_results:
                model = result.get('model', 'Unknown')
                data = models_data.get(model, {'correct': 0, 'total': 0})
                row.append(f"{data['correct']}/{data['total']}")
            writer.writerow(row)
    
    print(f"✓ Phenomena: {phenomena_path}")
    
    print("\n" + "="*60)
    print("✅ All comparison reports generated!")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Compare multiple models on BLIMP evaluation")
    parser.add_argument("--data", required=True, help="JSONL file with minimal pairs")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--hf-models", nargs="+", help="HuggingFace models to evaluate")
    parser.add_argument("--ollama-models", nargs="+", help="Ollama models to evaluate")
    parser.add_argument("--all-small", action="store_true", help="Test all small/fast models")
    parser.add_argument("--all-medium", action="store_true", help="Test all medium models")
    parser.add_argument("--use-existing", action="store_true", help="Use existing result files without running evaluations")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    results_files = []
    
    # If use-existing flag, find all JSON result files
    if args.use_existing:
        print("Using existing result files...")
        for json_file in output_dir.glob("*_results.json"):
            if json_file.exists():
                print(f"  Found: {json_file.name}")
                results_files.append(str(json_file))
        
        if not results_files:
            print("Error: No existing result files found!")
            print(f"Looking in: {output_dir}")
            return
        
        create_comparison_csv(results_files, output_dir)
        
        print("\nGenerated comparison files:")
        print("  - comparison_summary.csv     : Overall accuracy")
        print("  - comparison_categories.csv  : Per-category breakdown")
        print("  - comparison_pairwise.csv    : Pair-by-pair scores")
        print("  - comparison_errors.csv      : All errors by model")
        print("  - comparison_phenomena.csv   : Performance by phenomenon")
        return
    
    # Default model sets
    if args.all_small:
        hf_models = [("gpt2", "clm"), ("distilbert-base-uncased", "mlm")]
        ollama_models = ["qwen2.5:1.5b", "llama3.2:1b"]
    elif args.all_medium:
        hf_models = [("gpt2", "clm"), ("bert-base-uncased", "mlm"), ("roberta-base", "mlm")]
        ollama_models = ["qwen2.5:3b", "deepseek-r1:7b", "llama3.1:8b"]
    else:
        hf_models = []
        ollama_models = []
    
    # Add user-specified models
    if args.hf_models:
        for model in args.hf_models:
            model_type = "clm" if "gpt" in model.lower() else "mlm"
            hf_models.append((model, model_type))
    
    if args.ollama_models:
        ollama_models.extend(args.ollama_models)
    
    # Check if any models specified
    if not hf_models and not ollama_models:
        print("Error: No models specified!")
        print("\nUsage:")
        print("  --use-existing                    : Use existing result files")
        print("  --all-small                       : Run all small models")
        print("  --all-medium                      : Run all medium models")
        print("  --hf-models MODEL1 MODEL2         : Specify HuggingFace models")
        print("  --ollama-models MODEL1 MODEL2     : Specify Ollama models")
        print("\nExample:")
        print("  python3 scripts/compare_all_models.py --data data/extensive_test_pairs.jsonl --use-existing")
        print("  python3 scripts/compare_all_models.py --data data/extensive_test_pairs.jsonl --all-medium")
        return
    
    # Run HuggingFace models
    for model_name, model_type in hf_models:
        result_file = run_hf_model(model_name, model_type, args.data, output_dir)
        if result_file:
            results_files.append(result_file)
    
    # Run Ollama models
    for model_name in ollama_models:
        result_file = run_ollama_model(model_name, args.data, output_dir)
        if result_file:
            results_files.append(result_file)
    
    # Create comparison
    if results_files:
        create_comparison_csv(results_files, output_dir)
        
        print("\nGenerated comparison files:")
        print("  - comparison_summary.csv     : Overall accuracy")
        print("  - comparison_categories.csv  : Per-category breakdown")
        print("  - comparison_pairwise.csv    : Pair-by-pair scores")
        print("  - comparison_errors.csv      : All errors by model")
        print("  - comparison_phenomena.csv   : Performance by phenomenon")

if __name__ == "__main__":
    main()
