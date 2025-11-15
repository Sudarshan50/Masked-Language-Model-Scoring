"""
Comparative sentence scoring across multiple models.
Runs scoring on all specified models and generates comparative CSV output.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.eval_plausibility.eval import score_sentence_clm, score_sentence_mlm_pll_word_l2r
from tqdm import tqdm


def load_sentences(input_file: str) -> List[str]:
    """Load sentences from file, ignoring comments."""
    sentences = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                sentences.append(line)
    return sentences


def score_with_model(model_config: Dict[str, str], sentences: List[str], device: str = "cpu") -> List[Dict[str, Any]]:
    """Score all sentences with a single model."""
    model_name = model_config['name']
    model_type = model_config['type']
    model_id = model_config['id']
    
    print(f"\nScoring with {model_name}...")
    results = []
    
    for sentence in tqdm(sentences, desc=f"{model_name}"):
        try:
            if model_type == 'clm':
                log_prob = score_sentence_clm(model_id, sentence, device=device)
            else:  # mlm
                log_prob = score_sentence_mlm_pll_word_l2r(model_id, sentence, device=device)
            
            # Count words for normalization
            word_count = len(sentence.split())
            normalized_score = log_prob / word_count if word_count > 0 else log_prob
            
            results.append({
                'sentence': sentence,
                'log_probability': log_prob,
                'normalized_score': normalized_score,
                'word_count': word_count
            })
        except Exception as e:
            print(f"Error scoring '{sentence[:50]}...' with {model_name}: {e}")
            results.append({
                'sentence': sentence,
                'log_probability': None,
                'normalized_score': None,
                'word_count': len(sentence.split())
            })
    
    return results


def save_individual_results(results: List[Dict[str, Any]], model_name: str, output_dir: Path):
    """Save individual model results to CSV."""
    output_file = output_dir / f"{model_name.lower().replace(' ', '_')}_scores.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Sentence', 'Log Probability', 'Normalized Score', 'Word Count'])
        
        # Sort by normalized score (descending - higher is better)
        sorted_results = sorted(results, key=lambda x: x['normalized_score'] if x['normalized_score'] is not None else float('-inf'), reverse=True)
        
        for rank, result in enumerate(sorted_results, 1):
            writer.writerow([
                rank,
                result['sentence'],
                f"{result['log_probability']:.4f}" if result['log_probability'] is not None else "N/A",
                f"{result['normalized_score']:.4f}" if result['normalized_score'] is not None else "N/A",
                result['word_count']
            ])
    
    print(f"Saved {model_name} results to {output_file}")


def save_comparative_results(all_results: Dict[str, List[Dict[str, Any]]], sentences: List[str], output_dir: Path):
    """Save comparative results across all models to CSV."""
    output_file = output_dir / "comparative_scores.csv"
    
    # Prepare headers
    headers = ['Sentence', 'Word Count']
    for model_name in all_results.keys():
        headers.extend([
            f"{model_name} Log Prob",
            f"{model_name} Normalized",
            f"{model_name} Rank"
        ])
    
    # Calculate ranks for each model
    model_ranks = {}
    for model_name, results in all_results.items():
        sorted_results = sorted(results, key=lambda x: x['normalized_score'] if x['normalized_score'] is not None else float('-inf'), reverse=True)
        model_ranks[model_name] = {r['sentence']: rank for rank, r in enumerate(sorted_results, 1)}
    
    # Write comparative CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for sentence in sentences:
            row = [sentence, len(sentence.split())]
            
            for model_name, results in all_results.items():
                # Find result for this sentence
                result = next((r for r in results if r['sentence'] == sentence), None)
                
                if result and result['log_probability'] is not None:
                    row.extend([
                        f"{result['log_probability']:.4f}",
                        f"{result['normalized_score']:.4f}",
                        model_ranks[model_name][sentence]
                    ])
                else:
                    row.extend(["N/A", "N/A", "N/A"])
            
            writer.writerow(row)
    
    print(f"\nSaved comparative results to {output_file}")


def generate_summary_stats(all_results: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Generate summary statistics comparing models."""
    output_file = output_dir / "summary_statistics.json"
    
    summary = {}
    for model_name, results in all_results.items():
        valid_results = [r for r in results if r['normalized_score'] is not None]
        
        if valid_results:
            normalized_scores = [r['normalized_score'] for r in valid_results]
            summary[model_name] = {
                'total_sentences': len(results),
                'successful_scores': len(valid_results),
                'failed_scores': len(results) - len(valid_results),
                'mean_normalized_score': sum(normalized_scores) / len(normalized_scores),
                'min_normalized_score': min(normalized_scores),
                'max_normalized_score': max(normalized_scores),
                'top_5_sentences': [
                    {
                        'sentence': r['sentence'],
                        'normalized_score': r['normalized_score']
                    }
                    for r in sorted(valid_results, key=lambda x: x['normalized_score'], reverse=True)[:5]
                ],
                'bottom_5_sentences': [
                    {
                        'sentence': r['sentence'],
                        'normalized_score': r['normalized_score']
                    }
                    for r in sorted(valid_results, key=lambda x: x['normalized_score'])[:5]
                ]
            }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary statistics to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Comparative sentence scoring across multiple models")
    parser.add_argument("--input", required=True, help="Input file with sentences")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu/cuda)")
    parser.add_argument("--models", nargs='+', default=['gpt2', 'bert'], 
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'bert', 'roberta', 'distilbert'],
                        help="Models to evaluate")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define model configurations
    all_model_configs = {
        'gpt2': {'name': 'GPT-2', 'type': 'clm', 'id': 'gpt2'},
        'gpt2-medium': {'name': 'GPT-2 Medium', 'type': 'clm', 'id': 'gpt2-medium'},
        'gpt2-large': {'name': 'GPT-2 Large', 'type': 'clm', 'id': 'gpt2-large'},
        'bert': {'name': 'BERT', 'type': 'mlm', 'id': 'bert-base-uncased'},
        'roberta': {'name': 'RoBERTa', 'type': 'mlm', 'id': 'roberta-base'},
        'distilbert': {'name': 'DistilBERT', 'type': 'mlm', 'id': 'distilbert-base-uncased'}
    }
    
    # Select requested models
    models_to_eval = [all_model_configs[m] for m in args.models if m in all_model_configs]
    
    print(f"Loading sentences from {args.input}...")
    sentences = load_sentences(args.input)
    print(f"Loaded {len(sentences)} sentences")
    
    print(f"\nEvaluating with {len(models_to_eval)} model(s): {', '.join([m['name'] for m in models_to_eval])}")
    
    # Score with each model
    all_results = {}
    for model_config in models_to_eval:
        results = score_with_model(model_config, sentences, device=args.device)
        all_results[model_config['name']] = results
        
        # Save individual model results
        save_individual_results(results, model_config['name'], output_dir)
    
    # Save comparative results
    if len(models_to_eval) > 1:
        save_comparative_results(all_results, sentences, output_dir)
    
    # Generate summary statistics
    generate_summary_stats(all_results, output_dir)
    
    print("\n✓ Comparative scoring complete!")
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
