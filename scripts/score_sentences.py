#!/usr/bin/env python3
"""Score individual sentences and output probability/plausibility scores.

This tool takes a list of sentences and returns log-probability scores,
which indicate how plausible/natural each sentence is according to the model.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_plausibility.eval import score_sentence_clm, score_sentence_mlm_pll_word_l2r


def score_sentences(input_file, model_name, model_type, device="cpu", batch_size=16, output_format="table"):
    """Score sentences from input file."""
    
    # Load sentences
    sentences = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                sentences.append(line)
    
    if not sentences:
        print("Error: No sentences found in input file", file=sys.stderr)
        return
    
    print(f"Loaded {len(sentences)} sentences")
    print(f"Model: {model_name} ({model_type.upper()})")
    print(f"Scoring method: {'PLL-word-l2r' if model_type == 'mlm' else 'Chain-rule log-likelihood'}")
    print()
    
    # Score each sentence
    results = []
    for i, sentence in enumerate(sentences, 1):
        if model_type == "clm":
            score = score_sentence_clm(model_name, sentence, device=device)
        else:
            score = score_sentence_mlm_pll_word_l2r(model_name, sentence, device=device, batch_size=batch_size)
        
        # Compute normalized score (per-word)
        num_words = len(sentence.split())
        normalized_score = score / num_words if num_words > 0 else score
        
        results.append({
            "index": i,
            "sentence": sentence,
            "log_probability": score,
            "normalized_score": normalized_score,
            "num_words": num_words
        })
        
        if output_format == "table":
            print(f"{i:3d}. {sentence[:50]:<50} | Score: {score:8.2f} | Per-word: {normalized_score:7.2f}")
    
    # Output results
    if output_format == "json":
        output = {
            "model": model_name,
            "model_type": model_type,
            "num_sentences": len(sentences),
            "results": results
        }
        print(json.dumps(output, indent=2))
    elif output_format == "csv":
        print("\nindex,sentence,log_probability,normalized_score,num_words")
        for r in results:
            # Escape quotes in sentence
            sentence_escaped = r["sentence"].replace('"', '""')
            print(f'{r["index"]},"{sentence_escaped}",{r["log_probability"]:.4f},{r["normalized_score"]:.4f},{r["num_words"]}')
    elif output_format == "ranked":
        print("\n" + "="*80)
        print("SENTENCES RANKED BY PLAUSIBILITY (highest normalized score = most plausible)")
        print("="*80)
        ranked = sorted(results, key=lambda x: x["normalized_score"], reverse=True)
        for rank, r in enumerate(ranked, 1):
            print(f"\n{rank:2d}. Normalized: {r['normalized_score']:7.2f} | Total: {r['log_probability']:8.2f} | Words: {r['num_words']}")
            print(f"    {r['sentence']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Score sentences for semantic plausibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score sentences in a text file
  python score_sentences.py --input sentences.txt --model bert-base-uncased --type mlm
  
  # Output as JSON
  python score_sentences.py --input sentences.txt --model gpt2 --type clm --format json
  
  # Show ranked by plausibility
  python score_sentences.py --input sentences.txt --model bert-base-uncased --type mlm --format ranked
  
  # Output as CSV for spreadsheet
  python score_sentences.py --input sentences.txt --model gpt2 --type clm --format csv > results.csv

Input file format:
  - One sentence per line
  - Lines starting with # are ignored (comments)
  - Empty lines are ignored
        """
    )
    parser.add_argument("--input", "-i", required=True, help="Input file with sentences (one per line)")
    parser.add_argument("--model", "-m", required=True, help="HuggingFace model name")
    parser.add_argument("--type", "-t", choices=["mlm", "clm"], required=True, help="Model type")
    parser.add_argument("--format", "-f", choices=["table", "json", "csv", "ranked"], 
                       default="table", help="Output format")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for MLM scoring")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    # Redirect output if specified
    if args.output:
        sys.stdout = open(args.output, 'w', encoding='utf-8')
    
    try:
        score_sentences(
            input_file=args.input,
            model_name=args.model,
            model_type=args.type,
            device=args.device,
            batch_size=args.batch_size,
            output_format=args.format
        )
    finally:
        if args.output:
            sys.stdout.close()


if __name__ == "__main__":
    main()
