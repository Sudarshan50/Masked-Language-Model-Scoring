#!/usr/bin/env python3
"""Compare scores across multiple models for the same sentences."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_plausibility.eval import score_sentence_clm, score_sentence_mlm_pll_word_l2r


def compare_models(sentences, device="cpu"):
    """Compare sentence scores across multiple models."""
    
    models = [
        ("bert-base-uncased", "mlm"),
        ("gpt2", "clm"),
    ]
    
    print("="*90)
    print("MULTI-MODEL SENTENCE SCORING COMPARISON")
    print("="*90)
    print("\nScores are NORMALIZED (divided by number of words)")
    print("Higher (less negative) = more plausible\n")
    
    for sentence in sentences:
        num_words = len(sentence.split())
        print(f"\n{'─'*90}")
        print(f"Sentence: {sentence}")
        print(f"Words: {num_words}")
        print(f"{'─'*90}")
        
        scores = []
        for model_name, model_type in models:
            print(f"Scoring with {model_name} ({model_type})...", end=" ", flush=True)
            
            if model_type == "clm":
                score = score_sentence_clm(model_name, sentence, device=device)
            else:
                score = score_sentence_mlm_pll_word_l2r(model_name, sentence, device=device, batch_size=8)
            
            normalized = score / num_words
            scores.append((model_name, model_type, score, normalized))
            print(f"✓")
        
        print("\nResults:")
        print(f"{'Model':<25} {'Type':<6} {'Total Score':>12} {'Normalized':>12}")
        print("─"*60)
        for model_name, model_type, score, normalized in scores:
            print(f"{model_name:<25} {model_type.upper():<6} {score:>12.2f} {normalized:>12.2f}")
    
    print(f"\n{'='*90}\n")


def main():
    sentences = [
        "The cat sat on the mat.",
        "She ate the apple.",
        "The idea ate the computer.",
        "She drank the chair.",
        "Colorless green ideas sleep furiously.",
        "He drove to work yesterday.",
        "The wall gave John the button.",
    ]
    
    compare_models(sentences)


if __name__ == "__main__":
    main()
