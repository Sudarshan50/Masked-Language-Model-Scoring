#!/usr/bin/env python3
"""Demo script showing semantic plausibility scoring in action."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_plausibility.eval import score_sentence_mlm_pll_word_l2r


def demo_minimal_pairs():
    """Demonstrate scoring of classic minimal pairs."""
    print("=" * 70)
    print("SEMANTIC PLAUSIBILITY SCORING DEMO")
    print("=" * 70)
    print("\nUsing BERT (MLM with PLL-word-l2r scoring)")
    print("\nThis demo shows how LLMs can distinguish semantically plausible")
    print("from implausible sentences using distributional semantics.\n")
    
    model = "bert-base-uncased"
    
    pairs = [
        ("I gave John the button.", "I gave John the wall."),
        ("She ate the apple.", "She ate the computer."),
        ("He put the key in his pocket.", "He put the house in his pocket."),
        ("The dog chased the cat.", "The dog chased the theory."),
        ("She opened the door.", "She opened the democracy."),
    ]
    
    print(f"Loading model: {model}...")
    
    correct = 0
    total = len(pairs)
    
    for i, (good, bad) in enumerate(pairs, 1):
        print(f"\n{'─' * 70}")
        print(f"Pair {i}:")
        print(f"  ✓ Plausible:   {good}")
        print(f"  ✗ Implausible: {bad}")
        
        good_score = score_sentence_mlm_pll_word_l2r(model, good, batch_size=8)
        bad_score = score_sentence_mlm_pll_word_l2r(model, bad, batch_size=8)
        
        is_correct = good_score > bad_score
        if is_correct:
            correct += 1
        
        print(f"\n  Scores (log-probability):")
        print(f"    Plausible:   {good_score:7.2f} {'←' if is_correct else ''}")
        print(f"    Implausible: {bad_score:7.2f} {'←' if not is_correct else ''}")
        print(f"  Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    
    print(f"\n{'=' * 70}")
    print(f"Overall: {correct}/{total} correct ({correct/total:.0%})")
    print(f"{'=' * 70}\n")
    
    print("Key Insights:")
    print("• Higher (less negative) log-probability = more plausible")
    print("• Model learned selectional preferences from training data")
    print("• No explicit rules programmed—emergent from distribution")
    print("• Based on the Distributional Hypothesis: 'You shall know a word")
    print("  by the company it keeps' (Firth, 1957)")
    print("\nFor detailed theory, see the research paper included with this code.\n")


if __name__ == "__main__":
    demo_minimal_pairs()
