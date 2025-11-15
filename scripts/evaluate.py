#!/usr/bin/env python3
"""CLI to evaluate minimal pairs for semantic plausibility using CLM/MLM scoring.

Example:
  python scripts/evaluate.py --model bert-base-uncased --type mlm --data data/minimal_pairs.jsonl
  python scripts/evaluate.py --model gpt2 --type clm --data data/minimal_pairs.jsonl
"""
import argparse
import json
from pathlib import Path

from eval_plausibility.eval import (
    score_sentence_clm,
    score_sentence_mlm_pll_word_l2r,
)


def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append(obj)
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--type", choices=["mlm", "clm"], required=True)
    parser.add_argument("--data", required=True, help="JSONL of minimal pairs: {\"good\":...,\"bad\":...}")
    parser.add_argument("--device", default="cpu", help="device, e.g., cpu or cuda")
    args = parser.parse_args()

    pairs = load_pairs(args.data)
    correct = 0
    total = 0

    for p in pairs:
        good = p["good"]
        bad = p["bad"]

        if args.type == "clm":
            gscore = score_sentence_clm(args.model, good, device=args.device)
            bscore = score_sentence_clm(args.model, bad, device=args.device)
        else:
            gscore = score_sentence_mlm_pll_word_l2r(args.model, good, device=args.device)
            bscore = score_sentence_mlm_pll_word_l2r(args.model, bad, device=args.device)

        # higher score = more probable
        is_correct = gscore > bscore
        total += 1
        if is_correct:
            correct += 1
        print(f"Pair {total}: good={gscore:.4f} bad={bscore:.4f} -> {'OK' if is_correct else 'WRONG'}")

    print(f"Accuracy: {correct}/{total} = {correct/total:.4%}")


if __name__ == "__main__":
    main()
