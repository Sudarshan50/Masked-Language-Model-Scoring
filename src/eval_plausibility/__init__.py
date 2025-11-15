"""Evaluation utilities for semantic plausibility scoring.
Provides functions to score sentences using autoregressive (CLM) chain-rule
and masked (MLM) models using the PLL-word-l2r scoring method.
"""

__all__ = [
    "score_sentence_clm",
    "score_sentence_mlm_pll_word_l2r",
]
