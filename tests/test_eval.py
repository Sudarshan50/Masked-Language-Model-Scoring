"""Unit tests for evaluation scoring functions."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


class TestTokenAlignment:
    """Test token-to-word alignment functionality."""
    
    def test_offset_mapping_basic(self):
        """Test basic offset mapping with simple sentence."""
        from eval_plausibility.eval import _find_token_spans_for_words
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        sentence = "I gave John the button"
        
        spans = _find_token_spans_for_words(tokenizer, sentence)
        
        # Should have 5 words
        assert len(spans) == 5
        
        # Each span should have start, end, and token_ids
        for start, end, token_ids in spans:
            assert start < end
            assert len(token_ids) > 0
    
    def test_multitoken_words(self):
        """Test alignment with words that split into multiple tokens."""
        from eval_plausibility.eval import _find_token_spans_for_words
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        sentence = "The antidisestablishmentarianism is complex"
        
        spans = _find_token_spans_for_words(tokenizer, sentence)
        
        # Should have 4 words
        assert len(spans) == 4
        
        # The long word should be split into multiple tokens
        long_word_span = spans[1]  # "antidisestablishmentarianism"
        assert long_word_span[1] - long_word_span[0] > 1  # Multiple tokens


class TestScoringFunctions:
    """Test scoring functions with mock/small models."""
    
    def test_clm_scoring_returns_float(self):
        """Test CLM scoring returns a valid float."""
        from eval_plausibility.eval import score_sentence_clm
        
        # Use tiny model for testing
        model = "gpt2"
        sentence = "The cat sat on the mat"
        
        score = score_sentence_clm(model, sentence, device="cpu")
        
        assert isinstance(score, float)
        assert score < 0  # Log-probabilities are negative
    
    def test_clm_prefers_plausible(self):
        """Test CLM assigns higher probability to plausible sentence."""
        from eval_plausibility.eval import score_sentence_clm
        
        model = "gpt2"
        good = "I gave John the button"
        bad = "I gave John the wall"
        
        good_score = score_sentence_clm(model, good, device="cpu")
        bad_score = score_sentence_clm(model, bad, device="cpu")
        
        # Good sentence should have higher (less negative) score
        assert good_score > bad_score
    
    def test_mlm_scoring_returns_float(self):
        """Test MLM scoring returns a valid float."""
        from eval_plausibility.eval import score_sentence_mlm_pll_word_l2r
        
        model = "bert-base-uncased"
        sentence = "The cat sat on the mat"
        
        score = score_sentence_mlm_pll_word_l2r(model, sentence, device="cpu", batch_size=4)
        
        assert isinstance(score, float)
        assert score < 0  # Log-probabilities are negative
    
    def test_mlm_prefers_plausible(self):
        """Test MLM assigns higher probability to plausible sentence."""
        from eval_plausibility.eval import score_sentence_mlm_pll_word_l2r
        
        model = "bert-base-uncased"
        good = "She ate the apple"
        bad = "She ate the computer"
        
        good_score = score_sentence_mlm_pll_word_l2r(model, good, device="cpu", batch_size=4)
        bad_score = score_sentence_mlm_pll_word_l2r(model, bad, device="cpu", batch_size=4)
        
        # Good sentence should have higher (less negative) score
        assert good_score > bad_score


class TestBLIMPEvaluator:
    """Test BLIMP evaluator functionality."""
    
    def test_evaluator_initialization(self):
        """Test evaluator can be initialized."""
        from eval_plausibility.blimp_evaluator import BLIMPEvaluator
        
        evaluator = BLIMPEvaluator("bert-base-uncased", "mlm", device="cpu")
        
        assert evaluator.model_name == "bert-base-uncased"
        assert evaluator.model_type == "mlm"
        assert evaluator.device == "cpu"
    
    def test_evaluate_simple_pairs(self):
        """Test evaluation on simple minimal pairs."""
        from eval_plausibility.blimp_evaluator import BLIMPEvaluator
        
        pairs = [
            {"good": "The cat sat", "bad": "The cat building"},
            {"good": "She ate food", "bad": "She ate wall"},
        ]
        
        evaluator = BLIMPEvaluator("bert-base-uncased", "mlm", device="cpu", batch_size=4)
        results = evaluator.evaluate(pairs, show_progress=False)
        
        assert results["total"] == 2
        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0
    
    def test_category_breakdown(self):
        """Test category-wise accuracy breakdown."""
        from eval_plausibility.blimp_evaluator import BLIMPEvaluator
        
        pairs = [
            {"good": "The cat sat", "bad": "The cat building", "category": "CAT1"},
            {"good": "She ate food", "bad": "She ate wall", "category": "CAT2"},
        ]
        
        evaluator = BLIMPEvaluator("bert-base-uncased", "mlm", device="cpu", batch_size=4)
        results = evaluator.evaluate(pairs, show_progress=False)
        
        assert "categories" in results
        assert "CAT1" in results["categories"]
        assert "CAT2" in results["categories"]


class TestDataLoading:
    """Test data loading utilities."""
    
    def test_load_jsonl(self, tmp_path):
        """Test loading JSONL files."""
        from eval_plausibility.blimp_evaluator import load_jsonl
        
        # Create temporary JSONL file
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"good": "A", "bad": "B"}\n{"good": "C", "bad": "D"}\n')
        
        pairs = load_jsonl(str(test_file))
        
        assert len(pairs) == 2
        assert pairs[0]["good"] == "A"
        assert pairs[1]["bad"] == "D"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
