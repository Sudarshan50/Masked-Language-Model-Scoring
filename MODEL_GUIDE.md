# Model Evaluation Framework - Quick Reference

## Available Models

### 1. HuggingFace Models (Original)
These run locally using transformers library:

| Model | Type | Size | Speed | Command |
|-------|------|------|-------|---------|
| GPT-2 | CLM | 124M | Fast | `--model gpt2 --type clm` |
| BERT-base | MLM | 110M | Fast | `--model bert-base-uncased --type mlm` |
| RoBERTa-base | MLM | 125M | Fast | `--model roberta-base --type mlm` |
| DistilBERT | MLM | 66M | Very Fast | `--model distilbert-base-uncased --type mlm` |

### 2. Ollama Models (NEW!)
These run via Ollama API:

#### DeepSeek-R1 (OpenAI Alternative)
| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| deepseek-r1:1.5b | 1.5B | ⚡⚡⚡ Fast | Quick tests |
| deepseek-r1:7b | 7B | ⚡⚡ Medium | **Recommended** |
| deepseek-r1:14b | 14B | ⚡ Slow | High accuracy |

#### Qwen2.5 (Alibaba)
| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| qwen2.5:1.5b | 1.5B | ⚡⚡⚡ Fast | Quick tests |
| qwen2.5:3b | 3B | ⚡⚡ Medium | **Recommended** |
| qwen2.5:7b | 7B | ⚡ Slow | High accuracy |

#### Llama 3 (Meta)
| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| llama3.2:1b | 1B | ⚡⚡⚡ Fast | Quick tests |
| llama3.2:3b | 3B | ⚡⚡ Medium | Balance |
| llama3.1:8b | 8B | ⚡ Slow | **Recommended** |

## Quick Start Commands

### Setup (One-time)

1. **Install Ollama** (if using Ollama models):
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server (keep running)
ollama serve
```

2. **Pull a model**:
```bash
# Choose one or more:
ollama pull deepseek-r1:7b    # Recommended
ollama pull qwen2.5:3b        # Fast
ollama pull llama3.1:8b       # Reliable
```

### Run Evaluations

#### HuggingFace Models (Original)
```bash
# GPT-2
python3 scripts/evaluate_blimp.py \
  --model gpt2 --type clm \
  --data data/extensive_test_pairs.jsonl \
  --output results/gpt2_results.json

# BERT
python3 scripts/evaluate_blimp.py \
  --model bert-base-uncased --type mlm \
  --data data/extensive_test_pairs.jsonl \
  --output results/bert_results.json
```

#### Ollama Models (NEW!)
```bash
# DeepSeek-R1 7B
python3 scripts/evaluate_ollama.py \
  --model deepseek-r1:7b \
  --data data/extensive_test_pairs.jsonl \
  --output results/deepseek_7b_results.json

# Qwen2.5 3B
python3 scripts/evaluate_ollama.py \
  --model qwen2.5:3b \
  --data data/extensive_test_pairs.jsonl \
  --output results/qwen_3b_results.json

# Llama 3.1 8B
python3 scripts/evaluate_ollama.py \
  --model llama3.1:8b \
  --data data/extensive_test_pairs.jsonl \
  --output results/llama_8b_results.json
```

#### Compare All Models
```bash
# Compare medium-sized models
python3 scripts/compare_all_models.py \
  --data data/extensive_test_pairs.jsonl \
  --all-medium

# Compare specific models
python3 scripts/compare_all_models.py \
  --data data/extensive_test_pairs.jsonl \
  --hf-models gpt2 bert-base-uncased \
  --ollama-models deepseek-r1:7b qwen2.5:3b
```

## Output Files

Each evaluation creates:
1. **JSON file**: Detailed results with all pair scores
2. **CSV file**: Spreadsheet-ready format with:
   - Summary statistics
   - Per-category breakdown
   - Pair-wise details (sentence, scores, ✓/✗)

## Recommended Workflow

### For Quick Testing (< 2 minutes)
```bash
# Small HuggingFace model
python3 scripts/evaluate_blimp.py --model gpt2 --type clm \
  --data data/extensive_test_pairs.jsonl --output results/gpt2.json

# Small Ollama model
ollama pull qwen2.5:1.5b
python3 scripts/evaluate_ollama.py --model qwen2.5:1.5b \
  --data data/extensive_test_pairs.jsonl --output results/qwen_1.5b.json
```

### For Production Quality (5-15 minutes)
```bash
# Best HuggingFace models
python3 scripts/evaluate_blimp.py --model roberta-base --type mlm \
  --data data/extensive_test_pairs.jsonl --output results/roberta.json

# Best Ollama models
ollama pull deepseek-r1:7b
python3 scripts/evaluate_ollama.py --model deepseek-r1:7b \
  --data data/extensive_test_pairs.jsonl --output results/deepseek_7b.json

ollama pull llama3.1:8b
python3 scripts/evaluate_ollama.py --model llama3.1:8b \
  --data data/extensive_test_pairs.jsonl --output results/llama_8b.json
```

### For Comprehensive Comparison
```bash
# Run all medium models and compare
python3 scripts/compare_all_models.py \
  --data data/extensive_test_pairs.jsonl \
  --all-medium

# Opens: results/model_comparison.csv
```

## Previous Results

Based on earlier runs with 50 test pairs:

| Model | Type | Accuracy | ARG_STRUCTURE | SELECTIONAL_PREF |
|-------|------|----------|---------------|------------------|
| GPT-2 | CLM | **100%** | 100% | 100% |
| RoBERTa-base | MLM | **98%** | 95.8% | 100% |
| BERT-base | MLM | **84%** | 79.2% | 88.5% |

*(Ollama models not yet tested with your data)*

## Troubleshooting

### Ollama Issues

**"Model not found"**
```bash
ollama pull deepseek-r1:7b
```

**"Connection refused"**
```bash
# Make sure Ollama is running
ollama serve
```

**Check available models**
```bash
ollama list
```

### Python Issues

**Import errors**
```bash
pip install -r requirements.txt
```

**Check Python environment**
```bash
python3 --version
which python3
```

## File Structure
```
AIS710/
├── scripts/
│   ├── evaluate_blimp.py      # HuggingFace models
│   ├── evaluate_ollama.py     # Ollama models (NEW!)
│   └── compare_all_models.py  # Compare all (NEW!)
├── data/
│   ├── extensive_test_pairs.jsonl  # 50 test pairs
│   └── minimal_pairs.jsonl         # Original pairs
├── results/
│   ├── *.json                 # Detailed results
│   ├── *.csv                  # Spreadsheet format
│   └── model_comparison.csv   # Comparison table
├── OLLAMA_SETUP.md            # Detailed Ollama guide
└── requirements.txt           # Python dependencies
```

## Next Steps

1. **Install Ollama** (if not done): See OLLAMA_SETUP.md
2. **Pull models** you want to test
3. **Run evaluations** using commands above
4. **Open CSV files** in Excel/Google Sheets to analyze results
5. **Compare performance** across models

For detailed Ollama setup and troubleshooting, see: **OLLAMA_SETUP.md**
