# Quick Start Guide - BLIMP Model Evaluation

## ✅ What You Have Working

### HuggingFace Models (Fast, Local)
- ✅ GPT-2: **100% accuracy** (perfect score!)
- ✅ BERT: **84% accuracy** 
- ✅ RoBERTa: **98% accuracy**

### Ollama Models (NEW! - Running Now)
- 🔄 Qwen2.5 3B: Currently evaluating...

## 📊 Your Current Results

From your `/results/` folder:

| Model | Accuracy | ARG_STRUCTURE | SELECTIONAL_PREF | Speed |
|-------|----------|---------------|------------------|-------|
| GPT-2 | 100% | 100% (24/24) | 100% (26/26) | ⚡⚡⚡ Fast |
| BERT | 84% | 79.2% (19/24) | 88.5% (23/26) | ⚡⚡⚡ Fast |
| Qwen2.5 3B | Running... | - | - | ⚡⚡ Medium |

## 🚀 Running More Evaluations

### Use Bash Terminal (Your Current Setup)

Since you're in bash, always use this prefix for homebrew commands:
```bash
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### Quick Commands

**Check Ollama status:**
```bash
eval "$(/opt/homebrew/bin/brew shellenv)" && ollama list
```

**Pull more models:**
```bash
# Small/Fast models (1-2GB download)
eval "$(/opt/homebrew/bin/brew shellenv)" && ollama pull qwen2.5:1.5b
eval "$(/opt/homebrew/bin/brew shellenv)" && ollama pull llama3.2:1b

# Medium models (4-6GB download) - RECOMMENDED
eval "$(/opt/homebrew/bin/brew shellenv)" && ollama pull deepseek-r1:7b
eval "$(/opt/homebrew/bin/brew shellenv)" && ollama pull llama3.1:8b
```

**Run evaluations:**
```bash
# Qwen 1.5B (fastest)
python3 scripts/evaluate_ollama.py \
  --model qwen2.5:1.5b \
  --data data/extensive_test_pairs.jsonl \
  --output results/qwen_1.5b_results.json

# DeepSeek R1 7B (best reasoning)
python3 scripts/evaluate_ollama.py \
  --model deepseek-r1:7b \
  --data data/extensive_test_pairs.jsonl \
  --output results/deepseek_7b_results.json

# Llama 3.1 8B (reliable)
python3 scripts/evaluate_ollama.py \
  --model llama3.1:8b \
  --data data/extensive_test_pairs.jsonl \
  --output results/llama_8b_results.json
```

### Run More HuggingFace Models

```bash
# RoBERTa (best MLM model)
python3 scripts/evaluate_blimp.py \
  --model roberta-base --type mlm \
  --data data/extensive_test_pairs.jsonl \
  --output results/roberta_results.json

# DistilBERT (faster, smaller)
python3 scripts/evaluate_blimp.py \
  --model distilbert-base-uncased --type mlm \
  --data data/extensive_test_pairs.jsonl \
  --output results/distilbert_results.json
```

## 📁 Output Files

After each run, you get:
- **`results/model_name_results.json`** - Detailed scores
- **`results/model_name_results.csv`** - Excel-friendly format

The CSV contains:
1. **Summary**: Overall accuracy, correct/total
2. **Category Breakdown**: ARG_STRUCTURE vs SELECTIONAL_PREFERENCE
3. **Pair-wise Details**: Each sentence pair with scores and ✓/✗

## ⏱️ Estimated Times

For 50 test pairs:

| Model | Size | Time | Memory |
|-------|------|------|--------|
| GPT-2 | 124M | ~30 sec | 500MB |
| BERT | 110M | ~1 min | 500MB |
| Qwen 1.5B | 1.5B | ~2 min | 2GB |
| Qwen 3B | 3B | ~5 min | 4GB |
| DeepSeek 7B | 7B | ~10 min | 8GB |
| Llama 8B | 8B | ~12 min | 10GB |

## 🔧 Troubleshooting

### "Command not found: ollama"
```bash
eval "$(/opt/homebrew/bin/brew shellenv)" && ollama list
```

### "Model not found"
```bash
eval "$(/opt/homebrew/bin/brew shellenv)" && ollama pull qwen2.5:3b
```

### Check if Ollama is running
```bash
eval "$(/opt/homebrew/bin/brew shellenv)" && brew services list | grep ollama
```

### Restart Ollama
```bash
eval "$(/opt/homebrew/bin/brew shellenv)" && brew services restart ollama
```

### Stop a running evaluation
Press `Ctrl+C` in the terminal

## 📊 Viewing Results

### Open CSV files:
- Double-click any `.csv` file in `results/` folder
- Opens in Excel, Numbers, or Google Sheets
- See all pair-wise details with ✓/✗ marks

### Check current evaluation progress:
The progress bar shows: `Evaluating: 45%|████████░░| 23/50 [02:15<02:45, 6.1s/it]`

## 🎯 Recommended Workflow

1. **Start with fast models** to test setup:
   ```bash
   # Already done:
   ✅ GPT-2
   ✅ BERT
   
   # Running now:
   🔄 Qwen2.5 3B
   ```

2. **Add medium models** for better results:
   ```bash
   eval "$(/opt/homebrew/bin/brew shellenv)" && ollama pull deepseek-r1:7b
   python3 scripts/evaluate_ollama.py --model deepseek-r1:7b \
     --data data/extensive_test_pairs.jsonl \
     --output results/deepseek_7b_results.json
   ```

3. **Compare all results**:
   ```bash
   # All CSV files are in results/ folder
   open results/
   ```

## 📖 More Info

- **Detailed Ollama Guide**: `OLLAMA_SETUP.md`
- **All Models Reference**: `MODEL_GUIDE.md`
- **Your Project**: MSL304 - MediFlow Operations Management

## ✨ Tips

- Smaller models (1-3B) are great for testing
- Medium models (7-8B) give best balance
- Use CSV files for easy comparison
- Run multiple models overnight for comparison
- GPT-2's 100% score is impressive but it's a small model

---

**Current Status**: Qwen2.5 3B is evaluating now. Check progress in terminal!
