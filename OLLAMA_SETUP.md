# Using Ollama Models (DeepSeek, Qwen, Llama)

This guide explains how to evaluate BLIMP minimal pairs using Ollama models including DeepSeek-R1, Qwen2.5, and Llama.

## Prerequisites

### 1. Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Or download from:** https://ollama.com/download

### 2. Install Python Dependencies

```bash
pip install ollama requests
# Or update all dependencies:
pip install -r requirements.txt
```

### 3. Start Ollama Server

```bash
ollama serve
```

Leave this running in a separate terminal.

## Available Models

### DeepSeek-R1 (Reasoning Model)
- `deepseek-r1:1.5b` - Smallest, fastest (1.5B parameters)
- `deepseek-r1:7b` - Good balance (7B parameters) **RECOMMENDED**
- `deepseek-r1:14b` - Larger model (14B parameters)
- `deepseek-r1:32b` - Very large (32B parameters)
- `deepseek-r1:70b` - Largest (70B parameters)

### Qwen2.5 (Alibaba Model)
- `qwen2.5:0.5b` - Tiny (500M parameters)
- `qwen2.5:1.5b` - Small (1.5B parameters)
- `qwen2.5:3b` - Medium (3B parameters) **RECOMMENDED**
- `qwen2.5:7b` - Large (7B parameters)
- `qwen2.5:14b` - Very large (14B parameters)
- `qwen2.5:32b` - Huge (32B parameters)
- `qwen2.5:72b` - Largest (72B parameters)

### Llama 3 (Meta Model)
- `llama3.2:1b` - Smallest (1B parameters)
- `llama3.2:3b` - Small (3B parameters)
- `llama3.1:8b` - Medium (8B parameters) **RECOMMENDED**
- `llama3.1:70b` - Largest (70B parameters)

### Other Models
- `mistral:7b` - Mistral AI model (7B)
- `mixtral:8x7b` - Mixture of Experts (8×7B)
- `phi4:14b` - Microsoft Phi-4 (14B)

## Quick Start

### Step 1: Pull a Model

Choose a model and pull it (this downloads it locally):

```bash
# Recommended starting models:
ollama pull deepseek-r1:7b        # Good reasoning, 7B params
ollama pull qwen2.5:3b            # Fast, decent performance
ollama pull llama3.1:8b           # Popular, reliable

# Smaller models (faster but less accurate):
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5:1.5b
ollama pull llama3.2:1b

# Larger models (slower but more accurate):
ollama pull deepseek-r1:14b
ollama pull qwen2.5:7b
```

### Step 2: Run Evaluation

```bash
# Evaluate with DeepSeek-R1 7B
python3 scripts/evaluate_ollama.py \
  --model deepseek-r1:7b \
  --data data/extensive_test_pairs.jsonl \
  --output results/deepseek_r1_7b_results.json

# Evaluate with Qwen2.5 3B
python3 scripts/evaluate_ollama.py \
  --model qwen2.5:3b \
  --data data/extensive_test_pairs.jsonl \
  --output results/qwen2.5_3b_results.json

# Evaluate with Llama 3.1 8B
python3 scripts/evaluate_ollama.py \
  --model llama3.1:8b \
  --data data/extensive_test_pairs.jsonl \
  --output results/llama3.1_8b_results.json
```

## Usage Examples

### Basic Evaluation
```bash
python3 scripts/evaluate_ollama.py --model deepseek-r1:7b --data data/extensive_test_pairs.jsonl
```

### Save Results to JSON and CSV
```bash
python3 scripts/evaluate_ollama.py \
  --model qwen2.5:3b \
  --data data/extensive_test_pairs.jsonl \
  --output results/qwen_results.json
```
This automatically creates both `.json` and `.csv` files.

### Custom CSV Path
```bash
python3 scripts/evaluate_ollama.py \
  --model llama3.1:8b \
  --data data/minimal_pairs.jsonl \
  --csv results/llama_custom.csv
```

### Disable Progress Bar
```bash
python3 scripts/evaluate_ollama.py \
  --model deepseek-r1:7b \
  --data data/extensive_test_pairs.jsonl \
  --no-progress
```

## Comparing Models

Run multiple models and compare results:

```bash
# Small models comparison
python3 scripts/evaluate_ollama.py --model qwen2.5:1.5b --data data/extensive_test_pairs.jsonl --output results/qwen_1.5b.json
python3 scripts/evaluate_ollama.py --model llama3.2:1b --data data/extensive_test_pairs.jsonl --output results/llama_1b.json
python3 scripts/evaluate_ollama.py --model deepseek-r1:1.5b --data data/extensive_test_pairs.jsonl --output results/deepseek_1.5b.json

# Medium models comparison
python3 scripts/evaluate_ollama.py --model qwen2.5:3b --data data/extensive_test_pairs.jsonl --output results/qwen_3b.json
python3 scripts/evaluate_ollama.py --model deepseek-r1:7b --data data/extensive_test_pairs.jsonl --output results/deepseek_7b.json
python3 scripts/evaluate_ollama.py --model llama3.1:8b --data data/extensive_test_pairs.jsonl --output results/llama_8b.json
```

## Model Recommendations

### For Speed (Testing)
- `qwen2.5:1.5b` - Very fast, reasonable accuracy
- `llama3.2:1b` - Fastest, lower accuracy

### For Balance (Development)
- `qwen2.5:3b` - Good speed/accuracy balance ⭐
- `deepseek-r1:7b` - Better reasoning, slower ⭐
- `llama3.1:8b` - Reliable, well-tested ⭐

### For Accuracy (Production)
- `deepseek-r1:14b` - Strong reasoning
- `qwen2.5:7b` - Very capable
- `llama3.1:70b` - Highest quality (very slow)

## Troubleshooting

### "Model not found"
Pull the model first:
```bash
ollama pull deepseek-r1:7b
```

### "Error connecting to Ollama"
Make sure Ollama is running:
```bash
ollama serve
```

### Check Available Models
```bash
ollama list
```

### Remove a Model (Free Space)
```bash
ollama rm deepseek-r1:70b
```

### Update a Model
```bash
ollama pull deepseek-r1:7b
```

## Performance Notes

- **Smaller models (1-3B)**: Fast evaluation (~30-60 seconds for 50 pairs)
- **Medium models (7-8B)**: Moderate speed (~2-5 minutes for 50 pairs)
- **Large models (14B+)**: Slow (~5-15+ minutes for 50 pairs)

Memory requirements:
- 1-3B models: ~2-4GB RAM
- 7-8B models: ~8-12GB RAM
- 14B+ models: ~16GB+ RAM

## Comparison with HuggingFace Models

You can still use the original HuggingFace models (BERT, GPT-2, RoBERTa):

```bash
# HuggingFace models (original script)
python3 scripts/evaluate_blimp.py --model gpt2 --type clm --data data/extensive_test_pairs.jsonl --output results/gpt2.json
python3 scripts/evaluate_blimp.py --model bert-base-uncased --type mlm --data data/extensive_test_pairs.jsonl --output results/bert.json

# Ollama models (new script)
python3 scripts/evaluate_ollama.py --model deepseek-r1:7b --data data/extensive_test_pairs.jsonl --output results/deepseek.json
python3 scripts/evaluate_ollama.py --model qwen2.5:3b --data data/extensive_test_pairs.jsonl --output results/qwen.json
```

All results are saved in the same CSV format for easy comparison!
