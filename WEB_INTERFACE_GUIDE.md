# BLIMP Evaluation Web Interface Guide

## Overview

The BLIMP Evaluation Web Interface provides an interactive, user-friendly way to evaluate minimal sentence pairs using multiple language models. It features a beautiful, modern design with real-time progress tracking.

**Architecture:** This web application uses `scripts/evaluate_ollama.py` as the main backend logic, ensuring consistency between command-line and web-based evaluations. The core evaluation classes (`OllamaEvaluator`, `BLIMPEvaluator`) and utility functions (`is_ollama_model`) are imported directly from `evaluate_ollama.py`.

## Features

### 🎯 Single Evaluation Mode
- Enter one good sentence and one bad sentence
- Select multiple models (Ollama or HuggingFace)
- View results in a clean, organized table
- See scores, verdicts, and time taken for each model

### 📊 Bulk Evaluation Mode
- Upload CSV files with multiple sentence pairs
- Real-time progress bar showing evaluation status
- Summary statistics (total pairs, accuracy, avg time)
- Download results as CSV
- Cancel evaluation mid-process

### 🤖 Model Support
- **Ollama Models**: DeepSeek-R1, Qwen2.5, Llama3, Mistral, Phi4, etc.
- **HuggingFace Models**: BERT, RoBERTa, GPT-2, DistilBERT, ALBERT

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Make Sure Ollama is Running

```bash
ollama serve
```

### 3. Pull Required Models (Optional)

```bash
# Example: Pull some Ollama models
ollama pull deepseek-r1:7b
ollama pull qwen2.5:3b
ollama pull llama3.1:8b
```

## Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### Single Evaluation

1. Open your browser to `http://localhost:5000`
2. Click on the **"📝 Single Evaluation"** tab
3. Enter a grammatically correct sentence in "Good Sentence"
4. Enter an ungrammatical sentence in "Bad Sentence"
5. Select one or more models by clicking the model cards
6. Click **"Evaluate"**
7. View results in the table below

**Example:**
- Good: "The cat is sleeping on the mat."
- Bad: "The cat are sleeping on the mat."

### Bulk Evaluation

1. Click on the **"📊 Bulk Evaluation"** tab
2. Prepare a CSV file with columns: `good_sentence` and `bad_sentence`
3. Upload the CSV by clicking the upload area or dragging & dropping
4. Select one or more models
5. Click **"Start Evaluation"**
6. Watch the real-time progress bar
7. View summary statistics and detailed results
8. Click **"💾 Download Results"** to save as CSV

**CSV Format Example:**
```csv
good_sentence,bad_sentence
"The dog runs fast.","The dog run fast."
"She is reading a book.","She are reading a book."
"They have finished their work.","They has finished their work."
```

## API Endpoints

The backend provides RESTful API endpoints:

### `GET /api/models`
Returns available Ollama and HuggingFace models.

**Response:**
```json
{
  "ollama": ["deepseek-r1:7b", "qwen2.5:3b", ...],
  "huggingface": ["bert-base-uncased", "gpt2", ...]
}
```

### `POST /api/evaluate/single`
Evaluates a single minimal pair.

**Request:**
```json
{
  "good_sentence": "The cat is sleeping.",
  "bad_sentence": "The cat are sleeping.",
  "models": ["deepseek-r1:7b", "bert-base-uncased"]
}
```

**Response:**
```json
{
  "results": [
    {
      "model": "deepseek-r1:7b",
      "good_score": 8.5,
      "bad_score": 3.2,
      "score_difference": 5.3,
      "verdict": "✓ Correct",
      "time_taken": 2.1
    },
    ...
  ]
}
```

### `POST /api/evaluate/bulk`
Starts bulk evaluation from CSV file.

**Request:** Multipart form data with:
- `file`: CSV file
- `models[]`: Array of model names

**Response:**
```json
{
  "status": "started"
}
```

### `GET /api/progress`
Returns current progress of bulk evaluation.

**Response:**
```json
{
  "current": 45,
  "total": 100,
  "status": "running",
  "results": [...]
}
```

### `POST /api/cancel`
Cancels ongoing bulk evaluation.

### `GET /api/download/csv`
Downloads bulk evaluation results as CSV.

## Design Features

### 🎨 Aesthetic UI Elements
- **Gradient Headers**: Beautiful purple-blue gradients
- **Card-Based Layout**: Clean, modern card design
- **Smooth Animations**: Fade-in effects and transitions
- **Color-Coded Results**: Green for correct, red for incorrect
- **Hover Effects**: Interactive feedback on all clickable elements

### 📱 Responsive Design
- Works on desktop, tablet, and mobile
- Adaptive grid layouts
- Touch-friendly buttons and controls

### ⚡ Real-Time Updates
- Progress bar updates every 500ms
- Live evaluation count
- Dynamic result rendering

### 🎯 User Experience
- Clear visual hierarchy
- Intuitive tab navigation
- Informative alerts and messages
- Drag-and-drop file upload
- One-click model selection

## Troubleshooting

### Models Not Loading
- Ensure Ollama is running: `ollama serve`
- Check that you have pulled at least one model: `ollama list`

### Evaluation Fails
- Verify the sentences are properly formatted
- Check that the selected models are available
- Look at browser console for error messages

### CSV Upload Issues
- Ensure CSV has headers: `good_sentence` and `bad_sentence`
- Check that the file is valid UTF-8 encoded
- Verify there are no empty rows

### Performance Issues
- Start with fewer models for bulk evaluation
- Use smaller batch sizes
- Consider using lighter models (e.g., 1.5b or 3b variants)

## Technical Details

### Backend Stack
- **Flask**: Web framework
- **Flask-CORS**: Cross-origin resource sharing
- **Ollama Python SDK**: Local model inference
- **Transformers**: HuggingFace model support

### Frontend Stack
- **Vanilla JavaScript**: No framework dependencies
- **CSS3**: Modern animations and gradients
- **Fetch API**: Asynchronous HTTP requests
- **Polling**: Progress updates (500ms interval)

### Progress Tracking
The backend uses a global `progress_tracker` dictionary to maintain state:
- `current`: Current evaluation count
- `total`: Total evaluations to perform
- `status`: idle/running/completed/cancelled
- `results`: Array of evaluation results

Bulk evaluations run in a separate daemon thread to avoid blocking the main Flask process.

## Advanced Usage

### Custom Model Configuration
Edit `app.py` to add more HuggingFace models:

```python
models["huggingface"] = [
    "bert-base-uncased",
    "roberta-base",
    "gpt2",
    "your-custom-model-name"
]
```

### Changing Port
Edit the last line of `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=8080)  # Use port 8080
```

### Production Deployment
For production, use a WSGI server:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Screenshots

### Single Evaluation Tab
Clean interface for evaluating one pair at a time with immediate results.

### Bulk Evaluation Tab
Upload CSV files and track progress with live updates and summary statistics.

### Results Table
Organized, color-coded results showing scores, verdicts, and timing information.

## Contributing

To add new features:
1. Backend: Add endpoints in `app.py`
2. Frontend: Update `templates/index.html`
3. Test with various models and sentence pairs
4. Ensure responsive design is maintained

## License

Same as the main project license.

## Support

For issues or questions:
1. Check this guide first
2. Review error messages in browser console
3. Check Flask logs in terminal
4. Ensure all dependencies are installed correctly

---

**Happy Evaluating! 🎯**
