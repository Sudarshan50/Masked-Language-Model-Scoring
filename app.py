#!/usr/bin/env python3
"""Flask web application for interactive BLIMP evaluation.

This application uses evaluate_ollama.py as the main backend logic for model evaluation.
It provides a web interface with:
- Single sentence pair evaluation
- Bulk CSV evaluation with progress tracking
- Support for both Ollama and HuggingFace models

The core evaluation logic (OllamaEvaluator, is_ollama_model) is imported directly
from scripts/evaluate_ollama.py to ensure consistency with command-line evaluation.
"""

import os
import sys
import json
import csv
import time
import threading
from pathlib import Path
from io import StringIO, BytesIO
from typing import Dict, List

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import ollama
import torch

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

# Import from evaluate_ollama.py - the main backend logic
from evaluate_ollama import OllamaEvaluator, is_ollama_model

try:
    from eval_plausibility.blimp_evaluator import BLIMPEvaluator
except ImportError:
    BLIMPEvaluator = None

app = Flask(__name__)
CORS(app)

# Auto-detect device for HuggingFace models (same logic as evaluate_blimp_hf.py)
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Auto-detected device for HuggingFace models: {DEVICE}")

# Global progress tracker
progress_tracker = {
    "current": 0,
    "total": 0,
    "status": "idle",
    "results": [],
    "current_model": "",
    "current_pair": 0
}


def get_available_models():
    """Get list of available Ollama models and HF models.
    
    Uses the same detection logic as evaluate_ollama.py.
    """
    models = {"ollama": [], "huggingface": []}
    
    # Get Ollama models (same logic as evaluate_ollama.py)
    try:
        ollama_models = ollama.list()
        if isinstance(ollama_models, dict):
            models["ollama"] = [m.get('name', m.get('model', '')) for m in ollama_models.get('models', []) if m.get('name') or m.get('model')]
        else:
            models["ollama"] = [
                getattr(m, 'name', getattr(m, 'model', '')) 
                for m in getattr(ollama_models, 'models', [])
            ]
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
    
    # HuggingFace models - add the ones you have installed
    # These will work if you have transformers and torch installed
    models["huggingface"] = ["gpt2", "roberta-base", "bert-base-uncased"]
    
    return models


def evaluate_single_pair(good_sentence: str, bad_sentence: str, model_names: List[str]) -> List[Dict]:
    """Evaluate a single minimal pair with multiple models using evaluate_ollama.py logic."""
    results = []
    
    for model_name in model_names:
        try:
            # Use the same logic as evaluate_ollama.py
            if is_ollama_model(model_name):
                # Use OllamaEvaluator from evaluate_ollama.py
                evaluator = OllamaEvaluator(model_name=model_name)
                good_score, bad_score, is_correct, time_taken = evaluator.score_pair(good_sentence, bad_sentence)
            else:
                # HuggingFace model
                if BLIMPEvaluator is None:
                    results.append({
                        "model": model_name,
                        "good_score": 0,
                        "bad_score": 0,
                        "verdict": "Error",
                        "time_taken": 0,
                        "error": "BLIMPEvaluator not available"
                    })
                    continue
                
                # Infer model type (same logic as evaluate_ollama.py)
                model_type = "clm" if "gpt" in model_name.lower() else "mlm"
                evaluator = BLIMPEvaluator(model_name=model_name, model_type=model_type, device=DEVICE)
                
                # Use score_pair method from BLIMPEvaluator
                good_score, bad_score, is_correct, time_taken = evaluator.score_pair(good_sentence, bad_sentence)
            
            verdict = "✓ Correct" if good_score > bad_score else "✗ Incorrect"
            
            results.append({
                "model": model_name,
                "good_score": round(good_score, 4),
                "bad_score": round(bad_score, 4),
                "score_difference": round(good_score - bad_score, 4),
                "verdict": verdict,
                "time_taken": round(time_taken, 2)
            })
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"Error evaluating {model_name}: {error_msg}")
            results.append({
                "model": model_name,
                "good_score": 0,
                "bad_score": 0,
                "verdict": "Error",
                "time_taken": 0,
                "error": str(e)
            })
    
    return results


def is_ollama_model(model_name: str) -> bool:
    """Check if a model is available in Ollama."""
    try:
        models = ollama.list()
        if isinstance(models, dict):
            available_models = [m.get('name', '') for m in models.get('models', [])]
        else:
            available_models = [getattr(m, 'name', getattr(m, 'model', '')) for m in getattr(models, 'models', [])]
        
        return any(
            model_name == m or m.startswith(model_name + ':') or m.startswith(model_name)
            for m in available_models if m
        )
    except:
        return False


def evaluate_bulk_csv(csv_content: str, model_names: List[str]):
    """Evaluate bulk CSV file with progress tracking using evaluate_ollama.py logic.
    
    Optimized for M2 chip: Process all pairs for one model before moving to the next.
    This prevents repeatedly loading/unloading models and maximizes GPU efficiency.
    """
    global progress_tracker
    
    # Parse CSV
    csv_file = StringIO(csv_content)
    reader = csv.DictReader(csv_file)
    pairs = list(reader)
    
    progress_tracker["total"] = len(pairs) * len(model_names)
    progress_tracker["current"] = 0
    progress_tracker["status"] = "running"
    progress_tracker["results"] = []
    progress_tracker["current_model"] = ""
    progress_tracker["current_pair"] = 0
    
    # Process all pairs for each model (optimized for M2 chip)
    # This loads each model once and processes all pairs, then moves to next model
    for model_name in model_names:
        if progress_tracker["status"] == "cancelled":
            break
        
        progress_tracker["current_model"] = model_name
        evaluator = None
        
        try:
            # Initialize evaluator once per model
            if is_ollama_model(model_name):
                # Use OllamaEvaluator from evaluate_ollama.py
                evaluator = OllamaEvaluator(model_name=model_name)
                print(f"Loaded Ollama model: {model_name}")
            else:
                # HuggingFace model
                if BLIMPEvaluator is None:
                    print(f"Skipping {model_name}: BLIMPEvaluator not available")
                    progress_tracker["current"] += len(pairs)
                    continue
                
                # Infer model type (same logic as evaluate_ollama.py)
                model_type = "clm" if "gpt" in model_name.lower() else "mlm"
                evaluator = BLIMPEvaluator(model_name=model_name, model_type=model_type, device=DEVICE)
                print(f"Loaded HuggingFace model: {model_name} on {DEVICE}")
            
            # Evaluate all pairs with this model
            for idx, pair in enumerate(pairs, 1):
                if progress_tracker["status"] == "cancelled":
                    break
                
                good = pair.get("good_sentence", pair.get("good", ""))
                bad = pair.get("bad_sentence", pair.get("bad", ""))
                progress_tracker["current_pair"] = idx
                
                try:
                    # Use score_pair method from evaluator
                    good_score, bad_score, is_correct, time_taken = evaluator.score_pair(good, bad)
                    
                    verdict = "✓ Correct" if good_score > bad_score else "✗ Incorrect"
                    
                    progress_tracker["results"].append({
                        "pair_id": idx,
                        "model": model_name,
                        "good_sentence": good,
                        "bad_sentence": bad,
                        "good_score": round(good_score, 4),
                        "bad_score": round(bad_score, 4),
                        "score_difference": round(good_score - bad_score, 4),
                        "verdict": verdict,
                        "time_taken": round(time_taken, 2)
                    })
                    
                except Exception as e:
                    import traceback
                    error_msg = f"{str(e)}\n{traceback.format_exc()}"
                    print(f"Error evaluating pair {idx} with {model_name}: {error_msg}")
                    progress_tracker["results"].append({
                        "pair_id": idx,
                        "model": model_name,
                        "good_sentence": good,
                        "bad_sentence": bad,
                        "good_score": 0,
                        "bad_score": 0,
                        "verdict": "Error",
                        "time_taken": 0,
                        "error": str(e)
                    })
                
                progress_tracker["current"] += 1
            
            # Clean up model from memory before loading next one
            del evaluator
            if DEVICE in ["mps", "cuda"]:
                torch.cuda.empty_cache() if DEVICE == "cuda" else torch.mps.empty_cache()
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"Error loading model {model_name}: {error_msg}")
            # Mark all pairs as errors for this model
            for idx, pair in enumerate(pairs, 1):
                good = pair.get("good_sentence", pair.get("good", ""))
                bad = pair.get("bad_sentence", pair.get("bad", ""))
                progress_tracker["results"].append({
                    "pair_id": idx,
                    "model": model_name,
                    "good_sentence": good,
                    "bad_sentence": bad,
                    "good_score": 0,
                    "bad_score": 0,
                    "verdict": "Error",
                    "time_taken": 0,
                    "error": f"Model loading failed: {str(e)}"
                })
                progress_tracker["current"] += 1
        
        if progress_tracker["status"] == "cancelled":
            break
    
    progress_tracker["status"] = "completed"
    progress_tracker["current_model"] = ""
    progress_tracker["current_pair"] = 0


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/api/models', methods=['GET'])
def api_models():
    """Get available models."""
    models = get_available_models()
    return jsonify(models)


@app.route('/api/evaluate/single', methods=['POST'])
def api_evaluate_single():
    """Evaluate a single minimal pair."""
    data = request.json
    good_sentence = data.get('good_sentence', '')
    bad_sentence = data.get('bad_sentence', '')
    models = data.get('models', [])
    
    if not good_sentence or not bad_sentence:
        return jsonify({"error": "Both sentences are required"}), 400
    
    if not models:
        return jsonify({"error": "At least one model must be selected"}), 400
    
    results = evaluate_single_pair(good_sentence, bad_sentence, models)
    return jsonify({"results": results})


@app.route('/api/evaluate/bulk', methods=['POST'])
def api_evaluate_bulk():
    """Start bulk evaluation from CSV."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    models = request.form.getlist('models[]')
    
    if not models:
        return jsonify({"error": "At least one model must be selected"}), 400
    
    # Read CSV content
    csv_content = file.read().decode('utf-8')
    
    # Start evaluation in background thread
    thread = threading.Thread(target=evaluate_bulk_csv, args=(csv_content, models))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started"})


@app.route('/api/progress', methods=['GET'])
def api_progress():
    """Get current progress of bulk evaluation."""
    return jsonify(progress_tracker)


@app.route('/api/cancel', methods=['POST'])
def api_cancel():
    """Cancel ongoing bulk evaluation."""
    global progress_tracker
    progress_tracker["status"] = "cancelled"
    return jsonify({"status": "cancelled"})


@app.route('/api/download/csv', methods=['GET'])
def api_download_csv():
    """Download results as CSV."""
    if not progress_tracker["results"]:
        return jsonify({"error": "No results available"}), 400
    
    # Generate CSV in memory
    output = StringIO()
    fieldnames = ["pair_id", "model", "good_sentence", "bad_sentence", 
                  "good_score", "bad_score", "score_difference", "verdict", "time_taken"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for result in progress_tracker["results"]:
        row = {k: result.get(k, '') for k in fieldnames}
        writer.writerow(row)
    
    # Convert to bytes for send_file
    output.seek(0)
    byte_output = BytesIO(output.getvalue().encode('utf-8'))
    byte_output.seek(0)
    
    return send_file(
        byte_output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='evaluation_results.csv'
    )


if __name__ == '__main__':
    print("Starting BLIMP Evaluation Web Interface...")
    print("Open your browser at: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
