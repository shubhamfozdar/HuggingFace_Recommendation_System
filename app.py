from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import list_models
import requests
from functools import lru_cache
import time
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load small sentence similarity model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Define Hugging Face tasks with more detailed descriptions
TASKS = {
    "summarization": "summarize or shorten long text like articles or documents",
    "translation": "translate text from one language to another",
    "text-classification": "classify text into categories or topics",
    "sentiment-analysis": "detect emotions or sentiment in text (positive, negative, neutral)",
    "question-answering": "answer questions based on given context or knowledge",
    "token-classification": "identify entities like names, locations, or dates in text",
    "text-generation": "generate creative or informative text from a prompt",
    "fill-mask": "predict missing words in sentences or paragraphs",
    "image-classification": "identify what's shown in images",
    "zero-shot-classification": "classify text without specific training examples",
}

# Pre-encode task descriptions
task_embeddings = model.encode(list(TASKS.values()), convert_to_tensor=True)
task_names = list(TASKS.keys())

def detect_task(prompt: str):
    """Detect closest Hugging Face task from user prompt with confidence score"""
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    similarities = util.cos_sim(prompt_embedding, task_embeddings)[0]
    best_idx = similarities.argmax().item()
    task = task_names[best_idx]
    score = similarities[best_idx].item()
    return task, score

@lru_cache(maxsize=128)
def fetch_model_details(model_id):
    """Fetch additional details about a model"""
    try:
        api_url = f"https://huggingface.co/api/models/{model_id}"
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            return {
                "description": data.get("description", "No description available"),
                "last_modified": data.get("lastModified", "Unknown"),
                "tags": data.get("tags", []),
                "languages": [tag.replace("language:", "") for tag in data.get("tags", []) if tag.startswith("language:")]
            }
    except Exception as e:
        logger.error(f"Error fetching details for {model_id}: {str(e)}")
    return {}

def recommend_models(prompt: str, sort_by="downloads", limit=5, min_confidence=0.5):
    """Recommend models based on the prompt with enhanced details"""
    task, score = detect_task(prompt)
    
    # If confidence is too low, try to improve the prompt
    if score < min_confidence:
        logger.info(f"Low confidence ({score:.2f}) for task detection. Original task: {task}")
    
    try:
        models = list_models(filter=task, sort=sort_by, direction=-1, limit=limit)
        
        results = []
        for m in models:
            model_data = {
                "id": m.modelId,
                "name": m.modelId.split("/")[-1] if "/" in m.modelId else m.modelId,
                "author": m.modelId.split("/")[0] if "/" in m.modelId else "Unknown",
                "downloads": m.downloads,
                "likes": m.likes,
                "pipeline_tag": m.pipeline_tag,
                "link": f"https://huggingface.co/{m.modelId}",
                "last_modified": getattr(m, "lastModified", "Unknown"),
            }
            
            # Add additional details for top models
            if len(results) < 5:
                details = fetch_model_details(m.modelId)
                model_data.update(details)
                
            results.append(model_data)
            
        return task, score, results
        
    except Exception as e:
        logger.error(f"Error recommending models: {str(e)}")
        return task, score, []

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    detected_task = None
    confidence = None
    sort_by = "downloads"
    limit = 5
    error = None
    user_prompt = ""

    if request.method == "POST":
        user_prompt = request.form["prompt"]
        sort_by = request.form.get("sort_by", "downloads")
        limit = int(request.form.get("limit", 5))

        try:
            task, score, models = recommend_models(
                user_prompt, 
                sort_by=sort_by, 
                limit=limit
            )
            detected_task = task
            confidence = round(score * 100, 2)  # percentage
            recommendations = models
        except Exception as e:
            error = f"An error occurred: {str(e)}"
            logger.error(f"Error processing request: {str(e)}")

    return render_template(
        "index.html",
        recommendations=recommendations,
        task=detected_task,
        confidence=confidence,
        sort_by=sort_by,
        limit=limit,
        prompt=user_prompt,
        error=error,
        tasks=TASKS,
    )

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """API endpoint for programmatic access"""
    try:
        data = request.json
        prompt = data.get("prompt", "")
        sort_by = data.get("sort_by", "downloads")
        limit = int(data.get("limit", 5))
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
            
        task, score, models = recommend_models(prompt, sort_by=sort_by, limit=limit)
        
        return jsonify({
            "task": task,
            "confidence": round(score * 100, 2),
            "models": models
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)