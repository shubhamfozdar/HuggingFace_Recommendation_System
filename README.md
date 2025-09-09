# 🤗 Hugging Face Model Recommender

An interactive web app to help you find the best Hugging Face models for your AI tasks! Powered by Flask, sentence-transformers, and the Hugging Face Hub.
## Features

- 🔍 **Smart Task Detection:** Enter your task in plain English and the app will detect the most relevant Hugging Face pipeline.
- 🚀 **Model Recommendations:** Get a ranked list of top models for your task, with popularity, likes, and detailed descriptions.
- 🎨 **Modern UI:** Beautiful, responsive interface with quick task pills, confidence meter, and animated cards.
- ⚡ **Fast & Cached:** Uses efficient sentence-transformers and caching for quick results.
- 🧑‍💻 **API Access:** Programmatic endpoint for integration with other tools.



## Getting Started
### 1. Clone the repository

```bash
### 2. Set up Python environment

It is recommended to use a virtual environment:
```bash
python -m venv myenv
myenv\Scripts\activate  # On Windows
# or
source myenv/bin/activate  # On Mac/Linux

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the app

```bash
python app.py
# or
py app.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.
## Usage

1. **Describe your task** in the search box (e.g., "I want to classify news articles by topic").
2. **Choose sorting and number of results** (optional).
3. **Browse recommended models** with details, tags, and direct links to Hugging Face.
4. **Try quick task pills** for inspiration.
## API

POST `/api/recommend`
**Request JSON:**
```json
{
	"prompt": "your task description",
	"sort_by": "downloads",  // or "likes"
	"limit": 5
}
```

**Response JSON:**
```json
{
	"task": "detected-task",
	"confidence": 97.5,
	"models": [
		{ "id": "model_id", "name": "...", ... }
	]
}
```

## Technologies Used
- [Flask]
- [sentence-transformers]
- [huggingface_hub]
- [Bootstrap 5]
- [Font Awesome]

