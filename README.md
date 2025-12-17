# SHL Assessment Recommendation System

An intelligent RAG (Retrieval-Augmented Generation) system that recommends SHL assessments based on job descriptions or queries using Google Gemini AI for semantic search and reranking.

## Overview

This project provides an AI-powered recommendation engine for SHL (Saville and Holdsworth Limited) assessments. It uses advanced embedding-based retrieval combined with Gemini AI's reranking capabilities to match job requirements with the most relevant assessment tests from SHL's product catalog.

## Features

- **Semantic Search**: Uses Google's `text-embedding-004` model for vector-based similarity search
- **Intelligent Reranking**: Leverages Gemini 2.5 Flash with Google Search grounding for accurate recommendations
- **REST API**: FastAPI-based endpoint for easy integration
- **Evaluation Framework**: Built-in evaluation scripts to measure recommendation quality
- **Web Scraping**: Automated data collection from SHL product catalog
- **Embedding Caching**: Pre-computed embeddings stored for faster performance

## Tech Stack

- **FastAPI** - Modern, fast web framework for building APIs
- **Google Gemini AI** - For embeddings and reranking
- **BeautifulSoup** - Web scraping
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Uvicorn** - ASGI server

## Installation

### Prerequisites

- Python 3.8+
- Google API Key for Gemini AI

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd shlproject
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Project Structure

```
shlproject/
├── app.py                      # FastAPI application and API endpoints
├── rag_gemini_reranker.py     # Core recommendation engine
├── crawl.py                    # Web scraping script for SHL data
├── evaluate.py                 # Evaluation framework
├── requirements.txt            # Python dependencies
├── shl_data_structured.json   # Structured SHL assessment data
├── embeddings.pkl              # Cached embeddings (generated)
└── README.md                   # This file
```

## Usage

### Running the API Server

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, access:
- **Interactive API docs**: `http://localhost:8000/docs`
- **Alternative docs**: `http://localhost:8000/redoc`

### API Endpoint

#### POST `/recommend`

Get assessment recommendations for a query.

**Request Body:**
```json
{
  "query": "Content Writer required, expert in English and SEO",
  "top_k": 5
}
```

**Response:**
```json
[
  {
    "rank": 1,
    "url": "https://www.shl.com/products/product-catalog/view/...",
    "reasoning": "Recommended because..."
  },
  ...
]
```

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"query": "Software Engineer with Python experience", "top_k": 5}'
```

### Command Line Usage

You can also use the recommendation engine directly:

```python
from rag_gemini_reranker import GeminiAssessmentRecommender

recommender = GeminiAssessmentRecommender(
    data_path="shl_data_structured.json",
    embedding_file="embeddings.pkl"
)

results = recommender.recommend("Software Developer with Java skills")
print(results)
```

### Data Collection

To scrape SHL assessment data:

```bash
python crawl.py
```

This will generate `shl_data_structured.json` containing structured assessment information.

### Evaluation

Run the evaluation script to measure recommendation quality:

```bash
python evaluate.py
```

This requires an Excel file (`Gen_AI Dataset.xlsx`) with ground truth data containing:
- `Query` - Test queries
- `Assessment_url` - Expected assessment URLs

Results are saved to:
- `evaluation_results.xlsx` - Detailed per-query results
- `evaluation_summary.json` - Summary statistics including mean recall

## How It Works

1. **Data Collection**: Web scraping extracts assessment metadata (description, job levels, languages, duration) from SHL's catalog

2. **Embedding Generation**: Assessment descriptions are converted to embeddings using Google's text-embedding-004 model

3. **Retrieval**: For a given query, semantic similarity search retrieves top-k candidate assessments

4. **Reranking**: Gemini 2.5 Flash model with Google Search grounding reranks candidates based on:
   - Relevance to query
   - Job level matching
   - Assessment vs. report filtering
   - URL verification

5. **Response**: Returns top recommendations with reasoning

## Configuration

Key parameters in `rag_gemini_reranker.py`:

- `embedding_model`: Embedding model (default: `"text-embedding-004"`)
- `rerank_model`: Reranking model (default: `"gemini-2.5-flash"`)
- `top_k`: Number of initial candidates retrieved (default: 30)
- `thinking_budget`: Token budget for Gemini reasoning (default: 1024)



