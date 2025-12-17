from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from rag_gemini_reranker import GeminiAssessmentRecommender

# Load .env ONLY for local development
load_dotenv()

# Validate API key (works for local + cloud)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Please configure it as an environment variable."
    )

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Recommends SHL assessments based on job description or query",
    version="1.0"
)

# Initialize recommender ONCE at startup
recommender = GeminiAssessmentRecommender(
    data_path="shl_data_structured.json",
    embedding_file="embeddings.pkl"
)

# ---------- Request & Response Schemas ----------

class RecommendationRequest(BaseModel):
    query: str
    top_k: int = 5

class AssessmentResponse(BaseModel):
    rank: int
    url: str
    reasoning: str

# ---------- API Endpoint ----------

@app.post("/recommend", response_model=list[AssessmentResponse])
def recommend_assessments(request: RecommendationRequest):
    try:
        results = recommender.recommend(request.query)

        # Safely limit results
        results = results[: request.top_k]

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
