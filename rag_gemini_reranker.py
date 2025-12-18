import json
import re
import numpy as np
import pandas as pd
import time
import os
import pickle
from dotenv import load_dotenv

# NEW SDK IMPORTS
import google.genai as genai
from google.genai import types

# -------------------------------------------------
# ENV + CLIENT SETUP
# -------------------------------------------------
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API Key not found in environment")

client = genai.Client(api_key=api_key)

# -------------------------------------------------
# RECOMMENDER CLASS
# -------------------------------------------------
class GeminiAssessmentRecommender:
    def __init__(
        self,
        data_path,
        embedding_model="text-embedding-004",
        rerank_model="gemini-2.5-flash",
        embedding_file="embeddings.pkl"
    ):
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.embedding_file = embedding_file

        # 1. Load data
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.df = pd.DataFrame(self.data)

        # 2. Heuristic preprocessing
        self.df["duration_mins"] = self.df["assessment_length"].apply(
            self._extract_minutes
        )

        # 3. Semantic chunks (UNCHANGED)
        self.df["semantic_chunk"] = self.df.apply(
            lambda x: (
                f"URL: {x['url']}\n"
                f"Title: {x['url'].split('/')[-2].replace('-', ' ').title()}\n"
                f"Level: {x['job_levels']}\n"
                f"Description: {x['description']}\n"
                f"Languages: {x['languages']}\n"
                f"Duration: {x['assessment_length']}"
            ),
            axis=1
        )

        # -------------------------------------------------
        # 4. FORCE FRESH EMBEDDINGS (prevents zero-vector bugs)
        # -------------------------------------------------
        print("⚠️ Generating fresh embeddings...")
        self.embeddings = self._generate_document_embeddings(
            self.df["semantic_chunk"].tolist()
        )

        # 🔑 NORMALIZATION (CRITICAL)
        self.embeddings = self._normalize(self.embeddings)

        with open(self.embedding_file, "wb") as f:
            pickle.dump(self.embeddings, f)

        print("✅ Embeddings generated and normalized")

    # -------------------------------------------------
    # EMBEDDINGS
    # -------------------------------------------------
    def _generate_document_embeddings(self, texts, batch_size=50):
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                result = client.models.embed_content(
                    model=self.embedding_model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        title="SHL Assessment Catalog"
                    ),
                )
                all_embeddings.extend([e.values for e in result.embeddings])
                time.sleep(0.5)

            except Exception as e:
                print(f"❌ Embedding error: {e}")
                # DO NOT silently poison similarity
                all_embeddings.extend([[1e-6] * 768] * len(batch))

        return np.array(all_embeddings, dtype=np.float32)

    # -------------------------------------------------
    # UTIL
    # -------------------------------------------------
    def _extract_minutes(self, text):
        if not isinstance(text, str):
            return 999
        match = re.search(r"minutes\s*=\s*(\d+)", text, re.IGNORECASE)
        return int(match.group(1)) if match else 999

    def _normalize(self, vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms

    # -------------------------------------------------
    # RETRIEVAL (FIXED)
    # -------------------------------------------------
    def retrieve_candidates(self, query, top_k=30):
        query_response = client.models.embed_content(
            model=self.embedding_model,
            contents=query,
            config=types.EmbedContentConfig(
                # 🔑 MUST MATCH DOCUMENT TASK
                task_type="RETRIEVAL_QUERY"
            ),
        )

        query_embedding = np.array(
            query_response.embeddings[0].values, dtype=np.float32
        )
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]

        candidates = []
        for idx in top_indices:
            record = self.df.iloc[idx]
            candidates.append({
                "id": int(idx),
                "url": str(record["url"]),
                "content": str(record["semantic_chunk"]),
                "score": float(scores[idx])
            })

        return candidates

    # -------------------------------------------------
    # PHASE 1: GROUNDED RERANKING (UNCHANGED)
    # -------------------------------------------------
    def rerank_grounded(self, query, candidates):
        system_instruction = """
You are an expert Psychometrician and Technical Recruiter.

MANDATORY RULES:
1. You MUST use Google Search to verify every URL before ranking.
2. If a URL cannot be verified, discard or downgrade it.
3. Recommend ONLY assessments (not reports).
4. Entry-level queries must NOT receive senior/managerial tests.
5. Use your reasoning budget to compare candidates.
6. Return a MINIMUM of 7 assessments and MAXIMUM of 10 assessments, but only those which are strictly relevant.

Produce a human-readable ranked analysis.
"""

        prompt = f"""
USER QUERY:
{query}

CANDIDATES:
{json.dumps(candidates, indent=2)}

Rank the best assessments and explain briefly.
"""

        response = client.models.generate_content(
            model=self.rerank_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=1024
                ),
            ),
        )

        return response.text

    # -------------------------------------------------
    # PHASE 2: STRICT JSON EXTRACTION (UNCHANGED)
    # -------------------------------------------------
    def extract_json(self, text):
        extraction_prompt = f"""
Extract the FINAL top 5 recommendations from the text below.

Return STRICT JSON ONLY in this format:
[
  {{
    "id": <int>,
    "rank": <int>,
    "reasoning": "<string>"
  }}
]

TEXT:
{text}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=extraction_prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json"
            ),
        )

        return json.loads(response.text)

    # -------------------------------------------------
    # PIPELINE (UNCHANGED)
    # -------------------------------------------------
    def recommend(self, query):
        candidates = self.retrieve_candidates(query)
        grounded_text = self.rerank_grounded(query, candidates)
        rankings = self.extract_json(grounded_text)

        final = []
        for r in rankings:
            match = next((c for c in candidates if c["id"] == r["id"]), None)
            if not match:
                continue

            final.append({
                "rank": r["rank"],
                "url": match["url"],
                "reasoning": r["reasoning"]
            })

        return final


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    recommender = GeminiAssessmentRecommender("shl_data_structured.json")

    query = "Content Writer required, expert in English and SEO."
    results = recommender.recommend(query)

    print(json.dumps(results, indent=2))
