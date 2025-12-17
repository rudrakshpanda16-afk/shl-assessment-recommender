import pandas as pd
import numpy as np
import json
from urllib.parse import urlparse

from rag_gemini_reranker import GeminiAssessmentRecommender

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
EXCEL_PATH = "Gen_AI Dataset.xlsx"
TOP_K = 5

# -------------------------------------------------
# URL NORMALIZATION
# -------------------------------------------------
def normalize_shl_url(url: str) -> str:
    """
    Canonicalize SHL URLs by:
    - Removing '/solutions'
    - Ignoring protocol + domain
    - Lowercasing
    - Removing trailing slash
    """
    if not isinstance(url, str):
        return ""

    parsed = urlparse(url)
    path = parsed.path.lower().rstrip("/")
    path = path.replace("/solutions", "")
    return path

# -------------------------------------------------
# LOAD EXCEL
# -------------------------------------------------
df = pd.read_excel(EXCEL_PATH)

expected_cols = {"Query", "Assessment_url"}
if not expected_cols.issubset(df.columns):
    raise ValueError(f"Excel must contain columns: {expected_cols}")

# Clean text
df["Query"] = df["Query"].astype(str).str.strip()
df["Assessment_url"] = df["Assessment_url"].astype(str).str.strip()

# Normalize ground-truth URLs
df["normalized_url"] = df["Assessment_url"].apply(normalize_shl_url)

# -------------------------------------------------
# GROUP GROUND TRUTH (PANDAS)
# -------------------------------------------------
gt_df = (
    df.groupby("Query", as_index=False)
      .agg({"normalized_url": lambda x: set(x)})
      .rename(columns={"normalized_url": "ground_truth_urls"})
)

# -------------------------------------------------
# INIT RECOMMENDER
# -------------------------------------------------
recommender = GeminiAssessmentRecommender("shl_data_structured.json")

# -------------------------------------------------
# EVALUATION LOOP
# -------------------------------------------------
results = []

for _, row in gt_df.iterrows():
    query = row["Query"]
    gt_urls = row["ground_truth_urls"]

    print(f"Evaluating query: {query}")

    try:
        recs = recommender.recommend(query)

        # Normalize predicted URLs
        pred_urls = {
            normalize_shl_url(r["url"])
            for r in recs
            if "url" in r
        }

        hits = pred_urls.intersection(gt_urls)
        recall = len(hits) / len(gt_urls) if gt_urls else 0.0

    except Exception as e:
        print(f"Error evaluating query '{query}': {e}")
        recall = 0.0
        hits = set()
        pred_urls = set()

    results.append({
        "query": query,
        "recall": recall,
        "ground_truth_count": len(gt_urls),
        "predicted_count": len(pred_urls),
        "hit_count": len(hits),
        "hits": list(hits)
    })

# -------------------------------------------------
# RESULTS DATAFRAME
# -------------------------------------------------
results_df = pd.DataFrame(results)

mean_recall_percent = results_df["recall"].mean() * 100

print("\n================ EVALUATION SUMMARY ================")
print(f"Total unique queries: {len(results_df)}")
print(f"Mean Recall@{TOP_K}: {mean_recall_percent:.2f}%")

# -------------------------------------------------
# SAVE OUTPUTS
# -------------------------------------------------
results_df.to_excel("evaluation_results.xlsx", index=False)

with open("evaluation_summary.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "mean_recall_percent": mean_recall_percent,
            "per_query_results": results
        },
        f,
        indent=2
    )

print("Saved outputs:")
print("- evaluation_results.xlsx")
print("- evaluation_summary.json")
