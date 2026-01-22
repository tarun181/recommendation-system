from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sparse

app = FastAPI(title="Amazon Recommender API")

# Global variables
als_model = None
ranker = None
user_item_matrix = None
item_map = {}  # Dictionary to map {41476 -> 'B000X...'}


@app.on_event("startup")
def load_artifacts():
    global als_model, ranker, user_item_matrix, item_map

    # Load Models
    with open("models/retrieval_model.pkl", "rb") as f:
        als_model = pickle.load(f)
    ranker = lgb.Booster(model_file="models/ranker_model.txt")
    user_item_matrix = sparse.load_npz("models/sparse_user_item.npz")

    # Load Item Mapping (Internal ID -> Amazon ASIN)
    with open("models/item_map.pkl", "rb") as f:
        item_map = pickle.load(f)

    print("✅ All artifacts loaded successfully.")


class Request(BaseModel):
    user_idx: int
    top_k: int = 10


@app.post("/recommend")
def recommend(req: Request):
    if req.user_idx >= user_item_matrix.shape[0]:
        raise HTTPException(status_code=404, detail="User ID not found")

    # 1. Retrieval
    user_history = user_item_matrix[req.user_idx]
    ids, _ = als_model.recommend(req.user_idx, user_history, N=50)

    # 2. Feature Engineering (Simple Dot Product)
    user_vec = als_model.user_factors[req.user_idx]
    item_vecs = als_model.item_factors[ids]
    dot_prods = (user_vec * item_vecs).sum(axis=1)

    features = pd.DataFrame({'dot_prod': dot_prods})

    # 3. Ranking
    final_scores = ranker.predict(features)

    # 4. Sort & Format Output
    sorted_indices = np.argsort(final_scores)[::-1]
    top_ids = ids[sorted_indices][:req.top_k]

    # Convert Internal IDs -> Amazon ASINs
    recommendations = []
    for iid in top_ids:
        asin = item_map.get(iid, "Unknown")
        recommendations.append({
            "internal_id": int(iid),  # numpy int to python int
            "asin": asin,
            "amazon_link": f"https://www.amazon.com/dp/{asin}"
        })

    return {
        "user_idx": req.user_idx,
        "recommendations": recommendations
    }