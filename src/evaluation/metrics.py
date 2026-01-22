import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sparse
from src.utils.common import get_logger, load_config

logger = get_logger("Evaluation")


def calculate_metrics(k=50):
    logger.info("Loading Data & Models for Evaluation...")

    # 1. Load Data
    try:
        test_df = pd.read_parquet("data/processed/test.parquet")
        train_df = pd.read_parquet("data/processed/train.parquet")
    except FileNotFoundError:
        logger.error("Processed data not found. Run preprocessing first.")
        return

    # 2. Load Retrieval Model
    try:
        with open("models/retrieval_model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        logger.error("Retrieval model not found. Run train_als.py first.")
        return

    # 3. Build Training Set Sparse Matrix
    # We need this to "mask" items. If a user bought an item in TRAIN,
    # we should not recommend it again in TEST (or at least, the model knows about it).
    # We pass this to the recommend function to tell it "don't suggest these, show me NEW things".
    train_user_item = sparse.csr_matrix(
        (train_df['interaction'].astype(float), (train_df['user_idx'], train_df['item_idx'])),
        shape=(model.user_factors.shape[0], model.item_factors.shape[0])
    )

    # 4. Prepare Ground Truth (What did users ACTUALLY buy in Test?)
    # Dictionary: { user_idx: [item_idx_1, item_idx_2] }
    test_ground_truth = test_df.groupby('user_idx')['item_idx'].apply(list).to_dict()

    hits = 0
    total_targets = 0

    # We evaluate on a sample if the dataset is huge (e.g., first 1000 users) for speed
    # or all users for accuracy. Let's do all users here.
    logger.info(f"Evaluating on {len(test_ground_truth)} users...")

    for user_id, true_item_ids in test_ground_truth.items():
        # Safety check: if user_id from test is outside model range (shouldn't happen with correct filtering)
        if user_id >= model.user_factors.shape[0]:
            continue

        # Get Recommendations (The "Prediction")
        # filter_already_liked_items=True prevents recommending things from Train
        ids, _ = model.recommend(user_id, train_user_item[user_id], N=k)

        # Calculate Overlap (Intersection)
        relevant_hits = np.intersect1d(true_item_ids, ids)

        hits += len(relevant_hits)
        total_targets += len(true_item_ids)

    # 5. Final Calculation
    recall = hits / total_targets if total_targets > 0 else 0

    logger.info(f"--------------------------------------------------")
    logger.info(f"Global Recall@{k}: {recall:.4f}")
    logger.info(f"--------------------------------------------------")

    return recall


if __name__ == "__main__":
    calculate_metrics(k=40)
