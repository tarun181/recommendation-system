import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
from src.utils.common import load_config, get_logger

logger = get_logger("Ranking")


def generate_features_vectorized(user_factors, item_factors, df, n_neg=7):
    """
    Vectorized feature generation (100x faster than loops).
    """
    n_rows = len(df)
    n_users, n_factors = user_factors.shape
    n_items = item_factors.shape[0]

    # 1. Prepare Positive Samples
    # We repeat the user/item indices for the dataframe
    u_indices = df['user_idx'].values
    i_indices = df['item_idx'].values

    # Calculate Dot Product for Positives (Vectorized)
    # (Rows, Factors) * (Rows, Factors) -> Sum along axis 1
    pos_scores = (user_factors[u_indices] * item_factors[i_indices]).sum(axis=1)

    # Create the base arrays for the result
    # We will have (1 positive + n_neg negatives) for each row
    total_samples = n_rows * (1 + n_neg)

    users_arr = np.zeros(total_samples, dtype=int)
    items_arr = np.zeros(total_samples, dtype=int)
    labels_arr = np.zeros(total_samples, dtype=int)
    scores_arr = np.zeros(total_samples, dtype=float)

    # Fill Positives (every (n_neg+1)-th position)
    # Indices: 0, 5, 10... (if n_neg=4)
    step = n_neg + 1
    indices = np.arange(0, total_samples, step)

    users_arr[indices] = u_indices
    items_arr[indices] = i_indices
    labels_arr[indices] = 1
    scores_arr[indices] = pos_scores

    # 2. Generate Negatives
    # We need n_rows * n_neg random items
    # Randomly sample items
    neg_items = np.random.randint(0, n_items, size=n_rows * n_neg)

    # We need corresponding users repeated
    # If users are [A, B], and n_neg=2, we need [A, A, B, B]
    neg_users = np.repeat(u_indices, n_neg)

    # Calculate Dot Product for Negatives
    neg_scores = (user_factors[neg_users] * item_factors[neg_items]).sum(axis=1)

    # Fill Negatives
    # We need to fill positions 1,2,3,4, 6,7,8,9...
    # Create a mask for all positions, set positives to False
    mask = np.ones(total_samples, dtype=bool)
    mask[indices] = False

    users_arr[mask] = neg_users
    items_arr[mask] = neg_items
    labels_arr[mask] = 0  # Label 0
    scores_arr[mask] = neg_scores

    return pd.DataFrame({
        'user_idx': users_arr,
        'item_idx': items_arr,
        'label': labels_arr,
        'dot_prod': scores_arr
    })


def train_ranker(config_path="configs/model.yaml"):
    conf = load_config(config_path)

    try:
        with open(conf['retrieval']['artifact_path'], 'rb') as f:
            als_model = pickle.load(f)
        train_df = pd.read_parquet("data/processed/train.parquet")
    except FileNotFoundError:
        logger.error("Required data or models not found.")
        return

    logger.info(f"Loaded {len(train_df)} training rows. Generating Ranking Features...")

    # Vectorized generation
    rank_df = generate_features_vectorized(als_model.user_factors, als_model.item_factors, train_df)

    logger.info(f"Ranking Dataset Created: {len(rank_df)} rows")

    X = rank_df[['dot_prod']]
    y = rank_df['label']

    # LGBM Grouping
    rank_df = rank_df.sort_values('user_idx')
    q_groups = rank_df.groupby('user_idx').size().to_list()

    params = conf['ranking']
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=params["n_estimators"],  # Increased for real data
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        min_data_in_leaf=params["min_data_in_leaf"],
        random_state=42
    )

    logger.info("Training LightGBM Ranker...")
    ranker.fit(X, y, group=q_groups)

    ranker.booster_.save_model(conf['ranking']['artifact_path'])
    logger.info(f"Ranker saved to {conf['ranking']['artifact_path']}")


if __name__ == "__main__":
    train_ranker()

