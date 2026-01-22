import implicit
import scipy.sparse as sparse
import pickle
import pandas as pd
from src.utils.common import get_logger, load_config

logger = get_logger("Retrieval")


def train_retrieval(config_path="configs/model.yaml", data_config_path="configs/data.yaml"):
    conf = load_config(config_path)
    data_conf = load_config(data_config_path)

    try:
        train_df = pd.read_parquet(f"{data_conf['processed_data_path']}/train.parquet")
    except FileNotFoundError:
        logger.error("Train parquet not found. Run transformers.py first.")
        return

    # FIX: Matrix must be (Users, Items) for implicit > 0.6
    # Previously it was (item, user) which caused the swap error
    sparse_user_item = sparse.csr_matrix(
        (train_df['interaction'].astype(float), (train_df['user_idx'], train_df['item_idx']))
    )

    params = conf['retrieval']
    model = implicit.als.AlternatingLeastSquares(
        factors=params['factors'],
        iterations=params['iterations'],
        regularization=params['regularization'],
        alpha=params['alpha'],
        random_state=42
    )

    logger.info("Training ALS...")
    model.fit(sparse_user_item)

    with open(params['artifact_path'], 'wb') as f:
        pickle.dump(model, f)

    # We also need to save the sparse matrix for the API to use during recommendation
    sparse.save_npz("models/sparse_user_item.npz", sparse_user_item)
    logger.info(f"ALS model and sparse matrix saved to models/")


if __name__ == "__main__":
    train_retrieval()