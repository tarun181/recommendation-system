import pandas as pd
import pickle
import os
from src.utils.common import get_logger, load_config

logger = get_logger("Preprocessing")


def process_data(config):
    logger.info("Loading Amazon 5-Core JSON Data...")

    # 1. Load JSON Data
    try:
        df = pd.read_json(config['raw_data_path'], lines=True)
        cols = config['col_map']
        df = df[[cols['user'], cols['item'], cols['rating'], cols['time']]]

        df = df.rename(columns={
            cols['user']: 'user_id',
            cols['item']: 'item_id',
            cols['rating']: 'rating',
            cols['time']: 'timestamp'
        })

    except ValueError as e:
        logger.error(f"Error reading JSON. Check path/format. Error: {e}")
        return

    logger.info(f"Loaded Rows: {len(df)}")

    # 2. Implicit Feedback
    # We treat ratings >= 4 as a "Positive Interaction" (1)
    df = df[df['rating'] >= config['min_rating']].copy()
    df['interaction'] = 1

    # 3. Categorical Encoding
    # Convert string IDs (A3SPTOK...) to Integers (0, 1, 2...)
    df['user_id'] = df['user_id'].astype("category")
    df['item_id'] = df['item_id'].astype("category")

    df['user_idx'] = df['user_id'].cat.codes
    df['item_idx'] = df['item_id'].cat.codes

    # Save Mappings (Crucial for the API to reverse the process)
    user_map = dict(enumerate(df['user_id'].cat.categories))
    item_map = dict(enumerate(df['item_id'].cat.categories))

    os.makedirs("models", exist_ok=True)
    with open("models/user_map.pkl", "wb") as f:
        pickle.dump(user_map, f)
    with open("models/item_map.pkl", "wb") as f:
        pickle.dump(item_map, f)

    # 4. Time-based Split
    # Since 'unixReviewTime' is already an integer, we don't need parsing
    max_time = df['timestamp'].max()
    test_cutoff = max_time - (config['test_days'] * 86400)  # 86400 sec = 1 day

    train = df[df['timestamp'] < test_cutoff]
    test = df[df['timestamp'] >= test_cutoff]

    # 5. Cold Start Cleanup
    # We remove users from Test who were never seen in Train
    train_users = set(train['user_idx'].unique())
    test = test[test['user_idx'].isin(train_users)]

    # We remove items from Test that were never seen in Train
    train_items = set(train['item_idx'].unique())
    test = test[test['item_idx'].isin(train_items)]

    logger.info(f"Final Split - Train: {len(train)}, Test: {len(test)}")

    os.makedirs(config['processed_data_path'], exist_ok=True)
    train.to_parquet(f"{config['processed_data_path']}/train.parquet")
    test.to_parquet(f"{config['processed_data_path']}/test.parquet")


if __name__ == "__main__":
    config = load_config("configs/data.yaml")
    process_data(config)