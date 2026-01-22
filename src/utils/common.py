import yaml
import logging

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_logger(name: str):
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(name)s - %(message)s')
    return logging.getLogger(name)