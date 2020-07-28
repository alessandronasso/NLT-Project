import json

def load_config(path):
    with open(path) as f:
        config = json.load(f)
    return config

def save_config(path, config):
    with open(path, "w") as f:
        json.dump(config, f)