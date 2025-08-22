import yaml
from pathlib import Path

_CONFIG = None
CONFIG_PATH = Path("config")

def get_config():
    global _CONFIG
    if _CONFIG is None:
        with open(CONFIG_PATH / "model.yaml") as f:
            _CONFIG = yaml.SafeLoader(f.read()).get_data()
    return _CONFIG
