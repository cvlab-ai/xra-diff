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


def set_config_path(config_path):
    global CONFIG_PATH
    if isinstance(config_path, str):
        config_path = Path(config_path)
    CONFIG_PATH = config_path
