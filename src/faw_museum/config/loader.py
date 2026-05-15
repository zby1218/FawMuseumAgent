from pathlib import Path
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_intents() -> dict:
    with open(_PROJECT_ROOT / "config" / "intents.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_llm_config() -> dict:
    with open(_PROJECT_ROOT / "config" / "llm.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)