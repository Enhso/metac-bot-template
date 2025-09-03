from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple, Dict

import yaml

from forecasting_tools import GeneralLlm


def load_bot_config(config_path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load bot and LLM configuration from a YAML file.

    Returns (bot_config, llms) where:
    - bot_config is a flat dict of EnsembleForecaster kwargs (excluding llms)
    - llms is a dict mapping llm names to either GeneralLlm instances or provider strings
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    bot_cfg = data.get("bot", {}) or {}
    llms_cfg = data.get("llms", {}) or {}

    llms: Dict[str, Any] = {}
    for name, spec in llms_cfg.items():
        # Allow shorthand: string provider identifier
        if isinstance(spec, str):
            llms[name] = spec
            continue

        if not isinstance(spec, dict):
            raise TypeError(f"Invalid llm spec for '{name}': expected dict or str, got {type(spec)}")

        spec_copy = dict(spec)
        spec_type = spec_copy.pop("type", "general")

        if spec_type == "general":
            # Remaining keys map to GeneralLlm kwargs
            llms[name] = GeneralLlm(**spec_copy)
        else:
            # Unknown typed config; pass raw dict through so caller can handle if needed
            llms[name] = spec

    return bot_cfg, llms


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "bot_config.yaml"
