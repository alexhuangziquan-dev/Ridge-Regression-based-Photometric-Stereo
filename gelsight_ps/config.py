"""Configuration loader for gelsight_ps.

Supports three file formats:

- ``.py``          – executed with ``runpy.run_path``; the first dict variable
                     named ``CFG``, ``CONFIG``, or ``config`` is returned.
- ``.yaml`` / ``.yml`` – loaded with ``yaml.safe_load``; top level must be a dict.
- ``.json``        – loaded with ``json.load``; top level must be a dict.
"""

import os
import json
from typing import Any, Dict
from runpy import run_path


def load_config(path: str) -> Dict[str, Any]:
    """Loads a configuration file and returns it as a plain Python dict.

    Args:
        path: Absolute or relative path to the config file.

    Returns:
        A dict containing the full configuration.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file content is malformed or the extension is
            unsupported.
        ImportError: If PyYAML is not installed when a YAML file is requested.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        ns = run_path(path)
        for key in ("CFG", "CONFIG", "config"):
            if key in ns and isinstance(ns[key], dict):
                return ns[key]
        raise ValueError(f"{path}: Python config must define dict variable CFG/CONFIG/config")
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except Exception as e:
            raise ImportError("PyYAML not installed: pip install pyyaml") from e
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: YAML top-level must be a dict")
        return data
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: JSON top-level must be a dict")
        return data
    else:
        raise ValueError(f"Unsupported config extension: {ext}")
