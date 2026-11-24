"""Published cyst-counting parameters, read from the single config file.

The values live only in cystscan/data/published_params.json; Python and
MATLAB both read that file.
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np

PARAM_ORDER = (
    "minPeakSepSmall", "minPeakSepLarge", "largeCystVolThresh",
    "maxPeakDistMerge", "thresholdFraction", "gaussianSigma",
)


def load_published_params(path: str | Path | None = None) -> np.ndarray:
    """Return the six published parameters (shape (6,)) in PARAM_ORDER.

    path: a JSON file with the same shape; default is the packaged config.
    """
    if path is None:
        text = resources.files("cystscan").joinpath(
            "data", "published_params.json").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    config = json.loads(text)
    if tuple(config["order"]) != PARAM_ORDER:
        raise ValueError(f"unexpected parameter order: {config['order']}")
    values = np.asarray(config["values"], dtype=float)
    if values.shape != (6,):
        raise ValueError(f"expected 6 values, got shape {values.shape}")
    return values
