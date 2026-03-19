"""download_models.py
Download nnUNet model weights from Zenodo and set up the directory structure.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Zenodo file URLs  (record: https://doi.org/10.5281/zenodo.19097241)
# ---------------------------------------------------------------------------
ZENODO_URLS: dict[str, str | None] = {
    "n5":    "https://zenodo.org/records/19097241/files/nnUNet_n5_weights.zip",
    "n10":   "https://zenodo.org/records/19097241/files/nnUNet_n10_weights.zip",
    "n15":   "https://zenodo.org/records/19097241/files/nnUNet_n15_weights.zip",
    "n20":   "https://zenodo.org/records/19097241/files/nnUNet_n20_weights.zip",
    "n5_1C": "https://zenodo.org/records/19097241/files/nnUNet_n5_1C_weights.zip",
}

# Internal nnUNet subdirectory that the zip extracts into
_RESULTS_SUBPATH = "nnUNet_results"
_DATASET_PATH = (
    "nnUNet_results/Dataset101_Kidney/nnUNetTrainer__nnUNetPlans__3d_fullres"
)


def download_model(model_name: str, dest_dir: str | Path) -> Path:
    """Download and extract a model's weights from Zenodo.

    Parameters
    ----------
    model_name : str
        One of: 'n5', 'n10', 'n15', 'n20', 'n5_1C'.
    dest_dir : str or Path
        Root directory where nnUNet_results/ will be placed.

    Returns
    -------
    Path
        Path to the extracted nnUNet_results/ directory.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    url = ZENODO_URLS.get(model_name)
    if url is None:
        raise ValueError(
            f"No Zenodo URL configured for model '{model_name}'. "
            "Edit ZENODO_URLS in segmentation/download_models.py after uploading to Zenodo."
        )

    zip_path = dest_dir / f"nnUNet_{model_name}_weights.zip"
    if not zip_path.exists():
        print(f"Downloading {model_name} weights from Zenodo ...")
        _download_with_progress(url, zip_path)
    else:
        print(f"Archive already present: {zip_path}")

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    results_dir = dest_dir / _RESULTS_SUBPATH
    if not results_dir.exists():
        raise RuntimeError(
            f"Expected {results_dir} after extraction but it was not found. "
            "Check the zip file structure."
        )

    print(f"Model weights ready at: {results_dir}")
    return results_dir


def _download_with_progress(url: str, dest_path: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest_path.name,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def setup_nnunet_env(models_root: str | Path) -> None:
    """Set nnUNet environment variables pointing at models_root.

    Must be called before running nnUNetv2_predict.

    Parameters
    ----------
    models_root : str or Path
        Directory that contains (or will contain) nnUNet_results/.
    """
    root = Path(models_root).resolve()
    os.environ["nnUNet_results"]      = str(root / "nnUNet_results")
    os.environ["nnUNet_raw"]          = str(root / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(root / "nnUNet_preprocessed")


def model_is_downloaded(model_name: str, models_root: str | Path) -> bool:
    """Return True if all 5 fold checkpoints are present for model_name."""
    results_dir = Path(models_root) / _DATASET_PATH
    if not results_dir.exists():
        return False
    for fold in range(5):
        ckpt = results_dir / f"fold_{fold}" / "checkpoint_best.pth"
        if not ckpt.exists():
            return False
    return True
