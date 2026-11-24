"""Download nnUNet model weights from Zenodo and set up the directory structure.

Every archive is checked against the md5 that Zenodo lists for it before it
is extracted.
"""

from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Zenodo files (record: https://doi.org/10.5281/zenodo.19097241).
# md5 values copied from https://zenodo.org/api/records/19097241,
# field files[].checksum.
# ---------------------------------------------------------------------------
_RECORD = "https://zenodo.org/records/19097241/files"
ZENODO_FILES: dict[str, tuple[str, str]] = {
    "n5":    (f"{_RECORD}/nnUNet_n5_weights.zip",    "4e7a9e53f0463579b5238136bfbb43cf"),
    "n10":   (f"{_RECORD}/nnUNet_n10_weights.zip",   "135f2c35399b88a9e5a0353442409bf0"),
    "n15":   (f"{_RECORD}/nnUNet_n15_weights.zip",   "521655a76a0424b2fda3107c65c13592"),
    "n20":   (f"{_RECORD}/nnUNet_n20_weights.zip",   "cd4d51fe273ea96be4e9036cc16f9a27"),
    "n5_1C": (f"{_RECORD}/nnUNet_n5_1C_weights.zip", "587c5e0de65d657463a95499044a2e72"),
}
# Kept for code that imported the URL table directly.
ZENODO_URLS: dict[str, str] = {name: url for name, (url, _) in ZENODO_FILES.items()}

# Zenodo refuses some generic HTTP clients; identify the tool.
_USER_AGENT = "cystscan-weights-download"

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

    if model_name not in ZENODO_FILES:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose one of: {', '.join(ZENODO_FILES)}."
        )
    url, md5 = ZENODO_FILES[model_name]

    zip_path = dest_dir / f"nnUNet_{model_name}_weights.zip"
    if not zip_path.exists():
        print(f"Downloading {model_name} weights from Zenodo ...")
        part = zip_path.with_name(zip_path.name + ".part")
        try:
            _download_with_progress(url, part)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        part.replace(zip_path)
    else:
        print(f"Archive already present: {zip_path}")

    if file_md5(zip_path) != md5:
        zip_path.unlink()
        raise RuntimeError(
            f"checksum mismatch for {model_name}: the archive was deleted; "
            "run the download again."
        )

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    results_dir = dest_dir / _RESULTS_SUBPATH
    if not results_dir.exists():
        raise RuntimeError(
            f"Expected {results_dir} after extraction but it was not found. "
            "Check the zip file structure."
        )

    # Write a marker so model_is_downloaded() can identify this model
    marker = dest_dir / f".model_{model_name}_ok"
    marker.write_text(model_name)

    print(f"Model weights ready at: {results_dir}")
    return results_dir


def file_md5(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hex md5 of a file, read in chunks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _http_get(url: str):
    import requests

    return requests.get(url, stream=True, timeout=60,
                        headers={"User-Agent": _USER_AGENT})


def _download_with_progress(url: str, dest_path: Path) -> None:
    response = _http_get(url)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    try:
        from tqdm import tqdm
        bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                   desc=dest_path.name)
    except ImportError:
        bar = None
    try:
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                if bar is not None:
                    bar.update(len(chunk))
    finally:
        if bar is not None:
            bar.close()


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
    """Return True if model_name has been downloaded and extracted.

    Each successful download writes a small marker file so that different
    models (which share the same nnUNet directory structure) can be
    distinguished.
    """
    marker = Path(models_root) / f".model_{model_name}_ok"
    if not marker.exists():
        return False
    results_dir = Path(models_root) / _DATASET_PATH
    if not results_dir.exists():
        return False
    for fold in range(5):
        ckpt = results_dir / f"fold_{fold}" / "checkpoint_best.pth"
        if not ckpt.exists():
            return False
    return True
