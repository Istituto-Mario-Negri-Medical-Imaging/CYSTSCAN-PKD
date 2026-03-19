"""package_models_for_zenodo.py
One-time script: packages nnUNet model weights for upload to Zenodo.

For each model, includes only the files required for inference:
  nnUNet_results/Dataset101_Kidney/nnUNetTrainer__nnUNetPlans__3d_fullres/
      plans.json
      dataset.json
      dataset_fingerprint.json
      fold_0/checkpoint_best.pth
      fold_1/checkpoint_best.pth
      fold_2/checkpoint_best.pth
      fold_3/checkpoint_best.pth
      fold_4/checkpoint_best.pth

Output: .tmp/zenodo_packages/nnUNet_{name}_weights.zip (~1.2 GB each)

After uploading the zips to Zenodo, fill in ZENODO_URLS in
segmentation/download_models.py with the direct download URLs.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Default model names — source directories are provided via CLI arguments
# (see --help).  Run with --help to see usage.
# ---------------------------------------------------------------------------
MODEL_NAMES = ["n5", "n10", "n15", "n20", "n5_1C"]

_RESULTS_BASE = Path(
    "nnUNet_results/Dataset101_Kidney/nnUNetTrainer__nnUNetPlans__3d_fullres"
)
_JSON_FILES   = ["plans.json", "dataset.json", "dataset_fingerprint.json"]
_NUM_FOLDS    = 5
_CHECKPOINT   = "checkpoint_best.pth"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / ".tmp" / "zenodo_packages"


def _collect_files(source_root: Path) -> list[tuple[Path, str]]:
    """Return (absolute_path, archive_path) pairs for a single model."""
    files: list[tuple[Path, str]] = []
    results_dir = source_root / _RESULTS_BASE

    for name in _JSON_FILES:
        src = results_dir / name
        if src.exists():
            files.append((src, str(_RESULTS_BASE / name)))
        else:
            print(f"  WARNING: {src} not found — skipping.")

    for fold in range(_NUM_FOLDS):
        src = results_dir / f"fold_{fold}" / _CHECKPOINT
        if src.exists():
            files.append((src, str(_RESULTS_BASE / f"fold_{fold}" / _CHECKPOINT)))
        else:
            print(f"  WARNING: {src} not found — skipping.")

    return files


def package_model(model_name: str, source_root: Path, output_dir: Path) -> Path:
    zip_path = output_dir / f"nnUNet_{model_name}_weights.zip"
    print(f"\nPackaging {model_name}  →  {zip_path.name}")

    files = _collect_files(source_root)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for abs_path, arc_path in files:
            size_mb = abs_path.stat().st_size / 1e6
            print(f"  Adding  {arc_path}  ({size_mb:.0f} MB)")
            zf.write(abs_path, arc_path)

    size_gb = zip_path.stat().st_size / 1e9
    print(f"  Done. Archive size: {size_gb:.2f} GB")
    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Package nnUNet model weights for upload to Zenodo. "
            "Provide the root directory that contains the nnUNet_results/ folder "
            "for each model. Models not provided are skipped."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Example:",
            "  python tools/package_models_for_zenodo.py \\",
            "      --n5    /path/to/nnUNet_n5 \\",
            "      --n10   /path/to/nnUNet_n10 \\",
            "      --n20   /path/to/nnUNet_n20",
        ])
    )
    for name in MODEL_NAMES:
        parser.add_argument(
            f"--{name}",
            type=Path,
            default=None,
            metavar="DIR",
            help=f"Root directory for model {name} (contains nnUNet_results/)",
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        metavar="DIR",
        help=f"Output directory for zip files (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out}")

    any_packaged = False
    for name in MODEL_NAMES:
        source_root = getattr(args, name.replace("-", "_"), None)
        if source_root is None:
            continue
        if not source_root.exists():
            print(f"\nWARNING: Source not found for {name} at {source_root} — skipping.")
            continue
        package_model(name, source_root, out)
        any_packaged = True

    if not any_packaged:
        parser.print_help()
        return

    print(
        "\nAll models packaged."
        "\nNext steps:"
        "\n  1. Upload the .zip files to Zenodo."
        "\n  2. Copy the direct download URLs into ZENODO_URLS in"
        "\n     segmentation/download_models.py."
    )


if __name__ == "__main__":
    main()
