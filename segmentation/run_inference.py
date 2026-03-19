"""run_inference.py
Run nnUNet segmentation on a raw microCT scan.

Usage
-----
python -m segmentation.run_inference \
    --input  path/to/scan.nii.gz \
    --model  n20 \
    --output path/to/output/ \
    --models-dir ./models/

The script downloads the model weights automatically on first use.
After running, pass --output to segmentation/count_cysts.py to count cysts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

# nnUNet inference settings
_DATASET_ID    = "101"
_CONFIGURATION = "3d_fullres"
_CHECKPOINT    = "checkpoint_best.pth"
_FOLDS         = ["0", "1", "2", "3", "4"]

TWO_CHANNEL_MODELS = {"n5", "n10", "n15", "n20"}
ONE_CHANNEL_MODELS = {"n5_1C"}
ALL_MODELS = TWO_CHANNEL_MODELS | ONE_CHANNEL_MODELS


def run_inference(
    input_scan: Path,
    model_name: str,
    output_dir: Path,
    models_dir: Path,
) -> None:
    """Run nnUNet segmentation on a single microCT scan.

    Parameters
    ----------
    input_scan : Path
        Raw microCT scan (.nii or .nii.gz).
    model_name : str
        One of: 'n5', 'n10', 'n15', 'n20', 'n5_1C'.
    output_dir : Path
        Directory where the segmentation mask will be saved.
    models_dir : Path
        Root directory for model weights. Weights are downloaded here if absent.
    """
    _seg_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(_seg_dir))
    from download_models import download_model, model_is_downloaded, setup_nnunet_env
    from preprocessing import prepare_nnunet_input

    if model_name not in ALL_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {sorted(ALL_MODELS)}")

    # Download weights if needed
    if not model_is_downloaded(model_name, models_dir):
        download_model(model_name, models_dir)

    setup_nnunet_env(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive case_id from filename (strip .nii.gz or .nii)
    stem = input_scan.name
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    case_id = stem

    two_channel = model_name in TWO_CHANNEL_MODELS

    with tempfile.TemporaryDirectory(prefix="nnunet_input_") as tmp_in:
        print(f"Preparing nnUNet input (two_channel={two_channel}) ...")
        prepare_nnunet_input(
            scan_path=input_scan,
            output_dir=tmp_in,
            case_id=case_id,
            two_channel=two_channel,
        )

        print(f"Running nnUNetv2_predict (model={model_name}, case={case_id}) ...")
        cmd = [
            "nnUNetv2_predict",
            "-i", tmp_in,
            "-o", str(output_dir),
            "-d", _DATASET_ID,
            "-c", _CONFIGURATION,
            "-f", *_FOLDS,
            "-chk", _CHECKPOINT,
        ]
        subprocess.run(cmd, check=True)

    print(f"\nSegmentation mask saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Segment kidneys and cysts in a microCT scan using nnUNet."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to input microCT scan (.nii or .nii.gz)",
    )
    parser.add_argument(
        "--model", required=True, choices=sorted(ALL_MODELS),
        help="Model to use for segmentation",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Directory where the segmentation mask will be saved",
    )
    parser.add_argument(
        "--models-dir", default=Path("models"), type=Path,
        dest="models_dir",
        help="Directory where model weights are stored/downloaded (default: ./models/)",
    )
    args = parser.parse_args()

    run_inference(
        input_scan=args.input,
        model_name=args.model,
        output_dir=args.output,
        models_dir=args.models_dir,
    )


if __name__ == "__main__":
    main()
