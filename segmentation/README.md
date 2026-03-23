# Segmentation module

Kidney and cyst segmentation of rat microCT scans using pre-trained nnUNet v2 models.

## Requirements

This repository requires **Python 3.10+**. In addition to `nnunetv2` (which itself requires Python 3.8+, torch ≥ 2.2):

```bash
pip install SimpleITK scipy requests tqdm
```

## Models

Five models are available, differing in training set size and input channels:

| Model | Training samples | Input channels | Notes |
|-------|-----------------|----------------|-------|
| `n5`    | 5  | microCT + Sobel | — |
| `n10`   | 10 | microCT + Sobel | — |
| `n15`   | 15 | microCT + Sobel | — |
| `n20`   | 20 | microCT + Sobel | Best generalisation |
| `n5_1C` | 5  | microCT only    | Single-channel baseline |

All models output a 3-class NIfTI mask: `0` = background, `1` = kidney, `2` = cysts.

Weights (~1.2 GB per model) are downloaded automatically from Zenodo on first use.

## Usage

### Segment a scan

```bash
python -m segmentation.run_inference \
    --input  path/to/scan.nii.gz \
    --model  n20 \
    --output path/to/output/ \
    --models-dir ./models/
```

`--models-dir` is where weights are cached (default: `./models/`).

> **Note for Zenodo dataset users:** The test-set images on Zenodo are named
> `KRATS_XXX_0000.nii.gz` (nnUNet channel suffix). Pass the file as-is —
> the script creates the two-channel nnUNet input internally and uses the
> full filename (including `_0000`) as the case identifier, which is harmless.

### Count cysts from the segmentation output

```bash
python -m segmentation.count_cysts \
    --input-dir path/to/output/ \
    --output    cyst_counts.csv
```

Applies the published optimal parameters (from `python/run_evaluation_paper.py`)
to every `.nii.gz` mask in the directory. Prints counts to stdout and optionally
saves a CSV (`case_id, cyst_count`).

### Full pipeline in one go

```bash
OUT=results/my_case

python -m segmentation.run_inference \
    --input scan.nii.gz --model n20 \
    --output $OUT/mask/ --models-dir ./models/

python -m segmentation.count_cysts \
    --input-dir $OUT/mask/ --output $OUT/cyst_counts.csv
```

## Files

| File | Description |
|------|-------------|
| `preprocessing.py` | 2D Sobel edge computation per axial slice + nnUNet input file preparation |
| `download_models.py` | Download weights from Zenodo; set nnUNet environment variables |
| `run_inference.py` | CLI entry point: raw scan → segmentation mask |
| `count_cysts.py` | CLI entry point: segmentation masks → cyst counts |

## Zenodo model weights

Model weights are hosted on Zenodo at
<https://doi.org/10.5281/zenodo.19097241>.
The download URLs are configured in `ZENODO_URLS` inside `download_models.py`.
