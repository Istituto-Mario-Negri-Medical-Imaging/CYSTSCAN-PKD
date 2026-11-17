# Segmentation module

Kidney and cyst segmentation of rat microCT scans using pre-trained nnUNet v2 models.

## Requirements

Python 3.10+. From the repository root:

```bash
pip install -e ".[segmentation]"
```

This installs `nnunetv2`, PyTorch, SimpleITK, requests and tqdm. The code lives
in the `cystscan.segmentation` package; the files in this folder are thin
wrappers kept so the old `python -m segmentation.…` commands keep working.

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

Weights (~1.2 GB per model) are downloaded automatically from Zenodo on first use
and checked against the md5 listed in the Zenodo record.

## Usage

### Segment a scan

```bash
cystscan segment \
    --input  path/to/scan.nii.gz \
    --model  n20 \
    --output path/to/output/ \
    --models-dir ./models/
# same as: python -m segmentation.run_inference with the same options
```

`--models-dir` is where weights are cached (default: `./models/`).

File names: the test-set images on Zenodo are named `KRATS_XXX_0000.nii.gz`
(nnU-Net channel suffix). Pass the file as it is. The case id is the file name
without the extension and without a trailing `_0000`, so `KRATS_XXX.nii.gz`
and `KRATS_XXX_0000.nii.gz` both give the mask `KRATS_XXX.nii.gz`.

### Count cysts from the segmentation output

```bash
cystscan count \
    --input-dir path/to/output/ \
    --output    cyst_counts.csv
# same as: python -m segmentation.count_cysts with the same options
```

Applies the published parameters (`cystscan/data/published_params.json`)
to every `.nii.gz` mask in the directory. Prints counts to stdout and optionally
saves a CSV (`case_id, cyst_count`).

### Full pipeline in one go

```bash
OUT=results/my_case

cystscan segment --input scan.nii.gz --model n20 \
    --output $OUT/mask/ --models-dir ./models/

cystscan count --input-dir $OUT/mask/ --output $OUT/cyst_counts.csv
```

## Files

| Module | Wrapper in this folder | Description |
|--------|------------------------|-------------|
| `cystscan.segmentation.preprocessing` | `preprocessing.py` | 2D Sobel edge computation per axial slice + nnUNet input file preparation |
| `cystscan.segmentation.download` | `download_models.py` | Download weights from Zenodo, check md5; set nnUNet environment variables |
| `cystscan.segmentation.inference` | `run_inference.py` | Raw scan → segmentation mask |
| `cystscan.segmentation.count` | `count_cysts.py` | Segmentation masks → cyst counts |

## Zenodo model weights

Model weights are hosted on Zenodo at
<https://doi.org/10.5281/zenodo.19097241>.
URLs and md5 checksums are in `ZENODO_FILES` in `cystscan/segmentation/download.py`.
