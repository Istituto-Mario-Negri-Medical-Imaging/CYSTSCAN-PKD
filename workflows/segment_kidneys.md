# Workflow: Kidney & Cyst Segmentation

## Objective

Segment kidneys and cysts in a rat microCT scan using a pre-trained nnUNet v2
model, then count the cysts using the published optimal parameters.

## Prerequisites

- Python 3.10+ in the `medimg` conda environment (or any env with nnunetv2)
- Required packages (in addition to nnunetv2):

```bash
pip install SimpleITK scipy requests tqdm
```

- The `segmentation/` module in this repository (no separate installation needed)

---

## Inputs

| Item | Description |
|------|-------------|
| `scan.nii.gz` | Raw microCT scan of the kidney, in NIfTI format |
| Model name | One of: `n5`, `n10`, `n15`, `n20` (2-channel) or `n5_1C` (1-channel) |

> **Which model to use?** `n20` (trained on 20 samples) gives the best
> generalisation performance. Use `n5_1C` only for comparison with the
> single-channel baseline described in the paper.

---

## Step 1 — Run segmentation

```bash
python -m segmentation.run_inference \
    --input  path/to/scan.nii.gz \
    --model  n20 \
    --output path/to/segmentation_output/ \
    --models-dir ./models/
```

On first run the model weights (~1.2 GB) are downloaded automatically from
Zenodo ([doi.org/10.5281/zenodo.19097241](https://doi.org/10.5281/zenodo.19097241))
into `--models-dir`. Subsequent runs reuse the cached weights.

**Output:** a single `.nii.gz` mask in `segmentation_output/` with labels:
- `0` = background
- `1` = kidney parenchyma
- `2` = cysts

---

## Step 2 — Count cysts

```bash
python -m segmentation.count_cysts \
    --input-dir path/to/segmentation_output/ \
    --output    path/to/cyst_counts.csv
```

Point `--input-dir` at the same folder used as `--output` in Step 1.
The script applies the published optimal parameters (from
`python/run_evaluation_paper.py`) to every `.nii.gz` file it finds.

**Output:** printed counts + optional CSV (`case_id, cyst_count`).

---

## Running both steps in sequence

```bash
OUT=results/my_case

python -m segmentation.run_inference \
    --input scan.nii.gz --model n20 \
    --output $OUT/mask/ --models-dir ./models/

python -m segmentation.count_cysts \
    --input-dir $OUT/mask/ --output $OUT/cyst_counts.csv
```

---

## One-time: packaging models for Zenodo (maintainers only)

Before the model weights are on Zenodo, run this script once to create the
upload-ready zip files:

```bash
python tools/package_models_for_zenodo.py
```

Zips are written to `.tmp/zenodo_packages/`. After uploading to Zenodo, fill
in the direct download URLs in `ZENODO_URLS` inside
`segmentation/download_models.py`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No Zenodo URL configured` | Fill in `ZENODO_URLS` (see above) |
| `nnUNetv2_predict: command not found` | Activate the `medimg` conda environment |
| Unexpected cyst counts | Verify the Sobel channel: mean of channel 1 in the preprocessed input should be ≈75.8 (training fingerprint) |
| Wrong model directory structure after extraction | Ensure the zip contains `nnUNet_results/…` at the root |
