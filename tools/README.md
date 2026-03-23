# tools/

This directory contains maintainer-only scripts used to prepare data or model
packages for upload to Zenodo. They are **not needed** for running the
analysis pipelines described in the main README.

| Script | Purpose |
|--------|---------|
| `package_models_for_zenodo.py` | Zip trained nnUNet weights for Zenodo upload; writes to `.tmp/zenodo_packages/` |
