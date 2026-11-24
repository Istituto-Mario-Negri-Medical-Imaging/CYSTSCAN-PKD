"""cystscan.segmentation.count (formerly segmentation/count_cysts.py)
Count cysts in nnUNet segmentation masks using the published optimal parameters.

Scans a directory for .nii.gz mask files and applies the cyst-counting
algorithm (detect_cyst_peaks) with the published parameters
(cystscan.counting.PUBLISHED_PARAMS). Does not modify any existing evaluation code.

Usage
-----
python -m segmentation.count_cysts --input-dir path/to/inference_output/
python -m segmentation.count_cysts --input-dir path/to/masks/ --output results.csv

Typical workflow
----------------
1. python -m segmentation.run_inference --input scan.nii.gz --model n20 --output masks/
2. python -m segmentation.count_cysts  --input-dir masks/ --output cyst_counts.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cystscan.counting import PUBLISHED_PARAMS, detect_cyst_peaks


def count_cysts_in_dir(
    input_dir: Path,
    output_csv: Path | None = None,
) -> dict[str, int]:
    """Count cysts in all .nii.gz masks found in input_dir.

    Parameters
    ----------
    input_dir : Path
        Directory containing segmentation masks (.nii.gz).
        Typically the output directory of run_inference.py.
    output_csv : Path, optional
        If provided, results are saved as a two-column CSV
        (case_id, cyst_count).

    Returns
    -------
    dict[str, int]
        Mapping of case_id → cyst count.
    """
    masks = sorted(input_dir.glob("*.nii.gz"))
    if not masks:
        print(f"No .nii.gz files found in {input_dir}")
        return {}

    print(f"Found {len(masks)} mask(s). Using published parameters:")
    print(f"  {PUBLISHED_PARAMS.tolist()}\n")

    results: dict[str, int] = {}
    for mask_path in masks:
        # Strip .nii.gz or .nii extension to get case ID
        stem = mask_path.name
        for ext in (".nii.gz", ".nii"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        case_id = stem

        count = detect_cyst_peaks(str(mask_path), PUBLISHED_PARAMS)
        results[case_id] = count
        print(f"  {case_id}: {count} cyst(s)")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "cyst_count"])
            for case_id, count in results.items():
                writer.writerow([case_id, count])
        print(f"\nResults saved to: {output_csv}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Count cysts in segmentation masks using the published optimal parameters. "
            "Point --input-dir at the folder where run_inference.py saved its masks."
        )
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path,
        help="Directory containing .nii.gz segmentation masks",
    )
    parser.add_argument(
        "--output", default=None, type=Path,
        help="Optional path for a CSV file with results (default: print only)",
    )
    args = parser.parse_args()

    count_cysts_in_dir(
        input_dir=args.input_dir,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
