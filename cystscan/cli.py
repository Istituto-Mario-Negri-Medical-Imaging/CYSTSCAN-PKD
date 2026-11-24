"""Command-line interface: `cystscan count | segment | demo`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cystscan",
        description="Segment and count cysts in micro-CT scans of PKD rat kidneys.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    count = sub.add_parser("count", help="count cysts in a folder of segmentation masks")
    count.add_argument("--input-dir", required=True, type=Path,
                       help="folder with .nii.gz masks (0 background, 1 kidney, 2 cyst)")
    count.add_argument("--output", type=Path, default=None,
                       help="optional CSV file for the counts")

    seg = sub.add_parser("segment", help="segment a scan with an nnU-Net model")
    seg.add_argument("--input", required=True, type=Path, help="micro-CT scan (.nii/.nii.gz)")
    seg.add_argument("--model", required=True, help="n5, n10, n15, n20 or n5_1C")
    seg.add_argument("--output", required=True, type=Path, help="folder for the mask")
    seg.add_argument("--models-dir", type=Path, default=Path("models"),
                     help="folder for model weights (default: ./models)")

    sub.add_parser("demo", help="run the end-to-end demo (not available yet)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "count":
        from cystscan.segmentation.count import count_cysts_in_dir
        count_cysts_in_dir(input_dir=args.input_dir, output_csv=args.output)
    elif args.command == "segment":
        from cystscan.segmentation.inference import run_inference
        run_inference(input_scan=args.input, model_name=args.model,
                      output_dir=args.output, models_dir=args.models_dir)
    elif args.command == "demo":
        print("The demo is not available yet.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
