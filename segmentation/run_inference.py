"""Wrapper kept for `python -m segmentation.run_inference` (see README).

The code lives in cystscan.segmentation.inference.
"""
from cystscan.segmentation.inference import *  # noqa: F401,F403
from cystscan.segmentation.inference import main, run_inference  # noqa: F401

if __name__ == "__main__":
    main()
