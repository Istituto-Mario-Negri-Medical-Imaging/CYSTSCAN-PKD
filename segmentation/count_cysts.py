"""Wrapper kept for `python -m segmentation.count_cysts` (see README).

The code lives in cystscan.segmentation.count.
"""
from cystscan.segmentation.count import count_cysts_in_dir, main  # noqa: F401

if __name__ == "__main__":
    main()
