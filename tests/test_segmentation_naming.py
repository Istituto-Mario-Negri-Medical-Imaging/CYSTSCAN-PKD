"""Scan file names map to one case id, with or without the nnU-Net _0000 suffix."""
from pathlib import Path

import pytest

from cystscan.segmentation.inference import case_id_from_path


@pytest.mark.parametrize("name", ["KRATS_024.nii.gz", "KRATS_024_0000.nii.gz",
                                  "KRATS_024.nii", "KRATS_024_0000.nii"])
def test_case_id_drops_extension_and_channel_suffix(name):
    assert case_id_from_path(Path("scans") / name) == "KRATS_024"


def test_other_suffixes_are_kept():
    assert case_id_from_path(Path("KRATS_024_0001.nii.gz")) == "KRATS_024_0001"
