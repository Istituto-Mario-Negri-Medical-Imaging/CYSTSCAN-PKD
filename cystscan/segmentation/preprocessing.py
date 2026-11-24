"""preprocessing.py
Prepare a raw microCT scan as nnUNet-formatted input.

Channel 0 (_0000): the microCT scan as-is
Channel 1 (_0001): 2D Sobel edge magnitude, computed per axial slice
                   (replicates ImageJ/Fiji Enhance Edges, normalised to [0, 255])
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def compute_sobel_2d_per_slice(volume: np.ndarray) -> np.ndarray:
    """Compute 2D Sobel edge magnitude per axial slice.

    Replicates ImageJ/Fiji 'Enhance Edges': applies a 2D Sobel filter
    independently to each axial (Z) slice, then normalises the result to
    [0, 255].

    Parameters
    ----------
    volume : np.ndarray
        3D array with shape (Z, Y, X), as returned by
        SimpleITK.GetArrayFromImage().

    Returns
    -------
    np.ndarray
        Sobel magnitude, same shape as input, dtype uint8, values in [0, 255].
    """
    vol = volume.astype(np.float32)
    result = np.zeros_like(vol, dtype=np.float32)

    for z in range(vol.shape[0]):
        sx = ndimage.sobel(vol[z], axis=0)
        sy = ndimage.sobel(vol[z], axis=1)
        result[z] = np.sqrt(sx ** 2 + sy ** 2)

    max_val = result.max()
    if max_val > 0:
        result = result / max_val * 255.0

    return result.astype(np.uint8)


def prepare_nnunet_input(
    scan_path: str | Path,
    output_dir: str | Path,
    case_id: str,
    two_channel: bool = True,
) -> list[Path]:
    """Write a scan (and optionally its Sobel channel) as nnUNet input files.

    Parameters
    ----------
    scan_path : str or Path
        Path to the raw microCT scan (.nii or .nii.gz).
    output_dir : str or Path
        Directory where nnUNet input files will be written.
    case_id : str
        Case identifier used in output filenames (e.g. 'KRATS_001').
    two_channel : bool
        If True, also writes the Sobel edge channel as _0001.nii.gz.
        Set to False for the single-channel model (n5_1C).

    Returns
    -------
    list[Path]
        Paths of the written files ([_0000] or [_0000, _0001]).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sitk_img = sitk.ReadImage(str(scan_path))
    volume = sitk.GetArrayFromImage(sitk_img)  # shape: (Z, Y, X)

    # Channel 0: microCT as-is
    ch0_path = output_dir / f"{case_id}_0000.nii.gz"
    sitk.WriteImage(sitk_img, str(ch0_path))
    written = [ch0_path]

    if two_channel:
        # Channel 1: 2D Sobel magnitude per axial slice
        sobel_vol = compute_sobel_2d_per_slice(volume)
        sitk_sobel = sitk.GetImageFromArray(sobel_vol)
        sitk_sobel.CopyInformation(sitk_img)  # preserve spacing / origin / direction
        ch1_path = output_dir / f"{case_id}_0001.nii.gz"
        sitk.WriteImage(sitk_sobel, str(ch1_path))
        written.append(ch1_path)

    return written
