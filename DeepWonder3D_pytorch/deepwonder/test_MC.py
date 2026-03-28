import os
import numpy as np
import tifffile as tiff
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation


def _clip_shift(dy_dx, max_shift):
    dy, dx = float(dy_dx[0]), float(dy_dx[1])
    dy = np.clip(dy, -max_shift, max_shift)
    dx = np.clip(dx, -max_shift, max_shift)
    return dy, dx


def run_single_view_motion_correction(
    input_path,
    input_folder,
    output_path,
    output_folder='STEP_0_MC',
    max_shift=15,
    upsample_factor=10,
):
    """
    Simple rigid single-view motion correction for (T,H,W) tif stacks.

    Each frame is aligned to a reference image (median over time) using
    phase cross-correlation, then shifted with bilinear interpolation.
    """
    in_dir = os.path.join(input_path, input_folder)
    out_dir = os.path.join(output_path, output_folder)
    os.makedirs(out_dir, exist_ok=True)

    tif_files = sorted([f for f in os.listdir(in_dir) if f.endswith('.tif')])
    if len(tif_files) == 0:
        raise FileNotFoundError(f'No .tif files found in {in_dir}')

    for filename in tif_files:
        stack = tiff.imread(os.path.join(in_dir, filename)).astype(np.float32)
        if stack.ndim != 3:
            raise ValueError(f'Expected tif with shape (T,H,W), got {stack.shape} for {filename}')

        reference = np.median(stack, axis=0)
        corrected = np.zeros_like(stack, dtype=np.float32)

        for t in range(stack.shape[0]):
            moving = stack[t]
            shift_yx, _, _ = phase_cross_correlation(
                reference, moving, upsample_factor=upsample_factor
            )
            dy, dx = _clip_shift(shift_yx, max_shift=max_shift)
            corrected[t] = ndi_shift(moving, shift=(dy, dx), order=1, mode='nearest')

        tiff.imwrite(os.path.join(out_dir, filename), corrected.astype(np.float32))

    return out_dir
