import os
import re
import numpy as np
import tifffile as tiff
from scipy.io import savemat
from deepwonder.VM.utils import loadmat_auto


def _load_spatial_coords(neuron_coords_file):
    ext = os.path.splitext(neuron_coords_file)[1].lower()
    if ext == '.mat':
        data = loadmat_auto(neuron_coords_file)
        if 'spatial_3D' in data:
            coords = data['spatial_3D']
        elif 'coords' in data:
            coords = data['coords']
        else:
            raise KeyError('No spatial_3D/coords found in neuron_coords_file')
    elif ext == '.npy':
        coords = np.load(neuron_coords_file)
    elif ext in ('.csv', '.txt'):
        coords = np.loadtxt(neuron_coords_file, delimiter=',')
    else:
        raise ValueError(f'Unsupported coordinate format: {ext}')

    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError('Neuron coordinates should have shape (N, 3+)')
    return coords[:, :3]


def _parse_view_id(filename):
    matched = re.search(r'_view_(\d+)', filename)
    if not matched:
        raise ValueError(f'Failed to parse view id from filename: {filename}')
    return int(matched.group(1))


def _build_view_shifts(psffit_matrix, view_ids, z_values, cen_id, nnum, upsample_rate):
    shifts = np.zeros((len(view_ids), len(z_values), 2), dtype=np.float32)
    z_scale = (z_values / nnum * upsample_rate).astype(np.float32)

    for vi, view_id in enumerate(view_ids):
        if view_id < cen_id:
            ind = np.logical_and(psffit_matrix[:, 0] == view_id, psffit_matrix[:, 1] == cen_id)
            if not np.any(ind):
                continue
            ax = -float(psffit_matrix[ind, 2][0])
            ay = -float(psffit_matrix[ind, 4][0])
        elif view_id > cen_id:
            ind = np.logical_and(psffit_matrix[:, 0] == cen_id, psffit_matrix[:, 1] == view_id)
            if not np.any(ind):
                continue
            ax = float(psffit_matrix[ind, 2][0])
            ay = float(psffit_matrix[ind, 4][0])
        else:
            ax = 0.0
            ay = 0.0

        shifts[vi, :, 0] = ax * z_scale
        shifts[vi, :, 1] = ay * z_scale

    return shifts


def _extract_pixel_traces(stack, ys, xs):
    h, w = stack.shape[1], stack.shape[2]
    ys = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
    xs = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
    return stack[:, ys, xs].T  # (N, T)


def run_fast_trace_pipeline(
    input_path,
    input_folder,
    neuron_coords_file,
    psffit_matrix_file,
    output_dir,
    output_folder='STEP_FAST_TRACE',
    cen_id=113,
    Nnum=15,
    upsample_rate=2,
):
    """Fast trace extraction using pre-known neuron spatial coordinates."""
    data_dir = os.path.join(input_path, input_folder)
    save_dir = os.path.join(output_dir, output_folder)
    os.makedirs(save_dir, exist_ok=True)

    neuron_xyz = _load_spatial_coords(neuron_coords_file)
    lateral = neuron_xyz[:, :2]
    z_values = neuron_xyz[:, 2]

    psf = loadmat_auto(psffit_matrix_file)['psffit_matrix']

    tif_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tif')])
    if len(tif_files) == 0:
        raise FileNotFoundError(f'No .tif files found in {data_dir}')

    view_ids = [_parse_view_id(f) for f in tif_files]
    shifts = _build_view_shifts(psf, view_ids, z_values, cen_id, Nnum, upsample_rate)

    view_traces = []
    for file_i, filename in enumerate(tif_files):
        stack = tiff.imread(os.path.join(data_dir, filename)).astype(np.float32)
        if stack.ndim != 3:
            raise ValueError(f'Expected 3D tif stack (T,H,W), got shape={stack.shape} for {filename}')

        proj = lateral - shifts[file_i]
        ys = proj[:, 0]
        xs = proj[:, 1]
        trace_nt = _extract_pixel_traces(stack, ys, xs)
        view_traces.append(trace_nt)

    all_view_traces = np.stack(view_traces, axis=0)  # (V,N,T)
    merged_trace = np.median(all_view_traces, axis=0)  # (N,T)

    out_mat = os.path.join(save_dir, 'result_fast_trace.mat')
    savemat(
        out_mat,
        {
            'spatial_3D': neuron_xyz,
            'all_single_neuron_trace': merged_trace,
            'all_view_traces': all_view_traces,
            'view_ids': np.array(view_ids),
        },
    )

    return neuron_xyz, merged_trace
