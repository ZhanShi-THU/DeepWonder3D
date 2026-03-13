import os
import numpy as np
from deepwonder.VM.utils import *
from natsort import natsorted
import re


def run_view_merging_pipeline(
    DATA,
    SAVE,
    psffit_matrix_file,
    upsample_rate=2,
    dz=1.5,
    Nnum=15,
    min_view_num=5,
    min_loc_num=3,
    cen_id=113,
    cutoff_spatial=20,
    cutoff_temporal=0.75,
):
    '''
    Run the spatiotemporal view-merging and 3D localization test pipeline.

    This function performs the full VM workflow:
        1. Load per-view neuron traces and centroids from ``DATA``.
        2. Merge putative neurons across views using spatiotemporal criteria.
        3. Estimate 3D positions using a precomputed PSF fitting matrix.
        4. Save merged 3D coordinates and cleaned temporal traces to ``SAVE``.

    Args:
        DATA (str): Directory containing the input ``.mat`` files. Each file
            is expected to contain a ``final_mask_list`` structure with per-neuron
            temporal traces and spatial centroids.
        SAVE (str): Output directory where intermediate and final VM results
            (such as ``all_data.mat``, ``R_matrix.mat``, and ``result.mat``)
            will be stored.
        psffit_matrix_file (str): Path to the ``.mat`` file containing the
            PSF fitting matrix (with key ``'psffit_matrix'``).
        upsample_rate (int, optional): Axial upsampling factor used for PSF
            fitting and Z-estimation. Default is 2.
        dz (float, optional): Physical Z-step per PSF plane (e.g., in microns).
            Default is 1.5.
        Nnum (int, optional): Maximum number of candidate views used when
            estimating Z for a given neuron. Default is 15.
        min_view_num (int, optional): Minimum number of distinct views that
            must contribute to a candidate neuron cluster. Default is 5.
        min_loc_num (int, optional): Minimum number of localization events
            required to accept a neuron during Z-estimation. Default is 3.
        cen_id (int, optional): ID of the central field-of-view used as a
            reference view for PSF fitting. Default is 113.
        cutoff_spatial (float, optional): Maximum spatial distance (in pixels)
            used when linking neuron candidates across views. Default is 20.
        cutoff_temporal (float, optional): Minimum temporal correlation between
            traces for them to be clustered together. Default is 0.75.

    Returns:
        tuple:
            - spatial_3D (np.ndarray): Array of shape ``(N, 3)`` containing
              estimated 3D coordinates of valid neurons.
            - valid_single_neuron_trace (np.ndarray): Array of shape
              ``(N, T)`` with temporal traces for the same neurons.
    '''

    # Step 1: Load all data
    print(f'Loading data from {DATA}...')
    os.makedirs(SAVE, exist_ok=True)
    all_mat_path = os.path.join(SAVE, 'all_data.mat')

    if os.path.exists(all_mat_path):
        print(f'{all_mat_path} exists! Loading directly...')
        loaded_data = loadmat_auto(all_mat_path)
        T_trace_array = loaded_data['all_trace']
        S_center_array = loaded_data['S_center_list']
        id_array = loaded_data['all_index']
        view_list = loaded_data['all_view_num']
    else:
        print(f'{all_mat_path} does NOT exist! Loading each .mat file...')

        files_name = [f for f in os.listdir(DATA) if f.endswith('.mat')]
        files_name = natsorted(files_name)

        # extract view IDs from file names: *_view_{id}.mat
        view_list = []
        view_map = {}
        for filename in files_name:
            matched = re.search(r'_view_(\d+)', filename)
            if not matched:
                raise ValueError(f'Failed to parse view id from filename: {filename}')
            view_id = int(matched.group(1))
            view_list.append(view_id)
            view_map[filename] = view_id

        # containers for all neurons across views
        T_trace_list = []
        S_center_list = []
        id_list = []

        # load each view-level .mat file and aggregate neuron data
        for filename in files_name:
            print(f'{filename} loaded ...')
            data = loadmat_auto(os.path.join(DATA, filename))
            final_mask_list = data['final_mask_list']

            for neuron_id in range(final_mask_list.shape[1]):
                temporal_trace = final_mask_list[0, neuron_id]['trace'][0][0]
                spatial_center = final_mask_list[0, neuron_id]['centroid'][0][0]

                T_trace_list.append(np.squeeze(temporal_trace))
                S_center_list.append(np.squeeze(spatial_center))
                id_list.append([view_map[filename], neuron_id])

        # convert to numpy arrays
        T_trace_array = np.array(T_trace_list)
        S_center_array = np.array(S_center_list)
        id_array = np.array(id_list)
        view_array = np.array(view_list)

        # save consolidated data for faster subsequent runs
        print(f'{all_mat_path} saving...')
        savemat(
            all_mat_path,
            {
                'all_trace': T_trace_array,
                'S_center_list': S_center_array,
                'all_index': id_array,
                'all_view_num': view_array,
            },
            do_compression=True,
        )
        print(f'{all_mat_path} saved!')

    # Step 2: Spatiotemporal view merging
    print('Spatiotemporal view merging...')
    R_path = os.path.join(SAVE, 'R_matrix.mat')
    if os.path.exists(R_path):
        print(f'{R_path} exists! Loading directly...')
        R = loadmat_auto(R_path)['R']
    else:
        R = calculate_coef_matrix(T_trace_array, id_array, R_path)

    neuron_group = spatio_temporal_clustering(
        R, id_array, S_center_array, cutoff_spatial, cutoff_temporal, min_view_num
    )
    view_merge_C, view_merge_id, all_single_neuron_trace = group_save(
        neuron_group, S_center_array, id_array, T_trace_array, cutoff_temporal, SAVE
    )

    # Step 3: 3D Localization
    print('Estimating Z...')
    psffit_data = loadmat_auto(psffit_matrix_file)
    psffit_matrix = psffit_data['psffit_matrix']

    spatial_3D, neuron_num, invalid_flag = f_estimateZ(
        view_merge_C,
        view_merge_id,
        psffit_matrix,
        min_loc_num,
        Nnum,
        upsample_rate,
        dz,
        cen_id,
    )
    valid_single_neuron_trace = all_single_neuron_trace[~invalid_flag, :]

    # save the final results
    matpath = os.path.join(SAVE, 'result.mat')
    print(f'Saving final result to {matpath}...')
    savemat(
        matpath,
        {
            'spatial_3D': spatial_3D,
            'all_single_neuron_trace': valid_single_neuron_trace,
        },
    )
    print('Done!')

    return spatial_3D, valid_single_neuron_trace
