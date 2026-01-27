import os
import numpy as np
from deepwonder.VM.utils import *

# ==============================================================================
# Step 0: Global Configuration & Hyperparameters
# ==============================================================================

# --- Optical & PSF (Point Spread Function) Parameters ---
# Axial upsampling factor utilized for high-resolution PSF fitting and Z-estimation
upsample_rate = 2 
# Physical Z-step size (e.g., in microns) between individual PSF planes
dz = 1.5 
# Maximum number of candidate views considered during the Z-estimation process
Nnum = 15 
# The specific ID of the central Field-of-View, used as the reference view
cen_id = 113 

# --- Merging & Filtering Constraints ---
# Minimum number of distinct camera views required to form a valid neuron cluster
min_view_num = 5 
# Minimum number of successful localization events required to accept a neuron's Z-estimate
min_loc_num = 3 
# Maximum allowable Euclidean distance (in pixels) for linking neurons across views
cutoff_spatial = 20  
# Minimum temporal correlation coefficient required to cluster traces together
cutoff_temporal = 0.7

# --- Input/Output Paths ---
# Source directory containing per-view .mat files with traces and centroids
DATA = './result/STEP_6_MN/mat_0' 
# Destination directory for intermediate data and final 3D results
SAVE = './result/STEP_7_VM' 
# Path to the precomputed PSF fitting matrix file
psffit_matrix_file = './pth/psffit_matrix.mat'

# ==============================================================================
# Step 1: Data Aggregation & Preprocessing
# ==============================================================================
# In this phase, we harvest data from individual view files and consolidate 
# them into a single structure to facilitate cross-view comparison.

print(f"Loading data from {DATA}...")
os.makedirs(SAVE, exist_ok=True)
all_mat_path = os.path.join(SAVE, f'all_data.mat')

# Check for an existing aggregate file to bypass redundant processing
if os.path.exists(all_mat_path):
    print(f"Found existing aggregate data at {all_mat_path}. Loading...")
    loaded_data = loadmat_auto(all_mat_path)
    T_trace_array = loaded_data['all_trace']
    S_center_array = loaded_data['S_center_list']
    id_array = loaded_data['all_index']
    view_list = loaded_data['all_view_num']
else:
    print(f"Aggregate file not found. Processing individual .mat files...")

    # Sort files numerically to ensure the view indexing remains consistent
    files_name = [f for f in os.listdir(DATA) if f.endswith('.mat')]
    files_name = natsorted(files_name)

    # Parse view IDs directly from the filenames
    view_list = []
    for filename in files_name:
        ind1 = filename.index('_view_')
        subtmp = filename[ind1 + 6:]
        ind2 = subtmp.index('.')
        view_id = int(subtmp[:ind2])
        view_list.append(view_id)

    # Initialize storage containers for features across all views
    T_trace_list = []
    S_center_list = []
    id_list = []

    # Iterate through each view file to extract per-neuron masks and traces
    for filename in files_name:
        print(f"Extracting data from: {filename}")
        data = loadmat_auto(os.path.join(DATA, filename))
        final_mask_list = data['final_mask_list']

        for neuron_id in range(final_mask_list.shape[1]):
            # Retrieve the raw fluorescence temporal trace
            Temporal_trace = final_mask_list[0, neuron_id]['trace'][0][0]
            # Temporal_trace = zscore(Temporal_trace, axis=1)

            # Retrieve the 2D spatial centroid (X, Y coordinates)
            Spatial_center = final_mask_list[0, neuron_id]['centroid'][0][0]

            T_trace_list.append(np.squeeze(Temporal_trace))
            S_center_list.append(np.squeeze(Spatial_center))
            # Tag each entry with its source view and local neuron index
            id_list.append([view_list[files_name.index(filename)], neuron_id])

    # Cast lists to NumPy arrays for efficient vectorized calculations
    T_trace_array = np.array(T_trace_list)
    S_center_array = np.array(S_center_list)
    id_array = np.array(id_list)
    view_array = np.array(view_list)

    # Save the consolidated dataset for future use (optional)
    print(f"Saving aggregated data to {all_mat_path}...")
    savemat(all_mat_path, {
        'all_trace': T_trace_array,
        'S_center_list': S_center_array,
        'all_index': id_array,
        'all_view_num': view_array
    }, do_compression=True)


# ==============================================================================
# Step 2: Spatiotemporal View Merging
# ==============================================================================
# This stage identifies "putative neurons" by linking signals across views 
# that share both spatial proximity and temporal synchronization.

print(f"Executing spatiotemporal view merging...")
# Construct a correlation matrix to quantify trace similarity across all detections
R_path = os.path.join(SAVE, f'R_matrix.mat')
R = calculate_coef_matrix(T_trace_array, id_array, R_path)

# Cluster neurons using the specified spatial and temporal cutoffs
neuron_group = spatio_temporal_clustering(R, id_array, S_center_array, 
                                          cutoff_spatial, cutoff_temporal, min_view_num)

# Consolidate clusters into merged traces and record their constituent view IDs
view_merge_C, view_merge_id, all_single_neuron_trace = group_save(
    neuron_group, S_center_array, id_array, T_trace_array, cutoff_temporal, SAVE
)


# ==============================================================================
# Step 3: 3D Localization (Axial Z-Estimation)
# ==============================================================================
# By fitting the multi-view data to a PSF model, we resolve the depth (Z-axis) 
# for each merged neuron.

print(f"Estimating axial (Z) coordinates...")
# Load the precomputed PSF calibration matrix
psffit_data = loadmat_auto(psffit_matrix_file)
psffit_matrix = psffit_data['psffit_matrix']

# Calculate 3D coordinates; invalid_flag marks neurons that don't meet localization criteria
spatial_3D, neuron_num, invalid_flag = f_estimateZ(
    view_merge_C, view_merge_id, psffit_matrix,
    min_loc_num, Nnum, upsample_rate, dz, cen_id
)

# Retain only the temporal traces for neurons that were successfully localized in 3D
valid_single_neuron_trace = all_single_neuron_trace[~invalid_flag, :]

# Export final results: 3D coordinates and their corresponding cleaned traces
matpath = os.path.join(SAVE, f"result.mat")
print(f"Saving final 3D localized results to {matpath}...")
savemat(
    matpath,
    {'spatial_3D': spatial_3D, 'all_single_neuron_trace': valid_single_neuron_trace}
)

print('Process complete!')