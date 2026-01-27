import h5py
from scipy.io import loadmat, savemat
import os
import numpy as np
from natsort import natsorted
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.optimize import curve_fit


def is_in(array, element):
    """
    Check if an element exists in an array or list.
    
    This is a simple membership check function that serves as a wrapper around
    Python's 'in' operator for consistency with the codebase.
    
    Args:
        array: Array or list-like object (numpy array, Python list, etc.)
            The container to search within.
        element: Element to check for membership
            Must be compatible with the array element types.
    
    Returns:
        bool: True if element is found in array, False otherwise.
    
    Note:
        This function is equivalent to Python's built-in 'in' operator.
    """
    return element in array


def convert_data_structure(data_list):
    """
    Wrap a Python list into a MATLAB-compatible structure array format.
    
    Converts a Python list of arrays into a numpy object array that can be
    properly saved and loaded by MATLAB's .mat file format.
    
    Args:
        data_list (list): Python list containing arrays or lists
            Format: [array1, array2, ..., arrayN], where each element can be
            a numpy array of arbitrary dimensions or a Python list.
    
    Returns:
        np.ndarray: Object-type array with shape (N, 1)
            Format: numpy array with dtype=object, suitable for saving to
            MATLAB .mat files. Each element contains a numpy array.
    
    Note:
        This function is primarily used for data serialization when saving
        Python data structures to MATLAB-compatible .mat files.
    """
    N = len(data_list)
    new_data = np.empty((N, 1), dtype=object)
    for i in range(N):
        sublist = data_list[i]
        subarray = np.array(sublist)
        wrapped_array = np.array(['123', subarray], dtype=object)
        wrapped_array = wrapped_array[1:]
        new_data[i] = wrapped_array
    return new_data


def loadmat_auto(mat_path):
    """
    Automatically detect MAT file version and load all variables.
    
    Supports both MATLAB v7.3 (HDF5 format) and legacy formats with automatic
    detection. Excludes MATLAB metadata variables (those starting with '__').
    
    Args:
        mat_path (str): Path to the MAT file
            Format: String path, e.g., './data/file.mat'
    
    Returns:
        dict: Dictionary containing all variables from the MAT file
            Format: {variable_name: data}, where variable_name is a string and
            data is a numpy array or other Python object.
            Note: MATLAB metadata variables (__header__, __version__, __globals__)
            are automatically excluded.
    
    Raises:
        FileNotFoundError: If the specified MAT file does not exist.
    
    Note:
        - For v7.3 format (HDF5), uses h5py library and automatically transposes
          data to match MATLAB's column-major storage order.
        - For legacy formats, uses scipy.io.loadmat.
        - All data is converted to numpy array format.
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    # Detect if MAT file is v7.3 (HDF5 format)
    with open(mat_path, 'rb') as f:
        header = f.read(128)
        is_v73 = b'MATLAB 7.3' in header

    variables = {}
    if is_v73:
        # v7.3 use h5py
        with h5py.File(mat_path, 'r') as file:
            for var in file.keys():
                if var.startswith("__"):
                    continue  # Skip MATLAB metadata
                data = file[var][:]
                ndim = data.ndim
                if ndim > 1:
                    data = np.transpose(data, tuple(range(ndim - 1, -1, -1)))
                variables[var] = data
    else:
        # non-v7.3 use scipy.io.loadmat
        mat_data = loadmat(mat_path)
        for var, data in mat_data.items():
            if var.startswith("__"):
                continue  # Skip metadata
            variables[var] = data

    return variables


def calculate_coef_matrix(all_trace, all_index, R_path=''):
    """
    Calculate correlation coefficient matrix and zero out correlations between
    neurons from the same view.
    
    Computes pairwise Pearson correlation coefficients for all neuron pairs.
    If an existing R_matrix.mat file is found, loads it directly to avoid
    redundant computation.
    
    Args:
        all_trace (np.ndarray): Temporal traces for all neurons
            Format: 2D array with shape (n_cells, n_frames)
            - n_cells: Number of neurons
            - n_frames: Number of time frames
            Each row represents a neuron's temporal signal.
        
        all_index (np.ndarray): Index information for all neurons
            Format: 2D array with shape (n_cells, 2) or more columns
            - First column: View ID
            - Subsequent columns: Additional index information (e.g., neuron ID
              within that view)
        
        R_path (str, optional): File path for saving/loading the R matrix
            Format: String path, e.g., './result/R_matrix.mat'
            If empty string, the matrix will not be saved.
    
    Returns:
        R (np.ndarray): Correlation coefficient matrix
            Format: 2D array with shape (n_cells, n_cells)
            - Value range: [-1, 1], representing Pearson correlation coefficients
            - Correlations between neurons from the same view are set to 0
            - Diagonal elements are 1 (self-correlation)
    
    Note:
        - Uses numpy.corrcoef to compute Pearson correlation coefficients
        - Zeroing same-view correlations prevents erroneous merging in subsequent
          clustering steps
        - If R_path exists and file is found, loads directly to save computation time
    """
    if os.path.exists(R_path):
        data = loadmat_auto(R_path)
        R = data['R']
    else:
        # Compute correlation coefficient matrix
        R = np.corrcoef(all_trace)

        # Get unique view IDs
        unique_views = np.unique(all_index[:, 0])

        # Initialize mask
        mask = np.zeros_like(R, dtype=bool)

        # Create mask for same view
        for view in unique_views:
            idx = (all_index[:, 0] == view)
            mask[np.ix_(idx, idx)] = True

        # Remove diagonal
        mask &= ~np.eye(mask.shape[0], dtype=bool)

        # Set correlations from same view to 0
        R[mask] = 0

        # Save
        if R_path:
            os.makedirs(os.path.dirname(R_path), exist_ok=True)
            savemat(R_path, {'R': R}, do_compression=True)
    return R


def spatial_cluster(S_center_list, cutoff_spatial, if_show=False):
    """
    Perform hierarchical spatial clustering to filter out erroneous neuron components.
    
    Groups neurons with similar spatial locations using Euclidean distance and
    single-linkage hierarchical clustering. This helps identify and exclude
    spatially inconsistent neuron detections.
    
    Args:
        S_center_list (np.ndarray): Spatial centroid coordinates for neurons
            Format: 2D array with shape (N, 2)
            - N: Number of neurons
            - First column: y-coordinate (row coordinate)
            - Second column: x-coordinate (column coordinate)
        
        cutoff_spatial (float): Spatial clustering distance threshold
            Format: Positive float, units in pixels
            Meaning: If the Euclidean distance between two neurons is less than
            this threshold, they may belong to the same spatial cluster.
        
        if_show (bool, optional): Whether to visualize clustering results
            Format: Boolean, default False
            If True, generates and saves a scatter plot showing clustering results.
    
    Returns:
        T_spatial (np.ndarray): Spatial cluster labels for each neuron
            Format: 1D integer array with shape (N,)
            Meaning: 1-based labels; neurons with the same label belong to the
            same spatial cluster.
        
        N_spatial_num (int): Number of spatial clusters
            Format: Positive integer
            Meaning: Total number of spatial clusters after clustering.
    
    Note:
        - Uses scipy's hierarchical clustering method (single linkage)
        - Clustering is based on Euclidean distance
        - If if_show=True, generates a color-coded scatter plot saved as
          'spatial_clustering.png'
    """
    # Compute Euclidean distance matrix
    Y_spatial = pdist(S_center_list, metric='euclidean')

    # Hierarchical clustering
    Z_spatial = linkage(Y_spatial, method='single')

    # Generate clusters based on distance threshold
    T_spatial = fcluster(Z_spatial, t=cutoff_spatial, criterion='distance')
    N_spatial_num = np.max(T_spatial)

    # Visualization (optional)
    if if_show:
        import matplotlib.pyplot as plt
        plt.scatter(S_center_list[:, 1], S_center_list[:, 0], c=T_spatial, cmap='tab20')
        plt.title(f'Spatial Clustering (cutoff={cutoff_spatial})')
        plt.axis('equal')
        plt.show()
        plt.savefig('spatial_clustering.png')

    return T_spatial, N_spatial_num


def spatio_temporal_clustering(R, all_index, S_center_list,
                               cutoff_spatial, corr_thre, min_view_num,
                               if_show_spatial_clusters=False):
    """
    Perform spatio-temporal clustering to merge neurons across multiple views.
    
    Combines temporal correlation and spatial location information to identify
    and merge neurons detected in different views that correspond to the same
    physical neuron. The algorithm filters candidates by temporal correlation,
    refines using spatial clustering, and selects the best match per view.
    
    Args:
        R (np.ndarray): Correlation coefficient matrix
            Format: 2D array with shape (n_cells, n_cells)
            Meaning: Pearson correlation coefficient matrix between neurons,
            value range [-1, 1]
        
        all_index (np.ndarray): Index information for all neurons
            Format: 2D array with shape (n_cells, 2) or more columns
            Meaning: First column contains view ID for identifying which view
            each neuron belongs to.
        
        S_center_list (np.ndarray): Spatial centroid coordinates for neurons
            Format: 2D array with shape (n_cells, 2)
            Meaning: Spatial position coordinates [y, x] for each neuron.
        
        cutoff_spatial (float): Spatial clustering distance threshold
            Format: Positive float, units in pixels
            Meaning: Euclidean distance threshold for spatial clustering.
        
        corr_thre (float): Temporal correlation threshold
            Format: Float, typically in range [0, 1]
            Meaning: If temporal correlation between two neurons exceeds this
            threshold, they are considered potential matches for the same neuron.
        
        min_view_num (int): Minimum number of views required
            Format: Positive integer
            Meaning: A neuron group must appear in at least this many different
            views to be considered valid.
        
        if_show_spatial_clusters (bool, optional): Whether to show spatial
            clustering visualization
            Format: Boolean, default False
    
    Returns:
        neuron_group (list): List of neuron groups
            Format: list of list[int], where each inner list contains neuron
            indices from multiple views that belong to the same neuron.
            Meaning: Each element represents a merged neuron, containing the
            corresponding neuron indices from various views.
    
    Note:
        Algorithm workflow:
        1. Binarize correlation matrix (set values > threshold to 1)
        2. For each neuron, find candidate neurons with high correlation
        3. Apply spatial clustering to candidates, excluding spatially
           inconsistent matches
        4. Select neuron with maximum correlation in each view
        5. Ensure final group contains at least min_view_num different views
        - Selected neurons are cleared from correlation matrix to avoid
          duplicate assignments
    """
    # === Binarize correlation matrix ===
    R_bina = (R > corr_thre).astype(int)

    print('Start spatio-temporal clustering...')
    neuron_group = []
    neuron_num = 0

    for i in range(R_bina.shape[0]):
        R_bina_r = R_bina[i, :]
        right_idx = np.where(R_bina_r == 1)[0]

        # === Spatial clustering ===
        if len(right_idx) >= min_view_num:
            right_S_center_list = S_center_list[right_idx, :]
            T_spatial, N_spatial_num = spatial_cluster(
                right_S_center_list, cutoff_spatial, if_show_spatial_clusters
            )

            if N_spatial_num > 1:
                most_T_spatial_index = np.bincount(T_spatial).argmax()  # mode
                most_index = np.where(T_spatial == most_T_spatial_index)[0]
                right_idx = right_idx[most_index]

        # === Select neuron with max correlation in each view ===
        if len(right_idx) >= min_view_num:
            current_corr_vector = R[i, right_idx]  # correlation with neuron i
            views_in_right_p = all_index[right_idx, 0]  # view IDs

            unique_views = np.unique(views_in_right_p)
            selected_neurons = []

            for view in unique_views:
                mask_in_view = (views_in_right_p == view)
                candidates_in_view = right_idx[mask_in_view]
                max_idx = np.argmax(current_corr_vector[mask_in_view])
                selected_neurons.append(candidates_in_view[max_idx])

            right_idx = np.array(selected_neurons, dtype=int)

        # === Check min view count ===
        if len(right_idx) >= min_view_num:
            # Clear selected neurons from correlation matrix to avoid duplicates
            R_bina[right_idx, :] = 0
            R_bina[:, right_idx] = 0

            neuron_num += 1
            neuron_group.append(right_idx)

    print(f'Spatio-temporal clustering done. Found {len(neuron_group)} neurons.')
    return neuron_group


def group_save(neuron_group, S_center_list, all_index, all_trace, corr_thre, SAVE):
    """
    Save neuron grouping results to .mat file and merge signals from the same
    neuron across different views.
    
    Saves clustered neuron groups in MATLAB format, including spatial coordinates,
    index information, and merged temporal traces.
    
    Args:
        neuron_group (list): List of neuron groups
            Format: list of list[int], where each inner list contains neuron
            indices from multiple views belonging to the same neuron.
            Meaning: Each element represents a merged neuron group.
        
        S_center_list (np.ndarray): Spatial centroid coordinates for all neurons
            Format: 2D array with shape (n_cells, 2)
            Meaning: Spatial position [y, x] for each neuron.
        
        all_index (np.ndarray): Index information for all neurons
            Format: 2D array with shape (n_cells, 2) or more columns
            Meaning: First column is view ID, second column is neuron ID within
            that view.
        
        all_trace (np.ndarray): Temporal trace signals for all neurons
            Format: 2D array with shape (n_cells, n_frames)
            Meaning: Each row represents a neuron's temporal trace.
        
        corr_thre (float): Correlation threshold used for clustering
            Format: Float
            Meaning: Correlation threshold parameter used during clustering,
            saved for record-keeping.
        
        SAVE (str): Output folder path
            Format: String path, e.g., './result/STEP_7_VM'
            Meaning: Results will be saved to this folder.
    
    Returns:
        view_merge_C (list): List of spatial coordinates for each merged neuron
            Format: list of np.ndarray, each array with shape (n_views, 2)
            Meaning: Spatial coordinates for each neuron across different views.
        
        view_merge_id (list): List of index information for each merged neuron
            Format: list of np.ndarray, each array with shape (n_views, 2)
            Meaning: Index information for each neuron across different views.
        
        all_single_neuron_trace (np.ndarray): Merged temporal traces for single neurons
            Format: 2D array with shape (n_neurons, n_frames)
            Meaning: Averaged temporal trace signal for each merged neuron.
    
    Note:
        - Signals from the same neuron across different views are merged by averaging
        - Results are saved to 'view_merging.mat' file, containing view_merge_C,
          view_merge_id, all_single_neuron_trace, and corr_thre
    """

    def merge_signals(signals):
        """
        Merge multiple signals by averaging.
        
        Args:
            signals (np.ndarray): Temporal traces for multiple signals
                Format: 2D array with shape (n_signals, n_frames)
        
        Returns:
            np.ndarray: Merged average signal
                Format: 1D array with shape (n_frames,)
        """
        return np.mean(signals, axis=0)

    view_merge_C = []
    view_merge_id = []
    all_single_neuron_trace = []

    for group in neuron_group:
        single_neuron_index = np.array(group)
        view_merge_C.append(S_center_list[single_neuron_index, :])
        view_merge_id.append(all_index[single_neuron_index, :])

        neuron_group_trace = all_trace[single_neuron_index, :]
        all_single_neuron_trace.append(merge_signals(neuron_group_trace))

    all_single_neuron_trace = np.array(all_single_neuron_trace)

    # Save
    matpath = os.path.join(SAVE, 'view_merging.mat')
    print(f'Saving view merging results to {matpath}...')
    savemat(
        matpath,
        {
            'view_merge_C': convert_data_structure(view_merge_C),
            'view_merge_id': convert_data_structure(view_merge_id),
            'all_single_neuron_trace': all_single_neuron_trace,
            'corr_thre': corr_thre
        }
    )
    print('Done.')

    return view_merge_C, view_merge_id, all_single_neuron_trace


def f_estimateZ(view_merge_C, view_merge_id, psffit_matrix,
                min_loc_num, Nnum, upsample_rate, dz, cen_id):
    """
    Estimate 3D spatial positions (lateral + axial) of neurons based on
    multi-view centroid locations.
    
    Uses PSF fitting matrix and spatial position information from multiple views
    to estimate z-axis positions via curve fitting, then corrects lateral
    positions to obtain final 3D coordinates.
    
    Args:
        view_merge_C (list): List of spatial coordinates for each merged neuron
            Format: list of np.ndarray, each array with shape (n_views, 2)
            Meaning: 2D spatial coordinates [y, x] for each neuron across
            different views.
        
        view_merge_id (list): List of index information for each merged neuron
            Format: list of np.ndarray, each array with shape (n_views, 2)
            Meaning: First column contains view ID for looking up PSF fitting
            parameters.
        
        psffit_matrix (np.ndarray): PSF fitting matrix
            Format: 2D array with shape (n_pairs, 6)
            Meaning: Each row is [i, j, ax, bx, ay, by], representing PSF
            fitting parameters between views i and j.
            - ax, bx: Linear fitting parameters for x-direction (dx = ax * z + bx)
            - ay, by: Linear fitting parameters for y-direction (dy = ay * z + by)
        
        min_loc_num (int): Minimum number of localizations required
            Format: Positive integer
            Meaning: A neuron must appear in at least this many different views
            to perform z-axis estimation.
        
        Nnum (int): Maximum number of views
            Format: Positive integer
            Meaning: Used for normalizing z-axis position calculations.
        
        upsample_rate (int): Upsampling rate
            Format: Positive integer
            Meaning: Scaling factor used in z-axis position calculations.
        
        dz (float): Z-axis step size
            Format: Positive float, units in micrometers
            Meaning: Sampling interval along the z-axis.
        
        cen_id (int): Central view ID
            Format: Integer
            Meaning: Reference central view ID used for calculating lateral
            position offsets.
    
    Returns:
        spatial_3D (np.ndarray): 3D spatial coordinates for each neuron
            Format: 2D array with shape (n_neurons, 3)
            Meaning: Each row is [x, y, z] coordinates, units in pixels (x, y)
            and micrometers (z).
            Note: For invalid neurons (insufficient views), coordinates may be 0.
        
        neuron_num (int): Number of successfully reconstructed neurons
            Format: Positive integer
            Meaning: Total number of neurons satisfying min_loc_num requirement.
        
        invalid_flag (np.ndarray): Invalid neuron flags
            Format: Boolean array with shape (n_neurons,)
            Meaning: True indicates the neuron appears in fewer than min_loc_num
            views and cannot undergo z-axis estimation.
    
    Note:
        Algorithm workflow:
        1. For each neuron, collect spatial coordinates from all views
        2. Calculate position differences and PSF fitting parameters between
           different view pairs
        3. Estimate z-axis position using curve fitting
        4. Correct lateral positions based on z-axis position and PSF parameters
        5. Average corrected positions across all views to obtain final lateral
           coordinates
        - Neurons appearing in fewer than min_loc_num views are marked as
          invalid and skipped
    """

    def func(alpha_uv, z):
        """
        Linear PSF fitting function for curve fitting.
        
        Args:
            alpha_uv (float): PSF fitting parameter (slope)
            z (float): Z-axis position
        
        Returns:
            float: alpha_uv * z, representing position difference
        """
        return z * alpha_uv

    neuron_num = 0
    spatial_3D = np.zeros((len(view_merge_C), 3))
    invalid_flag = []

    for i, view_merge_C_uni in enumerate(view_merge_C):
        view_id = view_merge_id[i][:, 0]
        view_id_uni, uind = np.unique(view_id, return_index=True)
        view_merge_C_uni = view_merge_C_uni[uind, :]

        if view_merge_C_uni.shape[0] < min_loc_num:
            invalid_flag.append(1)
            continue  # skip this neuron
        else:
            invalid_flag.append(0)

        neuron_num += 1

        diffpos = np.zeros((view_merge_C_uni.shape[0] * (view_merge_C_uni.shape[0] - 1) // 2, 2))
        diffpar = np.zeros((view_merge_C_uni.shape[0] * (view_merge_C_uni.shape[0] - 1) // 2, 2))
        count = 0

        for j in range(view_merge_C_uni.shape[0] - 1):
            for k in range(j + 1, view_merge_C_uni.shape[0]):
                count += 1
                diffpos[count - 1] = view_merge_C_uni[j, :] - view_merge_C_uni[k, :]
                ind = np.logical_and(psffit_matrix[:, 0] == view_id_uni[j],
                                     psffit_matrix[:, 1] == view_id_uni[k])
                diffpar[count - 1] = np.squeeze(np.array([psffit_matrix[ind, 2],
                                                          psffit_matrix[ind, 4]]))

        z_pos, _ = curve_fit(func, diffpar.ravel(), diffpos.ravel(),
                             method='trf', ftol=1e-6)
        z_pos = z_pos * Nnum / upsample_rate * dz
        shiftpar = np.zeros((view_merge_C_uni.shape[0], 2))

        for j in range(view_merge_C_uni.shape[0]):
            if view_id_uni[j] < cen_id:
                ind = np.logical_and(psffit_matrix[:, 0] == view_id_uni[j],
                                     psffit_matrix[:, 1] == cen_id)
                shiftpar[j, :] = np.squeeze(np.array([-psffit_matrix[ind, 2],
                                                      -psffit_matrix[ind, 4]]))
            elif view_id_uni[j] > cen_id:
                ind = np.logical_and(psffit_matrix[:, 0] == cen_id,
                                     psffit_matrix[:, 1] == view_id_uni[j])
                shiftpar[j, :] = np.squeeze(np.array([psffit_matrix[ind, 2],
                                                      psffit_matrix[ind, 4]]))
            else:
                shiftpar[j, :] = np.array([0, 0])

        lateral_pos_list = view_merge_C_uni + shiftpar * z_pos / Nnum * upsample_rate
        lateral_pos = np.mean(lateral_pos_list, axis=0)

        spatial_3D[neuron_num - 1, :] = np.concatenate([lateral_pos, z_pos])

        if neuron_num % 20 == 0:
            print(f"{neuron_num} neuron done...")

    invalid_flag = np.array(invalid_flag, dtype=bool)

    return spatial_3D, neuron_num, invalid_flag
