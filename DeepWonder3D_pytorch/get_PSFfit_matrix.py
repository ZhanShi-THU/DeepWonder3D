import os
import tifffile
import numpy as np
from natsort import natsorted
from scipy.io import savemat


def psf_weighted_centroids_array(tiff_path):
    """
    Read PSF TIFF file and compute intensity-weighted centroid for each frame.
    
    Calculates intensity-weighted centroids for each frame in the PSF image
    sequence, used for subsequent PSF fitting and 3D localization.
    
    Args:
        tiff_path (str): Path to PSF TIFF file
            Format: String path, e.g., './psf_data/view_113.tif'
            Meaning: TIFF file containing PSF image sequence, typically a 3D
            stack (z-stack)
    
    Returns:
        centroids_array (np.ndarray): Array of centroid coordinates for each frame
            Format: 2D float array with shape (Nz, 2)
            Meaning: Each row is (y, x) coordinates in pixels
            - Nz: Number of z-axis slices (frames)
            - If total intensity of a frame is 0, corresponding position is
              [np.nan, np.nan]
    
    Note:
        - Uses intensity-weighted method to calculate centroid:
          centroid = sum(position * intensity) / sum(intensity)
        - If total intensity of a frame is 0 (completely black), centroid is
          set to NaN
        - Centroid coordinates use floats for sub-pixel precision
        - Requires tifffile library for reading TIFF files
    """
    psf_stack = tifffile.imread(tiff_path)
    centroids = []

    for frame in psf_stack:
        total_intensity = np.sum(frame)
        if total_intensity == 0:
            centroids.append([np.nan, np.nan])
            continue

        y_indices, x_indices = np.indices(frame.shape)
        y_center = np.sum(y_indices * frame) / total_intensity
        x_center = np.sum(x_indices * frame) / total_intensity
        centroids.append([y_center, x_center])

    # convert to numpy array
    centroids_array = np.array(centroids, dtype=np.float64)
    return centroids_array


def compute_psf_fit(psf, all_z):
    """
    Compute linear fitting parameters for centroid differences between views.
    
    Performs linear fitting on centroid position differences between views at
    different depths to obtain PSF fitting matrix, used for subsequent 3D
    localization and z-axis estimation.
    
    Args:
        psf (np.ndarray): Centroid coordinates for each view at different depths
            Format: 3D array with shape (N_views, N_z, 2)
            Meaning:
            - N_views: Number of views
            - N_z: Number of z-axis depth slices
            - Third dimension: [x, y] coordinates (Note: different order from
              [y, x] returned by psf_weighted_centroids_array)
        
        all_z (np.ndarray): Z-axis depth values array
            Format: 1D array with shape (N_z,)
            Meaning: Depth value corresponding to each z-axis slice, units in
            micrometers
            Example: (np.arange(101) - 50) * 3 represents -150 to 150 um with
            3 um step size
    
    Returns:
        psffit_matrix (np.ndarray): PSF fitting matrix
            Format: 2D array with shape (N_views*(N_views-1), 6)
            Meaning: Each row is [i, j, ax, bx, ay, by], representing fitting
            parameters between views i and j
            - i, j: View indices (integers)
            - ax, bx: Linear fitting parameters for x-direction, satisfying
              dx = ax * z + bx
            - ay, by: Linear fitting parameters for y-direction, satisfying
              dy = ay * z + by
            Note: Only contains view pairs with i != j, total of
            N_views*(N_views-1) rows
    
    Note:
        - Uses numpy.polyfit for first-order linear fitting
        - For each view pair (i, j), calculates position differences at
          different depths, then fits as linear function of z
        - Fitting results are used in subsequent 3D localization algorithms to
          estimate neuron z-axis positions
    """
    N_views = psf.shape[0]
    psffit_matrix = []

    for i in range(N_views):
        for j in range(N_views):
            if i == j:
                continue

            # vectorized computation of dx, dy
            dx = psf[i, :, 0] - psf[j, :, 0]
            dy = psf[i, :, 1] - psf[j, :, 1]

            # linear fitting
            ax, bx = np.polyfit(all_z, dx, 1)
            ay, by = np.polyfit(all_z, dy, 1)

            psffit_matrix.append([i, j, ax, bx, ay, by])

    return np.array(psffit_matrix)


# PARAMETERS
PSF_folder = 'your_PSF_folder_path'
all_z = (np.arange(101) - 50) * 3  ##### e.g. 101 z-slices, from -150 um to 150 um, z-step = 3 um

# Example usage
centroids_array = []

for tiff_file in natsorted(os.listdir(PSF_folder)):
    if tiff_file.endswith('.tif') or tiff_file.endswith('.tiff'):
        print('Processing file: ', tiff_file)
        centroids = psf_weighted_centroids_array(os.path.join(PSF_folder, tiff_file))
        centroids_array.append(centroids)

centroids_array = np.array(centroids_array)
psffit_matrix = compute_psf_fit(centroids_array, all_z)
savemat('psffit_matrix.mat', {'psffit_matrix': psffit_matrix})
