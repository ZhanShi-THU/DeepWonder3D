import numpy as np


def adjust_time_resolution(input_path,
                           input_folder,
                           output_path,
                           output_folder,
                           t_resolution):
    """
    Adjust temporal resolution of image sequence.
    
    Resamples image sequence temporal resolution from original to target
    resolution using linear interpolation.
    
    Args:
        input_path (str): Base path for input data
            Format: String path, e.g., './data' or '/path/to/input'
            Meaning: Parent directory containing input folder
        
        input_folder (str): Input folder name
            Format: String, e.g., 'raw_data'
            Meaning: Folder name containing original TIFF image files
        
        output_path (str): Base path for output data
            Format: String path, e.g., './result' or '/path/to/output'
            Meaning: Parent directory for output folder
        
        output_folder (str): Output folder name
            Format: String, e.g., 'adjusted_data'
            Meaning: Folder name for saving adjusted images
        
        t_resolution (int): Target temporal resolution (milliseconds)
            Format: Positive integer, e.g., 10, 20, 50, etc.
            Meaning: Target temporal resolution, assuming original resolution
            is 10ms
            Note: If t_resolution=10, maintains original resolution; if
            t_resolution=20, temporal resolution is halved
    
    Returns:
        None: Processed images are saved directly to output folder
    
    Note:
        - Uses linear interpolation (numpy.interp) for temporal dimension
          resampling
        - Assumes original temporal resolution is 10ms
        - New frame count = original frame count / (t_resolution / 10)
        - Interpolates each pixel's time series independently
        - Creates output folder if it doesn't exist
        - Requires tifffile library for reading/writing TIFF files
    """
    data_path = input_path + '//' + input_folder
    import os
    for im_name in list(os.walk(data_path, topdown=False))[-1][-1]:
        if '.tif' in im_name:
            import tifffile as tiff
            im_dir = data_path + '//' + im_name
            im = tiff.imread(im_dir)

            im_w = im.shape[2]
            im_h = im.shape[1]
            im_s = im.shape[0]

            new_im_s = int(im_s / (t_resolution / 10))
            new_im = np.zeros((new_im_s, im_h, im_w))
            # print(new_im.shape)

            im_s_list = np.linspace(0, im_s - 1, im_s)
            new_im_s_list = np.linspace(0, im_s - 1, new_im_s)
            for i in range(0, im_h):
                for ii in range(0, im_w):
                    new_im[:, i, ii] = np.interp(new_im_s_list, im_s_list, im[:, i, ii])

            # print(new_im.shape)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            if not os.path.exists(output_path + '//' + output_folder):
                os.mkdir(output_path + '//' + output_folder)
            from skimage import io
            # io.imsave(output_path+'//'+output_folder+'//'+im_name, new_im)
            tiff.imwrite(output_path + '//' + output_folder + '//' + im_name, new_im)


def get_data_fingerprint(data_path):
    """
    Get dimension information for all image files in data folder.
    
    Scans all TIFF files in the specified folder and collects their width,
    height, and depth (temporal frame count) information.
    
    Args:
        data_path (str): Data folder path
            Format: String path, e.g., './data/raw' or '/path/to/data'
            Meaning: Folder path containing TIFF image files
    
    Returns:
        tuple: (im_w_list, im_h_list, im_s_list, min_im_w, min_im_h, min_im_s)
            Image dimension information
            Format: (list, list, list, int, int, int)
            Meaning:
            - im_w_list: List of widths for all images
            - im_h_list: List of heights for all images
            - im_s_list: List of temporal frame counts for all images
            - min_im_w: Minimum width
            - min_im_h: Minimum height
            - min_im_s: Minimum temporal frame count
    
    Note:
        - Only scans TIFF format files (.tif or .tiff extension)
        - Assumes image format is 3D array with shape (time, height, width)
        - Used during data preprocessing to determine uniform crop size
        - Requires tifffile library for reading TIFF files
    """
    im_w_list = []
    im_h_list = []
    im_s_list = []
    import os
    for im_name in list(os.walk(data_path, topdown=False))[-1][-1]:
        if '.tif' in im_name:
            import tifffile as tiff
            im_dir = data_path + '//' + im_name
            im = tiff.imread(im_dir)

            im_w = im.shape[2]
            im_h = im.shape[1]
            im_s = im.shape[0]

            im_w_list.append(im_w)
            im_h_list.append(im_h)
            im_s_list.append(im_s)

    min_im_w = min(im_w_list)
    min_im_h = min(im_h_list)
    min_im_s = min(im_s_list)
    return im_w_list, im_h_list, im_s_list, min_im_w, min_im_h, min_im_s
