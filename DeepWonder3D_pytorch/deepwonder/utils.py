import os
import json
import datetime
import numpy as np
from skimage import io
import yaml
import psutil
from thop import profile
import torch
import re


def fullwidth_to_halfwidth(s: str) -> str:
    """
    Convert full-width characters to half-width characters.
    
    Converts Unicode full-width characters (commonly produced by Chinese input
    methods) to their half-width equivalents for consistent string processing.
    
    Args:
        s (str): Input string that may contain full-width characters
            Format: Any Unicode string
    
    Returns:
        str: Converted string with all full-width characters converted to
            half-width equivalents
            Format: String with the same length as input
    
    Note:
        - Full-width space (0x3000) is converted to half-width space (0x20)
        - Full-width character range (0xFF01-0xFF5E) is converted to
          corresponding half-width characters
        - Primarily used for handling full-width characters produced by
          Chinese input methods
    """
    result = []
    for char in s:
        code = ord(char)
        if code == 0x3000:  # Full-width space
            code = 32
        elif 0xFF01 <= code <= 0xFF5E:  # Full-width characters
            code -= 0xFEE0
        result.append(chr(code))
    return ''.join(result)


def validate_gpu_index(gpu_index: str) -> str:
    """
    Validate and normalize GPU index string.
    
    Performs validation and cleaning of GPU index input, including full-width
    character conversion, format checking, and validity verification against
    available GPUs on the system.
    
    Args:
        gpu_index (str): GPU index string
            Format: Comma-separated non-negative integers, e.g., "0,1,2" or "0, 1, 2"
            May contain full-width characters, e.g., "０，１，２"
    
    Returns:
        str: Validated and normalized GPU index string
            Format: Comma-separated sorted non-negative integers, e.g., "0,1,2"
            Returns empty string "" if input is invalid
    
    Note:
        Validation rules:
        1. Automatically converts full-width characters to half-width
        2. Must be a comma-separated list of non-negative integers
        3. Each index must be less than the total number of GPUs on the system
        4. Automatically removes duplicates and sorts indices
        5. Returns empty string if validation fails
    """
    # Convert full-width to half-width
    gpu_index = fullwidth_to_halfwidth(gpu_index.strip())

    # Format check
    pattern = re.compile(r'^\d+(,\d+)*$')
    if not pattern.match(gpu_index):
        return ''

    try:
        indices = sorted(set(int(i) for i in gpu_index.split(',')))
    except ValueError:
        return ''

    total_gpus = torch.cuda.device_count()

    if any(i < 0 or i >= total_gpus for i in indices):
        return ''

    # Return the cleaned string
    return ','.join(map(str, indices))


def save_times_json(times: dict, output_dir: str):
    """
    Save timing data dictionary to a JSON file with timestamp in filename.
    
    Creates a 'times' subfolder within the specified output directory and saves
    timing data as a JSON file with a timestamp-based filename.
    
    Args:
        times (dict): Dictionary containing timing data
            Format: {key: value}, where key is a string (e.g., 'DENO', 'SR')
            and value is a float (time in seconds)
            Example: {'DENO': 120.5, 'SR': 89.3, 'SEG': 45.2}
        
        output_dir (str): Base output directory path
            Format: String path, e.g., './result' or '/path/to/output'
            Meaning: JSON file will be saved in the 'times' subfolder of this
            directory
    
    Returns:
        str: Full path of the saved JSON file
            Format: String path, e.g., './result/times/times_20231225143022.json'
            Filename format: times_YYYYMMDDHHMMSS.json, where timestamp is the
            file creation time
    
    Note:
        - Creates 'times' subfolder if it doesn't exist
        - Saves with UTF-8 encoding, supporting Chinese characters
        - JSON file uses 4-space indentation for readability
    """
    # Create the 'times' subfolder inside the output directory
    times_folder = os.path.join(output_dir, 'times')
    os.makedirs(times_folder, exist_ok=True)

    # Generate a filename with the current timestamp
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    json_filename = f'times_{current_time}.json'

    # Full path to the JSON file
    json_path = os.path.join(times_folder, json_filename)

    # Save the dictionary as a JSON file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(times, f, indent=4, ensure_ascii=False)

    return json_path


def replace_large_pixels_with_min(image, threshold):
    """
    Replace pixel values exceeding threshold with the image minimum value.
    
    Used to remove outlier high-value pixels (outliers), typically for image
    preprocessing. This helps eliminate artifacts such as camera saturation or
    noise spikes.
    
    Args:
        image (np.ndarray): Input image array
            Format: Numpy array of arbitrary dimensions, typically 2D or 3D image
            Data type: Typically float or uint16
        
        threshold (float): Threshold value
            Format: Numeric type (int or float)
            Meaning: Pixels exceeding this value will be replaced with the
            image minimum value
    
    Returns:
        np.ndarray: Processed image array
            Format: Same shape and dtype as input image
            Meaning: All pixels exceeding threshold have been replaced with
            image minimum value
    
    Note:
        - This function modifies the input array in-place
        - Used to remove outlier high values in images, such as camera
          saturation or noise spikes
    """
    # Find the minimum value in the image
    min_value = np.min(image)

    # Create a mask of pixels that are greater than the threshold
    mask = image > threshold

    # Use the mask to replace the large pixels with the minimum value
    image[mask] = min_value

    return image


def process_image_sequence(image_sequence, threshold):
    """
    Process image sequence to remove outlier high-value pixels from each frame.
    
    Applies outlier removal processing to each frame in the image sequence
    independently.
    
    Args:
        image_sequence (np.ndarray): Image sequence array
            Format: Numpy array with shape (n_frames, height, width) or
            (n_frames, depth, height, width)
            Meaning: First dimension is time frames, subsequent dimensions
            are spatial dimensions
        
        threshold (float): Outlier threshold
            Format: Numeric type (int or float)
            Meaning: Pixels exceeding this value will be replaced with the
            minimum value of that frame
    
    Returns:
        np.ndarray: Processed image sequence
            Format: Same shape and dtype as input image sequence
            Meaning: Outlier high-value pixels in each frame have been replaced
            with the corresponding frame's minimum value
    
    Note:
        - Each frame in the sequence is processed independently using its own
          minimum value
        - This function modifies the input array in-place
    """
    # Get the number of images in the sequence
    num_images = image_sequence.shape[0]

    # Process each image in the sequence
    for i in range(num_images):
        image_sequence[i] = replace_large_pixels_with_min(image_sequence[i], threshold)

    return image_sequence


def get_gpu_mem_info(gpu_id=0):
    """
    Get GPU memory usage information by GPU ID.
    
    Uses NVIDIA Management Library (NVML) to retrieve memory usage information
    for the specified GPU device.
    
    Args:
        gpu_id (int, optional): GPU device ID
            Format: Non-negative integer, default 0
            Meaning: GPU device index to query, starting from 0
    
    Returns:
        tuple: (total, used, free) GPU memory information triple
            Format: (float, float, float), units in MB
            - total: Total GPU memory size (MB)
            - used: Currently used GPU memory (MB)
            - free: Currently available GPU memory (MB)
            Returns (0, 0, 0) if GPU does not exist
    
    Note:
        - Requires pynvml library to be installed
        - If specified GPU ID is invalid, prints error message and returns
          (0, 0, 0)
        - Memory sizes are rounded to 2 decimal places
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} does not correspond to an existing GPU!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free


def get_cpu_mem_info():
    """
    Get memory information for the current machine.
    
    Uses psutil library to retrieve system memory and current process memory
    usage information.
    
    Args:
        No parameters.
    
    Returns:
        tuple: (mem_total, mem_free, mem_process_used) Memory information triple
            Format: (float, float, float), units in MB
            - mem_total: Total system memory size (MB)
            - mem_free: Currently available system memory (MB)
            - mem_process_used: Memory used by current process (MB, RSS -
              Resident Set Size)
    
    Note:
        - Requires psutil library to be installed
        - Memory sizes are rounded to 2 decimal places
        - mem_process_used represents the actual physical memory usage of the
          current Python process
    """
    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    return mem_total, mem_free, mem_process_used


#########################################################################
#########################################################################
def print_dict(dictionary):
    """
    Print dictionary contents in colorized format.
    
    Uses ANSI escape codes to display dictionary keys in red and values in
    default color for enhanced readability.
    
    Args:
        dictionary (dict): Dictionary to print
            Format: Any Python dictionary object
            Meaning: Key-value pairs will be formatted and printed
    
    Returns:
        None: No return value
    
    Note:
        - Uses ANSI escape code \033[91m to set red foreground color,
          \033[0m to reset color
        - In terminals supporting ANSI escape codes, keys will appear in red
        - Primarily used for debugging and parameter display
    """
    for key, value in dictionary.items():
        print(f"\033[91m{key}\033[0m: {value}")


def get_netpara(model):
    """
    Calculate the total number of trainable parameters in a model.
    
    Counts all parameters with requires_grad=True, which are the parameters
    that will be updated during training.
    
    Args:
        model (torch.nn.Module): PyTorch model object
            Format: Neural network model inheriting from torch.nn.Module
            Meaning: Model for which to count parameters
    
    Returns:
        int: Total number of trainable parameters in the model
            Format: Non-negative integer
            Meaning: Total number of elements in all parameters with
            requires_grad=True
    
    Note:
        - Only counts parameters with requires_grad=True (trainable parameters)
        - Uses numel() method to get total number of elements in each parameter
          tensor
        - Used for model complexity analysis and memory estimation
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_netparaflops(model, input):
    """
    Calculate and print model FLOPs (floating point operations) and parameter count.
    
    Uses thop library to analyze model computational complexity and parameter count.
    Results are printed directly to the console.
    
    Args:
        model (torch.nn.Module): PyTorch model object
            Format: Neural network model inheriting from torch.nn.Module
            Meaning: Model for which to analyze computational complexity
        
        input (torch.Tensor): Model input tensor
            Format: torch.Tensor with shape matching model input requirements
            Meaning: Example input for FLOPs calculation, typically a single
            batch input
    
    Returns:
        None: Results are printed directly to console
    
    Note:
        - Requires thop library to be installed (pip install thop)
        - FLOPs printed in G (Giga, 10^9) units
        - Parameter count printed in M (Mega, 10^6) units
        - Used for model performance analysis and optimization
    """
    flops, params = profile(model, inputs=(input,))

    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))


def save_para_dict(save_path, para_dict):
    """
    Save parameter dictionary to YAML or TXT format file.
    
    Automatically selects save format based on file extension: .yaml saves as
    YAML format, .txt saves as text format.
    
    Args:
        save_path (str): File path for saving
            Format: String path, e.g., './config/para.yaml' or './config/para.txt'
            Meaning: If path contains '.yaml', saves as YAML format; if contains
            '.txt', saves as text format
        
        para_dict (dict): Parameter dictionary
            Format: Any Python dictionary object
            Meaning: Parameter configuration to save in key-value pair format
    
    Returns:
        None: No return value
    
    Note:
        - YAML format: Uses yaml.dump() to save, preserving dictionary structure
        - TXT format: One key-value pair per line, format "key : value"
        - If path contains both .yaml and .txt, both formats will be saved
        - Requires PyYAML library for YAML format saving
    """
    if '.yaml' in save_path:
        with open(save_path, "w") as yaml_file:
            yaml.dump(para_dict, yaml_file)

    if '.txt' in save_path:
        para_dict_key = para_dict.keys()
        with open(save_path, 'w') as f:  # 设置文件对象
            for now_key in para_dict_key:
                now_para = para_dict[now_key]
                # print(now_key,' -----> ',now_para)
                now_str = now_key + " : " + str(now_para) + '\n'
                f.write(now_str)


def save_img(all_img, norm_factor, output_path, im_name, tag='', if_nor=1,
             if_filter_ourliers=0, ourliers_thres=60000):
    """
    Save image array as TIFF file with support for normalization and outlier filtering.
    
    Saves numpy array image data as 16-bit TIFF format with optional normalization
    and outlier removal processing.
    
    Args:
        all_img (np.ndarray): Image array
            Format: Numpy array of arbitrary dimensions, typically 2D or 3D image
            Data type: Typically float32 or float64
        
        norm_factor (float): Normalization factor
            Format: Positive float
            Meaning: Image data is multiplied by this factor first, used for unit
            conversion or scaling
        
        output_path (str): Output directory path
            Format: String path, e.g., './result' or '/path/to/output'
            Meaning: Image file will be saved to this directory
        
        im_name (str): Image filename (without extension)
            Format: String, e.g., 'image' or 'image.tif'
            Meaning: If contains '.tif', used directly; otherwise '.tif' extension
            is added
        
        tag (str, optional): Filename tag
            Format: String, default empty string
            Meaning: Added to filename, e.g., 'image_tag.tif'
        
        if_nor (int, optional): Whether to normalize
            Format: 0 or 1, default 1
            Meaning: 1 means normalize to [0, 65535] range and convert to uint16;
            0 means no normalization
        
        if_filter_ourliers (int, optional): Whether to filter outliers
            Format: 0 or 1, default 0
            Meaning: 1 means filter outlier high-value pixels exceeding threshold;
            0 means no filtering
        
        ourliers_thres (int, optional): Outlier threshold
            Format: Positive integer, default 60000
            Meaning: If if_filter_ourliers=1, pixels exceeding this value will
            be replaced with image minimum value
    
    Returns:
        None: Image is saved directly to file
    
    Note:
        - Saves as 16-bit TIFF format (uint16), value range [0, 65535]
        - If if_nor=1, normalizes to [0, 65535] range first
        - If if_filter_ourliers=1, removes outliers first
        - Uses skimage.io.imsave to save image
    """
    all_img = all_img.squeeze().astype(np.float32) * norm_factor
    if '.tif' not in im_name:
        all_img_name = output_path + '/' + im_name + tag + '.tif'
    if '.tif' in im_name:
        all_img_name = output_path + '/' + im_name + tag
    # ssprint('all_img_name : ',all_img_name)
    if if_nor:
        all_img = all_img - np.min(all_img)
        # print(np.max(all_img), np.min(all_img))
        all_img = all_img / np.max(all_img) * 65535
        # print(np.max(all_img), np.min(all_img))
        all_img = np.clip(all_img, 0, 65535).astype('uint16')
    '''
    , if_cut=0
    if if_cut:
        all_img = all_img[0:-opt.img_s+1,:,:]
    '''
    # print(np.max(all_img), np.min(all_img))
    all_img = all_img - np.min(all_img)
    # print('all_img -----> ',all_img.shape)
    if if_filter_ourliers:
        all_img = process_image_sequence(all_img, ourliers_thres)
        # print('if_filter_ourliers')
        # pass

    all_img = np.clip(all_img, 0, 65535).astype('uint16')
    io.imsave(all_img_name, all_img)


def save_img_train(u_in_all, output_path, epoch, index, input_name_list, norm_factor, label_name):
    """
    Save image data during training process.
    
    Saves image batches from training as individual TIFF files with filenames
    containing epoch, index, and other metadata.
    
    Args:
        u_in_all (torch.Tensor): Image batch tensor
            Format: 4D tensor with shape (batch_size, channels, height, width)
            Meaning: Image data for one batch, typically from model output or input
        
        output_path (str): Base output directory path
            Format: String path, e.g., './result/train'
            Meaning: Image files will be saved to label_name subfolder within
            this directory
        
        epoch (int): Current training epoch
            Format: Non-negative integer
            Meaning: Used in filename generation, e.g., '10_image_0_input.tif'
        
        index (int): Current batch index
            Format: Non-negative integer
            Meaning: Used in filename generation to distinguish different batches
        
        input_name_list (list): List of input filenames
            Format: list of str, length equals batch_size
            Meaning: Each element is file path or name of input image, used for
            generating output filenames
        
        norm_factor (float): Normalization factor
            Format: Positive float
            Meaning: Image data is multiplied by this factor first, used for unit
            conversion or scaling
        
        label_name (str): Label name
            Format: String, e.g., 'input', 'output', 'label', etc.
            Meaning: Used to create subfolder and filename, identifying image type
    
    Returns:
        None: Images are saved directly to files
    
    Note:
        - Creates label_name subfolder under output_path if it doesn't exist
        - Each image in the batch is saved as a separate TIFF file
        - Filename format: {epoch}_{index}_{image_index}_{input_name}_{label_name}.tif
        - Images are converted from GPU tensor to numpy array before saving
    """
    u_in_path = output_path + '//' + label_name
    if not os.path.exists(u_in_path):
        os.mkdir(u_in_path)
    # print('-----> ',u_in_all.shape)
    for u_in_i in range(0, u_in_all.shape[0]):
        input_name = os.path.basename(input_name_list[u_in_i])
        u_in = u_in_all[u_in_i, :, :, :]
        u_in = u_in.cpu().detach().numpy()
        u_in = u_in.squeeze()

        u_in = u_in.squeeze().astype(np.float32) * norm_factor
        u_in_name = u_in_path + '//' + str(epoch) + '_' + str(index) + '_' + str(
            u_in_i) + '_' + input_name + '_' + label_name + '.tif'
        # print(label_name,' -----> ', u_in.max(), u_in.min())
        io.imsave(u_in_name, u_in)


def UseStyle(string, mode='', fore='', back=''):
    """
    Add ANSI terminal styling (colors, modes, etc.) to a string.
    
    Uses ANSI escape codes to add terminal display styling to strings, including
    foreground color, background color, and display modes.
    
    Args:
        string (str): String to add styling to
            Format: Any string
            Meaning: Text content that needs formatting
        
        mode (str, optional): Display mode
            Format: String, valid values: 'mormal', 'bold', 'underline', 'blink',
            'invert', 'hide'
            Meaning: Text display mode, such as bold, underline, etc.
            Note: 'mormal' appears to be a typo for 'normal'
        
        fore (str, optional): Foreground color (text color)
            Format: String, valid values: 'black', 'red', 'green', 'yellow',
            'blue', 'purple', 'cyan', 'white'
            Meaning: Foreground color of the text
        
        back (str, optional): Background color
            Format: String, valid values: 'black', 'red', 'green', 'yellow',
            'blue', 'purple', 'cyan', 'white'
            Meaning: Background color of the text
    
    Returns:
        str: String with ANSI escape codes added
            Format: String containing ANSI escape sequences
            Meaning: Will display as styled text in terminals supporting ANSI
            escape codes
    
    Note:
        - Uses ANSI escape code \033[XXm to set style, \033[0m to reset style
        - Only effective in terminals supporting ANSI escape codes (e.g., Linux/Mac
          terminals, some Windows 10+ terminals)
        - Primarily used for beautifying console output
        - Note: 'mormal' appears to be a typo for 'normal' in the code
    """
    STYLE = {
        'fore':
            {
                'black': 30,
                'red': 31,
                'green': 32,
                'yellow': 33,
                'blue': 34,
                'purple': 35,
                'cyan': 36,
                'white': 37,
            },

        'back':
            {
                'black': 40,
                'red': 41,
                'green': 42,
                'yellow': 43,
                'blue': 44,
                'purple': 45,
                'cyan': 46,
                'white': 47,
            },

        'mode':
            {
                'mormal': 0,
                'bold': 1,
                'underline': 4,
                'blink': 5,
                'invert': 7,
                'hide': 8,
            },

        'default':
            {
                'end': 0,
            },
    }

    mode = '%s' % STYLE['mode'][mode] if mode in STYLE['mode'] else ''  # .has_key(mode) else ''
    fore = '%s' % STYLE['fore'][fore] if fore in STYLE['fore'] else ''  # .has_key(fore) else ''
    back = '%s' % STYLE['back'][back] if back in STYLE['back'] else ''  # .has_key(back) else ''
    style = ';'.join([s for s in [mode, fore, back] if s])
    style = '\033[%sm' % style if style else ''
    end = '\033[%sm' % STYLE['default']['end'] if style else ''
    return '%s%s%s' % (style, string, end)
