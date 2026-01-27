import numpy as np
import os
import tifffile as tiff
import random
import math
import torch
from torch.utils.data import Dataset

"""
Pre-processing utilities for RMBG training and inference.

This module provides dataset classes and helper functions to:
    - Extract 3D patches from large TIFF stacks.
    - Map patch indices back to their locations in the original volume.
    - Shuffle datasets in a consistent way across inputs, targets, and names.
"""


class trainset(Dataset):
    """
    Training dataset for paired noisy input and target volumes.

    Args:
        name_list (list[str]): List of patch identifiers. Each identifier must
            exist as a key in ``coordinate_list``.
        coordinate_list (dict): Mapping from patch name to a coordinate dict
            with keys ``'init_h'``, ``'end_h'``, ``'init_w'``, ``'end_w'``,
            ``'init_s'``, ``'end_s'`` describing the 3D crop.
        noise_img (np.ndarray): Source volume for network inputs with shape
            ``(depth, height, width)``.
        noise_img2 (np.ndarray): Source volume for targets with the same shape
            as ``noise_img``.

    __getitem__ output:
        - input (torch.Tensor): Shape ``(1, D, H, W)`` float tensor slice.
        - target (torch.Tensor): Shape ``(1, D, H, W)`` float tensor slice.
    """

    def __init__(self, name_list, coordinate_list, noise_img, noise_img2):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img = noise_img
        self.noise_img2 = noise_img2

    def __getitem__(self, index):
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate["init_h"]
        end_h = single_coordinate["end_h"]
        init_w = single_coordinate["init_w"]
        end_w = single_coordinate["end_w"]
        init_s = single_coordinate["init_s"]
        end_s = single_coordinate["end_s"]
        input_patch = self.noise_img[init_s:end_s:1, init_h:end_h, init_w:end_w]
        target_patch = self.noise_img2[init_s:end_s:1, init_h:end_h, init_w:end_w]

        input_tensor = torch.from_numpy(np.expand_dims(input_patch, 0))
        target_tensor = torch.from_numpy(np.expand_dims(target_patch, 0))
        return input_tensor, target_tensor

    def __len__(self):
        return len(self.name_list)


class trainsetMul(Dataset):
    """
    Training dataset for multiple input/ground-truth volumes.

    Each element uses a coordinate description to crop corresponding patches
    from a dict of ground-truth and input volumes.

    Args:
        coor_list (list[dict]): List of coordinate dictionaries, each with:
            - ``'name'``: key into ``GT_list`` and ``input_list``.
            - ``'init_h'``, ``'init_w'``, ``'init_s'``: start indices.
        GT_list (dict[str, np.ndarray]): Mapping from image name to ground-truth
            volume with shape ``(depth, height, width)``.
        input_list (dict[str, np.ndarray]): Mapping from image name to input
            volume with the same shape as GT.
        opt: Options object with attributes ``img_s``, ``img_w``, ``img_h``.

    __getitem__ output:
        - input (torch.Tensor): Cropped input patch, shape ``(1, D, H, W)``.
        - target (torch.Tensor): Cropped GT patch, shape ``(1, D, H, W)``.
    """

    def __init__(self, coor_list, GT_list, input_list, opt):
        self.coor_list = coor_list
        self.GT_list = GT_list
        self.input_list = input_list
        self.opt = opt

    def __getitem__(self, index):
        per_coor = self.coor_list[index]
        train_im_name = per_coor["name"]
        init_w = per_coor["init_w"]
        init_h = per_coor["init_h"]
        init_s = per_coor["init_s"]
        GT_im = self.GT_list[train_im_name]
        input_im = self.input_list[train_im_name]

        GT_patch = GT_im[
            init_s : init_s + self.opt.img_s,
            init_w : init_w + self.opt.img_w,
            init_h : init_h + self.opt.img_h,
        ]
        input_patch = input_im[
            init_s : init_s + self.opt.img_s,
            init_w : init_w + self.opt.img_w,
            init_h : init_h + self.opt.img_h,
        ]

        input_tensor = torch.from_numpy(np.expand_dims(input_patch, 0))
        target_tensor = torch.from_numpy(np.expand_dims(GT_patch, 0))
        return input_tensor, target_tensor

    def __len__(self):
        return len(self.coor_list)


class testset(Dataset):
    """
    Test dataset that yields patches and their coordinate metadata.

    Args:
        name_list (list[str]): List of patch identifiers.
        coordinate_list (dict): Mapping from patch name to coordinate dict
            (same structure as in ``trainset``).
        noise_img (np.ndarray): Full input volume to sample from.

    __getitem__ output:
        - noise_patch (torch.Tensor): Shape ``(1, D, H, W)``.
        - single_coordinate (dict): Coordinate dictionary for this patch.
    """

    def __init__(self, name_list, coordinate_list, noise_img):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img = noise_img

    def __getitem__(self, index):
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate["init_h"]
        end_h = single_coordinate["end_h"]
        init_w = single_coordinate["init_w"]
        end_w = single_coordinate["end_w"]
        init_s = single_coordinate["init_s"]
        end_s = single_coordinate["end_s"]
        noise_patch = self.noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
        noise_patch = torch.from_numpy(np.expand_dims(noise_patch, 0))
        return noise_patch, single_coordinate

    def __len__(self):
        return len(self.name_list)


def singlebatch_test_save(single_coordinate, output_image, raw_image):
    """
    Paste a single prediction patch back into the full-volume canvas.

    Args:
        single_coordinate (dict): Coordinate dict with keys
            ``'stack_start_*'``, ``'stack_end_*'``, ``'patch_start_*'``,
            ``'patch_end_*'`` describing both stack and patch ranges in
            width (``w``), height (``h``), and slice (``s``) dimensions.
        output_image (np.ndarray): Network prediction volume for this patch.
        raw_image (np.ndarray): Realigned input volume for this patch.

    Returns:
        tuple:
            - patch_out (np.ndarray): Cropped prediction patch.
            - patch_in (np.ndarray): Cropped input patch.
            - stack_start_w, stack_end_w, stack_start_h, stack_end_h,
              stack_start_s, stack_end_s (ints): Placement indices in the
              original full-volume canvas.
    """
    stack_start_w = int(single_coordinate["stack_start_w"])
    stack_end_w = int(single_coordinate["stack_end_w"])
    patch_start_w = int(single_coordinate["patch_start_w"])
    patch_end_w = int(single_coordinate["patch_end_w"])

    stack_start_h = int(single_coordinate["stack_start_h"])
    stack_end_h = int(single_coordinate["stack_end_h"])
    patch_start_h = int(single_coordinate["patch_start_h"])
    patch_end_h = int(single_coordinate["patch_end_h"])

    stack_start_s = int(single_coordinate["stack_start_s"])
    stack_end_s = int(single_coordinate["stack_end_s"])
    patch_start_s = int(single_coordinate["patch_start_s"])
    patch_end_s = int(single_coordinate["patch_end_s"])

    patch_out = output_image[
        patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w
    ]
    patch_in = raw_image[
        patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w
    ]
    return (
        patch_out,
        patch_in,
        stack_start_w,
        stack_end_w,
        stack_start_h,
        stack_end_h,
        stack_start_s,
        stack_end_s,
    )


def multibatch_test_save(single_coordinate, id, output_image, raw_image):
    """
    Paste one element of a batched prediction back into the full-volume canvas.

    Args:
        single_coordinate (dict[str, torch.Tensor]): Coordinate tensors where
            each value is a vector over the batch dimension.
        id (int): Index in the batch to place.
        output_image (np.ndarray): Batched prediction array of shape
            ``(B, D, H, W)`` or ``(B, ...)``.
        raw_image (np.ndarray): Batched input array with the same shape.

    Returns:
        tuple:
            - patch_out (np.ndarray): Cropped prediction patch for sample ``id``.
            - patch_in (np.ndarray): Cropped input patch for sample ``id``.
            - stack_start_w, stack_end_w, stack_start_h, stack_end_h,
              stack_start_s, stack_end_s (ints): Placement indices in the
              original canvas.
    """
    stack_start_w_id = single_coordinate["stack_start_w"].numpy()
    stack_start_w = int(stack_start_w_id[id])
    stack_end_w_id = single_coordinate["stack_end_w"].numpy()
    stack_end_w = int(stack_end_w_id[id])
    patch_start_w_id = single_coordinate["patch_start_w"].numpy()
    patch_start_w = int(patch_start_w_id[id])
    patch_end_w_id = single_coordinate["patch_end_w"].numpy()
    patch_end_w = int(patch_end_w_id[id])

    stack_start_h_id = single_coordinate["stack_start_h"].numpy()
    stack_start_h = int(stack_start_h_id[id])
    stack_end_h_id = single_coordinate["stack_end_h"].numpy()
    stack_end_h = int(stack_end_h_id[id])
    patch_start_h_id = single_coordinate["patch_start_h"].numpy()
    patch_start_h = int(patch_start_h_id[id])
    patch_end_h_id = single_coordinate["patch_end_h"].numpy()
    patch_end_h = int(patch_end_h_id[id])

    stack_start_s_id = single_coordinate["stack_start_s"].numpy()
    stack_start_s = int(stack_start_s_id[id])
    stack_end_s_id = single_coordinate["stack_end_s"].numpy()
    stack_end_s = int(stack_end_s_id[id])
    patch_start_s_id = single_coordinate["patch_start_s"].numpy()
    patch_start_s = int(patch_start_s_id[id])
    patch_end_s_id = single_coordinate["patch_end_s"].numpy()
    patch_end_s = int(patch_end_s_id[id])

    output_image_id = output_image[id]
    raw_image_id = raw_image[id]
    patch_out = output_image_id[
        patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w
    ]
    patch_in = raw_image_id[
        patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w
    ]

    return (
        patch_out,
        patch_in,
        stack_start_w,
        stack_end_w,
        stack_start_h,
        stack_end_h,
        stack_start_s,
        stack_end_s,
    )


def shuffle_datasets(train_raw, train_GT, name_list):
    """
    Shuffle raw and ground-truth stacks using the same random permutation.

    Args:
        train_raw (np.ndarray or sequence): Input volumes with shape
            ``(N, D, H, W)``.
        train_GT (np.ndarray or sequence): Target volumes with matching shape.
        name_list (list[str]): List of sample identifiers of length ``N``.

    Returns:
        tuple:
            - new_train_raw (np.ndarray): Shuffled inputs.
            - new_train_GT (np.ndarray): Shuffled targets.
            - new_name_list (list[str]): Shuffled names.
    """
    index_list = list(range(0, len(name_list)))
    random.shuffle(index_list)
    random_index_list = index_list
    new_name_list = list(range(0, len(name_list)))
    train_raw = np.array(train_raw)
    train_GT = np.array(train_GT)
    new_train_raw = train_raw
    new_train_GT = train_GT
    for i in range(0, len(random_index_list)):
        new_train_raw[i, :, :, :] = train_raw[random_index_list[i], :, :, :]
        new_train_GT[i, :, :, :] = train_GT[random_index_list[i], :, :, :]
        new_name_list[i] = name_list[random_index_list[i]]
    return new_train_raw, new_train_GT, new_name_list


def get_gap_s(args, img, stack_num):
    """
    Compute an appropriate z-gap (slice step) for splitting a stack into patches.

    The z-gap is chosen so that the desired total number of training patches
    (``args.train_datasets_size``) is approximately met given the in-plane
    configuration.

    Args:
        args: Configuration object with attributes:
            - ``img_w``, ``img_h``, ``img_s``: patch dimensions.
            - ``gap_w``, ``gap_h``: in-plane patch strides.
            - ``train_datasets_size`` (int): desired number of training patches.
        img (np.ndarray): Input volume with shape ``(depth, height, width)``.
        stack_num (int): Number of stacks that will be sampled from.

    Returns:
        int: Gap along the slice dimension (``gap_s``).
    """
    whole_w = img.shape[2]
    whole_h = img.shape[1]
    whole_s = img.shape[0]
    w_num = math.floor((whole_w - args.img_w) / args.gap_w) + 1
    h_num = math.floor((whole_h - args.img_h) / args.gap_h) + 1
    s_num = math.ceil(args.train_datasets_size / w_num / h_num / stack_num)
    gap_s = math.floor((whole_s - args.img_s * 2) / (s_num - 1))
    return gap_s


def test_preprocess_lessMemoryNoTail_chooseOne(args, N):
    """
    Pre-process a single TIFF stack into overlapping 3D patches for testing.

    Args:
        args: Namespace-like object with attributes:
            - ``img_h``, ``img_w``, ``img_s``: patch size.
            - ``gap_h``, ``gap_w``, ``gap_s``: strides between patches.
            - ``datasets_path`` (str): Root path of datasets.
            - ``datasets_folder`` (str): Subfolder containing input TIFF files.
            - ``test_datasize`` (int): Maximum number of slices to keep.
            - ``normalize_factor`` (float): Scalar applied to normalize input.
        N (int): Index of the file to use within ``datasets_folder`` after
            sorting file names.

    Returns:
        tuple:
            - name_list (list[str]): List of generated patch identifiers.
            - noise_im (np.ndarray): Normalized, mean-subtracted input stack.
            - coordinate_list (dict): Mapping from patch name to coordinate
              dictionary with both crop and placement indices.
    """
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s2 = args.gap_s
    cut_w = (img_w - gap_w) / 2
    cut_h = (img_h - gap_h) / 2
    cut_s = (img_s2 - gap_s2) / 2
    im_folder = args.datasets_path + "//" + args.datasets_folder

    name_list = []
    coordinate_list = {}
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    img_list.sort()

    im_name = img_list[N]

    im_dir = im_folder + "//" + im_name
    print("im_dir -----> ", im_dir)
    noise_im = tiff.imread(im_dir)
    if noise_im.shape[0] > args.test_datasize:
        noise_im = noise_im[0 : args.test_datasize, :, :]
    noise_im = noise_im.astype(np.float32) / args.normalize_factor

    noise_im_ave_single = np.mean(noise_im, axis=0)
    noise_im_ave = np.zeros(noise_im.shape)
    for i in range(0, noise_im.shape[0]):
        noise_im_ave[i, :, :] = noise_im_ave_single
    noise_im = noise_im - noise_im_ave

    whole_w = noise_im.shape[2]
    whole_h = noise_im.shape[1]
    whole_s = noise_im.shape[0]

    num_w = math.ceil((whole_w - img_w + gap_w) / gap_w)
    num_h = math.ceil((whole_h - img_h + gap_h) / gap_h)
    num_s = math.ceil((whole_s - img_s2 + gap_s2) / gap_s2)
    for x in range(0, num_h):
        for y in range(0, num_w):
            for z in range(0, num_s):
                single_coordinate = {
                    "init_h": 0,
                    "end_h": 0,
                    "init_w": 0,
                    "end_w": 0,
                    "init_s": 0,
                    "end_s": 0,
                }
                if x != (num_h - 1):
                    init_h = gap_h * x
                    end_h = gap_h * x + img_h
                else:
                    init_h = whole_h - img_h
                    end_h = whole_h

                if y != (num_w - 1):
                    init_w = gap_w * y
                    end_w = gap_w * y + img_w
                else:
                    init_w = whole_w - img_w
                    end_w = whole_w

                if z != (num_s - 1):
                    init_s = gap_s2 * z
                    end_s = gap_s2 * z + img_s2
                else:
                    init_s = whole_s - img_s2
                    end_s = whole_s
                single_coordinate["init_h"] = init_h
                single_coordinate["end_h"] = end_h
                single_coordinate["init_w"] = init_w
                single_coordinate["end_w"] = end_w
                single_coordinate["init_s"] = init_s
                single_coordinate["end_s"] = end_s

                if (num_w - 1) == 0:
                    single_coordinate["stack_start_w"] = 0
                    single_coordinate["stack_end_w"] = whole_w
                    single_coordinate["patch_start_w"] = 0
                    single_coordinate["patch_end_w"] = img_w
                elif (num_w - 1) > 0:
                    if y == 0:
                        single_coordinate["stack_start_w"] = y * gap_w
                        single_coordinate["stack_end_w"] = y * gap_w + img_w - cut_w
                        single_coordinate["patch_start_w"] = 0
                        single_coordinate["patch_end_w"] = img_w - cut_w
                    elif y == num_w - 1:
                        single_coordinate["stack_start_w"] = whole_w - img_w + cut_w
                        single_coordinate["stack_end_w"] = whole_w
                        single_coordinate["patch_start_w"] = cut_w
                        single_coordinate["patch_end_w"] = img_w
                    else:
                        single_coordinate["stack_start_w"] = y * gap_w + cut_w
                        single_coordinate["stack_end_w"] = y * gap_w + img_w - cut_w
                        single_coordinate["patch_start_w"] = cut_w
                        single_coordinate["patch_end_w"] = img_w - cut_w

                if (num_h - 1) == 0:
                    single_coordinate["stack_start_h"] = 0
                    single_coordinate["stack_end_h"] = whole_h
                    single_coordinate["patch_start_h"] = 0
                    single_coordinate["patch_end_h"] = img_h
                elif (num_h - 1) > 0:
                    if x == 0:
                        single_coordinate["stack_start_h"] = x * gap_h
                        single_coordinate["stack_end_h"] = x * gap_h + img_h - cut_h
                        single_coordinate["patch_start_h"] = 0
                        single_coordinate["patch_end_h"] = img_h - cut_h
                    elif x == num_h - 1:
                        single_coordinate["stack_start_h"] = whole_h - img_h + cut_h
                        single_coordinate["stack_end_h"] = whole_h
                        single_coordinate["patch_start_h"] = cut_h
                        single_coordinate["patch_end_h"] = img_h
                    else:
                        single_coordinate["stack_start_h"] = x * gap_h + cut_h
                        single_coordinate["stack_end_h"] = x * gap_h + img_h - cut_h
                        single_coordinate["patch_start_h"] = cut_h
                        single_coordinate["patch_end_h"] = img_h - cut_h

                if (num_s - 1) == 0:
                    single_coordinate["stack_start_s"] = 0
                    single_coordinate["stack_end_s"] = whole_s
                    single_coordinate["patch_start_s"] = 0
                    single_coordinate["patch_end_s"] = img_s2
                elif (num_s - 1) > 0:
                    if z == 0:
                        single_coordinate["stack_start_s"] = z * gap_s2
                        single_coordinate["stack_end_s"] = (
                            z * gap_s2 + img_s2 - cut_s
                        )
                        single_coordinate["patch_start_s"] = 0
                        single_coordinate["patch_end_s"] = img_s2 - cut_s
                    elif z == num_s - 1:
                        single_coordinate["stack_start_s"] = (
                            whole_s - img_s2 + cut_s
                        )
                        single_coordinate["stack_end_s"] = whole_s
                        single_coordinate["patch_start_s"] = cut_s
                        single_coordinate["patch_end_s"] = img_s2
                    else:
                        single_coordinate["stack_start_s"] = z * gap_s2 + cut_s
                        single_coordinate["stack_end_s"] = (
                            z * gap_s2 + img_s2 - cut_s
                        )
                        single_coordinate["patch_start_s"] = cut_s
                        single_coordinate["patch_end_s"] = img_s2 - cut_s

                print(
                    "RMBG stack_start_w -----> ",
                    single_coordinate["stack_start_w"],
                    " patch_start_w -----> ",
                    single_coordinate["patch_start_w"],
                    "stack_end_w -----> ",
                    single_coordinate["stack_end_w"],
                    " patch_end_w -----> ",
                    single_coordinate["patch_end_w"],
                )
                print(
                    "RMBG stack_start_h -----> ",
                    single_coordinate["stack_start_h"],
                    " patch_start_h -----> ",
                    single_coordinate["patch_start_h"],
                    "stack_end_h -----> ",
                    single_coordinate["stack_end_h"],
                    " patch_end_h -----> ",
                    single_coordinate["patch_end_h"],
                )
                patch_name = (
                    args.datasets_folder + "_x" + str(x) + "_y" + str(y) + "_z" + str(z)
                )
                name_list.append(patch_name)
                coordinate_list[patch_name] = single_coordinate

    return name_list, noise_im, coordinate_list


def test_preprocess_lessMemoryNoTail_SubImg(args, sub_img):
    """
    Pre-process a given 3D sub-volume into overlapping patches for testing.

    Args:
        args (dict): Configuration dictionary with keys:
            - ``'img_h'``, ``'img_w'``, ``'img_s'``: patch size.
            - ``'gap_h'``, ``'gap_w'``, ``'gap_s'``: patch strides.
            - ``'normalize_factor'`` (float): Normalization scalar.
        sub_img (np.ndarray): Input volume with shape ``(depth, height, width)``.

    Returns:
        tuple:
            - name_list (list[str]): Generated patch identifiers.
            - noise_im (np.ndarray): Normalized, mean-subtracted version of
              ``sub_img``.
            - coordinate_list (dict): Mapping from patch name to coordinate dict
              with crop and placement indices.
    """
    img_h = args["img_h"]
    img_w = args["img_w"]
    img_s2 = args["img_s"]
    gap_h = args["gap_h"]
    gap_w = args["gap_w"]
    gap_s2 = args["gap_s"]
    cut_w = (img_w - gap_w) / 2
    cut_h = (img_h - gap_h) / 2
    cut_s = (img_s2 - gap_s2) / 2

    noise_im = sub_img
    noise_im = noise_im.astype(np.float32) / args["normalize_factor"]

    noise_im_ave_single = np.mean(noise_im, axis=0)
    noise_im_ave = np.zeros(noise_im.shape)
    for i in range(0, noise_im.shape[0]):
        noise_im_ave[i, :, :] = noise_im_ave_single
    noise_im = noise_im - noise_im_ave
    noise_im = noise_im / 1

    whole_w = noise_im.shape[2]
    whole_h = noise_im.shape[1]
    whole_s = noise_im.shape[0]

    num_w = math.ceil((whole_w - img_w + gap_w) / gap_w)
    num_h = math.ceil((whole_h - img_h + gap_h) / gap_h)
    num_s = math.ceil((whole_s - img_s2 + gap_s2) / gap_s2)
    name_list = []
    coordinate_list = {}
    for x in range(0, num_h):
        for y in range(0, num_w):
            for z in range(0, num_s):
                single_coordinate = {
                    "init_h": 0,
                    "end_h": 0,
                    "init_w": 0,
                    "end_w": 0,
                    "init_s": 0,
                    "end_s": 0,
                }
                if x != (num_h - 1):
                    init_h = gap_h * x
                    end_h = gap_h * x + img_h
                else:
                    init_h = whole_h - img_h
                    end_h = whole_h

                if y != (num_w - 1):
                    init_w = gap_w * y
                    end_w = gap_w * y + img_w
                else:
                    init_w = whole_w - img_w
                    end_w = whole_w

                if z != (num_s - 1):
                    init_s = gap_s2 * z
                    end_s = gap_s2 * z + img_s2
                else:
                    init_s = whole_s - img_s2
                    end_s = whole_s
                single_coordinate["init_h"] = init_h
                single_coordinate["end_h"] = end_h
                single_coordinate["init_w"] = init_w
                single_coordinate["end_w"] = end_w
                single_coordinate["init_s"] = init_s
                single_coordinate["end_s"] = end_s

                if (num_w - 1) == 0:
                    single_coordinate["stack_start_w"] = 0
                    single_coordinate["stack_end_w"] = whole_w
                    single_coordinate["patch_start_w"] = 0
                    single_coordinate["patch_end_w"] = img_w
                elif (num_w - 1) > 0:
                    if y == 0:
                        single_coordinate["stack_start_w"] = y * gap_w
                        single_coordinate["stack_end_w"] = y * gap_w + img_w - cut_w
                        single_coordinate["patch_start_w"] = 0
                        single_coordinate["patch_end_w"] = img_w - cut_w
                    elif y == num_w - 1:
                        single_coordinate["stack_start_w"] = whole_w - img_w + cut_w
                        single_coordinate["stack_end_w"] = whole_w
                        single_coordinate["patch_start_w"] = cut_w
                        single_coordinate["patch_end_w"] = img_w
                    else:
                        single_coordinate["stack_start_w"] = y * gap_w + cut_w
                        single_coordinate["stack_end_w"] = y * gap_w + img_w - cut_w
                        single_coordinate["patch_start_w"] = cut_w
                        single_coordinate["patch_end_w"] = img_w - cut_w

                if (num_h - 1) == 0:
                    single_coordinate["stack_start_h"] = 0
                    single_coordinate["stack_end_h"] = whole_h
                    single_coordinate["patch_start_h"] = 0
                    single_coordinate["patch_end_h"] = img_h
                elif (num_h - 1) > 0:
                    if x == 0:
                        single_coordinate["stack_start_h"] = x * gap_h
                        single_coordinate["stack_end_h"] = x * gap_h + img_h - cut_h
                        single_coordinate["patch_start_h"] = 0
                        single_coordinate["patch_end_h"] = img_h - cut_h
                    elif x == num_h - 1:
                        single_coordinate["stack_start_h"] = whole_h - img_h + cut_h
                        single_coordinate["stack_end_h"] = whole_h
                        single_coordinate["patch_start_h"] = cut_h
                        single_coordinate["patch_end_h"] = img_h
                    else:
                        single_coordinate["stack_start_h"] = x * gap_h + cut_h
                        single_coordinate["stack_end_h"] = x * gap_h + img_h - cut_h
                        single_coordinate["patch_start_h"] = cut_h
                        single_coordinate["patch_end_h"] = img_h - cut_h

                if (num_s - 1) == 0:
                    single_coordinate["stack_start_s"] = 0
                    single_coordinate["stack_end_s"] = whole_s
                    single_coordinate["patch_start_s"] = 0
                    single_coordinate["patch_end_s"] = img_s2
                elif (num_s - 1) > 0:
                    if z == 0:
                        single_coordinate["stack_start_s"] = z * gap_s2
                        single_coordinate["stack_end_s"] = (
                            z * gap_s2 + img_s2 - cut_s
                        )
                        single_coordinate["patch_start_s"] = 0
                        single_coordinate["patch_end_s"] = img_s2 - cut_s
                    elif z == num_s - 1:
                        single_coordinate["stack_start_s"] = (
                            whole_s - img_s2 + cut_s
                        )
                        single_coordinate["stack_end_s"] = whole_s
                        single_coordinate["patch_start_s"] = cut_s
                        single_coordinate["patch_end_s"] = img_s2
                    else:
                        single_coordinate["stack_start_s"] = z * gap_s2 + cut_s
                        single_coordinate["stack_end_s"] = (
                            z * gap_s2 + img_s2 - cut_s
                        )
                        single_coordinate["patch_start_s"] = cut_s
                        single_coordinate["patch_end_s"] = img_s2 - cut_s

                patch_name = "x" + str(x) + "_y" + str(y) + "_z" + str(z)
                name_list.append(patch_name)
                coordinate_list[patch_name] = single_coordinate

    return name_list, noise_im, coordinate_list
