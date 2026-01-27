import numpy as np
import os
import tifffile as tiff
import random
import math
import torch
from torch.utils.data import Dataset


def img_remove_time_ave(img):
    """
    Remove frame-wise temporal mean from a 3D stack.

    For each spatial location, the mean over time is subtracted so that
    each frame is centered around zero.

    Args:
        img (np.ndarray): Input array with shape ``(depth, height, width)``.

    Returns:
        np.ndarray: Mean-subtracted array with the same shape as ``img``.
    """
    noise_im_ave_single = np.mean(img, axis=0)
    noise_im_ave = np.zeros(img.shape)
    for i in range(0, img.shape[0]):
        noise_im_ave[i, :, :] = noise_im_ave_single
    img = img - noise_im_ave
    return img


class testset_RMBG(Dataset):
    """
    RMBG test dataset that yields patches and patch metadata.

    Args:
        img (np.ndarray): Full-volume array with shape ``(depth, height, width)``.
        per_patch_list (list[str]): List of patch names to iterate over.
        per_coor_list (dict): Mapping from patch name to a coordinate dict
            with keys ``'init_h'``, ``'end_h'``, ``'init_w'``, ``'end_w'``,
            ``'init_s'``, ``'end_s'``.

    __getitem__ output:
        - img_patch (torch.Tensor): Cropped patch with shape ``(1, D, H, W)``.
        - per_coor (dict): Coordinate dictionary for this patch.
        - patch_name (str): Identifier string for the patch.
    """

    def __init__(self, img, per_patch_list, per_coor_list):
        self.per_patch_list = per_patch_list
        self.per_coor_list = per_coor_list
        self.img = img

    def __getitem__(self, index):
        patch_name = self.per_patch_list[index]
        per_coor = self.per_coor_list[patch_name]
        init_h = per_coor["init_h"]
        end_h = per_coor["end_h"]
        init_w = per_coor["init_w"]
        end_w = per_coor["end_w"]
        init_s = per_coor["init_s"]
        end_s = per_coor["end_s"]
        img_patch = self.img[init_s:end_s, init_h:end_h, init_w:end_w]
        img_patch = torch.from_numpy(np.expand_dims(img_patch, 0))
        return img_patch, per_coor, patch_name

    def __len__(self):
        return len(self.per_patch_list)


def test_preprocess_lessMemory_RMBG(args):
    """
    Pre-process all TIFF stacks into RMBG patches for testing (low-memory path).

    Args:
        args: Configuration object with fields:
            - ``RMBG_img_h``, ``RMBG_img_w``, ``RMBG_img_s``: patch size.
            - ``RMBG_gap_h``, ``RMBG_gap_w``, ``RMBG_gap_s``: patch strides.
            - ``RMBG_input_pretype`` (str): Reserved flag for input pre-type.
            - ``RMBG_datasets_path`` (str): Path to dataset root.
            - ``RMBG_datasets_folder`` (str): Folder containing TIFF stacks.
            - ``RMBG_select_img_num`` (int): Maximum number of slices to keep.
            - ``RMBG_norm_factor`` (float): Normalization factor.

    Returns:
        tuple:
            - name_list (list[str]): Per-stack names (file names).
            - patch_name_list (dict): Mapping from stack name to list of patch
              identifiers in that stack.
            - img_list (dict): Mapping from stack name to normalized image array.
            - coordinate_list (dict): Mapping from stack name to per-patch
              coordinate dictionaries.
    """
    img_h = args.RMBG_img_h
    img_w = args.RMBG_img_w
    img_s2 = args.RMBG_img_s
    gap_h = args.RMBG_gap_h
    gap_w = args.RMBG_gap_w
    gap_s2 = args.RMBG_gap_s

    input_pretype = args.RMBG_input_pretype
    datasets_path = args.RMBG_datasets_path
    datasets_folder = args.RMBG_datasets_folder
    select_img_num = args.RMBG_select_img_num
    normalize_factor = args.RMBG_norm_factor

    cut_w = (img_w - gap_w) / 2
    cut_h = (img_h - gap_h) / 2
    cut_s = (img_s2 - gap_s2) / 2

    im_folder = datasets_path + "//" + datasets_folder
    patch_name_list = {}
    name_list = []
    coordinate_list = {}
    img_list = {}

    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        name_list.append(im_name)
        im_dir = im_folder + "//" + im_name
        noise_im = tiff.imread(im_dir)
        if noise_im.shape[0] > select_img_num:
            noise_im = noise_im[-select_img_num:, :, :]

        noise_im = noise_im.astype(np.float32) / normalize_factor

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]

        img_list[im_name] = noise_im

        single_im_coordinate_list, sub_patch_name_list = get_test_patch_list(
            im_name,
            whole_w,
            whole_h,
            whole_s,
            img_w,
            img_h,
            img_s2,
            gap_w,
            gap_h,
            gap_s2,
            cut_w,
            cut_h,
            cut_s,
        )
        coordinate_list[im_name] = single_im_coordinate_list
        patch_name_list[im_name] = sub_patch_name_list
    return name_list, patch_name_list, img_list, coordinate_list


def test_preprocess_lessMemory_RMBG_lm(args):
    """
    Pre-process TIFF stacks into RMBG patches, returning only file paths.

    Compared to ``test_preprocess_lessMemory_RMBG``, this variant avoids
    keeping the full images in memory and instead returns per-image paths
    together with patch coordinate metadata.

    Args:
        args: Same configuration as in ``test_preprocess_lessMemory_RMBG``.

    Returns:
        tuple:
            - name_list (list[str]): Stack identifiers (file names).
            - patch_name_list (dict): Mapping from stack name to list of patch
              identifiers.
            - img_dir_list (dict): Mapping from stack name to TIFF file path.
            - coordinate_list (dict): Mapping from stack name to patch
              coordinate dictionaries.
    """
    img_h = args.RMBG_img_h
    img_w = args.RMBG_img_w
    img_s2 = args.RMBG_img_s
    gap_h = args.RMBG_gap_h
    gap_w = args.RMBG_gap_w
    gap_s2 = args.RMBG_gap_s

    input_pretype = args.RMBG_input_pretype
    datasets_path = args.RMBG_datasets_path
    datasets_folder = args.RMBG_datasets_folder
    select_img_num = args.RMBG_select_img_num
    normalize_factor = args.RMBG_norm_factor

    cut_w = (img_w - gap_w) / 2
    cut_h = (img_h - gap_h) / 2
    cut_s = (img_s2 - gap_s2) / 2

    im_folder = datasets_path + "//" + datasets_folder
    print("im_folder ----> ", im_folder)
    patch_name_list = {}
    name_list = []
    coordinate_list = {}

    img_dir_list = {}

    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        if ".tif" in im_name:
            im_dir = im_folder + "//" + im_name
            print("im_dir -----> ", im_dir)
            name_list.append(im_name)

            img_dir_list[im_name] = im_dir

            noise_im = tiff.imread(im_dir)
            if noise_im.shape[0] > select_img_num:
                noise_im = noise_im[-select_img_num:, :, :]

            noise_im = noise_im.astype(np.float32) / normalize_factor

            whole_w = noise_im.shape[2]
            whole_h = noise_im.shape[1]
            whole_s = noise_im.shape[0]

            del noise_im

            single_im_coordinate_list, sub_patch_name_list = get_test_patch_list(
                im_name,
                whole_w,
                whole_h,
                whole_s,
                img_w,
                img_h,
                img_s2,
                gap_w,
                gap_h,
                gap_s2,
                cut_w,
                cut_h,
                cut_s,
            )

            coordinate_list[im_name] = single_im_coordinate_list
            patch_name_list[im_name] = sub_patch_name_list
    return name_list, patch_name_list, img_dir_list, coordinate_list


def get_test_patch_list(
    im_name,
    whole_w,
    whole_h,
    whole_s,
    img_w,
    img_h,
    img_s2,
    gap_w,
    gap_h,
    gap_s2,
    cut_w,
    cut_h,
    cut_s,
):
    """
    Generate a list of patch coordinates for a single stack.

    Args:
        im_name (str): Source image name (used when forming patch IDs).
        whole_w, whole_h, whole_s (int): Full-volume dimensions (width, height, depth).
        img_w, img_h, img_s2 (int): Patch dimensions.
        gap_w, gap_h, gap_s2 (int): Strides between adjacent patches.
        cut_w, cut_h, cut_s (float): Half-overlap crop sizes used for stitching.

    Returns:
        tuple:
            - single_im_coordinate_list (dict): Mapping from patch name to
              coordinate dictionary describing crop and placement indices.
            - sub_patch_name_list (list[str]): All patch names for this stack.
    """
    num_w = math.ceil((whole_w - img_w + gap_w) / gap_w)
    num_h = math.ceil((whole_h - img_h + gap_h) / gap_h)
    num_s = math.ceil((whole_s - img_s2 + gap_s2) / gap_s2)

    single_im_coordinate_list = {}
    sub_patch_name_list = []
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

                if y == 0:
                    single_coordinate["stack_start_w"] = y * gap_w
                    single_coordinate["stack_end_w"] = y * gap_w + img_w
                    single_coordinate["patch_start_w"] = 0
                    single_coordinate["patch_end_w"] = img_w
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

                if x == 0:
                    single_coordinate["stack_start_h"] = x * gap_h
                    single_coordinate["stack_end_h"] = x * gap_h + img_h
                    single_coordinate["patch_start_h"] = 0
                    single_coordinate["patch_end_h"] = img_h
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

                if z == 0:
                    single_coordinate["stack_start_s"] = z * gap_s2
                    single_coordinate["stack_end_s"] = z * gap_s2 + img_s2
                    single_coordinate["patch_start_s"] = 0
                    single_coordinate["patch_end_s"] = img_s2
                elif z == num_s - 1:
                    single_coordinate["stack_start_s"] = whole_s - img_s2 + cut_s
                    single_coordinate["stack_end_s"] = whole_s
                    single_coordinate["patch_start_s"] = cut_s
                    single_coordinate["patch_end_s"] = img_s2
                else:
                    single_coordinate["stack_start_s"] = z * gap_s2 + cut_s
                    single_coordinate["stack_end_s"] = z * gap_s2 + img_s2 - cut_s
                    single_coordinate["patch_start_s"] = cut_s
                    single_coordinate["patch_end_s"] = img_s2 - cut_s

                patch_name = (
                    im_name.replace(".tif", "")
                    + "_x"
                    + str(init_h)
                    + "_y"
                    + str(init_w)
                    + "_z"
                    + str(init_s)
                )
                sub_patch_name_list.append(patch_name)
                single_im_coordinate_list[patch_name] = single_coordinate

    return single_im_coordinate_list, sub_patch_name_list
