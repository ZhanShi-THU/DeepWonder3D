import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
import sys
import math
import scipy.io as scio

from skimage import io
import numpy as np
import math

from .SEG_utils import FFDrealign4, inv_FFDrealign4
from .SEG_network import SEG_Network_3D_Unet
from .SEG_data_process import (
    test_preprocess_lessMemoryNoTail_SubImgSEG,
    testset,
    singlebatch_test_save,
    multibatch_test_save,
)


def seg_3dunet_ffd(
    net,
    sub_img,
    SEG_ffd: bool = True,
    SEG_GPU: str = "0",
    SEG_batch_size: int = 1,
    SEG_img_w: int = 256,
    SEG_img_h: int = 256,
    SEG_img_s: int = 64,
    SEG_gap_w: int = 224,
    SEG_gap_h: int = 224,
    SEG_gap_s: int = 32,
    SEG_normalize_factor: float = 1000,
):
    """
    Run 3D U-Net based segmentation with optional FFD realignment.

    The input volume is split into overlapping 3D patches, optionally
    realigned with 4-way FFD before being passed through the network.
    Predictions are then inverse-aligned and stitched back into a 3D
    segmentation mask.

    Args:
        net (torch.nn.Module): Segmentation network (e.g. ``SEG_Network_3D_Unet``).
        sub_img (np.ndarray): Input 3D volume of shape
            ``(depth, height, width)``.
        SEG_ffd (bool): If True, enable FFDrealign4 / inv_FFDrealign4.
        SEG_GPU (str): CUDA device index as string.
        SEG_batch_size (int): Inference batch size.
        SEG_img_w (int): Patch width.
        SEG_img_h (int): Patch height.
        SEG_img_s (int): Patch depth (number of slices).
        SEG_gap_w (int): Horizontal stride between patches.
        SEG_gap_h (int): Vertical stride between patches.
        SEG_gap_s (int): Depth stride between patches.
        SEG_normalize_factor (float): Normalization factor used in
            preprocessing.

    Returns:
        np.ndarray: 3D segmentation mask of shape
        ``(num_patches_in_depth, height, width)`` where the first dimension
        corresponds to stacked patch indices along depth.
    """
    opt = {}
    opt["GPU"] = SEG_GPU
    opt["batch_size"] = SEG_batch_size
    opt["img_w"] = SEG_img_w
    opt["img_h"] = SEG_img_h
    opt["img_s"] = SEG_img_s

    opt["gap_w"] = SEG_gap_w
    opt["gap_h"] = SEG_gap_h
    opt["gap_s"] = SEG_gap_s

    opt["normalize_factor"] = SEG_normalize_factor

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt["GPU"])
    if torch.cuda.is_available():
        net = net.cuda()
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    name_list, noise_img, coordinate_list = test_preprocess_lessMemoryNoTail_SubImgSEG(
        opt, sub_img
    )

    prev_time = time.time()
    time_start = time.time()

    num_s = math.ceil((noise_img.shape[0] - SEG_img_s + SEG_gap_s) / SEG_gap_s)
    denoise_img = np.zeros((num_s, noise_img.shape[1], noise_img.shape[2]))

    test_data = testset(name_list, coordinate_list, noise_img)
    testloader = DataLoader(
        test_data, batch_size=opt["batch_size"], shuffle=False, num_workers=4, pin_memory=False
    )
    for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
        noise_patch = noise_patch.cuda().float()

        if SEG_ffd:
            real_A = FFDrealign4(noise_patch)
        else:
            real_A = noise_patch
        real_A = Variable(real_A)

        fake_B = net(real_A)

        batches_done = iteration
        batches_left = 1 * len(testloader) - batches_done
        time_left_seconds = int(batches_left * (time.time() - prev_time))
        time_left = datetime.timedelta(seconds=time_left_seconds)
        prev_time = time.time()
        if iteration % 1 == 0:
            time_end = time.time()
            time_cost = time_end - time_start
            print(
                "\r\033[1;31m[SEG]\033[0m [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]    "
                % (iteration + 1, len(testloader), time_cost, time_left_seconds),
                end=" ",
            )

        if (iteration + 1) % len(testloader) == 0:
            print("\n", end=" ")

        if SEG_ffd:
            fake_B_realign = inv_FFDrealign4(fake_B)
        else:
            fake_B_realign = fake_B

        output_image = np.squeeze(fake_B_realign.cpu().detach().numpy())
        real_A_realign = real_A
        raw_image = np.squeeze(real_A_realign.cpu().detach().numpy())
        if output_image.ndim == 2:
            turn = 1
        else:
            turn = output_image.shape[0]

        if turn > 1:
            for id in range(turn):
                (
                    aaaa,
                    stack_start_w,
                    stack_end_w,
                    stack_start_h,
                    stack_end_h,
                    stack_start_s,
                ) = multibatch_test_save(single_coordinate, id, output_image)
                denoise_img[
                    stack_start_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w
                ] = aaaa

        else:
            (
                aaaa,
                stack_start_w,
                stack_end_w,
                stack_start_h,
                stack_end_h,
                stack_start_s,
            ) = singlebatch_test_save(single_coordinate, output_image)
            denoise_img[
                stack_start_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w
            ] = aaaa
    return denoise_img

