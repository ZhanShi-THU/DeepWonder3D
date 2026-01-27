import tifffile as tiff
import cv2
import numpy as np
from skimage import io

import math
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import math

from .connect_components import cal_pccs
from .connect_components import Two_Pass, four_cc_label, Seed_Filling
from .connect_components2 import New_Two_Pass

import torch

import time
import datetime
import os
import scipy.io as scio
import multiprocessing as mp
import random



def z_group(img, z_num):
    """
    Group consecutive z-slices by taking maximum projection.

    Reduces temporal/depth dimension by grouping consecutive slices and taking
    the maximum value across each group. Used for temporal downsampling or
    depth compression.

    Args:
        img (np.ndarray): 3D image stack with shape (T, H, W)
            where T is number of time frames or depth slices
        z_num (int): Number of consecutive slices to group together

    Returns:
        np.ndarray: Grouped image with shape (T', H, W) where T' = ceil(T/z_num)
            Each slice is the maximum projection of z_num consecutive slices

    Note:
        - Last group may contain fewer slices if T is not divisible by z_num
        - Uses maximum projection (max pooling) across grouped slices
    """
    img_z = img.shape[0]
    img_z_group = math.ceil(img_z/z_num)
    img_grouped = np.zeros((img_z_group, img.shape[1], img.shape[2]))
    for i in range(0, img_z_group):
        if i<(img_z_group-1):
            img_sub = img[i*z_num:i*z_num+z_num,:,:]
            img_grouped[i,:,:] = np.max(img_sub, axis=0)
        if i==(img_z_group-1):
            img_sub = img[img_z-z_num:,:,:]
            img_grouped[i,:,:] = np.max(img_sub, axis=0)
    return img_grouped


def Get_Contours(position, mask_h, mask_w):
    """
    Extract contours from a set of pixel positions.

    Converts a list of pixel coordinates into OpenCV contours by creating
    a binary mask and finding contours.

    Args:
        position (np.ndarray): Array of pixel positions with shape (N, 2)
            Each row is [row, col] coordinates
        mask_h (int): Height of the mask
        mask_w (int): Width of the mask

    Returns:
        list: List of contours found by cv2.findContours
            Each contour is a numpy array of points

    Note:
        - Creates a binary mask with pixels at given positions set to 255
        - Uses cv2.findContours with RETR_TREE and CHAIN_APPROX_SIMPLE
        - Returns all contours found in the mask
    """
    mask_j = np.zeros((mask_h, mask_w), np.uint8)
    for p_i in range(0, position.shape[0]):
        # print(position[ii,:])
        now_position = position[p_i,:]
        mask_j[now_position[0], now_position[1]] = 255
    mask_jRGB = cv2.cvtColor(mask_j, cv2.COLOR_GRAY2BGR)
    
    mask_jgray = cv2.cvtColor(mask_jRGB, cv2.COLOR_BGR2GRAY)  
    ret, binary = cv2.threshold(mask_jgray,100,255,cv2.THRESH_BINARY)    
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours


def initial_mask_list(mask, 
                        quit_round_rate = 0.5, 
                        good_round_rate = 0.85,
                        smallest_neuron_area = 64):
    """
    Initialize neuron list from segmentation mask by filtering connected components.

    Performs connected component labeling on the mask and filters components
    based on roundness and area criteria. Separates neurons into "good" and
    "bad" categories based on roundness threshold.

    Args:
        mask (np.ndarray): Segmentation mask with shape (H, W)
        quit_round_rate (float, optional): Minimum roundness rate to keep neuron.
            Default is 0.5.
        good_round_rate (float, optional): Threshold for "good" roundness.
            Neurons with roundness >= good_round_rate are classified as good.
            Default is 0.85.
        smallest_neuron_area (int, optional): Minimum area in pixels to consider
            as a neuron. Default is 64.

    Returns:
        tuple: (good_neuron_list, bad_neuron_list)
            - good_neuron_list (list): List of dictionaries for good neurons,
              each containing 'position', 'split', 'round_rate'
            - bad_neuron_list (list): List of dictionaries for bad neurons,
              same structure as good_neuron_list

    Note:
        - Uses 8-connectivity for connected component labeling
        - Roundness calculated as: 4*π*area / perimeter²
        - Only components with single contour are considered
        - Good neurons have roundness >= good_round_rate
        - Bad neurons have roundness between quit_round_rate and good_round_rate
    """
    mask_nor = mask/np.max(mask)*255
    neuron_size = 20
    neuron_area = math.pi*neuron_size*neuron_size/4
    # smallest_neuron_area = 50
    max_single_neuron = neuron_area*1.2

    cc_mask = New_Two_Pass(mask_nor, 'NEIGHBOR_HOODS_8')
    max_cc_mask = int(np.max(cc_mask))
    min_cc_mask = int(np.min(cc_mask))
    good_neuron_list = []
    bad_neuron_list = []

    for i in range(min_cc_mask+1, max_cc_mask+1):
        # print('i -----> ',i)
        position = np.argwhere(cc_mask == i)

        if position.shape[0] != 0:

            mask_j = np.zeros(mask.shape, np.uint8)
            for p_i in range(0, position.shape[0]):
                # print(position[ii,:])
                now_position = position[p_i,:]
                mask_j[now_position[0], now_position[1]] = 255
            mask_jRGB = cv2.cvtColor(mask_j, cv2.COLOR_GRAY2BGR)
            
            mask_jgray = cv2.cvtColor(mask_jRGB, cv2.COLOR_BGR2GRAY)  
            ret, binary = cv2.threshold(mask_jgray, 100, 255, cv2.THRESH_BINARY)    
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours)==1:
                cnt = contours[0]
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt,True)
                if perimeter!=0:
                    round_rate = 4*math.pi*area/perimeter/perimeter

                    if round_rate>quit_round_rate:
                        if area>smallest_neuron_area:
                            # if area<max_single_neuron:
                            if True:
                                if round_rate>=good_round_rate:
                                    good_single_neuron = {}
                                    good_single_neuron['position'] = []
                                    good_single_neuron['split'] = 0
                                    good_single_neuron['round_rate'] = round_rate

                                    for ii in range(1, position.shape[0]):
                                        position_list = list(position[ii,:])
                                        good_single_neuron['position'].append(position_list)
                                    # single_neuron['split'] = 0
                                    good_neuron_list.append(good_single_neuron)

                                if round_rate<good_round_rate:
                                    bad_single_neuron = {}
                                    bad_single_neuron['position'] = []
                                    bad_single_neuron['split'] = 0
                                    bad_single_neuron['round_rate'] = round_rate

                                    for ii in range(1, position.shape[0]):
                                        position_list = list(position[ii,:])
                                        bad_single_neuron['position'].append(position_list)
                                    # single_neuron['split'] = 0
                                    bad_neuron_list.append(bad_single_neuron)
    return good_neuron_list, bad_neuron_list


def neuron_filter(mask_list, min_area, max_area, round_rate):
    """
    Filter neuron list based on area and roundness criteria.

    Placeholder function for filtering neurons by area and roundness thresholds.
    Currently not implemented (pass statement).

    Args:
        mask_list (list): List of neuron dictionaries
        min_area (int): Minimum area threshold
        max_area (int): Maximum area threshold
        round_rate (float): Minimum roundness threshold

    Returns:
        None: Function not yet implemented

    Note:
        - This function is a placeholder and currently does nothing
        - Intended to filter neurons based on area and roundness criteria
    """
    pass

def Mining_rest_neuron(w_g_neuron_list, w_b_neuron_list, img, quit_round_rate = 0.5, smallest_neuron_area = 100):
    """
    Mine additional neurons from remaining regions after removing good neurons.

    Extracts neurons from regions that remain after subtracting good neuron
    masks from bad neuron masks. This helps recover neurons that were
    partially occluded or merged with good neurons.

    Args:
        w_g_neuron_list (list): List of good neuron dictionaries
            Each dictionary contains 'position' key with pixel coordinates
        w_b_neuron_list (list): List of bad neuron dictionaries
            Same structure as w_g_neuron_list
        img (np.ndarray): Image stack with shape (T, H, W) for reference
        quit_round_rate (float, optional): Minimum roundness rate to keep neuron.
            Default is 0.5.
        smallest_neuron_area (int, optional): Minimum area in pixels.
            Default is 100.

    Returns:
        list: List of newly discovered neuron dictionaries
            Each dictionary contains 'position', 'split', 'round_rate'
            Only includes neurons with roundness > quit_round_rate and
            area > smallest_neuron_area

    Note:
        - Computes difference: rest_mask = bad_neuron_mask - good_neuron_mask
        - Finds contours in remaining regions
        - Selects largest contour from each remaining region
        - Filters by roundness and area criteria
    """
    w_g_neuron_mask = np.zeros((img.shape[1],img.shape[2]))
    for i in range(0, len(w_g_neuron_list)):
        g_neuron = w_g_neuron_list[i]
        g_neuron_position = g_neuron['position']
        for p_i in range(0, len(g_neuron_position)):
            w_g_neuron_mask[g_neuron_position[p_i][0], g_neuron_position[p_i][1]] = 1
    
    rest_mask_stack = np.zeros((len(w_b_neuron_list),img.shape[1],img.shape[2]))
    add_neuron_list =[]
    for i in range(0, len(w_b_neuron_list)):
        b_neuron = w_b_neuron_list[i]
        b_neuron_position = b_neuron['position']
        b_neuron_mask = np.zeros((img.shape[1],img.shape[2]))
        for p_i in range(0, len(b_neuron_position)):
            b_neuron_mask[b_neuron_position[p_i][0], b_neuron_position[p_i][1]] = 1
        rest_mask = b_neuron_mask-w_g_neuron_mask
        rest_mask[rest_mask>0] = 255
        rest_mask[rest_mask<0] = 0
        rest_mask_stack[i,:,:] = rest_mask
        rest_mask1 = rest_mask.astype(np.uint8)
        if np.sum(rest_mask)>0:
            # print(p_i,' Mining_rest_neuron ---> ', len(b_neuron_position))
            position = np.argwhere(rest_mask1 == 255)
            rest_maskRGB = cv2.cvtColor(rest_mask1, cv2.COLOR_GRAY2BGR)
            
            rest_maskgray = cv2.cvtColor(rest_maskRGB, cv2.COLOR_BGR2GRAY)  
            ret, binary = cv2.threshold(rest_maskgray,100,255,cv2.THRESH_BINARY)    
            contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            max_area_index = 0
            max_area = 0
            for iii in range(0, len(contours)):
                if cv2.contourArea(contours[iii])>max_area:
                    max_area = cv2.contourArea(contours[iii])
                    max_area_index = iii
            cnt = contours[max_area_index]
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)
            if area>smallest_neuron_area:
                round_rate = 4*math.pi*area/perimeter/perimeter
                # print('round_rate ---> ',round_rate, max_area_index, area)
                clear_mask_j = np.zeros((img.shape[1], img.shape[2]), np.uint8)
                clear_mask_j =cv2.drawContours(clear_mask_j,contours,max_area_index,255,cv2.FILLED)

                position11 = np.argwhere(clear_mask_j == 255)
                if round_rate>quit_round_rate:
                    single_neuron = {}
                    single_neuron['position'] = []
                    single_neuron['split'] = 0
                    single_neuron['round_rate'] = round_rate   
                    for ii in range(1, position11.shape[0]):
                        position_list = list(position11[ii,:])
                        single_neuron['position'].append(position_list)
                    add_neuron_list.append(single_neuron)
    # io.imsave(save_folder+'//'+'rest.tif', rest_mask_stack)
    return add_neuron_list
    

def Neuron_List_Initial(mask, 
                        image, 
                        quit_round_rate = 0.5, 
                        good_round_rate = 0.8, 
                        good_round_size_rate = 0.5):
    """
    Initialize neuron list from segmentation mask with comprehensive filtering.

    Performs connected component labeling and filters neurons based on
    roundness, area, and size criteria. More comprehensive than initial_mask_list,
    includes additional filtering and processing steps.

    Args:
        mask (np.ndarray): Segmentation mask with shape (H, W)
        image (np.ndarray): Original image stack with shape (T, H, W) for reference
        quit_round_rate (float, optional): Minimum roundness rate to keep neuron.
            Default is 0.5.
        good_round_rate (float, optional): Threshold for "good" roundness.
            Default is 0.8.
        good_round_size_rate (float, optional): Size rate threshold for good neurons.
            Default is 0.5.

    Returns:
        list: List of neuron dictionaries, each containing:
            - 'position': List of [row, col] pixel coordinates
            - 'split': Split flag (0 = not split)
            - 'round_rate': Calculated roundness value
            - Additional properties based on filtering criteria

    Note:
        - Uses 8-connectivity for connected component labeling
        - Filters by roundness, area, and size criteria
        - Calculates roundness as: 4*π*area / perimeter²
        - Only processes components with single contour
    """
    neuron_size = 20
    neuron_area = math.pi*neuron_size*neuron_size/4
    smallest_neuron_area = 50
    mask_nor = mask/np.max(mask)*255
    rest_rate = 1

    cc_mask = New_Two_Pass(mask_nor, 'NEIGHBOR_HOODS_8')
    max_cc_mask = int(np.max(cc_mask))
    min_cc_mask = int(np.min(cc_mask))
    neuron_list = []

    for i in range(min_cc_mask+1, max_cc_mask+1):
        # print('i -----> ',i)
        position = np.argwhere(cc_mask == i)

        if position.shape[0] != 0:

            mask_j = np.zeros(mask.shape, np.uint8)
            for p_i in range(0, position.shape[0]):
                # print(position[ii,:])
                now_position = position[p_i,:]
                mask_j[now_position[0], now_position[1]] = 255
            mask_jRGB = cv2.cvtColor(mask_j, cv2.COLOR_GRAY2BGR)
            
            mask_jgray = cv2.cvtColor(mask_jRGB, cv2.COLOR_BGR2GRAY)  
            ret, binary = cv2.threshold(mask_jgray,100,255,cv2.THRESH_BINARY)    
            contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            if len(contours)==1:
                cnt = contours[0]
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt,True)
                if perimeter!=0:
                    round_rate = 4*math.pi*area/perimeter/perimeter
                    # print('area -----> ', area, ' perimeter -----> ', perimeter, ' round_rate -----> ', round_rate,)
                    
                    if round_rate>quit_round_rate:
                        if (area<=neuron_area*rest_rate) and (area>smallest_neuron_area):
                            ellipse = cv2.fitEllipse(cnt)
                            (xc,yc),(d1,d2),angle = ellipse
                            size_rate = d2/neuron_size
                            nli_size_rate = size_rate**2+1/(size_rate**2)-1
                            round_size_rate = round_rate/nli_size_rate

                            if round_rate>=good_round_rate:
                                single_neuron = {}
                                single_neuron['position'] = []
                                single_neuron['split'] = 0

                                for ii in range(1, position.shape[0]):
                                    position_list = list(position[ii,:])
                                    single_neuron['position'].append(position_list)
                                # single_neuron['split'] = 0
                                neuron_list.append(single_neuron)
            
                        if area>neuron_area*rest_rate:
                            ellipse = cv2.fitEllipse(cnt)
                            (xc,yc),(d1,d2),angle = ellipse
                            size_rate = d2/neuron_size
                            nli_size_rate = size_rate**2+1/(size_rate**2)-1
                            round_size_rate = round_rate/nli_size_rate

                            if round_size_rate>=good_round_size_rate:
                                single_neuron = {}
                                single_neuron['position'] = []
                                single_neuron['split'] = 0

                                for ii in range(1, position.shape[0]):
                                    position_list = list(position[ii,:])
                                    single_neuron['position'].append(position_list)
                                # single_neuron['split'] = 0
                                neuron_list.append(single_neuron)
                                # print('ADD')

                            if round_size_rate<good_round_size_rate:
                                # print('round_size_rate -----> ',round_size_rate,good_round_size_rate)
                                # print('round_rate -----> ',round_rate)
                                single_neuron = {}
                                single_neuron['position'] = []
                                single_neuron['split'] = 0
                                for ii in range(1, position.shape[0]):
                                    position_list = list(position[ii,:])
                                    single_neuron['position'].append(position_list)
                                Split_Neuron_list = Split_Neuron(single_neuron, image, 0.6, 1)
                                # print('Split_Neuron_list  ',len(Split_Neuron_list))
                                neuron_list.extend(Split_Neuron_list)
    '''
    for i in range(0,len(neuron_list)):
        single_neuron = neuron_list[i]
        if 'split' in single_neuron:
            print('split -----> ',single_neuron['split'])
        if not 'split' in single_neuron:
            print('split -----> no key')
    '''
    return neuron_list


def Split_Neuron(single_neuron, image, quit_round_rate, rest_rate):
    """
    Split a large neuron into multiple smaller neurons using NMF decomposition.

    Uses Non-negative Matrix Factorization (NMF) to decompose a large neuron
    mask into multiple components based on temporal signal patterns. Each
    component is then validated and converted to a separate neuron if it meets
    roundness criteria.

    Args:
        single_neuron (dict): Neuron dictionary containing 'position' key with
            list of [row, col] pixel coordinates
        image (np.ndarray): Image stack with shape (T, H, W)
            Temporal dimension T, spatial dimensions H, W
        quit_round_rate (float): Minimum roundness rate to keep split neuron
        rest_rate (float): Rate parameter for calculating NMF dimensions

    Returns:
        list: List of neuron dictionaries after splitting
            Each dictionary contains 'position', 'trace', 'split' keys
            - If neuron cannot be split (nmf_dim <= 1), returns original neuron
            - If split, returns multiple neurons with 'split' = 1

    Note:
        - Uses NMF to decompose temporal signals at neuron pixel positions
        - Number of components: round((area - neuron_area*rest_rate) / neuron_area/rest_rate) + 1
        - Each NMF component is thresholded and converted to a mask
        - Only components with single contour and roundness > quit_round_rate are kept
        - Uses 'nndsvd' initialization for NMF
    """
    position = single_neuron['position']
    len_p = len(position)
    mask_matrix = np.zeros((image.shape[0], len_p))
    for i in range(0, len_p):
        mask_matrix[:,i] = image[:, position[i][0], position[i][1]]

    neuron_list = []
    mask_matrix = mask_matrix.T
    neuron_size = 20
    neuron_area = math.pi*neuron_size*neuron_size/4
    whole_area = len(position)
    # rest_rate = 1
    nmf_dim = round((whole_area-neuron_area*rest_rate)/neuron_area/rest_rate)+1
    if nmf_dim<=1:
        single_neuron['split'] = 0
        neuron_list.append(single_neuron)

    if nmf_dim>1:
        # print('nmf_dim ',nmf_dim, whole_area) 'random' nndsvd
        nmf_model = NMF(n_components=nmf_dim, init='nndsvd', random_state=0, max_iter=200)
        W = nmf_model.fit_transform(mask_matrix)
        H = nmf_model.components_
        # print('H -----> ',H.shape)
        # pccs1 = cal_pccs(H, H, H.shape[0])
        pccs = np.corrcoef(H, H)
        # print('pccs -----> ',pccs)
        mask_W = np.zeros(W.shape)
        for i in range(0,W.shape[0]):
            ww = W[i,:]
            ww_mask = np.zeros(ww.shape)
            max_ww = np.max(ww)
            min_ww = np.min(ww)
            threshold_ww = (max_ww-min_ww)*0.2+min_ww

            ww_mask[ww>threshold_ww] = 1
            mask_W[i,:] = ww_mask

        for i in range(0, mask_W.shape[1]):
            mask_no = np.zeros((image.shape[1], image.shape[2]))

            mask_j = np.zeros((image.shape[1], image.shape[2]), np.uint8)
            for j in range(0, mask_W.shape[0]):
                if mask_W[j,i] == 1:
                    mask_j[position[j][0], position[j][1]] = 255

            mask_jRGB = cv2.cvtColor(mask_j, cv2.COLOR_GRAY2BGR)
            
            mask_jgray = cv2.cvtColor(mask_jRGB, cv2.COLOR_BGR2GRAY)  
            ret, binary = cv2.threshold(mask_jgray,100,255,cv2.THRESH_BINARY)    
            contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # print('len(contours) -----> ',len(contours))

            if_hollow = 0
            max_con_len =0
            max_index = 0
            for con_i in range(0, len(contours)):
                if len(contours[con_i])>max_con_len:
                    max_con_len = len(contours[con_i])
                    max_con = contours[con_i]
                    max_index = con_i
                if len(contours[con_i])>10:
                    if_hollow =1+if_hollow

            if if_hollow==1:
                cnt = max_con
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt,True)

                clear_mask_j = np.zeros((image.shape[1], image.shape[2]), np.uint8)
                clear_mask_j =cv2.drawContours(clear_mask_j,contours,max_index,255,cv2.FILLED)

                if perimeter!=0:
                    round_rate = 4*math.pi*area/perimeter/perimeter
                    if round_rate>(quit_round_rate):
                        new_single_neuron={}
                        new_single_neuron['position'] = []
                        new_single_neuron['trace'] = H[i,:]
                        new_single_neuron['split'] = 1
                        new_position = np.argwhere(clear_mask_j == 255)
                        # io.imsave(save_folder+'//test'+str(whole_area)+'_'+str(i)+'_clear.tif', clear_mask_j)
                        for iii in range(1, new_position.shape[0]):
                            position_list = list(new_position[iii,:])
                            new_single_neuron['position'].append(position_list)
                        new_single_neuron['split'] = 1
                        neuron_list.append(new_single_neuron)

    return neuron_list


def SingleAddtrace1(single_seg, image, mode='add'):
    """
    Add temporal trace and centroid to a single neuron segment.

    Computes the average temporal trace and centroid coordinates for a neuron
    by averaging across all pixel positions in the neuron mask.

    Args:
        single_seg (dict): Neuron dictionary containing 'position' key with
            list of [row, col] pixel coordinates
        image (np.ndarray): Image stack with shape (T, H, W)
            Temporal dimension T, spatial dimensions H, W
        mode (str, optional): Operation mode. Default is 'add'.
            - 'add': Only add trace/centroid if they don't exist
            - 'update': Always update trace/centroid

    Returns:
        dict: Modified neuron dictionary with added/updated keys:
            - 'trace': Average temporal trace with shape (T,)
            - 'centroid': Centroid coordinates [row, col] with shape (2,)

    Note:
        - Trace is computed as mean of temporal signals at all pixel positions
        - Centroid is computed as mean of all pixel coordinates
        - In 'add' mode, only adds if 'trace' or 'centroid' keys don't exist
    """
    position = single_seg['position']
    if mode=='add':
        if not 'trace' in single_seg or not 'centroid' in single_seg:
            trace = np.zeros((image.shape[0], ))
            centroid = np.zeros((2,))
            for ii in range(0, len(position)):
                now_position = position[ii]
                single_trace = image[:, now_position[0], now_position[1]].squeeze()
                trace = trace+image[:, now_position[0], now_position[1]].squeeze()
                centroid = now_position+centroid
            centroid = centroid/len(position)
            ave_trace = trace/len(position)
            single_seg['centroid'] = centroid
            single_seg['trace'] = ave_trace
    if mode=='update':
        trace = np.zeros((image.shape[0], ))
        centroid = np.zeros((2,))
        for ii in range(0, len(position)):
            now_position = position[ii]
            single_trace = image[:, now_position[0], now_position[1]].squeeze()
            trace = trace+image[:, now_position[0], now_position[1]].squeeze()
            centroid = now_position+centroid
        centroid = centroid/len(position)
        ave_trace = trace/len(position)
        single_seg['centroid'] = centroid
        single_seg['trace'] = ave_trace
    return single_seg


def SingleAddtrace(single_seg, image):
    """
    Add temporal trace and centroid to a single neuron segment (always update).

    Computes and updates the average temporal trace and centroid coordinates
    for a neuron by averaging across all pixel positions.

    Args:
        single_seg (dict): Neuron dictionary containing 'position' key with
            list of [row, col] pixel coordinates
        image (np.ndarray): Image stack with shape (T, H, W)

    Returns:
        dict: Modified neuron dictionary with updated keys:
            - 'trace': Average temporal trace with shape (T,)
            - 'centroid': Centroid coordinates [row, col] with shape (2,)

    Note:
        - Always updates trace and centroid (unlike SingleAddtrace1 with mode='add')
        - Trace is mean of temporal signals at all pixel positions
        - Centroid is mean of all pixel coordinates
    """
    position = single_seg['position']
    trace = np.zeros((image.shape[0], ))
    centroid = np.zeros((2,))
    for ii in range(0, len(position)):
        now_position = position[ii]
        single_trace = image[:, now_position[0], now_position[1]].squeeze()
        trace = trace+image[:, now_position[0], now_position[1]].squeeze()
        centroid = now_position+centroid
    centroid = centroid/len(position)
    ave_trace = trace/len(position)
    single_seg['centroid'] = centroid
    single_seg['trace'] = ave_trace
    return single_seg


def listAddtrace3(list, image, mode='add'):
    """
    Add temporal traces to a list of neurons using multiprocessing.

    Computes temporal traces and centroids for all neurons in the list using
    parallel processing for improved performance.

    Args:
        list (list): List of neuron dictionaries, each containing 'position' key
        image (np.ndarray): Image stack with shape (T, H, W)
        mode (str, optional): Operation mode. Default is 'add' (not used in this version).

    Returns:
        list: List of neuron dictionaries with added 'trace' and 'centroid' keys
            Note: Currently returns original list (implementation may be incomplete)

    Note:
        - Uses multiprocessing with number of cores equal to CPU count
        - Processes neurons in parallel using pool.apply_async
        - Note: Return statement returns original list, not processed list
    """
    # mode update add
    # print('listAddtrace ---> ',len(list))
    num_cores = int(mp.cpu_count())
    pool = mp.Pool(num_cores)
    single_seg_dict = {}
    for i in range(0, len(list)):
        single_seg_dict[str(i)] = list[i]
    
    results = [pool.apply_async(SingleAddtrace, args=(single_seg, image)) for name, single_seg in single_seg_dict.items()]

    new_list = []
    for p in results:
        single_seg_new = p.get()
        new_list.append(single_seg_new)
    '''
    for i in range(0, len(list)):
        single_seg = list[i]
        single_seg = SingleAddtrace(single_seg, image, mode='add')
    '''
    return list


def listAddtrace2(list, image, mode='add'):
    """
    Add temporal traces to a list of neurons (sequential processing).

    Computes temporal traces and centroids for all neurons in the list using
    sequential processing.

    Args:
        list (list): List of neuron dictionaries, each containing 'position' key
        image (np.ndarray): Image stack with shape (T, H, W)
        mode (str, optional): Operation mode. Default is 'add'.

    Returns:
        list: List of neuron dictionaries with added 'trace' and 'centroid' keys

    Note:
        - Processes neurons sequentially (one at a time)
        - Uses SingleAddtrace1 with mode='add' for each neuron
    """
    # mode update add
    # print('listAddtrace ---> ',len(list))
    for i in range(0, len(list)):
        single_seg = list[i]
        single_seg = SingleAddtrace(single_seg, image, mode='add')
    return list


def listAddtrace4(list, image, mode='add'):
    """
    Add temporal traces to a list of neurons (inline implementation).

    Computes temporal traces and centroids for all neurons using inline
    computation without calling separate functions.

    Args:
        list (list): List of neuron dictionaries, each containing 'position' key
        image (np.ndarray): Image stack with shape (T, H, W)
        mode (str, optional): Operation mode. Default is 'add'.
            - 'add': Only add if 'trace' or 'centroid' don't exist
            - 'update': Always update trace and centroid

    Returns:
        list: List of neuron dictionaries with added/updated 'trace' and 'centroid' keys

    Note:
        - Inline implementation for performance
        - Same functionality as SingleAddtrace1 but processes entire list
    """
    # mode update add
    # print('listAddtrace ---> ',len(list))
    for i in range(0, len(list)):
        single_seg = list[i]
        position = single_seg['position']
        if mode=='add':
            if not 'trace' in single_seg or not 'centroid' in single_seg:
                trace = np.zeros((image.shape[0], ))
                centroid = np.zeros((2,))
                for ii in range(0, len(position)):
                    now_position = position[ii]
                    single_trace = image[:, now_position[0], now_position[1]].squeeze()
                    trace = trace+image[:, now_position[0], now_position[1]].squeeze()
                    centroid = now_position+centroid
                centroid = centroid/len(position)
                ave_trace = trace/len(position)
                single_seg['centroid'] = centroid
                single_seg['trace'] = ave_trace
        if mode=='update':
            trace = np.zeros((image.shape[0], ))
            centroid = np.zeros((2,))
            for ii in range(0, len(position)):
                now_position = position[ii]
                single_trace = image[:, now_position[0], now_position[1]].squeeze()
                trace = trace+image[:, now_position[0], now_position[1]].squeeze()
                centroid = now_position+centroid
            centroid = centroid/len(position)
            ave_trace = trace/len(position)
            single_seg['centroid'] = centroid
            single_seg['trace'] = ave_trace
    return list


def listAddtrace(list, image, mode='add', trace_mode='sample'):
    """
    Add temporal traces to a list of neurons with sampling option.

    Computes temporal traces and centroids for all neurons, with option to
    sample a subset of pixels for faster computation.

    Args:
        list (list): List of neuron dictionaries, each containing 'position' key
        image (np.ndarray): Image stack with shape (T, H, W)
        mode (str, optional): Operation mode. Default is 'add'.
            - 'add': Only add if 'trace' or 'centroid' don't exist
            - 'update': Always update trace and centroid
        trace_mode (str, optional): Pixel sampling mode. Default is 'sample'.
            - 'sample': Randomly sample 10 pixels per neuron
            - 'all': Use all pixels in the neuron

    Returns:
        list: List of neuron dictionaries with added/updated 'trace' and 'centroid' keys

    Note:
        - Sampling mode can significantly speed up computation for large neurons
        - Random sampling uses 10 pixels per neuron
        - Trace and centroid computed from sampled/all pixel positions
    """
    # mode update add
    # print('listAddtrace ---> ',len(list))
    for i in range(0, len(list)):
        single_seg = list[i]
        position1 = single_seg['position']
        if trace_mode=='sample':
            position = random.sample(position1,10)
        if trace_mode=='all':
            position = position1
        if mode=='add':
            if not 'trace' in single_seg or not 'centroid' in single_seg:
                trace = np.zeros((image.shape[0], ))
                centroid = np.zeros((2,))
                for ii in range(0, len(position)):
                    now_position = position[ii]
                    single_trace = image[:, now_position[0], now_position[1]].squeeze()
                    trace = trace+image[:, now_position[0], now_position[1]].squeeze()
                    centroid = now_position+centroid
                centroid = centroid/len(position)
                ave_trace = trace/len(position)
                single_seg['centroid'] = centroid
                single_seg['trace'] = ave_trace
        if mode=='update':
            trace = np.zeros((image.shape[0], ))
            centroid = np.zeros((2,))
            for ii in range(0, len(position)):
                now_position = position[ii]
                single_trace = image[:, now_position[0], now_position[1]].squeeze()
                trace = trace+image[:, now_position[0], now_position[1]].squeeze()
                centroid = now_position+centroid
            centroid = centroid/len(position)
            ave_trace = trace/len(position)
            single_seg['centroid'] = centroid
            single_seg['trace'] = ave_trace
    return list



def listAdd_remain_trace(list, image, dict_name='remain_trace', mode='add', trace_mode='sample'):
    """
    Add temporal traces for remaining regions of neurons.

    Computes temporal traces for remaining (non-primary) regions of neurons,
    stored under a custom dictionary key name.

    Args:
        list (list): List of neuron dictionaries, each containing 'remain_position' key
        image (np.ndarray): Image stack with shape (T, H, W)
        dict_name (str, optional): Dictionary key name for storing trace.
            Default is 'remain_trace'.
        mode (str, optional): Operation mode. Default is 'add'.
            - 'add': Only add if dict_name doesn't exist
            - 'update': Always update trace
        trace_mode (str, optional): Pixel sampling mode. Default is 'sample'.
            - 'sample': Randomly sample 10 pixels
            - 'all': Use all pixels in remain_position

    Returns:
        list: List of neuron dictionaries with added/updated dict_name key
            containing temporal trace for remaining regions

    Note:
        - Used for computing traces from remaining (non-primary) neuron regions
        - Similar to listAddtrace but uses 'remain_position' instead of 'position'
        - Trace stored under custom key name (default 'remain_trace')
    """
    # mode update add
    # print('listAddtrace ---> ',len(list))
    # print(list[0])
    # print('listAdd_remain_trace : ',dict_name)
    for i in range(0, len(list)):
        single_seg = list[i]
        position1 = single_seg['remain_position']
        if trace_mode=='sample':
            position = random.sample(position1,10)
        if trace_mode=='all':
            position = position1
        if mode=='add':
            if not dict_name in single_seg:
                trace = np.zeros((image.shape[0], ))
                centroid = np.zeros((2,))
                for ii in range(0, len(position)):
                    now_position = position[ii]
                    single_trace = image[:, now_position[0], now_position[1]].squeeze()
                    trace = trace+image[:, now_position[0], now_position[1]].squeeze()
                    centroid = now_position+centroid
                centroid = centroid/len(position)
                ave_trace = trace/len(position)
                single_seg[dict_name] = ave_trace
        if mode=='update':
            trace = np.zeros((image.shape[0], ))
            centroid = np.zeros((2,))
            for ii in range(0, len(position)):
                now_position = position[ii]
                single_trace = image[:, now_position[0], now_position[1]].squeeze()
                trace = trace+image[:, now_position[0], now_position[1]].squeeze()
                centroid = now_position+centroid
            centroid = centroid/len(position)
            ave_trace = trace/len(position)
            single_seg[dict_name] = ave_trace
    return list




def list_add_mask(list, image):
    """
    Create binary mask stack from list of neuron positions.

    Converts a list of neuron dictionaries into a 3D binary mask stack where
    each slice represents one neuron's spatial mask.

    Args:
        list (list): List of neuron dictionaries, each containing 'position' key
            with list of [row, col] pixel coordinates
        image (np.ndarray): Image stack with shape (T, H, W) for reference dimensions

    Returns:
        np.ndarray: Binary mask stack with shape (N, H, W)
            where N is number of neurons, H and W are spatial dimensions
            Value 1 indicates neuron pixel, 0 indicates background

    Note:
        - Each neuron's mask is stored in a separate slice
        - Mask values are 1 for neuron pixels, 0 for background
        - Spatial dimensions match the last two dimensions of input image
    """
    all_mask = np.zeros((len(list), image.shape[-2], image.shape[-1]))
    for i in range(0, len(list)):
        single_seg = list[i]
        position = single_seg['position']
        mask = np.zeros((image.shape[-2], image.shape[-1]))
        for ii in range(0, len(position)):
            now_position = position[ii]
            mask[now_position[0], now_position[1]]=1
        # single_seg['mask'] = mask
        all_mask[i,:,:] = mask
    return all_mask



def remain_mask(all_mask):
    """
    Compute remaining mask regions after removing overlaps.

    For each neuron mask, computes the region that remains after subtracting
    all other neuron masks. This identifies non-overlapping regions of each neuron.

    Args:
        all_mask (np.ndarray): Binary mask stack with shape (N, H, W)
            where N is number of neurons, H and W are spatial dimensions

    Returns:
        np.ndarray: Remaining mask stack with shape (N, H, W)
            Each slice contains the non-overlapping region of that neuron
            Value 1 indicates remaining region, 0 indicates overlap or background

    Note:
        - For each neuron, subtracts all other neuron masks
        - Result is thresholded to binary (0 or 1)
        - Used to identify unique regions of each neuron
    """
    all_remain_mask = np.zeros((all_mask.shape[-3], all_mask.shape[-2], all_mask.shape[-1]))
    for i in range(0, all_mask.shape[-3]):
        now_mask = all_mask[i,:,:].copy()
        '''
        other_mask = all_mask.copy()
        other_mask[i,:,:] = 0
        other_mask_max = np.max(other_mask, axis=0)
        '''
        other_mask_max = np.zeros((all_mask.shape[-2], all_mask.shape[-1]))
        for ii in range(0, all_mask.shape[-3]):
            if ii!=i:
                other_mask_max = other_mask_max+all_mask[ii,:,:]

        now_mask = now_mask - other_mask_max
        now_mask[now_mask<0] = 0
        now_mask[now_mask>0] = 1
        all_remain_mask[i,:,:] = now_mask
    return all_remain_mask



def remain_mask_test_list(list):
    """
    Test function for computing remaining positions (debugging).

    Computes remaining positions for neurons after removing overlaps from
    other neurons. Used for testing and debugging position overlap detection.

    Args:
        list (list): List of neuron dictionaries, each containing 'position' key

    Returns:
        list: List of [row, col] positions (returns last neuron's position)
            Note: Function appears incomplete, mainly for debugging

    Note:
        - Computes remaining positions by subtracting other neurons' positions
        - Prints debug information about position counts
        - Function implementation may be incomplete
    """
    for i in range(0,  len(list)):
        single_seg = list[i]
        position = single_seg['position']
        # position_l = position.tolist()
        # print('position -----> ',len(position))
        all_other_position_list = []
        for ii in range(0, len(list)):
            single_seg1 = list[ii]
            other_position = single_seg1['position']
            # other_position_l = other_position.tolist()
            if ii==i:
                # print(ii,i,len(all_other_position_list),len(other_position))
                other_position = []
            all_other_position_list = all_other_position_list + other_position
            if ii==i:
                print(ii,i,len(all_other_position_list),len(other_position))
        # print('all_other_position_list -----> ',len(all_other_position_list),len(other_position))
        remain_position_list = [ val for val in position if val not in all_other_position_list]
        
        # remain_position_list = list(set(position)-set(all_other_position_list))
        if len(remain_position_list)<len(position):
            print('remain_position_list -----> ',len(remain_position_list))

    return position


def add_remain_mask_list(list, all_remain_mask):
    """
    Add remaining mask positions to neuron dictionaries.

    Extracts pixel positions from remaining mask regions and adds them to
    each neuron dictionary under the 'remain_position' key.

    Args:
        list (list): List of neuron dictionaries to update
        all_remain_mask (np.ndarray): Remaining mask stack with shape (N, H, W)
            where N matches length of list

    Returns:
        list: List of neuron dictionaries with added 'remain_position' key
            Each 'remain_position' contains array of [row, col] coordinates
            for non-overlapping regions

    Note:
        - Uses np.argwhere to find all positions where remain_mask == 1
        - Adds 'remain_position' key to each neuron dictionary
        - Positions are in [row, col] format
    """
    # print(list[0])
    for i in range(0, len(list)):
        remain_mask = all_remain_mask[i,:,:]
        remain_p =  np.argwhere(remain_mask == 1)
        single_seg = list[i]
        single_seg['remain_position'] = remain_p
        # print('remain_position : ',i,remain_p[0,:])
    # print(list[0])
    return list




def neuron_max_filter(w_g_neuron_list,
                    all_neuron_mask,
                    raw_image_max,
                    max_thres =0.01):
    """
    Filter neurons based on maximum intensity value in their mask region.

    Removes neurons whose maximum intensity value (within their mask region)
    is below a threshold relative to the global maximum intensity.

    Args:
        w_g_neuron_list (list): List of neuron dictionaries
        all_neuron_mask (np.ndarray): Binary mask stack with shape (N, H, W)
            where N matches length of w_g_neuron_list
        raw_image_max (np.ndarray): Maximum projection image with shape (H, W)
            Maximum intensity across time dimension
        max_thres (float, optional): Threshold as fraction of global maximum.
            Default is 0.01 (1% of max intensity).

    Returns:
        list: Filtered list of neuron dictionaries
            Only includes neurons with max_value > max_thres * global_max
            Each neuron dictionary has added 'max_value' key

    Note:
        - Computes max intensity within each neuron's mask region
        - Filters neurons below threshold
        - Adds 'max_value' key to each neuron dictionary
    """
    max_thres_value = np.max(raw_image_max)*max_thres
    new_w_g_neuron_list = []
    for i in range(0, len(w_g_neuron_list)):
        # print('neuron_max_filter -----> ',i)
        single_seg = w_g_neuron_list[i]
        position = single_seg['position']
        # mask = single_seg['mask']
        mask = all_neuron_mask[i, :, :]
        mask_max = mask*raw_image_max
        mask_max_value = np.max(mask_max)
        single_seg['max_value'] = mask_max_value
        if mask_max_value>max_thres_value:
            new_w_g_neuron_list.append(single_seg)
            # print(mask_max_value,' --- ',max_thres_value, ' --- ',np.max(raw_image_max))
    return new_w_g_neuron_list



def delete_edge_neuron(w_g_neuron_list,
                    all_neuron_mask,
                    edge_value=10):
    """
    Remove neurons located too close to image edges.

    Filters out neurons whose centroids are within edge_value pixels of
    image boundaries, as these are likely incomplete or artifacts.

    Args:
        w_g_neuron_list (list): List of neuron dictionaries, each containing
            'centroid' key with [row, col] coordinates
        all_neuron_mask (np.ndarray): Binary mask stack with shape (N, H, W)
            for reference dimensions
        edge_value (int, optional): Edge margin in pixels. Default is 10.

    Returns:
        list: Filtered list of neuron dictionaries
            Only includes neurons with centroids away from edges by at least
            edge_value pixels

    Note:
        - Checks that centroid is at least edge_value pixels from all edges
        - Removes neurons too close to top, bottom, left, or right edges
        - Helps remove incomplete neurons at image boundaries
    """
    new_w_g_neuron_list = []
    size_h = all_neuron_mask.shape[1]
    size_w = all_neuron_mask.shape[2]
    for i in range(0, len(w_g_neuron_list)):
        single_seg = w_g_neuron_list[i]
        position = single_seg['position']
        centroid = single_seg['centroid']

        if centroid[0]>edge_value:
            if centroid[0]<size_h-edge_value:
                if centroid[1]>edge_value:
                    if centroid[1]<size_w-edge_value:
                        new_w_g_neuron_list.append(single_seg)
                        # print(mask_max_value,' --- ',max_thres_value, ' --- ',np.max(raw_image_max))
    return new_w_g_neuron_list



def list2mask(final_mask_list, mask_h, mask_w):
    """
    Convert list of neuron dictionaries to labeled mask images.

    Creates two mask representations: a single combined mask with sequential
    labels, and a 3D stack with each neuron in a separate slice.

    Args:
        final_mask_list (list): List of neuron dictionaries, each containing
            'position' key with list of [row, col] coordinates
        mask_h (int): Height of output masks
        mask_w (int): Width of output masks

    Returns:
        tuple: (final_mask, whole_mask)
            - final_mask (np.ndarray): Combined mask with shape (H, W)
              Each neuron has a unique sequential label (1, 2, 3, ...)
            - whole_mask (np.ndarray): Stack of individual masks with shape (N, H, W)
              Each slice contains one neuron's mask with its label value

    Note:
        - Labels start from 1 (background is 0)
        - Overlapping neurons will have overlapping labels in final_mask
        - Each neuron's mask is stored separately in whole_mask
    """
    final_mask = np.zeros((mask_h, mask_w))
    whole_mask = np.zeros((len(final_mask_list), mask_h, mask_w))
    for i in range(0, len(final_mask_list)):
        single_seg = final_mask_list[i]
        position_list = single_seg['position']
        # print('value_final -----> ',value_final)
        # print('position_list -----> ',len(position_list))
        for ii in range(0, len(position_list)):
            position = position_list[ii]
            final_mask[position[0], position[1]] = i+1
            whole_mask[int(i), position[0], position[1]] = i+1

    return final_mask, whole_mask


def centroid_distance(centroid1, centroid2):
    """
    Calculate Euclidean distance between two centroids.

    Computes the 2D Euclidean distance between two centroid coordinates.

    Args:
        centroid1 (np.ndarray or list): First centroid coordinates [row, col]
        centroid2 (np.ndarray or list): Second centroid coordinates [row, col]

    Returns:
        float: Euclidean distance between the two centroids in pixels

    Note:
        - Formula: sqrt((row1-row2)² + (col1-col2)²)
        - Used for spatial distance calculations between neurons
    """
    a = centroid1[0]-centroid2[0]
    b = centroid1[1]-centroid2[1]
    distance = (a**2+b**2)**0.5
    return distance


def list_union(list1,list2):
    """
    Compute union of two lists (combine unique elements).

    Creates a new list containing all unique elements from both input lists.
    Elements from list1 that are not in list2 are added to the result.

    Args:
        list1 (list): First list
        list2 (list): Second list (base list)

    Returns:
        list: Union of list1 and list2
            Contains all elements from list2 plus elements from list1
            that are not already in list2

    Note:
        - Preserves order: list2 elements first, then list1 elements
        - Does not remove duplicates within each list
        - Only ensures elements from list1 are not duplicated in result
    """
    union = list2
    for i in range(0, len(list1)):
        a = list1[i]
        if not a in list2:
            union.append(a)
    return union


def list_inter(list1,list2):
    """
    Compute intersection of two lists (common elements).

    Creates a new list containing only elements that appear in both input lists.

    Args:
        list1 (list): First list
        list2 (list): Second list

    Returns:
        list: Intersection of list1 and list2
            Contains only elements that are in both lists
            Order follows list1

    Note:
        - Preserves order from list1
        - Only includes elements present in both lists
        - Used for finding common elements between neuron position lists
    """
    inter = []
    for i in range(0, len(list1)):
        a = list1[i]
        # print('a',a)
        if a in list2:
            inter.append(a)
    return inter


def listAddcontours_Laplacian(list, mask_h, mask_w,width=3):
    """
    Add Laplacian-based contours to neuron list.

    Computes contours for each neuron using Laplacian edge detection.
    Creates a mask from neuron positions, applies Laplacian filter, and
    extracts contours from the filtered result.

    Args:
        list (list): List of neuron dictionaries, each containing 'position' key
        mask_h (int): Height of mask
        mask_w (int): Width of mask
        width (int, optional): Width of Laplacian kernel. Default is 3.

    Returns:
        list: List of neuron dictionaries with added 'contours' key
            Each dictionary contains 'contours' with extracted contour data

    Note:
        - Uses 3x3 Laplacian kernel for edge detection
        - Applies convolution with custom kernel
        - Extracts contours from Laplacian-filtered mask
    """
    new_list = []
    print('len(list) -----> ',len(list))
    for aaaaa in range(0, len(list)):
        # print('listAddcontours -----> ',aaaaa)
        new_single_seg = {}
        new_single_seg['contours'] = []
        single_seg = list[aaaaa]
        position = single_seg['position']
        value = aaaaa+1
        trace = single_seg['trace']

        mask_j = np.zeros((mask_h, mask_w), np.uint8)
        for p_i in range(0, len(position)):
            # print(position[ii,:])
            now_position = position[p_i]
            mask_j[now_position[0], now_position[1]] = value

        ker_Laplacian = np.zeros([3,3]) #[[0,1,0],[1,-4,1],[0,1,0]]
        ker_Laplacian[0,0] = 0
        ker_Laplacian[0,1] = 1
        ker_Laplacian[0,2] = 0

        ker_Laplacian[1,0] = 1
        ker_Laplacian[1,1] = -4
        ker_Laplacian[1,2] = 1

        ker_Laplacian[2,0] = 0
        ker_Laplacian[2,1] = 1
        ker_Laplacian[2,2] = 0
        ker_width = np.ones([width,width])
    
        mask_Laplacian = conv(mask_j, ker_Laplacian, stride=1, padding=1)
        mask_Laplacian[mask_Laplacian>0] = value
        mask_Laplacian[mask_Laplacian<0] =0

        mask_Laplacian_width = conv(mask_Laplacian, ker_width, stride=1, padding=1)
        mask_Laplacian_width[mask_Laplacian_width>0] = value
            
        contours_p = np.argwhere(mask_Laplacian_width == value)
        '''
        print('contours_p ----> ',contours_p.shape)
        for ii in range(0, contours_p.shape[0]):
            # c_position_list = list(contours_p[ii,:])
            new_single_seg['contours'].append( list(contours_p[ii,:]))
        '''
        new_single_seg['contours'] = contours_p
        new_single_seg['value'] = aaaaa+1
        new_single_seg['position'] = single_seg['position']
        new_single_seg['trace'] = single_seg['trace']
        new_list.append(new_single_seg)

    return new_list


def listAddcontours_Laplacian_pytorch(list, mask_h, mask_w,width=3):
    new_list = []
    # print('len(list) -----> ',len(list))
    for aaaaa in range(0, len(list)):
        # print('listAddcontours -----> ',aaaaa)
        new_single_seg = {}
        new_single_seg['contours'] = []
        single_seg = list[aaaaa]
        position = single_seg['position']
        value = aaaaa+1
        if 'trace' in single_seg:
            trace = single_seg['trace']

        mask_j = np.zeros((mask_h, mask_w), np.uint8)
        for p_i in range(0, len(position)):
            # print(position[ii,:])
            now_position = position[p_i]
            mask_j[now_position[0], now_position[1]] = value

        ker_Laplacian = np.zeros([3,3]) #[[0,1,0],[1,-4,1],[0,1,0]]
        ker_Laplacian[0,0] = 0
        ker_Laplacian[0,1] = 1
        ker_Laplacian[0,2] = 0

        ker_Laplacian[1,0] = 1
        ker_Laplacian[1,1] = -4
        ker_Laplacian[1,2] = 1

        ker_Laplacian[2,0] = 0
        ker_Laplacian[2,1] = 1
        ker_Laplacian[2,2] = 0
        ker_Laplacian = ker_Laplacian[np.newaxis, np.newaxis, :, :]
        ker_Laplacian = torch.Tensor(ker_Laplacian).cuda()

        ker_width = np.ones([width,width])
        ker_width = ker_width[np.newaxis, np.newaxis, :, :]
        ker_width = torch.Tensor(ker_width).cuda()

        mask_j_tensor = mask_j[np.newaxis, np.newaxis, :, :]
        mask_j_tensor = torch.Tensor(mask_j_tensor).cuda()
        # mask_Laplacian = conv(mask_j, ker_Laplacian, stride=1, padding=1)\
        mask_Laplacian_tensor = torch.nn.functional.conv2d(mask_j_tensor, ker_Laplacian, stride=1, padding=1)
        mask_Laplacian = mask_Laplacian_tensor.cpu().detach().numpy().squeeze()
        # print('mask_Laplacian ',np.max(mask_Laplacian),np.min(mask_Laplacian))
        mask_Laplacian[mask_Laplacian>0] = value
        mask_Laplacian[mask_Laplacian<0] =0

        # mask_Laplacian_width = conv(mask_Laplacian, ker_width, stride=1, padding=1)
        mask_Laplacian_tensor1 = mask_Laplacian[np.newaxis, np.newaxis, :, :]
        mask_Laplacian_tensor1 = torch.Tensor(mask_Laplacian_tensor1).cuda()
        mask_Laplacian_width_tensor = torch.nn.functional.conv2d(mask_Laplacian_tensor1, ker_width, stride=1, padding=1)
        mask_Laplacian_width = mask_Laplacian_width_tensor.cpu().detach().numpy().squeeze()
        # print('mask_Laplacian_width ',np.max(mask_Laplacian_width),np.min(mask_Laplacian_width))
        mask_Laplacian_width[mask_Laplacian_width>0] = value
            
        contours_p = np.argwhere(mask_Laplacian_width == value)
        '''
        print('contours_p ----> ',contours_p.shape)
        for ii in range(0, contours_p.shape[0]):
            # c_position_list = list(contours_p[ii,:])
            new_single_seg['contours'].append( list(contours_p[ii,:]))
        '''
        new_single_seg['mask'] = mask_j
        new_single_seg['contours'] = contours_p
        new_single_seg['value'] = aaaaa+1
        new_single_seg['position'] = single_seg['position']
        # remain_trace
        if 'remain_trace' in single_seg:
            new_single_seg['remain_trace'] = single_seg['remain_trace']
        if 'remain_position' in single_seg:
            new_single_seg['remain_position'] = single_seg['remain_position']
        if 'trace' in single_seg:
            new_single_seg['trace'] = single_seg['trace']
        if 'centroid' in single_seg:
            new_single_seg['centroid'] = single_seg['centroid']
        new_list.append(new_single_seg)

    return new_list


def listAddcontours_Laplacian_pytorch_Merge_Mask(list, mask_h, mask_w,width=3):
    new_list = []
    # print('len(list) -----> ',len(list))
    for aaaaa in range(0, len(list)):
        # print('listAddcontours -----> ',aaaaa)
        new_single_seg = {}
        new_single_seg['contours'] = []
        single_seg = list[aaaaa]
        position = single_seg['position'][0][0]
        value = aaaaa+1
        if 'trace' in single_seg:
            trace = single_seg['trace'][0][0][0]

        mask_j = np.zeros((mask_h, mask_w), np.uint8)
        for p_i in range(0, len(position)):
            # print(position[ii,:])
            now_position = position[p_i]
            mask_j[now_position[0], now_position[1]] = value

        ker_Laplacian = np.zeros([3,3]) #[[0,1,0],[1,-4,1],[0,1,0]]
        ker_Laplacian[0,0] = 0
        ker_Laplacian[0,1] = 1
        ker_Laplacian[0,2] = 0

        ker_Laplacian[1,0] = 1
        ker_Laplacian[1,1] = -4
        ker_Laplacian[1,2] = 1

        ker_Laplacian[2,0] = 0
        ker_Laplacian[2,1] = 1
        ker_Laplacian[2,2] = 0
        ker_Laplacian = ker_Laplacian[np.newaxis, np.newaxis, :, :]
        ker_Laplacian = torch.Tensor(ker_Laplacian).cuda()

        ker_width = np.ones([width,width])
        ker_width = ker_width[np.newaxis, np.newaxis, :, :]
        ker_width = torch.Tensor(ker_width).cuda()

        mask_j_tensor = mask_j[np.newaxis, np.newaxis, :, :]
        mask_j_tensor = torch.Tensor(mask_j_tensor).cuda()
        # mask_Laplacian = conv(mask_j, ker_Laplacian, stride=1, padding=1)\
        mask_Laplacian_tensor = torch.nn.functional.conv2d(mask_j_tensor, ker_Laplacian, stride=1, padding=1)
        mask_Laplacian = mask_Laplacian_tensor.cpu().detach().numpy().squeeze()
        # print('mask_Laplacian ',np.max(mask_Laplacian),np.min(mask_Laplacian))
        mask_Laplacian[mask_Laplacian>0] = value
        mask_Laplacian[mask_Laplacian<0] =0

        # mask_Laplacian_width = conv(mask_Laplacian, ker_width, stride=1, padding=1)
        mask_Laplacian_tensor1 = mask_Laplacian[np.newaxis, np.newaxis, :, :]
        mask_Laplacian_tensor1 = torch.Tensor(mask_Laplacian_tensor1).cuda()
        mask_Laplacian_width_tensor = torch.nn.functional.conv2d(mask_Laplacian_tensor1, ker_width, stride=1, padding=1)
        mask_Laplacian_width = mask_Laplacian_width_tensor.cpu().detach().numpy().squeeze()
        # print('mask_Laplacian_width ',np.max(mask_Laplacian_width),np.min(mask_Laplacian_width))
        mask_Laplacian_width[mask_Laplacian_width>0] = value
            
        contours_p = np.argwhere(mask_Laplacian_width == value)
        '''
        print('contours_p ----> ',contours_p.shape)
        for ii in range(0, contours_p.shape[0]):
            # c_position_list = list(contours_p[ii,:])
            new_single_seg['contours'].append( list(contours_p[ii,:]))
        '''
        new_single_seg['contours'] = contours_p
        new_single_seg['value'] = aaaaa+1
        new_single_seg['position'] = single_seg['position'][0][0]
        if 'trace' in single_seg:
            new_single_seg['trace'] = single_seg['trace'][0][0][0]
        if 'centroid' in single_seg:
            new_single_seg['centroid'] = single_seg['centroid'][0][0][0]
        new_list.append(new_single_seg)

    return new_list



def list2contours(final_mask_list, mask_h, mask_w):
    end_mark = len(final_mask_list)
    final_mask = np.zeros((mask_h, mask_w))
    whole_mask = np.zeros((end_mark, mask_h, mask_w))
    for i in range(0, len(final_mask_list)):
        single_seg = final_mask_list[i]
        contours = single_seg['contours']
        value_final = i+1
        for ii in range(0, contours.shape[0]):
            position = contours[ii,:]
            # print('position -----> ',position)
            final_mask[int(position[0]), int(position[1])] = value_final
            whole_mask[i, int(position[0]), int(position[1])] = value_final

    return final_mask , whole_mask



def list2masks(final_mask_list, mask_h, mask_w):
    end_mark = len(final_mask_list)
    final_mask = np.zeros((mask_h, mask_w))
    whole_mask = np.zeros((end_mark, mask_h, mask_w))
    for i in range(0, len(final_mask_list)):
        single_seg = final_mask_list[i]
        mask = single_seg['mask']
        value_final = i+1
        # print('value_final -----> ',value_final)
        mask = mask*value_final
        whole_mask[i,:,:] = mask
        final_mask = final_mask+mask

    return final_mask , whole_mask






def conv(img, ker, stride=1, padding=0):
    size = list(img.shape)
    pad_img = np.zeros([size[0] + 2 * padding, size[1] + 2 * padding])
    pad_img[padding:-padding, padding:-padding] = img
    # print('pad_img -----> ', pad_img.shape)
    img = pad_img
    out_size0 = (img.shape[0] - ker.shape[0]) // stride + 1
    out_size1 = (img.shape[1] - ker.shape[1]) // stride + 1
    res = np.zeros([out_size0, out_size1])

    for hi in range(0, out_size0 * stride, stride):
        for wi in range(0, out_size1 * stride, stride):
            region = img[hi:hi + ker.shape[0], wi:wi + ker.shape[0]]
            res[hi // stride, wi // stride] = np.sum(region * ker[:, :])
    return res


def nmf_defined(V,R,K,W,if_W_fixed=True):
    V[V==0] = 0.001
    W[W==0] = 0.001
    [m, n] = V.shape
    H = np.ones((R,n))
    for i in range(0,K):
        WH_W = np.dot(W.T, W)
        WH_W_H = np.dot(WH_W, H)
        WH_V = np.dot(W.T, V)
        H = H*WH_V/WH_W_H
    return H


def group_mask(mask_list, img):
    mask_j = np.zeros((img.shape[1],img.shape[2]), np.uint8)
    for i in range(0, len(mask_list)):
        now_neuron = mask_list[i]
        now_neuron_position = now_neuron['position']
        # print('now_neuron_position ---> ',len(now_neuron_position))
        # if len(now_neuron_position)==97:
            # print('now_neuron_position ---> ',now_neuron_position)
        for p_i in range(0, len(now_neuron_position)):
            # now_position = now_neuron_position[p_i,:]
            mask_j[now_neuron_position[p_i][0], now_neuron_position[p_i][1]] = 255
        mask_jRGB = cv2.cvtColor(mask_j, cv2.COLOR_GRAY2BGR)

    cc_mask_j = New_Two_Pass(mask_j, 'NEIGHBOR_HOODS_8')
    max_cc_mask_j = int(np.max(cc_mask_j))
    min_cc_mask_j = int(np.min(cc_mask_j))

    group_neuron_list = []
    for i in range(min_cc_mask_j+1, max_cc_mask_j+1):
        # print('i -----> ',i)
        position = np.argwhere(cc_mask_j == i)
        if position.shape[0]>0:
            single_neuron = {}
            single_neuron['position'] = []
            for ii in range(0, position.shape[0]):
                position_list = list(position[ii,:])
                single_neuron['position'].append(position_list)
            # print('single_neuron[''] ---> ',len(single_neuron['position']))
            group_neuron_list.append(single_neuron)

    arranged_index = []
    for i in range(0, len(group_neuron_list)):
        group_neuron = group_neuron_list[i]
        group_neuron_position = group_neuron['position']
        coor_patch_list = []
        for ii in range(0, len(mask_list)):
            now_neuron1 = mask_list[ii]
            now_neuron_position1 = now_neuron1['position']
            list_inter_len = len(list_inter(now_neuron_position1, group_neuron_position))
            # print(len(now_neuron_position1),' --> ',len(group_neuron_position)) now_neuron_position1 
            # if len(now_neuron_position1)<100 & len(group_neuron_position)<100:
            '''
            if len(now_neuron_position1)==len(group_neuron_position):
                print(len(now_neuron_position1),' len(now_neuron_position1) --> ',)
                print(len(group_neuron_position),' len(group_neuron_position) --> ',)
                print('list_inter --> ',list_inter(now_neuron_position1, group_neuron_position))
                # print('list_inter group_neuron_position --> ',list_inter(group_neuron_position, group_neuron_position))
                # print('list_inter now_neuron_position1 --> ',list_inter(now_neuron_position1, now_neuron_position1))
            '''
            if list_inter_len>0:
                # print(i,' --> ',ii,' --> ',list_inter_len)
                coor_patch_list.append(ii)
        # print(i,' --> ',len(group_neuron_list),len(mask_list),'coor_patch_list --> ',coor_patch_list,len(group_neuron_position))
        arranged_index.append(coor_patch_list)
    return group_neuron_list, arranged_index, cc_mask_j


def calculate_trace(img, group_neuron_list, mask_list, arranged_index):
    new_mask_list = []
    for i in range(0, len(group_neuron_list)):
        sub_index = arranged_index[i]
        group_neuron = group_neuron_list[i]

        # print(i,' len(sub_index) -----> ',len(sub_index))
        new_single_neuron = {}
        if len(sub_index)==1:
            mask_index = sub_index[0]
            single_mask = mask_list[mask_index]
            new_single_neuron['true_trace'] = single_mask['trace']
            new_single_neuron['trace'] = single_mask['trace']
            new_single_neuron['position'] = single_mask['position']
            new_single_neuron['split'] = single_mask['split']  
            new_single_neuron['centroid'] = single_mask['centroid']
            new_mask_list.append(new_single_neuron)
        if len(sub_index)>1:
            group_neuron_position = group_neuron['position']
            len_g_p = len(group_neuron_position)
            mask_matrix = np.zeros((img.shape[0], len_g_p))
            for ii in range(0, len_g_p):
                mask_matrix[:,ii] = img[:, group_neuron_position[ii][0], group_neuron_position[ii][1]]

            mask_matrix = mask_matrix.T
            W = np.zeros((len(sub_index), len_g_p))
            for ii in range(0, len(sub_index)):
                sub_mask = mask_list[sub_index[ii]]
                sub_mask_postion = sub_mask['position']
                for iii in range(0, len_g_p):
                    group_neuron_position_p = group_neuron_position[iii]
                    if group_neuron_position_p in sub_mask_postion:
                        W[ii, iii] = 1
            W = W.T
            H = nmf_defined(mask_matrix,len(sub_index),100,W)

            for ii in range(0, len(sub_index)):
                single_mask = mask_list[sub_index[ii]]
                # sub_mask['true_trace'] = H[ii,:]
                new_single_neuron['true_trace'] = H[ii,:]
                new_single_neuron['trace'] = single_mask['trace']
                new_single_neuron['position'] = single_mask['position']
                new_single_neuron['split'] = single_mask['split']  
                new_single_neuron['centroid'] = single_mask['centroid']
                new_mask_list.append(new_single_neuron)
    return new_mask_list


def Joint_Mask_List_Simple1(mask_list1, mask_list2, corr_mark, area_mark=0.9, active_rate=0.02, if_coor=True, if_area=True, if_merge=True):
    w_mask_list = mask_list1+mask_list2

    arranged_index = []
    processed_index = []
    for ii in range(0, len(w_mask_list)):
        # if ii not in processed_index:
        # processed_index.append(ii)
        now_neuron = w_mask_list[ii]
        now_neuron_trace = now_neuron['trace']
        now_neuron_position = now_neuron['position']
        now_neuron_centroid = now_neuron['centroid']
        # print(' ii -----> ',str(ii))
        if_coor_neuron = 0
        coor_patch_list = []
        if_close_neuron = 0
        posi_patch_list = []
        max_pccs = 0

        for iii in range(0, len(w_mask_list)):
            # if iii not in processed_index:
            # processed_index.append(iii)
            old_neuron = w_mask_list[iii]
            old_neuron_trace = old_neuron['trace']
            # print('old_neuron_trace -----> ',old_neuron_trace.shape)
            old_neuron_position = old_neuron['position']
            old_neuron_centroid = old_neuron['centroid']

            distance = centroid_distance(old_neuron_centroid, now_neuron_centroid)
            if distance<20 and distance>1:
                list_inter_len = len(list_inter(now_neuron_position, old_neuron_position))
                now_neuron_position_len = len(now_neuron_position)
                old_neuron_position_len = len(old_neuron_position)
                if (list_inter_len/old_neuron_position_len)>0.4 or (list_inter_len/now_neuron_position_len)>0.4:
                    if_close_neuron = 1

                if if_coor:
                    pccs1 = cal_pccs(old_neuron_trace, now_neuron_trace, now_neuron_trace.shape[0])
                    # print('pccs1 ----- ',pccs1)
                    if pccs1>corr_mark:
                        if_coor_neuron = 1
                        coor_patch_list.append(iii)
                        # processed_index.append(iii)
                        # print('coor_patch_list -----> ',len(coor_patch_list))
                if if_area:
                    list_inter_len = len(list_inter(now_neuron_position, old_neuron_position))
                    now_neuron_position_len = len(now_neuron_position)
                    old_neuron_position_len = len(old_neuron_position)
                    if (list_inter_len/old_neuron_position_len)>area_mark or (list_inter_len/now_neuron_position_len)>area_mark:
                        if_coor_neuron = 1
                        coor_patch_list.append(iii)
                        # processed_index.append(iii)
                        # print('overlap')

        if if_coor_neuron ==0:
            mask_list = []
            mask_list.append(ii)
            # mask_cell['list'] = mask_list
            # mask_cell['close'] = if_close_neuron
            arranged_index.append(mask_list)
        # if (if_close_neuron==1)&(if_coor_neuron==1):
        if if_coor_neuron==1:
            mask_list = coor_patch_list
            mask_list.append(ii)
            arranged_index.append(mask_list)
            # print('coor_patch_list -----> ',len(coor_patch_list))
        '''
        if if_close_neuron==0:
            print('if_close_neuron -----> ',if_close_neuron)
        '''
    final_mask_list = []
    max_sub_len = 0
    for i in range(0,len(arranged_index)):
        sub_list = arranged_index[i]
        if len(sub_list)>max_sub_len:
            max_sub_len = len(sub_list)

    for i in range(0,len(arranged_index)):
        sub_list = arranged_index[i]
        if not if_merge:
            # print('Not Merge')
            # print('len(sub_list) -----> ',len(sub_list))
            if len(sub_list)>max_sub_len*active_rate or if_close_neuron==0:
                max_index = 0
                max_len = 0
                for ii in range(0,len(sub_list)):
                    sub_mask = w_mask_list[sub_list[ii]]
                    sub_mask_position = sub_mask['position']
                    sub_mask_position_len = len(sub_mask_position)
                    if sub_mask_position_len>max_len:
                        max_len = sub_mask_position_len
                        max_index = ii

                final_mask = w_mask_list[sub_list[max_index]]
                final_mask_list.append(final_mask)
        if if_merge:
            # print('Merge')
            final_mask = w_mask_list[sub_list[0]]
            for ii in range(1,len(sub_list)):
                add_mask = w_mask_list[sub_list[ii]]
                final_mask_position = final_mask['position']
                add_mask_position = add_mask['position']
                final_mask_position = list_union(final_mask_position, add_mask_position)
                final_mask['position'] = final_mask_position
            final_mask_list.append(final_mask)

    return final_mask_list


def Joint_Mask_List_Mul(mask_list, corr_mark, area_mark=0.9, active_rate=0.02, if_coor=True, if_area=True, if_merge=True):
    group_size = 500
    group_num = math.ceil(len(mask_list)/group_size)
    f_mask_list=[]

    for i in range(0, group_num):
        init = i*group_size
        if i<(group_num-1):
            end = init+group_size
        if i==(group_num-1):
            end = len(mask_list)
        # print(len(mask_list),' init ---> ',init,'end ---> ',end)
        sub_mask_list = mask_list[init:end]
        f_mask_list = Joint_Mask_List_Simple(f_mask_list, sub_mask_list, corr_mark, area_mark, active_rate, if_coor, if_area, if_merge)
        # print('f_mask_list ---> ',len(f_mask_list))
    return f_mask_list


def Joint_Mask_List_Simple(mask_list1, mask_list2, corr_mark, area_mark=0.9, active_rate=0.02, if_coor=True, if_area=True, if_merge=True):
    w_mask_list = mask_list1+mask_list2

    arranged_index = []
    processed_index = []
    for ii in range(0, len(w_mask_list)):
        if ii not in processed_index:
            processed_index.append(ii)
            now_neuron = w_mask_list[ii]
            now_neuron_trace = now_neuron['trace']
            now_neuron_position = now_neuron['position']
            now_neuron_centroid = now_neuron['centroid']
            # print(' ii -----> ',str(ii))
            if_coor_neuron = 0
            coor_patch_list = []
            if_close_neuron = 0
            posi_patch_list = []
            max_pccs = 0

            for iii in range(0, len(w_mask_list)):
                if iii not in processed_index:
                    # processed_index.append(iii)
                    old_neuron = w_mask_list[iii]
                    old_neuron_trace = old_neuron['trace']
                    # print('old_neuron_trace -----> ',old_neuron_trace.shape)
                    old_neuron_position = old_neuron['position']
                    old_neuron_centroid = old_neuron['centroid']

                    distance = centroid_distance(old_neuron_centroid, now_neuron_centroid)
                    if distance<20:
                        list_inter_len = len(list_inter(now_neuron_position, old_neuron_position))
                        now_neuron_position_len = len(now_neuron_position)
                        old_neuron_position_len = len(old_neuron_position)
                        if (list_inter_len/old_neuron_position_len)>0.4 or (list_inter_len/now_neuron_position_len)>0.4:
                            if_close_neuron = 1

                        if if_coor:
                            pccs1 = cal_pccs(old_neuron_trace, now_neuron_trace, now_neuron_trace.shape[0])
                            # print('pccs1 ----- ',pccs1)
                            if pccs1>corr_mark:
                                if_coor_neuron = 1
                                coor_patch_list.append(iii)
                                processed_index.append(iii)
                                # print('coor_patch_list -----> ',len(coor_patch_list))
                        if if_area:
                            list_inter_len = len(list_inter(now_neuron_position, old_neuron_position))
                            now_neuron_position_len = len(now_neuron_position)
                            old_neuron_position_len = len(old_neuron_position)
                            if (list_inter_len/old_neuron_position_len)>area_mark or (list_inter_len/now_neuron_position_len)>area_mark:
                                if_coor_neuron = 1
                                coor_patch_list.append(iii)
                                processed_index.append(iii)
                                # print('overlap')

            if if_coor_neuron ==0:
                mask_list = []
                mask_list.append(ii)
                # mask_cell['list'] = mask_list
                # mask_cell['close'] = if_close_neuron
                arranged_index.append(mask_list)
            # if (if_close_neuron==1)&(if_coor_neuron==1):
            if if_coor_neuron==1:
                mask_list = coor_patch_list
                mask_list.append(ii)
                arranged_index.append(mask_list)
                # print('coor_patch_list -----> ',len(coor_patch_list))
            '''
            if if_close_neuron==0:
                print('if_close_neuron -----> ',if_close_neuron)
            '''
    final_mask_list = []
    max_sub_len = 0
    for i in range(0,len(arranged_index)):
        sub_list = arranged_index[i]
        if len(sub_list)>max_sub_len:
            max_sub_len = len(sub_list)

    for i in range(0,len(arranged_index)):
        sub_list = arranged_index[i]
        if not if_merge:
            # print('Not Merge')
            # print('len(sub_list) -----> ',len(sub_list))
            if len(sub_list)>max_sub_len*active_rate or if_close_neuron==0:
                max_index = 0
                max_len = 0
                for ii in range(0,len(sub_list)):
                    sub_mask = w_mask_list[sub_list[ii]]
                    sub_mask_position = sub_mask['position']
                    sub_mask_position_len = len(sub_mask_position)
                    if sub_mask_position_len>max_len:
                        max_len = sub_mask_position_len
                        max_index = ii

                final_mask = w_mask_list[sub_list[max_index]]
                final_mask_list.append(final_mask)
        if if_merge:
            # print('Merge')
            final_mask = w_mask_list[sub_list[0]]
            for ii in range(1,len(sub_list)):
                add_mask = w_mask_list[sub_list[ii]]
                final_mask_position = final_mask['position']
                add_mask_position = add_mask['position']
                final_mask_position = list_union(final_mask_position, add_mask_position)
                final_mask['position'] = final_mask_position
            final_mask_list.append(final_mask)

    return final_mask_list


def Joint_Mask_List_Simple_Merge_Mask(mask_list1, mask_list2, corr_mark, area_mark=0.9, active_rate=0.02, if_coor=True, if_area=True, if_merge=True):
    w_mask_list = mask_list1+mask_list2

    arranged_index = []
    processed_index = []
    for ii in range(0, len(w_mask_list)):
        if ii not in processed_index:
            processed_index.append(ii)
            now_neuron = w_mask_list[ii]
            now_neuron_trace = now_neuron['trace'][0][0][0]
            now_neuron_position = now_neuron['position'][0][0]
            now_neuron_centroid = now_neuron['centroid'][0][0][0]
            # print(' ii -----> ',str(ii))
            if_coor_neuron = 0
            coor_patch_list = []
            if_close_neuron = 0
            posi_patch_list = []
            max_pccs = 0

            for iii in range(0, len(w_mask_list)):
                if iii not in processed_index:
                    # processed_index.append(iii)
                    old_neuron = w_mask_list[iii]
                    old_neuron_trace = old_neuron['trace'][0][0][0]
                    # print('old_neuron_trace -----> ',old_neuron_trace.shape)
                    old_neuron_position = old_neuron['position'][0][0]
                    old_neuron_centroid = old_neuron['centroid'][0][0][0]

                    distance = centroid_distance(old_neuron_centroid, now_neuron_centroid)
                    if distance<20:
                        list_inter_len = len(list_inter(now_neuron_position, old_neuron_position))
                        now_neuron_position_len = len(now_neuron_position)
                        old_neuron_position_len = len(old_neuron_position)
                        if (list_inter_len/old_neuron_position_len)>0.4 or (list_inter_len/now_neuron_position_len)>0.4:
                            if_close_neuron = 1

                        if if_coor:
                            pccs1 = cal_pccs(old_neuron_trace, now_neuron_trace, now_neuron_trace.shape[0])
                            # print('pccs1 ----- ',pccs1)
                            if pccs1>corr_mark:
                                if_coor_neuron = 1
                                coor_patch_list.append(iii)
                                processed_index.append(iii)
                                # print('coor_patch_list -----> ',len(coor_patch_list))
                        if if_area:
                            list_inter_len = len(list_inter(now_neuron_position, old_neuron_position))
                            now_neuron_position_len = len(now_neuron_position)
                            old_neuron_position_len = len(old_neuron_position)
                            if (list_inter_len/old_neuron_position_len)>area_mark or (list_inter_len/now_neuron_position_len)>area_mark:
                                if_coor_neuron = 1
                                coor_patch_list.append(iii)
                                processed_index.append(iii)
                                # print('overlap')

            if if_coor_neuron ==0:
                mask_list = []
                mask_list.append(ii)
                # mask_cell['list'] = mask_list
                # mask_cell['close'] = if_close_neuron
                arranged_index.append(mask_list)
            # if (if_close_neuron==1)&(if_coor_neuron==1):
            if if_coor_neuron==1:
                mask_list = coor_patch_list
                mask_list.append(ii)
                arranged_index.append(mask_list)
                # print('coor_patch_list -----> ',len(coor_patch_list))
            '''
            if if_close_neuron==0:
                print('if_close_neuron -----> ',if_close_neuron)
            '''
    final_mask_list = []
    max_sub_len = 0
    for i in range(0,len(arranged_index)):
        sub_list = arranged_index[i]
        if len(sub_list)>max_sub_len:
            max_sub_len = len(sub_list)

    for i in range(0,len(arranged_index)):
        sub_list = arranged_index[i]
        if not if_merge:
            # print('Not Merge')
            # print('len(sub_list) -----> ',len(sub_list))
            if len(sub_list)>max_sub_len*active_rate or if_close_neuron==0:
                max_index = 0
                max_len = 0
                for ii in range(0,len(sub_list)):
                    sub_mask = w_mask_list[sub_list[ii]]
                    sub_mask_position = sub_mask['position']
                    sub_mask_position_len = len(sub_mask_position)
                    if sub_mask_position_len>max_len:
                        max_len = sub_mask_position_len
                        max_index = ii

                final_mask = w_mask_list[sub_list[max_index]]
                final_mask_list.append(final_mask)
        if if_merge:
            # print('Merge')
            final_mask = w_mask_list[sub_list[0]]
            for ii in range(1,len(sub_list)):
                add_mask = w_mask_list[sub_list[ii]]
                final_mask_position = final_mask['position'][0][0]
                add_mask_position = add_mask['position'][0][0]
                final_mask_position = list_union(final_mask_position, add_mask_position)
                final_mask['position'] = final_mask_position
            final_mask_list.append(final_mask)

    return final_mask_list


def clear_neuron(mask_list1, mask_list2, area_mark=0.4, area_size=300):
    w_mask_list = mask_list1+mask_list2

    final_mask_list = []
    arranged_index = []
    processed_index = []
    for ii in range(0, len(w_mask_list)):
        now_neuron = w_mask_list[ii]
        now_neuron_trace = now_neuron['trace']
        now_neuron_position = now_neuron['position']
        now_neuron_centroid = now_neuron['centroid']
        # print(' ii -----> ',str(ii))
        if_coor_neuron = 0
        coor_patch_list = []
        if_close_neuron = 0
        if_close_small_neuron = 0
        posi_patch_list = []
        max_pccs = 0

        for iii in range(0, len(w_mask_list)):
            if iii!=ii:
                old_neuron = w_mask_list[iii]
                old_neuron_trace = old_neuron['trace']
                # print('old_neuron_trace -----> ',old_neuron_trace.shape)
                old_neuron_position = old_neuron['position']
                old_neuron_centroid = old_neuron['centroid']

                distance = centroid_distance(old_neuron_centroid, now_neuron_centroid)
                if distance<20:
                    list_inter_len = len(list_inter(now_neuron_position, old_neuron_position))
                    now_neuron_position_len = len(now_neuron_position)
                    old_neuron_position_len = len(old_neuron_position)
                    if (list_inter_len/old_neuron_position_len)>area_mark or (list_inter_len/now_neuron_position_len)>area_mark:
                        if_close_neuron = 1
                        if now_neuron_position_len<area_size:
                            if_close_small_neuron = 1
                            # print('if_close_small_neuron ---> ',if_close_small_neuron)

        if if_close_small_neuron ==0:
            final_mask_list.append(now_neuron)

    return final_mask_list


def correct_position(position, init_h, init_w):
    cor_position = []
    for i in range(0, len(position)):
        now_position = position[i].copy()
        # print('now_position ',now_position)
        # cor_now_position = np.zeros(now_position.shape)
        now_position[0] = now_position[0]+init_h
        now_position[1] = now_position[1]+init_w
        cor_position.append(now_position)
    return cor_position


def correct_contours(contours, init_h, init_w):
    cor_contours = np.zeros(contours.shape)
    cor_contours[:,0] = contours[:,0]+init_h
    cor_contours[:,1] = contours[:,1]+init_w
    return cor_contours

def correct_centroid(centroid, init_h, init_w):
    cor_centroid = centroid.copy()
    cor_centroid[0] = centroid[0]+init_h
    cor_centroid[1] = centroid[1]+init_w
    return cor_centroid

'''
def delete_edge_neuron(sub_neuron_list, img_h, img_w):
    delete_index=[]
    for i in range(0, len(sub_neuron_list)):
        pass
'''


def joint_neuron(whole_neuron_list, sub_neuron_list, init_h, init_w):
    corr_mark = 0.9
    if len(whole_neuron_list)>1:
        list_len = len(sub_neuron_list)

        for i in range(0, len(sub_neuron_list)):
            now_neuron = sub_neuron_list[i]
            now_neuron_trace = now_neuron['trace']
            now_neuron_position = now_neuron['position']
            now_neuron_centroid = now_neuron['centroid']
            now_neuron_centroid = correct_centroid(now_neuron_centroid, init_h, init_w)

            if_coor_neuron = 0
            coor_neuron_list = []
            if_close_neuron = 0
            posi_neuron_list = []
            for ii in range(0, len(whole_neuron_list)):
                old_neuron = whole_neuron_list[ii]
                old_neuron_centroid = old_neuron['centroid']
                old_neuron_position = old_neuron['position']
                old_neuron_trace = old_neuron['trace']

                distance = centroid_distance(old_neuron_centroid, now_neuron_centroid)
                
                if distance<20:
                    if_close_neuron = 1
                    pccs1 = cal_pccs(old_neuron_trace, now_neuron_trace, now_neuron_trace.shape[0])
                    if pccs1>corr_mark:
                        print('distance -----> ',distance,' pccs1 -----> ',pccs1)
                        if_coor_neuron = 1
                        coor_neuron_list.append(ii)

            if if_coor_neuron ==0:
                # print('Add Neuron =====> ',i,' ==> ',ii,' ==> ',Add_num)
                new_single_neuron = {}
                now_neuron_position = correct_position(now_neuron_position, init_h, init_w)
                new_single_neuron['position'] = now_neuron_position
                new_single_neuron['trace'] = now_neuron_trace
                now_neuron_centroid = correct_centroid(now_neuron_centroid, init_h, init_w)
                new_single_neuron['centroid'] = now_neuron_centroid
                whole_neuron_list.append(new_single_neuron)

            if (if_close_neuron==1)&(if_coor_neuron==1):
                
                # print('Merge Neuron =====> ',i,' ==> ',ii)
                same_neuron_index = coor_neuron_list[0]
                same_neuron = whole_neuron_list[same_neuron_index]
                same_neuron_position = same_neuron['position']
                now_neuron_position = correct_position(now_neuron_position, init_h, init_w)
                # print('same_neuron_position -----> ',len(same_neuron_position),' now_neuron_position -----> ',len(now_neuron_position))
                # print('same_neuron_position -----> ',same_neuron_position,' now_neuron_position -----> ',now_neuron_position)
                same_neuron_position = list_union(same_neuron_position, now_neuron_position) #list(set(same_patch_position)|set(now_patch_position))
                # print('same_neuron_position -----> ',len(same_neuron_position))

                same_neuron['position'] = same_neuron_position
                whole_neuron_list[same_neuron_index] = same_neuron
                '''
                for iii in range(0, len(coor_neuron_list)):
                    same_neuron_index = coor_neuron_list[iii]
                    same_neuron = whole_neuron_list[same_neuron_index]
                    same_neuron_position = same_neuron['position']
                    now_neuron_position = correct_position(now_neuron_position, init_h, init_w)
                    same_neuron_position = list_union(same_neuron_position, now_neuron_position)
                '''
                

    if len(whole_neuron_list)<1:
        for i in range(0, len(sub_neuron_list)):
            now_neuron = sub_neuron_list[i]
            now_neuron_trace = now_neuron['trace']
            now_neuron_centroid = now_neuron['centroid']
            now_neuron_position = now_neuron['position']

            new_single_neuron = {}
            new_single_neuron['value'] = i+1

            now_neuron_position = correct_position(now_neuron_position, init_h, init_w)
            new_single_neuron['position'] = now_neuron_position
            new_single_neuron['trace'] = now_neuron_trace
            now_neuron_centroid = correct_centroid(now_neuron_centroid, init_h, init_w)
            new_single_neuron['centroid'] = now_neuron_centroid
            whole_neuron_list.append(new_single_neuron)
    return whole_neuron_list



def joint_neuron2(whole_mask_list, sub_mask_list, init_h, init_w):
    corr_mark = 0.1
    if len(whole_mask_list)>1:
        list_len = len(sub_mask_list)
        for i in range(0, len(sub_mask_list)):
            now_patch = sub_mask_list[i]
            now_patch_trace = now_patch['trace']
            now_patch_position = now_patch['position']
            now_patch_centroid = now_patch['centroid']
            now_patch_centroid[0] = now_patch_centroid[0]+init_h
            now_patch_centroid[1] = now_patch_centroid[1]+init_w

            if_coor_neuron = 0
            coor_patch_list = []
            if_close_neuron = 0
            posi_patch_list = []
            for ii in range(0, len(whole_mask_list)):
                old_patch = whole_mask_list[ii]
                old_patch_centroid = old_patch['centroid']
                old_patch_position = old_patch['position']
                old_patch_trace = old_patch['trace']

                distance = centroid_distance(old_patch_centroid, now_patch_centroid)
                if distance<20:
                    if_close_neuron = 1
                    pccs1 = cal_pccs(old_patch_trace, now_patch_trace, now_patch_trace.shape[0])
                    if pccs1>corr_mark:
                        if_coor_neuron = 1
                        coor_patch_list.append(ii)

            if if_coor_neuron ==0:
                # print('Add Neuron =====> ',i,' ==> ',ii,' ==> ',Add_num)
                new_single_patch = {}
                now_patch_position = correct_position(now_patch_position, init_h, init_w)
                new_single_patch['position'] = now_patch_position
                new_single_patch['trace'] = now_patch_trace
                now_patch_centroid = correct_centroid(now_patch_centroid, init_h, init_w)
                new_single_patch['centroid'] = now_patch_centroid
                whole_mask_list.append(new_single_patch)

            if (if_close_neuron==1)&(if_coor_neuron==1):
                # print('Merge Neuron =====> ',i,' ==> ',ii)
                same_patch_index = coor_patch_list[0]
                same_patch = whole_mask_list[same_patch_index]
                same_patch_position = same_patch['position']
                now_patch_position = correct_position(now_patch_position, init_h, init_w)
                same_patch_position = list_union(same_patch_position, now_patch_position) #list(set(same_patch_position)|set(now_patch_position))

                same_patch['position'] = same_patch_position
                whole_mask_list[same_patch_index] = same_patch

    if len(whole_mask_list)<1:
        for i in range(0, len(sub_mask_list)):
            now_patch = sub_mask_list[i]
            now_patch_trace = now_patch['trace']
            now_patch_centroid = now_patch['centroid']
            now_patch_position = now_patch['position']

            new_single_seg = {}
            new_single_seg['value'] = i+1

            now_patch_position = correct_position(now_patch_position, init_h, init_w)
            new_single_seg['position'] = now_patch_position
            new_single_seg['trace'] = now_patch_trace
            now_patch_centroid = correct_centroid(now_patch_centroid, init_h, init_w)
            new_single_seg['centroid'] = now_patch_centroid
            whole_mask_list.append(new_single_seg)
    return whole_mask_list

