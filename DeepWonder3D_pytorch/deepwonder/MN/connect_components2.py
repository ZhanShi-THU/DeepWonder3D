import cv2
import numpy as np
import scipy.io as scio

def neighbor_label_list(i, j, label_matrix, binary_img, neighbor_hoods):
    """
    Get list of unique neighbor labels for a given pixel position.

    Examines neighboring pixels (4- or 8-connectivity) and collects all unique
    non-zero labels from foreground neighbors.

    Args:
        i (int): Row coordinate of the pixel
        j (int): Column coordinate of the pixel
        label_matrix (np.ndarray): Label matrix with shape (H, W)
            Contains current label assignments for all pixels
        binary_img (np.ndarray): Binary image with shape (H, W)
            Pixels != 0 are considered foreground
        neighbor_hoods (str): Connectivity type, either 'NEIGHBOR_HOODS_4' for
            4-connectivity or 'NEIGHBOR_HOODS_8' for 8-connectivity

    Returns:
        list: List of unique neighbor label values (integers)
            Only includes labels from foreground neighbors (pixel != 0)
            Empty list if no foreground neighbors found

    Note:
        - For 4-connectivity: checks up, down, left, right neighbors
        - For 8-connectivity: includes diagonal neighbors as well
        - Only includes labels from foreground pixels (binary_img != 0)
        - Duplicate labels are automatically removed
    """
    nb_label_list=[]

    if neighbor_hoods == 'NEIGHBOR_HOODS_4':
        if i > 0:
            label_up = label_matrix[i-1, j].copy()
            pixel_up = binary_img[i-1, j].copy()
            if pixel_up!=0:
                if label_up not in nb_label_list:
                    nb_label_list.append(label_up)
        if i < label_matrix.shape[0]-1:
            label_down = label_matrix[i+1, j].copy()
            pixel_down = binary_img[i+1, j].copy()
            if pixel_down!=0:
                if label_down not in nb_label_list:
                    nb_label_list.append(label_down)
        if j > 0:
            label_left = label_matrix[i, j-1].copy()
            pixel_left = binary_img[i, j-1].copy()
            if pixel_left!=0:
                if label_left not in nb_label_list:
                    nb_label_list.append(label_left)
        if j < label_matrix.shape[1]-1:
            label_right = label_matrix[i, j+1].copy()
            pixel_right = binary_img[i, j+1].copy()
            if pixel_right!=0:
                if label_right not in nb_label_list:
                    nb_label_list.append(label_right)

    if neighbor_hoods == 'NEIGHBOR_HOODS_8':
        if i > 0:
            label_up = label_matrix[i-1, j].copy()
            pixel_up = binary_img[i-1, j].copy()
            if pixel_up!=0:
                if label_up not in nb_label_list:
                    nb_label_list.append(label_up)
        if i < label_matrix.shape[0]-1:
            label_down = label_matrix[i+1, j].copy()
            pixel_down = binary_img[i+1, j].copy()
            if pixel_down!=0:
                if label_down not in nb_label_list:
                    nb_label_list.append(label_down)
        if j > 0:
            label_left = label_matrix[i, j-1].copy()
            pixel_left = binary_img[i, j-1].copy()
            if pixel_left!=0:
                if label_left not in nb_label_list:
                    nb_label_list.append(label_left)
        if j < label_matrix.shape[1]-1:
            label_right = label_matrix[i, j+1].copy()
            pixel_right = binary_img[i, j+1].copy()
            if pixel_right!=0:
                if label_right not in nb_label_list:
                    nb_label_list.append(label_right)

        if (i > 0) and (j > 0):
            label_up_left = label_matrix[i-1, j-1].copy()
            pixel_up_left = binary_img[i-1, j-1].copy()
            if pixel_up_left!=0:
                if label_up_left not in nb_label_list:
                    nb_label_list.append(label_up_left)
        if (i > 0) and (j < label_matrix.shape[1]-1):
            label_up_right = label_matrix[i-1, j+1].copy()
            pixel_up_right = binary_img[i-1, j+1].copy()
            if pixel_up_right!=0:
                if label_up_right not in nb_label_list:
                    nb_label_list.append(label_up_right)
        if (i < label_matrix.shape[0]-1) and (j > 0):
            label_down_left = label_matrix[i+1,  j-1].copy()
            pixel_down_left = binary_img[i+1,  j-1].copy()
            if pixel_down_left!=0:
                if label_down_left not in nb_label_list:
                    nb_label_list.append(label_down_left)
        if (i < label_matrix.shape[0]-1) and (j < label_matrix.shape[1]-1):
            label_down_right = label_matrix[i+1, j+1].copy()
            pixel_down_right = binary_img[i+1, j+1].copy()
            if pixel_down_right!=0:
                if label_down_right not in nb_label_list:
                    nb_label_list.append(label_down_right)
    return nb_label_list


def New_Two_Pass(binary_img: np.array, neighbor_hoods):
    """
    Perform optimized two-pass connected component labeling algorithm.

    Labels all connected components in a binary image using an improved
    two-pass algorithm that resolves label equivalences on-the-fly during
    the first pass.

    Args:
        binary_img (np.ndarray): Binary image with shape (H, W)
            Pixels != 0 are considered foreground, pixels == 0 are background
        neighbor_hoods (str): Connectivity type, either 'NEIGHBOR_HOODS_4' for
            4-connectivity or 'NEIGHBOR_HOODS_8' for 8-connectivity

    Returns:
        np.ndarray: Labeled image with shape (H, W)
            Each connected component has a unique sequential label (1, 2, 3, ...)
            Background pixels remain 0

    Raises:
        ValueError: If neighbor_hoods is not 'NEIGHBOR_HOODS_4' or 'NEIGHBOR_HOODS_8'

    Note:
        - Single-pass algorithm that resolves label equivalences immediately
        - When multiple neighbor labels are found, assigns minimum label and
          merges all equivalent labels in the label matrix
        - More efficient than classic two-pass algorithm for some cases
        - Handles edge cases where neighbors have label 0 (background)
    """
    if neighbor_hoods == 'NEIGHBOR_HOODS_4':
        pass
    elif neighbor_hoods == 'NEIGHBOR_HOODS_8':
        pass
    else:
        raise ValueError

    rows, cols = binary_img.shape
    label_matrix = np.zeros((rows, cols))
    label_idx = 1
    for i in range(0,rows):
        for j in range(0,cols):
            pixel_center = binary_img[i,j]
            if pixel_center !=0:
                
                nb_label_list = neighbor_label_list(i, j, label_matrix, binary_img, neighbor_hoods)
                # print('i --- ',i,' j --- ',j,' pixel_center --- ',pixel_center,' nb_label_list --- ',nb_label_list, \
                # ' label_matrix04 --- ',label_matrix[4,0],np.min(label_matrix),np.max(label_matrix))
                if (len(nb_label_list)==1) and (min(nb_label_list)==0):
                    label_matrix[i,j]=label_idx
                    label_idx = label_idx+1
                    # print('i --- ',i,' j --- ',j,' label_idex --- ',label_idx)
                if (len(nb_label_list)==1) and (min(nb_label_list)!=0):
                    label_matrix[i,j]=min(nb_label_list)
                if (len(nb_label_list)>1):
                    if 0 in nb_label_list:
                        if (len(nb_label_list)==2):
                            nb_label_list.remove(0)
                            if min(nb_label_list)!=0:
                                label_matrix[i,j]=min(nb_label_list)
                        if (len(nb_label_list)>2):
                            # if 0 in nb_label_list:
                            nb_label_list.remove(0)
                            if min(nb_label_list)!=0:
                                label_matrix[i,j]=min(nb_label_list)
                                min_nb_label_list=min(nb_label_list)
                                nb_label_list.remove(min_nb_label_list)
                                # print('i --- ',i,' j --- ',j,' pixel_center --- ',pixel_center,' remove nb_label_list --- ',nb_label_list)
                                for idx_nb_label_list in range(0, len(nb_label_list)):
                                    nb_value = nb_label_list[idx_nb_label_list]
                                    # print('nb_value --- ',nb_value)
                                    # print('np.argwhere(label_matrix == nb_value) --- ',np.argwhere(label_matrix == nb_value))
                                    # print('np.argwhere(label_matrix == nb_value) --- ',np.argwhere(label_matrix == 3))
                                    label_matrix[label_matrix == nb_value] = min_nb_label_list
                                    # print('np.argwhere(label_matrix == nb_value) --- ',np.argwhere(label_matrix == 3))
                    if 0 not in nb_label_list:
                        label_matrix[i,j]=min(nb_label_list)
                        min_nb_label_list=min(nb_label_list)
                        nb_label_list.remove(min_nb_label_list)
                        for idx_nb_label_list in range(0, len(nb_label_list)):
                            nb_value = nb_label_list[idx_nb_label_list]
                            label_matrix[label_matrix == nb_value] = min_nb_label_list
    return label_matrix