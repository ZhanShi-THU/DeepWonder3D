import cv2
import numpy as np
import scipy.io as scio

NEIGHBOR_HOODS_4 = True
OFFSETS_4 = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]

NEIGHBOR_HOODS_8 = False
OFFSETS_8 = [[-1, -1], [0, -1], [1, -1],
             [-1,  0], [0,  0], [1,  0],
             [-1,  1], [0,  1], [1,  1]]



def reorganize(binary_img: np.array):
    """
    Reorganize labeled binary image and extract point lists for each label.

    Converts a labeled binary image into sequential labels (1, 2, 3, ...) and
    collects all pixel coordinates for each connected component.

    Args:
        binary_img (np.ndarray): Labeled binary image with shape (H, W)
            Background pixels should be < 0.5, object pixels have various labels

    Returns:
        tuple: (binary_img, points)
            - binary_img (np.ndarray): Reorganized image with sequential labels
              (1, 2, 3, ...) for each unique component
            - points (list): List of point lists, where each inner list contains
              [row, col] coordinates for pixels belonging to that component

    Note:
        - Pixels with value < 0.5 are considered background and skipped
        - Each unique label value becomes a sequential number starting from 1
        - Point lists are indexed by component label (0-indexed in the list)
    """
    index_map = []
    points = []
    index = -1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var < 0.5:
                continue
            if var in index_map:
                index = index_map.index(var)
                num = index + 1
            else:
                index = len(index_map)
                num = index + 1
                index_map.append(var)
                points.append([])
            binary_img[row][col] = num
            points[index].append([row, col])
    return binary_img, points



def neighbor_value(binary_img: np.array, offsets, reverse=False):
    """
    Label connected components by propagating minimum neighbor labels.

    Scans the binary image and assigns each foreground pixel the minimum label
    value found among its neighbors. Can scan in forward or reverse order.

    Args:
        binary_img (np.ndarray): Binary image with shape (H, W)
            Background pixels should be < 0.5, foreground pixels can have labels
        offsets (list): List of [row_offset, col_offset] pairs defining
            neighbor connectivity (e.g., 4-connectivity or 8-connectivity)
        reverse (bool, optional): If True, scan from bottom-right to top-left.
            Default is False (top-left to bottom-right).

    Returns:
        np.ndarray: Labeled binary image with same shape as input
            Each foreground pixel is assigned the minimum label from its neighbors

    Note:
        - If no labeled neighbors found, assigns a new sequential label
        - Neighbor coordinates are clamped to image boundaries
        - Used as part of the two-pass connected component labeling algorithm
    """
    rows, cols = binary_img.shape
    label_idx = 0
    rows_ = [0, rows, 1] if reverse == False else [rows-1, -1, -1]
    cols_ = [0, cols, 1] if reverse == False else [cols-1, -1, -1]
    for row in range(rows_[0], rows_[1], rows_[2]):
        for col in range(cols_[0], cols_[1], cols_[2]):
            label = 256
            if binary_img[row][col] < 0.5:
                continue
            for offset in offsets:
                neighbor_row = min(max(0, row+offset[0]), rows-1)
                neighbor_col = min(max(0, col+offset[1]), cols-1)
                neighbor_val = binary_img[neighbor_row, neighbor_col]
                if neighbor_val < 0.5:
                    continue
                label = neighbor_val if neighbor_val < label else label
            if label == 255:
                label_idx += 1
                label = label_idx
            binary_img[row][col] = label
    return binary_img

def Two_Pass(binary_img: np.array, neighbor_hoods):
    """
    Perform two-pass connected component labeling algorithm.

    Labels all connected components in a binary image using the classic two-pass
    algorithm. First pass assigns provisional labels, second pass resolves
    label equivalences.

    Args:
        binary_img (np.ndarray): Binary image with shape (H, W)
            Background pixels should be 0, object pixels should be 255 (or > 0.5)
        neighbor_hoods (str): Connectivity type, either 'NEIGHBOR_HOODS_4' for
            4-connectivity or 'NEIGHBOR_HOODS_8' for 8-connectivity

    Returns:
        np.ndarray: Labeled image with shape (H, W)
            Each connected component has a unique sequential label (1, 2, 3, ...)
            Background pixels remain 0

    Raises:
        ValueError: If neighbor_hoods is not 'NEIGHBOR_HOODS_4' or 'NEIGHBOR_HOODS_8'

    Note:
        - First pass: forward scan assigning provisional labels
        - Second pass: reverse scan resolving label equivalences
        - 4-connectivity: up, down, left, right neighbors
        - 8-connectivity: includes diagonal neighbors
    """
    if neighbor_hoods == 'NEIGHBOR_HOODS_4':
        offsets = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]
    elif neighbor_hoods == 'NEIGHBOR_HOODS_8':
        offsets = [[-1, -1], [0, -1], [1, -1],
             [-1,  0], [0,  0], [1,  0],
             [-1,  1], [0,  1], [1,  1]]
    else:
        raise ValueError

    binary_img = neighbor_value(binary_img, offsets, False)
    binary_img = neighbor_value(binary_img, offsets, True)

    return binary_img



def recursive_seed(binary_img: np.array, seed_row, seed_col, offsets, num, max_num=100):
    """
    Recursively fill connected component starting from a seed point.

    Uses seed-filling algorithm to label all pixels in a connected component
    starting from a seed position. Recursively visits all connected neighbors.

    Args:
        binary_img (np.ndarray): Binary image with shape (H, W)
        seed_row (int): Starting row coordinate for seed filling
        seed_col (int): Starting column coordinate for seed filling
        offsets (list): List of [row_offset, col_offset] pairs defining
            neighbor connectivity
        num (int): Label value to assign to this component
        max_num (int, optional): Maximum value threshold. Pixels with value
            >= max_num are considered unlabeled and will be filled. Default is 100.

    Returns:
        np.ndarray: Modified binary image with the connected component labeled

    Note:
        - Recursively visits all neighbors that have value >= max_num
        - Neighbor coordinates are clamped to image boundaries
        - Used by Seed_Filling for connected component labeling
    """
    rows, cols = binary_img.shape
    binary_img[seed_row][seed_col] = num
    for offset in offsets:
        neighbor_row = min(max(0, seed_row+offset[0]), rows-1)
        neighbor_col = min(max(0, seed_col+offset[1]), cols-1)
        var = binary_img[neighbor_row][neighbor_col]
        if var < max_num:
            continue
        binary_img = recursive_seed(binary_img, neighbor_row, neighbor_col, offsets, num, max_num)
    return binary_img

def Seed_Filling(binary_img, neighbor_hoods, max_num=1000000):
    """
    Perform seed-filling algorithm for connected component labeling.

    Labels all connected components in a binary image using seed-filling
    (flood-fill) algorithm. Finds unlabeled foreground pixels and recursively
    fills the entire connected component.

    Args:
        binary_img (np.ndarray): Binary image with shape (H, W)
            Pixels with value > max_num are considered unlabeled foreground
        neighbor_hoods (str): Connectivity type, either 'NEIGHBOR_HOODS_4' for
            4-connectivity or 'NEIGHBOR_HOODS_8' for 8-connectivity
        max_num (int, optional): Maximum value threshold. Pixels with value
            > max_num are considered unlabeled and will be filled. Default is 1000000.

    Returns:
        np.ndarray: Labeled image with shape (H, W)
            Each connected component has a unique sequential label (1, 2, 3, ...)

    Raises:
        ValueError: If neighbor_hoods is not 'NEIGHBOR_HOODS_4' or 'NEIGHBOR_HOODS_8'

    Note:
        - Scans image row by row, column by column
        - When an unlabeled foreground pixel is found, starts recursive seed filling
        - Each component gets a unique sequential label
        - max_num parameter allows filtering of already-labeled pixels
    """
    if neighbor_hoods == 'NEIGHBOR_HOODS_4':
        offsets = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]
    elif neighbor_hoods == 'NEIGHBOR_HOODS_8':
        offsets = [[-1, -1], [0, -1], [1, -1],
             [-1,  0], [0,  0], [1,  0],
             [-1,  1], [0,  1], [1,  1]]
    else:
        raise ValueError

    num = 1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var <= max_num:
                continue
            binary_img = recursive_seed(binary_img, row, col, offsets, num, max_num=1000000)
            num += 1
    return binary_img


def cal_pccs(x, y, n):
    """
    Calculate Pearson correlation coefficient (PCC) between two variables.

    Computes the Pearson correlation coefficient using the formula:
    PCC = (n*Σxy - Σx*Σy) / sqrt((n*Σx² - (Σx)²) * (n*Σy² - (Σy)²))

    Args:
        x (np.ndarray): Variable 1, must be numpy array
        y (np.ndarray): Variable 2, must be numpy array with same shape as x
        n (int): Number of elements in x (should match x.size)

    Returns:
        float: Pearson correlation coefficient in range [-1, 1]
            - 1: Perfect positive correlation
            - -1: Perfect negative correlation
            - 0: No correlation

    Note:
        - Input data format must be numpy array
        - Both arrays are flattened before calculation
        - Returns NaN if denominator is zero (constant arrays)
    """
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc


def four_cc_label(img):
    """
    Perform 4-connected component labeling using optimized algorithm.

    Labels all 4-connected components in a binary image using an optimized
    algorithm with lookup table (LUT) for label equivalence resolution.

    Args:
        img (np.ndarray): Binary image with shape (H, W)
            Pixels > 0 are considered foreground, pixels == 0 are background

    Returns:
        np.ndarray: Labeled image with shape (H, W) and dtype uint8
            Each connected component has a unique label value
            Background pixels remain 0

    Note:
        - Uses 4-connectivity (up, down, left, right neighbors)
        - Implements LUT-based label equivalence resolution for efficiency
        - Scans top-to-bottom, left-to-right
        - Checks neighbors: top-left, top, top-right, left
        - Prints progress every 500 pixels for large images
    """
    height, width = img.shape
    label = np.zeros((height, width), dtype=np.int32)
    LUT = np.zeros(height * width,dtype=np.uint8)

    COLORS = range(0,1000000)
    out = np.zeros((height, width), dtype=np.uint8)
    label[img[:,:] > 0] = 1

    n = 1
    for y in range(height):
        for x in range(width):
            if y%500==0:
                if x%500==0:
                    print(y,' --- ',height,' --- ',x,' --- ',width)
            if label[y, x] == 0:
                continue
            c2 = label[max(y - 1, 0), min(x + 1, width - 1)]
            c3 = label[max(y - 1, 0), x]
            c4 = label[max(y - 1, 0), max(x - 1, 0)]
            c5 = label[y, max(x - 1, 0)]
            if c3 < 2 and c5 < 2 and c2 < 2 and c4 < 2:
                n += 1
                label[y, x] = n
            else:
                _vs = [c3, c5, c2, c4]
                vs = [a for a in _vs if a > 1]
                v = min(vs)
                label[y, x] = v

                minv = v
                for _v in vs:
                    if LUT[_v] != 0:
                        minv = min(minv, LUT[_v])
                for _v in vs:
                    LUT[_v] = minv

    count = 1
    for l in range(2, n + 1):
        flag = True
        for i in range(n + 1):
            if LUT[i] == l:
                if flag:
                    count += 1
                    flag = False
                LUT[i] = count


    for i, lut in enumerate(LUT[2:]):
        if i%500==0:
            print('LUT',i,len(LUT[2:]))
        out[label == (i + 2)] = COLORS[lut - 2]
    return out



if __name__ == "__main__":
    binary_img = np.zeros((4, 7), dtype=np.int16)
    index = [[0, 2], [0, 5],
            [1, 0], [1, 1], [1, 2], [1, 4], [1, 5], [1, 6],
            [2, 2], [2, 5],
            [3, 1], [3, 2], [3, 4], [3, 6]]
    for i in index:
        binary_img[i[0], i[1]] = np.int16(255)

    print("原始二值图像")
    print(binary_img)

    print("Two_Pass")
    binary_img = Two_Pass(binary_img, NEIGHBOR_HOODS_8)
    binary_img, points = reorganize(binary_img)
    print(binary_img, points)

    print("Seed_Filling")
    binary_img = Seed_Filling(binary_img, NEIGHBOR_HOODS_8)
    binary_img, points = reorganize(binary_img)
    print(binary_img, points)