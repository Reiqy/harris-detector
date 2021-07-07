import sys
import numpy as np
from scipy import ndimage, signal


def get_sobel_operator_x():
    return np.array(
        [
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]
    )


def get_sobel_operator_y():
    return get_sobel_operator_x().transpose()


# img must be float
def compute_harris_response(img, k, differential_operator_x, differential_operator_y, sigma, path):
    img_dx = None
    img_dy = None
    if path == "ndimage":
        img = img / 255
        img_dx = ndimage.convolve(img, differential_operator_x)
        img_dy = ndimage.convolve(img, differential_operator_y)
    elif path == "signal":
        img_dx = signal.convolve2d(img, differential_operator_x, mode="same", boundary="symm")
        img_dy = signal.convolve2d(img, differential_operator_y, mode="same", boundary="symm")
    else:
        print(f"Error: Path {path} is undefined. Choose either ndimage or signal.", file=sys.stderr)
        exit(1)

    h_tl = ndimage.gaussian_filter(img_dx * img_dy, sigma)
    h_br = ndimage.gaussian_filter(img_dy * img_dy, sigma)
    h_tr = ndimage.gaussian_filter(img_dx * img_dy, sigma)

    return ((h_tl * h_br) - np.square(h_tr)) - k * np.square(h_tl + h_br)


def find_local_maxima(discrete_function, neighborhood_size, threshold):
    # consider using peak_local_max() from skimage
    function_max = ndimage.maximum_filter(discrete_function, neighborhood_size)
    function_min = ndimage.minimum_filter(discrete_function, neighborhood_size)

    threshold_mask = (function_max - function_min) > threshold

    local_maximum_img = np.zeros(discrete_function.shape)
    local_maximum_img[discrete_function == function_max] = discrete_function[discrete_function == function_max]
    local_maximum_img[threshold_mask == 0] = 0

    return np.asarray(local_maximum_img.nonzero()).transpose()


def sort_indices_by_value(discrete_function, indices):
    return np.asarray(sorted(indices, key=lambda x: discrete_function[x[0]][x[1]], reverse=True))


def decluster_indices(indices, rho):
    if len(indices) <= 0:
        print("test")
        print("Error: No corner points found!", file=sys.stderr)
        exit(1)

    result = [indices[0]]

    for index in indices[1:]:
        index_numpy = np.array(index)
        if np.min(np.linalg.norm(np.array(result) - index_numpy, axis=1)) > rho:
            result.append(index)

    return np.array(result)
