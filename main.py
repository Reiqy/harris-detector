import cv2
import harris
import numpy as np


def load_img_grayscale(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


def convert_to_color(img_grayscale):
    return cv2.cvtColor(img_grayscale, cv2.COLOR_GRAY2BGR)


def draw_points_red(img_color, indices):
    img_copy = img_color.copy()
    for index in indices:
        cv2.circle(img_copy, (index[1], index[0]), 3, color=(0, 0, 255))
    return img_copy


def show_img(img_any, window_name, wait_and_destroy=False):
    cv2.imshow(window_name, img_any)
    if wait_and_destroy:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def show_with_points(img_color, indices, window_name, wait_and_destroy=False):
    show_img(draw_points_red(img_color, indices), window_name, wait_and_destroy)


def demo():
    img_grayscale = load_img_grayscale("data/church.jpeg")
    img_color = convert_to_color(img_grayscale)

    sobel_x = harris.get_sobel_operator_x()
    sobel_y = harris.get_sobel_operator_y()
    response_signal = harris.compute_harris_response(img_grayscale, 0.05, sobel_x, sobel_y, 1, path="signal")

    maxima_indices_signal = harris.find_local_maxima(response_signal, 20, 22000)
    maxima_indices_signal = harris.sort_indices_by_value(response_signal, maxima_indices_signal)
    maxima_indices_signal = harris.decluster_indices(maxima_indices_signal, 20)

    response_ndimage = harris.compute_harris_response(img_grayscale, 0.05, sobel_x, sobel_y, 1, path="ndimage")

    maxima_indices_ndimage = harris.find_local_maxima(response_ndimage, 20, 0.001)
    maxima_indices_ndimage = harris.sort_indices_by_value(response_ndimage, maxima_indices_ndimage)
    maxima_indices_ndimage = harris.decluster_indices(maxima_indices_ndimage, 20)

    show_img(img_grayscale, "Original Image")
    show_with_points(img_color, maxima_indices_signal, "Corner Points (Signal for Convolution)")
    show_with_points(img_color, maxima_indices_ndimage, "Corner Points (ndImage for Convolution)", wait_and_destroy=True)

    cv2.imwrite("readme/church_corners.jpeg", draw_points_red(img_color, maxima_indices_signal))


if __name__ == '__main__':
    demo()
