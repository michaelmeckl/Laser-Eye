#!/usr/bin/python3
# -*- coding:utf-8 -*-

from numpy import ndarray
from scipy.spatial import distance as dist
import cv2
# import dlib


def preprocess_frame(frame: ndarray, kernel_size=5, keep_dim=True) -> ndarray:
    """
    Converts bgr image to grayscale using opencv. If keep_dim is set to True it will
    be converted back to BGR afterwards so that the third dimension will still be 3.
    This is necessary for some mxnet detectors that require a 4-Dim image input.
    """
    # Convert to grayscale and keep dimensions as required for the face (alignment) detectors
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if keep_dim:
        grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

    # blur to reduce noise; use smaller kernels to improve speed
    blurred_image = cv2.GaussianBlur(grayscale_image, (kernel_size, kernel_size), 0)
    return blurred_image


def convert_to_grayscale(rgb_img):
    """
    Convert linear RGB values to linear grayscale values.
    This function was taken from https://gitlab.com/brohrer/lodgepole/-/blob/main/lodgepole/image_tools.py;
    read https://e2eml.school/convert_rgb_to_grayscale.html for an explanation of the values
    """
    red = rgb_img[:, :, 0]
    green = rgb_img[:, :, 1]
    blue = rgb_img[:, :, 2]

    gray_img = (0.299 * red + 0.587 * green + 0.114 * blue)
    return gray_img


def eye_aspect_ratio(eye):
    """
    Computes the eye-aspect-ration (EAR) for the given eye landmarks.
    This function was taken from https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
    and adjusted to work with 8 instead of 6 eye markers.

    Args:
        eye: an array with the landmarks for this eye

    Returns:
        the eye-aspect-ratio (ear)
    """
    # compute the euclidean distances between the sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[7])
    B = dist.euclidean(eye[2], eye[6])
    C = dist.euclidean(eye[3], eye[5])
    # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    D = dist.euclidean(eye[0], eye[4])

    # compute the eye aspect ratio
    ear = (A + B + C) / (3.0 * D)  # 3*D as we have three vertical but only one horizontal coordinate pair
    return ear


def __zoom(img, center=None):
    """
    It takes the size of the image, finds the center value, calculates the size according to the scale,
    crops it accordingly, and increases the size to the original size to return.

    This function was taken from https://github.com/harimkang/openCV-with-Zoom/blob/master/Camera.py
    """
    scale = 10
    # zoom하는 실제 함수
    height, width = img.shape[:2]
    if center is None:
        #   중심값이 초기값일 때의 계산
        center_x = int(width / 2)
        center_y = int(height / 2)
        radius_x, radius_y = int(width / 2), int(height / 2)
    else:
        #   특정 위치 지정시 계산
        rate = height / width
        center_x, center_y = center

        #   비율 범위에 맞게 중심값 계산
        if center_x < width * (1 - rate):
            center_x = width * (1 - rate)
        elif center_x > width * rate:
            center_x = width * rate
        if center_y < height * (1 - rate):
            center_y = height * (1 - rate)
        elif center_y > height * rate:
            center_y = height * rate

        center_x, center_y = int(center_x), int(center_y)
        left_x, right_x = center_x, int(width - center_x)
        up_y, down_y = int(height - center_y), center_y
        radius_x = min(left_x, right_x)
        radius_y = min(up_y, down_y)

    # 실제 zoom 코드
    radius_x, radius_y = int(scale * radius_x), int(scale * radius_y)

    # size 계산
    min_x, max_x = center_x - radius_x, center_x + radius_x
    min_y, max_y = center_y - radius_y, center_y + radius_y

    # size에 맞춰 이미지를 자른다
    cropped = img[min_y:max_y, min_x:max_x]
    # 원래 사이즈로 늘려서 리턴
    new_cropped = cv2.resize(cropped, (width, height))

    return new_cropped
