#!/usr/bin/python3
# -*- coding:utf-8 -*-

import cv2


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
