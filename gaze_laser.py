#!/usr/bin/python3
# -*- coding:utf-8 -*-

from service.head_pose import HeadPoseEstimator
from service.face_alignment import CoordinateAlignmentModel
from service.face_detector import MxnetDetectionModel
from service.iris_localization import IrisLocalizationModel
import cv2
import numpy as np
from numpy import sin, cos, pi, arctan
from numpy.linalg import norm
import sys

SIN_LEFT_THETA = 2 * sin(pi / 4)
SIN_UP_THETA = sin(pi / 6)


def calculate_3d_gaze(frame, poi, scale=256):
    starts, ends, pupils, centers = poi

    eye_length = norm(starts - ends, axis=1)
    ic_distance = norm(pupils - centers, axis=1)
    zc_distance = norm(pupils - starts, axis=1)

    s0 = (starts[:, 1] - ends[:, 1]) * pupils[:, 0]
    s1 = (starts[:, 0] - ends[:, 0]) * pupils[:, 1]
    s2 = starts[:, 0] * ends[:, 1]
    s3 = starts[:, 1] * ends[:, 0]

    delta_y = (s0 - s1 + s2 - s3) / eye_length / 2
    delta_x = np.sqrt(abs(ic_distance ** 2 - delta_y ** 2))

    delta = np.array((delta_x * SIN_LEFT_THETA,
                      delta_y * SIN_UP_THETA))
    delta /= eye_length
    theta, pha = np.arcsin(delta)

    # print(f"THETA:{180 * theta / pi}, PHA:{180 * pha / pi}")
    # delta[0, abs(theta) < 0.1] = 0
    # delta[1, abs(pha) < 0.03] = 0

    inv_judge = zc_distance ** 2 - delta_y ** 2 < eye_length ** 2 / 4

    delta[0, inv_judge] *= -1
    theta[inv_judge] *= -1
    delta *= scale

    # mark pupil and pupil centers
    # TODO centers don't really work well, they are basically just bouncing around
    cv2.circle(frame, tuple(pupils[0].astype(int)), 2, (0, 255, 255), -1)
    cv2.circle(frame, tuple(pupils[1].astype(int)), 2, (0, 255, 255), -1)

    cv2.circle(frame, tuple(centers[0].astype(int)), 1, (0, 0, 255), -1)
    cv2.circle(frame, tuple(centers[1].astype(int)), 1, (0, 0, 255), -1)

    return theta, pha, delta.T


def draw_sticker(src, offset, pupils, landmarks,
                 blink_thd=0.22,
                 arrow_color=(0, 125, 255), copy=False):
    if copy:
        src = src.copy()  # make a copy of the current frame

    left_eye_height = landmarks[33, 1] - landmarks[40, 1]
    left_eye_width = landmarks[39, 0] - landmarks[35, 0]

    right_eye_height = landmarks[87, 1] - landmarks[94, 1]
    right_eye_width = landmarks[93, 0] - landmarks[89, 0]

    print(f"Left eye sizes: width: {left_eye_width}, height: {left_eye_height}")
    print(f"Right eye sizes: width: {right_eye_width}, height: {right_eye_height}")

    for mark in landmarks.reshape(-1, 2).astype(int):
        cv2.circle(src, tuple(mark), radius=1, color=(0, 0, 255), thickness=-1)

    # TODO use blink_treshold (0.22) and eye dimension ration to detect blinking ??
    #  -> wann eye closed ist, klappt im moment nur extrem ungenau! treshold zu groß?
    if left_eye_height / left_eye_width > blink_thd:
        cv2.arrowedLine(src, tuple(pupils[0].astype(int)),
                        tuple((offset + pupils[0]).astype(int)), arrow_color, 2)
    else:
        print("The user's left eye is closed!")

    if right_eye_height / right_eye_width > blink_thd:
        cv2.arrowedLine(src, tuple(pupils[1].astype(int)),
                        tuple((offset + pupils[1]).astype(int)), arrow_color, 2)
    else:
        print("The user's right eye is closed!")

    return src


def main(video, gpu_ctx=-1):
    capture = cv2.VideoCapture(video)
    video_width, video_height = capture.get(3), capture.get(4)

    face_detector = MxnetDetectionModel("weights/16and32", 0, .6, gpu=gpu_ctx)
    face_alignment = CoordinateAlignmentModel('weights/2d106det', 0, gpu=gpu_ctx)
    # face_alignment = CoordinateAlignmentModel('weights/model-hg2d3-cab/model', 0, gpu=gpu_ctx)  # does not work :(
    iris_locator = IrisLocalizationModel("weights/iris_landmark.tflite")
    head_pos = HeadPoseEstimator("weights/object_points.npy", video_width, video_height)

    c = 0
    while True:
        return_code, frame = capture.read()  # read from input video or webcam

        if not return_code:
            # break loop if getting frame was not successful
            sys.stderr.write("Unknown error while trying to get current frame!")
            break

        bboxes = face_detector.detect(frame)

        for landmarks in face_alignment.get_landmarks(frame, bboxes, calibrate=True):
            # print("\nCurrent landmarks: ", landmarks)

            # calculate head pose
            _, euler_angle = head_pos.get_head_pose(landmarks)
            pitch, yaw, roll = euler_angle[:, 0]

            eye_markers = np.take(landmarks, face_alignment.eye_bound, axis=0)
            # print(f"Eye markers: {eye_markers}")

            eye_centers = np.average(eye_markers, axis=1)
            # eye_centers = landmarks[[34, 88]]
            print(f"Eye centers: {eye_centers}")

            # eye_lengths = np.linalg.norm(landmarks[[39, 93]] - landmarks[[35, 89]], axis=1)
            eye_lengths = (landmarks[[39, 93]] - landmarks[[35, 89]])[:, 0]
            # print(f"Eye lengths: {eye_lengths}")

            iris_left = iris_locator.get_mesh(frame, eye_lengths[0], eye_centers[0])
            pupil_left, _ = iris_locator.draw_pupil(iris_left, frame, thickness=1)

            iris_right = iris_locator.get_mesh(frame, eye_lengths[1], eye_centers[1])
            pupil_right, _ = iris_locator.draw_pupil(iris_right, frame, thickness=1)

            print(f"Pupil left: {pupil_left}")
            print(f"Pupil right: {pupil_right}")

            pupils = np.array([pupil_left, pupil_right])

            poi = landmarks[[35, 89]], landmarks[[39, 93]], pupils, eye_centers
            theta, pha, delta = calculate_3d_gaze(frame, poi)

            if yaw > 30:
                end_mean = delta[0]
            elif yaw < -30:
                end_mean = delta[1]
            else:
                end_mean = np.average(delta, axis=0)

            if end_mean[0] < 0:
                zeta = arctan(end_mean[1] / end_mean[0]) + pi
            else:
                zeta = arctan(end_mean[1] / (end_mean[0] + 1e-7))

            # print(zeta * 180 / pi)
            # print(zeta)
            if roll < 0:
                roll += 180
                print(f"Head Roll Position is >= 0 ({roll}°)")
            else:
                roll -= 180
                print(f"Head Roll Position is < 0 ({roll}°)")

            real_angle = zeta + roll * pi / 180
            # real_angle = zeta

            # print("end mean:", end_mean)
            # print(roll, real_angle * 180 / pi)

            # calculate the norm of the vector (i.e. the length)
            R = norm(end_mean)
            offset = R * cos(real_angle), R * sin(real_angle)
            # print(f"R = {R}, offset = {offset}")

            landmarks[[38, 92]] = landmarks[[34, 88]] = eye_centers

            iris_locator.draw_eye_markers(eye_markers, frame, thickness=1)
            draw_sticker(frame, offset, pupils, landmarks)

        # show current annotated frame and save it as a png
        cv2.imshow('res', frame)
        # cv2.imwrite(f'tracking_images/frame_{c}.png', frame)
        c += 1

        # infinite loop until the user presses 'q' on the keyboard
        # replace 1 with 0 to only show 'frame-by-frame'
        if cv2.waitKey(1) == ord('q'):
            break

    # cleanup opencv stuff
    capture.release()
    cv2.destroyAllWindows()


# TODO we need:
"""
- (annotated) webcam images  ✔️
- pupil positions and pupil sizes (diameter)  ✔️ (but probably not good enough; most likely only noise)
- fixations and saccades (count, mean, std)   ❌ # TODO
- blinks (rate, number, etc.)   ❌ (basic approaches are there; need to be expanded to actually be useful)

- gaze direction ?   (✔️) (only a basic estimation, but probably not even needed)
"""

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        # fall back to webcam if no input video was provided
        main(0)
