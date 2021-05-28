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

# TODO debug only:
blink_counter_left, blink_counter_right = 0, 0


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
    cv2.circle(frame, tuple(pupils[0].astype(int)), 2, (0, 255, 255), -1)
    cv2.circle(frame, tuple(pupils[1].astype(int)), 2, (0, 255, 255), -1)
    # cv2.circle(frame, tuple(centers[0].astype(int)), 1, (0, 0, 255), -1)
    # cv2.circle(frame, tuple(centers[1].astype(int)), 1, (0, 0, 255), -1)

    return theta, pha, delta.T


def draw_sticker(src, offset, pupils, landmarks, left_eye_size, right_eye_size,
                 blink_thd=0.22,
                 arrow_color=(0, 125, 255), copy=False):
    # TODO remove after debugging:
    global blink_counter_left, blink_counter_right

    left_eye_height, left_eye_width = left_eye_size[1], left_eye_size[0]
    right_eye_height, right_eye_width = right_eye_size[1], right_eye_size[0]

    if copy:
        src = src.copy()  # make a copy of the current frame

    for mark in landmarks.reshape(-1, 2).astype(int):
        cv2.circle(src, tuple(mark), radius=1, color=(0, 0, 255), thickness=-1)

    # TODO use blink_treshold (0.22) and eye dimension ration to detect blinking ??
    #  -> treshold zu groß? bzw. woher kommt der wert?
    if left_eye_height / left_eye_width > blink_thd:
        cv2.arrowedLine(src, tuple(pupils[0].astype(int)),
                        tuple((offset + pupils[0]).astype(int)), arrow_color, 2)
    else:
        print(f"The user blinked {blink_counter_left} times with his left eye!")
        blink_counter_left += 1

    if right_eye_height / right_eye_width > blink_thd:
        cv2.arrowedLine(src, tuple(pupils[1].astype(int)),
                        tuple((offset + pupils[1]).astype(int)), arrow_color, 2)
    else:
        print(f"The user blinked {blink_counter_right} times with his right eye!")
        blink_counter_right += 1

    return src


def show_eye_region(frame, region_center, min_x, max_x, min_y, max_y):
    # draw circle at eye region center (the middle point between both eyes)
    cv2.circle(frame, (int(region_center[0]), int(region_center[1])),
               5, color=(0, 255, 0))
    # draw circle around the whole eye region
    cv2.circle(frame, (int(region_center[0]), int(region_center[1])),
               int(region_center[0] - min_x), color=(170, 0, 255))
    # draw a rectangle around the whole eye region
    cv2.rectangle(frame, (min_x.astype(int), min_y.astype(int)),
                  (max_x.astype(int), max_y.astype(int)), (0, 255, 255), 3)

    # draw a square around the eye region
    center_y = int(region_center[1])
    eye_region_width = max_x - min_x
    min_y_rect = center_y - int(eye_region_width / 2)
    max_y_rect = center_y + int(eye_region_width / 2)
    cv2.rectangle(frame, (min_x.astype(int), min_y_rect),
                  (max_x.astype(int), max_y_rect), (0, 222, 222), 2)


def get_eye_region(frame, eye_markers):
    # TODO get eye regions for both eyes separately instead??
    print("Eyemarkers:")

    # find the outermost eye markers: i.e. the smallest and largest x and y values in the matrix
    min_vals = np.amin(eye_markers, axis=1)
    min_x = np.min(min_vals[:, 0])
    min_y = np.min(min_vals[:, 1])
    # print(f"min val über beide Augen:\n{[min_x, min_y]}")
    max_vals = np.amax(eye_markers, axis=1)
    max_x = np.max(max_vals[:, 0])
    max_y = np.max(max_vals[:, 1])
    # print(f"max val über beide Augen:\n{[max_x, max_y]}")

    # get the center point of the eye region by adding half of the eye region width
    # and height to the min x and y values; Important: as opencv needs pixels to
    # draw, we must provide integers which is why we use the floor division!
    region_center = (min_x + (max_x - min_x) // 2, min_y + (max_y - min_y) // 2)
    print(f"Eye region center is at {region_center}")

    # calculate a squared bbox around the eye region; we consider only the region width
    # as the eyes I know are usually far wider than large (correct me if I'm wrong ...)
    center_y = int(region_center[1])
    eye_region_width = max_x - min_x
    min_y_rect = center_y - int(eye_region_width / 2)
    max_y_rect = center_y + int(eye_region_width / 2)

    # visualize it!
    show_eye_region(frame, region_center, min_x, max_x, min_y, max_y)

    # TODO Lösungsansätze für Problem mit unterschiedlichen Bilddimensionen pro Frame:
    # 1. kleinere bilder mit padding versehen bis alle gleich groß wie größtes
    # 2. größere bilder runterskalieren bis alle gleich groß wie kleinstes (oder alternativ crop)
    # 3. jetzt erstmal unterschiedlich lassen und dann später beim CNN vorverarbeiten!
    #      -> vermtl. eh am besten weil später neue Bilder ja auch erstmal vorverarbeitet werden müssen!
    # TODO: => fürs Erste jetzt ignorieren und nur Rechtecke nehmen!

    # eye_ROI = frame[min_y.astype(int): max_y.astype(int), min_x.astype(int): max_x.astype(int)]

    # we need square images later for our CNN!
    eye_ROI = frame[min_y_rect: max_y_rect, min_x.astype(int): max_x.astype(int)]
    return eye_ROI


def get_pupil_bboxes(frame, left_pupil, right_pupil, left_eye_size, right_eye_size):
    """
    Get both pupils as separate (square-sized) images.
    """
    x_min = left_pupil[0]  # no need to check for min as the left eye SHOULD be further left than the right eye ;)
    x_max = right_pupil[0]
    y_min = min(left_pupil[1], right_pupil[1])
    y_max = max(left_pupil[1], right_pupil[1])
    # print(f"{y_min}, {y_max}; {x_min}, {x_max}")

    # calculate the bounding box that encompasses both pupils
    left_eye_height, left_eye_width = left_eye_size[1], left_eye_size[0]
    right_eye_height, right_eye_width = right_eye_size[1], right_eye_size[0]
    max_height = max(left_eye_height, right_eye_height)
    max_width = max(left_eye_width, right_eye_width)

    min_y_rect = int(y_min - max_height / 2 - 10)  # 10 pixel padding so it isn't cut off immediately at the edges
    max_y_rect = int(y_min + max_height / 2 + 10)
    min_x_rect = int(x_min - max_height / 2 - 10)
    max_x_rect = int(x_max + max_height / 2 + 10)
    pupils_bbox = [(min_x_rect, min_y_rect), (max_x_rect, max_y_rect)]
    cv2.rectangle(frame, pupils_bbox[0], pupils_bbox[1], (222, 0, 222), 2)  # not a square of course ...

    # calculate bboxes for both pupils separately
    # we use a half the max eye width as the square size; this turned out to be 
    # pretty accurate most of the time and is the best we can get from the provided landmarks
    left_pupil_left_x = int(x_min - max_width / 4)
    left_pupil_right_x = int(x_min + max_width / 4)
    left_pupil_min_y = int(left_pupil[1] - max_width / 4)
    left_pupil_max_y = int(left_pupil[1] + max_width / 4)
    cv2.rectangle(frame, (int(left_pupil_left_x), int(left_pupil_min_y)),
                  (int(left_pupil_right_x), int(left_pupil_max_y)), (0, 120, 222), 2)

    right_pupil_left_x = int(x_max - max_width / 4)
    right_pupil_right_x = int(x_max + max_width / 4)
    right_pupil_min_y = int(right_pupil[1] - max_width / 4)
    right_pupil_max_y = int(right_pupil[1] + max_width / 4)
    cv2.rectangle(frame, (int(right_pupil_left_x), int(right_pupil_min_y)),
                  (int(right_pupil_right_x), int(right_pupil_max_y)), (0, 120, 222), 2)

    left_pupil_bbox = frame[left_pupil_min_y: left_pupil_max_y,
                            left_pupil_left_x: left_pupil_right_x]
    right_pupil_bbox = frame[right_pupil_min_y: right_pupil_max_y,
                             right_pupil_left_x: right_pupil_right_x]
    return left_pupil_bbox, right_pupil_bbox


def main(video, gpu_ctx=-1):
    capture = cv2.VideoCapture(video)
    video_width, video_height = capture.get(3), capture.get(4)

    face_detector = MxnetDetectionModel("weights/16and32", 0, .6, gpu=gpu_ctx)
    face_alignment = CoordinateAlignmentModel('weights/2d106det', 0, gpu=gpu_ctx)
    # face_alignment = CoordinateAlignmentModel('weights/model-hg2d3-cab/model', 0, gpu=gpu_ctx)  # does not work :(
    iris_locator = IrisLocalizationModel("weights/iris_landmark.tflite")
    head_pos = HeadPoseEstimator("weights/object_points.npy", video_width, video_height)

    # init the different counters for saving images
    # TODO save timestamps instead of counter
    c, rn, pn_l, pn_r = 0, 0, 0, 0

    while True:
        return_code, frame = capture.read()  # read from input video or webcam

        if not return_code:
            # break loop if getting frame was not successful
            sys.stderr.write("Unknown error while trying to get current frame!")
            break

        bboxes = face_detector.detect(frame)

        for landmarks in face_alignment.get_landmarks(frame, bboxes, calibrate=True):
            # calculate head pose
            _, euler_angle = head_pos.get_head_pose(landmarks)
            pitch, yaw, roll = euler_angle[:, 0]

            eye_markers = np.take(landmarks, face_alignment.eye_bound, axis=0)
            # print(f"Eye markers: {eye_markers}")

            eye_centers = np.average(eye_markers, axis=1)
            # eye_centers = landmarks[[34, 88]]
            # print(f"Eye centers: {eye_centers}")

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
                # print(f"Head Roll Position is >= 0 ({roll}°)")
            else:
                roll -= 180
                # print(f"Head Roll Position is < 0 ({roll}°)")

            real_angle = zeta + roll * pi / 180
            # real_angle = zeta

            # print("end mean:", end_mean)
            # print(roll, real_angle * 180 / pi)

            # calculate the norm of the vector (i.e. the length)
            R = norm(end_mean)
            offset = R * cos(real_angle), R * sin(real_angle)
            # print(f"R = {R}, offset = {offset}")

            landmarks[[38, 92]] = landmarks[[34, 88]] = eye_centers

            left_eye_height = landmarks[33, 1] - landmarks[40, 1]
            left_eye_width = landmarks[39, 0] - landmarks[35, 0]
            left_eye_size = [left_eye_width, left_eye_height]

            right_eye_height = landmarks[87, 1] - landmarks[94, 1]
            right_eye_width = landmarks[93, 0] - landmarks[89, 0]
            right_eye_size = [right_eye_width, right_eye_height]

            print(f"Left eye sizes: width: {left_eye_width}, height: {left_eye_height}")
            print(f"Right eye sizes: width: {right_eye_width}, height: {right_eye_height}")

            iris_locator.draw_eye_markers(eye_markers, frame, thickness=1)
            draw_sticker(frame, offset, pupils, landmarks, left_eye_size, right_eye_size)

            eye_region = get_eye_region(frame, eye_markers)
            # crashes if eye_region is empty matrix (e.g. if face is not fully visible)
            if eye_region.size:
                # cv2.imwrite(f'eye_regions/region_{rn}.png', eye_region)
                rn += 1

            left_pupil_bbox, right_pupil_bbox = get_pupil_bboxes(frame,
                                                                 pupil_left,
                                                                 pupil_right,
                                                                 left_eye_size,
                                                                 right_eye_size)
            if left_pupil_bbox.size:
                # the left pupil is actually the right one as it is mirrored:
                cv2.imwrite(f'pupil_regions/pupil_right_{pn_l}.png', left_pupil_bbox)
                pn_l += 1

            if right_pupil_bbox.size:
                cv2.imwrite(f'pupil_regions/pupil_left_{pn_r}.png', right_pupil_bbox)
                pn_r += 1

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
- (annotated) webcam images  (✔)️
- pupil positions and pupil sizes (diameter)  ✔ (but probably not good enough; most likely only noise)
- fixations and saccades (count, mean, std)   ❌ # TODO
- blinks (rate, number, etc.)   ❌ (basic approaches are there; need to be expanded to actually be useful)

- gaze direction ?   (✔ ️) (only a basic estimation, but probably not even needed)
"""

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        # fall back to webcam if no input video was provided
        main(0)
