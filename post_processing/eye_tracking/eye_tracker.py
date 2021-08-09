import pathlib
import threading
import cv2
import numpy as np
import pyautogui as pyautogui
from numpy import sin, cos, pi, arctan
from numpy.linalg import norm
from post_processing.eye_tracking.ProcessingLogger import ProcessingLogger, ProcessingData
from post_processing.eye_tracking.image_utils import detect_pupils, apply_threshold, show_image_window
from post_processing_service.blink_detector import BlinkDetector
from post_processing_service.saccade_fixation_detector import SaccadeFixationDetector
from post_processing_service.face_alignment import CoordinateAlignmentModel
from tracking.TrackingLogger import get_timestamp
from tracking_service.face_detector import MxnetDetectionModel
from tracking.tracking_utils import extract_image_region
from post_processing_service.head_pose import HeadPoseEstimator
from post_processing_service.iris_localization import IrisLocalizationModel


# we need:
"""
- webcam images of eyes and pupils ✔
- pupil positions  ✔
- pupil sizes (diameter)  ❌ TODO
- average pupilsize; peak pupil size  ❌ TODO
- fixations and saccades (count, mean, std)   ❌ TODO
- blinks (rate, number, etc.)   ❌ (basic approaches are there; need to be expanded to actually be useful)
"""


# noinspection PyAttributeOutsideInit
class EyeTracker:

    def __init__(self, enable_annotation=False, debug_active=False, gpu_ctx=-1):
        self.__debug = debug_active
        self.__annotation_enabled = enable_annotation

        self.__screenWidth, self.__screenHeight = pyautogui.size()
        self.gaze_left, self.gaze_right = None, None

        self.__init_logger()

        self.saccade_detector = SaccadeFixationDetector()
        self.blink_detector = BlinkDetector(show_annotation=self.__annotation_enabled)

        weights_path = pathlib.Path(__file__).parent.parent.parent / "weights"
        self.face_detector = MxnetDetectionModel(f"{weights_path / '16and32'}", 0, .6, gpu=gpu_ctx)
        self.face_alignment = CoordinateAlignmentModel(f"{weights_path / '2d106det'}", 0, gpu=gpu_ctx)
        self.iris_locator = IrisLocalizationModel(f"{weights_path / 'iris_landmark.tflite'}")
        self.head_pose_estimator = HeadPoseEstimator(f"{weights_path / 'object_points.npy'}")

    def __init_logger(self):
        self.__logger = ProcessingLogger()
        # use the name of the enum as dict key as otherwise it would always generate a new key the next time
        # and would therefore always append new columns to our pandas dataframe!
        self.__tracked_data = {key.name: None for key in ProcessingData}

    def set_camera_matrix(self, frame_width, frame_height):
        # print(f"frame width: {frame_width}, frame height: {frame_height}")
        self.head_pose_estimator.set_camera_matrix(frame_width, frame_height)

    def set_current_participant(self, participant_folder, fps_val):
        self.__logger.set_participant(participant_folder)
        self.blink_detector.set_participant_fps(fps_val)

    def set_current_difficulty(self, difficulty):
        self.__logger.set_difficulty(difficulty)

    def reset_blink_detector(self):
        self.blink_detector.reset_blink_detection()

    def process_current_frame(self, frame: np.ndarray):
        """
        Args:
            frame: video frame in the format [width, height, channels]
        """

        # processed_frame = preprocess_frame(frame, kernel_size=3, keep_dim=True)
        self.__current_frame = frame

        bboxes = self.face_detector.detect(self.__current_frame)
        for landmarks in self.face_alignment.get_landmarks(self.__current_frame, bboxes, calibrate=True):
            self.__landmarks = landmarks

            # calculate head pose
            self.set_camera_matrix(frame_width=frame.shape[1], frame_height=frame.shape[0])
            _, euler_angle = self.head_pose_estimator.get_head_pose(landmarks)
            self.__pitch, self.__yaw, self.__roll = euler_angle[:, 0]

            # calculate eye size, iris and pupil positions
            self.__get_eye_features()

            self.__track_gaze()
            # self.__calculate_gaze_point()

            if self.__annotation_enabled:
                self.__draw_face_landmarks()
                self.iris_locator.draw_eye_markers(self.__eye_markers, self.__current_frame, thickness=1)

            # check if user blinked
            self.blink_detector.set_current_values(self.__current_frame, self.__left_eye, self.__right_eye,
                                                   (self.__left_eye_width, self.__left_eye_height),
                                                   (self.__right_eye_width, self.__right_eye_height))
            self.blink_detector.detect_blinks()

            # extract different parts of the eye region and save them as pngs
            eye_region_bbox = self.__extract_eye_region()
            left_eye_bbox, right_eye_bbox, left_eye_bbox_small, right_eye_bbox_small = self.__extract_eyes()

            self.__left_pupil_diameter, self.__right_pupil_diameter = detect_pupils(left_eye_bbox, right_eye_bbox,
                                                                                    self.__annotation_enabled)
            self.__log(eye_region_bbox, left_eye_bbox, right_eye_bbox, left_eye_bbox_small, right_eye_bbox_small)

            # new_eye_region = improve_image(eye_region_bbox)
            # self.__logger.log_image("eye_regions_improved", "region", new_eye_region, get_timestamp())

        return self.__current_frame

    """
    def show_face_region(self, bboxes):
        for face in bboxes:
            face_region = extract_image_region(self.__current_frame, face[0], face[1], face[2], face[3])
            cv2.rectangle(self.__current_frame, (int(face[0]), int(face[1])), (int(face[2]), int(face[3])),
                          color=(0, 0, 255))
            cv2.imshow("extracted", face_region)
            break  # break to take only the first face (in most cases there should be only one anyway)
    """

    def __log(self, eye_region_bbox, left_eye_bbox, right_eye_bbox, left_eye_bbox_small, right_eye_bbox_small):
        # fill dict with all relevant data so we don't have to pass all params manually
        self.__tracked_data.update({
            ProcessingData.HEAD_POS_ROLL_PITCH_YAW.name: (self.__roll, self.__pitch, self.__yaw),
            ProcessingData.LEFT_EYE.name: self.__left_eye,
            ProcessingData.RIGHT_EYE.name: self.__right_eye,
            ProcessingData.LEFT_EYE_CENTER.name: self.__eye_centers[0],
            ProcessingData.RIGHT_EYE_CENTER.name: self.__eye_centers[1],
            ProcessingData.LEFT_EYE_WIDTH.name: self.__left_eye_width,
            ProcessingData.RIGHT_EYE_WIDTH.name: self.__right_eye_width,
            ProcessingData.LEFT_EYE_HEIGHT.name: self.__left_eye_height,
            ProcessingData.RIGHT_EYE_HEIGHT.name: self.__right_eye_height,
            ProcessingData.LEFT_PUPIL_POS.name: self.__pupils[0],
            ProcessingData.RIGHT_PUPIL_POS.name: self.__pupils[1],
            ProcessingData.LEFT_PUPIL_DIAMETER.name: self.__left_pupil_diameter,
            ProcessingData.RIGHT_PUPIL_DIAMETER.name: self.__right_pupil_diameter,
            # ProcessingData.FACE_LANDMARKS.name: self.__landmarks,  # .tolist(),
        })

        # save timestamp separately as it has to be the same for all the frames and the log data! otherwise it
        # can't be matched later!
        log_timestamp = get_timestamp()
        self.__logger.log_frame_data(frame_id=log_timestamp, data=self.__tracked_data)

        self.__logger.log_image("eye_regions", "region", eye_region_bbox, log_timestamp)
        # self.__logger.log_image("eyes", "left_eye", left_eye_bbox, log_timestamp)
        # self.__logger.log_image("eyes", "right_eye", right_eye_bbox, log_timestamp)
        # self.__logger.log_image("eyes_small", "left_eye_small", left_eye_bbox_small, log_timestamp)
        # self.__logger.log_image("eyes_small", "right_eye_small", right_eye_bbox_small, log_timestamp)

    def log_information(self):
        print("Saving log data ...")
        blink_metrics = self.blink_detector.get_blink_metrics()
        self.__logger.log_blink_info(blink_metrics)

        # save the logs on a background thread so the ui won't freeze during this!
        self.save_log_thread = threading.Thread(target=self.__save_post_processing_data, name="SaveLogs", daemon=True)
        self.save_log_thread.start()

        # we still need to wait for it to finish as otherwise the new data would be added to the same list
        self.save_log_thread.join()

    def __save_post_processing_data(self):
        self.__logger.save_tracking_data()

    # def stop_tracking(self):
    #     self.__logger.stop_scheduling()

    def __get_eye_features(self):
        self.__eye_markers = np.take(self.__landmarks, self.face_alignment.eye_bound, axis=0)
        self.__left_eye = self.__eye_markers[1]
        self.__right_eye = self.__eye_markers[0]

        self.__eye_centers = np.average(self.__eye_markers, axis=1)
        # swap both eye centers as the left eye is actually the right one
        # (because the defined eye bounds are using the mirrored image)
        self.__eye_centers[[0, 1]] = self.__eye_centers[[1, 0]]
        if self.__debug:
            print(f"Eye markers: {self.__eye_markers}")
            print(f"Eye centers: {self.__eye_centers}")

        self.__find_pupils()
        self.__get_eye_sizes()

    def __find_pupils(self):
        eye_lengths = (self.__landmarks[[39, 93]] - self.__landmarks[[35, 89]])[:, 0]
        frame_copy = self.__current_frame.copy()

        self.__iris_left = self.iris_locator.get_mesh(frame_copy, eye_lengths[1], self.__eye_centers[0])
        pupil_left, self.__iris_left_radius = self.iris_locator.draw_pupil(
            self.__iris_left, frame_copy, annotations_on=self.__annotation_enabled, thickness=1)

        self.__iris_right = self.iris_locator.get_mesh(frame_copy, eye_lengths[0], self.__eye_centers[1])
        pupil_right, self.__iris_right_radius = self.iris_locator.draw_pupil(
            self.__iris_right, frame_copy, annotations_on=self.__annotation_enabled, thickness=1)

        self.__pupils = np.array([pupil_left, pupil_right])

        if self.__annotation_enabled:
            show_image_window(frame_copy, window_name="pupils", x_pos=1000, y_pos=50)

        if self.__debug:
            print(f"Pupil left: {pupil_left}")
            print(f"Pupil right: {pupil_right}")

    def __get_eye_sizes(self):
        self.__landmarks[[92, 38]] = self.__landmarks[[88, 34]] = self.__eye_centers

        self.__right_eye_height = self.__landmarks[33, 1] - self.__landmarks[40, 1]
        self.__right_eye_width = self.__landmarks[39, 0] - self.__landmarks[35, 0]
        self.__left_eye_height = self.__landmarks[87, 1] - self.__landmarks[94, 1]
        self.__left_eye_width = self.__landmarks[93, 0] - self.__landmarks[89, 0]

        if self.__debug:
            print(f"Left eye sizes: width: {self.__left_eye_width}, height: {self.__left_eye_height}")
            print(f"Right eye sizes: width: {self.__right_eye_width}, height: {self.__right_eye_height}")

    def __extract_eye_region(self):
        """
        Get the smallest squared frame that encompasses the whole eye region.
        """
        # find the outermost eye markers: i.e. the smallest and largest x and y values in the matrix
        min_vals = np.amin(self.__eye_markers, axis=1)
        min_x = np.min(min_vals[:, 0])
        min_y = np.min(min_vals[:, 1])
        max_vals = np.amax(self.__eye_markers, axis=1)
        max_x = np.max(max_vals[:, 0])
        max_y = np.max(max_vals[:, 1])

        # get the center point of the eye region by adding half of the eye region width
        # and height to the min x and y values; Important: as opencv needs pixels to
        # draw, we must provide integers which is why we use the floor division!
        region_center = (min_x + (max_x - min_x) // 2, min_y + (max_y - min_y) // 2)
        if self.__debug:
            print(f"Eye region center is at {region_center}")

        min_y_rect = int(round(min_y))
        max_y_rect = int(round(max_y))
        min_x_rect = int(round(min_x))
        max_x_rect = int(round(max_x))
        eye_ROI = extract_image_region(self.__current_frame, min_x_rect, min_y_rect, max_x_rect, max_y_rect, padding=20)
        return eye_ROI

    def __extract_eyes(self):
        """
        Get both eyes separately as squared images.
        """
        left_eye_center, right_eye_center = self.__eye_centers[0], self.__eye_centers[1]

        left_eye_x_min = int(round(left_eye_center[0] - self.__left_eye_width / 2))
        left_eye_x_max = int(round(left_eye_center[0] + self.__left_eye_width / 2))
        left_eye_y_min = int(round(left_eye_center[1] - self.__left_eye_height / 2))
        left_eye_y_max = int(round(left_eye_center[1] + self.__left_eye_height / 2))

        right_eye_x_min = int(round(right_eye_center[0] - self.__right_eye_width / 2))
        right_eye_x_max = int(round(right_eye_center[0] + self.__right_eye_width / 2))
        right_eye_y_min = int(round(right_eye_center[1] - self.__right_eye_height / 2))
        right_eye_y_max = int(round(right_eye_center[1] + self.__right_eye_height / 2))

        # extract region and add some padding so we get a little bit more; also make sure the extracted region +
        # padding still lies in the image bounding box
        padding = 10
        left_eye_box = extract_image_region(self.__current_frame, left_eye_x_min, left_eye_y_min, left_eye_x_max,
                                            left_eye_y_max, padding=padding)
        right_eye_box = extract_image_region(self.__current_frame, right_eye_x_min, right_eye_y_min, right_eye_x_max,
                                             right_eye_y_max, padding=padding)

        # use a mask to get only the eyes themselves
        frame_shape = (self.__current_frame.shape[0], self.__current_frame.shape[1])
        mask = np.zeros(frame_shape, dtype=np.uint8)  # black mask
        # mask = np.full(frame_shape, fill_value=255, dtype=np.uint8)  # white mask
        points = np.array(self.__eye_markers, np.int32)
        cv2.fillPoly(mask, points, 255)

        gray_frame = cv2.cvtColor(self.__current_frame, cv2.COLOR_BGR2GRAY)
        eye_mask = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
        if self.__annotation_enabled and eye_mask is not None:
            show_image_window(eye_mask, "Eye Mask", 320, 450)

        left_eye_box_small = extract_image_region(eye_mask, left_eye_x_min, left_eye_y_min, left_eye_x_max,
                                                  left_eye_y_max, padding=0)
        right_eye_box_small = extract_image_region(eye_mask, right_eye_x_min, right_eye_y_min,
                                                   right_eye_x_max, right_eye_y_max, padding=0)

        # TODO different value per person to work correctly!  -> probably needs to be done manually for each tp
        threshold_val = 18
        apply_threshold(left_eye_box_small, threshold_val=threshold_val, is_gray=True,
                        show_annotation=self.__annotation_enabled)

        # TODO: get diameter of the black region by finding the contours of the connected components!
        # see https://stackoverflow.com/questions/57317579/find-contiguous-black-pixels-in-image for an example

        # if self.__annotation_enabled and left_eye_box_small is not None:
        #     show_image_window(left_eye_box_small, "Left eyebox small", 450, 200)

        return left_eye_box, right_eye_box, left_eye_box_small, right_eye_box_small

    def __draw_face_landmarks(self):
        frame_copy = self.__current_frame.copy()  # make a copy so we don't edit the original frame
        for mark in self.__landmarks.reshape(-1, 2).astype(int):
            cv2.circle(frame_copy, tuple(mark), radius=1, color=(0, 0, 255), thickness=-1)
        show_image_window(frame_copy, window_name="face landmarks", x_pos=800, y_pos=350)

    def __track_gaze(self):
        # landmarks[[35, 89]] and landmarks[[39, 93]] are the start and end marks
        # (i.e. the leftmost and rightmost) for each eye
        poi = self.__landmarks[[35, 89]], self.__landmarks[[39, 93]], self.__pupils, self.__eye_centers
        theta, pha, delta = self.__calculate_3d_gaze(self.__current_frame, poi)

        if self.__yaw > 30:
            end_mean = delta[0]
        elif self.__yaw < -30:
            end_mean = delta[1]
        else:
            end_mean = np.average(delta, axis=0)

        if end_mean[0] < 0:
            zeta = arctan(end_mean[1] / end_mean[0]) + pi
        else:
            zeta = arctan(end_mean[1] / (end_mean[0] + 1e-7))

        if self.__roll < 0:
            self.__roll += 180
        else:
            self.__roll -= 180

        real_angle = zeta + self.__roll * pi / 180
        if self.__debug:
            print(f"Gaze angle: {real_angle}°")

        # calculate the norm of the vector (i.e. the length)
        R = norm(end_mean)
        offset = R * cos(real_angle), R * sin(real_angle)

        if self.__annotation_enabled:
            self.__draw_gaze(offset)

    def __calculate_gaze_point(self):
        """
        Algorithm below taken from this answer: https://stackoverflow.com/a/52922636/14345809
        """
        # Scaling factor
        scale_x_left = self.__screenWidth / self.__left_eye_width
        scale_y_left = self.__screenHeight / self.__left_eye_height
        scale_x_right = self.__screenWidth / self.__right_eye_width
        scale_y_right = self.__screenHeight / self.__right_eye_height

        # difference iris center - eye center (direction of iris and pupil)
        eye_center_deviation_x_left = self.__pupils[0][0] - self.__eye_centers[0][0]
        eye_center_deviation_y_left = self.__pupils[0][1] - self.__eye_centers[0][1]
        eye_center_deviation_x_right = self.__pupils[1][0] - self.__eye_centers[1][0]
        eye_center_deviation_y_right = self.__pupils[1][1] - self.__eye_centers[1][1]

        gaze_point_left_x = (self.__screenWidth / 2) + scale_x_left * eye_center_deviation_x_left
        gaze_point_left_y = (self.__screenHeight / 2) + scale_y_left * eye_center_deviation_y_left

        gaze_point_right_x = (self.__screenWidth / 2) + scale_x_right * eye_center_deviation_x_right
        gaze_point_right_y = (self.__screenHeight / 2) + scale_y_right * eye_center_deviation_y_right

        if gaze_point_left_x > self.__screenWidth or gaze_point_right_x > self.__screenWidth:
            print(f"Gaze points x are out of screen bounds! ({gaze_point_left_x, gaze_point_right_x}")
        if gaze_point_left_y > self.__screenHeight or gaze_point_right_y > self.__screenHeight:
            print(f"Gaze points y are out of screen bounds! ({gaze_point_left_y, gaze_point_right_y})")

        self.gaze_left = (gaze_point_left_x, gaze_point_left_y)
        print("Gaze left: ", self.gaze_left)
        self.gaze_right = (gaze_point_right_x, gaze_point_right_y)

    def get_gaze_points(self):
        return self.gaze_left, self.gaze_right

    def __calculate_3d_gaze(self, frame, poi, scale=256):
        SIN_LEFT_THETA = 2 * sin(pi / 4)
        SIN_UP_THETA = sin(pi / 6)

        starts, ends, pupils, centers = poi

        # swap pupils and eye centers back to their original format for the gaze direction calculation
        pupils[[0, 1]] = pupils[[1, 0]]
        centers[[0, 1]] = centers[[1, 0]]

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

        inv_judge = zc_distance ** 2 - delta_y ** 2 < eye_length ** 2 / 4

        delta[0, inv_judge] *= -1
        theta[inv_judge] *= -1
        delta *= scale

        if self.__annotation_enabled:
            # highlight pupil centers
            frame_copy = frame.copy()
            cv2.circle(frame_copy, tuple(pupils[0].astype(int)), 1, (0, 255, 255), -1)
            cv2.circle(frame_copy, tuple(pupils[1].astype(int)), 1, (0, 255, 255), -1)
            show_image_window(frame_copy, window_name="pupil centers", x_pos=600, y_pos=50)

        return theta, pha, delta.T

    def __draw_gaze(self, offset, blink_thd=0.22, arrow_color=(0, 125, 255)):
        # show gaze direction as arrows
        src = self.__current_frame.copy()

        # if self.__left_eye_height / self.__left_eye_width > blink_thd:
        cv2.arrowedLine(src, tuple(self.__pupils[0].astype(int)),
                        tuple((offset + self.__pupils[0]).astype(int)),
                        arrow_color, 2)
        # if self.__right_eye_height / self.__right_eye_width > blink_thd:
        cv2.arrowedLine(src, tuple(self.__pupils[1].astype(int)),
                        tuple((offset + self.__pupils[1]).astype(int)),
                        arrow_color, 2)
        show_image_window(src, window_name="gaze direction", x_pos=1000, y_pos=350)
