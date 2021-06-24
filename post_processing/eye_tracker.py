import cv2
from datetime import datetime
from plyer import notification
# import sounddevice as sd
# import soundfile as sf
import numpy as np
import pyautogui as pyautogui
from numpy import sin, cos, pi, arctan
from numpy.linalg import norm
from post_processing.ProcessingLogger import ProcessingLogger, LogData, get_timestamp
from post_processing.image_utils import preprocess_frame
from post_processing_service.blink_detector import BlinkDetector
# from post_processing_service.saccade_fixation_detector import SaccadeFixationDetector
from post_processing_service.face_alignment import CoordinateAlignmentModel
from tracking_service.face_detector import MxnetDetectionModel
from post_processing_service.head_pose import HeadPoseEstimator
from post_processing_service.iris_localization import IrisLocalizationModel


# noinspection PyAttributeOutsideInit
class EyeTracker:

    def __init__(self, video_width: int, video_height: int, debug_active=False,
                 enable_annotation=False, show_video=True, gpu_ctx=-1):
        self.__debug = debug_active
        self.__annotation_enabled = enable_annotation
        self.__show_video = show_video

        self.frame_count = 0
        self.t1 = None
        self.t2 = None

        self.__screenWidth, self.__screenHeight = pyautogui.size()
        self.gaze_left, self.gaze_right = None, None

        self.__init_logger()

        # self.saccade_detector = SaccadeFixationDetector()
        self.blink_detector = BlinkDetector()
        self.face_detector = MxnetDetectionModel("weights/16and32", 0, .6, gpu=gpu_ctx)
        self.face_alignment = CoordinateAlignmentModel('weights/2d106det', 0, gpu=gpu_ctx)
        # self.face_alignment = CoordinateAlignmentModel('weights/model-hg2d3-cab/model', 0, gpu=gpu_ctx)
        self.iris_locator = IrisLocalizationModel("../weights/iris_landmark.tflite")
        self.head_pose_estimator = HeadPoseEstimator("../weights/object_points.npy", video_width, video_height)

    def __init_logger(self):
        self.__logger = ProcessingLogger()
        # self.__tracked_data = dict.fromkeys(LogData, None)

        # use the name of the enum as dict key as otherwise it would always generate a new key the next time
        # and would therefore always append new columns to our pandas dataframe!
        self.__tracked_data = {key.name: None for key in LogData}

    def measure_frame_count(self):
        self.frame_count += 1
        print(f"########\nFrame {self.frame_count} at {datetime.now()}\n#######")
        if self.frame_count % 2 == 1:
            self.t1 = get_timestamp()
        elif self.frame_count % 2 == 0:
            self.t2 = get_timestamp()
            print(f"########\nTime between frames {(self.t2 - self.t1):.2f} seconds\n#######")

    # TODO the only things that could actually be necessary live would be resizing and grayscale
    # conversion for faster transfer
    def process_current_frame(self, frame: np.ndarray):
        """
        Args:
            frame: video frame in the format [width, height, channels]
        """

        # measure actual frame count per second with current implementation:
        # ca. 100 ms avg zwischen Frames; bei meiner meiner 30 FPS cam heißt das:
        # 1000/30 = 33.3ms => 33.3 + 100 = 133ms zwischen Frames! => 1000/133 ~= 7.5 frames pro sekunde!
        # self.measure_frame_count()

        # TODO überlegen, ob multiprocessing / parallelisieren irgendwo sinnvoll sein könnte! (frühestens nach der head
        #  pos); sinnvoll, womöglich blink und saccaden detection (aber wieder joinen befor logging)
        # use PoolProcessExecuter instead of multiprocessing module probably

        processed_frame = preprocess_frame(frame, kernel_size=3, keep_dim=True)
        self.__current_frame = processed_frame

        # eye_region_bbox = None

        bboxes = self.face_detector.detect(self.__current_frame)
        for landmarks in self.face_alignment.get_landmarks(self.__current_frame, bboxes, calibrate=True):
            self.__landmarks = landmarks
            # calculate head pose
            _, euler_angle = self.head_pose_estimator.get_head_pose(landmarks)
            self.__pitch, self.__yaw, self.__roll = euler_angle[:, 0]

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
            left_eye_bbox, right_eye_bbox = self.__extract_eyes()
            left_pupil_bbox, right_pupil_bbox = self.__extract_pupils()
            self.__log(eye_region_bbox, left_eye_bbox, right_eye_bbox, left_pupil_bbox, right_pupil_bbox)

            # find_pupil(left_eye_bbox, save=True)

        if self.__show_video:
            # show current annotated frame and save it as a png
            cv2.imshow('res', self.__current_frame)

        return self.__current_frame
        # if eye_region_bbox is not None:
        #     return resize_image(eye_region_bbox, size=150)
        # else:
        #     return None

    def __log(self, eye_region_bbox, left_eye_bbox, right_eye_bbox, left_pupil_bbox, right_pupil_bbox):
        # fill dict with all relevant data so we don't have to pass all params manually
        self.__tracked_data.update({
            LogData.HEAD_POS_ROLL_PITCH_YAW.name: (self.__roll, self.__pitch, self.__yaw),
            LogData.FACE_LANDMARKS.name: self.__landmarks,  # .tolist(),
            LogData.LEFT_EYE.name: self.__left_eye,
            LogData.RIGHT_EYE.name: self.__right_eye,
            LogData.LEFT_EYE_CENTER.name: self.__eye_centers[0],
            LogData.RIGHT_EYE_CENTER.name: self.__eye_centers[1],
            LogData.LEFT_EYE_WIDTH.name: self.__left_eye_width,
            LogData.RIGHT_EYE_WIDTH.name: self.__right_eye_width,
            LogData.LEFT_EYE_HEIGHT.name: self.__left_eye_height,
            LogData.RIGHT_EYE_HEIGHT.name: self.__right_eye_height,
            LogData.LEFT_PUPIL_POS.name: self.__pupils[0],
            LogData.RIGHT_PUPIL_POS.name: self.__pupils[1],
            LogData.LEFT_PUPIL_DIAMETER.name: 0,  # TODO wie pupillen durchmesser bestimmen?
            LogData.RIGHT_PUPIL_DIAMETER.name: 0,
        })
        # TODO count, avg and std of blinks, fixations, saccades (also duration and peak)
        # TODO wann loggen, pro Frame macht keinen Sinn!

        # save timestamp separately as it has to be the same for all the frames and the log data! otherwise it
        # can't be matched later!
        log_timestamp = get_timestamp()
        self.__logger.log_frame_data(frame_id=log_timestamp, data=self.__tracked_data)

        # TODO some images are not squared but one pixel larger in one dimension (fix later in postprocessing)

        # TODO resize images to larger ones before saving?
        # new_eye_region = improve_image(eye_region_bbox)
        # self.__logger.log_image("eye_regions_improved", "region", new_eye_region, log_timestamp)

        self.__logger.log_image("eye_regions", "region", eye_region_bbox, log_timestamp)
        self.__logger.log_image("eyes", "left_eye", left_eye_bbox, log_timestamp)
        self.__logger.log_image("eyes", "right_eye", right_eye_bbox, log_timestamp)
        self.__logger.log_image("pupils", "pupil_left", left_pupil_bbox, log_timestamp)
        self.__logger.log_image("pupils", "pupil_right", right_pupil_bbox, log_timestamp)

    def stop_tracking(self):
        self.__logger.stop_scheduling()

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

        self.__iris_left = self.iris_locator.get_mesh(self.__current_frame, eye_lengths[1], self.__eye_centers[0])
        pupil_left, self.__iris_left_radius = self.iris_locator.draw_pupil(
            self.__iris_left, self.__current_frame, annotations_on=self.__annotation_enabled, thickness=1
        )
        self.__iris_right = self.iris_locator.get_mesh(self.__current_frame, eye_lengths[0], self.__eye_centers[1])
        pupil_right, self.__iris_right_radius = self.iris_locator.draw_pupil(
            self.__iris_right, self.__current_frame, annotations_on=self.__annotation_enabled, thickness=1
        )
        self.__pupils = np.array([pupil_left, pupil_right])

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

        # calculate a squared bbox around the eye region as we need square images later for our CNN;
        # we consider only the region width as the eyes I know are usually far wider than large
        center_y = int(region_center[1])
        eye_region_width = max_x - min_x
        min_y_rect = int(round(center_y - eye_region_width / 2))
        max_y_rect = int(round(center_y + eye_region_width / 2))

        if self.__annotation_enabled:
            # visualize the squared eye region
            self.__highlight_eye_region(self.__current_frame, region_center, min_x, max_x, min_y, max_y)

        eye_ROI = self.__current_frame[min_y_rect: max_y_rect, int(round(min_x)): int(round(max_x))]
        return eye_ROI

    def __extract_eyes(self):
        """
        Get both eyes separately as squared images.
        """
        left_eye_center, right_eye_center = self.__eye_centers[0], self.__eye_centers[1]

        left_eye_x_min = int(round(left_eye_center[0] - self.__left_eye_width / 2))
        left_eye_x_max = int(round(left_eye_center[0] + self.__left_eye_width / 2))
        left_eye_y_min = int(round(left_eye_center[1] - self.__left_eye_width / 2))
        left_eye_y_max = int(round(left_eye_center[1] + self.__left_eye_width / 2))

        right_eye_x_min = int(round(right_eye_center[0] - self.__right_eye_width / 2))
        right_eye_x_max = int(round(right_eye_center[0] + self.__right_eye_width / 2))
        right_eye_y_min = int(round(right_eye_center[1] - self.__right_eye_width / 2))
        right_eye_y_max = int(round(right_eye_center[1] + self.__right_eye_width / 2))

        if self.__annotation_enabled:
            # draw rectangles around both eyes
            cv2.rectangle(self.__current_frame, (left_eye_x_min, left_eye_y_min),
                          (left_eye_x_max, left_eye_y_max), (0, 0, 255))
            cv2.rectangle(self.__current_frame, (right_eye_x_min, right_eye_y_min),
                          (right_eye_x_max, right_eye_y_max), (255, 0, 0))

        left_eye_box = self.__current_frame[left_eye_y_min: left_eye_y_max, left_eye_x_min: left_eye_x_max]
        right_eye_box = self.__current_frame[right_eye_y_min: right_eye_y_max, right_eye_x_min: right_eye_x_max]
        return left_eye_box, right_eye_box

    def __extract_pupils(self):
        """
        Get both pupils as separate (square-sized) images.
        """
        left_pupil, right_pupil = self.__pupils[0], self.__pupils[1]

        # calculate bboxes for both pupils separately
        left_pupil_left_x = int(left_pupil[0] - self.__iris_left_radius)
        left_pupil_right_x = int(left_pupil[0] + self.__iris_left_radius)
        left_pupil_min_y = int(left_pupil[1] - self.__iris_left_radius)
        left_pupil_max_y = int(left_pupil[1] + self.__iris_left_radius)

        right_pupil_left_x = int(right_pupil[0] - self.__iris_left_radius)
        right_pupil_right_x = int(right_pupil[0] + self.__iris_left_radius)
        right_pupil_min_y = int(right_pupil[1] - self.__iris_left_radius)
        right_pupil_max_y = int(right_pupil[1] + self.__iris_left_radius)

        if self.__annotation_enabled:
            # draw rectangles around both pupils
            cv2.rectangle(self.__current_frame, (left_pupil_left_x, left_pupil_min_y),
                          (left_pupil_right_x, left_pupil_max_y), (0, 120, 222), 2)
            cv2.rectangle(self.__current_frame, (right_pupil_left_x, right_pupil_min_y),
                          (right_pupil_right_x, right_pupil_max_y), (0, 120, 222), 2)

        left_pupil_bbox = self.__current_frame[left_pupil_min_y: left_pupil_max_y,
                                               left_pupil_left_x: left_pupil_right_x]
        right_pupil_bbox = self.__current_frame[right_pupil_min_y: right_pupil_max_y,
                                                right_pupil_left_x: right_pupil_right_x]
        return left_pupil_bbox, right_pupil_bbox

    def __draw_face_landmarks(self):
        for mark in self.__landmarks.reshape(-1, 2).astype(int):
            cv2.circle(self.__current_frame, tuple(mark), radius=1, color=(0, 0, 255), thickness=-1)

    def __highlight_eye_region(self, frame, region_center, min_x, max_x, min_y, max_y):
        # draw circle at eye region center (the middle point between both eyes)
        cv2.circle(frame, (int(region_center[0]), int(region_center[1])),
                   5, color=(0, 255, 0))
        # draw a rectangle around the whole eye region
        cv2.rectangle(frame, (min_x.astype(int), min_y.astype(int)),
                      (max_x.astype(int), max_y.astype(int)), (0, 255, 255), 3)

        # draw a square around the eye region
        center_y = int(region_center[1])
        eye_region_width = max_x - min_x
        min_y_rect = center_y - int(eye_region_width / 2)
        max_y_rect = center_y + int(eye_region_width / 2)
        cv2.rectangle(frame, (int(min_x), min_y_rect), (int(max_x), max_y_rect), (0, 222, 222), 2)

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

        # TODO
        # self.__watch_head_position()

        real_angle = zeta + self.__roll * pi / 180
        if self.__debug:
            print(f"Gaze angle: {real_angle}°")

        # calculate the norm of the vector (i.e. the length)
        R = norm(end_mean)
        offset = R * cos(real_angle), R * sin(real_angle)

        if self.__annotation_enabled:
            self.__draw_gaze(offset)

    def __watch_head_position(self):
        # TODO use euler angles to give warnings if the head pos changed too much
        #  -> only show one notification at a time (use boolean flag maybe)
        if not -15 <= self.__roll <= 15:
            notification.notify(title="Head Position wrong (Roll)!",
                                message="Bringe deinen Kopf in eine gerade, aufrechte Position!",
                                timeout=5)
            """
            # also play a short audio
            try:
                data, fs = sf.read("./asset/notify.wav", dtype='float32')
                sd.play(data, fs)
                status = sd.wait()
            except Exception as e:
                sys.stderr.write(type(e).__name__ + ': ' + str(e))
            """
        if not -10 <= self.__pitch <= 10:
            notification.notify(title="Head Position wrong (Pitch)!",
                                message="Bringe deinen Kopf in eine gerade, aufrechte Position!",
                                timeout=5)
        if not -20 <= self.__yaw <= 20:
            notification.notify(title="Head Position wrong (Yaw)!",
                                message="Bringe deinen Kopf in eine gerade, aufrechte Position!",
                                timeout=5)
        # print(f"Yaw: {self.__yaw}, Pitch: {self.__pitch}, Roll: {self.__roll}")

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
            cv2.circle(frame, tuple(pupils[0].astype(int)), 2, (0, 255, 255), -1)
            cv2.circle(frame, tuple(pupils[1].astype(int)), 2, (0, 255, 255), -1)

        return theta, pha, delta.T

    def __draw_gaze(self, offset, blink_thd=0.22, arrow_color=(0, 125, 255), copy=False):
        src = self.__current_frame
        if copy:
            src = src.copy()  # make a copy of the current frame

        # show gaze direction as arrows
        if self.__left_eye_height / self.__left_eye_width > blink_thd:
            cv2.arrowedLine(src, tuple(self.__pupils[0].astype(int)),
                            tuple((offset + self.__pupils[0]).astype(int)),
                            arrow_color, 2)

        if self.__right_eye_height / self.__right_eye_width > blink_thd:
            cv2.arrowedLine(src, tuple(self.__pupils[1].astype(int)),
                            tuple((offset + self.__pupils[1]).astype(int)),
                            arrow_color, 2)
        return src
