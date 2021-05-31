#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
import cv2
from image_utils import eye_aspect_ratio
from scipy.spatial import distance as dist


# noinspection PyAttributeOutsideInit
class BlinkDetector:

    BLINK_COUNTER, TOTAL_BLINKS = 0, 0
    
    def set_current_values(self, curr_frame, left_eye, right_eye, left_eye_size, right_eye_size):
        self.__current_frame = curr_frame
        self.__left_eye, self.__right_eye = left_eye, right_eye
        self.__left_eye_width, self.__left_eye_height = left_eye_size
        self.__right_eye_width, self.__right_eye_height = right_eye_size
    
    def detect_blinks(self, method=2):
        if method == 0:
            self.__detect_blinks()
        elif method == 1:
            self.detect_blinks_alternative()
        elif method == 2:
            self.detect_blinks_2()
        elif method == 3:
            self.detect_blinks_3()
        elif method == 4:
            self.detect_blinks_4()
        else:
            sys.stderr.write("Wrong method parameter! Only values between 0 and 4 are allowed!")

    # TODO blink frequency and average blink duration und amplitude auch noch (mean unterschied zw. größten und
    #  kleinstem)
    def __detect_blinks(self):
        EAR_TRESH = 0.28
        EAR_CONSEC_FRAMES = 3

        leftEAR = eye_aspect_ratio(self.__left_eye)
        rightEAR = eye_aspect_ratio(self.__right_eye)

        # average the eye aspect ratio together for both eyes for better estimate,
        # see Cech, J., & Soukupova, T. (2016). Real-time eye blink detection using facial landmarks.
        # Cent. Mach. Perception, Dep. Cybern. Fac. Electr. Eng. Czech Tech. Univ. Prague, 1-8.
        # Note: this assumes that a person blinks with both eyes at the same time!
        # -> if the user blinks with one eye only this might not be detected
        ear = (leftEAR + rightEAR) / 2.0

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EAR_TRESH:
            BlinkDetector.BLINK_COUNTER += 1
        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if BlinkDetector.BLINK_COUNTER >= EAR_CONSEC_FRAMES:
                BlinkDetector.TOTAL_BLINKS += 1
            # reset the eye frame counter
            BlinkDetector.BLINK_COUNTER = 0

        cv2.putText(self.__current_frame, "Blinks: {}".format(BlinkDetector.TOTAL_BLINKS), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(self.__current_frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        print(f"\n##############\nBlink Counter: {BlinkDetector.TOTAL_BLINKS}\n##############")

    def detect_blinks_alternative(self):
        A_left = dist.euclidean(self.__left_eye[1], self.__left_eye[7])
        B_left = dist.euclidean(self.__left_eye[2], self.__left_eye[6])
        C_left = dist.euclidean(self.__left_eye[3], self.__left_eye[5])
        A_right = dist.euclidean(self.__right_eye[1], self.__right_eye[7])
        B_right = dist.euclidean(self.__right_eye[2], self.__right_eye[6])
        C_right = dist.euclidean(self.__right_eye[3], self.__right_eye[5])

        eye_treshold = 6.8
        eye_consec_count = 2

        # TODO only take the middle ones, i.e. B (as they are the most interesting?)

        # if A_left <= eye_treshold or B_left <= eye_treshold or C_left <= eye_treshold:
        if (A_left <= eye_treshold and C_left <= eye_treshold) or (
                A_right <= eye_treshold and C_right <= eye_treshold):
            BlinkDetector.BLINK_COUNTER += 1
        else:
            if BlinkDetector.BLINK_COUNTER >= eye_consec_count:
                BlinkDetector.TOTAL_BLINKS += 1
            BlinkDetector.BLINK_COUNTER = 0

        cv2.putText(self.__current_frame, "Blinks: {}".format(BlinkDetector.TOTAL_BLINKS), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def detect_blinks_2(self):
        B_left = dist.euclidean(self.__left_eye[2], self.__left_eye[6])
        B_right = dist.euclidean(self.__right_eye[2], self.__right_eye[6])

        eye_treshold = 9.2
        eye_consec_count = 2

        # only take the middle ones this time, i.e. B as they are the most interesting for this
        # if (B_left + B_right) / 2 <= eye_treshold:
        if B_left <= eye_treshold or B_right <= eye_treshold:
            BlinkDetector.BLINK_COUNTER += 1
        else:
            if BlinkDetector.BLINK_COUNTER >= eye_consec_count:
                BlinkDetector.TOTAL_BLINKS += 1
            BlinkDetector.BLINK_COUNTER = 0

        cv2.putText(self.__current_frame, "Blinks: {}".format(BlinkDetector.TOTAL_BLINKS), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def detect_blinks_3(self):
        blink_thd = 0.17
        consec_frames = 2

        if self.__left_eye_height / self.__left_eye_width <= blink_thd:
            print(f"The user blinked with his left eye!")
            BlinkDetector.BLINK_COUNTER += 1
        elif self.__right_eye_height / self.__right_eye_width <= blink_thd:
            print(f"The user blinked with his right eye!")
            BlinkDetector.BLINK_COUNTER += 1
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if BlinkDetector.BLINK_COUNTER >= consec_frames:
                BlinkDetector.TOTAL_BLINKS += 1
            # reset the eye frame counter
            BlinkDetector.BLINK_COUNTER = 0

        cv2.putText(self.__current_frame, "Blinks: {}".format(BlinkDetector.TOTAL_BLINKS), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def detect_blinks_4(self):
        blinking_treshold = 3.7  # 3.8
        frame_consec_count = 2

        left_eye_ratio = self.__left_eye_width / self.__left_eye_height
        right_eye_ratio = self.__right_eye_width / self.__right_eye_height
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > blinking_treshold:
            print(f"The user blinked {BlinkDetector.TOTAL_BLINKS} times!")
            BlinkDetector.BLINK_COUNTER += 1
        else:
            if BlinkDetector.BLINK_COUNTER >= frame_consec_count:
                BlinkDetector.TOTAL_BLINKS += 1
            BlinkDetector.BLINK_COUNTER = 0

        cv2.putText(self.__current_frame, "Blinks: {}".format(BlinkDetector.TOTAL_BLINKS), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
