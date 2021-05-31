#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
import cv2
from image_utils import eye_aspect_ratio


# TODO blink frequency and average blink duration und amplitude auch noch (mean unterschied zw. größten und
#  kleinstem)

# noinspection PyAttributeOutsideInit
class BlinkDetector:

    BLINK_COUNTER, TOTAL_BLINKS = 0, 0
    
    def set_current_values(self, curr_frame, left_eye, right_eye, left_eye_size, right_eye_size):
        self.__current_frame = curr_frame
        self.__left_eye, self.__right_eye = left_eye, right_eye
        self.__left_eye_width, self.__left_eye_height = left_eye_size
        self.__right_eye_width, self.__right_eye_height = right_eye_size
    
    def detect_blinks(self, method=1):
        # TODO all methods work better with recorded (slower) video than live!
        if method == 0:
            self.__detect_blinks()
            # video1: 13 erkannt, 3 oder 4 Fehler; EAR_TRESH = 0.22,EAR_CONSEC_FRAMES = 3
            # video2: 3 erkannt; 6 Fehler (viel zu wenig)
            # live sehr schlecht, weil fast gar nichts erkannt (live müsste consec vermtl auf 1 sein)
        elif method == 1:
            self.__detect_blinks_alt()
            # video1: 14 erkannt; 2 oder 3 Fehler; 3.7, 2
            # video2: 5 erkannt; 4 Fehler (paar zu wenig erkannt)
        else:
            sys.stderr.write("Wrong method parameter! Only values 0 and 1 are allowed!")

    def __detect_blinks(self):
        # TODO does only work ok with recorded video
        EAR_TRESH = 0.22
        EAR_CONSEC_FRAMES = 3

        leftEAR = eye_aspect_ratio(self.__left_eye)
        rightEAR = eye_aspect_ratio(self.__right_eye)

        # average the eye aspect ratio together for both eyes for better estimate,
        # see Cech, J., & Soukupova, T. (2016). Real-time eye blink detection using facial landmarks.
        # Cent. Mach. Perception, Dep. Cybern. Fac. Electr. Eng. Czech Tech. Univ. Prague, 1-8.
        # Note: this assumes that a person blinks with both eyes at the same time!
        # -> if the user blinks with one eye only this might not be detected
        ear = (leftEAR + rightEAR) / 2.0

        cv2.putText(self.__current_frame, "Blinks: {}".format(BlinkDetector.TOTAL_BLINKS), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(self.__current_frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EAR_TRESH:
            print(f"The user blinked {BlinkDetector.TOTAL_BLINKS} times!")
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

    def __detect_blinks_alt(self):
        blinking_treshold = 2.9  # 3.7 for recorded video
        frame_consec_count = 2

        left_eye_ratio = self.__left_eye_width / self.__left_eye_height
        right_eye_ratio = self.__right_eye_width / self.__right_eye_height
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        cv2.putText(self.__current_frame, "Blinks: {}".format(BlinkDetector.TOTAL_BLINKS), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(self.__current_frame, "Ratio: {:.2f}".format(blinking_ratio), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if blinking_ratio > blinking_treshold:
            print(f"The user blinked {BlinkDetector.TOTAL_BLINKS} times!")
            BlinkDetector.BLINK_COUNTER += 1
        else:
            if BlinkDetector.BLINK_COUNTER >= frame_consec_count:
                BlinkDetector.TOTAL_BLINKS += 1
            BlinkDetector.BLINK_COUNTER = 0
