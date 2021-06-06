#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
import cv2
from utils.image_utils import eye_aspect_ratio


# TODO blink frequency and average blink duration und amplitude auch noch (mean unterschied zw. größten und
#  kleinstem)

# noinspection PyAttributeOutsideInit
class BlinkDetector:

    BLINK_COUNTER, TOTAL_BLINKS = 0, 0
    BLINK_COUNTER_NEW, TOTAL_BLINKS_NEW = 0, 0
    FRAME_COUNTER = 0

    def __init__(self):
        self.last_ratio = None
        self.onset = False
    
    def set_current_values(self, curr_frame, left_eye, right_eye, left_eye_size, right_eye_size):
        self.__current_frame = curr_frame
        self.__left_eye, self.__right_eye = left_eye, right_eye
        self.__left_eye_width, self.__left_eye_height = left_eye_size
        self.__right_eye_width, self.__right_eye_height = right_eye_size
    
    def detect_blinks(self, method=2):
        # TODO all methods work better with recorded (slower) video than live!
        if method == 0:
            self.__detect_blinks()
            # video1: 13 erkannt, 3 oder 4 Fehler; EAR_TRESH = 0.22,EAR_CONSEC_FRAMES = 3
            # video2: 3 erkannt; 6 Fehler (viel zu wenig)
            # live sehr schlecht, weil fast gar nichts erkannt (live müsste consec vermtl auf 1 sein)
        elif method == 1:
            self.__detect_blinks_alternative()
            # video1: 14 erkannt; 2 oder 3 Fehler; 3.7, 2
            # video2: 5 erkannt; 4 Fehler (paar zu wenig erkannt)
        elif method == 2:
            self.__detect_blinks_custom()
        else:
            sys.stderr.write("Wrong method parameter! Only values 0 and 1 are allowed!")

    def __reset_blink(self):
        self.onset = False
        BlinkDetector.FRAME_COUNTER = 0

    def __detect_blinks_custom(self, diff_threshold=0.05, max_frame_count=4):
        """Idea:
        Detect frame with blink onset (if eye ratio below a certain threshold or if it is much smaller than in the
        last frame); count frames until the eye ratio becomes bigger again (eye opens wider than a certain treshold)

        => if this happens in a short time (e.g. below 5 frames or 200-300 ms): **user blinked**!

        -> if it takes too long (blinks usually don't take longer than 300 ms): reset frame count and onset
        """
        leftEAR = eye_aspect_ratio(self.__left_eye)
        rightEAR = eye_aspect_ratio(self.__right_eye)
        ear = (leftEAR + rightEAR) / 2.0

        if self.last_ratio is None:
            # for the first frame we only set the current ratio and return immediately as we don't have a previous frame
            self.last_ratio = ear
            return

        cv2.putText(self.__current_frame, f"Blinks: {BlinkDetector.TOTAL_BLINKS}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(self.__current_frame, f"Onset: {self.onset}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(self.__current_frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(self.__current_frame, f"LAST EAR: {self.last_ratio:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # if difference between last and current ear is larger than a treshold, blink onset
        # if current ear is far smaller than the last one, the eye closed up
        if self.last_ratio - ear > diff_threshold:
            if self.onset is not True:
                self.onset = True

            # TODO here too?
            # if BlinkDetector.BLINK_COUNTER > max_frame_count:
                # self.__reset_blink()

            BlinkDetector.FRAME_COUNTER += 1
        # if current ear is far larger than the last one, the eye opened up
        elif ear - self.last_ratio > diff_threshold:
            if self.onset:
                # blink actually happened!
                BlinkDetector.TOTAL_BLINKS += 1
                self.__reset_blink()
        else:
            # some small eye size change happened between this and the last frame
            if BlinkDetector.FRAME_COUNTER > max_frame_count:
                # this is taking too long; was probably just noise, so reset
                self.__reset_blink()
            elif self.onset:
                # if onset is currently active, increment counter
                BlinkDetector.FRAME_COUNTER += 1

        self.last_ratio = ear

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

    def __detect_blinks_alternative(self):
        blinking_treshold = 3.0  # 3.7 for recorded video
        frame_consec_count = 3

        left_eye_ratio = self.__left_eye_width / self.__left_eye_height
        right_eye_ratio = self.__right_eye_width / self.__right_eye_height
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        cv2.putText(self.__current_frame, "Blinks: {}".format(BlinkDetector.TOTAL_BLINKS), (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(self.__current_frame, "New Blinks: {}".format(BlinkDetector.TOTAL_BLINKS_NEW), (5, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(self.__current_frame, "Ratio: {:.2f}".format(blinking_ratio), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if left_eye_ratio > blinking_treshold and right_eye_ratio > blinking_treshold:
            BlinkDetector.BLINK_COUNTER_NEW += 1
        else:
            if BlinkDetector.BLINK_COUNTER_NEW >= frame_consec_count:
                BlinkDetector.TOTAL_BLINKS_NEW += 1
            BlinkDetector.BLINK_COUNTER_NEW = 0

        if blinking_treshold > blinking_treshold:
            BlinkDetector.BLINK_COUNTER += 1
        else:
            if BlinkDetector.BLINK_COUNTER >= frame_consec_count:
                BlinkDetector.TOTAL_BLINKS += 1
            BlinkDetector.BLINK_COUNTER = 0
