import cv2
from post_processing.eye_tracking.image_utils import eye_aspect_ratio, show_image_window
import numpy as np


# noinspection PyAttributeOutsideInit
class BlinkDetector:

    # The average blink duration of an adult varies between studies but 100-400 ms is usually a good value, see
    # https://bionumbers.hms.harvard.edu/bionumber.aspx?s=y&id=100706&ver=0 and
    # Kwon, K. A., Shipley, R. J., Edirisinghe, M., Ezra, D. G., Rose, G., Best, S. M., & Cameron, R. E. (2013).
    # High-speed camera characterization of voluntary eye blinking kinematics. Journal of the Royal Society Interface,
    # 10(85), 20130227.
    BLINK_MAX_DURATION = 250  # in ms

    def __init__(self, show_annotation: bool):
        self.__show_annotation = show_annotation
        self.last_ratio = None
        self.onset = False

        self.blink_counter, self.frame_count, self.consecutive_frames_counter, self.total_blinks = 0, 0, 0, 0

        self.eye_aspect_ratios = []
        self.blink_durations = []

    def set_current_values(self, curr_frame, left_eye, right_eye, left_eye_size, right_eye_size):
        self.__current_frame = curr_frame
        self.__left_eye, self.__right_eye = left_eye, right_eye
        self.__left_eye_width, self.__left_eye_height = left_eye_size
        self.__right_eye_width, self.__right_eye_height = right_eye_size

    def set_participant_fps(self, fps_val):
        self.__participant_fps = fps_val
        # calculate the maximal frame count (i.e. the max blink duration) based on the user's camera fps
        frames_per_blink_duration = (self.BLINK_MAX_DURATION / 1000) * self.__participant_fps
        self.__max_frame_count = int(round(frames_per_blink_duration))

        # print("Current fps in blink detector: ", fps_val)
        # print("max_frame_count in blink detector: ", self.__max_frame_count)

    def get_blink_metrics(self):
        duration_in_minutes = (self.frame_count / self.__participant_fps) / 60
        blinks_per_minute = self.total_blinks / duration_in_minutes

        # TODO check if these values can be confirmed!
        # -> "averaging around 10 blinks per minute in a laboratory setting. [...] when the eyes are focused [...] the
        # rate of blinking decreases to about 3 to 4 times per minute" (see https://en.wikipedia.org/wiki/Blinking)

        return {"total_blinks": self.total_blinks,
                "avg_blinks_per_minute": blinks_per_minute,
                "min_aspect_ratio": min(self.eye_aspect_ratios),
                "max_aspect_ratio": max(self.eye_aspect_ratios),
                "avg_aspect_ratio": np.average(self.eye_aspect_ratios),
                "min_blink_duration_in_ms": min(self.blink_durations),
                "max_blink_duration_in_ms": max(self.blink_durations),
                "avg_blink_duration_in_ms": np.average(self.blink_durations),
                }

    def detect_blinks(self):
        # self.__detect_blinks()
        self.__detect_blinks_custom()

    def __reset_blink(self):
        self.onset = False
        self.consecutive_frames_counter = 0

    def __detect_blinks_custom(self, diff_threshold=0.05):
        """
        Detect frame with blink onset (if eye aspect ratio (EAR) below a certain threshold or if it is much smaller
        than in the last frame); count frames until the eye ratio becomes bigger again (eye opens wider than a
        certain threshold)
        => if this happens in a short time (e.g. 200-300 ms): ** the user blinked **!
        -> if it takes too long (blinks usually don't take longer than 400 ms): reset frame count and onset

        Args:
            diff_threshold: the threshold for the EAR difference between two frames that must be reached to switch the
                            onset from True to False or vice versa; determined by testing with several participants
        """
        self.frame_count += 1

        # calculate the EAR for both eyes
        leftEAR = eye_aspect_ratio(self.__left_eye)
        rightEAR = eye_aspect_ratio(self.__right_eye)
        # Take the average eye aspect ratio of both eyes as a better and more stable estimate,
        # see Cech, J., & Soukupova, T. (2016). Real-time eye blink detection using facial landmarks.
        # Cent. Mach. Perception, Dep. Cybern. Fac. Electr. Eng. Czech Tech. Univ. Prague, 1-8.
        # Note: this assumes that a person blinks with both eyes at the same time!
        # -> if the user blinks with one eye only this might not be detected
        ear = (leftEAR + rightEAR) / 2.0
        self.eye_aspect_ratios.append(ear)

        if self.last_ratio is None:
            # for the first frame we only set the current ratio and return immediately as we don't have a previous frame
            self.last_ratio = ear
            return

        if self.__show_annotation:
            # show the current number of detected blinks and the EAR values on a copy of the current frame
            frame_copy = self.__current_frame.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame_copy, f"Blinks: {self.total_blinks}", (10, 20), font, 0.6, (0, 0, 255), 2)
            cv2.putText(frame_copy, f"Onset: {self.onset}", (10, 50), font, 0.5, (0, 0, 255), 2)
            cv2.putText(frame_copy, f"EAR: {ear:.2f}", (10, 150), font, 0.5, (0, 0, 255), 2)
            cv2.putText(frame_copy, f"LAST EAR: {self.last_ratio:.2f}", (10, 180), font, 0.5, (0, 0, 255), 2)
            show_image_window(frame_copy, window_name="blink detection", x_pos=600, y_pos=350)

        if self.last_ratio - ear > diff_threshold:
            # If the current EAR is far smaller than the last one (i.e. the difference is greater than the specified
            # threshold), the eye is closing, so we register a possible blink onset
            if not self.onset:
                self.onset = True

            # count the number of frames that have passed since the blink onset
            self.consecutive_frames_counter += 1

        elif ear - self.last_ratio > diff_threshold:
            # If the current EAR is far larger than the last one, the eye is widening
            if self.onset:
                # if we registered a blink onset before this (i.e. before the eye opens up), a blink occurred!
                self.total_blinks += 1

                # the blink duration is the number of frames needed for the blink times the frame duration for this user
                blink_duration = self.consecutive_frames_counter * (1000 / self.__participant_fps)
                self.blink_durations.append(blink_duration)
                self.__reset_blink()

        elif self.onset:
            # only some small eye size change happened between this and the last frame;
            # check if we reached the maximum number of frames, i.e. the duration of a blink
            if self.consecutive_frames_counter > self.__max_frame_count:
                # this is taking too long; was probably just some noise, so we reset the counter
                self.__reset_blink()
            else:
                # if the blink onset is still active and nothing happened, simply increment the counter
                self.consecutive_frames_counter += 1

        self.last_ratio = ear

    """
    def __detect_blinks(self):
        EAR_TRESH = 0.22
        EAR_CONSEC_FRAMES = 3

        leftEAR = eye_aspect_ratio(self.__left_eye)
        rightEAR = eye_aspect_ratio(self.__right_eye)
        ear = (leftEAR + rightEAR) / 2.0

        frame_copy = self.__current_frame.copy()
        cv2.putText(frame_copy, "Blinks: {}".format(self.TOTAL_BLINKS), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame_copy, "EAR: {:.2f}".format(ear), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("blink detector", frame_copy)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EAR_TRESH:
            # print(f"The user blinked {self.TOTAL_BLINKS} times!")
            self.blink_counter += 1
        # otherwise, the eye aspect ratio is not below the blink threshold
        else:
            # if the eyes were closed for a sufficient number of frames then increment the total number of blinks
            if self.blink_counter >= EAR_CONSEC_FRAMES:
                self.TOTAL_BLINKS += 1
            # reset the eye frame counter
            self.blink_counter = 0
    """
