#!/usr/bin/python3
# -*- coding:utf-8 -*-

from scipy.spatial import distance as dist
from post_processing.ProcessingLogger import get_timestamp


# noinspection PyAttributeOutsideInit
class SaccadeFixationDetector:
    """
    This velocity-based saccade detection algorithm is based on the one described in:
    Stuart, S., Galna, B., Lord, S., Rochester, L., & Godfrey, A. (2014, August).
    Quantifying saccades while walking: validity of a novel velocity-based algorithm for mobile eye tracking.
    In 2014 36th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (pp.
    5739-5742). IEEE.
    """

    def __init__(self):
        self.t1 = None
        self.last_coords_left = None
        self.last_velocity = None
        self.fixation_count = 0
        self.saccade_count = 0

    def set_current_frame_vals(self, pupils):
        self.__pupils = pupils

    def find_saccades_fixations(self, debug=True):
        if self.t1 is None or self.last_coords_left is None:
            self.t1 = get_timestamp()
            self.last_coords_left = self.__pupils[0]
            return

        t2 = get_timestamp()
        dist_frames = dist.euclidean(self.last_coords_left, self.__pupils[0])
        time_frames = t2 - self.t1
        current_velocity = dist_frames / time_frames

        if self.last_velocity is not None:
            pixel_degrees_conversion_rate = 0.31
            acceleration = (self.last_velocity - current_velocity) / time_frames
            dist_in_degrees = dist_frames * pixel_degrees_conversion_rate
            current_velocity_degrees = dist_in_degrees / time_frames
            last_velocity_degrees = self.last_velocity * pixel_degrees_conversion_rate
            acceleration_degrees = (last_velocity_degrees - current_velocity_degrees) / time_frames

            if debug:
                print("dist_frames: ", dist_frames)
                print("time_frames: ", time_frames)
                print("current_velocity: ", current_velocity)
                print("dist_in_degrees: ", dist_in_degrees)
                print("current_velocity_degrees: ", current_velocity_degrees)
                print("acceleration_degrees: ", acceleration_degrees)

            # check if flicker happened; else skip this
            # TODO should also exclude blinks
            if current_velocity_degrees < 1000 and acceleration_degrees < 100.0:

                # saccades shouldn't take longer than 100 ms, so sth probably went wrong; discard it
                # for a 30 hz webcam 100ms would be 3 frames!
                # TODO this algorithm doesn't make any sense live -> needs to be done with the times of the captured
                #  frames AFTERWARDS
                if current_velocity_degrees > 240 and acceleration_degrees > 3 and time_frames < 100:
                    self.saccade_count += 1

                # fixations usually take longer than 100 ms, so sth probably went wrong if not; discard it
                if current_velocity_degrees < 240 and acceleration_degrees < 3 and time_frames > 100:
                    self.fixation_count += 1

            # TODO missing:
            # 1. group together all adjacent saccade and fixation frames that are found
            # 2. calculate saccade distance by summing up the distances of all adjacent saccade frames in [1]
            # 3. calculate peaks in velocity and distance at the end overall

        self.last_velocity = current_velocity
        self.t1 = get_timestamp()
        self.last_coords_left = self.__pupils[0]  # pupil left

        if debug:
            print("Saccades: ", self.saccade_count)
            print("Fixations: ", self.fixation_count)
