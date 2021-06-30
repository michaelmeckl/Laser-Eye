#!/usr/bin/python3
# -*- coding:utf-8 -*-

import math
import time
from datetime import datetime
import platform
import sys
import threading
from pathlib import Path
from tracking_utils import find_face_mxnet_resized
import cv2   # pip install opencv-python
import numpy as np
import psutil
from gpuinfo.windows import get_gpus  # pip install gpu-info
import pyautogui
from PyQt5.QtCore import Qt
from plyer import notification
from tracking_service.face_detector import MxnetDetectionModel
from TrackingLogger import TrackingData, get_timestamp
from TrackingLogger import Logger as TrackingLogger
from FpsMeasuring import FpsMeasurer
import keyboard
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMessageBox
# from ThreadedWebcamCapture import WebcamStream


class TrackingSystem(QtWidgets.QWidget):

    def __init__(self, debug_active=True):
        super(TrackingSystem, self).__init__()
        self.__tracking_active = False
        self.progress = None  # the upload progress
        self.debug = debug_active

        self.__current_frame = None
        self.frame_count = 0
        self.t1 = None
        self.t2 = None

        # use a custom threaded video capture to increase fps;
        # see https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
        # self.capture = WebcamStream(src=0)

        # necessary for building the exe file with pyinstaller with the --one-file option as the path changes;
        # see https://stackoverflow.com/questions/7674790/bundling-data-files-with-pyinstaller-onefile for more
        if getattr(sys, 'frozen', False):
            # change the path if we are executing the exe file; the lint warning here can be ignored ;)
            folder = Path(sys._MEIPASS)
            data_path = folder/'weights/16and32'
        else:
            folder = Path(__file__).parent
            data_path = folder/'../weights/16and32'

        self.face_detector = MxnetDetectionModel(data_path, 0, .6, gpu=-1)

        self.__setup_gui()
        self.__init_logger()

        if self.debug:
            self.fps_measurer = FpsMeasurer()
            # capture_fps = self.get_stream_fps()
            # self.fps_measurer.show_optimal_fps(capture_fps)

    def __setup_gui(self):
        self.layout = QtWidgets.QVBoxLayout()

        # show status of tracking
        self.tracking_status = QtWidgets.QLabel(self)
        self.tracking_status.setContentsMargins(0, 10, 0, 10)
        self.tracking_status.setAlignment(Qt.AlignCenter)
        self.__set_tracking_status_ui()
        self.layout.addWidget(self.tracking_status)

        # show some instructions
        self.label = QtWidgets.QLabel(self)
        self.label.setText(
            "Verwendung:\n\nMit Ctrl + Shift + A kann das Tracking gestartet und mit Ctrl + Shift + Q wieder "
            "gestoppt werden.\n\nDieses Fenster muss nach Beginn der Studie solange geöffnet bleiben, bis der "
            "Hochladevorgang beendet ist (100% auf dem Fortschrittsbalken unterhalb). "
            "Abhängig von der Internetgeschwindigkeit und Leistung des Rechners kann dies einige Zeit in "
            "Anspruch nehmen! Während dieser Zeit muss der Rechner eingeschaltet bleiben!"
        )
        self.label.setContentsMargins(10, 10, 10, 10)
        self.label.setWordWrap(True)
        self.label.setStyleSheet("QLabel {font-size: 9pt;}")
        self.layout.addWidget(self.label)

        # show a progress bar for the upload
        self.__setup_progress_bar()

        self.label_general = QtWidgets.QLabel(self)
        self.label_general.setText("Hinweise:")
        self.label_general.setStyleSheet("QLabel {font-size: 10pt; margin-top: 40px;}")
        self.layout.addWidget(self.label_general)

        self.general_instructions = QtWidgets.QTextEdit(self)
        self.general_instructions.setStyleSheet("QTextEdit {font-size: 9pt; background-color : rgba(0, 0, 0, 0%); "
                                                "border: 0; margin-top: 20px}")
        self.general_instructions.setReadOnly(True)  # don't let user edit this text field
        self.general_instructions.setHtml(
            "<ul>"
            "<li>Bitte versuchen Sie während das Tracking aktiv ist, <b>möglichst ruhig zu sitzen</b> und den Kopf "
            "nicht zu viel oder zu schnell zu bewegen (normale Kopfbewegungen sind selbstverständlich in Ordnung).</li>"
            "<li>Bitte tragen Sie während des Trackings <b>keine Brille</b>, da die dabei auftretenden Reflexionen "
            "ein Problem bei der Verarbeitung der Daten darstellen!</li>"
            "<li>Versuchen Sie <b>nicht zu weit weg von der Kamera</b> zu sein. Der Abstand zwischen Kamera und "
            "Gesicht sollte nicht mehr als 60-70 cm betragen.</li>"
            "<li>Die Kamera sollte beim Tracking möglichst gerade und frontal zum Gesicht positioniert sein, sodass "
            "das gesamte Gesicht von der Kamera erfasst werden kann.</li>"
            "<li>Entfernen Sie vor Beginn des Trackings bitte etwaige Webcam Abdeckungen!</li>"
            "<li>Die zu Beginn genannten Hotkeys zum Starten und Stoppen funktionieren nur einmal! Nach Stoppen des "
            "Trackings muss das Programm erneut gestartet werden, um das Tracking wieder zu starten!</li>"
            "</ul>"
        )
        self.layout.addWidget(self.general_instructions)

        # add buttons to manually start and stop tracking
        # self.__setup_button_layout()  # TODO only for debugging!

        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.layout)
        self.setGeometry(100, 100, 1000, 800)
        self.setWindowTitle("Tracking System")

    def __setup_button_layout(self):
        self.start_button = QtWidgets.QPushButton(self)
        self.start_button.setText("Start tracking")
        self.start_button.setStyleSheet("QPushButton {background-color: rgb(87, 205, 0); color: black; "
                                        "padding: 10px 10px 10px 10px; border-radius: 2px;}")
        self.start_button.clicked.connect(self.__activate_tracking)  # connect start method to this button

        self.stop_button = QtWidgets.QPushButton(self)
        self.stop_button.setText("Stop tracking")
        self.stop_button.setStyleSheet("QPushButton {background-color: rgb(153, 25, 25); color: white; "
                                       "padding: 10px 10px 10px 10px; border-radius: 2px;}")
        self.stop_button.clicked.connect(self.__stop_tracking)  # connect stop method to this button

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.start_button, alignment=Qt.AlignLeft)
        button_layout.addWidget(self.stop_button, alignment=Qt.AlignRight)
        self.layout.addLayout(button_layout)

    def __setup_progress_bar(self):
        # show a progressbar for the upload
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        progress_bar_layout = QtWidgets.QHBoxLayout()
        progress_bar_layout.addStretch(1)
        progress_bar_layout.addWidget(self.progress_bar, stretch=7)
        progress_bar_layout.addStretch(1)
        progress_bar_layout.setContentsMargins(0, 20, 0, 0)
        self.layout.addLayout(progress_bar_layout)

        # show how much files have been uploaded
        self.label_current = QtWidgets.QLabel(self)
        self.label_all = QtWidgets.QLabel(self)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.label_current)
        hbox.addWidget(self.label_all)
        hbox.setAlignment(Qt.AlignCenter)
        hbox.setSpacing(5)
        self.layout.addLayout(hbox)

        # show approximate time for the upload
        self.label_eta = QtWidgets.QLabel(self)
        self.label_eta.setAlignment(Qt.AlignCenter)
        self.label_eta.setStyleSheet("font-weight: bold")
        self.layout.addWidget(self.label_eta)

    def __set_tracking_status_ui(self):
        if self.__tracking_active:
            self.tracking_status.setText("Tracking aktiv")
            self.tracking_status.setStyleSheet("QLabel {color: green;}")
        else:
            self.tracking_status.setText("Tracking nicht aktiv")
            self.tracking_status.setStyleSheet("QLabel {color: red;}")

    def __init_logger(self):
        self.__logger = TrackingLogger(self.__on_upload_progress)
        self.__tracked_data = {key.name: None for key in TrackingData}

    def __on_upload_progress(self, current, overall):
        if overall == 0:
            return

        if current != 0:
            seconds_per_frame = (time.time() - self.__upload_start) / current
            eta_seconds = (overall - current) * seconds_per_frame
            minutes = math.floor(eta_seconds / 60)
            seconds = round(eta_seconds % 60)
            self.label_eta.setText(f"Verbliebene Zeit: {minutes} min, {seconds} Sekunden")
        else:
            self.label_eta.setText(f"Mehr Daten werden benötigt ...")

        self.label_current.setText(str(current))
        self.label_all.setText(f"/ {overall}")
        self.progress = int((current / overall) * 100)

        # TODO remove later:
        if current in [30, 100, 500, 1200]:
            needed_time = time.time() - self.__upload_start
            # print(f"Time needed to upload {current} images: {needed_time:.3f} seconds")

        self.progress_bar.setValue(self.progress)

    def listen_for_hotkey(self, hotkey_start="ctrl+shift+a", hotkey_stop="ctrl+shift+q"):
        keyboard.add_hotkey(hotkey_start, self.__activate_tracking, suppress=False, trigger_on_release=False)
        keyboard.add_hotkey(hotkey_stop, self.__stop_tracking, suppress=False, trigger_on_release=False)

    def __activate_tracking(self):
        # activate tracking on hotkey press
        if not self.__tracking_active:
            # start tracking on a background thread
            self.__tracking_active = True
            self.__set_tracking_status_ui()
            # TODO enable again:
            notification.notify(title="Tracking aktiv", message="Das Tracking wurde gestartet!", timeout=1)
            # self.capture.start()  # start reading frames from webcam
            self.tracking_thread = threading.Thread(target=self.__start_tracking, name="TrackingThread", daemon=True)
            self.tracking_thread.start()
        else:
            notification.notify(title="Tracking kann nicht gestartet werden!", message="Das Tracking läuft bereits!",
                                timeout=2)

    def __start_tracking(self):
        """
        This function runs on a background thread.
        """
        self.capture = cv2.VideoCapture(0)
        self.__log_static_data()
        self.__logger.start_saving_images_to_disk()  # start saving webcam frames to disk
        self.__logger.start_async_upload()  # start uploading data to sftp server

        self.__upload_start = time.time()
        if self.debug:
            self.fps_measurer.start()

        while self.__tracking_active:
            # read the next frame from the webcam
            return_val, frame = self.capture.read()
            if not return_val:
                sys.stderr.write("Unknown error while trying to get current frame!")
                break
            if frame is None:
                print("Frame from webcam is None!")
                continue

            processed_frame = self.__process_frame(frame)
            if processed_frame is None:
                continue
            self.__current_frame = processed_frame
            # self.__measure_frame_count()  # TODO delete
            if self.debug:
                self.fps_measurer.update()
                cv2.putText(processed_frame, f"current FPS: {self.fps_measurer.get_current_fps():.3f}",
                            (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("current_frame", processed_frame)
                # read until the video is finished or if no video was provided until the
                # user presses 'q' on the keyboard;
                # replace 1 with 0 to step manually through the video "frame-by-frame"
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.capture.release()  # cleanup the webcam capture

    def __measure_frame_count(self):
        self.frame_count += 1
        print(f"########\nFrame {self.frame_count} at {datetime.now()}\n#######")
        if self.frame_count % 2 == 1:
            self.t1 = get_timestamp()
        elif self.frame_count % 2 == 0:
            self.t2 = get_timestamp()
            print(f"########\nTime between frames {(self.t2 - self.t1):.2f} seconds\n#######")

    def __process_frame(self, frame: np.ndarray) -> np.ndarray:
        face_image = find_face_mxnet_resized(self.face_detector, frame)
        # scaled = scale_image(frame, scale_factor=0.5, show_scaled=True)
        # face_image = to_gray(face_image)

        if face_image is not None:
            # save timestamp separately as it has to be the same for all the frames and the log data! otherwise it
            # can't be matched later!
            log_timestamp = get_timestamp()
            self.__logger.add_image_to_queue("capture", face_image, log_timestamp)
        return frame

    # TODO in der einverständniserklärung unbedingt mit angeben, dass das auch geloggt wird!
    def __log_static_data(self):
        # get the dimensions of the webcam
        video_width, video_height = self.get_stream_dimensions()
        # get the dimensions of the primary monitor.
        screenWidth, screenHeight = pyautogui.size()

        system_info = platform.uname()._asdict()
        ram_info = psutil.virtual_memory()._asdict()
        cpu_current_freq, _, cpu_max_freq = psutil.cpu_freq()  # in Mhz
        try:
            gpu_name = [x.name for x in get_gpus()]
        except Exception:
            gpu_name = ["no gpu"]

        # noinspection PyProtectedMember
        self.__tracked_data.update({
            TrackingData.SCREEN_WIDTH.name: screenWidth,
            TrackingData.SCREEN_HEIGHT.name: screenHeight,
            TrackingData.CAPTURE_WIDTH.name: video_width,
            TrackingData.CAPTURE_HEIGHT.name: video_height,
            TrackingData.CAPTURE_FPS.name: self.get_stream_fps(),
            TrackingData.CORE_COUNT.name: psutil.cpu_count(logical=True),
            TrackingData.CORE_COUNT_PHYSICAL.name: psutil.cpu_count(logical=False),
            TrackingData.CORE_COUNT_AVAILABLE.name: len(psutil.Process().cpu_affinity()),  # number of usable cpus by
            # this process
            TrackingData.CPU_FREQUENCY_MHZ.name: f"Max: {cpu_max_freq}, current: {cpu_current_freq}",
            TrackingData.GPU_INFO.name: gpu_name,
            TrackingData.SYSTEM.name: system_info["system"],
            TrackingData.SYSTEM_VERSION.name: system_info["release"],
            TrackingData.MODEL_NAME.name: system_info["node"],
            TrackingData.PROCESSOR.name: system_info["machine"],
            TrackingData.RAM_OVERALL_GB.name: ram_info["total"] / 1000000000,  # convert from Bytes to GB
            TrackingData.RAM_AVAILABLE_GB.name: ram_info["available"] / 1000000000,
            TrackingData.RAM_FREE_GB.name: ram_info["free"] / 1000000000,
        })
        self.__logger.log_csv_data(data=self.__tracked_data)

    def get_stream_fps(self):
        return self.capture.get(cv2.CAP_PROP_FPS)

    def get_stream_dimensions(self):
        return self.capture.get(3), self.capture.get(4)

    def get_current_frame(self) -> np.ndarray:
        return self.__current_frame

    # TODO: upload the folders from the unity system as well when the user presses the stop hot key!
    #  -> make sure we wait until this upload has finished!
    #  -> shouldn't take too long (800ms ca. for 50kb) ->
    def __stop_tracking(self):
        """
        Stop and cleanup active webcam captures and destroy open windows if any.
        Also stop the logging.
        """
        if self.__tracking_active:
            # TODO enable again:
            notification.notify(title="Tracking nicht mehr aktiv", message="Das Tracking wurde gestoppt!", timeout=1)
            self.__tracking_active = False
            self.__set_tracking_status_ui()
            # self.capture.stop()
            cv2.destroyAllWindows()

            self.__logger.finish_logging()

            # remove and disconnect all hotkeys and signals to prevent user from starting again without restarting
            # the program itself
            # self.start_button.disconnect()
            # self.stop_button.disconnect()
            keyboard.remove_all_hotkeys()

            if self.debug:
                self.fps_measurer.stop()
                print(f"[INFO] elapsed time: {self.fps_measurer.elapsed():.2f} seconds")
                print(f"[INFO] approx. FPS on background thread: {self.fps_measurer.fps():.2f}")
                print("Frames on main thread:", self.fps_measurer._numFrames)

    def finish_tracking_system(self):
        self.__stop_tracking()
        self.__logger.stop_upload()

    def closeEvent(self, event):
        if self.progress == 100 or self.progress is None:
            self.finish_tracking_system()
            event.accept()
        else:
            # show warning if the upload progress hasn't reach 100% yet to prevent users from accidentally
            # closing the system
            # TODO translate the buttons as well!
            choice = QMessageBox.question(self, 'Tracking-System beenden?',
                                          "Bitte schließen Sie das Tracking-System erst, wenn der Fortschrittsbalken "
                                          "bei 100% angekommen ist! Möchten Sie das System wirklich beenden?",
                                          QMessageBox.Yes | QMessageBox.No)
            if choice == QMessageBox.Yes:
                # TODO log this as well? (small txt. with "User quitted too early") ?
                self.finish_tracking_system()
                event.accept()
            else:
                event.ignore()

    # TODO delete later:
    def debug_me(self):
        self.__tracking_active = True
        self.__set_tracking_status_ui()
        # self.capture.start()  # start reading frames from webcam
        self.tracking_thread = threading.Thread(target=self.__start_tracking, name="TrackingThread", daemon=True)
        self.tracking_thread.start()


def main():
    app = QApplication(sys.argv)
    tracking_system = TrackingSystem()
    tracking_system.show()
    tracking_system.listen_for_hotkey()

    # TODO only for debugging fps:
    """
    tracking_system.debug_me()
    c = 0
    start_time = datetime.now()
    while True:
        curr_frame = tracking_system.get_current_frame()
        if curr_frame is None:
            continue
        c += 1
        elapsed_time = (datetime.now() - start_time).total_seconds()
        fps = c / elapsed_time if elapsed_time != 0 else c
        cv2.putText(curr_frame, f"mainthread FPS: {fps:.3f}",
                    (350, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("fps_main_thread", curr_frame)
        if cv2.waitKey(1) & 0xFF == ord('f'):
            break
    """
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

"""
Creating an exe file:

1. Comment out the config parsing in the logger and add server credentials directly in the code.

For creating exe with auto-py-to-exe: select --onefile and window based add-data 
C:/Users/Michael/Documents/GitHub/Praxisseminar-Webcam-Tracking-System/weights add-data 
C:/Users/Michael/AppData/Local/Programs/Python/Python39/Lib/site-packages/mxnet add-data 
C:/Users/Michael/Documents/GitHub/Praxisseminar-Webcam-Tracking-System/tracking_service
add hidden-imports: pandas, pysftp, plyer.platforms.win.notification and requests
# for plyer import see 
https://stackoverflow.com/questions/56281839/issue-with-plyer-library-of-python-when-creating-a-executable-using-pyinstaller 

2. Pyinstaller command for the above:
pyinstaller --noconfirm --onefile --windowed 
--add-data "C:/Users/Michael/AppData/Local/Programs/Python/Python39/Lib/site-packages/mxnet;mxnet/" 
--add-data "C:/Users/Michael/Documents/GitHub/Praxisseminar-Webcam-Tracking-System/weights;weights/" 
--add-data "C:/Users/Michael/Documents/GitHub/Praxisseminar-Webcam-Tracking-System/tracking_service;tracking_service/" 
--hidden-import "plyer.platforms.win.notification" 
--hidden-import "pandas" 
--hidden-import "pysftp" 
--hidden-import "requests"  
"C:/Users/Michael/Documents/GitHub/Praxisseminar-Webcam-Tracking-System/tracking/tracker.py"
"""
