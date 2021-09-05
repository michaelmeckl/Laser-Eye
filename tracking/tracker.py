#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Creating an exe file:

1. Comment out the config parsing in the logger ('get_server_credentials()') and add server credentials directly in
the code.

For creating exe with auto-py-to-exe: select --onefile and window based
add-data C:/Users/Michael/Documents/GitHub/Praxisseminar-Webcam-Tracking-System/weights
add-data C:/Users/Michael/AppData/Local/Programs/Python/Python39/Lib/site-packages/mxnet
add-data C:/Users/Michael/Documents/GitHub/Praxisseminar-Webcam-Tracking-System/tracking_service
add hidden-imports: pandas, pysftp, plyer.platforms.win.notification and requests
# for plyer import see
https://stackoverflow.com/questions/56281839/issue-with-plyer-library-of-python-when-creating-a-executable-using-pyinstaller
and
exclude-module matplotlib

2. Pyinstaller command for the above:
(add --debug "all" to the command below if things go wrong...)
pyinstaller --noconfirm --onefile --windowed --add-data "C:/Users/Michael/AppData/Local/Programs/Python/Python39/Lib/site-packages/mxnet;mxnet/" --add-data "C:/Users/Michael/Documents/GitHub/Praxisseminar-Webcam-Tracking-System/weights;weights/" --add-data "C:/Users/Michael/Documents/GitHub/Praxisseminar-Webcam-Tracking-System/tracking_service;tracking_service/" --hidden-import "plyer.platforms.win.notification" --hidden-import "pandas" --hidden-import "pysftp" --hidden-import "requests" --exclude-module "matplotlib"  "C:/Users/Michael/Documents/GitHub/Praxisseminar-Webcam-Tracking-System/tracking/tracker.py"
"""

import math
import platform
import sys
import threading
import time
from pathlib import Path
import cv2  # pip install opencv-python
import numpy as np
import psutil
import pyautogui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox, QPushButton
from gpuinfo.windows import get_gpus as get_amd  # pip install gpu-info
from gpuinfo.nvidia import get_gpus as get_nvidia
from plyer import notification
from FpsMeasuring import FpsMeasurer
from TrackingLogger import Logger as TrackingLogger
from TrackingLogger import TrackingData, get_timestamp
from tracking_service.face_detector import MxnetDetectionModel
from tracking_utils import find_face_mxnet_resized
# import keyboard  # for hotkeys


class TrackingSystem(QtWidgets.QWidget):

    def __init__(self, debug_active=False):
        super(TrackingSystem, self).__init__()
        self.__tracking_active = False
        self.__progress = None  # the upload progress
        self.__debug = debug_active
        self.__selected_camera = 0  # use the in-built camera (index 0) per default

        self.__load_face_detection_model()
        self.__setup_gui()
        self.__init_logger()

        self.logger.init_server_connection()
        self.fps_measurer = FpsMeasurer()
        """
        available_indexes = find_attached_cameras()
        if len(available_indexes) > 0:
            # set the available camera available_indexes in the gui dropdown (and convert them to strings before)
            self.camera_selection.addItems(map(str, available_indexes))

            self.logger.init_server_connection()
            self.fps_measurer = FpsMeasurer()
        else:
            # there seems to be no available camera!
            self.error_label.setText("Leider wurde keine verfügbare Kamera gefunden! Bitte stellen Sie sicher, dass "
                                     "eine Kamera an den Computer angeschlossen (oder eingebaut ist), und "
                                     "starten Sie dann das Programm erneut!")
            self.logger.log_error("Keine verfügbare Kamera gefunden!")
        """

    def __load_face_detection_model(self):
        # necessary for building the exe file with pyinstaller with the --one-file option as the path changes;
        # see https://stackoverflow.com/questions/7674790/bundling-data-files-with-pyinstaller-onefile for more
        if getattr(sys, 'frozen', False):
            # change the path if we are executing the exe file; the lint warning here can be ignored
            folder = Path(sys._MEIPASS)
            data_path = folder / 'weights/16and32'
        else:
            folder = Path(__file__).parent
            data_path = folder / '../weights/16and32'

        self.face_detector = MxnetDetectionModel(data_path, 0, .6, gpu=-1)

    def __setup_gui(self):
        self.layout = QtWidgets.QVBoxLayout()  # set base layout (vertically aligned box)

        # show status of tracking
        self.tracking_status = QtWidgets.QLabel(self)
        self.tracking_status.setContentsMargins(0, 10, 0, 10)
        self.tracking_status.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.tracking_status)

        # show some usage notes
        usage_label = QtWidgets.QLabel(self)
        usage_label.setText(
            "Verwendung:\n\nMit den Buttons unten kann das Tracking gestartet und wieder "
            "gestoppt werden.\n\nDieses Fenster muss nach Beginn der Studie solange geöffnet bleiben, bis der "
            "Hochladevorgang beendet ist (100% auf dem Fortschrittsbalken unterhalb). "
            "Abhängig von der Internetgeschwindigkeit und Leistung des Rechners kann dies einige Zeit in "
            "Anspruch nehmen. Während dieser Zeit muss der Rechner eingeschaltet bleiben!"
        )
        usage_label.setContentsMargins(10, 10, 10, 10)
        usage_label.setWordWrap(True)
        usage_label.setStyleSheet("QLabel {font-size: 9pt;}")
        self.layout.addWidget(usage_label)

        self.__setup_progress_bar()  # show a progress bar for the upload
        self.__show_instructions()
        # self.__setup_camera_selection()

        self.error_label = QtWidgets.QLabel(self)
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setStyleSheet("QLabel {color: rgba(255, 0, 0);}")
        self.layout.addWidget(self.error_label)

        self.__setup_button_layout()  # add buttons to start and stop tracking
        self.__set_tracking_status_ui()

        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.layout)
        self.setGeometry(100, 50, 1200, 950)  # set initial size of gui window (top left and bottom right positions)
        self.setWindowTitle("Tracking System")

    def __show_instructions(self):
        label_general = QtWidgets.QLabel(self)
        label_general.setText("Hinweise:")
        label_general.setStyleSheet("QLabel {font-size: 10pt; margin-top: 40px;}")
        self.layout.addWidget(label_general)

        general_instructions = QtWidgets.QTextEdit(self)
        general_instructions.setStyleSheet("QTextEdit {font-size: 9pt; background-color : rgba(0, 0, 0, 0%); "
                                           "border: 0; margin-top: 20px}")
        general_instructions.setReadOnly(True)  # don't let user edit this text field
        general_instructions.setHtml(
            "<ul>"
            "<li>Bitte versuchen Sie während das Tracking aktiv ist, <b>möglichst ruhig zu sitzen</b> und den Kopf "
            "nicht zu viel oder zu schnell zu bewegen (normale Kopfbewegungen sind selbstverständlich in Ordnung).</li>"
            "<li>Bitte tragen Sie während des Trackings <b>keine Brille</b>, da die dabei auftretenden Reflexionen "
            "ein Problem bei der Verarbeitung der Daten darstellen.</li>"
            "<li>Versuchen Sie <b>nicht zu weit weg von der Kamera</b> zu sein. Der Abstand zwischen Kamera und "
            "Gesicht sollte nicht mehr als maximal 50-60 cm betragen.</li>"
            "<li>Die <b>Kamera</b> sollte beim Tracking möglichst <b>gerade und frontal zum Gesicht positioniert</b> "
            "sein, sodass das gesamte Gesicht von der Kamera erfasst werden kann.</li>"
            "<li>Bitte achten Sie auf <b>gute Lichtverhältnisse</b> während der Studie. Der von der Webcam "
            "sichtbare Bereich sollte gut ausgeleuchtet sein.</li>"
            "<li>Entfernen Sie vor Beginn des Trackings bitte etwaige Webcam Abdeckungen.</li>"
            "<li>Die Buttons zum Starten und Stoppen funktionieren nur einmal. Nach Stoppen des "
            "Trackings muss das Programm erneut gestartet werden, um das Tracking wieder zu starten.</li>"
            "<li>Bei Beenden dieser Anwendung werden <b>automatisch</b> alle für die Studie nicht mehr benötigten "
            "<b>Dateien in diesem Ordner gelöscht</b>. Bitte verschieben Sie diese Datei deshalb in keinen anderen "
            "Ordner!</li>"
            "</ul>"
        )
        self.layout.addWidget(general_instructions)

    def __setup_camera_selection(self):
        """
        Show a dropdown menu to choose between all available cameras for tracking.
        """
        camera_select_label = QtWidgets.QLabel(self)
        camera_select_label.setStyleSheet("QLabel {font-size: 9pt;")
        camera_select_label.setText("Kamera-Auswahl für Tracking (0 sollte i.d.R. die Richtige sein):")
        self.camera_selection = QtWidgets.QComboBox(self)
        self.camera_selection.setStyleSheet("QComboBox {min-width: 1em; border: 1px solid gray; border-radius: 2px; "
                                            "padding: 1px 18px 1px 3px;}")
        self.camera_selection.currentIndexChanged.connect(self.__selected_cam_changed)

        dropdown_layout = QtWidgets.QHBoxLayout()
        dropdown_layout.addWidget(camera_select_label)
        dropdown_layout.addWidget(self.camera_selection, alignment=Qt.AlignLeft)
        dropdown_layout.setContentsMargins(0, 0, 0, 15)
        dropdown_layout.setAlignment(Qt.AlignCenter)
        self.layout.addLayout(dropdown_layout)

    def __selected_cam_changed(self, index):
        self.__selected_camera = index

    def __setup_button_layout(self):
        button_common_style = "QPushButton {font: 16px; padding: 12px; min-width: 10em; border-radius: 6px;} " \
                              ":disabled {background-color: lightGray; color: gray;}"
        self.start_button = QtWidgets.QPushButton(self)
        self.start_button.setText("Starte Tracking")
        self.start_button.setStyleSheet(f"{button_common_style}"
                                        ":enabled {background-color: rgb(87, 205, 0); color: black;}"
                                        ":pressed {background-color: rgb(47, 165, 0);}")
        self.start_button.clicked.connect(self.__show_start_info_box)  # connect start method to this button

        self.stop_button = QtWidgets.QPushButton(self)
        self.stop_button.setText("Tracking stoppen")
        self.stop_button.setStyleSheet(f"{button_common_style}"
                                       ":enabled {background-color: rgb(153, 25, 25); color: white;}"
                                       ":pressed {background-color: rgb(141, 12, 12);}")
        self.stop_button.clicked.connect(self.__stop_study)  # connect stop method to this button
        self.stop_button.setEnabled(False)  # disable stop button at the start

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.start_button, alignment=Qt.AlignLeft)
        button_layout.addWidget(self.stop_button, alignment=Qt.AlignRight)
        button_layout.setContentsMargins(20, 10, 20, 10)
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
            self.tracking_status.setStyleSheet("QLabel {color: green; font-size: 10pt;}")
        else:
            self.tracking_status.setText("Tracking nicht aktiv")
            self.tracking_status.setStyleSheet("QLabel {color: red; font-size: 10pt;}")

    def __show_start_info_box(self):
        self.__preview_box_is_showing = True

        # webcam preview needs to be shown in separate thread as otherwise the qt info box and the preview would block
        # each other
        self.preview_thread = threading.Thread(target=self.__show_webcam_preview, name="PreviewThread", daemon=True)
        self.preview_thread.start()

        preview_box = QMessageBox(QMessageBox.Information, "Webcam-Vorschau",
                                  "Eine Vorschau Ihrer Webcam sollte sich in wenigen Sekunden in einem separaten "
                                  "Fenster öffnen. Stellen Sie bitte sicher, dass Ihr Gesicht vollständig im Bild und "
                                  "gut sichtbar ist. Sobald Sie zufrieden sind, drücken Sie bitte auf 'Weiter'",
                                  parent=self)
        # create and set custom buttons
        yes_button = QPushButton("Weiter")
        preview_box.addButton(yes_button, QMessageBox.YesRole)
        no_button = QPushButton("Abbrechen")
        preview_box.addButton(no_button, QMessageBox.NoRole)
        preview_box.exec_()

        if preview_box.clickedButton() == yes_button:
            self.__preview_box_is_showing = False
            self.__activate_tracking()
        else:
            self.__preview_box_is_showing = False

    def __show_webcam_preview(self):
        # setup webcam capture
        self.capture = cv2.VideoCapture(self.__selected_camera)
        # show webcam capture to user
        while self.__preview_box_is_showing:
            return_val, frame = self.capture.read()
            if not return_val:
                self.logger.log_error("Couldn't get frame from webcam in preview!")
                break
            cv2.imshow("Webcam Vorschau", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.__cleanup_webcam_capture()

    def __on_upload_progress(self, current, overall):
        if overall == 0:
            return

        if current != 0:
            seconds_per_frame = (time.time() - self.__upload_start) / current
            eta_seconds = (overall - current) * seconds_per_frame
            minutes = math.floor(eta_seconds / 60)
            seconds = round(eta_seconds % 60)
            self.label_eta.setText(f"Verbleibende Zeit: {minutes} min, {seconds} Sekunden")

        self.label_current.setText(str(current))
        self.label_all.setText(f"/ {overall}")
        self.__progress = int((current / overall) * 100)
        self.progress_bar.setValue(self.__progress)

    def __on_error(self):
        failed_uploads = self.logger.get_failed_uploads()
        if len(failed_uploads) == 0:
            self.error_label.setText("")  # reset error text if everything has been fixed automatically
        else:
            failed_list_string = "\n".join([f"- {failed_upload}" for failed_upload in failed_uploads])
            self.error_label.setText("Bei der Verbindung ist ein Fehler aufgetreten! Folgende Dateien konnten deshalb "
                                     "nicht übertragen werden und müssen nach Ende der Studie selbst an die "
                                     "Versuchsleiter übermittelt werden:\n" + failed_list_string)

    def __init_logger(self):
        self.logger = TrackingLogger(self.__on_upload_progress, self.__on_error)
        self.__tracked_data = {key.name: None for key in TrackingData}

    # The following could be used instead of the buttons to automatically start and stop tracking via hotkeys:
    # def listen_for_hotkey(self, hotkey_start="ctrl+a", hotkey_stop="ctrl+q"):
    #     keyboard.add_hotkey(hotkey_start, self.__show_start_info_box, suppress=False, trigger_on_release=False)
    #     keyboard.add_hotkey(hotkey_stop, self.__stop_study, suppress=False, trigger_on_release=False)

    def __activate_tracking(self):
        if not self.__tracking_active:
            self.__tracking_active = True

            self.__set_tracking_status_ui()
            self.label_eta.setText(f"Mehr Daten werden benötigt ...")
            # disable camera selection after tracking has started
            # self.camera_selection.setEnabled(False)
            # toggle button active status
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            notification.notify(title="Tracking aktiv", message="Das Tracking wurde gestartet!", timeout=1)

            # start tracking on a background thread
            self.tracking_thread = threading.Thread(target=self.__start_tracking, name="TrackingThread", daemon=True)
            self.tracking_thread.start()
        else:
            notification.notify(title="Tracking kann nicht gestartet werden!", message="Das Tracking läuft bereits!",
                                timeout=2)

    def __start_tracking(self):
        """
        This function runs on a background thread so the fps of the video games the user is playing aren't reduced.
        """
        self.capture = cv2.VideoCapture(self.__selected_camera)
        # Start the logging and the uploading on other background threads (so the reading from the webcam isn't
        # blocked by the processing there)
        self.__log_system_data()
        self.logger.start_saving_images_to_disk()  # start saving webcam frames to disk
        self.logger.start_async_upload()  # start uploading data to sftp server

        self.__upload_start = time.time()
        self.fps_measurer.start()

        while self.__tracking_active:
            # read the next frame from the webcam
            return_val, frame = self.capture.read()
            if not return_val:
                self.logger.log_error("Couldn't get current frame while tracking!")
                break
            if frame is None:
                sys.stderr.write("Frame from webcam is None!")
                continue

            processed_frame = self.__process_frame(frame)
            if processed_frame is None:
                continue

            # update the average frame rate
            self.fps_measurer.update()
            current_fps = self.fps_measurer.get_current_fps()
            if self.__debug:
                cv2.putText(processed_frame, f"current FPS: {current_fps:.3f}",
                            (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("current_frame", processed_frame)
                # read until the video is finished or if no video was provided until the
                # user presses 'q' on the keyboard;
                # replace 1 with 0 to step manually through the video "frame-by-frame"
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # cleanup the webcam capture at the end
        self.__cleanup_webcam_capture()

    def __process_frame(self, frame: np.ndarray) -> np.ndarray:
        face_image = find_face_mxnet_resized(self.face_detector, frame, show_result=True if self.__debug else False)

        if face_image is not None:
            # save timestamp separately as it has to be the same for all the frames and the log data! otherwise it
            # can't be matched later!
            log_timestamp = get_timestamp()
            self.logger.add_image_to_queue("capture", face_image, log_timestamp)
        return frame

    def __cleanup_webcam_capture(self):
        self.capture.release()
        cv2.destroyAllWindows()

    def __log_system_data(self):
        # get the dimensions of the webcam
        video_width, video_height = self.__get_stream_dimensions()
        # get the dimensions of the primary monitor.
        screenWidth, screenHeight = pyautogui.size()

        system_info = platform.uname()._asdict()
        ram_info = psutil.virtual_memory()._asdict()
        cpu_current_freq, _, cpu_max_freq = psutil.cpu_freq()  # in Mhz

        # try to find gpus if any
        found_gpus = []
        try:
            gpu_list = find_gpu()
            if len(gpu_list) == 0:
                # try to check for nvidia gpu instead
                found_gpus = [x.name for x in get_nvidia()]
            else:
                found_gpus.extend(gpu_list)
        except Exception:
            found_gpus = ["no gpu found"]

        # noinspection PyProtectedMember
        self.__tracked_data.update({
            TrackingData.SCREEN_WIDTH.name: screenWidth,
            TrackingData.SCREEN_HEIGHT.name: screenHeight,
            TrackingData.CAPTURE_WIDTH.name: video_width,
            TrackingData.CAPTURE_HEIGHT.name: video_height,
            TrackingData.CAPTURE_FPS.name: self.__get_stream_fps(),
            TrackingData.CORE_COUNT.name: psutil.cpu_count(logical=True),
            TrackingData.CORE_COUNT_PHYSICAL.name: psutil.cpu_count(logical=False),
            TrackingData.CORE_COUNT_AVAILABLE.name: len(psutil.Process().cpu_affinity()),  # number of usable cpus by
            # this process
            TrackingData.CPU_FREQUENCY_MHZ.name: f"Max: {cpu_max_freq}, current: {cpu_current_freq}",
            TrackingData.GPU_INFO.name: found_gpus,
            TrackingData.SYSTEM.name: system_info["system"],
            TrackingData.SYSTEM_VERSION.name: system_info["release"],
            TrackingData.MODEL_NAME.name: system_info["node"],
            TrackingData.MACHINE.name: system_info["machine"],
            TrackingData.PROCESSOR.name: platform.processor(),
            TrackingData.RAM_OVERALL_GB.name: ram_info["total"] / 1000000000,  # convert from Bytes to GB
            TrackingData.RAM_AVAILABLE_GB.name: ram_info["available"] / 1000000000,
            TrackingData.RAM_FREE_GB.name: ram_info["free"] / 1000000000,
        })
        self.logger.log_system_info(data=self.__tracked_data)

    def __get_stream_fps(self):
        return self.capture.get(cv2.CAP_PROP_FPS)

    def __get_stream_dimensions(self):
        return self.capture.get(3), self.capture.get(4)

    def __stop_tracking(self):
        """
        Stop and cleanup active webcam captures and destroy open windows if any.
        Also stop the logging by setting a boolean flag to False.
        """
        if self.__tracking_active:
            notification.notify(title="Tracking nicht mehr aktiv", message="Das Tracking wurde gestoppt!", timeout=1)
            self.__tracking_active = False
            self.__set_tracking_status_ui()
            # disable stop button but don't enable start button (not implemented to restart tracking!)
            self.stop_button.setEnabled(False)

            # remove and disconnect all hotkeys and signals to prevent user from starting again without restarting
            # the program itself
            # keyboard.remove_all_hotkeys()

            # get fps info and log them
            self.fps_measurer.stop()
            fps_values = self.fps_measurer.get_fps_list()
            elapsed_time = self.fps_measurer.elapsed()
            avg_fps_overall = self.fps_measurer.fps()
            frame_count = self.fps_measurer.get_frames_count()

            self.logger.finish_logging(fps_values, elapsed_time, avg_fps_overall, frame_count)
            """
            print(f"[INFO] elapsed time: {elapsed_time:.2f} seconds")
            print(f"[INFO] approx. FPS on background thread: {avg_fps_overall:.2f}")
            print(f"[INFO] frame count on background thread: {frame_count}")
            """

    def __stop_study(self):
        """
        Called when the user clicks on the stop button or presses the stop hotkey to stop recording new frames and
        also upload the game data if started as exe.
        """
        self.__stop_tracking()
        # upload the log folders from the unity game when tracking has finished
        game_data_thread = threading.Thread(target=self.logger.upload_game_data, name="GameDataUpload", daemon=True)
        game_data_thread.start()

    def __finish_tracking_system(self):
        """
        Called when the user clicks on the close symbol in the top right of the window and closes the system completely.
        """
        self.tracking_status.setText("Anwendung wird beendet. Bitte warten...")
        self.__stop_tracking()
        self.logger.stop_upload()

    def closeEvent(self, event):
        """
        Overwrite `closeEvent()` to be able to react to presses on the 'x' in the top-right corner of the window.
        """
        if self.__progress == 100 or self.__progress is None:
            self.__finish_tracking_system()
            event.accept()
        else:
            # show warning if the upload progress hasn't reach 100% yet to prevent users from accidentally
            # closing the system
            message_box = QMessageBox(QMessageBox.Warning, "Tracking-System beenden?",
                                      "Bitte schließen Sie das Tracking-System erst, wenn der Fortschrittsbalken bei "
                                      "100% angekommen ist! Möchten Sie das System wirklich beenden?", parent=self)
            # create and set custom buttons for yes and no
            yes_button = QPushButton("Ja")
            message_box.addButton(yes_button, QMessageBox.YesRole)
            no_button = QPushButton("Nein")
            message_box.addButton(no_button, QMessageBox.NoRole)
            message_box.exec_()  # show the message box

            if message_box.clickedButton() == yes_button:
                self.logger.log_too_early_quit()
                self.__finish_tracking_system()
                event.accept()
            else:
                event.ignore()


def find_gpu():
    try:
        return [x.name for x in get_amd()]
    except Exception:
        return []


def find_attached_cameras():
    # Taken from https://stackoverflow.com/questions/8044539/listing-available-devices-in-python-opencv
    # Checks the first 10 indexes for available cameras and returns all positions where a working camera was found.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr


def main():
    app = QApplication(sys.argv)
    tracking_system = TrackingSystem()
    tracking_system.show()
    # tracking_system.listen_for_hotkey()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
