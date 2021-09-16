import time
from numpy import ndarray
from scipy.spatial import distance as dist
import cv2
import numpy as np


def show_image_window(src, window_name, x_pos, y_pos):
    # shows a named opencv window with the given src content at the specified position on the screen
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, x_pos, y_pos)
    cv2.imshow(window_name, src)


def apply_threshold(eye_img, threshold_val=20, is_gray=False, show_annotation=False):
    if not is_gray:
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)

    _, threshold_eye_img = cv2.threshold(eye_img, threshold_val, 255, cv2.THRESH_BINARY)
    if show_annotation and threshold_eye_img is not None:
        # threshold_eye_img = cv2.resize(threshold_eye_img, None, fx=5, fy=5)
        show_image_window(threshold_eye_img, window_name="Threshold eye_img: ", x_pos=300, y_pos=200)

    return threshold_eye_img


def improve_image(image):
    """
    Different operations to improve the quality of the resized image.
    fastNlMeansDenoising works best but takes for ever (1 or 2 seconds); gauß blur and erode/dilate work quite well too
    """
    # resized_image = resize_image(image, size=500)
    # cv2.imshow("im_resized", resized_image)
    resized_image = image.copy()

    # denoised = cv2.fastNlMeansDenoising(resized_image, None, 10, 7, 21)
    # cv2.fastNlMeansDenoisingMulti()  # for image sequence
    # cv2.imshow("im_denoised", denoised)

    kernel_size = 5
    blurred_image = cv2.GaussianBlur(resized_image, (kernel_size, kernel_size), 0)
    cv2.imshow("im_blurred_5", blurred_image)

    new_frame = cv2.bilateralFilter(resized_image, 10, 15, 15)
    cv2.imshow("im_bil", new_frame)
    return new_frame


def historgramEqualization(img, clahe=False):
    if clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equ_img = clahe.apply(img)
    else:
        equ_img = cv2.equalizeHist(img)
    return equ_img


def gray_blurred(img, blur_l, gray=False, blur="Median", Lab=False):
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if Lab:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 1]

    if blur == "Median":
        return cv2.medianBlur(img, blur_l)
    elif blur == "gaussian":
        return cv2.GaussianBlur(img, blur_l, 0)


def detect_pupils(cropped_l_e_img, cropped_r_e_img, show_annotation=False):
    # Function taken and adjusted from https://github.com/anmolduainter/Pupil-Dilation/blob/master/pupilMeasurement.py

    cropped_l_e_img = cv2.cvtColor(cropped_l_e_img, cv2.COLOR_BGR2GRAY)
    cropped_r_e_img = cv2.cvtColor(cropped_r_e_img, cv2.COLOR_BGR2GRAY)
    blurred_img_l_g = gray_blurred(cropped_l_e_img, 19, gray=False)
    blurred_img_r_g = gray_blurred(cropped_r_e_img, 19, gray=False)

    blurred_img_l = historgramEqualization(blurred_img_l_g, clahe=False)
    blurred_img_r = historgramEqualization(blurred_img_r_g, clahe=False)

    # Left Eye
    _, threshold_l = cv2.threshold(blurred_img_l, 5, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    cnt = contours[0]
    center_l, radius_l = cv2.minEnclosingCircle(cnt)

    # Right Eye
    _, threshold_r = cv2.threshold(blurred_img_r, 5, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    cnt = contours[0]
    center_r, radius_r = cv2.minEnclosingCircle(cnt)

    if show_annotation:
        if center_l and radius_l:
            cv2.circle(cropped_l_e_img, tuple(np.array([center_l[0], center_l[1]]).astype(int)),
                       int(round(radius_l)), (255, 34, 34))
            show_image_window(cropped_l_e_img, window_name="left eye", x_pos=20, y_pos=50)
        if center_r and radius_r:
            cv2.circle(cropped_r_e_img, tuple(np.array([center_r[0], center_r[1]]).astype(int)),
                       int(round(radius_r)), (255, 34, 34))
            show_image_window(cropped_r_e_img, window_name="right eye", x_pos=300, y_pos=50)

    return 2 * radius_l, 2 * radius_r  # return the diameter of both pupils


def find_pupil(frame, pupil_thresh=30, save=False):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thres = cv2.threshold(image, pupil_thresh, 255, cv2.THRESH_BINARY)

    thresh = cv2.erode(thres, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    output = cv2.bitwise_and(image, image, mask=thresh)

    img_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if img_contours:
        cv2.drawContours(image, img_contours, -1, (0, 0, 255), thickness=1)

    for (i, c) in enumerate(img_contours):
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 34, 34))
        # ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        # cv2.circle(image, (int(cX), int(cY)), int(radius), (0, 0, 255), 1)
        break  # only the first

    # show the images
    cv2.imshow("images", np.hstack([image, output]))
    if save and image.size:
        cv2.imwrite(f'processing_data/img_contours__{time.time()}.png', image)


def invert_image(image):
    return cv2.bitwise_not(image)


def detect_iris(eye_frame):
    """Detects the iris and estimates the position of the iris by
    calculating the centroid.
    Taken from https://github.com/antoinelame/GazeTracking

    Arguments:
        eye_frame (numpy.ndarray): Frame containing an eye and nothing else
    """
    eye_fr = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    iris_frame = image_processing(eye_fr, 50)

    canny_output = cv2.Canny(iris_frame, 70, 70 * 2)

    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea)

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        cv2.drawContours(drawing, contours, i, (0, 255, 176), 2, cv2.LINE_8, hierarchy, 0)
    # Show in a window
    cv2.imshow('Eye', drawing)


def image_processing(eye_frame, threshold):
    """Performs operations on the eye frame to isolate the iris.
    Taken from https://github.com/antoinelame/GazeTracking

    Arguments:
        eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        threshold (int): Threshold value used to binarize the eye frame

    Returns:
        A frame with a single element representing the iris
    """
    kernel = np.ones((3, 3), np.uint8)
    new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
    new_frame = cv2.erode(new_frame, kernel, iterations=3)
    new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

    return new_frame


def apply_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ratio, kernel_size = 3, 3
    low_threshold = 50
    detected_edges = cv2.Canny(gray, low_threshold, low_threshold * ratio, kernel_size)
    return detected_edges


def apply_hough_circles(image):
    rows = image.shape[0]
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=rows / 16,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=40)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(image, center, radius, (255, 0, 255), 3)

    return image


def convert_color_space(image):
    """
    Converts the captured BGR images from opencv to RGB images.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def scale_image(frame, scale_factor=0.25, show_scaled=False):
    # Resize frame of video to 'scale_factor' of original size for faster processing
    scaled_image = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    if show_scaled:
        cv2.imshow(f"Scale={scale_factor}", scaled_image)
    return scaled_image


# @timeit
def preprocess_frame(frame: ndarray, kernel_size=5, keep_dim=True) -> ndarray:
    """
    Converts bgr image to grayscale using opencv. If keep_dim is set to True it will
    be converted back to BGR afterwards so that the third dimension will still be 3.
    This is necessary for some mxnet detectors that require a 4-Dim image input.
    """
    # Convert to grayscale and keep dimensions as required for the face (alignment) detectors
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if keep_dim:
        grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

    # blur to reduce noise; use smaller kernels to improve speed
    blurred_image = cv2.GaussianBlur(grayscale_image, (kernel_size, kernel_size), 0)
    return blurred_image


def eye_aspect_ratio(eye):
    """
    Computes the eye-aspect-ration (EAR) for the given eye landmarks.
    This function was taken from https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
    and adjusted to work with 8 instead of 6 eye markers.

    Args:
        eye: an array with the landmarks for this eye

    Returns:
        the eye-aspect-ratio (ear)
    """
    # compute the euclidean distances between the sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[7])
    B = dist.euclidean(eye[2], eye[6])
    C = dist.euclidean(eye[3], eye[5])
    # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    D = dist.euclidean(eye[0], eye[4])

    # compute the eye aspect ratio
    ear = (A + B + C) / (3.0 * D)  # 3*D as we have three vertical but only one horizontal coordinate pair
    return ear
