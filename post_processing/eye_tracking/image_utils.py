import io
import time
import imutils
from numpy import ndarray
from scipy.spatial import distance as dist
import cv2
import numpy as np
# import dlib
# for morphological filtering
from skimage.morphology import opening
from skimage.morphology import disk


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

    """
    thresh = cv2.erode(new_frame, None, iterations=2)
    new_frame = cv2.dilate(thresh, None, iterations=4)
    cv2.imshow("im_bil_erode_dilate", new_frame)

    thresh = cv2.erode(resized_image, None, iterations=2)
    new_frame = cv2.dilate(thresh, None, iterations=4)
    cv2.imshow("im_erode_dilate", new_frame)
    """
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


def detect_pupils(cropped_l_e_img, cropped_r_e_img=None):
    all_images_arr = []

    if cropped_r_e_img is None:
        all_images_arr.append(cropped_l_e_img)
        img = gray_blurred(cropped_l_e_img, 19, gray=True)
        all_images_arr.append(img)
        blurred_img = historgramEqualization(img, clahe=False)
        all_images_arr.append(blurred_img)
        # Left Eye
        _, threshold_l = cv2.threshold(blurred_img, 5, 255, cv2.THRESH_BINARY_INV)
        all_images_arr.append(threshold_l)
        contours, _ = cv2.findContours(threshold_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        cnt = contours[0]
        center_l, radius_l = cv2.minEnclosingCircle(cnt)

        if center_l and radius_l:
            cv2.circle(cropped_l_e_img, tuple(np.array([center_l[0], center_l[1]]).astype(int)),
                       int(round(radius_l)), (255, 34, 34))

        return (center_l, radius_l), all_images_arr

    else:
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

        if center_l and radius_l:
            cv2.circle(cropped_l_e_img, tuple(np.array([center_l[0], center_l[1]]).astype(int)),
                       int(round(radius_l)), (255, 34, 34))
            cv2.imshow('left eye', cropped_l_e_img)
        if center_r and radius_r:
            cv2.circle(cropped_r_e_img, tuple(np.array([center_r[0], center_r[1]]).astype(int)),
                       int(round(radius_r)), (255, 34, 34))
            cv2.imshow('right eye', cropped_r_e_img)

        all_images_arr.append(cropped_l_e_img)
        all_images_arr.append(blurred_img_l_g)
        all_images_arr.append(blurred_img_l)
        all_images_arr.append(threshold_l)

        all_images_arr.append(cropped_r_e_img)
        all_images_arr.append(blurred_img_r_g)
        all_images_arr.append(blurred_img_r)
        all_images_arr.append(threshold_r)
        return ((center_l, radius_l), (center_r, radius_r)), all_images_arr


def connected_component_threshold(image):
    """
    Taken from this stackoverflow answer: https://stackoverflow.com/a/52006700
    """
    black = 0
    white = 255
    threshold = 50

    pixels = np.array(image)[:, :, 0]
    pixels[pixels > threshold] = white
    pixels[pixels < threshold] = black

    cv2.imshow("alternative", pixels)

    # Morphological opening
    blobSize = 1  # Select the maximum radius of the blobs you would like to remove
    structureElement = disk(blobSize)  # you can define different shapes, here we take a disk shape
    # We need to invert the image such that black is background and white foreground to perform the opening
    pixels = np.invert(opening(np.invert(pixels), structureElement))

    # Find the connected components (black objects in your image)
    # Because the function searches for white connected components on a black background, we need to invert the image
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.invert(pixels), connectivity=8)

    # For every connected component in your image, you can obtain the number of pixels from the stats variable in the
    # last column. We remove the first entry from sizes, because this is the entry of the background connected component
    sizes = stats[1:, -1]
    nb_components -= 1

    # Define the minimum size (number of pixels) a component should consist of
    minimum_size = 100

    # Create a new image
    newPixels = np.ones(pixels.shape) * 255

    # Iterate over all components in the image, only keep the components larger than minimum size
    for i in range(1, nb_components):
        if sizes[i] > minimum_size:
            newPixels[output == i + 1] = 0

    cv2.imshow("alternative_connected", newPixels)


def find_pupil(frame, pupil_thresh=30, save=False):
    # TODO mit dem threshold könnte man vllt auch ganz gut blinks erkennen...
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


def increase_image_contrast(image):
    kernel = np.ones((3, 3), np.uint8)
    new_frame = cv2.bilateralFilter(image, 10, 15, 15)
    new_frame = cv2.erode(new_frame, kernel, iterations=3)
    cv2.imshow("increased contrast", image)
    return new_frame


def invert_image(image):
    return cv2.bitwise_not(image)


def circler(im):
    """
    Taken and updated from https://github.com/pupal-deep-learning/PuPal-Beta
    """
    # find contours first
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thres = cv2.threshold(im, 40, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Contours:\n{contours}")

    ratio, kernel_size = 3, 3
    low_threshold = 50
    detected_edges = cv2.Canny(thres, low_threshold, low_threshold * ratio, kernel_size)

    result = apply_hough_circles(thres)
    cv2.imshow('hough circles', result)

    # try to extract contours of iris and pupil assuming no other contours have been detected
    # TODO this assumption would only work if given an image with only the eye and round pupils
    if len(contours) == 2:
        # iris
        cnt_iris = contours[0]
        (xi, yi), radius_iris = cv2.minEnclosingCircle(cnt_iris)
        center_i = (int(xi), int(yi))
        radius_i = int(radius_iris)

        # pupil
        cnt_pupil = contours[1]
        (x_p, y_p), radius_pupil = cv2.minEnclosingCircle(cnt_pupil)
        center_p = (int(x_p), int(y_p))
        radius_p = int(radius_pupil)

        # cv2.polylines(im, cnt_pupil, True, (0, 255, 220), thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(im, center_p, radius_p, color=(0, 255, 0))

        # ratio
        ratio = round((radius_pupil / radius_iris), 4)
        if 0.2 < ratio < 0.8:
            return ratio, radius_i, center_i, radius_p, center_p

    cv2.imshow('edges', detected_edges)
    cv2.imshow('thres', thres)
    cv2.imshow('img', im)


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


def apply_automatic_canny(frame, show_edges=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edgeMap = imutils.auto_canny(gray)
    if show_edges:
        cv2.imshow("Original", frame)
        cv2.imshow("Automatic Edge Map", edgeMap)
    return edgeMap


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


def find_marker(image):
    # convert the image to grayscale and blur to detect edges

    blurred_frame = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([38, 86, 0])
    upper_blue = np.array([121, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    con = max(contours, key=cv2.contourArea)

    return cv2.minAreaRect(con)


def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the image to the camera
    return (knownWidth * focalLength) / perWidth


def get_distance(image):
    # TODO compute these two values during calibration:
    KNOWN_DISTANCE = 30
    KNOWN_WIDTH = 3

    marker = find_marker(image)
    print("Marker: ", marker)
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
    # alternativ:
    # dis = 0.367 * 14 / (h * 0.00064)
    # dis = math.floor(dis)

    print("focalLength: ", focalLength)
    CM = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
    print("Distance: ", CM)


def convert_color_space(image):
    """
    Converts the captured BGR images from opencv to RGB images.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_bgr_to_rgb(frame):
    rgb_frame = frame[:, :, ::-1]
    return rgb_frame


def scale_image(frame, scale_factor=0.25, show_scaled=False):
    # Resize frame of video to 'scale_factor' of original size for faster processing
    scaled_image = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    if show_scaled:
        cv2.imshow(f"Scale={scale_factor}", scaled_image)
    return scaled_image


def resize_image(image, size=150, show_resized=False):
    resized = imutils.resize(image, width=size, inter=cv2.INTER_AREA)
    if show_resized:
        cv2.imshow(f"Size={size}dpx", resized)
    return resized


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


def convert_to_bytes(image):
    data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return bytes(data)


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


def encode_image(image, img_format=".jpeg"):
    """
    Encodes the given image as a byte stream in working memory for faster transfer.
    See https://jdhao.github.io/2019/07/06/python_opencv_pil_image_to_bytes/
    """
    is_success, im_buf_arr = cv2.imencode(img_format, image)
    # byte_im = im_buf_arr.tobytes()

    byte_stream = io.BytesIO(im_buf_arr)
    # byte_im = byte_stream.getvalue()
    # byte_stream.seek(0)
    return byte_stream


def decode_byte_image(byte_image_path):
    """
    Decodes the byte image at the given path back to a real image.

    Args:
        byte_image_path: the path to the image in byte format

    Returns: the decoded image as np.ndarray
    """
    with open(byte_image_path, "rb") as img:
        byte_stream = io.BytesIO(img.read())
        decoded_img = cv2.imdecode(np.frombuffer(byte_stream.getbuffer(), np.uint8), cv2.IMREAD_UNCHANGED)
        cv2.imshow("decoded byte image", decoded_img)
        return decoded_img
