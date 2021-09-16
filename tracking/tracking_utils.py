import cv2


def to_gray(frame):
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grayscale_image


def find_face_mxnet(face_detector, frame):
    bboxes = face_detector.detect(frame)
    face_region = None
    for face in bboxes:
        face_region = extract_image_region(frame, face[0], face[1], face[2], face[3])
        cv2.imshow("extracted_region_mxnet", face_region)
        break  # break to take only the first face (in most cases there should be only one anyway)
    return face_region


def find_face_mxnet_resized(face_detector, frame, scale_factor=0.5, show_result=True):
    """
    The same algorithm as above but scales the input image smaller first so the face detection is faster,
    then upscales the resulting face image by the same amount the frame was scaled at the start.

    Taken from https://github.com/spmallick/learnopencv/tree/master/FaceDetectionComparison and adapted.
    """
    image_frame = frame.copy()
    frameHeight = image_frame.shape[0]
    frameWidth = image_frame.shape[1]
    image_frame_small = cv2.resize(image_frame, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    scaleHeight = frameHeight / (frameHeight * scale_factor)
    scaleWidth = frameWidth / (frameWidth * scale_factor)

    bboxes = face_detector.detect(image_frame_small)
    face_region = None
    for face in bboxes:
        face_region = extract_image_region(frame,
                                           face[0] * scaleWidth,
                                           face[1] * scaleHeight,
                                           face[2] * scaleWidth,
                                           face[3] * scaleHeight)
        if show_result:
            cv2.imshow("extracted_region_mxnet", face_region)
        break  # break to take only the first face (in most cases there should be only one anyway)
    return face_region


# next three methods copy & pasted from post_processing/image_utils for easier bundling in exe file with pyinstaller

def scale_image(frame, scale_factor=0.25, show_scaled=False):
    # Resize frame of video to 'scale_factor' of original size for faster processing
    scaled_image = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    if show_scaled:
        cv2.imshow(f"Scale={scale_factor}", scaled_image)
    return scaled_image


def extract_image_region(image, x_start, y_start, x_end, y_end, padding=5):
    # add small padding to every corner point to make sure there is enough of the face visible for the post processing
    x_start, y_start, x_end, y_end = x_start - padding, y_start - padding, x_end + padding, y_end + padding
    # make sure the bounding box coordinates are located in the image's bounding box; else take the outermost values
    startX = max(0, x_start)
    startY = max(0, y_start)
    endX = min(x_end, image.shape[1])
    endY = min(y_end, image.shape[0])
    # round to integer so we can extract a region from the original image
    return image[int(round(startY)): int(round(endY)), int(round(startX)): int(round(endX))]
