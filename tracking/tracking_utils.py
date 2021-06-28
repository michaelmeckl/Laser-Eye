import cv2
import imutils


def to_gray(frame):
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grayscale_image


def find_face_hog(hog_detector, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # dlib requires RGB while opencv uses BGR per default
    bboxes = hog_detector(rgb_frame, 0)  # 0 so it won't be upsampled
    if len(bboxes) > 0:
        # only take the first face if more were found (in most cases there should be only one anyway)
        face = bboxes[0]
        region = extract_image_region(frame, face.left(), face.top(), face.right(), face.bottom())
        cv2.imshow("extracted_region_mxnet", region)
        return region


def find_face_mxnet(face_detector, frame):
    bboxes = face_detector.detect(frame)
    face_region = None
    for face in bboxes:
        face_region = extract_image_region(frame, face[0], face[1], face[2], face[3])
        cv2.imshow("extracted_region_mxnet", face_region)
        break  # break to take only the first face (in most cases there should be only one anyway)
    return face_region


def find_face_mxnet_resized(face_detector, frame, new_height=250, new_width=0):
    """
    The same algorithm as above but scales the input image smaller first so the face detection is faster,
    then upscales the resulting face image by the same amount the frame was scaled at the start.

    Taken from https://github.com/spmallick/learnopencv/tree/master/FaceDetectionComparison and adapted.
    """
    image_frame = frame.copy()
    frameHeight = image_frame.shape[0]
    frameWidth = image_frame.shape[1]
    if not new_width:
        new_width = int((frameWidth / frameHeight) * new_height)
    scaleHeight = frameHeight / new_height
    scaleWidth = frameWidth / new_width

    image_frame_small = cv2.resize(image_frame, (new_width, new_height))
    bboxes = face_detector.detect(image_frame_small)
    face_region = None
    for face in bboxes:
        face_region = extract_image_region(frame,
                                           face[0] * scaleWidth,
                                           face[1] * scaleHeight,
                                           face[2] * scaleWidth,
                                           face[3] * scaleHeight)
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


def resize_image(image, size=150, show_resized=False):
    resized = imutils.resize(image, width=size, inter=cv2.INTER_AREA)
    if show_resized:
        cv2.imshow(f"Size={size}dpx", resized)
    return resized


def extract_image_region(image, x_start, y_start, x_end, y_end):
    # make sure the bounding box coordinates are located in the image's bounding box; else take the outermost values
    startX = max(0, x_start)
    startY = max(0, y_start)
    endX = min(x_end, image.shape[1])
    endY = min(y_end, image.shape[0])
    # round to integer so we can extract a region from the original image
    return image[int(round(startY)): int(round(endY)), int(round(startX)): int(round(endX))]
