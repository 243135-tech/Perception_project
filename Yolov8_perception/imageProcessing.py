import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from sort import Sort


def resize_image(input_image_path, width=217, height=225):
    """
    Resizes an input image to the specified dimensions and saves the result.

    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the resized image.
        width (int): Desired width of the resized image. Default is 217.
        height (int): Desired height of the resized image. Default is 225.

    Returns:
        None
    """
    # Load the input image
    image = cv2.imread(input_image_path)

    if image is None:
        raise ValueError("Input image not found or unable to read.")

    # Resize the image to the specified dimensions
    resized_image = cv2.resize(image, (width, height))

    return resized_image

def calculate_occlusion_area(box, overlay_rect):

    """
    Calculates the area of intersection (occlusion) between two rectangles (bounding boxes)

    Args:
        box (list): Bounding box representing the object's coordinates [x1, y1, x2, y2]
        overlay_rect (list): Overlaying rectangle coordinates [x1, y1, x2, y2]

    Returns:
        int: The area of intersection between the two rectangles. If they do not overlap, returns 0.
    """

    x1 = max(box[0], overlay_rect[0])
    y1 = max(box[1], overlay_rect[1])
    x2 = min(box[2], overlay_rect[2])
    y2 = min(box[3], overlay_rect[3])
    
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    return intersection_width * intersection_height

def template_matching(img, overlap_image):

    """
    Perform template matching to locate a template image within a larger image 
    and return the coordinates of the detected region.

    This function uses OpenCV's template matching algorithm (`cv2.matchTemplate`) 
    with the method `TM_CCOEFF_NORMED` to find the best match of the template 
    (`overlap_image`) within the target image (`img`). If the match exceeds a 
    predefined threshold, it calculates the bounding rectangle of the matched 
    region and overlays it on the target image.

    Args:
        img (numpy.ndarray): The target image in which the template will be searched.
        overlap_image (numpy.ndarray): The template image to locate within the target image.
    """

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(overlap_image, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    threshold = 0.4

    if max_val >= threshold:
        top_left = (max_loc[0], max_loc[1])
        h, w = gray_template.shape[:2]
        bottom_right = (top_left[0] + w + 20, top_left[1] + h)
        overlay_rect = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

    return overlay_rect

def calculate_occlusion(x1, x2, y1, y2, overlay_rect):

    """
    Calculates occlusion:
    occlusion = 3 if occlusion_rate = 100
    occlusion = 2 if 50 <= occlusion_rate < 100 
    occlusion = 1 if 0 < occlusion_rate < 50
    occlusion = 0 if occlusion_rate = 0

    Args:
        x1, x2, y1, y2: object's bounding box
        overlay_rect: occlusion rectangle
    """

    # Calculate occlusion area
    box_area = (x2 - x1) * (y2 - y1)
    occlusion_area = calculate_occlusion_area((x1, y1, x2, y2), overlay_rect)
    occlusion_rate = (occlusion_area / box_area) * 100 

    # Map occlusion rate to visibility state
    occlusion = (3 if occlusion_rate == 100 else 2 if 100 > occlusion_rate >= 50 else 1 if 50 > occlusion_rate > 0 else 0)

    return occlusion



def write_to_file(objects, file_path, frame):
    """
    Write tracking data to a file.
    Each line contains: frame, track_id, label, occlusion, bbox (4 coordinates), height, width.
    """
    with open(file_path, 'w') as f:
        for obj in objects:
            track_id = obj.id
            label = obj.label
            occlusion = obj.occlusion
            bbox = obj.bbox  # [x1, y1, x2, y2]
            height = obj.get_height()
            width = obj.get_width()
            
            # Write a line in the specified format
            f.write(f"{frame} {track_id} {label} {occlusion} "
                    f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} "
                    f"{height} {width}\n")