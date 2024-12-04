import cv2

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
    x1 = max(box[0], overlay_rect[0])
    y1 = max(box[1], overlay_rect[1])
    x2 = min(box[2], overlay_rect[2])
    y2 = min(box[3], overlay_rect[3])
    
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    return intersection_width * intersection_height

def create_outputs(outputs,occlusion_rate,frame_path,label,track_id,x1,y1,x2,y2,x_center,y_center,
                   distance, bearing, rotation, height, width, length):
    if occlusion_rate == 100:
            outputs.append((frame_path.name,label,track_id, x1, y1, x2, y2, x_center, y_center,distance,
                            bearing, rotation, height, width, length ,3)) # not visible at all
    if occlusion_rate<100 and occlusion_rate >= 50:
            outputs.append((frame_path.name,label,track_id, x1, y1, x2, y2, x_center, y_center,distance,
                            bearing, rotation, height, width, length, 2)) # partially visible
    if occlusion_rate<50 and occlusion_rate > 0:
            outputs.append((frame_path.name,label,track_id, x1, y1, x2, y2, x_center, y_center,distance,
                            bearing, rotation, height, width, length, 1)) # mostly visible
    else:
            outputs.append((frame_path.name,label,track_id, x1, y1, x2, y2, x_center, y_center,distance,
                            bearing, rotation, height, width, length, 0)) # totally visible
    return outputs

def find_overlay_rectangle(img, overlap_image,set_img, threshold=0.2):
    """
    Finds the overlay rectangle (Mr. Wazovski) in the input image using template matching.
    """

    # Convert both the frame and template to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(overlap_image, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Initialize overlay rectangle as None
    overlay_rect = None

    # Check if the matching score exceeds the threshold
    if set_img == 1:
          overlay_rect = (0,0,0,0)

    elif max_val >= threshold:
        top_left = (max_loc[0], max_loc[1])  # Top-left corner of the match
        h, w = gray_template.shape[:2]
        bottom_right = (top_left[0] + w + 20, top_left[1] + h)  # Add margin to width
        overlay_rect = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

        # Draw the overlay rectangle on the image
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

    return overlay_rect, img