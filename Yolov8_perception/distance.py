import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

def calculate_distance_to_object(camera_mtx, bbox_left, bbox_right):
    """
    Calculate the distance to an object using bounding box coordinates from left and right images,
    along with the camera calibration parameters.
    
    Parameters:
        bbox_left (list): Bounding box coordinates of left image (x_left, y_left, x_right, y_right)
        bbox_right (list): Bounding box coordinates of right image (x_left, y_left, x_right, y_right)
        camera_mtx (array): Camera intrinsics
        
    Returns:
        float: Distance to the object (in meters)
        float: Disparity between the left and right images (in pixels)
    """
    baseline = 0.06

    # Get the center of the bounding box in the left and right images
    center_left = ((bbox_left[0] + bbox_left[2]) / 2, (bbox_left[1] + bbox_left[3]) / 2)
    center_right = ((bbox_right[0] + bbox_right[2]) / 2, (bbox_right[1] + bbox_right[3]) / 2)

    # Calculate disparity (horizontal pixel difference between the left and right image
    x_left = center_left[0]
    x_right = center_right[0]

    disparity = abs(x_left - x_right)
    
    if disparity == 0:
        return float('inf'), disparity  # If disparity is zero, the object is too far away or at the same location
    
    # Calculate the distance to the object using the formula
    focal_length = camera_mtx[0, 0]
    distance = (focal_length * baseline) / disparity
    

    return distance


def create_depth_map(camera_mtx, image_left, bbox_left, image_right, bbox_right):
    """
    Create a depth map for an object using stereo images and bounding boxes.

    Args:
        camera_mtx (numpy.ndarray): Camera intrinsic matrix (3x3).
        image_left (numpy.ndarray): Left image (grayscale or color).
        bbox_left (tuple): Bounding box in the left image (x_min, y_min, x_max, y_max).
        image_right (numpy.ndarray): Right image (grayscale or color).
        bbox_right (tuple): Bounding box in the right image (x_min, y_min, x_max, y_max).

    Returns:
        numpy.ndarray: Depth map (in meters) for the object within the bounding box.
    """
    # Baseline in meters
    baseline = 0.06  # Fixed at 6 cm

    # Extract focal length from the camera intrinsic matrix
    focal_length = camera_mtx[0, 0]  # Assuming fx (focal length in pixels)

    # Convert images to grayscale if needed
    if len(image_left.shape) == 3:
        image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    if len(image_right.shape) == 3:
        image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

    # Crop the regions of interest (ROIs) from the bounding boxes
    x_min_left, y_min_left, x_max_left, y_max_left = bbox_left
    x_min_right, y_min_right, x_max_right, y_max_right = bbox_right

    roi_left = image_left[y_min_left:y_max_left, x_min_left:x_max_left]
    roi_right = image_right[y_min_right:y_max_right, x_min_right:x_max_right]

    # Stereo block matching using cv2.StereoSGBM_create
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,  # Minimum possible disparity value. 
                        # Usually set to 0, but if the object is closer to the camera and disparities are expected to be large, increase this value.
        
        numDisparities=16 * 5,  # Maximum disparity minus minimum disparity.
                                # Must be divisible by 16. Increasing this value allows matching objects farther away (lower disparity), but increases computation time.
        
        blockSize=9,  # Size of the matching block (odd number).
                    # Larger values improve robustness to noise but may blur fine details. Smaller values provide better detail but are more sensitive to noise.
        
        P1=8 * 3 * 9 ** 2,  # Penalty for small changes in disparity (smoothness constraint).
                            # Increase for smoother disparity maps, especially in regions with gradual depth changes. 
                            # Use lower values for scenes with high texture variation.
        
        P2=32 * 3 * 9 ** 2,  # Penalty for large changes in disparity (discontinuity constraint).
                            # Higher values make the algorithm less sensitive to abrupt depth changes, improving smoothness but potentially losing sharp edges.
        
        disp12MaxDiff=1,  # Maximum allowed difference between left-right and right-left disparity computations.
                        # Smaller values improve accuracy but can lead to gaps in the disparity map.
        
        uniquenessRatio=10,  # Margin by which the best match should be better than the second-best match (in percentage).
                            # Lower values make the algorithm more permissive, increasing noise but capturing more detail. Higher values reduce noise but may miss details.
        
        speckleWindowSize=100,  # Maximum size of connected components considered as speckles (noise).
                                # Increase to remove larger noise patches in disparity maps. Smaller values may leave speckles but preserve details.
        
        speckleRange=32,  # Maximum disparity variation within connected components (speckle filtering threshold).
                        # Lower values remove speckles more aggressively, but this can also remove valid details. Increase to preserve more details.
    )

    # Compute disparity map for the cropped regions
    disparity_map = stereo.compute(roi_left, roi_right).astype(np.float32) / 16.0

    # Replace invalid disparities with NaN
    disparity_map[disparity_map <= 0] = np.nan

    # Calculate the depth map using the disparity map
    depth_map = (baseline * focal_length) / disparity_map

    return depth_map



def get_bearing(camera_mtx, bbox):
    '''
    Calculation of bearing from object relative to the camera
    camera_mtx is an array with camera's intrinsics: fx, fy, cx, cy
    object_coord are the coordinates of the object in the image frame: u, v
    '''
    fx = camera_mtx[0, 0]
    cx = camera_mtx[0, 2]
    x = (bbox[0] + bbox[2]) / 2
    norm_x = (x - cx) / fx
    bearing = np.arctan(norm_x)

    bearing = (bearing + np.pi) % (2 * np.pi) - np.pi

    return bearing


def calculate_rotation_y(camera_matrix, bbox, depth):
    """
    Calculate the rotation y (yaw angle) of an object relative to the camera.
    
    Parameters:
        camera_matrix (np.array): 3x3 intrinsic camera matrix.
        depth (float): Depth (z-coordinate) of the object in the camera frame (meters).
        image_center (tuple): (u, v) in pixel coordinates.
        
    Returns:
        float: Rotation y (yaw angle) in radians, in range [-pi, pi].
    """
    # Extract intrinsic parameters from the camera matrix
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Calculate the center of the bounding box in image coordinates
    u_center = (bbox[0] + bbox[2]) / 2
    v_center = (bbox[1] + bbox[3]) / 2
    
    # Back-project the bounding box center to normalized image coordinates
    x_norm = (u_center - cx) / fx
    y_norm = (v_center - cy) / fy
    
    # Compute the 3D coordinates of the object's center in the camera frame
    x_camera = x_norm * depth
    z_camera = depth  # Depth is already given
    
    # Calculate the rotation_y (yaw angle) using arctan2
    rotation_y = math.atan2(x_camera, z_camera)  # Yaw angle in radians
    
    # Ensure the result is in the range [-pi, pi]
    rotation_y = (rotation_y + math.pi) % (2 * math.pi) - math.pi
    
    return rotation_y


def camera_to_world(camera_coords, R, t):
    """
    Transform camera coordinates to world coordinates.

    Parameters:
        camera_coords (np.array): 3D point in camera coordinates (x_c, y_c, z_c).
        R (np.array): 3x3 rotation matrix (camera-to-world rotation).
        t (np.array): 3x1 translation vector (camera-to-world translation).
    
    Returns:
        np.array: 3D point in world coordinates (x_w, y_w, z_w).
    """
    # Convert camera_coords to homogeneous coordinates
    camera_coords_h = np.append(camera_coords, 1)  # Add 1 for homogeneous coordinates

    # Construct extrinsic matrix T_cw
    T_cw = np.eye(4)
    T_cw[:3, :3] = R
    T_cw[:3, 3] = t.flatten()

    # Transform to world coordinates
    world_coords_h = np.dot(T_cw, camera_coords_h)

    # Convert back from homogeneous to 3D
    world_coords = world_coords_h[:3]
    
    return world_coords


def get_height(bbox):
    """
    Calculate the height of an object in the world coordinate system.

    Args:
        bbox (tuple): Bounding box coordinates in the image 
                      (x_min, y_min, x_max, y_max) format.

    Returns:
        float: Height of the object in the world coordinate system. 
               Returns NaN if height is negative.
    """
    # Put the actual Rotation and Translation values here from camera calibration
    R = np.eye(3)  # Example: Identity matrix (replace with calibrated rotation matrix)
    T = np.array([1, 1, 1]).reshape(-1, 1)  # Example: Translation vector (replace with calibrated values)

    # Extract top and bottom corners of the bounding box
    top_corner = np.array([bbox[0], bbox[1]])
    bottom_corner = np.array([bbox[2], bbox[3]])

    # Transform the corners from camera to world coordinates
    top = camera_to_world(top_corner, R, T)
    bottom = camera_to_world(bottom_corner, R, T)

    # Calculate the height (difference in the y-axis)
    height = top[1] - bottom[1]

    # Handle cases where height is invalid (negative)
    if height < 0:
        print("Negative height")
        return np.nan

    return height


def get_width(bbox):
    """
    Calculate the width of an object in the world coordinate system.

    Args:
        bbox (tuple): Bounding box coordinates in the image 
                      (x_min, y_min, x_max, y_max) format.

    Returns:
        float: Width of the object in the world coordinate system.
    """
    # Put the actual Rotation and Translation values here from camera calibration
    R = np.eye(3)  # Example: Identity matrix (replace with calibrated rotation matrix)
    T = np.array([1, 1, 1]).reshape(-1, 1)  # Example: Translation vector (replace with calibrated values)

    # Extract top and bottom corners of the bounding box
    top_corner = np.array([bbox[0], bbox[1]])
    bottom_corner = np.array([bbox[2], bbox[3]])

    # Transform the corners from camera to world coordinates
    top = camera_to_world(top_corner, R, T)
    bottom = camera_to_world(bottom_corner, R, T)

    # Calculate the width (absolute difference in the x-axis)
    width = abs(top[0] - bottom[0])

    return width



def get_length(camera_mtx, image_left, bbox_left, image_right, bbox_right):
    """
    Calculate the length of an object along the z-axis (depth).

    Args:
        camera_mtx (numpy.ndarray): Camera intrinsic matrix (3x3).
        image_left (numpy.ndarray): Left stereo image (grayscale or color).
        bbox_left (tuple): Bounding box in the left image (x_min, y_min, x_max, y_max).
        image_right (numpy.ndarray): Right stereo image (grayscale or color).
        bbox_right (tuple): Bounding box in the right image (x_min, y_min, x_max, y_max).

    Returns:
        float: Length of the object along the z-axis in meters.
    """
    # Generate depth map for the object using stereo images and bounding boxes
    object_depth = create_depth_map(camera_mtx, image_left, bbox_left, image_right, bbox_right)

    # Filter out invalid depth values (e.g., 0 or NaN)
    valid_depth = object_depth[object_depth > 0]

    # Ensure there are valid depth values to calculate length
    if valid_depth.size == 0:
        print("No valid depth values found")
        return np.nan

    # Get maximum and minimum depth values
    max_depth = np.max(valid_depth)
    min_depth = np.min(valid_depth)

    # Calculate the length of the object along the z-axis (depth)
    object_length = max_depth - min_depth

    return object_length