import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import math

focal_length, baseline = 1063, 0.6
camera_mtx_left = np.array([[1.16250155e+03, 0.00000000e+00, 6.95968880e+02],
                            [0.00000000e+00, 1.13544076e+03, 2.56041949e+02],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_mtx_right = np.array([[963.70857227, 0.00000000e+00, 695.55992952],
                            [0.00000000e+00, 968.17648905, 254.82848325],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

R = np.array([[0.9912061811223318, -0.014465381388326603, 0.13153349096778827],
              [0.048329135491758615, 0.9649137269349327, -0.2580809838697095], 
              [-0.12318523112574098, 0.2621683663484751, 0.9571275497647476]])

T = np.array([[-26.98804619101692], [35.77753268478152], [-23.27277151603797]])

# Function to compute the Euclidean distance between two bounding box centers
def euclidean_distance(box1, box2):
    center1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2  # (x, y)
    center2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

# Function to compute the cost matrix for matching boxes
def compute_cost_matrix(left_boxes, right_boxes):
    cost_matrix = np.zeros((len(left_boxes), len(right_boxes)))
    for i, left_box in enumerate(left_boxes):
        for j, right_box in enumerate(right_boxes):
            cost_matrix[i, j] = euclidean_distance(left_box[:4], right_box[:4])  # Only compare the coordinates, not the label
    return cost_matrix

# Function to match boxes using the Hungarian algorithm
def match_boxes(left_boxes, right_boxes):
    cost_matrix = compute_cost_matrix(left_boxes, right_boxes)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

# Function to calculate distance (depth) using triangulation
def calculate_distance(left_boxes, right_boxes):
    # Match boxes using Hungarian algorithm
    left_indices, right_indices = match_boxes(left_boxes, right_boxes)
    
    distances = []
    
    for left_idx, right_idx in zip(left_indices, right_indices):
        left_box = left_boxes[left_idx]
        right_box = right_boxes[right_idx]
        
        # Calculate center points
        left_center_x = (left_box[0] + left_box[2]) / 2
        right_center_x = (right_box[0] + right_box[2]) / 2
        print(left_box)
        # Calculate disparity (difference in x-coordinates of the centers)
        disparity = left_center_x - right_center_x  # x1_left - x1_right
        # Ensure we have a non-zero disparity (avoid division by zero)
        if disparity > 0:
            # Calculate depth (Z = (focal_length * baseline) / disparity)
            depth = (focal_length * baseline) / disparity
            distances.append((left_box[1], right_box[1], depth))  # Append labels and distance
        else:
            distances.append((left_box[1], right_box[1], float('inf')))  # Set as infinity if no disparity (avoid division by zero)
    
    return distances

def create_depth_map(image_left, image_right, bbox_left, bbox_right):
    
    # Convert images to grayscale if needed
    if len(image_left.shape) == 3:
        gray_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    else:
        gray_left = image_left

    if len(image_right.shape) == 3:
        gray_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    else:
        gray_right = image_right

    # Crop the regions of interest using the bounding boxes
    roi_left = gray_left[int(bbox_left[1]):int(bbox_left[3]), int(bbox_left[0]):int(bbox_left[2])]
    roi_right = gray_right[int(bbox_right[1]):int(bbox_right[3]), int(bbox_right[0]):int(bbox_right[2])]


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
    roi_right = cv2.resize(roi_right, (roi_left.shape[1], roi_left.shape[0]))
    disparity_map = stereo.compute(roi_left, roi_right).astype(np.float32) / 16.0

    # Replace invalid disparities with NaN
    disparity_map[disparity_map <= 0] = np.nan

    # Calculate the depth map using the disparity map
    depth_map = (baseline * focal_length) / disparity_map

    return depth_map



def get_bearing_from_stereo(left_boxes, right_boxes):
    '''
    Calculate bearings and depths for objects in stereo images.
    
    camera_mtx: Intrinsics array of the left camera: fx, fy, cx, cy
    left_boxes, right_boxes: Detected bounding boxes in left and right images [(x1, y1, x2, y2), ...]
    focal_length: Focal length of the camera
    baseline: Distance between left and right camera (in meters)
    '''
    fx_left = camera_mtx_left[0, 0]
    cx_left = camera_mtx_left[0, 2]
    
    # Match boxes using Hungarian algorithm
    left_indices, right_indices = match_boxes(left_boxes, right_boxes)
    
    results = []
    
    for left_idx, right_idx in zip(left_indices, right_indices):
        left_box = left_boxes[left_idx]
        right_box = right_boxes[right_idx]
        
        # Calculate disparity (difference in x-coordinates between left and right)
        disparity = (left_box[0] + left_box[2]) / 2 - (right_box[0] + right_box[2]) / 2  # Center x-coordinates
        
        # Ensure we have a non-zero disparity (avoid division by zero)
        if disparity != 0:
            depth = (focal_length * baseline) / disparity
            
            # Bearing calculation from the left image
            x = (left_box[0] + left_box[2]) / 2  # Center x-coordinate of the left box
            norm_x = (x - cx_left) / fx_left
            bearing = np.arctan(norm_x)  # Horizontal angle in radians
            
            # Adjust bearing to be in range [-pi, pi]
            bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
            
            # Append results: (bearing, depth)
            results.append((left_box[1], right_box[1], bearing))

        else:
            results.append((left_box[1], right_box[1], np.nan))
    
    return results


def calculate_rotation_y(left_boxes, right_boxes):

    # Extract intrinsic parameters from the camera matrix
    fx_left = camera_mtx_left[0, 0]
    cx_left = camera_mtx_left[0, 2]

    # Match boxes using Hungarian algorithm
    left_indices, right_indices = match_boxes(left_boxes, right_boxes)

    results = []

    for left_idx, right_idx in zip(left_indices, right_indices):
        left_box = left_boxes[left_idx]
        right_box = right_boxes[right_idx]

        # Calculate disparity (difference in x-coordinates between left and right)
        disparity = (left_box[0] + left_box[2]) / 2 - (right_box[0] + right_box[2]) / 2  # Center x-coordinates
        
        # Ensure we have a non-zero disparity (avoid division by zero)
        if disparity != 0:

            depth = (focal_length * baseline) / disparity

            # Back-project the bounding box center to normalized image coordinates
            u_center = (left_box[0] + left_box[2]) / 2  # Center x-coordinate of the bounding box
            x_norm = (u_center - cx_left) / fx_left  # Normalize using intrinsic parameters

            # Compute the 3D coordinates of the object's center in the camera frame
            x_camera = x_norm * depth
            z_camera = depth

            # Calculate the yaw angle (rotation_y) using arctan2
            rotation_y = math.atan2(x_camera, z_camera)

            # Ensure the result is in the range [-pi, pi]
            rotation_y = (rotation_y + math.pi) % (2 * math.pi) - math.pi

            # Append results: (left_label, right_label, rotation_y)
            results.append((left_box[1], right_box[1], rotation_y))
        else:
            # Handle cases with zero disparity (undefined depth or rotation)
            results.append((left_box[1], right_box[1], np.nan))

    return results
    
def camera_to_world(pixel_coords):
    
    # Convert pixel_coords to homogeneous coordinates
    camera_coords_h = np.append(pixel_coords, 1)  # Add 1 for homogeneous coordinates

    # Construct extrinsic matrix T_cw
    T_cw = np.eye(4)
    T_cw[:3, :3] = R
    T_cw[:3, 3] = T.flatten()

    # Transform to world coordinates
    world_coords_h = np.dot(T_cw, camera_coords_h)

    # Convert back from homogeneous to 3D
    world_coords = world_coords_h[:3]
    
    return world_coords

def get_height(left_boxes, right_boxes, camera_mtx_left = np.array([[1.16250155e+03, 0.00000000e+00, 6.95968880e+02],
                            [0.00000000e+00, 1.13544076e+03, 2.56041949e+02],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), focal_length = 1063, baseline= 0.6):

    # Extract intrinsic parameters from the camera matrix
    fx_left = camera_mtx_left[0, 0]
    cx_left = camera_mtx_left[0, 2]
    fy_left = camera_mtx_left[1, 1]
    cy_left = camera_mtx_left[1, 2]

    # Match boxes using Hungarian algorithm
    left_indices, right_indices = match_boxes(left_boxes, right_boxes)

    results = []

    for left_idx, right_idx in zip(left_indices, right_indices):
        left_box = left_boxes[left_idx]
        right_box = right_boxes[right_idx]

        # Calculate disparity (difference in x-coordinates between left and right)
        disparity = (left_box[0] + left_box[2]) / 2 - (right_box[0] + right_box[2]) / 2  # Center x-coordinates

        # Ensure we have a non-zero disparity (avoid division by zero)
        if disparity != 0:
            # Calculate depth (Z = (focal_length * baseline) / disparity)
            depth = (focal_length * baseline) / disparity

            # Calculate 3D camera coordinates for the top of the bounding box (y1)
            u_center = (left_box[0] + left_box[2]) / 2  # Center x-coordinate
            v_top = left_box[1]  # Top y-coordinate (y1)
            x_norm_top = (u_center - cx_left) / fx_left
            y_norm_top = (v_top - cy_left) / fy_left
            x_camera_top = x_norm_top * depth
            y_camera_top = y_norm_top * depth
            z_camera_top = depth

            # Calculate 3D camera coordinates for the bottom of the bounding box (y2)
            v_bottom = left_box[3]  # Bottom y-coordinate (y2)
            y_norm_bottom = (v_bottom - cy_left) / fy_left
            x_camera_bottom = x_norm_top * depth  # Same x_norm for the same u_center
            y_camera_bottom = y_norm_bottom * depth
            z_camera_bottom = depth

            # Transform to world coordinates
            camera_coords_top = np.array([x_camera_top, y_camera_top, z_camera_top])
            camera_coords_bottom = np.array([x_camera_bottom, y_camera_bottom, z_camera_bottom])

            world_coords_top = camera_to_world(camera_coords_top)
            world_coords_bottom = camera_to_world(camera_coords_bottom)

            # Calculate height as the difference in world y-coordinates
            height = world_coords_top[1] - world_coords_bottom[1]

            # Append results: (left_label, right_label, height)
            results.append((left_box[1], right_box[1], height))
        else:
            # Handle cases with zero disparity (undefined depth or height)
            results.append((left_box[1], right_box[1], None))

    return results


def get_width(left_boxes, right_boxes):

    # Extract intrinsic parameters from the camera matrix
    fx_left = camera_mtx_left[0, 0]
    cx_left = camera_mtx_left[0, 2]

    # Match boxes using Hungarian algorithm
    left_indices, right_indices = match_boxes(left_boxes, right_boxes)

    results = []

    for left_idx, right_idx in zip(left_indices, right_indices):
        left_box = left_boxes[left_idx]
        right_box = right_boxes[right_idx]

        # Calculate disparity (difference in x-coordinates between left and right)
        disparity = (left_box[0] + left_box[2]) / 2 - (right_box[0] + right_box[2]) / 2  # Center x-coordinates

        # Ensure we have a non-zero disparity (avoid division by zero)
        if disparity != 0:
            # Calculate depth (Z = (focal_length * baseline) / disparity)
            depth = (focal_length * baseline) / disparity

            # Calculate 3D camera coordinates for the left side of the bounding box (x1)
            u_left = left_box[0]  # x1 from the bounding box
            x_norm_left = (u_left - cx_left) / fx_left
            x_camera_left = x_norm_left * depth
            z_camera_left = depth

            # Calculate 3D camera coordinates for the right side of the bounding box (x2)
            u_right = left_box[2]  # x2 from the bounding box
            x_norm_right = (u_right - cx_left) / fx_left
            x_camera_right = x_norm_right * depth
            z_camera_right = depth

            # Transform to world coordinates
            camera_coords_left = np.array([x_camera_left, 0, z_camera_left])  # 0 for y since it's not needed here
            camera_coords_right = np.array([x_camera_right, 0, z_camera_right])

            world_coords_left = camera_to_world(camera_coords_left)
            world_coords_right = camera_to_world(camera_coords_right)

            # Calculate width as the difference in world x-coordinates
            width = abs(world_coords_right[0] - world_coords_left[0])

            # Append results: (left_label, right_label, width)
            results.append((left_box[1], right_box[1], width))
        else:
            # Handle cases with zero disparity (undefined depth or width)
            results.append((left_box[1], right_box[1], None))

    return results

def get_length(left_boxes, right_boxes, image_left, image_right):

    # Match boxes using Hungarian algorithm
    left_indices, right_indices = match_boxes(left_boxes, right_boxes)

    results = []

    for left_idx, right_idx in zip(left_indices, right_indices):
        left_box = left_boxes[left_idx]
        right_box = right_boxes[right_idx]

        # Crop the bounding boxes from the images
        bbox_left = (left_box[0], left_box[1], left_box[2], left_box[3])
        bbox_right = (right_box[0], right_box[1], right_box[2], right_box[3])

        # Generate depth map for the object using stereo images and bounding boxes
        object_depth = create_depth_map(image_left, image_right, bbox_left, bbox_right)

        # Filter out invalid depth values (e.g., NaN)
        valid_depth = object_depth[~np.isnan(object_depth)]

        # Ensure there are valid depth values to calculate length
        if valid_depth.size > 0:
            # Get maximum and minimum depth values
            max_depth = np.max(valid_depth)
            min_depth = np.min(valid_depth)

            # Calculate the length of the object along the z-axis (depth)
            object_length = max_depth - min_depth

            # Append results: (left_label, right_label, length)
            results.append((left_box[1], right_box[1], object_length))
        else:
            # Handle cases where no valid depth values are available
            results.append((left_box[1], right_box[1], None))

    return results

def get_parameters(i,distances,bearings,rotations,heights,widths,lengths):
        if i >= len(distances):
            distance = 'inf'
        else:
            distance = distances[i][2]

        if i >= len(bearings):
            bearing = 'inf'
        else:
            bearing = bearings[i][2]

        if i >= len(rotations):
            rotation = 'inf'
        else:
            rotation = rotations[i][2]

        if i >= len(heights):
            height = 'inf'
        else:
            height = heights[i][2]  

        if i >= len(widths):
            width = 'inf'
        else:
            width = widths[i][2]   

        if i >= len(lengths):
            length = 'inf'
        else:
            length = lengths[i][2] 
    
        return distance, bearing, rotation, height, width, length
