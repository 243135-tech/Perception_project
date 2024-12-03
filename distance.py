import numpy as np
from scipy.optimize import linear_sum_assignment

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
def calculate_distance(left_boxes, right_boxes, focal_length, baseline):
    # Match boxes using Hungarian algorithm
    left_indices, right_indices = match_boxes(left_boxes, right_boxes)
    
    distances = []
    
    for left_idx, right_idx in zip(left_indices, right_indices):
        left_box = left_boxes[left_idx]
        right_box = right_boxes[right_idx]
        
        # Calculate disparity (difference in x-coordinates between left and right)
        disparity = left_box[0] - right_box[0]  # x1_left - x1_right
        
        # Ensure we have a non-zero disparity (avoid division by zero)
        if disparity != 0:
            # Calculate depth (Z = (focal_length * baseline) / disparity)
            depth = (focal_length * baseline) / disparity
            distances.append((left_box[1], right_box[1], depth))  # Append labels and distance
        else:
            distances.append((left_box[1], right_box[1], float('inf')))  # Set as infinity if no disparity (avoid division by zero)
    
    return distances


