import numpy as np
from occlusion import *

def initialize_kalman():
    kalman = {
        "x": np.array([0,
              0,
              0,
              0]),  # State vector
        "P": 1000 * np.eye(4),  # Initial uncertainty, a random high number
        "F":  np.array([[1, 1, 0, 0],  # x_pos
                        [0, 1, 0, 0],  # x_vel
                        [0, 0, 1, 1], # y_pos
                        [0, 0, 0, 1]]),  # y_vel # Transition matrix
        "u": np.zeros(4),  # External motion
        "H": np.array([[1, 0, 0, 0],  # Observe x position
                       [0, 0, 1, 0]]),  # Observe y position
        "R": 10 * np.eye(2),  # Measurement uncertainty
        "I": np.eye(4)  # Identity matrix
    }
    return kalman
def update(kalman, Z):
    x, P, H, R, I = kalman["x"], kalman["P"], kalman["H"], kalman["R"], kalman["I"]
    
    # Measurement residual y
    y = Z - np.dot(H, x)
    
    # Residual covariance S
    S = np.dot(H, np.dot(P, H.T)) + R
    
    # Kalman gain K
    K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))

    # Update state estimate x
    x = x + np.dot(K, y)
    
    # Update uncertainty P
    P = np.dot(I - np.dot(K, H), P)
    
    kalman["x"], kalman["P"] = x, P
    return kalman

def predict(kalman):
    x, P, F, u = kalman["x"], kalman["P"], kalman["F"], kalman["u"]
    
    Q = np.eye(4) * 0.1  # small noise

    # Predict state x
    x = np.dot(F, x) + u

    # Hardcoded increase in velocity
    x[1] = x[1] * 1.03
    x[3] = x[3] * 1.03
    
    # Predict uncertainty P
    P = np.dot(F, np.dot(P, F.T)) + Q
    
    kalman["x"], kalman["P"] = x, P
    return kalman

def new_kalman(track_id, tracked_predictions, x1, y1, x2, y2, occlusion_rate):
    if track_id not in tracked_predictions:
            kalman = initialize_kalman()
            tracked_predictions[track_id] = {"kalman": kalman, "width": x2 - x1, "height": y2 - y1, "occlusion_rate":occlusion_rate,}
    return tracked_predictions

def process_prediction(track_id, prediction, frame_name, overlay_rect, img, tracked_predictions,labels_dict,outputs):

    """
    Process a single Kalman filter prediction, update bounding box, and recalculate occlusion.
    """

    # Predict the next state using Kalman filter
    kalman = prediction['kalman']
    kalman = predict(kalman) 

    # Print predicted position and velocity
    predicted_position = (kalman['x'][0], kalman['x'][2])
    predicted_velocity = (kalman['x'][1], kalman['x'][3])
    print(f"Predicted position for {track_id} in {frame_name} is {predicted_position}")
    print(f"Predicted velocity for {track_id} in {frame_name} is {predicted_velocity}")

    # Update bounding box using the predicted position
    x, y = kalman['x'][0], kalman['x'][2]
    width = prediction["width"]
    height = prediction["height"]
    new_x1 = int(x - width / 2)
    new_x2 = int(x + width / 2)
    new_y1 = int(y - height / 2)
    new_y2 = int(y + height / 2)

    # Update the Kalman filter in tracked_predictions
    tracked_predictions[track_id]['kalman'] = kalman

    # Recalculate occlusion area
    box_area = width * height
    occlusion_area = calculate_occlusion_area((new_x1, new_y1, new_x2, new_y2), overlay_rect)
    occlusion_rate = (occlusion_area / box_area) * 100
    print(f"New Occlusion_rate for {track_id} in {frame_name} is {occlusion_rate}")

    # Update the occlusion rate in tracked_predictions
    tracked_predictions[track_id]['occlusion_rate'] = occlusion_rate

    # Find the label corresponding to the track ID
    label = labels_dict.get(track_id, "unknown")
        
    # Call create_outputs to store the result
    outputs = create_outputs(
            outputs, occlusion_rate, frame_name, label, track_id,
            new_x1, new_y1, new_x2, new_y2, x, y, 'o','o','o','o','o','o')

    # Draw predictions on the image if occlusion conditions are met
    if occlusion_rate > 0 and box_area > 3000:
        cv2.rectangle(img, (new_x1, new_y1), (new_x2, new_y2), (0, 0, 255), 2)
        cv2.putText(img, f"Pred: {track_id}", (new_x1, new_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return outputs, tracked_predictions

def process_tracked_object(d, img, overlay_rect, frame_path, label, outputs, tracked_predictions, 
                           distance, bearing, rotation, height, width, length):
    """
    Processes a single tracked object, updating its occlusion and Kalman filter states.
    """
    # Extract tracking information)

    x1, y1, x2, y2, track_id = map(int, d)
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    # Draw the bounding box and add the track ID
    #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
    #cv2.putText(img, f": {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate the occlusion area and rate
    box_area = (x2 - x1) * (y2 - y1)
    occlusion_area = calculate_occlusion_area((x1, y1, x2, y2), overlay_rect)
    occlusion_rate = (occlusion_area / box_area) * 100 if box_area > 0 else 0

    # Add occlusion information to outputs
    if label == 'unknow':
        label = 'kalman_pred'
    
    outputs = create_outputs(outputs, occlusion_rate, frame_path, label, track_id, x1, y1, x2, y2, x_center, y_center,
                             distance, bearing, rotation, height, width, length)

    # Check if this box is already being tracked
    if track_id not in tracked_predictions:
        tracked_predictions = new_kalman(track_id, tracked_predictions, x1, y1, x2, y2, occlusion_rate)

    # Update Kalman filter state if not occluded
    if occlusion_area == 0:
        Z = np.array([x_center, y_center])  # Current observation
        tracked_predictions[track_id]['kalman'] = update(tracked_predictions[track_id]['kalman'], Z)
        # Update dimensions in tracked_predictions
        tracked_predictions[track_id]['width'] = x2 - x1
        tracked_predictions[track_id]['height'] = y2 - y1

    return outputs, tracked_predictions
