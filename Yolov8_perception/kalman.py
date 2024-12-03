import numpy as np
from numpy import ndarray

class DetectedObjects():

    """
    A class to implement a basic Kalman Filter for state estimation and uncertainty prediction.

    The Kalman Filter is used for tracking or estimating the state of a system by combining noisy sensor measurements 
    with a predictive model. This implementation includes methods for updating the state based on new measurements 
    and predicting the next state based on the system's model.

    Attributes:
        x (numpy.ndarray): The current state vector of shape (6,). Initialized to zeros.
        P (numpy.ndarray): The current state covariance matrix of shape (6, 6). Initialized to a high uncertainty (10000).
    """

    def __init__(self, position, track_id: int, frame, label: str, bbox: list, occlusion: int):
        self.x = np.array(position).reshape(-1, 1)
        self.P = np.eye(6) * 10000
        self.id = track_id
        self.frame = frame
        self.label = label
        self.bbox = bbox
        self.occlusion = occlusion
    
    def get_width(self):
        return abs(self.bbox[2] - self.bbox[0])
    
    def get_height(self):
        return abs(self.bbox[3] - self.bbox[1])
    
    def set_x_coord(self, x):
        self.x[0] = x

    def set_y_coord(self, y):
        self.x[3] = y

    def get_x_coord(self):
        return self.x[0]

    def get_y_coord(self):
        return self.x[3]
    

    def set_bbox(self, bbox):
        if len(bbox) != 4:
            print("Invalid bounding box")
            return
        self.bbox = bbox

    def update(self, measurements: ndarray): # measurements shape (2, 1)

        """
        Updates the state vector (`x`) and state covariance matrix (`P`) using new sensor measurements.

            Args:
                measurements (numpy.ndarray): A measurement vector of shape (2, 1) representing sensor readings.

            Returns:
                Tuple[numpy.ndarray, numpy.ndarray]: Updated state vector and covariance matrix.
        """

        # Define H, R and I
        H = np.array([[1, 0, 0, 0, 0, 0],[0, 0, 0, 1, 0, 0]]).reshape(2, 6)
        R = np.eye(2) * 10
        I = np.eye(6)

        # Define x, P and Z just for simplicity
        x, P = self.x, self.P
        Z = np.array(measurements).reshape(-1, 1)

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

        # update x and P
        self.x, self.P = x, P

    
    def predict(self):

        """
            Predicts the next state vector (`x`) and state covariance matrix (`P`) based on the system dynamics.

            Returns:
                Tuple[numpy.ndarray, numpy.ndarray]: Predicted state vector and covariance matrix.
        """

        # Define F and u
        F = np.array([[1, 1, 0, 0, 0, 0],
                       [0, 1, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 1, 0],
                       [0, 0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 0, 1]])
        
        u = np.zeros(6).reshape(-1, 1)

        # Define x and P just for simplicity
        x, P = self.x, self.P

        Q = np.eye(6) * 0.01  # small noise

        # Predict state x   
        x = np.dot(F, x) + u    

        # Predict uncertainty P 
        P = np.dot(F, np.dot(P, F.T)) + Q

        self.x, self.P = x, P

    
    def smooth_update(self, measurement):

        # Define Z as measurements for simplicity
        Z = measurement

        # Get the previous position coordinates
        prev_x, prev_y = self.x[0], self.x[3]
        
        # Update step
        self.update(measurement)

        # Get position coordinates
        x, y = self.x[0], self.x[3]

        # Define alpha parameter
        alpha = 0.8

        # Utilize alpha to re-calculate position coordinates
        x = alpha * x + (1-alpha) * prev_x
        y = alpha * y + (1-alpha) * prev_y

        # Update position coordinates
        self.set_x_coord(x)
        self.set_y_coord(y)



        



"""
def smooth_update(track_id, x, y):
    kalman = tracked_predictions[track_id]['kalman']
    Z = np.array([x, y])  # Observation update
    tracked_predictions[track_id]['kalman'] = update(kalman, Z)
    alpha = 0.8

    # Smooth prediction
    x_center = alpha * kalman['x'][0] + (1 - alpha) * tracked_predictions[track_id].get('prev_x_center', x)
    y_center = alpha * kalman['x'][3] + (1 - alpha) * tracked_predictions[track_id].get('prev_y_center', y)

    # Store smoothed centers
    tracked_predictions[track_id]['prev_x_center'] = x_center
    tracked_predictions[track_id]['prev_y_center'] = y_center

    return tracked_predictions[track_id]
"""