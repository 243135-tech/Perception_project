import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

def filter_red_color(frame):
    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the red color range
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([7, 255, 255])
    lower_red2 = np.array([173, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the two masks
    mask = mask1 + mask2

    # Apply the mask to get only red parts of the image
    result = cv2.bitwise_and(frame, frame, mask=mask)

    return result

def update(x, P, Z, H, R):
    # Measurement residual y
    y = Z - np.dot(H, x)
    
    # Residual covariance S
    S = np.dot(H, np.dot(P, H.T)) + R
    
    # Kalman gain K
    K = np.dot(P, np.dot(H.T, np.linalg.pinv(S)))
    
    # Update state estimate X
    x = x + np.dot(K, y)
    
    # Update uncertainty P
    P = np.dot(I - np.dot(K, H), P)
    
    return x, P

def predict(x, P, F, u):
    # Predict state X'
    x = np.dot(F, x) + u
    
    # Predict uncertainty P'
    P = np.dot(F, np.dot(P, F.T))
    
    return x, P
    
### Initialize Kalman filter ###
# The initial state (6x1).  - position, velocity and acceleration in both x and y
x = np.array([0,
              0,
              0,
              0,
              0,
              0])

# The initial uncertainty (6x6). - I will just input a random and high number on the main diagonal
P = 1000 * np.eye(6)

# The external motion (6x1). - I assume it will be 0 since we have no external input on the ball trajectory
u = np.array([0,
              0,
              0,
              0,
              0,
              0])

# The transition matrix (6x6).  - explained on paper
F = np.array([[0, 1, 0.5, 0, 0, 0],
              [0, 1, 1, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 1, 0.5],
              [0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 1]])
              

# The observation matrix (2x6). - we will need position as output
H = np.array([[1, 0, 0, 0, 0, 0],  # Observe x position
              [0, 0, 0, 1, 0, 0]]) # Observe y position

# The measurement uncertainty.
R = np.array([[1, 0],
              [0, 1]])

I = np.eye(6)

# Load the video
cap = cv2.VideoCapture('rolling_ball.mp4')
if not cap.isOpened():
    print("Cannot open video")
    exit()

# Sharpening kernel
sharpen_kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])

# Initialize radius with a default value
radius = 0

# Looping through all the frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    ### Detect the ball ###
    # Filter by red color first
    red_frame = filter_red_color(frame)

    # Convert the red frame to grayscale for edge detection
    gray = cv2.cvtColor(red_frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(gray_blurred, 30, 50)

    # Apply HoughCircles to detect circles
    pieces = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 80, param1=100, param2=20, minRadius=40, maxRadius=125)

    # We now display the ball if it was found
    if pieces is not None:
        # Convert the (x, y, radius) values of the circles to integers
        pieces = np.uint16(np.around(pieces))
    
        # Draw the circles on the original image
        circle = pieces[0, 0]
        center = (circle[0], circle[1])
        radius = circle[2]

        cv2.circle(frame, center, radius, (0, 255, 0), 5)  # Green circle with thickness 2
    
        ### If the ball is found, update the Kalman filter ###
        Z = np.array(center)  # Z holds the "measured position" of the ball
        x, P = update(x, P, Z, H, R)
    
    ### Predict the next state
    x, P = predict(x, P, F, u)
    predicted_center = (int(x[0]), int(x[3]))
    ### Draw the predicted state on the image frame ###
    cv2.circle(frame, predicted_center, radius, (0, 0, 255), 5)
    

    # Get original video dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set a desired maximum window width or height, while maintaining the aspect ratio
    desired_width = 1000  # Change this as needed
    
    # Calculate the appropriate height based on the aspect ratio
    aspect_ratio = frame_height / frame_width
    new_height = int(desired_width * aspect_ratio)
    
    # Set up the window for display and resize it, keeping the aspect ratio
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)  # Create a window
    cv2.resizeWindow('Frame', desired_width, new_height)  # Resize keeping the aspect ratio

    # Show the frame
    cv2.imshow('Frame', frame)
    cv2.waitKey(150)
    
cap.release()
cv2.destroyAllWindows()
