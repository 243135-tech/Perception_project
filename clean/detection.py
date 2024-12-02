import cv2
import numpy as np


def process_frame(frame_path, model, confidence_threshold=0.5, valid_classes=["person", "car", "truck", "pedestrian"]):
    """
    Loads an image, performs object detection, and processes detections.
    """

    # Load the image
    img = cv2.imread(str(frame_path))
    ids = []
    
    # Check if the image is loaded correctly
    if img is None:
        print(f"Error: Could not load image {frame_path.name}")
        return None, None, None, None

    # Perform object detection
    results = model(img, conf=confidence_threshold, classes=[0, 1, 2, 7])
    detections = results[0].boxes

    dets = []
    labels = []

    # Log processing details
    print("Processing frame:", frame_path.name)
    print("Number of detections:", len(detections))

    # Process each detection
    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        labels.append(label)

        # Filter detections based on confidence and valid classes
        if confidence < confidence_threshold or label not in valid_classes:
            continue

        # Append valid detection
        dets.append([x1, y1, x2, y2, confidence])

    # Convert detections to NumPy array
    dets = np.array(dets)

    return dets, labels, img, ids