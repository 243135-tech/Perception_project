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

def draw_detected_objects(img, dets, labels):
    for det, label in zip(dets, labels):
        x1, y1, x2, y2, confidence = map(int, det)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(img, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)