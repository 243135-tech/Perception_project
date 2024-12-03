import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from sort import Sort
from kalman import DetectedObjects
from imageProcessing import *
from outputs import *
from distance import *

def main():

    # Initialize the YOLO model
    model = YOLO("yolov8l.pt")   # Moved to parameters 

    # Initialize SORT tracker
    mot_tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.5) # Not important for yolo

    overlap_image = resize_image("Overlap_image.png",width=217, height=225)

    # Define input and obj folders
    set_img = 1

    # Create obj folder
    data_folder_1, data_folder_2 = output(set_img)

    # List of detected objects
    objs = []

    output_file = "outputs.txt"

    for frame_path in sorted(Path(data_folder_1).glob("*.png")):
        
        img = cv2.imread(str(frame_path))
        # Get file name
        file_name = frame_path.name
        # Strip leading zeros and remove the ".png" extension
        file_name = file_name.lstrip("0").replace(".png", "")

        if img is None:
            print(f"Error: Could not load image {frame_path.name}")
            continue

        # Perform object detection
        results = model(img, conf=0.5, classes=[0, 1, 2, 7])
        detections = results[0].boxes

        dets = []
        labels = []

        # Process each detection
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            labels.append(label)

            if confidence < 0.5 or label not in ["person", "car", "truck"]:
                continue

            dets.append([x1, y1, x2, y2, confidence])

        dets = np.array(dets)

        # Update SORT tracker
        trackers = mot_tracker.update(dets)

        # Matching the overlaping template 
        overlay_rect = template_matching(img, overlap_image)

        # Create an ids list to keep track of ids
        ids = []

        # Process each tracked object
        for i, d in enumerate(trackers): 

            # Get coordinates, id and label
            x1, y1, x2, y2, track_id = map(int, d)
            label = labels[i]

             # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) 
            cv2.putText(img, f": {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 

            # Calculate occlusion (from 0 (totally visible) to 3 (not visible))
            occlusion = calculate_occlusion(x1, x2, y1, y2, overlay_rect)

            # Compute object center
            x, y = get_center(x1, y1, x2, y2)

            # Create new object if it does not exist already
            if track_id not in ids:

                pos = np.array([x, 0, 0, y, 0, 0])
                det_obj = DetectedObjects(pos, track_id, file_name, label, [x1, y1, x2, y2], occlusion)
                objs.append(det_obj)
                ids.append(track_id)


            # If the object is completely visible do a smooth update
            if occlusion == 0:
                Z = np.array([x, y])
                det_obj.smooth_update(Z)

            for obj in objs:

                obj.predict()
                x, y = int(obj.x[0]), int(obj.x[3])
                width, height = obj.get_width(), obj.get_height()

                # Calculate updated bounding box
                new_x1 = int(x - width / 2)
                new_x2 = int(x + width / 2)
                new_y1 = int(y + height / 2)
                new_y2 = int(y - height / 2)

                bbox = [new_x1, new_y1, new_x2, new_y2]

                # Recalculate occlusion area
                occlusion = calculate_occlusion(new_x1, new_x2, new_y1, new_y2, overlay_rect)

                obj.bbox = bbox
                obj.occlusion = occlusion

                if obj.id == track_id and obj.occlusion > 0:
                    cv2.rectangle(img, (new_x1, new_y1), (new_x2, new_y2), (0, 0, 255), 2)
                    cv2.putText(img, f"Pred: {track_id}", (new_x1, new_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        write_to_file(objs, output_file, file_name)

if __name__ == '__main__':
    main()