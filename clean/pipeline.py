import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from sort import Sort
from occlusion import *
from kalman import *
from detection import *
from export_video import *

# Initialize the YOLO model
model = YOLO("yolov8l.pt") 

# Initialize SORT tracker
mot_tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.5)

overlap_image = resize_image("clean\Overlap_image.png",width=217, height=225)

# Define input and obj folders
set_img = 2
if set_img == 1: # third sequence
    output_folder = "Perception_project/clean/outputs1"
    data_folder_1 = "Yolov8_perception/data/view1"  # Folder containing input frames
    data_folder_2 = "Yolov8_perception/data/view2"  # Folder containing input frames
if set_img == 2: # second sequence
    output_folder = "Perception_project/clean/outputs2"
    data_folder_1 = "Yolov8_perception/data/view3"  # Folder containing input frames
    data_folder_2 = "Yolov8_perception/data/view4"  # Folder containing input frames
else: # third sequence
    output_folder = "Perception_project/clean/outputs3"
    data_folder_1 = "Yolov8_perception/data/view5"  # Folder containing input frames
    data_folder_2 = "Yolov8_perception/data/view6"  # Folder containing input frames
os.makedirs(output_folder, exist_ok=True)  # Create the obj folder if it doesn't exist

# Initialize a list to store occluded predictions
tracked_predictions = {}
outputs = []
def iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    if union == 0:
        return 0  # Avoid division by zero
    return intersection / union

# Main loop for processing each frame
for frame_path in sorted(Path(data_folder_1).glob("*.png")):
    # Process the current frame: detects objects, their positions, and labels them
    dets, labels, img, ids = process_frame(frame_path, model, confidence_threshold=0.5)

    # Skip to the next frame if the image failed to load
    if img is None:
        continue

    draw_detected_objects(img, dets, labels)
    
    # Update SORT tracker
    trackers = mot_tracker.update(dets)

    # Finds Wazowski
    overlay_rect, img = find_overlay_rectangle(img, overlap_image, threshold=0.4)

    # Loop and process each tracked object individually
    for i, d in enumerate(trackers): 
        
        label = labels[i]

        # Process each tracked object (Kalman filter and occlusion detection)
        outputs, tracked_predictions = process_tracked_object(
            d=d,
            img=img,
            overlay_rect=overlay_rect,
            frame_path=frame_path,
            label=label,
            outputs=outputs,
            tracked_predictions=tracked_predictions
        )
    
    # Process prediction: draws bounding box if object is occluded, updates occlusion rate, and updates Kalman filter
    for track_id, prediction in tracked_predictions.items():
        tracked_predictions = process_prediction(
            track_id=track_id,
            prediction=prediction,
            frame_name=frame_path.name,
            overlay_rect=overlay_rect,
            img=img,
            tracked_predictions=tracked_predictions,
            label = label
        )

    # Save annotated frame
    print("Saving frame:", frame_path.name)
    output_path = os.path.join(output_folder, frame_path.name)
    cv2.imwrite(output_path, img)

# Specify the folder and output video path
image_folder = "Perception_project/clean/outputs2"  
output_video_path = "Perception_project/clean/video2.avi" 

# Create a video from images with a frame rate of 20 FPS and preview enabled
create_video_from_images(image_folder, output_video_path, frame_rate=20, preview=True)