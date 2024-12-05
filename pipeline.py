import cv2
import os
from pathlib import Path
from matplotlib import pyplot as plt
from ultralytics import YOLO
from sort import Sort
from occlusion import *
from kalman import *
from detection import *
from export_video import *
from distance import *



# Initialize the YOLO model
model = YOLO("yolov8l.pt") 

# Initialize SORT tracker
mot_tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.5)

overlap_image = resize_image("Overlap_image.png", width=217, height=225)

# Define input and obj folders
set_img = 2
if set_img == 1: # third sequence
    output_folder = "outputs1"
    data_folder_1 = "Yolov8_perception/data/view1"  # Folder containing input frames
    data_folder_2 = "Yolov8_perception/data/view2"  # Folder containing input frames
elif set_img == 2: # second sequence
    output_folder = "outputs2"
    data_folder_1 = "Yolov8_perception/data/view3"  # Folder containing input frames
    data_folder_2 = "Yolov8_perception/data/view4"  # Folder containing input frames
else: # third sequence
    output_folder = "outputs3"
    data_folder_1 = "Yolov8_perception/data/view5"  # Folder containing input frames
    data_folder_2 = "Yolov8_perception/data/view6"  # Folder containing input frames
os.makedirs(output_folder, exist_ok=True)  # Create the obj folder if it doesn't exist

# Initialize a list to store occluded predictions
tracked_predictions = {}
outputs = []
labels_dict = {}
lables_left = []
lables_right =[]
# Main loop for processing each frame
# Main loop for processing each frame
for left_frame_path, right_frame_path in zip (sorted(Path(data_folder_1).glob("*.png")), sorted(Path(data_folder_2).glob("*.png"))):
   
    # Process left frame: detects objects, their positions, and labels them
    dets, labels, img, ids, lables_left = process_frame(left_frame_path, model, confidence_threshold=0.5)
    
    # Process right frame: detects objects, their positions, and labels them
    dets_right, labels_right, img_right, ids_right, lables_right = process_frame(right_frame_path, model, confidence_threshold=0.5)
    
    # Match boxes between the left and right frames (e.g., using Euclidean distance and Hungarian algorithm)
    distances = calculate_distance(lables_left, lables_right)

    bearings = get_bearing_from_stereo(lables_left, lables_right)
    rotations = calculate_rotation_y(lables_left, lables_right)
    heights = get_height(lables_left, lables_right)
    widths = get_width(lables_left, lables_right)
    lengths = get_length(lables_left, lables_right, img, img_right)
    truncations = get_truncated(lables_left, lables_right, img, img_right)
    cam_positions = get_camera_position(lables_left, lables_right)
    
    # Skip to the next frame if the image failed to load
    if img is None:
        continue

    draw_detected_objects(img,dets,labels,distances)
    # Update SORT tracker
    trackers = mot_tracker.update(dets)

    # Finds Wazowski
    overlay_rect, img = find_overlay_rectangle(img, overlap_image,set_img, threshold=0.2)
    

    # Loop and process each tracked object individually
    for i, d in enumerate(trackers): 

        distance, bearing, rotation, height, width, length, truncated, x_cam, y_cam, z_cam = get_parameters(
            i, distances, bearings, rotations, heights, widths, lengths, truncations, cam_positions
        ) 

        label = labels[i]

        # Process each tracked object (Kalman filter and occlusion detection)
        outputs, tracked_predictions = process_tracked_object(
            d=d,
            img=img,
            overlay_rect=overlay_rect,
            frame_path=left_frame_path,
            label=label,
            outputs=outputs,
            tracked_predictions=tracked_predictions,
            distance = distance,
            bearing = bearing,
            rotation= rotation,
            height = height, 
            width= width, 
            length = length,
            truncated=truncated,
            x_cam=x_cam,
            y_cam=y_cam,
            z_cam=z_cam
        )
    
    # Process prediction: draws bounding box if object is occluded, updates occlusion rate, and updates Kalman filter
    for track_id, prediction in tracked_predictions.items():
        outputs, tracked_predictions = process_prediction(
            track_id=track_id,
            prediction=prediction,
            frame_name=left_frame_path,
            overlay_rect=overlay_rect,
            img=img,
            tracked_predictions=tracked_predictions,
            labels_dict=labels_dict,
            outputs=outputs
        )
    
    # Save annotated frame
    #print("Saving frame:", left_frame_path.name)
    output_path = os.path.join(output_folder, left_frame_path.name)
    cv2.imwrite(output_path, img)

# Specify the folder and output video path
image_folder = "outputs2"
output_video_path = "clean/video2.mp4"

# Create a video from images with a frame rate of 20 FPS and preview enabled
create_video_from_images(image_folder, output_video_path, frame_rate=20, preview=True)

# Print the outputs to a .txt file
output_path = "outputs_text.txt"
with open(output_path, "w") as file:
    for output in outputs:
        file.write(str(output) + "\n") 