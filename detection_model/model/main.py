import argparse
import os
from pathlib import Path
import cv2
from ultralytics import YOLO
from ..src.sort import Sort
from ..utils.occlusion import *
from ..src.kalman import *
from ..src.detection import *
from ..utils.export_video import *
from ..utils.distance import *

def main(set_img, output_video_path, frame_rate, confidence_threshold):
    # Initialize the YOLO model
    model = YOLO("yolov8l.pt") 

    # Initialize SORT tracker
    mot_tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.5)

    overlap_image = resize_image("detection_model/model/Overlap_image.png", width=217, height=225)

    # Define input and obj folders
    if set_img == 1:  # first sequence
        output_folder = "outputs/outputs1"
        data_folder_1 = "data/view1"
        data_folder_2 = "data/view2"
    elif set_img == 2:  # second sequence
        output_folder = "outputs/outputs2"
        data_folder_1 = "data/view3"
        data_folder_2 = "data/view4"
    else:  # third sequence
        output_folder = "outputs/outputs3"
        data_folder_1 = "data/view5"
        data_folder_2 = "data/view6"

    os.makedirs(output_folder, exist_ok=True)

    # Initialize a list to store occluded predictions
    tracked_predictions = {}
    outputs = []
    labels_dict = {}

    # Main loop for processing each frame
    for left_frame_path, right_frame_path in zip(
        sorted(Path(data_folder_1).glob("*.png")), 
        sorted(Path(data_folder_2).glob("*.png"))
    ):
        # Process left and right frames
        dets, labels, img, ids, lables_left = process_frame(left_frame_path, model, confidence_threshold)
        dets_right, labels_right, img_right, ids_right, lables_right = process_frame(right_frame_path, model, confidence_threshold)

        distances = calculate_distance(lables_left, lables_right)
        bearings = get_bearing_from_stereo(lables_left, lables_right)
        rotations = calculate_rotation_y(lables_left, lables_right)
        heights = get_height(lables_left, lables_right)
        widths = get_width(lables_left, lables_right)
        lengths = get_length(lables_left, lables_right, img, img_right)
        truncations = get_truncated(lables_left, lables_right, img, img_right)
        cam_positions = get_camera_position(lables_left, lables_right)

        if img is None:
            continue

        draw_detected_objects(img, dets, labels, distances)
        trackers = mot_tracker.update(dets)
        overlay_rect, img = find_overlay_rectangle(img, overlap_image, set_img, threshold=0.2)

        for i, d in enumerate(trackers):
            distance, bearing, rotation, height, width, length, truncated, x_cam, y_cam, z_cam = get_parameters(
                i, distances, bearings, rotations, heights, widths, lengths, truncations, cam_positions
            )
            label = labels[i]
            outputs, tracked_predictions = process_tracked_object(
                d=d,
                img=img,
                overlay_rect=overlay_rect,
                frame_path=left_frame_path,
                label=label,
                outputs=outputs,
                tracked_predictions=tracked_predictions,
                distance=distance,
                bearing=bearing,
                rotation=rotation,
                height=height,
                width=width,
                length=length,
                truncated=truncated,
                x_cam=x_cam,
                y_cam=y_cam,
                z_cam=z_cam
            )

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

        output_path = os.path.join(output_folder, left_frame_path.name)
        cv2.imwrite(output_path, img)

    # Create video from images
    create_video_from_images(output_folder, output_video_path, frame_rate, preview=False)

    # Save outputs to a text file
    with open("outputs_text.txt", "w") as file:
        for output in outputs:
            file.write(str(output) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video frames using YOLO and SORT tracker.")
    parser.add_argument("--set_img", type=int, default=2, help="Sequence number (1, 2, or 3)")
    parser.add_argument("--output_video_path", type=str, default="clean/video2.mp4", help="Output video path")
    parser.add_argument("--frame_rate", type=int, default=20, help="Frame rate for the output video")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold for object detection")

    args = parser.parse_args()
    main(
        set_img=args.set_img,
        output_video_path=args.output_video_path,
        frame_rate=args.frame_rate,
        confidence_threshold=args.confidence_threshold
    )
