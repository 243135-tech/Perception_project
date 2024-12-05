# Perception for Autonomous Systems Project

## Overview
The rapid advancements in artificial intelligence, particularly in autonomous systems, have ushered in a new era of transportation technologies. These systems aim to ensure efficiency and safety in complex urban environments.

This project focuses on the **detection, classification, and tracking** of entities in dynamic and unstructured environments—a critical challenge for autonomous vehicle perception systems. The primary goals include:
- **Real-time 3D tracking** using stereo vision for pedestrians, cyclists, and vehicles.
- **Handling occlusion** to ensure reliable perception.
- **Stereo camera calibration** for accurate depth estimation.

By integrating methodologies introduced during the course with state-of-the-art techniques, this project addresses key challenges and offers insights into advancing perception systems for autonomous vehicles.

---

## Features
- Real-time 3D object detection, classification, and tracking.
- Stereo vision integration for depth estimation.
- Robust handling of occlusions.
- Modular pipeline with configurable settings.
- Outputs compatible with downstream tasks like trajectory prediction.

---

## Prerequisites
To run this project, ensure you have the following:
- Python 3.8 or later

### Clone Repository
```bash
git clone https://github.com/243135-tech/Perception_project
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Download Dataset
Download the data provided by the course and save them in the following folders:
```bash
data_folder_1
data_folder_2
```

---

## Run
To execute the pipeline, run:

bash
Copiar código
python pipeline.py

css
Copiar código
(frame_path.name, track_id, label, truncated, occlusion, alpha, bbox, dimensions, location, rotation_y, x_center, y_center, distance)
Output Description
Values	Name	Description
1	frame_path.name	Frame within the sequence where the object appears.
1	track_id	Unique tracking ID of this object within this sequence.
1	label	Describes the type of object: Car, Pedestrian, Cyclist.
1	truncated	Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries.
1	occlusion	Integer (0, 1, 2, 3) indicating occlusion state: 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown.
1	alpha	Observation angle of object, ranging $[-\pi, \pi]$.
4	bbox	2D bounding box of object in the Rectified image (0-based index): contains left top (x1, y1) and right bottom (x2, y2) pixel coordinates.
3	dimensions	3D object dimensions: height, width, length (in meters).
3	location	3D object location $x, y, z$ in camera coordinates (in meters).
1	rotation_y	Rotation $r_y$ around the Y-axis in camera coordinates $[-\pi, \pi]$.
1	x_center	1D coordinate in pixel coordinates indicating the center of the bounding box along the x-axis.
1	y_center	1D coordinate in pixel coordinates indicating the center of the bounding box along the y-axis.
1	distance	Distance from the object to the camera.
Contributions
Contributions are welcome! Feel free to submit a pull request or raise issues

