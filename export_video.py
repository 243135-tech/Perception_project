import cv2
import os
from natsort import natsorted

def create_video_from_images(image_folder, output_video_path, frame_rate=20, preview=False):
    """
    Creates a video from a sequence of images in a folder.
    """

    # Get all .png files from the folder and sort them by name number
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = natsorted(images)  

    if not images:
        print("No .png images found in the folder.")
        return

    # Read the first image to get video dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)

    if frame is None:
        print(f"Error reading the first image: {first_image_path}")
        return

    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Loop through images and write them to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error reading image: {image_path}, skipping.")
            continue

        video.write(frame)

        # Display the frame if preview is enabled
        if preview:
            print(f"Processing {image}")
            cv2.imshow('Video Preview', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit preview
                break

    # Release the video writer and close any open windows
    video.release()
    if preview:
        cv2.destroyAllWindows()

    print(f"Video saved as {output_video_path}")