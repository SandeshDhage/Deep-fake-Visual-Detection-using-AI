import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=30):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:  # Extract frames at regular intervals (e.g., every 30th frame)
            # Construct frame filename with the video name for uniqueness
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            frame_filename = f"{video_name}_frame_{count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
        count += 1

    cap.release()

# Extract frames for all videos in the dataset
video_directory = "archive"
output_directory = "dataset"

# Update categories to match YOLOv8 structure expectations
category_map = {
    "Celeb-real": "real",
    "Celeb-synthesis": "fake"
}
for original_category, mapped_category in category_map.items():
    video_folder = os.path.join(video_directory, original_category)
    output_folder = os.path.join(output_directory, mapped_category)
    os.makedirs(output_folder, exist_ok=True)

    for video_file in os.listdir(video_folder):
        if video_file.endswith(".mp4") or video_file.endswith(".avi"):
            video_path = os.path.join(video_folder, video_file)
            extract_frames(video_path, output_folder, frame_rate=90)
