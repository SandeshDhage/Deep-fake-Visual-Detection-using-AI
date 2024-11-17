import os
import shutil
import random

def create_vivit_dataset(video_directory, output_directory, train_ratio=0.7, val_ratio=0.2):
    # Define category mappings
    category_map = {
        "Celeb-real": "real",
        "Celeb-synthesis": "fake"
    }

    # Create train, validation, and test folders
    splits = ["train", "validation", "test"]
    for split in splits:
        for category in category_map.values():
            os.makedirs(os.path.join(output_directory, split, category), exist_ok=True)

    # Split data into train, validation, and test
    dataset = []
    for original_category, mapped_category in category_map.items():
        video_folder = os.path.join(video_directory, original_category)
        video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]
        dataset += [(video, mapped_category) for video in video_files]

    random.shuffle(dataset)
    num_train = int(train_ratio * len(dataset))
    num_val = int(val_ratio * len(dataset))

    train_set = dataset[:num_train]
    val_set = dataset[num_train:num_train + num_val]
    test_set = dataset[num_train + num_val:]

    # Copy videos to respective folders
    def copy_videos(video_set, split):
        for video_path, category in video_set:
            output_folder = os.path.join(output_directory, split, category)
            shutil.copy(video_path, output_folder)

    # Process and copy train, validation, and test sets
    copy_videos(train_set, "train")
    copy_videos(val_set, "validation")
    copy_videos(test_set, "test")

# Directories for input videos and output dataset
video_directory = "archive"
output_directory = "vivit_dataset"

# Create the dataset for training ViViT
create_vivit_dataset(video_directory, output_directory)
