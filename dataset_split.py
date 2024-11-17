import os
import shutil
import random
from tqdm import tqdm

def train_test_val_split(dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Ensure ratios add up to 1.0
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Set up output directories
    for split in ['train', 'val', 'test']:
        for category in ['real', 'fake']:
            split_category_path = os.path.join(output_dir, split, category)
            os.makedirs(split_category_path, exist_ok=True)

    # Split data for each category
    for category in ['real', 'fake']:
        category_path = os.path.join(dataset_dir, category)
        images = os.listdir(category_path)

        # Shuffle images to ensure random distribution
        random.shuffle(images)

        # Calculate split indices
        total_images = len(images)
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)

        # Split images
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        # Copy images to respective folders
        for split, split_images in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
            print(f"\nCopying images for {split} set ({category})...")
            for image in tqdm(split_images, desc=f"{category} - {split}", unit="image"):
                src_path = os.path.join(category_path, image)
                dst_path = os.path.join(output_dir, split, category, image)
                shutil.copy2(src_path, dst_path)

# Input and output directories
dataset_directory = "dataset"  # Input folder containing 'real' and 'fake'
output_directory = "dataset_split"  # Output folder for train, val, test

# Split the dataset
train_test_val_split(dataset_directory, output_directory)
