import os
import cv2
from sklearn.model_selection import train_test_split

# CONFIGURATION
DATASET_PATH = "dataset"  # Raw dataset folder
OUTPUT_PATH = "processed_dataset"
IMG_SIZE = 64  # Resize images to 64x64

# Create output directories
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_PATH, split), exist_ok=True)

# Get list of all gesture classes
classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

for class_name in classes:
    class_path = os.path.join(DATASET_PATH, class_name)
    images = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith(".jpg")]

    # Split dataset into train/val/test (70% / 15% / 15%)
    train_files, temp_files = train_test_split(images, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    for split_name, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        split_class_path = os.path.join(OUTPUT_PATH, split_name, class_name)
        os.makedirs(split_class_path, exist_ok=True)

        for file_path in files:
            img = cv2.imread(file_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(os.path.join(split_class_path, os.path.basename(file_path)), img)

print("Preprocessing complete!")
