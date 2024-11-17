import os
from ultralytics import YOLO
from tqdm import tqdm
import cv2


def evaluate_model(model_path, input_folder):
    # Load the trained YOLOv8 model
    model = YOLO(model_path)

    # Initialize counters for correct and incorrect predictions
    correct_predictions = 0
    total_predictions = 0

    # Iterate through each class (e.g., real, fake)
    for class_name in os.listdir(input_folder):
        class_folder = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_folder):
            continue

        # Iterate through each image in the class folder
        for image_name in tqdm(os.listdir(class_folder), desc=f"Evaluating {class_name}", unit="image"):
            image_path = os.path.join(class_folder, image_name)
            if not (image_name.endswith('.jpg') or image_name.endswith('.png')):
                continue

            # Load the image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Predict the class of the image using the model
            results = model.predict(image, verbose=False)

            # Get the predicted class label
            predicted_class = results[0].names[results[0].probs.argmax()]

            # Compare the predicted class with the actual class
            if predicted_class.lower() == class_name.lower():
                correct_predictions += 1
            total_predictions += 1

    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"\nEvaluation completed.")
    print(f"Total Images Evaluated: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")


# Example usage
if __name__ == "__main__":
    model_path = "deepfake_yolov8_final.pt"  # Path to the trained model file
    input_folder = "dataset_split/test"  # Path to the folder containing images (e.g., test set)

    evaluate_model(model_path, input_folder)