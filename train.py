from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n-cls.pt')  # You can also use 'yolov8s-cls.pt' for a slightly larger model

# Train the model
model.train(data='dataset_split', epochs=2, imgsz=224, batch=16, name='deepfake_classification')
