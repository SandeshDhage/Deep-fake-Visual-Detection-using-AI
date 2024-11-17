# File: app.py

from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure paths
UPLOAD_FOLDER = 'static/uploads/'
MODEL_PATH = 'deepfake_yolov8_final.pt'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the YOLOv8 model
model = YOLO(MODEL_PATH)

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    filename = None
    predicted_class = None
    confidence = None

    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load the image using OpenCV
            image = cv2.imread(filepath)
            if image is None:
                return "Image not found", 404

            # Predict the class of the image using the model
            results = model.predict(image, verbose=False)

            # Get the predicted class label and confidence
            predicted_class = results[0].names[results[0].probs.top1]
            confidence = results[0].probs.top1conf * 100

    return render_template('index.html', filename=filename, predicted_class=predicted_class, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)