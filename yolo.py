from flask import Flask, request, jsonify
import cv2
from ultralytics import YOLO
import numpy as np
import uuid
import os
import tempfile
import time

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained YOLO model
model = YOLO('icon-det1.pt')

def process_image_with_yolo(image_path: str, output_image_path: str) -> list:
    """
    Process an image with YOLO model and save the output image with bounding boxes drawn.
    
    :param image_path: Path to the input image file.
    :param output_image_path: Path where the output image with bounding boxes will be saved.
    :return: List of detection results.
    """
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError('Failed to read the image')

    # Run YOLO model on the image
    results = model(image_path, conf=0.01, iou=0.45)

    # Extract and process detection results
    predictions = []
    for result in results[0].boxes:
        boxes = result.xyxy.cpu().numpy()
        confs = result.conf.cpu().numpy()
        cls_ids = result.cls.cpu().numpy()

        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = map(int, box)
            width_bbox = x2 - x1
            height_bbox = y2 - y1
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # Skip the bounding boxes with area greater than 30000 square pixels
            if width_bbox * height_bbox > 50000:
                continue

            detection_id = str(uuid.uuid4())
            class_name = model.names[int(cls_id)]

            predictions.append({
                "x": float(x_center),
                "y": float(y_center),
                "width": float(width_bbox),
                "height": float(height_bbox),
                "confidence": float(conf),
                "class": class_name,
                "class_id": int(cls_id),
                "detection_id": detection_id
            })

            # Draw bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the output image with bounding boxes
    cv2.imwrite(output_image_path, image)

    return predictions

@app.route('/get_boxes', methods=['POST'])
def get_boxes():
    start_time = time.time()  # Start time for calculating inference time

    # Check if an image file was provided in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    try:
        # Get the image file from the request
        image_file = request.files['image']
        
        # Create a temporary file for the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            image_path = temp_file.name
            image_file.save(image_path)

        # Create a temporary file for the output image with bounding boxes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file_out:
            output_image_path = temp_file_out.name
        
        # Process image with YOLO and save the output image
        predictions = process_image_with_yolo(image_path, output_image_path)

        # Get image dimensions
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        inference_time = time.time() - start_time  # Calculate inference time

        return jsonify({
            'inference_id': str(uuid.uuid4()),
            'time': inference_time,
            'image': {
                'width': width,
                'height': height
            },
            'predictions': predictions,
            'output_image_url': output_image_path
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up temporary files
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(output_image_path):
            os.remove(output_image_path)

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=8129)
