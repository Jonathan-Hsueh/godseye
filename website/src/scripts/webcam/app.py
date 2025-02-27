from flask import Flask, request, jsonify
import base64
from ultralytics import YOLO
import numpy as np
import cv2
import os
from flask_cors import CORS, cross_origin

app = Flask("VideoBackend")

CORS(app, resources={
    r"/*": {
        "origins": "http://localhost:5173",
        "supports_credentials": True,
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

model = YOLO("../trainedmodel.pt")

def _build_cors_preflight_response():
    response = jsonify({"message": "Preflight Accepted"})
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

@app.route('/upload_video_chunk', methods=['POST'])
def upload_video_chunk():
    if 'video_chunk' not in request.files:
        return jsonify({'error': 'No video chunk'}), 400

    chunk = request.files['video_chunk']
    
    # Save chunk to a temporary file
    temp_file = "/tmp/chunk.webm"
    chunk.save(temp_file)
    
    cap = cv2.VideoCapture(temp_file)
    success, frame = cap.read()
    cap.release()
    os.remove(temp_file)  # Cleanup
    
    if not success:
        return jsonify({'error': 'Failed to read video chunk'}), 400

    results = model(frame)
    annotated_frame = results[0].plot()
    
    _, buffer = cv2.imencode('.JPEG', annotated_frame)
    response = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({'annotated_chunk': response}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)