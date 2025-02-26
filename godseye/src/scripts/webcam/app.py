from flask import Flask, request, jsonify
from ultralytics import YOLO
import io
from PIL import Image
import numpy as np
import cv2
import os
from flask_cors import CORS, cross_origin

app = Flask("VideoBackend")

CORS(app)
model = YOLO("../backenddesign/trainedmodel.pt")


@app.route('/upload_video_chunk', methods=['POST'])
@cross_origin()
def upload_video_chunk(): 
    if 'video_chunk' not in request.files: 
        return jsonify({'error': 'No video chunk provided'}), 400
    
    video_chunk_file = request.files['video_chunk']
    video_chunk_bytes = video_chunk_file.read()
    
    nparr = np.frombuffer(video_chunk_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({'error': 'Invalid chunk data'}), 400
    
    results = model(frame)
    annotated_frame = results[0].plot()
    
    _, annotated_frame_bytes = cv2.imencode('.JPEG', annotated_frame)
    annotated_frame_base64 = base64.b64encode(annotated_frame_bytes).decode('utf-8')
    
    return jsonify({
        'annotated_chunk': annotated_frame_base64,
        'shape': annotated_frame.shape,
        'dtype': str(annotated_frame.dtype)
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)