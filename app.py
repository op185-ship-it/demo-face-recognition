from flask import Flask, jsonify
import requests
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import joblib
import os
from datetime import datetime

# -----------------------------
# ESP32 camera URL
# -----------------------------
ESP32_IP = "192.168.1.5"    # Replace with your ESP32 IP
ESP32_ENDPOINT = f"http://{ESP32_IP}/capture"

# -----------------------------
# Models path
# -----------------------------
MODELS_DIR = '../models'
SVM_MODEL_PATH = f'{MODELS_DIR}/face_svm_model.pkl'
ENCODER_PATH = f'{MODELS_DIR}/label_encoder.pkl'

# -----------------------------
# Device & Models
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

svm_model = joblib.load(SVM_MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Ensure folder to save images
# -----------------------------
SAVED_IMAGES_DIR = './captured_images'
os.makedirs(SAVED_IMAGES_DIR, exist_ok=True)

# -----------------------------
# Helper functions
# -----------------------------
def preprocess_faces(frame):
    boxes, _ = mtcnn.detect(frame)
    faces = []
    if boxes is None:
        return faces
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (160, 160))
        face = face.astype('float32') / 255.0
        faces.append(face)
    return faces

def get_embedding(face_pixels):
    face_tensor = torch.tensor(np.transpose(face_pixels, (2,0,1))).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = facenet_model(face_tensor)
    return embedding.cpu().numpy().flatten()

def recognize_faces_from_image(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = preprocess_faces(frame_rgb)

    recognized_names = set()
    for face in faces:
        emb = get_embedding(face)
        pred = svm_model.predict([emb])
        name = label_encoder.inverse_transform(pred)[0]
        recognized_names.add(name)
    return list(recognized_names)

# -----------------------------
# Flask Route
# -----------------------------
@app.route("/detect_students", methods=["GET"])
def detect_students():
    try:
        # Request image from ESP32
        response = requests.get(ESP32_ENDPOINT, timeout=5)
        if response.status_code != 200:
            return jsonify({"error": "Failed to get image from ESP32"}), 500

        # Convert image bytes to OpenCV image
        image_array = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Save image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(SAVED_IMAGES_DIR, f"capture_{timestamp}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"Image saved at {save_path}")

        # Recognize faces
        names = recognize_faces_from_image(frame)
        return jsonify({"present_students": names, "image_saved": save_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
