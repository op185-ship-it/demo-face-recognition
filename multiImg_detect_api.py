import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import joblib
from flask import Flask, request, jsonify
import tempfile
import os

# -----------------------------
# Paths
# -----------------------------
MODELS_DIR = '../models'  # adjust relative path if needed
SVM_MODEL_PATH = f'{MODELS_DIR}/face_svm_model.pkl'
ENCODER_PATH = f'{MODELS_DIR}/label_encoder.pkl'

# -----------------------------
# Device & Models
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load SVM and Label Encoder
svm_model = joblib.load(SVM_MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# -----------------------------
# Helper Functions
# -----------------------------
def preprocess_faces(frame):
    """Detect faces and return cropped, normalized faces"""
    boxes, _ = mtcnn.detect(frame)
    faces, coords = [], []

    if boxes is None:
        return faces, coords

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue
        face = cv2.resize(face, (160, 160))
        face = face.astype('float32') / 255.0
        faces.append(face)
        coords.append((x1, y1, x2, y2))

    return faces, coords

def get_embedding(face_pixels):
    """Convert face image to embedding"""
    face_tensor = torch.tensor(np.transpose(face_pixels, (2,0,1))).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = facenet_model(face_tensor)
    return embedding.cpu().numpy().flatten()

def recognize_faces(image_path):
    """Process single image and return recognized names"""
    frame = cv2.imread(image_path)
    if frame is None:
        return []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces, _ = preprocess_faces(frame_rgb)

    recognized_names = []

    for face in faces:
        emb = get_embedding(face)
        pred = svm_model.predict([emb])
        name = label_encoder.inverse_transform(pred)[0]
        recognized_names.append(name)

    return recognized_names

# -----------------------------
# Flask API
# -----------------------------
app = Flask(__name__)

@app.route("/recognize", methods=["POST"])
def recognize():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    all_names = set()  # ✅ use set for unique names

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            file.save(tmp.name)
            names = recognize_faces(tmp.name)
            all_names.update(names)  # ✅ add to set (avoids duplicates)
            os.unlink(tmp.name)

    return jsonify({
        "recognized_faces": list(all_names)
    })

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
