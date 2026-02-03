import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Paths (relative to this file)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings", "face_db.pkl")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "face_svm_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# -----------------------------
# Load embeddings
# -----------------------------
if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError(
        f"Cannot find embeddings database at {EMBEDDINGS_PATH}. "
        "Run generate_embeddings.py first."
    )

db = joblib.load(EMBEDDINGS_PATH)
embeddings = np.asarray(db.get("embeddings"))
labels = np.asarray(db.get("labels"))

if embeddings.size == 0 or labels.size == 0:
    raise ValueError("Embeddings database is empty. Add images and regenerate.")

# -----------------------------
# Train SVM + Label Encoder
# -----------------------------
os.makedirs(MODELS_DIR, exist_ok=True)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

svm = SVC(kernel="linear", probability=True)
svm.fit(embeddings, y)

joblib.dump(svm, SVM_MODEL_PATH)
joblib.dump(label_encoder, ENCODER_PATH)

print("Training complete.")
print(f"Saved SVM model: {SVM_MODEL_PATH}")
print(f"Saved label encoder: {ENCODER_PATH}")
