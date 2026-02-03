import os
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import joblib

# -----------------------------
# PATHS
# -----------------------------
DATASET_DIR = 'dataset'
EMBEDDINGS_DIR = 'embeddings'
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# -----------------------------
# MODELS
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# -----------------------------
# HELPER FUNCTION
# -----------------------------
def get_face_embedding(image_path):
    try:
        img = Image.open(image_path)
        face = mtcnn(img)
        if face is None:
            return None
        face_embedding = resnet(face.unsqueeze(0).to(device))
        return face_embedding.detach().cpu().numpy().flatten()
    except Exception as e:
        print(f"Error in {image_path}: {e}")
        return None

# -----------------------------
# MAIN LOOP
# -----------------------------
all_embeddings = []
all_labels = []

for student_name in os.listdir(DATASET_DIR):
    student_dir = os.path.join(DATASET_DIR, student_name)
    if not os.path.isdir(student_dir):
        continue

    print(f"Processing {student_name} ...")

    student_embeddings = []
    for img_file in os.listdir(student_dir):
        img_path = os.path.join(student_dir, img_file)
        emb = get_face_embedding(img_path)
        if emb is not None:
            student_embeddings.append(emb)

    if student_embeddings:
        avg_embedding = np.mean(student_embeddings, axis=0)
        all_embeddings.append(avg_embedding)
        all_labels.append(student_name)
        print(f"✅ Saved embedding for {student_name}")

# -----------------------------
# SAVE ALL EMBEDDINGS
# -----------------------------
DB = {
    'embeddings': np.array(all_embeddings),
    'labels': np.array(all_labels)
}
joblib.dump(DB, os.path.join(EMBEDDINGS_DIR, 'face_db.pkl'))

print("\n✅ All student embeddings generated and saved!")
