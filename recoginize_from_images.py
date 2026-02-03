import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import sys
import os

# -----------------------------
# CONFIG
# -----------------------------
EMBEDDINGS_PATH = 'embeddings/face_db.pkl'
THRESHOLD = 0.7  # similarity threshold
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# LOAD MODELS
# -----------------------------
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# -----------------------------
# LOAD DATABASE
# -----------------------------
if not os.path.exists(EMBEDDINGS_PATH):
    print("❌ face_db.pkl not found! Run generate_embeddings.py first.")
    sys.exit()

db = joblib.load(EMBEDDINGS_PATH)
known_embeddings = db['embeddings']
known_labels = db['labels']

# -----------------------------
# HELPER: Get Embedding
# -----------------------------
def get_embedding(face_img):
    # Temporarily set keep_all=False to get a single tensor
    single_mtcnn = MTCNN(keep_all=False, device=device)
    face_tensor = single_mtcnn(Image.fromarray(face_img))
    if face_tensor is None:
        return None
    # face_tensor is already [3,160,160]
    emb = resnet(face_tensor.unsqueeze(0).to(device))  # now [1,3,160,160]
    return emb.detach().cpu().numpy().flatten()

# -----------------------------
# HELPER: Recognize
# -----------------------------
def recognize_face(embedding):
    sims = cosine_similarity([embedding], known_embeddings)
    best_idx = np.argmax(sims)
    if sims[0][best_idx] > THRESHOLD:
        return known_labels[best_idx], sims[0][best_idx]
    return "Unknown", sims[0][best_idx]

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def recognize_from_image(image_path):
    if not os.path.exists(image_path):
        print("❌ Image not found:", image_path)
        return

    frame = cv2.imread(image_path)
    boxes, _ = mtcnn.detect(frame)

    recognized = []
    unknown_count = 0

    if boxes is None:
        print("⚠️ No faces detected.")
        return

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        face_img = frame[y1:y2, x1:x2]
        emb = get_embedding(face_img)

        if emb is not None:
            name, score = recognize_face(emb)
            if name == "Unknown":
                unknown_count += 1
                color = (0, 0, 255)
            else:
                recognized.append(name)
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} ({score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display result
    cv2.imshow("Recognized Faces", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print summary
    print("\n🎯 Recognition Summary:")
    print("--------------------------")
    print("✅ Recognized persons:", recognized if recognized else "None")
    print(f"❌ Unknown faces count: {unknown_count}")
    print("--------------------------")

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python recognize_from_image.py <image_path>")
        sys.exit()

    image_path = sys.argv[1]
    recognize_from_image(image_path)
