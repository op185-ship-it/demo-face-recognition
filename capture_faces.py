# capture_faces.py
import cv2, os, time
from facenet_pytorch import MTCNN

student_name = input("Enter student name: ")
save_dir = f"dataset/{student_name}"
os.makedirs(save_dir, exist_ok=True)

cam = cv2.VideoCapture(0)
mtcnn = MTCNN(keep_all=False)

instructions = [
    "Look straight 🧍",
    "Turn slightly LEFT ↩️",
    "Turn slightly RIGHT ↪️",
    "Look UP ⬆️",
    "Look DOWN ⬇️",
    "Smile 😄"
]

count = 0
while count < 100:
    ret, frame = cam.read()
    if not ret:
        break

    # show next instruction every 20 images
    instr = instructions[(count // 20) % len(instructions)]
    cv2.putText(frame, instr, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Capture Faces", frame)

    face = mtcnn(frame)
    if face is not None:
        path = f"{save_dir}/{count}.jpg"
        cv2.imwrite(path, frame)
        count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"✅ Captured {count} images for {student_name}")
cam.release()
cv2.destroyAllWindows()
