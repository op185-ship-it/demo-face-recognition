from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import os
import torch
import joblib
import requests
import time

from facenet_pytorch import InceptionResnetV1, MTCNN
# your recognition module (must exist in the same project)
from recognize_and_mark import (
    recognize_from_image,
    recognize_present_students,
    mark_attendance,
    recognize_and_mark_attendance
)

app = Flask(__name__)

# ----------------------------
# CONFIG - change these
# ----------------------------
ESP32_IP = "10.11.64.126"               # <<--- CHANGE to your ESP32 IP (no http://)
ESP32_CAPTURE_ENDPOINT = f"http://{ESP32_IP}/capture"
ESP32_TIMEOUT_SECONDS = 10              # how long Flask waits for the esp32 image to arrive
SAVED_ESP32_IMAGE = "esp32_latest.jpg"  # where we save the image sent by ESP32
UPLOADS_DIR = "uploads"
DATABASES_DIR = "databases"
MODELS_DIR = "models"

# ----------------------------
# SUBJECT MAPPING
# ----------------------------
subject_mapping = {
    "ece": {
        "1": ["ECE_101", "ECE_102", "ECE_103", "ECE_104"],
        "2": ["ECE_201", "ECE_202", "ECE_203", "ECE_204"],
        "3": ["ECE_301", "ECE_302", "ECE_303", "ECE_304"],
        "4": ["ECE_401", "ECE_402", "ECE_403", "ECE_404"]
    },
    "cse": {
        "1": ["CSE_101", "CSE_102", "CSE_103", "CSE_104"],
        "2": ["CSE_201", "CSE_202", "CSE_203", "CSE_204"],
        "3": ["CSE_301", "CSE_302", "CSE_303", "CSE_304"],
        "4": ["CSE_401", "CSE_402", "CSE_403", "CSE_404"]
    },
    "me": {
        "1": ["ME_101", "ME_102", "ME_103", "ME_104"],
        "2": ["ME_201", "ME_202", "ME_203", "ME_204"],
        "3": ["ME_301", "ME_302", "ME_303", "ME_304"],
        "4": ["ME_401", "ME_402", "ME_403", "ME_404"]
    }
}

# ----------------------------
# Load face recognition models (if needed in this file)
# ----------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# The next two lines are optional if your recognize_and_mark module loads models itself.
try:
    mtcnn = MTCNN(keep_all=True, device=DEVICE, post_process=True)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
except Exception as e:
    print("Warning: facenet/mtcnn init failed in this module (maybe handled elsewhere):", e)

# If you have SVM and encoder models in models/ directory, they will be loaded by your recognize module.
# If not needed here, these lines can remain or be removed based on your project.
try:
    svm_model = joblib.load(os.path.join(MODELS_DIR, "face_svm_model.pkl"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
except Exception as e:
    # not fatal here; recognition module might handle its own loading
    print("Info: could not load svm/labels here:", e)

# Ensure folders exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DATABASES_DIR, exist_ok=True)

# ----------------------------
# Routes - UI pages (unchanged)
# ----------------------------
@app.route("/")
def home():
    return render_template('Teacher_login.html')

@app.route("/Teacher_login", methods=['POST'])
def login():
    return redirect(url_for('teacher_dashboard'))

@app.route("/teacher_dashboard")
def teacher_dashboard():
    return render_template('Teacher_Dashboard.html')

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('teachers.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teachers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                email TEXT,
                password TEXT
            )
        ''')
        cursor.execute("INSERT INTO teachers (username, email, password) VALUES (?, ?, ?)",
                       (username, email, password))
        conn.commit()
        conn.close()

        return "<h3>Registered successfully! <a href='/'>Login here</a></h3>"

    return render_template("Teacher_register.html")

@app.route("/view_attandance", methods=['POST'])
def view_attandance():
    return render_template('Teacher_ViewStudents.html')

@app.route("/take_class", methods=['POST'])
def take_class():
    return render_template('Teacher_set_class.html')

# ----------------------------
# Endpoint to receive image from ESP32
# ESP32 may either:
#  - POST multipart/form-data with field 'image' (request.files['image'])
#  - or POST raw jpeg bytes (request.data) with content-type image/jpeg
# This endpoint handles both.
# ----------------------------
@app.route('/receive_image', methods=['POST'])
def receive_image():
    try:
        # multipart/form-data upload (preferred)
        if 'image' in request.files:
            image = request.files['image']
            if image.filename == '':
                return "No file uploaded", 400
            save_path = SAVED_ESP32_IMAGE
            image.save(save_path)
            print(f"📸 Received multipart image and saved to {save_path}")
            return "OK", 200

        # otherwise try raw body bytes (image/jpeg)
        data = request.get_data()
        if data and len(data) > 0:
            save_path = SAVED_ESP32_IMAGE
            with open(save_path, 'wb') as f:
                f.write(data)
            print(f"📸 Received raw bytes and saved to {save_path}")
            return "OK", 200

        return "No image data found", 400
    except Exception as e:
        print("❌ receive_image error:", e)
        return f"ERROR: {e}", 500

# ----------------------------
# set_class endpoint: triggers ESP32 to capture and POST image,
# waits for saved file, then runs recognition.
# ----------------------------
@app.route('/set_class', methods=['POST'])
def set_class():
    department = request.form.get("department")
    semester = request.form.get("semester")
    subject = request.form.get("subject")

    if not (department and semester and subject):
        return "❌ Invalid input: Please fill all fields", 400

    # Save current class info
    with open("current_class.txt", "w") as f:
        f.write(f"{department},{semester},{subject}")

    class_code = f"{department}-{semester}-{subject}"
    print(f"✅ Class set: {class_code}")

    # Remove old file if present
    if os.path.exists(SAVED_ESP32_IMAGE):
        try:
            os.remove(SAVED_ESP32_IMAGE)
        except Exception:
            pass

    # 1) Ask ESP32 to capture and send image
    try:
        esp_url = "http://10.11.64.126/capture"
        # print(f"📡 Requesting ESP32 to capture:" {esp_url})
        resp = requests.get(esp_url, timeout=8)   # ESP32 must have /capture route
        print("📥 ESP32 responded:", resp.status_code, resp.text if resp is not None else "no text")
    except Exception as e:
        print("❌ Error contacting ESP32:", e)
        return f"❌ ESP32 communication error: {e}", 500

    # 2) Wait for the ESP32 image to be saved by /receive_image
    waited = 0.0
    interval = 0.5
    max_wait = ESP32_TIMEOUT_SECONDS
    while waited < max_wait:
        if os.path.exists(SAVED_ESP32_IMAGE) and os.path.getsize(SAVED_ESP32_IMAGE) > 0:
            print("✅ ESP32 image found:", SAVED_ESP32_IMAGE)
            break
        time.sleep(interval)
        waited += interval
    else:
        print("❌ Timeout waiting for ESP32 image.")
        return "❌ Timeout waiting for ESP32 image", 504

    # 3) Run recognition on the saved image
    try:
        recognized_names = recognize_and_mark_attendance(SAVED_ESP32_IMAGE)
        print("✅ Face recognition done:", recognized_names)
    except Exception as e:
        print("❌ Face recognition error:", e)
        return f"❌ Face recognition error: {e}", 500

    return render_template("attendance_success.html", class_code=class_code)

# ----------------------------
# mark_attendance (manual upload) — unchanged
# ----------------------------
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance_route():
    dept = request.form['department'].lower()
    sem = request.form['semester']
    image = request.files['image']

    if image.filename == '':
        return "No file selected", 400

    os.makedirs(UPLOADS_DIR, exist_ok=True)
    image_path = os.path.join(UPLOADS_DIR, image.filename)
    image.save(image_path)

    # Save current class info to current_class.txt for consistency
    with open("current_class.txt", "w") as f:
        f.write(f"{dept},{sem}")

    present_students = recognize_present_students(image_path)
    mark_attendance(dept, sem, present_students)

    return render_template("attendance_success.html", students=present_students)

# ----------------------------
# show_all_students, show_one_student — unchanged (kept as you provided)
# ----------------------------
@app.route("/select_all")
def select_all():
    return render_template("Teacher_check_attendance.html")

@app.route("/show_all_students", methods=['POST'])
def show_all_students():
    department = request.form['department']
    semester = request.form['semester']
    db_name = f"{DATABASES_DIR}/{department.lower()}_sem{semester}.db"

    if not os.path.exists(db_name):
        return f"Database not found: {db_name}"

    subjects = subject_mapping[department.lower()][semester]

    # Build dynamic column list like: "ECE_301_total", "ECE_301_attended", ...
    column_parts = []
    for subject in subjects:
        column_parts.append(f'"{subject}_total"')
        column_parts.append(f'"{subject}_attended"')
    columns = ", ".join(["name", "roll", "department", "semester"] + column_parts)

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f'''
        SELECT {columns}
        FROM students
    ''')
    rows = cursor.fetchall()
    conn.close()

    students = []
    for row in rows:
        student = {
            "name": row[0],
            "roll": row[1],
            "department": row[2],
            "semester": row[3],
            "subjects": []
        }

        total_attended = 0
        total_classes = 0

        for i, subject in enumerate(subjects):
            total = row[4 + i * 2]
            attended = row[5 + i * 2]
            percentage = round((attended / total) * 100, 2) if total else 0
            student["subjects"].append({
                "name": subject,
                "total": total,
                "attended": attended,
                "percentage": percentage
            })
            total_attended += attended
            total_classes += total

        student["overall"] = round((total_attended / total_classes) * 100, 2) if total_classes else 0
        students.append(student)

    return render_template("students_list.html", students=students, dept=department, sem=semester)

@app.route("/select_one")
def select_one():
    return render_template("Teacher_ViewOne.html")

@app.route('/show_one_student', methods=['POST'])
def show_one_student():
    roll = request.form['roll']
    department = request.form['department']
    semester = request.form['semester']
    db_path = f"{DATABASES_DIR}/{department}_sem{semester}.db"

    if not os.path.exists(db_path):
        return "Database does not exist.", 404

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM students WHERE roll=?", (roll,))
    row = c.fetchone()
    conn.close()

    if not row:
        return "Student not found.", 404

    # Get subject names using subject_mapping
    subjects = subject_mapping[department.lower()][semester]

    subject_data = []
    total_classes = 0
    total_attended = 0

    # Skip index 0 (ID), so actual fields start from index 1
    for i in range(4):
        total = int(row[5 + i * 2])       # starts from 5 because: id=0, name=1, roll=2, dept=3, sem=4
        attended = int(row[6 + i * 2])
        percentage = round((attended / total) * 100, 2) if total else 0
        subject_data.append({
            "name": subjects[i],
            "total": total,
            "attended": attended,
            "percentage": percentage
        })
        total_classes += total
        total_attended += attended

    overall_percentage = round((total_attended / total_classes) * 100, 2) if total_classes else 0

    student = {
        "name": row[1],
        "roll": row[2],
        "department": row[3],
        "semester": row[4]
    }

    return render_template("OneStudent_list.html", student=student, subjects=subject_data, overall=overall_percentage)

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    # host=0.0.0.0 makes Flask reachable from other devices on LAN (ESP32).
    app.run(host='0.0.0.0', port=5000, debug=True)