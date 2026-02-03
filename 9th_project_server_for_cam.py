# server.py
from flask import Flask, request, jsonify
import os
import requests
from datetime import datetime
import uuid
import json

app = Flask(__name__)

# directory to store captures
CAPTURE_ROOT = "captures_new"
os.makedirs(CAPTURE_ROOT, exist_ok=True)

# a simple in-memory registry of running classes (for demo)
running_classes = {}

@app.route('/start-class', methods=['POST'])
def start_class():
    """
    Expected JSON:
    {
      "classname": "ClassA",
      "subject": "Math",
      "interval_seconds": 30,
      "duration_seconds": 3600,
      "esp_ip": "192.168.1.10",
      "esp_id": "esp01"    # optional ID
    }
    """
    data = request.get_json(force=True)
    required = ("classname", "subject", "interval_seconds", "esp_ip")
    for r in required:
        if r not in data:
            return jsonify({"error": f"missing {r}"}), 400

    esp_ip = data["esp_ip"]
    esp_id = data.get("esp_id", esp_ip)
    running_classes[esp_id] = {
        "classname": data["classname"],
        "subject": data["subject"],
        "interval_seconds": int(data["interval_seconds"]),
        "start_time": datetime.utcnow().isoformat(),
    }

    # POST to ESP to configure it
    try:
        url = f"http://{esp_ip}/configure"
        resp = requests.post(url, json=data, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        return jsonify({"error": "failed contacting ESP", "details": str(e)}), 500

    return jsonify({"status": "started", "esp": esp_ip})

@app.route('/captured', methods=['POST'])
def captured():
    """
    ESP POSTS raw JPEG bytes as request.data and sends metadata in headers:
    X-Classname, X-Subject, X-Image-No, X-Timestamp, X-Esp-Id
    """
    headers = request.headers
    classname = headers.get("X-Classname", "unknown")
    subject = headers.get("X-Subject", "unknown")
    esp_id = headers.get("X-Esp-Id", request.remote_addr)
    image_no = headers.get("X-Image-No", None)
    timestamp = headers.get("X-Timestamp", datetime.utcnow().isoformat())

    # make directory
    dirpath = os.path.join(CAPTURE_ROOT, classname, subject, esp_id)
    os.makedirs(dirpath, exist_ok=True)

    # generate filename
    if image_no is None:
        image_no = str(uuid.uuid4())
    filename = f"{timestamp.replace(':','-')}_img{image_no}.jpg"
    filepath = os.path.join(dirpath, filename)

    # save image bytes
    image_bytes = request.get_data()
    if not image_bytes:
        return jsonify({"error": "no image data"}), 400
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    # also save a small metadata json
    meta = {
        "classname": classname,
        "subject": subject,
        "esp_id": esp_id,
        "image_no": image_no,
        "timestamp": timestamp,
        "filename": filename,
        "saved_at": datetime.utcnow().isoformat()
    }
    with open(filepath + ".json", "w") as mf:
        json.dump(meta, mf, indent=2)

    return jsonify({"status": "saved", "path": filepath})

@app.route('/end-class', methods=['POST'])
def end_class():
    """
    JSON: {"esp_ip": "192.168.1.55", "esp_id": "esp01"}
    """
    data = request.get_json(force=True)
    esp_ip = data.get("esp_ip")
    if not esp_ip:
        return jsonify({"error": "esp_ip missing"}), 400

    esp_id = data.get("esp_id", esp_ip)
    running_classes.pop(esp_id, None)

    # tell ESP to end
    try:
        url = f"http://{esp_ip}/end"
        resp = requests.post(url, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        return jsonify({"error": "failed contacting ESP", "details": str(e)}), 500

    return jsonify({"status": "ended", "esp": esp_ip})

if __name__ == '__main__':
    # for debugging; production: use gunicorn/uwsgi
    app.run(host="0.0.0.0", port=5000, debug=True)
