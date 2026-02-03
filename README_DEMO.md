# Face Recognition Demo (Linux Instructions)

This folder lets you:
1) capture face images,
2) generate face embeddings,
3) train a classifier,
4) recognize people in group images.

Everything below is written so someone new to Python can follow it step by step on Linux.

## 1) Install Python

Install Python 3.10 or newer on Linux (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

## 2) Open a terminal in this project

Open a terminal and go to the project folder (adjust the path if yours is different):

```bash
cd /path/to/me_testing
```

## 3) Create and activate a virtual environment

```bash
python -m venv .venv
```

Activate it:
```bash
source .venv/bin/activate
```

## 4) Install dependencies

Install requirements from the project root (`me_testing`):

```bash
pip install -r requirements.txt
```

### Optional: CPU-only PyTorch
If the install fails due to GPU/CUDA issues, install CPU-only PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Then return to `me_testing` if you moved elsewhere.

## 5) Capture training images (per person)

Run:

```bash
python capture_faces.py
```

- Enter the person’s name when prompted.
- The webcam will capture ~100 images into `dataset/<person_name>/`.
- Repeat for each person you want the system to recognize.

## 6) Generate embeddings

```bash
python generate_embeddings.py
```

This creates `embeddings/face_db.pkl` containing one averaged embedding per person.

## 7) Train the classifier

```bash
python train_svm.py
```

This creates two files in `../models/`:
- `face_svm_model.pkl`
- `label_encoder.pkl`

## 8) Test recognition on a group image

Put a group photo in `me_testing/` (example: `group.jpg`).

Edit `detect_faces.py` to point to the image:

```python
IMAGE_PATH = 'group.jpg'
```

Run:

```bash
python detect_faces.py
```

It will print the names of recognized people.

## 9) Optional: Use the API for multiple images

Start the API:

```bash
python multiImg_detect_api.py
```

Send images (example using curl):

```bash
curl -X POST -F "files=@group.jpg" -F "files=@another.jpg" http://127.0.0.1:5000/recognize
```

You will get JSON with the recognized names.

## Common issues

- Webcam not opening: check permissions or try a different camera index.
- No faces detected: use clear, well-lit images with visible faces.
- Wrong names: add more images per person and rerun steps 6-7.

## Files in this demo

- `capture_faces.py`: collects images per person using the webcam
- `generate_embeddings.py`: creates embeddings from the dataset
- `train_svm.py`: trains the SVM classifier
- `detect_faces.py`: recognizes people in one image
- `multiImg_detect_api.py`: API to recognize faces in multiple images
