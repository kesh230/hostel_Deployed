# File: attendence.py

import os
import json
import csv
import numpy as np
import cv2

from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ----------------------------
# Constants & File Paths
# ----------------------------
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
DATASET_DIR = "dataset"             # Where face images are stored
LABELS_FILE = "labels.json"         # Maps label_id -> name
MODEL_FILE = "model_eigen.yml"      # Saved EigenFace model
ATTENDANCE_FILE = "attendance.csv"  # Logs attendance records
FIXED_SIZE = (200, 200)             # Force all faces to 200×200

# Initialize the EigenFace recognizer
recognizer = cv2.face.EigenFaceRecognizer_create()

# ----------------------------
# Helper Functions
# ----------------------------
def load_labels():
    """Load {label_id: name} from labels.json."""
    if not os.path.exists(LABELS_FILE):
        return {}
    with open(LABELS_FILE, "r") as f:
        try:
            return json.load(f)
        except:
            return {}

def save_labels(labels_dict):
    """Save {label_id: name} to labels.json."""
    with open(LABELS_FILE, "w") as f:
        json.dump(labels_dict, f)

def get_next_label_id(labels_dict):
    """Return the next numeric label ID (as string)."""
    if not labels_dict:
        return "1"
    existing_ids = list(map(int, labels_dict.keys()))
    return str(max(existing_ids) + 1)

def mark_attendance(name):
    """Append attendance entry to CSV with current timestamp."""
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

def detect_face(img):
    """
    1) Convert to grayscale
    2) Detect the *first* face using Haar cascade
    3) Crop and resize it to FIXED_SIZE
    Return the cropped grayscale face or None if no face found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    # scaleFactor & minNeighbors can be tuned
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return None

    # Use the first (or largest) face
    (x, y, w, h) = faces[0]
    cropped_face = gray[y : y + h, x : x + w]

    # **Resize** to match training size
    cropped_face = cv2.resize(cropped_face, FIXED_SIZE)
    return cropped_face

def train_model():
    """
    Scan dataset/<label_id>/ for images,
    load + resize them, then train EigenFace model.
    Save to MODEL_FILE.
    """
    labels_dict = load_labels()
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    faces = []
    labels = []

    for label_id_str, name in labels_dict.items():
        person_folder = os.path.join(DATASET_DIR, label_id_str)
        if not os.path.isdir(person_folder):
            continue

        # Read all images in that folder
        for filename in os.listdir(person_folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(person_folder, filename)
                gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if gray_img is None:
                    continue
                # Make sure it is resized
                gray_img = cv2.resize(gray_img, FIXED_SIZE)

                faces.append(gray_img)
                labels.append(int(label_id_str))

    # If we have no faces, skip
    if len(faces) == 0:
        return False

    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_FILE)
    return True

def load_model():
    """Load the trained EigenFace model if it exists."""
    if os.path.exists(MODEL_FILE):
        recognizer.read(MODEL_FILE)
        return True
    return False

# ----------------------------
# Flask Routes
# ----------------------------

@app.route("/api/register", methods=["POST"])
def register_face():
    """
    Expects:
      - form-data with "name" (str) and "image" (file)
    1) Detect + resize face
    2) Save to dataset/<label_id>/
    3) Retrain model
    """
    if "name" not in request.form or "image" not in request.files:
        return jsonify({"error": "Name and image file are required"}), 400

    name = request.form["name"]
    image_file = request.files["image"]
    image_bytes = image_file.read()

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    face_cropped = detect_face(img)  # includes resizing
    if face_cropped is None:
        return jsonify({"error": "No face detected"}), 200

    # Load or create label dict
    labels_dict = load_labels()
    # See if name already exists
    label_id_str = None
    for k, v in labels_dict.items():
        if v == name:
            label_id_str = k
            break

    if not label_id_str:
        label_id_str = get_next_label_id(labels_dict)
        labels_dict[label_id_str] = name
        save_labels(labels_dict)

    # Save the image
    person_folder = os.path.join(DATASET_DIR, label_id_str)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.jpg"
    save_path = os.path.join(person_folder, filename)
    cv2.imwrite(save_path, face_cropped)

    # Retrain
    trained = train_model()
    if not trained:
        return jsonify({"message": "Registered face, but no data to train yet."}), 200

    return jsonify({"message": f"Successfully registered {name}"}), 200

@app.route("/api/recognize", methods=["POST"])
def recognize_face():
    """
    Expects:
      - form-data with "image" (file)
    1) Detect + resize face
    2) predict with EigenFace
    3) If confidence < threshold => recognized
    """
    if "image" not in request.files:
        return jsonify({"error": "Image file is required"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    face_cropped = detect_face(img)  # resized
    if face_cropped is None:
        return jsonify({"message": "No face detected", "name": "Unknown"}), 200

    if not load_model():
        return jsonify({"message": "Model not trained yet", "name": "Unknown"}), 200

    # Predict
    label_id, confidence = recognizer.predict(face_cropped)
    # For EigenFace, lower confidence = better match
    # Typically we pick a threshold by trial, e.g. 5000 ~ 10000
    THRESHOLD = 5500.0

    if confidence < THRESHOLD:
        labels_dict = load_labels()
        name = labels_dict.get(str(label_id), "Unknown")
        # Mark attendance
        mark_attendance(name)
        return jsonify({
            "message": f"Recognized {name}",
            "name": name,
            "confidence": float(confidence)
        }), 200
    else:
        return jsonify({
            "message": "Face not recognized",
            "name": "Unknown",
            "confidence": float(confidence)
        }), 200

@app.route("/")
def index():
    return "EigenFace Attendance API is running."

# ----------------------------
# Main Entry
# ----------------------------
if __name__ == "__main__":
    # Ensure directories/files exist
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    if not os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "w") as f:
            json.dump({}, f)
    if not os.path.exists(ATTENDANCE_FILE):
        open(ATTENDANCE_FILE, "w").close()

    app.run(debug=True, port=5000)
