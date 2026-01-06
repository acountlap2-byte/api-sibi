from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import joblib
import time
from collections import deque
import os

# ===============================
# KONFIGURASI
# ===============================
MODEL_PATH = "model/model_sibi_landmark_FINAL.keras"
SCALER_PATH = "preprocessed/scaler.pkl"

BASE_CONF_THRESHOLD = 0.45
STABLE_TIME = 0.6
VOTE_SIZE = 5

# ===============================
# APP
# ===============================
app = Flask(__name__)
CORS(app)

# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
print("✅ API SIBI READY")

# ===============================
# STATE (ANTI SPAM & STABIL)
# ===============================
last_label = None
stable_start = None
vote_buffer = deque(maxlen=VOTE_SIZE)

# ===============================
# PREPROCESS
# ===============================
def preprocess_landmark(lm):
    lm = np.array(lm, dtype=np.float32)

    # Relatif terhadap wrist
    lm = lm - lm[0]

    # Mirror tangan kiri
    if lm[5][0] < lm[17][0]:
        lm[:, 0] = -lm[:, 0]

    # Normalisasi skala
    ref = np.linalg.norm(lm[9] - lm[0])
    if ref > 0:
        lm = lm / ref

    return lm.flatten()

# ===============================
# ROUTES
# ===============================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "API SIBI running",
        "stable_time": STABLE_TIME
    })

@app.route("/predict", methods=["POST"])
def predict():
    global last_label, stable_start, vote_buffer

    data = request.get_json()
    if not data or "landmark" not in data:
        return jsonify({"huruf": "-", "confidence": 0.0})

    lm = data["landmark"]
    if len(lm) != 21:
        return jsonify({"huruf": "-", "confidence": 0.0})

    # ===============================
    # PREPROCESS
    # ===============================
    feat = preprocess_landmark(lm)
    feat = scaler.transform(feat.reshape(1, -1))

    # ===============================
    # PREDIKSI
    # ===============================
    preds = model.predict(feat, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    label = LABELS[idx]

    # ===============================
    # ADAPTIVE THRESHOLD
    # ===============================
    threshold = BASE_CONF_THRESHOLD

    # Huruf U → toleran
    if label == "U":
        threshold = 0.33

    # Huruf Y → toleran
    if label == "Y":
        threshold = 0.35

    # Huruf Z → gerakan dinamis (proteksi)
    if label == "Z":
        vote_buffer.clear()
        stable_start = None
        return jsonify({
            "huruf": "-",
            "confidence": round(conf, 2),
            "status": "LOW_CONF"
        })

    if conf < threshold:
        vote_buffer.clear()
        stable_start = None
        return jsonify({
            "huruf": "-",
            "confidence": round(conf, 2),
            "status": "LOW_CONF"
        })

    # ===============================
    # VOTING (ANTI RAGU)
    # ===============================
    vote_buffer.append(label)

    if vote_buffer.count(label) < 3:
        return jsonify({
            "huruf": label,
            "confidence": round(conf, 2),
            "status": "SEARCHING"
        })

    # ===============================
    # STABILISASI WAKTU
    # ===============================
    now = time.time()

    if label == last_label:
        if stable_start is None:
            stable_start = now
        elif now - stable_start >= STABLE_TIME:
            return jsonify({
                "huruf": label,
                "confidence": round(conf, 2),
                "status": "FINAL"
            })
    else:
        stable_start = now
        last_label = label

    return jsonify({
        "huruf": label,
        "confidence": round(conf, 2),
        "status": "SEARCHING"
    })

# ===============================
# RUN (LOKAL & CLOUD AMAN)
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
