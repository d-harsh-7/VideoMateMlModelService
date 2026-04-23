import os
import logging
import warnings

# Set TensorFlow runtime flags before importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r"TensorFlow GPU support is not available on native Windows.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Skipping variable loading for optimizer 'rmsprop'.*",
)

from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import tempfile
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from tensorflow.keras.models import load_model

app = FastAPI()
model = load_model("deepfake_model.h5", compile=False)


# ---------- Feature Extraction ----------
def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
    return hist / (hist.sum() + 1e-6)


def extract_glcm(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, "contrast")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

    return np.array([contrast, correlation, energy, homogeneity])


# ---------- Core Prediction Flow ----------
def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (128, 128))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame / 255.0)

    cap.release()
    return np.array(frames)


def get_features(frames):
    lbp_features = np.array([extract_lbp((f * 255).astype("uint8")) for f in frames])
    glcm_features = np.array([extract_glcm((f * 255).astype("uint8")) for f in frames])

    glcm_features = glcm_features / (np.max(glcm_features) + 1e-6)
    features = np.concatenate([lbp_features, glcm_features], axis=1)

    return features


def predict_video(video_path):
    frames = get_frames(video_path)
    if len(frames) == 0:
        return {"error": "No frames extracted"}

    features = get_features(frames)

    preds = model.predict([frames, features])
    preds = (preds > 0.5).astype(int)

    final_pred = float(np.mean(preds))
    result = "Fake Video" if final_pred > 0.5 else "Real Video"

    return {
        "prediction": result,
        "confidence": final_pred,
        "frames_shape": tuple(frames.shape),
        "features_shape": tuple(features.shape),
    }


# ---------- API ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(await file.read())
            video_path = temp.name

        return predict_video(video_path)

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/health")
async def healthcheck():
    try:
        return {"message": "running fine"}
    except Exception as e:
        return {"error": str(e)}
