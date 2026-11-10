"""
Dog vs Cat Image Classification — Flask web app.

Upload a photo and the trained MobileNetV2 transfer-learning model
(model.keras, produced by train_model.py) predicts Dog or Cat with a
confidence percentage.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""

import io
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")
IMG_SIZE = 128  # must match training

app = Flask(__name__)
app.secret_key = "dog-cat-image-classification"
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload cap

_model = None


def get_model():
    """Load the trained model once, lazily (keeps import cheap for tests
    of the non-model routes, and /health usable without the model)."""
    global _model
    if _model is None:
        import tensorflow as tf

        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def preprocess(image: Image.Image) -> np.ndarray:
    """Same preprocessing as training: training fed the model plain
    [0, 255] float pixels (image_dataset_from_directory default) and the
    MobileNetV2 preprocessing step (x / 127.5 - 1.0) lives INSIDE the saved
    model graph as a Rescaling layer, so here we only resize and convert
    to float32. Augmentation layers in the saved model are inactive at
    inference."""
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict_image(image: Image.Image) -> dict:
    """Return {'label': 'Dog'|'Cat', 'confidence': float 0-100}.
    Class order in training was alphabetical: cat = 0, dog = 1, and the
    model outputs P(dog) via a single sigmoid unit."""
    model = get_model()
    prob_dog = float(model.predict(preprocess(image), verbose=0).flatten()[0])
    if prob_dog >= 0.5:
        return {"label": "Dog", "confidence": round(prob_dog * 100, 2)}
    return {"label": "Cat", "confidence": round((1 - prob_dog) * 100, 2)}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template(
            "index.html", error="Please choose an image file to upload."
        ), 400
    try:
        image = Image.open(io.BytesIO(file.read()))
        image.load()
    except Exception:
        return render_template(
            "index.html",
            error="That file could not be read as an image. "
                  "Please upload a JPG or PNG photo.",
        ), 400

    # keep an in-memory copy of the upload to show on the result page
    import base64

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    result = predict_image(image)
    return render_template(
        "result.html",
        label=result["label"],
        confidence=result["confidence"],
        img_b64=img_b64,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
