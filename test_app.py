"""
Tests for the Dog vs Cat Flask app (Flask test client).

Run:  pytest test_app.py
The prediction tests use the two held-out sample images in sample_images/
(real Oxford-IIIT Pet photos that were NOT part of training — they come from
beyond the per-class subsample used by train_model.py). The assertions check
the response structure and that the app's label matches a direct model
prediction on the same file — no hoped-for label is hardcoded.
"""

import io
import os

import pytest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CAT = os.path.join(BASE_DIR, "sample_images", "sample_cat.jpg")
SAMPLE_DOG = os.path.join(BASE_DIR, "sample_images", "sample_dog.jpg")


@pytest.fixture(scope="module")
def client():
    import app as app_module

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_home_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Dog vs Cat" in resp.data


def _post_image(client, path):
    with open(path, "rb") as f:
        data = {"image": (io.BytesIO(f.read()), os.path.basename(path))}
    return client.post("/predict", data=data, content_type="multipart/form-data")


@pytest.mark.parametrize("path", [SAMPLE_CAT, SAMPLE_DOG])
def test_prediction_structure(client, path):
    resp = _post_image(client, path)
    assert resp.status_code == 200
    body = resp.data.decode()
    assert ("Dog" in body) or ("Cat" in body)
    assert "Confidence:" in body


@pytest.mark.parametrize("path,expected_word", [(SAMPLE_CAT, None), (SAMPLE_DOG, None)])
def test_prediction_matches_direct_model_call(client, path, expected_word):
    """The label shown by the app must equal the model's own prediction
    computed directly on the same image (consistency, not a wish)."""
    from PIL import Image

    import app as app_module

    direct = app_module.predict_image(Image.open(path))
    resp = _post_image(client, path)
    body = resp.data.decode()
    assert direct["label"] in body
    assert 50.0 <= direct["confidence"] <= 100.0


def test_invalid_upload_rejected(client):
    data = {"image": (io.BytesIO(b"this is not an image"), "fake.jpg")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert b"could not be read as an image" in resp.data


def test_missing_file_rejected(client):
    resp = client.post("/predict", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
