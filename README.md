# 🐶 Dog vs Cat Image Classification 🐱

A deep-learning web app that tells whether a photo shows a **Dog** or a
**Cat** — with the model's confidence. Built with **transfer learning**:
a MobileNetV2 network pre-trained on ImageNet, fine-tuned on pet photos,
served through a small Flask app.

**Test accuracy: 97.71%** — measured on a held-out test split of
480 images that the model never saw during training
(238/248 cats and 231/232 dogs classified correctly; full confusion
counts in `eval_report.json`).

## Dataset

- **Source:** Oxford-IIIT Pet Dataset —
  https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
  (redirects to `thor.robots.ox.ac.uk`, ~792 MB, 7,349 images of
  37 cat/dog breeds)
- **Labels:** the 12 cat breeds (Abyssinian, Bengal, Birman, Bombay,
  British Shorthair, Egyptian Mau, Maine Coon, Persian, Ragdoll,
  Russian Blue, Siamese, Sphynx) → **Cat**; the other 25 breeds → **Dog**
- **Subset used:** 1,500 images per class, sampled with a fixed seed (42).
  One dog image in the original dataset (`newfoundland_133.jpg`) is
  corrupt and was removed, leaving **2,999 images** (1,500 cat, 1,499 dog).
- **Split:** 70% train (2,101 images) / 15% validation (480 images
  evaluated) / 15% test (480 images evaluated) — the batched split
  rounds up to whole batches of 32, so the evaluated splits are
  slightly larger than the exact 15% counts. Split is deterministic
  (seed 42); full numbers are in `eval_report.json`.
- The dataset itself is **not** committed to this repo — `train_model.py`
  downloads it automatically and builds the subset. The two photos in
  `sample_images/` are held-out Oxford-IIIT Pet images (not part of the
  training subset) used by the tests and for a quick demo.

> Note: the well-known ML-Edu `cats_and_dogs_filtered.zip` mirror was
> tried first but its bucket now returns 403 Access Denied, so the
> official Oxford-IIIT Pet source above was used instead.

## Approach

- **Base model:** MobileNetV2 (ImageNet weights), input 128×128
- **Phase 1 — head training (6 epochs):** base frozen; augmentation
  (random flip / rotation / zoom) → GlobalAveragePooling → Dropout 0.3 →
  single sigmoid output; Adam, lr 1e-3
- **Phase 2 — fine-tuning (3 epochs):** top 30 layers of the base
  unfrozen; Adam, lr 1e-5
- **Preprocessing:** MobileNetV2 scaling (x / 127.5 − 1.0) is a Rescaling
  layer *inside* the saved model, so the app and training always apply
  exactly the same preprocessing — the app only resizes to 128×128
- **Training curves** (train / validation accuracy per epoch) are saved
  in `training_history.json`; the final validation accuracy was 99.17%
- Trained on CPU (TensorFlow-CPU 2.21) in roughly 8 minutes for all
  9 epochs

## Features of the web app

- Upload any photo → **Dog / Cat prediction with confidence %**
- The uploaded photo is shown on the result page
- Friendly error handling for non-image / missing uploads (HTTP 400,
  no crashes); uploads capped at 8 MB
- `/health` endpoint returning JSON `{"status": "ok"}`
- Model loads lazily once and is reused across requests

## Tech stack

- Python, TensorFlow / Keras (tensorflow-cpu), MobileNetV2
- Flask, Pillow, NumPy
- pytest (Flask test client) — 8 tests, all passing

## How to run

### Train the model (optional — `model.keras` is already included)

```bash
pip install -r requirements.txt
python train_model.py     # downloads the dataset, trains, evaluates
```

### Run the web app

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 and upload a photo.

### Run the tests

```bash
pip install pytest
pytest test_app.py
```

## Files

- `app.py` — Flask app (routes `/`, `/predict`, `/health`)
- `train_model.py` — full reproducible pipeline: dataset download,
  subset build, training (2 phases), test evaluation
- `dog_cat_classification.ipynb` — notebook version of the same pipeline
- `model.keras` — the trained model (20.9 MB)
- `eval_report.json` — real test-set metrics (accuracy + confusion counts)
- `training_history.json` — per-epoch accuracy/loss
- `test_app.py` — pytest suite (health, home, prediction structure and
  app-vs-model consistency, invalid-upload handling)
- `templates/index.html`, `templates/result.html` — app pages
- `sample_images/` — 2 held-out demo/test photos (1 cat, 1 dog)
- `requirements.txt`, `.gitignore` (dataset folder is git-ignored)

## Honest notes

- This project was rebuilt from scratch in Nov 2026 to replace an
  earlier college project whose files were lost.
- 97.71% is the accuracy on this project's own held-out Oxford-IIIT Pet
  test split — not a claim about arbitrary photos from the internet,
  though the model generalises well to typical pet photos.
- Cats were slightly harder than dogs here: 10 of 248 test cats were
  predicted as dogs, while only 1 of 232 test dogs was predicted as a cat.

## Author

**Suraj Dhakad** — Data Science Intern @ GRRAS Solutions, Jaipur ·
B.Tech CSE (AI), Chartered Institute of Technology ·
GitHub: [Suraj881711](https://github.com/Suraj881711)
