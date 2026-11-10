"""
Dog vs Cat Image Classification — training script.

Approach: transfer learning with MobileNetV2 (ImageNet weights, frozen base)
plus a small dense classification head, trained on a subset of the
Oxford-IIIT Pet Dataset mapped to binary labels (Cat / Dog) via breed names.

Dataset source (downloaded automatically on first run):
    https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    (redirects to https://thor.robots.ox.ac.uk/pets/images.tar.gz, ~792 MB)

The 12 cat breeds in the dataset are:
    Abyssinian, Bengal, Birman, Bombay, British_Shorthair, Egyptian_Mau,
    Maine_Coon, Persian, Ragdoll, Russian_Blue, Siamese, Sphynx
All other breeds (25) are dogs.

Usage:
    python train_model.py
Outputs:
    model.keras         — trained model (also used by the Flask app)
    eval_report.json    — real test-set metrics (accuracy + confusion counts)
    training_history.json — per-epoch train/val accuracy and loss

Reproducibility: fixed random seed, deterministic subset sampling, fixed
train/val/test split (stratified, seed 42).
"""

import io
import json
import os
import random
import tarfile
import urllib.request

import numpy as np

SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 32
PER_CLASS_LIMIT = 1500          # cap images per class (subsampled, seed-fixed)
HEAD_EPOCHS = 6                 # train dense head with frozen base
FINETUNE_EPOCHS = 3             # then unfreeze top layers briefly
DATA_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "pets")
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")

CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx",
}


def breed_of(filename: str) -> str:
    """'British_Shorthair_12.jpg' -> 'British_Shorthair' (strip trailing _<id>)."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.rsplit("_", 1)[0]


def species_of(filename: str) -> str:
    return "cat" if breed_of(filename) in CAT_BREEDS else "dog"


def download_and_prepare() -> str:
    """Download the Oxford-IIIT Pet images (if needed) and build a
    class-folder layout: data/pets_subset/{cat,dog}/*.jpg (subsampled)."""
    subset_dir = os.path.join(BASE_DIR, "data", "pets_subset")
    if os.path.isdir(os.path.join(subset_dir, "cat")) and os.listdir(
        os.path.join(subset_dir, "cat")
    ):
        print(f"[data] subset already prepared at {subset_dir}")
        return subset_dir

    os.makedirs(DATA_DIR, exist_ok=True)
    archive = os.path.join(BASE_DIR, "data", "images.tar.gz")
    if not os.path.exists(archive):
        print(f"[data] downloading {DATA_URL} (~792 MB) ...")
        urllib.request.urlretrieve(DATA_URL, archive)

    print("[data] reading archive and subsampling subset ...")
    rng = random.Random(SEED)
    by_class = {"cat": [], "dog": []}
    members = {}
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar.getmembers():
            if m.name.startswith("images/") and m.name.lower().endswith(".jpg"):
                by_class[species_of(m.name)].append(m.name)
                members[m.name] = m
        for cls in ("cat", "dog"):
            rng.shuffle(by_class[cls])
            keep = by_class[cls][:PER_CLASS_LIMIT]
            out_dir = os.path.join(subset_dir, cls)
            os.makedirs(out_dir, exist_ok=True)
            for name in keep:
                f = tar.extractfile(members[name])
                if f is None:
                    continue
                data = f.read()
                with open(os.path.join(out_dir, os.path.basename(name)), "wb") as out:
                    out.write(data)
            print(f"[data] {cls}: kept {len(keep)} of {len(by_class[cls])} images")
    return subset_dir


def main():
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    print(f"[tf] TensorFlow {tf.__version__}")

    subset_dir = download_and_prepare()

    # --- datasets: split the subset 70/15/15 (train/val/test), stratified ---
    full = tf.keras.utils.image_dataset_from_directory(
        subset_dir, labels="inferred", label_mode="binary",
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        shuffle=True, seed=SEED,
    )
    class_names = [n.capitalize() for n in full.class_names]  # ['Cat', 'Dog']
    n = sum(1 for _ in full.unbatch())
    n_test = int(0.15 * n)
    n_val = int(0.15 * n)
    n_train = n - n_val - n_test
    print(f"[data] total={n} train={n_train} val={n_val} test={n_test}")

    raw = tf.keras.utils.image_dataset_from_directory(
        subset_dir, labels="inferred", label_mode="binary",
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        shuffle=True, seed=SEED,
    )
    test_ds = raw.take(n_test // BATCH_SIZE + 1)
    rest = raw.skip(n_test // BATCH_SIZE + 1)
    val_ds = rest.take(n_val // BATCH_SIZE + 1)
    train_ds = rest.skip(n_val // BATCH_SIZE + 1)

    # Keep exact test batches fixed for evaluation & reporting
    test_ds_eval = test_ds.cache()
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # --- model: MobileNetV2 transfer learning ---
    # NOTE: the input fed to the base model is kept in [0, 255] float pixels
    # and MobileNetV2's preprocess_input runs INSIDE the model graph as a
    # Lambda layer, so the saved model takes plain [0, 255] inputs and the
    # exact same preprocessing applies at inference time in the Flask app.
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmentation")

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augmentation(inputs)
    # MobileNetV2 preprocess_input is exactly x / 127.5 - 1.0; a Rescaling
    # layer (127.5 scale + -1.0 offset) does the same thing and serializes
    # cleanly (a Lambda wrapping preprocess_input does not reload reliably
    # in Keras 3). Verified: Rescaling output == preprocess_input output.
    x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0,
                                  name="mobilenet_preprocess")(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=lambda s: print("[model]", s))

    print(f"[train] phase 1: head only, {HEAD_EPOCHS} epochs")
    hist1 = model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS)

    print(f"[train] phase 2: fine-tune top 30 layers, {FINETUNE_EPOCHS} epochs")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    hist2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=HEAD_EPOCHS + FINETUNE_EPOCHS, initial_epoch=HEAD_EPOCHS,
    )

    # --- evaluation on the held-out test split ---
    print("[eval] evaluating on held-out test split ...")
    y_true, y_prob = [], []
    for images, labels in test_ds_eval:
        probs = model.predict(images, verbose=0).flatten()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().flatten().tolist())
    y_true = np.array(y_true, dtype=int)
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    # class order from image_dataset_from_directory is alphabetical: cat=0, dog=1
    tn = int(((y_true == 0) & (y_pred == 0)).sum())   # cat -> cat
    fp = int(((y_true == 0) & (y_pred == 1)).sum())   # cat -> dog
    fn = int(((y_true == 1) & (y_pred == 0)).sum())   # dog -> cat
    tp = int(((y_true == 1) & (y_pred == 1)).sum())   # dog -> dog
    accuracy = float((y_true == y_pred).mean())
    report = {
        "dataset": "Oxford-IIIT Pet Dataset (subset, cat/dog via breed)",
        "test_images": int(len(y_true)),
        "test_accuracy": round(accuracy, 4),
        "confusion": {
            "cat_correct": tn, "cat_as_dog": fp,
            "dog_correct": tp, "dog_as_cat": fn,
        },
        "class_names_alphabetical": class_names,
    }
    print("[eval]", json.dumps(report, indent=2))
    with open(os.path.join(BASE_DIR, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    history = {
        "accuracy": [float(v) for v in hist1.history["accuracy"] + hist2.history["accuracy"]],
        "val_accuracy": [float(v) for v in hist1.history["val_accuracy"] + hist2.history["val_accuracy"]],
        "loss": [float(v) for v in hist1.history["loss"] + hist2.history["loss"]],
        "val_loss": [float(v) for v in hist1.history["val_loss"] + hist2.history["val_loss"]],
    }
    with open(os.path.join(BASE_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    model.save(MODEL_PATH)
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"[done] model saved to {MODEL_PATH} ({size_mb:.1f} MB), test accuracy = {accuracy:.4f}")


if __name__ == "__main__":
    main()
