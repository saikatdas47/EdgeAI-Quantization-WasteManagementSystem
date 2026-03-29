import os, json, pathlib
import numpy as np
import tensorflow as tf

print("TF Version:", tf.__version__)

# ============================================================
# CONFIG (EDIT ONLY THESE)
# ============================================================
SAVEDMODEL_DIR = "/Users/saikatdas/Desktop/IOT BrenchMark/ConvertTflite/export_NASNetMobile"
DATASET_DIR    = "/Users/saikatdas/Desktop/IOT BrenchMark/ConvertTflite/TrashData"
OUT_DIR        = "/Users/saikatdas/Desktop/IOT BrenchMark/ConvertTflite/OUT NASNetMobile"

SEED = 42

# NASNet preprocess_input style (usually minus1_1)
REP_NORM_MODE = "minus1_1"

# For speed: build dataset with batch>1, but calibration MUST be batch=1
BATCH_SIZE_BUILD = 32
REP_SAMPLES = 200  # 100~300 recommended
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)
INT8_FULL   = os.path.join(OUT_DIR, "int8_full_int8io.tflite")
LABELS_TXT  = os.path.join(OUT_DIR, "labels.txt")
CLASSES_JSON = os.path.join(OUT_DIR, "classes.json")
META_JSON   = os.path.join(OUT_DIR, "export_meta.json")

print("SavedModel:", SAVEDMODEL_DIR)
print("Dataset   :", DATASET_DIR)
print("Out dir   :", OUT_DIR)

# ============================================================
# Detect SavedModel input size
# ============================================================
def get_savedmodel_input_hw(savedmodel_dir: str):
    loaded = tf.saved_model.load(savedmodel_dir)
    sig = loaded.signatures["serving_default"]
    inp = list(sig.structured_input_signature[1].values())[0]
    shape = inp.shape.as_list()  # [1, H, W, 3]
    if shape[1] is None or shape[2] is None:
        raise RuntimeError(f"SavedModel has dynamic input shape: {shape}. Export with fixed IMG_SIZE.")
    return int(shape[1]), int(shape[2])

MODEL_H, MODEL_W = get_savedmodel_input_hw(SAVEDMODEL_DIR)
IMG_SIZE = (MODEL_H, MODEL_W)
print("Detected input size:", IMG_SIZE)

# ============================================================
# Lock class order
# ============================================================
def class_names_from_dir(dataset_dir: str):
    p = pathlib.Path(dataset_dir)
    classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError("No class folders found in dataset dir.")
    return classes

class_names = class_names_from_dir(DATASET_DIR)

with open(LABELS_TXT, "w", encoding="utf-8") as f:
    for c in class_names:
        f.write(c + "\n")

with open(CLASSES_JSON, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

print("Classes:", class_names)
print("Saved:", LABELS_TXT)
print("Saved:", CLASSES_JSON)

# ============================================================
# Representative dataset (calibration) - FAST + CORRECT
# - representative_dataset MUST yield float32
# - MUST be batch=1 for stable int8
# ============================================================
AUTOTUNE = tf.data.AUTOTUNE
tf.random.set_seed(SEED)
np.random.seed(SEED)

def apply_rep_norm(x):
    x = tf.cast(x, tf.float32)
    m = (REP_NORM_MODE or "").strip().lower()
    if m == "minus1_1":
        return (x / 127.5) - 1.0
    if m == "0_1":
        return x / 255.0
    if m == "raw_255":
        return x
    raise ValueError("REP_NORM_MODE must be: minus1_1 | 0_1 | raw_255")

# labels=None => dataset yields ONLY images (faster + simpler)
rep_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels=None,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE_BUILD,
    shuffle=True,
    seed=SEED
)

rep_ds = rep_ds.map(apply_rep_norm, num_parallel_calls=AUTOTUNE)
rep_ds = rep_ds.unbatch().batch(1).prefetch(AUTOTUNE)  # ✅ MUST

def rep_data_gen():
    taken = 0
    for x in rep_ds.take(REP_SAMPLES):
        taken += 1
        yield [tf.cast(x, tf.float32)]
    if taken == 0:
        raise RuntimeError("Representative dataset empty. Check DATASET_DIR.")

def save_bytes(path, blob: bytes):
    with open(path, "wb") as f:
        f.write(blob)

# ============================================================
# FULL INT8 (int8 ops + int8 IO) ONLY
# ============================================================
print("\n[INT8] Converting FULL INT8 (int8 IO)...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tfl = converter.convert()
save_bytes(INT8_FULL, tfl)
print("Saved:", INT8_FULL)

# ============================================================
# Meta
# ============================================================
meta = {
    "tf_version": tf.__version__,
    "savedmodel_dir": SAVEDMODEL_DIR,
    "dataset_dir": DATASET_DIR,
    "detected_model_input_size": [MODEL_H, MODEL_W],
    "batch_size_build": BATCH_SIZE_BUILD,
    "rep_samples": REP_SAMPLES,
    "seed": SEED,
    "rep_norm_mode": REP_NORM_MODE,
    "classes": class_names,
    "output": os.path.basename(INT8_FULL),
    "labels": os.path.basename(LABELS_TXT),
    "classes_json": os.path.basename(CLASSES_JSON),
}

with open(META_JSON, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("\nDONE ✅")
print("Meta:", META_JSON)