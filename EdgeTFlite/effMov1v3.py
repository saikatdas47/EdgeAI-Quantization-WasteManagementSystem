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

BATCH_SIZE  = 32
REP_BATCHES = 15
SEED        = 42

# MUST match TRAINING input pipeline (because preprocess was done outside model)
# "minus1_1" | "0_1" | "raw_255"
REP_NORM_MODE = "raw_255"

# If you want hybrid int8 (weights int8, float IO) enable this
EXPORT_INT8_HYBRID_FLOATIO = True
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)

FP32_PATH   = os.path.join(OUT_DIR, "fp32.tflite")
FP16_PATH   = os.path.join(OUT_DIR, "fp16.tflite")
DRQ_PATH    = os.path.join(OUT_DIR, "drq.tflite")
INT8_HYB    = os.path.join(OUT_DIR, "int8_hybrid_floatio.tflite")
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
        raise RuntimeError(f"SavedModel has dynamic input shape: {shape}. Fix export with fixed IMG_SIZE.")
    return int(shape[1]), int(shape[2])

MODEL_H, MODEL_W = get_savedmodel_input_hw(SAVEDMODEL_DIR)
IMG_SIZE = (MODEL_H, MODEL_W)
print("Detected input size:", IMG_SIZE)

# ============================================================
# Lock class order (VERY IMPORTANT)
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
# Representative dataset (calibration)
# NOTE: representative_dataset MUST yield float32
# ============================================================
AUTOTUNE = tf.data.AUTOTUNE
tf.random.set_seed(SEED)
np.random.seed(SEED)

rep_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

def apply_rep_norm(x):
    x = tf.cast(x, tf.float32)
    if REP_NORM_MODE == "minus1_1":
        x = (x / 127.5) - 1.0
    elif REP_NORM_MODE == "0_1":
        x = x / 255.0
    elif REP_NORM_MODE == "raw_255":
        x = x
    else:
        raise ValueError("REP_NORM_MODE must be: minus1_1 | 0_1 | raw_255")
    return x

rep_ds = rep_ds.map(lambda x, y: apply_rep_norm(x), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

def rep_data_gen():
    taken = 0
    for x in rep_ds.take(REP_BATCHES):
        taken += 1
        yield [x]  # MUST be list of input(s)
    if taken == 0:
        raise RuntimeError("Representative dataset empty. Check DATASET_DIR.")

def save_bytes(path, blob: bytes):
    with open(path, "wb") as f:
        f.write(blob)

def convert_and_save(converter, out_path):
    tfl = converter.convert()
    save_bytes(out_path, tfl)
    print("Saved:", out_path)

# ============================================================
# 1) FP32 (no optimization)
# ============================================================
print("\n[1/4] FP32...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
convert_and_save(converter, FP32_PATH)

# ============================================================
# 2) FP16 (weights float16)
# ============================================================
print("\n[2/4] FP16...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
convert_and_save(converter, FP16_PATH)

# ============================================================
# 3) DRQ (dynamic range quant, weights int8, activations float)
# ============================================================
print("\n[3/4] DRQ...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
convert_and_save(converter, DRQ_PATH)

# ============================================================
# 4) FULL INT8 (int8 ops + int8 IO)
# ============================================================
print("\n[4/4] FULL INT8 (int8 IO)...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # integer-only ops  [oai_citation:1‡TensorFlow](https://www.tensorflow.org/lite/performance/post_training_quantization?hl=ja)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
convert_and_save(converter, INT8_FULL)

# ============================================================
# Optional: INT8 HYBRID float IO (weights int8, float input/output)
# ============================================================
if EXPORT_INT8_HYBRID_FLOATIO:
    print("\n[extra] INT8 HYBRID (float IO)...")
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # allows float fallback  [oai_citation:2‡TensorFlow](https://www.tensorflow.org/lite/performance/post_training_quantization?hl=ja)
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    convert_and_save(converter, INT8_HYB)

# ============================================================
# Meta
# ============================================================
meta = {
    "tf_version": tf.__version__,
    "savedmodel_dir": SAVEDMODEL_DIR,
    "dataset_dir": DATASET_DIR,
    "detected_model_input_size": [MODEL_H, MODEL_W],
    "batch_size": BATCH_SIZE,
    "rep_batches": REP_BATCHES,
    "seed": SEED,
    "rep_norm_mode": REP_NORM_MODE,
    "classes": class_names,
    "outputs": {
        "fp32": os.path.basename(FP32_PATH),
        "fp16": os.path.basename(FP16_PATH),
        "drq": os.path.basename(DRQ_PATH),
        "int8_full_int8io": os.path.basename(INT8_FULL),
        "labels": os.path.basename(LABELS_TXT),
        "classes_json": os.path.basename(CLASSES_JSON),
        "int8_hybrid_floatio": (os.path.basename(INT8_HYB) if EXPORT_INT8_HYBRID_FLOATIO else None)
    }
}

with open(META_JSON, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("\nDONE ✅")
print("Meta:", META_JSON)