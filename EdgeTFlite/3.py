import os, pathlib, json
import numpy as np
import tensorflow as tf

print("TF Version:", tf.__version__)

# ============================================================
# CONFIG (EDIT ONLY THESE)
# ============================================================
SAVEDMODEL_DIR = "/Users/saikatdas/Desktop/IOT BrenchMark/ConvertTflite/export_InceptionV3"
DATASET_DIR    = "/Users/saikatdas/Desktop/IOT BrenchMark/ConvertTflite/TrashData"
OUT_DIR        = "/Users/saikatdas/Desktop/IOT BrenchMark/ConvertTflite/New_InceptionV3"

BATCH_SIZE  = 32
REP_BATCHES = 15
SEED        = 42

# InceptionV3 usually uses preprocess_input => [-1, 1]
REP_NORM_MODE = "minus1_1"   # "minus1_1" | "0_1" | "raw_255"
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)

FP32_PATH   = os.path.join(OUT_DIR, "fp32.tflite")
FP16_PATH   = os.path.join(OUT_DIR, "fp16.tflite")
DRQ_PATH    = os.path.join(OUT_DIR, "drq.tflite")
INT8_HYB    = os.path.join(OUT_DIR, "int8_hybrid_floatio.tflite")
INT8_FULL   = os.path.join(OUT_DIR, "int8_full_int8io.tflite")
LABELS_TXT  = os.path.join(OUT_DIR, "labels.txt")
META_JSON   = os.path.join(OUT_DIR, "export_meta.json")

print("SavedModel:", SAVEDMODEL_DIR)
print("Dataset   :", DATASET_DIR)
print("Out dir   :", OUT_DIR)

# ============================================================
# Read SavedModel input size (important fix)
# ============================================================
def get_savedmodel_input_hw(savedmodel_dir):
    loaded = tf.saved_model.load(savedmodel_dir)
    # usually "serving_default"
    sig = loaded.signatures["serving_default"]
    # take first input tensor
    inp = list(sig.structured_input_signature[1].values())[0]
    shape = inp.shape.as_list()  # [1, H, W, 3]
    if shape[1] is None or shape[2] is None:
        raise RuntimeError(f"SavedModel has dynamic input shape: {shape}.")
    return int(shape[1]), int(shape[2])

MODEL_H, MODEL_W = get_savedmodel_input_hw(SAVEDMODEL_DIR)
IMG_SIZE = (MODEL_H, MODEL_W)
print("Detected model input size:", IMG_SIZE)

# ============================================================
# labels.txt
# ============================================================
def class_names_from_dir(dataset_dir):
    p = pathlib.Path(dataset_dir)
    classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError("No class folders found in dataset dir.")
    return classes

class_names = class_names_from_dir(DATASET_DIR)
with open(LABELS_TXT, "w") as f:
    for c in class_names:
        f.write(c + "\n")

print("Classes:", class_names)
print("Saved labels:", LABELS_TXT)

# ============================================================
# Representative dataset (calibration)
# MUST match training input scale AND model input size
# ============================================================
AUTOTUNE = tf.data.AUTOTUNE
tf.random.set_seed(SEED)
np.random.seed(SEED)

rep_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,          # ✅ always matches SavedModel
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

rep_ds = rep_ds.map(lambda x, y: (apply_rep_norm(x), y),
                    num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

def rep_data_gen():
    for x, _ in rep_ds.take(REP_BATCHES):
        yield [x]

def save_bytes(path, blob: bytes):
    with open(path, "wb") as f:
        f.write(blob)

def convert_and_save(converter, out_path):
    tfl = converter.convert()
    save_bytes(out_path, tfl)
    print("Saved:", out_path)

# ============================================================
# 1) FP32
# ============================================================
print("\n[1/5] FP32...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
convert_and_save(converter, FP32_PATH)

# ============================================================
# 2) FP16
# ============================================================
print("\n[2/5] FP16...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
convert_and_save(converter, FP16_PATH)

# ============================================================
# 3) DRQ
# ============================================================
print("\n[3/5] DRQ...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
convert_and_save(converter, DRQ_PATH)

# ============================================================
# 4) INT8 HYBRID (float IO)
# ============================================================
print("\n[4/5] INT8 HYBRID (float IO)...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32
convert_and_save(converter, INT8_HYB)

# ============================================================
# 5) FULL INT8 (int8 IO)
# ============================================================
print("\n[5/5] FULL INT8 (int8 IO)...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8
converter.experimental_new_converter = True
convert_and_save(converter, INT8_FULL)

# ============================================================
# Meta
# ============================================================
meta = {
    "tf_version": tf.__version__,
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
        "int8_hybrid_floatio": os.path.basename(INT8_HYB),
        "int8_full_int8io": os.path.basename(INT8_FULL),
    }
}
with open(META_JSON, "w") as f:
    json.dump(meta, f, indent=2)

print("\nDONE ✅")
print("Meta:", META_JSON)