import os, pathlib, json
import numpy as np
import tensorflow as tf

print("TF Version:", tf.__version__)

# ============================================================
# CONFIG (EDIT ONLY THESE)
# ============================================================
SAVEDMODEL_DIR = "/Users/saikatdas/Desktop/ConvertTflite/export_EfficientNetB0"
DATASET_DIR    = "/Users/saikatdas/Desktop/ConvertTflite/TrashData"     # representative dataset
OUT_DIR        = "/Users/saikatdas/Desktop/ConvertTflite/New_EfficientNetB0"

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
REP_BATCHES = 15
SEED        = 42
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)

FP32_PATH   = os.path.join(OUT_DIR, "fp32.tflite")
FP16_PATH   = os.path.join(OUT_DIR, "fp16.tflite")
DRQ_PATH    = os.path.join(OUT_DIR, "drq.tflite")
INT8_HYB    = os.path.join(OUT_DIR, "int8_hybrid_floatio.tflite")
INT8_FULL   = os.path.join(OUT_DIR, "int8_full_uint8io.tflite")
LABELS_TXT  = os.path.join(OUT_DIR, "labels.txt")
META_JSON   = os.path.join(OUT_DIR, "export_meta.json")

print("SavedModel:", SAVEDMODEL_DIR)
print("Dataset   :", DATASET_DIR)
print("Out dir   :", OUT_DIR)

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
# NOTE: MUST match training input scale.
# Here we keep float32 with 0..255 (common when training used raw images)
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

# Keep 0..255 float32 (change if your training used /255 or preprocess_input)
rep_ds = rep_ds.map(lambda x, y: (tf.cast(x, tf.float32), y),
                    num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

def rep_data_gen():
    for x, _ in rep_ds.take(REP_BATCHES):
        yield [x]

def save_bytes(path, blob: bytes):
    with open(path, "wb") as f:
        f.write(blob)

# ============================================================
# 1) FP32
# ============================================================
print("\n[1/5] FP32...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tfl = converter.convert()
save_bytes(FP32_PATH, tfl)
print("Saved:", FP32_PATH)

# ============================================================
# 2) FP16 (weights FP16)
# ============================================================
print("\n[2/5] FP16...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tfl = converter.convert()
save_bytes(FP16_PATH, tfl)
print("Saved:", FP16_PATH)

# ============================================================
# 3) DRQ (Dynamic range quant): weights int8, activations float
# ============================================================
print("\n[3/5] DRQ...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tfl = converter.convert()
save_bytes(DRQ_PATH, tfl)
print("Saved:", DRQ_PATH)

# ============================================================
# 4) INT8 HYBRID (weights int8, IO float32)  ✅ Pi-safe
# ============================================================
print("\n[4/5] INT8 HYBRID (float IO)...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# FORCE float IO (hybrid)
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32

tfl = converter.convert()
save_bytes(INT8_HYB, tfl)
print("Saved:", INT8_HYB)

# ============================================================
# 5) FULL INT8 (integer-only, IO uint8)  ✅ fastest if works
# ============================================================
print("\n[5/5] FULL INT8 (uint8 IO)...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data_gen

# Force integer-only kernels
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.uint8
converter.inference_output_type = tf.uint8

tfl = converter.convert()
save_bytes(INT8_FULL, tfl)
print("Saved:", INT8_FULL)

# ============================================================
# Save metadata for paper/debug
# ============================================================
meta = {
    "tf_version": tf.__version__,
    "img_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "rep_batches": REP_BATCHES,
    "seed": SEED,
    "classes": class_names,
    "outputs": {
        "fp32": os.path.basename(FP32_PATH),
        "fp16": os.path.basename(FP16_PATH),
        "drq": os.path.basename(DRQ_PATH),
        "int8_hybrid_floatio": os.path.basename(INT8_HYB),
        "int8_full_uint8io": os.path.basename(INT8_FULL),
    }
}
with open(META_JSON, "w") as f:
    json.dump(meta, f, indent=2)

print("\nDONE ✅")
print("Meta:", META_JSON)