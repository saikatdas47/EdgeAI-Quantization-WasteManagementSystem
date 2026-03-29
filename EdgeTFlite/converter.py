import os, pathlib
import tensorflow as tf

print("TF Version:", tf.__version__)

# =========================
# CONFIG
# =========================
OUT_DIR = "/Users/saikatdas/Desktop/converter/New_Resnet50"
os.makedirs(OUT_DIR, exist_ok=True)

# IMPORTANT: use SavedModel folder (exported from Keras3)
SAVEDMODEL_DIR = "/Users/saikatdas/Desktop/converter/savedmodel_export_OUT_ResNet50"

DATASET_DIR = "/Users/saikatdas/Desktop/converter/TrashData"  # for INT8 calibration

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
REP_BATCHES = 15
SEED = 42

FP32_PATH = os.path.join(OUT_DIR, "best_fp32.tflite")
DRQ_PATH  = os.path.join(OUT_DIR, "best_dynamic_range.tflite")
INT8_PATH = os.path.join(OUT_DIR, "best_int8.tflite")
LABELS_TXT = os.path.join(OUT_DIR, "labels.txt")

print("Converting SavedModel:", SAVEDMODEL_DIR)

# =========================
# labels.txt
# =========================
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
print("Saved labels:", LABELS_TXT)
print("Classes:", class_names)

# =========================
# FP32
# =========================
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
tflite_fp32 = converter.convert()
open(FP32_PATH, "wb").write(tflite_fp32)
print("Saved FP32:", FP32_PATH)

# =========================
# DRQ
# =========================
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_drq = converter.convert()
open(DRQ_PATH, "wb").write(tflite_drq)
print("Saved DRQ:", DRQ_PATH)

# =========================
# INT8 HYBRID (float IO, int8 weights)
# =========================
AUTOTUNE = tf.data.AUTOTUNE

rep_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

rep_ds = rep_ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

def rep_data_gen():
    for x, _ in rep_ds.take(REP_BATCHES):
        yield [x]

converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_int8 = converter.convert()
open(INT8_PATH, "wb").write(tflite_int8)
print("Saved INT8 hybrid:", INT8_PATH)

print("\nDONE ✅")