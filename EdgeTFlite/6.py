import os, json, pathlib
import numpy as np
import tensorflow as tf

print("TF Version:", tf.__version__)

# ============================================================
# CONFIG (EDIT ONLY THESE)
# ============================================================
SAVEDMODEL_DIR = "/Users/saikatdas/Desktop/IOT BrenchMark/ConvertTflite/export_MobileNetV1"
DATASET_DIR    = "/Users/saikatdas/Desktop/IOT BrenchMark/ConvertTflite/TrashData"
OUT_DIR        = "/Users/saikatdas/Desktop/IOT BrenchMark/ConvertTflite/OUT MobileNetV1"

# Model family (affects AUTO preprocessing)
MODEL_FAMILY = "mobilenetv1"  # efficientnet|mobilenetv1|mobilenetv3|nasnetmobile|resnet50|inceptionv3

# If your SavedModel already contains preprocessing layers (e.g., Rescaling/Normalization),
# then use raw_255 so you don't preprocess twice.
INCLUDE_PREPROCESSING_IN_MODEL = True

# Preprocess mode for representative dataset:
# "auto" | "minus1_1" | "0_1" | "raw_255" | "resnet_caffe"
REP_NORM_MODE = "auto"

BATCH_SIZE  = 32
REP_BATCHES = 15    # typical 100-500 samples total; here it's batches
SEED        = 42
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)

FP32_PATH   = os.path.join(OUT_DIR, "fp32.tflite")
FP16_PATH   = os.path.join(OUT_DIR, "fp16.tflite")
DRQ_PATH    = os.path.join(OUT_DIR, "drq.tflite")
INT8_FULL   = os.path.join(OUT_DIR, "int8_full_int8io.tflite")
LABELS_TXT  = os.path.join(OUT_DIR, "labels.txt")
META_JSON   = os.path.join(OUT_DIR, "export_meta.json")

# -------------------------
# Helpers
# -------------------------
def get_savedmodel_input_hw(savedmodel_dir: str):
    loaded = tf.saved_model.load(savedmodel_dir)
    sig = loaded.signatures["serving_default"]
    inp = list(sig.structured_input_signature[1].values())[0]
    shape = inp.shape.as_list()  # [1, H, W, 3]
    if len(shape) != 4:
        raise RuntimeError(f"Unexpected input shape: {shape}")
    if shape[1] is None or shape[2] is None:
        raise RuntimeError(f"SavedModel has dynamic H/W: {shape}")
    return int(shape[1]), int(shape[2])

def class_names_from_dir(dataset_dir: str):
    p = pathlib.Path(dataset_dir)
    classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError("No class folders found in dataset dir.")
    return classes

def resolve_norm_mode(model_family: str, requested: str, include_preproc: bool) -> str:
    mf = (model_family or "").lower().strip()
    req = (requested or "auto").lower().strip()

    if req != "auto":
        return req

    # AUTO logic:
    # If preprocessing is inside the model, feed raw pixels (0..255)
    if include_preproc:
        return "raw_255"

    # Otherwise, match common Keras app preprocessing
    if mf == "resnet50":
        return "resnet_caffe"
    if mf in [ "mobilenetv3", "inceptionv3", "nasnetmobile"]:
        return "minus1_1"
    if mf == "efficientnet":
        # EfficientNet often has preprocessing included; if not, many pipelines still use raw_255 + internal rescaling.
        return "raw_255"
    if mf == "mobilenetv1":
        return "raw_255"
    return "raw_255"

def apply_norm_tf(x, mode: str):
    """
    x: float32 image tensor in [0..255]
    returns float32 after preprocessing
    """
    mode = (mode or "raw_255").lower().strip()

    if mode == "raw_255":
        return x
    if mode == "0_1":
        return x / 255.0
    if mode == "minus1_1":
        return (x / 127.5) - 1.0
    if mode == "resnet_caffe":
        # RGB -> BGR + mean subtraction (caffe)
        x = x[..., ::-1]
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        return x - mean
    raise ValueError("REP_NORM_MODE must be: auto | minus1_1 | 0_1 | raw_255 | resnet_caffe")

def save_bytes(path, blob: bytes):
    with open(path, "wb") as f:
        f.write(blob)

def convert_and_save(converter, out_path):
    tfl = converter.convert()
    save_bytes(out_path, tfl)
    print("Saved:", out_path)

# -------------------------
# Main
# -------------------------
print("SavedModel:", SAVEDMODEL_DIR)
print("Dataset   :", DATASET_DIR)
print("Out dir   :", OUT_DIR)

MODEL_H, MODEL_W = get_savedmodel_input_hw(SAVEDMODEL_DIR)
IMG_SIZE = (MODEL_H, MODEL_W)
print("Detected model input size:", IMG_SIZE)

class_names = class_names_from_dir(DATASET_DIR)
with open(LABELS_TXT, "w", encoding="utf-8") as f:
    for c in class_names:
        f.write(c + "\n")
print("Classes:", class_names)
print("Saved labels:", LABELS_TXT)

norm_mode = resolve_norm_mode(MODEL_FAMILY, REP_NORM_MODE, INCLUDE_PREPROCESSING_IN_MODEL)
print("Resolved REP_NORM_MODE:", norm_mode)

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
    seed=SEED,
)

rep_ds = rep_ds.map(
    lambda x, y: apply_norm_tf(tf.cast(x, tf.float32), norm_mode),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

def rep_data_gen():
    # Must yield float32 tensors; list-of-inputs is accepted by TFLite converter.  [oai_citation:3‡TensorFlow](https://www.tensorflow.org/lite/performance/post_training_quantization?hl=zh-cn)
    taken = 0
    for x in rep_ds.take(REP_BATCHES):
        taken += 1
        yield [x]
    if taken == 0:
        raise RuntimeError("Representative dataset is empty. Check DATASET_DIR.")

# ============================================================
# 1) FP32
# ============================================================
print("\n[1/4] FP32...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
convert_and_save(converter, FP32_PATH)

# ============================================================
# 2) FP16 (weight quant)
# ============================================================
print("\n[2/4] FP16...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  #  [oai_citation:4‡TensorFlow](https://www.tensorflow.org/lite/performance/post_training_quantization?hl=zh-cn)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
convert_and_save(converter, FP16_PATH)

# ============================================================
# 3) DRQ (dynamic range quant)
# ============================================================
print("\n[3/4] DRQ...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  #  [oai_citation:5‡TensorFlow](https://www.tensorflow.org/lite/performance/post_training_quantization?hl=zh-cn)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
convert_and_save(converter, DRQ_PATH)

# ============================================================
# 4) FULL INT8 (int8 IO)
# ============================================================
print("\n[4/4] FULL INT8 (int8 IO)...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  #  [oai_citation:6‡TensorFlow](https://www.tensorflow.org/lite/performance/post_training_quantization?hl=zh-cn)
converter.inference_input_type  = tf.int8                               #  [oai_citation:7‡TensorFlow](https://www.tensorflow.org/lite/performance/post_training_quantization?hl=zh-cn)
converter.inference_output_type = tf.int8                               #  [oai_citation:8‡TensorFlow](https://www.tensorflow.org/lite/performance/post_training_quantization?hl=zh-cn)
convert_and_save(converter, INT8_FULL)

meta = {
    "tf_version": tf.__version__,
    "savedmodel_dir": SAVEDMODEL_DIR,
    "dataset_dir": DATASET_DIR,
    "detected_model_input_size": [MODEL_H, MODEL_W],
    "batch_size": BATCH_SIZE,
    "rep_batches": REP_BATCHES,
    "seed": SEED,
    "model_family": MODEL_FAMILY,
    "include_preprocessing_in_model": INCLUDE_PREPROCESSING_IN_MODEL,
    "resolved_rep_norm_mode": norm_mode,
    "classes": class_names,
    "outputs": {
        "fp32": os.path.basename(FP32_PATH),
        "fp16": os.path.basename(FP16_PATH),
        "drq": os.path.basename(DRQ_PATH),
        "int8_full_int8io": os.path.basename(INT8_FULL),
        "labels": os.path.basename(LABELS_TXT),
    }
}
with open(META_JSON, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("\nDONE ✅")
print("Meta:", META_JSON)