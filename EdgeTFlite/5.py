import os, json, pathlib
import numpy as np
import tensorflow as tf

print("TF Version:", tf.__version__)

# ============================================================
# CONFIG (EDIT ONLY THESE)
# ============================================================

MODEL_FAMILY  = "mobilenetv3"  # efficientnet | mobilenetv1 | mobilenetv3 | nasnetmobile | resnet50 | inceptionv3

SAVEDMODEL_DIR = "/Users/saikatdas/Desktop/IOT BrenchMark/ConvertTflite/export_MobileNetV3Small"
DATASET_DIR    = "/Users/saikatdas/Desktop/IOT BrenchMark/ConvertTflite/TrashData"
OUT_DIR        = "/Users/saikatdas/Desktop/IOT BrenchMark/ConvertTflite/New2 MobileNetV3Small"

BATCH_SIZE  = 32
REP_BATCHES = 15
SEED        = 42


REP_NORM_MODE = "auto"

# If you want to ALSO export int8_hybrid_floatio, set True
EXPORT_INT8_HYBRID_FLOATIO = False

# ============================================================


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_bytes(path, blob: bytes):
    with open(path, "wb") as f:
        f.write(blob)

def convert_and_save(converter: tf.lite.TFLiteConverter, out_path: str):
    tfl = converter.convert()
    save_bytes(out_path, tfl)
    print("Saved:", out_path)

def get_savedmodel_input_hw(savedmodel_dir: str):
    loaded = tf.saved_model.load(savedmodel_dir)
    sig = loaded.signatures.get("serving_default", None)
    if sig is None:
        raise RuntimeError("SavedModel missing 'serving_default' signature.")
    inp = list(sig.structured_input_signature[1].values())[0]
    shape = inp.shape.as_list()  # e.g. [1, H, W, 3]
    if len(shape) != 4:
        raise RuntimeError(f"Unexpected input rank: {shape}")
    if shape[1] is None or shape[2] is None:
        raise RuntimeError(f"Dynamic input shape not supported here: {shape}")
    return int(shape[1]), int(shape[2])

def class_names_from_dir(dataset_dir: str):
    p = pathlib.Path(dataset_dir)
    classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError("No class folders found in dataset dir.")
    return classes

def default_rep_norm_for_family(family: str) -> str:
    """
    IMPORTANT:
    This is a best-practice default, BUT your true source of truth is your training/serving pipeline.
    If your SavedModel already contains a Rescaling layer, then you often want 'raw_255'.
    """
    fam = (family or "").lower().strip()

    # Keras EfficientNet often uses internal rescaling in newer variants -> raw_255 works well.
    if fam == "efficientnet":
        return "raw_255"

    # InceptionV3 / MobileNetV1 classic preprocess_input -> [-1,1]
    if fam in ["mobilenetv1", "inceptionv3", "nasnetmobile"]:
        return "minus1_1"

    # MobileNetV3 is tricky across versions (some variants include preprocessing).
    # Defaulting to raw_255 is usually safest IF model contains rescaling.
    if fam == "mobilenetv3":
        return "raw_255"

    # ResNet50 classic: caffe style
    if fam == "resnet50":
        return "resnet_caffe"

    return "raw_255"

def apply_rep_norm(x, mode: str, family: str):
    """
    x: float32 batch [B,H,W,3], originally from image_dataset_from_directory => range 0..255 (float32)
    Returns float32 batch, normalized.
    """
    fam = (family or "").lower().strip()
    mode = (mode or "auto").lower().strip()

    if mode == "auto":
        mode = default_rep_norm_for_family(fam)

    if mode == "raw_255":
        return x  # 0..255 float32

    if mode == "0_1":
        return x / 255.0

    if mode == "minus1_1":
        return (x / 127.5) - 1.0

    if mode == "resnet_caffe":
        # RGB -> BGR then mean subtraction (caffe)
        # x is RGB in 0..255 float32
        bgr = x[..., ::-1]
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        return bgr - mean

    raise ValueError("REP_NORM_MODE must be one of: auto | raw_255 | 0_1 | minus1_1 | resnet_caffe")

def build_rep_dataset(dataset_dir: str, img_size_hw, batch_size: int, seed: int, rep_batches: int, family: str, rep_norm_mode: str):
    AUTOTUNE = tf.data.AUTOTUNE
    tf.random.set_seed(seed)
    np.random.seed(seed)

    ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size_hw,   # (H,W)
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    # IMPORTANT: representative_dataset generator MUST yield float32 tensors.  [oai_citation:4‡TensorFlow](https://www.tensorflow.org/lite/performance/post_training_quantization?hl=zh-cn&utm_source=chatgpt.com)
    ds = ds.map(lambda x, y: tf.cast(x, tf.float32), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x: apply_rep_norm(x, rep_norm_mode, family), num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)

    def gen():
        taken = 0
        for x in ds.take(rep_batches):
            taken += 1
            yield [x]  # MUST be a list of input tensors
        if taken == 0:
            raise RuntimeError("Representative dataset is empty. Check DATASET_DIR.")

    return gen

# ----------------------------
# Main export
# ----------------------------
def main():
    ensure_dir(OUT_DIR)

    FP32_PATH  = os.path.join(OUT_DIR, "fp32.tflite")
    FP16_PATH  = os.path.join(OUT_DIR, "fp16.tflite")
    DRQ_PATH   = os.path.join(OUT_DIR, "drq.tflite")
    INT8_FULL  = os.path.join(OUT_DIR, "int8_full_int8io.tflite")
    INT8_HYB   = os.path.join(OUT_DIR, "int8_hybrid_floatio.tflite")
    LABELS_TXT = os.path.join(OUT_DIR, "labels.txt")
    META_JSON  = os.path.join(OUT_DIR, "export_meta.json")

    print("Model family:", MODEL_FAMILY)
    print("SavedModel  :", SAVEDMODEL_DIR)
    print("Dataset     :", DATASET_DIR)
    print("Out dir     :", OUT_DIR)

    # Input size from SavedModel signature (best fix)
    H, W = get_savedmodel_input_hw(SAVEDMODEL_DIR)
    IMG_SIZE = (H, W)
    print("Detected model input size:", IMG_SIZE)

    # labels.txt (locks class order)
    class_names = class_names_from_dir(DATASET_DIR)
    with open(LABELS_TXT, "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(c + "\n")
    print("Classes:", class_names)
    print("Saved labels:", LABELS_TXT)

    # Representative dataset generator for INT8
    rep_gen = build_rep_dataset(
        dataset_dir=DATASET_DIR,
        img_size_hw=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        rep_batches=REP_BATCHES,
        family=MODEL_FAMILY,
        rep_norm_mode=REP_NORM_MODE
    )

    # 1) FP32
    print("\n[1/4] FP32...")
    c = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    convert_and_save(c, FP32_PATH)

    # 2) FP16
    print("\n[2/4] FP16...")
    c = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.target_spec.supported_types = [tf.float16]  #  [oai_citation:5‡TensorFlow](https://www.tensorflow.org/lite/performance/post_training_quantization?hl=zh-cn&utm_source=chatgpt.com)
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    convert_and_save(c, FP16_PATH)

    # 3) DRQ (Dynamic Range Quantization)
    print("\n[3/4] DRQ...")
    c = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    convert_and_save(c, DRQ_PATH)

    # 4) FULL INT8 (int8 ops + int8 IO)
    print("\n[4/4] FULL INT8 (int8 IO)...")
    c = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep_gen
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  #  [oai_citation:6‡TensorFlow](https://www.tensorflow.org/lite/performance/post_training_quantization?hl=zh-cn&utm_source=chatgpt.com)
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    convert_and_save(c, INT8_FULL)

  

    # Meta for reproducibility
    resolved_mode = REP_NORM_MODE
    if resolved_mode.lower().strip() == "auto":
        resolved_mode = default_rep_norm_for_family(MODEL_FAMILY)

    meta = {
        "tf_version": tf.__version__,
        "model_family": MODEL_FAMILY,
        "savedmodel_dir": SAVEDMODEL_DIR,
        "dataset_dir": DATASET_DIR,
        "detected_model_input_size": [H, W],
        "batch_size": BATCH_SIZE,
        "rep_batches": REP_BATCHES,
        "seed": SEED,
        "rep_norm_mode_requested": REP_NORM_MODE,
        "rep_norm_mode_resolved": resolved_mode,
        "classes": class_names,
        "outputs": {
            "fp32": os.path.basename(FP32_PATH),
            "fp16": os.path.basename(FP16_PATH),
            "drq": os.path.basename(DRQ_PATH),
            "int8_full_int8io": os.path.basename(INT8_FULL),
            "int8_hybrid_floatio": os.path.basename(INT8_HYB) if EXPORT_INT8_HYBRID_FLOATIO else None,
            "labels": os.path.basename(LABELS_TXT),
        }
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("\nDONE ✅")
    print("Meta:", META_JSON)

if __name__ == "__main__":
    main()