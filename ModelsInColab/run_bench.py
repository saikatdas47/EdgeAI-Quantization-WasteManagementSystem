#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, random, traceback, hashlib, gc
from pathlib import Path

import numpy as np
from PIL import Image
import psutil
from sklearn.metrics import classification_report, f1_score, accuracy_score


# ============================================================
# CONFIG (EDIT ONLY THESE)
# ============================================================

MODEL_FAMILY = "mobilenetv1"  # efficientnet|mobilenetv1|mobilenetv3|nasnetmobile|resnet50|inceptionv3

MODEL_FILES = {
    "fp32": "fp32.tflite",
    "fp16": "fp16.tflite",
    "drq": "drq.tflite",
    "int8_hybrid_floatio": "int8_hybrid_floatio.tflite",
    "int8_full_uint8io": "int8_full_uint8io.tflite"
}

MODEL_DIR = Path("./MobileNetV1")

DATASET_DIR  = Path("./Users/saikatdas/Desktop/lol/data2")

OUT_DIR = Path("./bench_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_RUNS  = 30
BENCH_RUNS   = 300
LOAD_REPS    = 5
SAMPLE_EVERY = 10
NUM_THREADS  = 4
SEED         = 42

CLASSES_JSON = Path("./classes.json")  # recommended

# Preprocess must match training.
# Options: "keras", "0_1", "minus1_1", "resnet_caffe"
FLOAT_PREPROCESS_MODE = "keras"

# ============================================================


# -------------------------
# Backend detection
# -------------------------
TFLITE_RT = None
TFLITE_RT_VER = "not_installed"
TF = None
TF_VER = "not_installed"

try:
    import tflite_runtime.interpreter as tflite_rt
    TFLITE_RT = tflite_rt
    TFLITE_RT_VER = getattr(tflite_rt, "__version__", "unknown")
except Exception:
    pass

try:
    import tensorflow as tf
    TF = tf
    TF_VER = getattr(tf, "__version__", "unknown")
except Exception:
    pass


# -------------------------
# Repro
# -------------------------
random.seed(SEED)
np.random.seed(SEED)


# -------------------------
# Utilities
# -------------------------
def sha1_file(path: Path, chunk_size=1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_temp_c():
    p = Path("/sys/class/thermal/thermal_zone0/temp")
    if p.exists():
        try:
            return float(p.read_text().strip()) / 1000.0
        except Exception:
            return None
    return None


def list_dataset(dataset_dir: Path, classes_json: Path):
    dataset_dir = Path(dataset_dir)

    if classes_json and classes_json.exists():
        classes = json.loads(classes_json.read_text(encoding="utf-8"))
        if not isinstance(classes, list) or not all(isinstance(x, str) for x in classes):
            raise ValueError("classes.json must be a JSON list like ['biological','metal','paper','plastic']")
        class_order_source = "classes.json"
    else:
        classes = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
        class_order_source = "folder_sorted"

    img_paths, labels = [], []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for i, c in enumerate(classes):
        class_dir = dataset_dir / c
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder missing: {class_dir}")

        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in valid_ext:
                img_paths.append(str(p))
                labels.append(i)

    if len(img_paths) == 0:
        raise RuntimeError(f"No images found under: {dataset_dir}")

    img_paths = np.array(img_paths)
    labels = np.array(labels)

    # Sanity print
    print("\n[Dataset]")
    print("Path:", dataset_dir)
    print("Classes:", classes, "| class_order_source:", class_order_source)
    print("Total images:", len(img_paths))
    for i, c in enumerate(classes):
        print(f"  {c}: {int(np.sum(labels == i))}")

    return img_paths, labels, classes, class_order_source


def family_input_size(family: str):
    fam = (family or "").lower().strip()
    if fam == "inceptionv3":
        return (299, 299)
    return (224, 224)


def preprocess_float(img_rgb_uint8: np.ndarray, family: str, mode: str) -> np.ndarray:
    """
    img_rgb_uint8: HxWx3 uint8 in [0..255]
    returns float32 HxWx3
    """
    x = img_rgb_uint8.astype(np.float32)
    mode = (mode or "keras").lower().strip()
    fam = (family or "").lower().strip()

    if mode == "0_1":
        return (x / 255.0).astype(np.float32)

    if mode == "minus1_1":
        return ((x / 127.5) - 1.0).astype(np.float32)

    if mode == "resnet_caffe":
        # RGB -> BGR + mean subtraction
        x = x[..., ::-1]
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68
        return x.astype(np.float32)

    # mode == "keras" (family-aware defaults)
    # IMPORTANT correction:
    # - EfficientNet in Keras typically expects [0..255] with internal rescaling
    # - MobileNetV1/V3, InceptionV3, NASNetMobile typically expect [-1..1]
    # - ResNet50 typically caffe
    if fam == "efficientnet":
        return x.astype(np.float32)

    if fam in ["mobilenetv1", "mobilenetv3", "inceptionv3", "nasnetmobile"]:
        return ((x / 127.5) - 1.0).astype(np.float32)

    if fam == "resnet50":
        return preprocess_float(img_rgb_uint8, family=fam, mode="resnet_caffe")

    # fallback
    return (x / 255.0).astype(np.float32)


def load_image_as_input(path: str, input_hw, input_details, family: str, float_mode: str):
    """
    Builds (1,H,W,C) tensor matching model input dtype.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((input_hw[1], input_hw[0]), Image.BILINEAR)
    rgb = np.array(img, dtype=np.uint8)

    in_dtype = input_details["dtype"]
    scale, zero_point = input_details.get("quantization", (0.0, 0))
    scale = float(scale)
    zero_point = int(zero_point)

    # Float input
    if in_dtype == np.float32:
        x = preprocess_float(rgb, family=family, mode=float_mode)
        return np.expand_dims(x, axis=0).astype(np.float32)

    # Integer input: quantize from raw 0..255 unless scale suggests otherwise
    x_f = rgb.astype(np.float32)

    if scale > 0:
        # If scale==1 and zp==0, this becomes raw uint8 (good for many uint8 IO models)
        x_q = np.round((x_f / scale) + zero_point)
    else:
        x_q = x_f + zero_point

    if in_dtype == np.uint8:
        x_q = np.clip(x_q, 0, 255).astype(np.uint8)
    elif in_dtype == np.int8:
        x_q = np.clip(x_q, -128, 127).astype(np.int8)
    else:
        x_q = x_q.astype(in_dtype)

    return np.expand_dims(x_q, axis=0)


def make_interpreter(model_path: Path, num_threads: int):
    if TFLITE_RT is not None:
        return TFLITE_RT.Interpreter(model_path=str(model_path), num_threads=int(num_threads)), "tflite_runtime"
    if TF is not None:
        return TF.lite.Interpreter(model_path=str(model_path), num_threads=int(num_threads)), "tensorflow_lite"
    raise RuntimeError("Neither tflite_runtime nor tensorflow is installed.")


def measure_load_time(model_path: Path, num_threads: int, reps: int):
    times = []
    backend = None
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        interp, backend = make_interpreter(model_path, num_threads)
        interp.allocate_tensors()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        del interp
        gc.collect()
    return float(np.mean(times)), float(np.std(times)), backend


def safe_argmax(output: np.ndarray) -> int:
    """
    Works for output shapes:
    - (1, num_classes)
    - (num_classes,)
    - any tensor that can be flattened
    """
    out = np.array(output)
    out = out.reshape(-1)
    return int(np.argmax(out))


def run_benchmark(model_family: str, model_path: Path, dataset_dir: Path):
    model_family = (model_family or "").lower().strip()
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    paths, labels, class_names, class_order_source = list_dataset(dataset_dir, CLASSES_JSON)

    load_mean, load_std, backend_used = measure_load_time(model_path, NUM_THREADS, LOAD_REPS)

    interp, backend_used = make_interpreter(model_path, NUM_THREADS)
    interp.allocate_tensors()

    in_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]

    input_hw = family_input_size(model_family)
    input_shape = list(in_details["shape"])
    input_dtype = str(in_details["dtype"])
    qscale, qzp = in_details.get("quantization", (0.0, 0))
    input_quant = {"scale": float(qscale), "zero_point": int(qzp)}

    proc = psutil.Process(os.getpid())
    temp_start = read_temp_c()

    # -------------------------
    # 1) Accuracy (full dataset)
    # -------------------------
    y_true = labels.tolist()
    y_pred = []

    for p in paths:
        x = load_image_as_input(p, input_hw, in_details, model_family, FLOAT_PREPROCESS_MODE)
        interp.set_tensor(in_details["index"], x)
        interp.invoke()
        out = interp.get_tensor(out_details["index"])
        y_pred.append(safe_argmax(out))

    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro"))
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # -------------------------
    # 2) Latency benchmark
    # -------------------------
    idxs = np.random.randint(0, len(paths), size=(WARMUP_RUNS + BENCH_RUNS,))
    sample_paths = paths[idxs]

    for i in range(WARMUP_RUNS):
        x = load_image_as_input(sample_paths[i], input_hw, in_details, model_family, FLOAT_PREPROCESS_MODE)
        interp.set_tensor(in_details["index"], x)
        interp.invoke()

    infer_times_ms = []
    cpu_samples = []
    ram_samples = []
    temp_samples = []

    _ = psutil.cpu_percent(percpu=True)  # prime

    for i in range(BENCH_RUNS):
        x = load_image_as_input(sample_paths[WARMUP_RUNS + i], input_hw, in_details, model_family, FLOAT_PREPROCESS_MODE)

        t0 = time.perf_counter()
        interp.set_tensor(in_details["index"], x)
        interp.invoke()
        _ = interp.get_tensor(out_details["index"])
        t1 = time.perf_counter()

        infer_times_ms.append((t1 - t0) * 1000.0)

        if (i % SAMPLE_EVERY) == 0:
            cpu_samples.append(float(np.sum(psutil.cpu_percent(percpu=True))))
            ram_samples.append(float(proc.memory_info().rss / (1024.0 * 1024.0)))
            tc = read_temp_c()
            if tc is not None:
                temp_samples.append(float(tc))

    infer_ms_mean = float(np.mean(infer_times_ms))
    infer_ms_std  = float(np.std(infer_times_ms))
    infer_ms_p50  = float(np.percentile(infer_times_ms, 50))
    infer_ms_p95  = float(np.percentile(infer_times_ms, 95))
    throughput    = float(1000.0 / infer_ms_mean) if infer_ms_mean > 0 else 0.0

    cpu_mean = float(np.mean(cpu_samples)) if cpu_samples else None
    cpu_peak = float(np.max(cpu_samples)) if cpu_samples else None
    ram_mean = float(np.mean(ram_samples)) if ram_samples else None
    ram_peak = float(np.max(ram_samples)) if ram_samples else None
    temp_mean = float(np.mean(temp_samples)) if temp_samples else None
    temp_peak = float(np.max(temp_samples)) if temp_samples else None

    bench_total_s = float((infer_ms_mean * BENCH_RUNS) / 1000.0)

    result = {
        "model": model_path.name,
        "family": model_family,
        "file_size_mb": float(model_path.stat().st_size / (1024.0 * 1024.0)),
        "file_sha1": sha1_file(model_path),

        "tflite_runtime_version": TFLITE_RT_VER,
        "tensorflow_version": TF_VER,
        "backend_used": backend_used,

        "dataset": {
            "path": str(dataset_dir),
            "num_images": int(len(paths)),
            "classes": class_names,
            "class_order_source": class_order_source,
        },

        "settings": {
            "warmup_runs": int(WARMUP_RUNS),
            "bench_runs": int(BENCH_RUNS),
            "load_reps": int(LOAD_REPS),
            "sample_every": int(SAMPLE_EVERY),
            "num_threads": int(NUM_THREADS),
            "float_preprocess_mode": str(FLOAT_PREPROCESS_MODE),
            "seed": int(SEED),
        },

        "load_time_mean_s": float(load_mean),
        "load_time_std_s": float(load_std),

        "infer_ms_mean": float(infer_ms_mean),
        "infer_ms_std": float(infer_ms_std),
        "infer_ms_p50": float(infer_ms_p50),
        "infer_ms_p95": float(infer_ms_p95),
        "throughput_img_s": float(throughput),

        "cpu_mean": cpu_mean,
        "cpu_peak": cpu_peak,
        "ram_mean_mb": ram_mean,
        "ram_peak_mb": ram_peak,

        "temp_start_c": temp_start,
        "temp_mean_c": temp_mean,
        "temp_peak_c": temp_peak,

        "bench_total_time_s": float(bench_total_s),

        "accuracy": float(acc),
        "macro_f1": float(macro),
        "per_class_report": report,

        "input_shape": input_shape,
        "input_dtype": input_dtype,
        "input_quant": input_quant,
    }

    return result

def main():
    try:
        all_results = {}

        for variant, filename in MODEL_FILES.items():
            model_path = MODEL_DIR / filename

            print("\n======================================")
            print("Running:", MODEL_FAMILY, "| Variant:", variant)
            print("Model file:", model_path)
            print("======================================")

            res = run_benchmark(MODEL_FAMILY, model_path, DATASET_DIR)

            # Save individual JSON
            out_name = f"{MODEL_FAMILY}__{variant}_benchmark.json"
            out_path = OUT_DIR / out_name
            out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")

            print("Saved:", out_path)
            print("Accuracy:", res["accuracy"])
            print("Latency (ms):", res["infer_ms_mean"])
            print("Throughput:", res["throughput_img_s"])

            all_results[variant] = res

        # Optional: Save combined summary
        summary_path = OUT_DIR / f"{MODEL_FAMILY}__summary.json"
        summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

        print("\n✅ ALL DONE for", MODEL_FAMILY)

    except Exception as e:
        print("\nERROR:", str(e))
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()