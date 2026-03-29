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
MODEL_FILES = {
    "fp32": "fp32.tflite",
    "fp16": "fp16.tflite",
    "drq": "drq.tflite",
    "int8_full_int8io": "int8_full_int8io.tflite",
    #"int8_hybrid_floatio": "int8_hybrid_floatio.tflite",
}

MODEL_DIR   = Path("./NASNetMobile")   # folder that contains above files
DATASET_DIR = Path("./data2")             # class subfolders

OUT_DIR = Path("./bench_out/NASNetMobile/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_RUNS  = 30
BENCH_RUNS   = 300
LOAD_REPS    = 5
SAMPLE_EVERY = 10
NUM_THREADS  = 4
SEED         = 42

# MUST match training preprocess AND converter REP_NORM_MODE
# "minus1_1" | "0_1" | "raw_255"
FLOAT_PREPROCESS_MODE = "raw_255"

# Class order lock (use the same classes.json you exported)
CLASSES_JSON = Path("./classes.json")
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

def to_jsonable(x):
    import numpy as np
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, (np.ndarray,)): return x.tolist()
    if isinstance(x, dict): return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [to_jsonable(v) for v in x]
    return x

def load_classes(dataset_dir: Path, classes_json: Path):
    dataset_dir = Path(dataset_dir)
    if classes_json and classes_json.exists():
        classes = json.loads(classes_json.read_text(encoding="utf-8"))
        if not isinstance(classes, list) or not all(isinstance(x, str) for x in classes):
            raise ValueError("classes.json must be a JSON list of class names")
        src = "classes.json"
    else:
        classes = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
        src = "folder_sorted"

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

    if not img_paths:
        raise RuntimeError(f"No images found under: {dataset_dir}")

    img_paths = np.array(img_paths)
    labels = np.array(labels)

    print("\n[Dataset]")
    print("Path:", dataset_dir)
    print("Classes:", classes, "| source:", src)
    print("Total images:", len(img_paths))
    for i, c in enumerate(classes):
        print(f"  {c}: {int(np.sum(labels == i))}")

    return img_paths, labels, classes, src

def preprocess_float(img_rgb_uint8: np.ndarray, mode: str) -> np.ndarray:
    x = img_rgb_uint8.astype(np.float32)
    mode = (mode or "minus1_1").lower().strip()

    if mode == "minus1_1":
        return ((x / 127.5) - 1.0).astype(np.float32)
    if mode == "0_1":
        return (x / 255.0).astype(np.float32)
    if mode == "raw_255":
        return x.astype(np.float32)

    raise ValueError("FLOAT_PREPROCESS_MODE must be: minus1_1 | 0_1 | raw_255")

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
    out = np.array(output).reshape(-1)
    return int(np.argmax(out))

def load_image_as_input(path: str, in_details, preprocess_mode: str):
    # read model input shape
    shape = in_details["shape"]  # [1,H,W,3]
    H, W = int(shape[1]), int(shape[2])

    img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    rgb = np.array(img, dtype=np.uint8)

    in_dtype = in_details["dtype"]
    scale, zero_point = in_details.get("quantization", (0.0, 0))
    scale = float(scale)
    zero_point = int(zero_point)

    # float input
    if in_dtype == np.float32:
        x = preprocess_float(rgb, preprocess_mode)
        return np.expand_dims(x, 0).astype(np.float32)

    # int8/uint8 input: preprocess float THEN quantize using scale/zp
    x_f = preprocess_float(rgb, preprocess_mode)  # float32
    if scale > 0:
        x_q = np.round((x_f / scale) + zero_point)
    else:
        x_q = np.round(x_f + zero_point)

    if in_dtype == np.uint8:
        x_q = np.clip(x_q, 0, 255).astype(np.uint8)
    elif in_dtype == np.int8:
        x_q = np.clip(x_q, -128, 127).astype(np.int8)
    else:
        x_q = x_q.astype(in_dtype)

    return np.expand_dims(x_q, 0)

def run_benchmark(model_path: Path):
    paths, labels, class_names, class_src = load_classes(DATASET_DIR, CLASSES_JSON)

    load_mean, load_std, backend_used = measure_load_time(model_path, NUM_THREADS, LOAD_REPS)

    interp, backend_used = make_interpreter(model_path, NUM_THREADS)
    interp.allocate_tensors()

    in_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]

    proc = psutil.Process(os.getpid())
    temp_start = read_temp_c()

    # -------------------------
    # Accuracy (full dataset)
    # -------------------------
    y_true = labels.tolist()
    y_pred = []

    for p in paths:
        x = load_image_as_input(p, in_details, FLOAT_PREPROCESS_MODE)
        interp.set_tensor(in_details["index"], x)
        interp.invoke()
        out = interp.get_tensor(out_details["index"])
        y_pred.append(safe_argmax(out))

    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro"))
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # -------------------------
    # Latency benchmark
    # -------------------------
    idxs = np.random.randint(0, len(paths), size=(WARMUP_RUNS + BENCH_RUNS,))
    sample_paths = paths[idxs]

    for i in range(WARMUP_RUNS):
        x = load_image_as_input(sample_paths[i], in_details, FLOAT_PREPROCESS_MODE)
        interp.set_tensor(in_details["index"], x)
        interp.invoke()

    infer_times_ms = []
    cpu_samples, ram_samples, temp_samples = [], [], []
    _ = psutil.cpu_percent(percpu=True)

    for i in range(BENCH_RUNS):
        x = load_image_as_input(sample_paths[WARMUP_RUNS + i], in_details, FLOAT_PREPROCESS_MODE)

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

    result = {
        "model": model_path.name,
        "file_size_mb": float(model_path.stat().st_size / (1024.0 * 1024.0)),
        "file_sha1": sha1_file(model_path),

        "tflite_runtime_version": TFLITE_RT_VER,
        "tensorflow_version": TF_VER,
        "backend_used": backend_used,

        "dataset": {
            "path": str(DATASET_DIR),
            "num_images": int(len(paths)),
            "classes": class_names,
            "class_order_source": class_src,
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

        "infer_ms_mean": infer_ms_mean,
        "infer_ms_std": infer_ms_std,
        "infer_ms_p50": infer_ms_p50,
        "infer_ms_p95": infer_ms_p95,
        "throughput_img_s": throughput,

        "cpu_mean": float(np.mean(cpu_samples)) if cpu_samples else None,
        "cpu_peak": float(np.max(cpu_samples)) if cpu_samples else None,
        "ram_mean_mb": float(np.mean(ram_samples)) if ram_samples else None,
        "ram_peak_mb": float(np.max(ram_samples)) if ram_samples else None,

        "temp_start_c": temp_start,
        "temp_mean_c": float(np.mean(temp_samples)) if temp_samples else None,
        "temp_peak_c": float(np.max(temp_samples)) if temp_samples else None,

        "accuracy": acc,
        "macro_f1": macro,
        "per_class_report": report,

        "input_shape": [int(x) for x in in_details["shape"]],
        "input_dtype": str(in_details["dtype"]),
        "input_quant": {
            "scale": float(in_details.get("quantization", (0.0, 0))[0]),
            "zero_point": int(in_details.get("quantization", (0.0, 0))[1]),
        },
    }

    return result

def main():
    try:
        all_results = {}
        for variant, filename in MODEL_FILES.items():
            model_path = MODEL_DIR / filename
            if not model_path.exists():
                print(f"Skip (missing): {model_path}")
                continue

            print("\n======================================")
            print("Variant:", variant)
            print("Model  :", model_path)
            print("======================================")

            res = run_benchmark(model_path)

            out_path = OUT_DIR / f"{variant}_benchmark.json"
            out_path.write_text(json.dumps(to_jsonable(res), indent=2), encoding="utf-8")
            print("Saved:", out_path)
            print("Accuracy:", res["accuracy"], "| MacroF1:", res["macro_f1"])
            print("Latency(ms):", res["infer_ms_mean"], "| Throughput:", res["throughput_img_s"])

            all_results[variant] = res

        summary_path = OUT_DIR / "summary.json"
        summary_path.write_text(json.dumps(to_jsonable(all_results), indent=2), encoding="utf-8")
        print("\n✅ ALL DONE")

    except Exception as e:
        print("\nERROR:", str(e))
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()