# Edge-Optimized CNN Deployment for Waste Classification  
### A Hardware-Aware Evaluation of Quantization Trade-offs on Raspberry Pi 4B

This repository contains the implementation, trained models, TensorFlow Lite conversions, benchmarking outputs, and visualization assets for a hardware-aware study of CNN deployment for waste classification on resource-constrained edge devices.

The project systematically evaluates how different CNN architectures behave under multiple post-training quantization strategies when deployed on a Raspberry Pi 4B. In addition to predictive performance, the study considers deployment-critical system metrics such as throughput, load time, CPU usage, RAM usage, and temperature.

---

## Overview

High cloud-stage accuracy does not always translate into practical edge performance. This repository supports a deployment-oriented benchmarking framework that evaluates six CNN architectures under four TensorFlow Lite formats:

- **CNN architectures:** EfficientNetB0, InceptionV3, MobileNetV1, MobileNetV3Small, NASNetMobile, ResNet50  
- **Quantization settings:** FP32, FP16, DRQ, INT8  
- **Target hardware:** Raspberry Pi 4B  

The main goal is to study the interaction between **network architecture** and **quantization strategy** under real hardware constraints.

---

## Overall Pipeline Figure

> Replace the placeholder below with your final system or workflow figure.

<p align="center">
<img width="250" height="" alt="quantization" src="https://github.com/user-attachments/assets/0db4b37b-0ada-44cf-b2bc-ea6cee0faf1f" />
</p>

**Suggested figure:** training-to-deployment pipeline or complete benchmarking workflow.

---

## Video Demonstration

> Add your YouTube demo link below.

**YouTube Demo:**  
`[PASTE_YOUR_YOUTUBE_LINK_HERE](https://youtu.be/LS8pOz9uuPw?si=7KCP6rmZhyfTJJfQ)`

---

## Main Contributions

- A **hardware-aware benchmarking framework** for waste classification on edge devices  
- A comparative evaluation of **six CNN architectures** across **four TensorFlow Lite deployment formats**  
- Real-device benchmarking on **Raspberry Pi 4B** using both predictive and system-level metrics  
- An **Edge Deployment Score (EDS)** framework for integrated deployment-oriented evaluation  
- Validation using a dedicated **edge-collected waste dataset** reflecting practical operating conditions  

---

## Repository Structure

```text
edge-waste-cnn-quantization/
├── Edge Info
├── EdgeResult/
│   ├── EfficientNetB0/
│   ├── InceptionV3/
│   ├── MobileNetV1/
│   ├── MobileNetV3Small/
│   ├── NASNetMobile/
│   ├── Resnet50/
│   └── run.py
├── EdgeTFlite/
│   ├── EfficientNetB0/
│   ├── InceptionV3/
│   ├── MobileNetV1/
│   ├── MobileNetV3Small/
│   ├── NASNetMobile/
│   ├── Resnet50/
│   ├── converter.py
│   └── convert2.py
├── ModelsInColab/
│   ├── keras/
│   ├── Performance/
│   ├── Traning/
│   └── run_bench.py
├── Plots/
│   ├── Edge/
│   ├── FinalAccuracyColab.png
│   └── plots.ipynb
├── LICENSE
└── README.md
```

Folder Description

ModelsInColab/
	•	keras/ → trained models and metrics
	•	Performance/ → training curves & confusion matrices
	•	Traning/ → training notebooks
	•	run_bench.py → benchmarking script

EdgeTFlite/
	•	Converted TFLite models (FP32, FP16, DRQ, INT8)
	•	Conversion scripts (converter.py, etc.)

EdgeResult/
	•	Benchmark outputs (JSON)
	•	Model-wise summaries

Plots/
	•	Final figures used in paper
	•	Visualization notebooks

⸻

Evaluated Models
	•	EfficientNetB0
	•	InceptionV3
	•	MobileNetV1
	•	MobileNetV3Small
	•	NASNetMobile
	•	ResNet50

⸻

Quantization Variants
	•	FP32
	•	FP16
	•	DRQ
	•	INT8

⸻

Benchmarking Metrics

Predictive
	•	Accuracy
	•	Precision
	•	Recall
	•	F1-score

Deployment
	•	Throughput
	•	Load time
	•	CPU usage
	•	RAM usage
	•	Temperature
	•	Model size

⸻

Edge Deployment Score (EDS)

EDS integrates predictive performance and hardware efficiency:
	•	Accuracy, F1-score
	•	Throughput
	•	Model size
	•	Load time
	•	CPU usage
	•	RAM usage
	•	Temperature

Two variants:
	•	Equal-weight
	•	Weighted (deployment-aware)

⸻

Datasets

Public Dataset

Garbage Classification v2
https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2

Edge Dataset

DATASet_WasteClass
https://github.com/saikatdas47/DATASet_WasteClass

⸻

Reproducibility

Steps:
	1.	Train models (ModelsInColab/Traning/)
	2.	Convert to TFLite (EdgeTFlite/)
	3.	Deploy on Raspberry Pi
	4.	Run benchmarking (EdgeResult/)
	5.	Generate plots (Plots/)

⸻

Example Commands
python ModelsInColab/run_bench.py

Key Findings
	•	Cloud accuracy ≠ deployment performance
	•	FP16 preserves accuracy
	•	INT8 improves speed but varies by architecture
	•	MobileNetV3Small shows strong robustness
	•	Deep models degrade under edge constraints
	•	Deployment requires hardware-aware evaluation

⸻

Hardware
	•	Raspberry Pi 4B
	•	8GB RAM
	•	ARM Cortex-A72
	•	CPU inference

⸻

Software
	•	TensorFlow / Keras
	•	TensorFlow Lite
	•	Python
Notes
	•	CPU-based edge evaluation
	•	Results may vary across hardware
	•	Experimental structure preserved
## EDS Heatmap

<p align="center">
  <img src="https://github.com/user-attachments/assets/a3ffeb9f-1edf-49ab-a070-d881520f4c50" width="45%" />
  <img src="https://github.com/user-attachments/assets/31b6790a-95d1-4482-81b7-0c30f43cd01a" width="45%" />
</p>

<p align="center">
  <b>Figure:</b> Edge Deployment Score (EDS) heatmaps comparing CNN architectures across quantization strategies (equal-weight and weighted formulations).
</p>
