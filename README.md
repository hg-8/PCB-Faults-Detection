# 🔍 PCB Defect Detection with YOLOv5 & YOLOv8

Automated detection and classification of Printed Circuit Board (PCB) surface defects using deep learning object detection models. This project compares **YOLOv5s** and **YOLOv8s** architectures on the **HRIPCB** dataset to evaluate their effectiveness in identifying six common types of PCB manufacturing defects.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Defect Classes](#defect-classes)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
- [Training Configuration](#training-configuration)
- [Performance Comparison](#performance-comparison)
- [Severity Analysis System](#severity-analysis-system)
- [Sample Predictions](#sample-predictions)
- [Getting Started](#getting-started)
- [License](#license)

---

## Overview

Quality control in PCB manufacturing is critical — undetected defects can lead to device failure, safety hazards, or costly recalls. This project leverages state-of-the-art YOLO (You Only Look Once) object detection models to automatically identify and localize defects on PCB surfaces, enabling faster and more reliable inspection than manual visual checks.

Both **YOLOv5s** and **YOLOv8s** (small variants) were trained and evaluated under comparable conditions, allowing a direct performance comparison.

---

## Dataset

| Property | Detail |
|---|---|
| **Name** | HRIPCB (High-Resolution Images of PCB) |
| **Total Images** | 693 |
| **Image Size** | Resized to 640×640 pixels |
| **Annotation Format** | Pascal VOC XML → converted to YOLO format |
| **Defect Classes** | 6 |

### Per-Class Image Distribution

| Defect Type | Images | Annotations |
|---|:---:|:---:|
| Missing Hole | 115 | 115 |
| Mouse Bite | 115 | 115 |
| Open Circuit | 116 | 116 |
| Short | 116 | 116 |
| Spur | 115 | 115 |
| Spurious Copper | 116 | 116 |

---

## Defect Classes

| # | Class | Description |
|:---:|---|---|
| 0 | **Missing Hole** | Absent drill hole where a component via or through-hole should exist |
| 1 | **Mouse Bite** | Irregular edge left from PCB depanelization (tab routing) |
| 2 | **Open Circuit** | Broken or interrupted trace causing electrical path failure |
| 3 | **Short** | Unintended copper bridge creating unwanted electrical connection |
| 4 | **Spur** | Small protrusion extending beyond intended trace boundary |
| 5 | **Spurious Copper** | Unwanted copper residue/deposit on the board surface |

---

## Project Structure

```
github_push/
├── README.md
├── yolov5/
│   ├── pcb-defects-detected-by-yolo-v5.ipynb   # Full training & inference notebook
│   ├── testing/                                 # Sample test inference results
│   │   ├── 01_missing_hole_05.jpg
│   │   ├── 01_mouse_bite_02.jpg
│   │   ├── 01_open_circuit_04.jpg
│   │   └── 01_spurious_copper_02.jpg
│   └── yolov5_results/
│       └── pcb_experiment/
│           ├── results.csv                      # Training metrics per epoch
│           ├── results.png                      # Training curves plot
│           ├── confusion_matrix.png             # Confusion matrix
│           ├── F1_curve.png                     # F1 vs confidence curve
│           ├── PR_curve.png                     # Precision-Recall curve
│           ├── P_curve.png                      # Precision vs confidence
│           ├── R_curve.png                      # Recall vs confidence
│           ├── labels.jpg                       # Dataset label distribution
│           ├── labels_correlogram.jpg           # Label correlogram
│           ├── hyp.yaml                         # Hyperparameters used
│           ├── opt.yaml                         # Training options/config
│           ├── train_batch*.jpg                 # Sample training batches
│           └── val_batch*_pred.jpg              # Validation predictions
│
└── yolov8/
    ├── pcb-defect-detection-with-yolov8.ipynb   # Full training & inference notebook
    ├── testing/                                 # Sample test inference results (34 images)
    │   ├── 04_missing_hole_04.jpg
    │   ├── 04_mouse_bite_05.jpg
    │   ├── ...
    │   └── 12_spur_03.jpg
    └── yolov8_results/
        ├── results.csv                          # Training metrics per epoch
        ├── results.png                          # Training curves plot
        ├── confusion_matrix.png                 # Confusion matrix
        ├── confusion_matrix_normalized.png      # Normalized confusion matrix
        ├── BoxF1_curve.png                      # Box F1 vs confidence
        ├── BoxPR_curve.png                      # Box Precision-Recall
        ├── BoxP_curve.png                       # Box Precision vs confidence
        ├── BoxR_curve.png                       # Box Recall vs confidence
        ├── labels.jpg                           # Dataset label distribution
        ├── train_batch*.jpg                     # Sample training batches
        └── val_batch*_pred.jpg                  # Validation predictions
```

---

## Model Architectures

| Feature | YOLOv5s | YOLOv8s |
|---|---|---|
| **Framework** | Ultralytics YOLOv5 (PyTorch) | Ultralytics YOLOv8 (PyTorch) |
| **Backbone** | CSPDarknet53 | CSPDarknet (modified) |
| **Neck** | PANet (FPN + PAN) | C2f modules + BiFPN-style |
| **Head** | Coupled (anchor-based) | Decoupled (anchor-free) |
| **Loss** | CIoU + BCE (obj + cls) | CIoU + DFL + BCE (cls) |
| **Pretrained Weights** | `yolov5s.pt` (COCO) | `yolov8s.pt` (COCO) |

---

## Training Configuration

| Parameter | YOLOv5s | YOLOv8s |
|---|:---:|:---:|
| **Epochs** | 100 | 180 |
| **Batch Size** | 16 | 16 |
| **Image Size** | 640×640 | 640×640 |
| **Optimizer** | SGD | AdamW (default) |
| **Initial LR** | 0.01 | 0.001 |
| **Final LR** | 0.01 | 0.0001 |
| **Momentum** | 0.937 | 0.937 (default) |
| **Weight Decay** | 0.0005 | 0.0005 (default) |
| **Warmup Epochs** | 3.0 | 3.0 (default) |
| **Mosaic** | 1.0 | 1.0 (default) |
| **Mixup** | 0.0 | 0.3 |
| **Train/Val Split** | 80/20 | 95/5 (3-fold cross-val) |
| **Cross-Validation** | No | Yes (K=3) |
| **Platform** | Google Colab (GPU) | Google Colab (GPU) |

---

## Performance Comparison

### 🏆 Best Epoch Metrics (mAP@0.5 optimized)

| Metric | YOLOv5s | YOLOv8s | Δ (v5 − v8) |
|---|:---:|:---:|:---:|
| **Best Epoch** | 83 / 100 | 152 / 180 | — |
| **Precision** | **0.9734** | 0.9679 | +0.55% |
| **Recall** | **0.9520** | 0.9071 | +4.49% |
| **mAP@0.5** | **0.9744** | 0.9568 | +1.76% |
| **mAP@0.5:0.95** | **0.5038** | 0.4586 | +4.52% |

### 📊 Best Epoch Metrics (mAP@0.5:0.95 optimized)

| Metric | YOLOv5s | YOLOv8s | Δ (v5 − v8) |
|---|:---:|:---:|:---:|
| **Best Epoch** | 96 / 100 | 73 / 180 | — |
| **Precision** | **0.9736** | 0.9548 | +1.88% |
| **Recall** | **0.9551** | 0.8959 | +5.92% |
| **mAP@0.5** | **0.9732** | 0.9479 | +2.53% |
| **mAP@0.5:0.95** | **0.5206** | 0.4835 | +3.71% |

### 📈 Final Epoch Metrics

| Metric | YOLOv5s (Epoch 99) | YOLOv8s (Epoch 173) | Δ (v5 − v8) |
|---|:---:|:---:|:---:|
| **Precision** | **0.9754** | 0.9656 | +0.98% |
| **Recall** | **0.9529** | 0.9181 | +3.48% |
| **mAP@0.5** | **0.9710** | 0.9434 | +2.76% |
| **mAP@0.5:0.95** | **0.5175** | 0.4566 | +6.09% |

### Key Takeaways

- ✅ **YOLOv5s outperforms YOLOv8s** on this dataset across all primary metrics (Precision, Recall, mAP@0.5, mAP@0.5:0.95).
- ✅ **YOLOv5s converges faster** — achieves best results at epoch 83–96 out of 100, whereas YOLOv8s peaks at epoch 73–152 out of 180.
- ✅ The **recall gap (~4–6%)** indicates YOLOv5s detects a higher proportion of actual defects, which is critical in quality inspection where missed defects are costly.
- ✅ The **mAP@0.5:0.95 gap (~4–6%)** shows YOLOv5s produces tighter bounding boxes across all IoU thresholds.
- ⚠️ **Training differences** — YOLOv5s used an 80/20 train/val split (no cross-validation), while YOLOv8s used a 95/5 split with 3-fold cross-validation and mixup augmentation (0.3). These methodological differences should be considered when interpreting results.
- ⚠️ YOLOv8s was trained for nearly **2× more epochs** (180 vs 100), yet did not surpass YOLOv5s performance.

---

## Severity Analysis System

The YOLOv5 notebook includes an integrated **4-level severity classification system** for detected defects:

| Level | Icon | Description |
|---|:---:|---|
| **CRITICAL** | 🔴 | Board must be scrapped immediately |
| **HIGH** | 🟠 | Significant defect, likely causes malfunction |
| **MODERATE** | 🟡 | May cause issues under stress or over time |
| **LOW** | 🟢 | Minor defect, unlikely to affect function |

Each detection is dynamically assessed based on:
- **Defect area** relative to board size
- **Position** on the board (edge vs center)
- **Detection confidence**

The system provides:
- Per-defect severity with corrective solutions
- Board-level **PASS/FAIL** verdict
- Safety risk evaluation and ship/repair decision

---

## Sample Predictions

### YOLOv5 Test Results
The `yolov5/testing/` folder contains inference results on unseen test images, demonstrating detections for missing holes, mouse bites, open circuits, and spurious copper defects.

### YOLOv8 Test Results
The `yolov8/testing/` folder contains 34 inference results across all 6 defect classes from multiple PCB board templates.

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch ≥ 2.0 with CUDA support
- Google Colab (recommended) or local GPU

### Running the Notebooks

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/pcb-defect-detection.git
   cd pcb-defect-detection
   ```

2. **YOLOv5 Training & Inference:**
   - Open `yolov5/pcb-defects-detected-by-yolo-v5.ipynb` in Google Colab
   - The notebook handles: dataset download, preprocessing, XML→YOLO conversion, training, inference, and severity analysis

3. **YOLOv8 Training & Inference:**
   - Open `yolov8/pcb-defect-detection-with-yolov8.ipynb` in Google Colab
   - The notebook handles: dataset loading, annotation parsing, image preprocessing, K-fold cross-validation, training, and inference visualization

### Dependencies

```bash
# YOLOv5
pip install -r yolov5/requirements.txt
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# YOLOv8
pip install ultralytics
```

---

## License

This project is for educational and research purposes. The HRIPCB dataset is used under its respective license terms.
