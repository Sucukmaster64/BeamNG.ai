# BeamNG.ai – Realtime Drivable Area Segmentation

<p align="center">
  <img src="assets/beamng_tech_logo.png" alt="BeamNG.tech Logo" width="420"/>
</p>

## Overview
BeamNG.ai is a research and development project focused on **real‑time drivable area segmentation** from monocular camera images and the direct use of this information for autonomous vehicle control.

The project is built on **BeamNG.tech** as a high‑fidelity, physics‑based simulation environment and combines:
- semantic segmentation
- realistic vehicle dynamics
- real‑time inference

The explicit goal is **real‑time performance** with **potential transferability to real vehicles**.

---

## Motivation
Many autonomous driving approaches rely on lane detection or high‑definition maps. This project intentionally follows a different paradigm:

> The vehicle learns **which areas are drivable**, instead of explicitly following lane markings.

This approach is more robust in scenarios such as:
- missing or degraded lane markings
- construction zones
- complex intersections
- visually ambiguous road layouts

---

## Dataset

### Source
- Generated in **BeamNG.tech**
- Front‑facing RGB camera
- Semantic annotation images

### Size
- **47,776 images**
- Resolution: **288 × 512**
- Train / Validation split: **80 % / 20 %**

### Classes (Multi‑Class Segmentation)
| ID | Class      | Description |
|----|-----------|-------------|
| 0  | OTHER     | Background / irrelevant |
| 1  | ROAD      | Drivable road surface |
| 2  | SHOULDER  | Road shoulder |
| 3  | SIDEWALK  | Sidewalk |
| 4  | TERRAIN   | Grass / dirt |
| 5  | OBSTACLE  | Buildings, fences, static obstacles |
| 6  | MARKING   | Lane markings, arrows, crosswalks |

MARKING is **explicitly separated** from ROAD to prevent false positives (e.g. arrows or symbols being interpreted as drivable area).

---

## Model

### Architecture
- **ENet** (Efficient Neural Network for real‑time semantic segmentation)
- Designed for low latency and embedded systems

### Training Setup
- Optimizer: AdamW
- Scheduler: OneCycleLR
- Mixed Precision Training (AMP)
- Class‑weighted Cross‑Entropy Loss

### Training Results (30 epochs)
- **Best mIoU:** ~0.66
- Very high ROAD IoU (>0.9)
- Stable convergence without overfitting

The model is optimized for **drivable area reliability**, not for pixel‑perfect visual segmentation.

---

## Real‑Time Objective

The model is designed to:
- run in real time (>20 FPS)
- directly support vehicle control (steering / throttle)
- serve as a foundation for real‑world deployment

The next development step is **live inference and closed‑loop vehicle control inside BeamNG.tech**.

---

## Project Structure

```
BeamNG.ai/
├── assets/            # Logos & media
├── data/
│   ├── raw/           # RGB images
│   ├── labels/        # Multi‑class labels (0–6)
│   ├── masks/         # Binary drivable masks (optional)
│   └── labels_vis/    # Visualization outputs
├── src/
│   ├── env/           # BeamNG interface & data collection
│   └── ml/
│       ├── datasets/  # PyTorch datasets & loaders
│       ├── models/    # ENet and additional models
│       └── train_seg.py
├── runs/              # TensorBoard logs & checkpoints
└── README.md
```

---

## Current Status

- Data collection: **completed**
- Segmentation model: **trained & validated**
- Real‑time inference: **in progress**

---

## Roadmap

Planned next steps:
1. Real‑time inference pipeline (`infer_seg.py`)
2. Drivable mask → target point → vehicle control
3. FPS and latency evaluation
4. Optional comparison with Fast‑SCNN / BiSeNet

---

## Disclaimer
This project is a **non‑commercial research and educational project**.

BeamNG.tech is used in accordance with its licensing terms.

---

## Contact
For research collaboration, academic interest, or technical discussion, feel free to get in touch.

