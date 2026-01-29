![Drive Logo](assets/Drive-Logo-White.png)

# BeamNG.ai — Realtime Drivable Area Perception

BeamNG.ai is a research-oriented project that focuses on **realtime semantic perception for autonomous driving** using the BeamNG.tech simulator.  
Instead of classical lane-line detection, the system learns to identify the **entire drivable area** using multi-class semantic segmentation.

The long-term goal is to develop a perception pipeline that is:
- realtime-capable
- robust to markings and visual noise
- transferable from simulation to real-world scenarios

---

## Key Idea

Traditional lane detection often fails in complex environments (intersections, worn markings, construction zones).  
This project therefore focuses on **semantic understanding of the road scene**, allowing the vehicle to reason about *where it can drive*, not just *where lines are*.

The system is built around:
- semantic segmentation
- explicit handling of road markings
- realtime inference constraints

---

## Simulator

- **BeamNG.tech**
- Python interface via `beamngpy`
- Front-facing RGB camera
- Semantic annotation images used for labeling

---

## Dataset Generation

Data is generated directly inside the simulator using a custom interactive labeling pipeline.

Each frame consists of:
- RGB image
- multi-class semantic label map
- optional binary drivable mask
- colorized label visualization for debugging

### Folder Structure

```
data/
├── raw/           # RGB camera images
├── labels/        # Multi-class labels (PNG, values 0..6)
├── labels_vis/    # Colorized label previews
└── masks/         # Optional binary drivable masks
```

---

## Label Definition

| Class ID | Name       | Description |
|--------:|------------|-------------|
| 0 | OTHER     | Undefined / background |
| 1 | ROAD      | Drivable road surface |
| 2 | SHOULDER  | Road shoulder |
| 3 | SIDEWALK  | Sidewalk / pedestrian area |
| 4 | TERRAIN   | Grass, dirt, off-road terrain |
| 5 | OBSTACLE  | Buildings, walls, vehicles, barriers |
| 6 | MARKING   | Road markings (lane lines, arrows, text, crosswalks) |

**Important:**  
Road markings are explicitly separated from the road surface to avoid false drivable signals during control.

---

## Dataset Status

- ~19,000 labeled frames
- Multi-class semantic labels
- Realistic class distribution
- Collected across different road situations and intersections

---

## Interactive Labeling

Labels are created by clicking on semantic annotation pixels inside BeamNG.tech.

Features:
- Class-based color collection
- Priority-based label assignment
- Visual label preview
- Optional drivable-mask derivation for controller testing

The labeling tool is located in:

```
src/env/
```

---

## Current State

- Dataset generation complete
- Multi-class semantic labeling implemented
- Realtime camera pipeline working in BeamNG.tech
- Road markings correctly separated from road surface

---

## Planned Next Steps

- Train a **lightweight realtime segmentation model** (e.g. ENet / BiSeNet)
- Export model to ONNX for fast inference
- Integrate inference back into BeamNG for closed-loop control
- Evaluate latency, stability, and robustness

---

## Realtime Focus

The project explicitly targets **realtime inference**, with future deployment in:
- simulation
- research prototypes
- potential real-world robotics platforms

---

## Attribution

This project uses **BeamNG.tech** for simulation and data generation.  
All trademarks and assets belong to their respective owners.

---

## Disclaimer

This project is intended for **research and educational purposes only**.
