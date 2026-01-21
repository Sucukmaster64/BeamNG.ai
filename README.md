# BeamNG.ai — Real-Time Drivable Area Segmentation for Autonomous Driving

This project builds an end-to-end, real-time autonomous driving pipeline in simulation using BeamNG.tech.
Instead of classical lane-line detection, the system estimates the *drivable area* (pixel-wise segmentation)
and derives steering/throttle commands from that output.

The main goal is robustness: the vehicle should drive even when road markings are missing, unreliable,
or visually degraded.

## Key Idea

**Input (camera frame) → Drivable area segmentation → Target point → Steering/Throttle → Vehicle control**

The ML model predicts a probability map of “drivable” vs “non-drivable”. A lightweight controller then:
1) analyzes the lower region of the image (where the road is closest),
2) selects a target point inside the drivable region,
3) converts the lateral error to steering,
4) smooths steering and adjusts throttle for stability.

## Why BeamNG.tech

BeamNG.tech provides a high-fidelity simulation environment with:
- camera sensors and consistent rendering,
- reproducible scenarios and automation,
- vehicle state access for evaluation,
- (optionally) ground-truth data generation and faster dataset creation.

This significantly improves data quality and reduces manual overhead compared to screen-only capture.

## Repository Structure

