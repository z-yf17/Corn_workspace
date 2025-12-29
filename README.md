# Corn Workspace

Corn Workspace is a unified robotics codebase that provides a practical **perception → pose → control** pipeline:

- **Multi-view camera extrinsic calibration** (multi-camera setups)
- **Image segmentation**
- **Pose estimation**
- **Polymetis control interface**

> This repository **does not ship model weights/checkpoints**.

---

## Contents

- [Features](#features)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Multi-view Extrinsic Calibration (RealSense)](#multi-view-extrinsic-calibration-realsense)
  - [Step-by-step Workflow](#step-by-step-workflow)
  - [Keyboard Controls](#keyboard-controls)
- [Segmentation](#segmentation)
  - [Grounded-SAM2 (Docker)](#grounded-sam2-docker)
  - [Running Segmentation in the Online Multi-view Pipeline](#running-segmentation-in-the-online-multi-view-pipeline)
- [Pose Estimation](#pose-estimation)
  - [FoundationPose (Docker)](#foundationpose-docker)
  - [Running Pose Estimation in the Online Multi-view Pipeline](#running-pose-estimation-in-the-online-multi-view-pipeline)
- [End-to-end Online Multi-view Runtime](#end-to-end-online-multi-view-runtime)
- [Policy (Example)](#policy-example)
- [Polymetis Control Interface](#polymetis-control-interface)
- [Models and Checkpoints](#models-and-checkpoints)
- [Third-party Components](#third-party-components)
- [License](#license)

---

## Features

### Calibration (Multi-view Extrinsics)
Tools under `calibration/` for labeling cameras and capturing extrinsic calibration samples from multiple views.

### Segmentation
Segmentation is implemented using **Grounded-SAM2**. The segmentation service subscribes to multi-view images, performs segmentation, and publishes results for downstream modules.

### Pose Estimation
Pose estimation is implemented using **FoundationPose**. The pose estimator consumes the (segmented) multi-view stream, estimates pose, and publishes results for downstream policy/control.

### Control (Polymetis)
A control interface designed to connect perception outputs (e.g., target pose) to robot commands via **Polymetis**.

---

## Repository Layout

```text
.
├── calibration/                 # multi-camera extrinsic calibration tools
├── FoundationPose/              # third-party component (modified)
├── Grounded-SAM-2/              # third-party component (modified)
├── naive_policy.py              # example policy (replace with your own)
└── README.md

