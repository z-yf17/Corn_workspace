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
- [Pose Estimation](#pose-estimation)
- [Polymetis Control Interface](#polymetis-control-interface)
- [Models and Checkpoints](#models-and-checkpoints)
- [Third-party Components](#third-party-components)
- [License](#license)

---

## Features

### Calibration (Multi-view Extrinsics)
Tools under `calibration/` for labeling cameras and capturing extrinsic calibration samples from multiple views.

### Segmentation
A segmentation stage intended to produce masks / ROIs for downstream pose estimation.

### Pose Estimation
Pose estimation components that consume images (+ optional segmentation output) and produce pose results suitable for robotics tasks.

### Control (Polymetis)
An interface designed to connect perception outputs (e.g., target pose) to robot commands via **Polymetis**.

---

## Repository Layout

```text
.
├── calibration/                 # multi-camera extrinsic calibration tools
├── FoundationPose/              # third-party component (modified)
├── Grounded-SAM-2/              # third-party component (modified)
└── README.md
```

---

## Prerequisites

- Linux recommended
- Python environment matching your local setup
- RealSense cameras connected (for calibration)
- **Polymetis must be running** before capturing calibration samples
- `zmq_publisher.py` must be running to publish robot joint states (required by capture)

---

## Multi-view Extrinsic Calibration (RealSense)

The `calibration/` folder is used for multi-camera extrinsic calibration.

### Notes (important)

- The commands below must be run **inside** the `calibration/` directory.
- Before capturing calibration data you must:
  1. **Start Polymetis**
  2. Run `zmq_publisher.py` to publish robot joint states (required by the capture script)

### Step-by-step Workflow

#### 0) Start Polymetis
Start Polymetis using your local setup (command depends on your environment).

#### 1) Start joint-state publisher (required)
Open a terminal and run:

```bash
cd calibration
python zmq_publisher.py
```

Keep this running while you capture samples.

#### 2) Label RealSense cameras as `left / front / right`
Open a second terminal and run:

```bash
cd calibration
python label_realsense_cams.py
```

This will show three camera streams (one per camera). Use the keyboard controls below to label them.

#### 3) Capture extrinsic calibration samples (per view)
Run the capture script for each view:

```bash
cd calibration
python capture_calib_extrinsic_multi_view.py --cam left
python capture_calib_extrinsic_multi_view.py --cam front
python capture_calib_extrinsic_multi_view.py --cam right
```

**Procedure**
1. The script opens the corresponding camera view.
2. Move the robot so that the calibration board is clearly visible.
3. Press `s` to **save** the current result/sample.

Repeat until you have enough samples for each view.

---

### Keyboard Controls

#### `label_realsense_cams.py`
- Press `1` / `2` / `3` to select the camera view
- Press `l` / `f` / `r` to label the selected view:
  - `l` → `left`
  - `f` → `front`
  - `r` → `right`

#### `capture_calib_extrinsic_multi_view.py --cam {left|front|right}`
- Press `s` to **save** the current sample/result

---

## Segmentation

This repository includes (or is designed to include) an image segmentation stage that produces masks/regions of interest for downstream pose estimation.

```bash
# TODO: add your actual command(s)
# python <your_entrypoint>.py ...
```

---

## Pose Estimation

This repository includes (or is designed to include) a pose estimation stage that consumes images (and optionally segmentation outputs) and produces pose outputs suitable for robotic tasks.

```bash
# TODO: add your actual command(s)
# python <your_entrypoint>.py ...
```

---

## Polymetis Control Interface

This repository provides an interface/bridge to command robots through **Polymetis**, enabling perception outputs (pose/targets) to be used in control loops.

```bash
# TODO: add your actual command(s)
# python <your_entrypoint>.py ...
```

---

## Models and Checkpoints

This repository **does not include** model weights/checkpoints.

- Download weights from official sources of the corresponding models
- Place them under your local weights directory (e.g., `checkpoints/` or `weights/`)
- Ensure your configs/scripts point to the correct paths

---

## Third-party Components

This repository includes (and may modify) third-party open-source projects, such as:

- `FoundationPose/`
- `Grounded-SAM-2/`

Please keep upstream attribution and license files intact within their directories.

---

## License

- Core code in this repository: TODO
- Third-party components: follow their respective upstream licenses

