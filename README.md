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
- RealSense cameras connected (for calibration and online pipeline)
- Docker installed (required for segmentation and pose estimation runtimes described below)
- **Polymetis must be running** before capturing calibration samples
- `zmq_publisher.py` must be running to publish robot joint states (required by calibration capture)

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

### Grounded-SAM2 (Docker)

Segmentation is implemented using **Grounded-SAM2**.

**Enter the Grounded-SAM2 segmentation Docker container:**
```bash
docker start -ai Corn_docker
```

Inside the container, activate the conda environment and run the multi-view segmentation subscriber/publisher:
```bash
conda activate sam
python docker_video_sub_pub_tracking_multi_view.py
```

This process:
- **subscribes** to a published multi-view image stream
- performs segmentation (Grounded-SAM2)
- **publishes** the segmented results for downstream consumers (e.g., pose estimation)

### Running Segmentation in the Online Multi-view Pipeline

In an online pipeline, segmentation expects that a multi-view image publisher is already running (see [End-to-end Online Multi-view Runtime](#end-to-end-online-multi-view-runtime)).

---

## Pose Estimation

### FoundationPose (Docker)

Pose estimation is implemented using **FoundationPose**.

**Enter the FoundationPose Docker container:**
```bash
cd /home/galbot/ros_noetic_docker/FoundationPose
bash docker/run_container.sh
```

Once inside the container, run the multi-view pose estimator:
```bash
python realtime_multi_view_filter.py
```

This process:
- consumes the (segmented) multi-view stream
- estimates pose
- publishes pose outputs for downstream policy/control

### Running Pose Estimation in the Online Multi-view Pipeline

Pose estimation should be started **after**:
1) the multi-view camera publisher is running, and  
2) the segmentation service is running and publishing segmented outputs.

See the end-to-end runtime section below.

---

## End-to-end Online Multi-view Runtime

Below is the recommended run order for the online multi-view pipeline (multiple terminals).

### Terminal 1 — Start multi-view camera publisher
Start the multi-view camera publisher:
```bash
python video_publisher_multi_view.py
```

### Terminal 2 — Start segmentation (Grounded-SAM2 in Docker)
```bash
docker start -ai Corn_docker
conda activate sam
python docker_video_sub_pub_tracking_multi_view.py
```

### Terminal 3 — Start pose estimation (FoundationPose in Docker)
```bash
cd /home/galbot/ros_noetic_docker/FoundationPose
bash docker/run_container.sh
python realtime_multi_view_filter.py
```

After these are running:
- the camera publisher provides multi-view images
- Grounded-SAM2 subscribes, segments, and republishes results
- FoundationPose subscribes, estimates pose, and republishes pose outputs for the policy

---

## Policy (Example)

An example policy is provided for demonstration and testing.

```bash
python naive_policy.py
```

> This example policy is intended to consume the published perception outputs (e.g., pose) and produce actions/commands for the robot.

---

## Polymetis Control Interface

This repository provides an interface/bridge to command robots through **Polymetis**, enabling perception outputs (pose/targets) to be used in control loops.

```bash
# TODO: add your actual command(s) for policy/control if different from naive_policy.py
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

