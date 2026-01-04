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
  - [Environment & configuration reference (Segmentation)](#environment--configuration-reference-segmentation)
  - [Running Segmentation in the Online Multi-view Pipeline](#running-segmentation-in-the-online-multi-view-pipeline)
- [Pose Estimation](#pose-estimation)
  - [FoundationPose (Docker)](#foundationpose-docker)
  - [Environment & configuration reference (Pose estimation)](#environment--configuration-reference-pose-estimation)
  - [Running Pose Estimation in the Online Multi-view Pipeline](#running-pose-estimation-in-the-online-multi-view-pipeline)
  - [NeRF OBJ Generation (BundleSDF)](#nerf-obj-generation-bundlesdf)
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
Pose estimation is implemented using **FoundationPose**. The pose estimator consumes the (segmented) multi-view stream, estimates pose outputs, and publishes results for downstream policy/control.

### Control (Polymetis)
A control interface designed to connect perception outputs (e.g., target pose) to robot commands via **Polymetis**.

---

## Repository Layout

~~~text
.
├── calibration/                 # multi-camera extrinsic calibration tools
├── FoundationPose/              # third-party component (modified)
├── Grounded-SAM-2/              # third-party component (modified)
├── naive_policy.py              # example policy (replace with your own)
└── README.md
~~~

---

## Prerequisites

- Linux recommended
- Python environment matching your local setup
- RealSense cameras connected (for calibration and online pipeline)
- Docker installed (required for segmentation and pose estimation runtimes described below)

### Polymetis (user-installed)

This project uses **Polymetis** for robot control. **Polymetis is not included in this repository**.

- Please install Polymetis yourself (according to your robot/hardware setup).
- Make sure the Polymetis server is running when you use:
  - calibration capture that depends on robot state publishing
  - policy/control execution that uses `polymetis.RobotInterface`

---

## Multi-view Extrinsic Calibration (RealSense)

The `calibration/` folder is used for multi-camera extrinsic calibration.

### Notes (important)

- The commands below must be run **inside** the `calibration/` directory.
- Before capturing calibration data you must:
  1. **Start Polymetis** (installed by the user)
  2. Run `zmq_publisher.py` to publish robot joint states (required by the capture script)

### Step-by-step Workflow

#### 0) Start Polymetis
Start Polymetis using your local setup (command depends on your environment).

#### 1) Start joint-state publisher (required)
Open a terminal and run:

~~~bash
cd calibration
python zmq_publisher.py
~~~

Keep this running while you capture samples.

#### 2) Label RealSense cameras as `left / front / right`
Open a second terminal and run:

~~~bash
cd calibration
python label_realsense_cams.py
~~~

This will show three camera streams (one per camera). Use the keyboard controls below to label them.

#### 3) Capture extrinsic calibration samples (per view)
Run the capture script for each view:

~~~bash
cd calibration
python capture_calib_extrinsic_multi_view.py --cam left
python capture_calib_extrinsic_multi_view.py --cam front
python capture_calib_extrinsic_multi_view.py --cam right
~~~

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
~~~bash
docker start -ai Corn_docker
~~~

Inside the container, activate the conda environment and run the multi-view segmentation subscriber/publisher:
~~~bash
conda activate sam
python docker_video_sub_pub_tracking_multi_view.py
~~~

This process:
- **subscribes** to a published multi-view image stream
- performs segmentation (Grounded-SAM2)
- **publishes** the segmented results for downstream consumers (e.g., pose estimation)

### Environment & configuration reference (Segmentation)

For Grounded-SAM2 environment setup, model weights, and configuration details, please refer to the upstream project:

- https://github.com/IDEA-Research/Grounded-SAM-2

### Running Segmentation in the Online Multi-view Pipeline

In an online pipeline, segmentation expects that a multi-view image publisher is already running (see [End-to-end Online Multi-view Runtime](#end-to-end-online-multi-view-runtime)).

---

## Pose Estimation

### FoundationPose (Docker)

Pose estimation is implemented using **FoundationPose**.

**Enter the FoundationPose Docker container:**
~~~bash
cd /home/galbot/ros_noetic_docker/FoundationPose
bash docker/run_container.sh
~~~

Once inside the container, run the multi-view pose estimator:
~~~bash
python realtime_multi_view_filter.py
~~~

This process:
- consumes the (segmented) multi-view stream
- estimates pose
- publishes pose outputs for downstream policy/control

### Environment & configuration reference (Pose estimation)

For FoundationPose environment setup, dependencies, and configuration details, please refer to the upstream project commit used as reference:

- https://github.com/NVlabs/FoundationPose/tree/e3d597b8c6b851d053094ebd6fa240191c5238f8

### Running Pose Estimation in the Online Multi-view Pipeline

Pose estimation should be started **after**:
1) the multi-view camera publisher is running, and
2) the segmentation service is running and publishing segmented outputs.

See the end-to-end runtime section below.

### NeRF OBJ Generation (BundleSDF)

You can run NeRF (BundleSDF) to generate an **OBJ** model under `FoundationPose/bundlesdf`.

#### 1) Record an RGBD session
Run the recorder from:
`FoundationPose/bundlesdf`

~~~bash
cd /home/galbot/ros_noetic_docker/FoundationPose/bundlesdf
python record.py
~~~

**Keyboard controls**
- `r`: start recording
- `t`: stop recording
- `q`: quit

**Recording requirements (important)**
- Ensure the **calibration board is fully visible** in the camera view during recording.
- **Only frames where the entire calibration board appears in view are valid.**
- After recording, data will be saved under:
  `FoundationPose/bundlesdf/rs_rgbd_aruco_record/left/`

#### 2) Run NeRF to generate the OBJ
From the same directory:

~~~bash
cd /home/galbot/ros_noetic_docker/FoundationPose/bundlesdf
python run_nerf_rs_session.py
~~~

Notes:
- This step can take **~1 hour** depending on hardware and recording length.
- You should **filter/select frames** with:
  - **good masks**
  - a **complete and accurate calibration-board bounding frame**
  - stable views where the full board is clearly visible

(See the script output/logs for the exact OBJ output directory produced by your configuration.)

---

## End-to-end Online Multi-view Runtime

Below is the recommended run order for the online multi-view pipeline (multiple terminals).

### Terminal 1 — Start multi-view camera publisher
Start the multi-view camera publisher:
~~~bash
python video_publisher_multi_view.py
~~~

### Terminal 2 — Start segmentation (Grounded-SAM2 in Docker)
~~~bash
docker start -ai Corn_docker
conda activate sam
python docker_video_sub_pub_tracking_multi_view.py
~~~

### Terminal 3 — Start pose estimation (FoundationPose in Docker)
~~~bash
cd /home/galbot/ros_noetic_docker/FoundationPose
bash docker/run_container.sh
python realtime_multi_view_filter.py
~~~

After these are running:
- the camera publisher provides multi-view images
- Grounded-SAM2 subscribes, segments, and republishes results
- FoundationPose subscribes, estimates pose, and republishes pose outputs for the policy

---

## Policy (Example)

An example policy is provided for demonstration and testing:

~~~bash
python naive_policy.py
~~~

### Replace with your own policy

`naive_policy.py` is **only an example**. It is provided as a reference implementation showing how to:
- subscribe to perception outputs (e.g., ZMQ topics)
- compute targets from incoming data
- command the robot via `polymetis.RobotInterface`

**You should replace `naive_policy.py` with your own policy implementation** that matches your task logic, safety constraints, and robot setup.

> The current `naive_policy.py` contains hard-coded parameters and topic names.
> Please modify them to fit your environment (e.g., ZMQ address/topic, control gains, safety limits, etc.).

---

## Polymetis Control Interface

This repository assumes Polymetis is available in your environment (installed by the user).
Your policy/control code can use `polymetis.RobotInterface` to execute robot commands.

If you have an additional policy entrypoint different from `naive_policy.py`, document it here.

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

