#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import signal
from typing import Dict, Optional, Tuple, List

import numpy as np
import cv2
import zmq
import trimesh
import torch
import open3d as o3d

# ==============================================================================
# 0) 关键：避免 nvdiffrast 编译 compute_89（你的 nvcc 不支持）：
#    - 强制只编译到 8.6 并带 PTX，让驱动在 8.9 上 JIT
#    - 同时临时 monkeypatch torch.cuda.get_device_capability -> (8,6)
# ==============================================================================
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6+PTX"

# 降低 CPU 线程争抢
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

# ===================== 参数配置 =====================
CAMS = ["front", "left", "right"]          # 融合点云用这三路
FRONT_CAM = "front"                        # FoundationPose 只用 front（保持之前效果）

# segd 发布端（你的 worker 是 PUB connect -> SUB bind，所以这里保持 bind）
ZMQ_MODE = "bind"                          # "bind" or "connect"
ZMQ_ADDR = "tcp://0.0.0.0:5556"            # bind 用
TOPIC_PREFIX = "segd/"                     # worker 输出 topic: segd/<cam>

# 外参：每个相机一份 npz，里面要有 T_cam_from_base (cam<-base)
EXTR_ROOT = "../calibration_latest/calib_extrinsic"
EXTR_NAME = "fr3_realsense_eye_to_hand.npz"

# FoundationPose mesh
MESH_FILE = "./demo_data/mugblack_002.obj"

# FoundationPose 参数（保持你之前的默认）
EST_REFINE_ITER = 5
TRACK_REFINE_ITER = 2
DEBUG = 1
DEBUG_DIR = "./debug_live"
MIN_MASK_PIX_FOR_REGISTER = 200
AXIS_SCALE = 0.1

# mask refine（保持之前）
MASK_REFINE_ENABLE = True
MASK_MIN_AREA = 800
MASK_OPEN_K = 3
MASK_CLOSE_K = 5
MASK_ERODE_ITER = 1

# 点云融合参数（稳一点）
DEPTH_MIN_M = 0.20
DEPTH_MAX_M = 2.00
STRIDE = 4
MAX_POINTS_PER_CAM = 12000
STALE_SEC = 1.0

# Open3D 降频
RENDER_HZ = 8.0

# OpenCV 可视化只显示 front
SHOW_FRONT_CV = True
CV_WIN = "front_foundationpose"

# 只取最新帧相关参数
POLL_TIMEOUT_MS = 1
# ====================================================

# ===================== float64 -> torch.float32 拦截（保持你之前） =====================
torch.set_default_dtype(torch.float32)

_orig_from_numpy = torch.from_numpy
def _from_numpy_force_f32(x):
    t = _orig_from_numpy(x)
    return t.float() if t.dtype == torch.float64 else t
torch.from_numpy = _from_numpy_force_f32

_orig_as_tensor = torch.as_tensor
def _as_tensor_force_f32(data, *args, **kwargs):
    t = _orig_as_tensor(data, *args, **kwargs)
    if ("dtype" not in kwargs or kwargs["dtype"] is None) and t.dtype == torch.float64:
        return t.float()
    return t
torch.as_tensor = _as_tensor_force_f32

_orig_tensor = torch.tensor
def _tensor_force_f32(data, *args, **kwargs):
    t = _orig_tensor(data, *args, **kwargs)
    if ("dtype" not in kwargs or kwargs["dtype"] is None) and t.dtype == torch.float64:
        return t.float()
    return t
torch.tensor = _tensor_force_f32
# ==============================================================================

STOP = False
def _on_sig(sig, frame):
    global STOP
    STOP = True
signal.signal(signal.SIGINT, _on_sig)
signal.signal(signal.SIGTERM, _on_sig)


# ------------------ 基本工具 ------------------
def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def decode_png16(depth_png_bytes: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(depth_png_bytes, dtype=np.uint8)
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        return None
    return depth


def decode_png8_gray(mask_png_bytes: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(mask_png_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def decode_jpg_to_bgr(jpg_bytes: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def meta_to_K_and_scale(meta: dict) -> Tuple[np.ndarray, float]:
    fx = float(meta["fx"])
    fy = float(meta["fy"])
    cx = float(meta["ppx"])
    cy = float(meta["ppy"])
    depth_scale = float(meta.get("depth_scale", 0.001))
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K, depth_scale


def refine_mask_u8(mask_u8, min_area=800, k_open=3, k_close=5, erode_iter=0):
    m = (mask_u8 > 0).astype(np.uint8) * 255
    k_open = max(1, int(k_open))
    k_close = max(1, int(k_close))

    if k_open > 1:
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k1)
    if k_close > 1:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return m

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    if stats[best, cv2.CC_STAT_AREA] < int(min_area):
        return np.zeros_like(m)

    m2 = (labels == best).astype(np.uint8) * 255

    if erode_iter > 0:
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m2 = cv2.erode(m2, ke, iterations=int(erode_iter))

    return m2


def backproject_masked_points(
    depth_m: np.ndarray,
    mask_u8: np.ndarray,
    K: np.ndarray,
    stride: int,
    zmin: float,
    zmax: float,
    max_points: int,
    color_bgr: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """pts_cam float32, cols_rgb float32"""
    H, W = depth_m.shape[:2]
    if mask_u8.shape[:2] != (H, W):
        mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)

    z = depth_m.astype(np.float32)
    m = (mask_u8 > 0)

    z_s = z[0:H:stride, 0:W:stride]
    m_s = m[0:H:stride, 0:W:stride]

    valid = m_s & (z_s > zmin) & (z_s < zmax)
    if not np.any(valid):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])

    us = np.arange(0, W, stride, dtype=np.float32)
    vs = np.arange(0, H, stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)

    Z = z_s[valid].astype(np.float32)
    U = uu[valid].astype(np.float32)
    V = vv[valid].astype(np.float32)

    X = (U - cx) * Z / fx
    Y = (V - cy) * Z / fy
    pts = np.stack([X, Y, Z], axis=1).astype(np.float32)

    if color_bgr is not None and color_bgr.shape[:2] == (H, W):
        c_s = color_bgr[0:H:stride, 0:W:stride, :]
        rgb = c_s[valid][:, ::-1].astype(np.float32) / 255.0
    else:
        rgb = np.zeros((pts.shape[0], 3), np.float32)

    n = pts.shape[0]
    if max_points > 0 and n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        pts = pts[idx]
        rgb = rgb[idx]

    return pts, rgb


# ------------------ ZMQ 最新帧 drain ------------------
def make_sub_socket(ctx: zmq.Context) -> Tuple[zmq.Socket, zmq.Poller]:
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 50)
    sub.setsockopt(zmq.LINGER, 0)

    if ZMQ_MODE == "bind":
        sub.bind(ZMQ_ADDR)
        print(f"[ZMQ] SUB bind {ZMQ_ADDR}")
    else:
        sub.connect(ZMQ_ADDR)
        print(f"[ZMQ] SUB connect {ZMQ_ADDR}")

    # 订阅 segd/ 前缀即可（会收到 segd/front segd/left segd/right）
    sub.setsockopt_string(zmq.SUBSCRIBE, TOPIC_PREFIX)
    print(f"[ZMQ] subscribe prefix '{TOPIC_PREFIX}'")

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)
    return sub, poller


def drain_all_latest(sub: zmq.Socket, poller: zmq.Poller, timeout_ms: int = 5) -> List[List[bytes]]:
    socks = dict(poller.poll(timeout_ms))
    if sub not in socks:
        return []
    out = []
    while True:
        try:
            parts = sub.recv_multipart(flags=zmq.NOBLOCK)
            out.append(parts)
        except zmq.Again:
            break
    return out


def parse_segd_parts(parts: List[bytes]) -> Optional[Tuple[str, float, dict, bytes, bytes, Optional[bytes]]]:
    """
    segd/<cam> multipart:
      5: topic, ts, meta, mask_png, depth_png
      6: + raw_rgb_jpg
      7: + raw_rgb_jpg, overlay_jpg
    返回: (cam, ts, meta_dict, mask_png, depth_png, raw_rgb_jpg_or_None)
    """
    if len(parts) not in (5, 6, 7):
        return None

    try:
        topic = parts[0].decode("utf-8")
    except Exception:
        return None
    if not topic.startswith(TOPIC_PREFIX):
        return None
    cam = topic[len(TOPIC_PREFIX):].strip()
    if cam not in CAMS:
        return None

    try:
        ts = float(parts[1].decode("utf-8"))
    except Exception:
        ts = time.time()

    try:
        meta = json.loads(parts[2].decode("utf-8"))
        if not isinstance(meta, dict):
            meta = {}
    except Exception:
        meta = {}

    mask_png = parts[3]
    depth_png = parts[4]
    raw_rgb = parts[5] if len(parts) >= 6 else None
    return cam, ts, meta, mask_png, depth_png, raw_rgb


def load_T_base_from_cam_for_all(cams: List[str]) -> Dict[str, np.ndarray]:
    T = {}
    for cam in cams:
        npz_path = os.path.join(EXTR_ROOT, cam, EXTR_NAME)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing extrinsic npz for {cam}: {npz_path}")
        r = np.load(npz_path, allow_pickle=True)
        if "T_cam_from_base" not in r:
            raise KeyError(f"{npz_path} missing key 'T_cam_from_base'")
        T_cam_from_base = r["T_cam_from_base"].astype(np.float64)
        T_base_from_cam = inv_T(T_cam_from_base)
        T[cam] = T_base_from_cam
        print(f"[EXTR] {cam}: loaded {npz_path}")
    return T


def main():
    global STOP
    os.makedirs(DEBUG_DIR, exist_ok=True)

    # 1) extrinsics for fusion
    T_base_from_cam = load_T_base_from_cam_for_all(CAMS)
    T_base_from_front = T_base_from_cam[FRONT_CAM].astype(np.float64)

    # 2) FoundationPose: load mesh
    if not os.path.exists(MESH_FILE):
        raise FileNotFoundError(f"MESH_FILE not found: {MESH_FILE}")
    mesh = trimesh.load(MESH_FILE)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # 3) Import FoundationPose stack (保持你之前结构)
    from estimater import (
        set_logging_format, set_seed,
        ScorePredictor, PoseRefinePredictor, FoundationPose,
        draw_posed_3d_box, draw_xyz_axis,
    )

    set_logging_format()
    set_seed(0)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()

    # 4) nvdiffrast：不使用 “import nvdiffrast.torch as dr”
    #    保险：临时把 capability 伪装成 8.6，防止它内部强行 set compute_89
    from nvdiffrast.torch import RasterizeCudaContext

    _orig_get_cap = None
    if torch.cuda.is_available():
        _orig_get_cap = torch.cuda.get_device_capability
        torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 6)

    try:
        glctx = RasterizeCudaContext()
    finally:
        if _orig_get_cap is not None:
            torch.cuda.get_device_capability = _orig_get_cap

    est = FoundationPose(
        model_pts=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
        model_normals=np.ascontiguousarray(mesh.vertex_normals, dtype=np.float32),
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=DEBUG_DIR,
        debug=DEBUG,
        glctx=glctx,
    )

    print("[INIT] FoundationPose ready.")
    print("[UI] Keys: S=register using FRONT mask | R=reset | Q/ESC=quit")

    # 5) ZMQ
    ctx = zmq.Context()
    sub, poller = make_sub_socket(ctx)

    # per-cam latest cache
    latest: Dict[str, dict] = {
        cam: {"t_recv": 0.0, "ts": 0.0, "meta": None, "mask_png": None, "depth_png": None, "rgb_jpg": None}
        for cam in CAMS
    }

    # 6) Open3D for fused point cloud
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Fused segd cloud (front+left+right)", width=1280, height=720)
    try:
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0, 0, 0], dtype=np.float64)
    except Exception:
        pass
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis.add_geometry(axis)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    def _quit(_vis):
        global STOP
        STOP = True
        return False
    vis.register_key_callback(ord('Q'), _quit)
    vis.register_key_callback(ord('q'), _quit)
    vis.register_key_callback(256, _quit)  # ESC

    # 7) OpenCV for front visualization only
    tracking = False
    pose = None
    need_register = False
    frame_idx = 0

    if SHOW_FRONT_CV:
        cv2.namedWindow(CV_WIN, cv2.WINDOW_NORMAL)

    render_dt = 1.0 / max(RENDER_HZ, 1e-6)
    t_last_render = 0.0
    last_front_vis = None

    try:
        while not STOP:
            if not vis.poll_events():
                break

            # ---- drain all messages (latest per cam) ----
            msgs = drain_all_latest(sub, poller, timeout_ms=POLL_TIMEOUT_MS)
            now = time.time()
            for parts in msgs:
                parsed = parse_segd_parts(parts)
                if parsed is None:
                    continue
                cam, ts, meta, mask_png, depth_png, raw_rgb = parsed
                latest[cam]["t_recv"] = now
                latest[cam]["ts"] = ts
                latest[cam]["meta"] = meta
                latest[cam]["mask_png"] = mask_png
                latest[cam]["depth_png"] = depth_png
                latest[cam]["rgb_jpg"] = raw_rgb

            # ---- FoundationPose (front only) ----
            front_item = latest.get(FRONT_CAM, None)
            if front_item and front_item["mask_png"] is not None and front_item["depth_png"] is not None and front_item["meta"] is not None:
                if (now - front_item["t_recv"]) <= STALE_SEC:
                    # decode front
                    try:
                        K, depth_scale = meta_to_K_and_scale(front_item["meta"])
                    except Exception:
                        K, depth_scale = None, None

                    depth_u16 = decode_png16(front_item["depth_png"])
                    mask_u8 = decode_png8_gray(front_item["mask_png"])
                    frame_bgr = decode_jpg_to_bgr(front_item["rgb_jpg"]) if front_item["rgb_jpg"] is not None else None

                    if K is not None and depth_u16 is not None and mask_u8 is not None and frame_bgr is not None:
                        if frame_bgr.shape[:2] != depth_u16.shape:
                            frame_bgr = cv2.resize(frame_bgr, (depth_u16.shape[1], depth_u16.shape[0]), interpolation=cv2.INTER_LINEAR)
                        if mask_u8.shape[:2] != depth_u16.shape:
                            mask_u8 = cv2.resize(mask_u8, (depth_u16.shape[1], depth_u16.shape[0]), interpolation=cv2.INTER_NEAREST)

                        if MASK_REFINE_ENABLE:
                            mask_u8 = refine_mask_u8(mask_u8, MASK_MIN_AREA, MASK_OPEN_K, MASK_CLOSE_K, MASK_ERODE_ITER)
                        else:
                            mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255

                        ob_mask = (mask_u8 > 0)
                        mask_pix = int(ob_mask.sum())
                        depth_m = np.ascontiguousarray(depth_u16.astype(np.float32) * np.float32(depth_scale), dtype=np.float32)
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                        frame_idx += 1

                        if need_register:
                            if mask_pix < MIN_MASK_PIX_FOR_REGISTER:
                                print(f"[POSE] FRONT mask too small ({mask_pix}), wait...")
                            else:
                                try:
                                    print(f"[POSE] register(FRONT)... mask_pix={mask_pix}")
                                    pose = est.register(
                                        K=np.ascontiguousarray(K, dtype=np.float32),
                                        rgb=frame_rgb,
                                        depth=depth_m,
                                        ob_mask=ob_mask.astype(bool),
                                        iteration=EST_REFINE_ITER,
                                    )
                                    tracking = True
                                    print("[POSE] register OK -> tracking ON.")
                                except Exception as e:
                                    print("[POSE] register failed:", e)
                                    tracking = False
                                    pose = None
                            need_register = False

                        if tracking and pose is not None:
                            try:
                                pose = est.track_one(
                                    rgb=frame_rgb,
                                    depth=depth_m,
                                    K=np.ascontiguousarray(K, dtype=np.float32),
                                    iteration=TRACK_REFINE_ITER,
                                )
                            except Exception as e:
                                print("[POSE] track_one failed:", e)
                                tracking = False
                                pose = None

                        # front visualization (保持你之前的样式)
                        if SHOW_FRONT_CV:
                            vis_bgr = frame_bgr.copy()

                            mask_small = cv2.resize(mask_u8, (vis_bgr.shape[1] // 4, vis_bgr.shape[0] // 4),
                                                    interpolation=cv2.INTER_NEAREST)
                            mask_small_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                            vis_bgr[0:mask_small_bgr.shape[0],
                                    vis_bgr.shape[1]-mask_small_bgr.shape[1]:vis_bgr.shape[1]] = mask_small_bgr

                            if pose is not None:
                                center_pose = pose @ np.linalg.inv(to_origin)
                                vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                                vis_rgb = draw_posed_3d_box(K, img=vis_rgb, ob_in_cam=center_pose, bbox=bbox)
                                vis_rgb = draw_xyz_axis(
                                    vis_rgb, ob_in_cam=center_pose, scale=AXIS_SCALE, K=K,
                                    thickness=3, transparency=0, is_input_rgb=True
                                )
                                vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

                            hud1 = "TRACKING" if tracking else "IDLE"
                            hud2 = "S=register(front mask) | R=reset | Q=quit"
                            cv2.putText(vis_bgr, f"{hud1} frame={frame_idx} front_mask_pix={mask_pix}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.putText(vis_bgr, hud2, (10, 65),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

                            last_front_vis = vis_bgr

            # ---- OpenCV key handling (front only) ----
            if SHOW_FRONT_CV:
                if last_front_vis is not None:
                    cv2.imshow(CV_WIN, last_front_vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("s"), ord("S")):
                    need_register = True
                    tracking = False
                    pose = None
                    print("[UI] register requested (FRONT).")
                if key in (ord("r"), ord("R")):
                    need_register = False
                    tracking = False
                    pose = None
                    print("[UI] reset -> IDLE.")

            # ---- Fuse point cloud (all cams) at low rate ----
            now = time.time()
            if (now - t_last_render) >= render_dt:
                t_last_render = now

                all_pts = []
                all_cols = []

                for cam in CAMS:
                    item = latest[cam]
                    if item["mask_png"] is None or item["depth_png"] is None or item["meta"] is None:
                        continue
                    if (now - item["t_recv"]) > STALE_SEC:
                        continue

                    try:
                        Kc, depth_scale = meta_to_K_and_scale(item["meta"])
                    except Exception:
                        continue

                    depth_u16 = decode_png16(item["depth_png"])
                    mask_u8 = decode_png8_gray(item["mask_png"])
                    if depth_u16 is None or mask_u8 is None:
                        continue

                    depth_m = depth_u16.astype(np.float32) * np.float32(depth_scale)

                    color_bgr = None
                    if item["rgb_jpg"] is not None:
                        color_bgr = decode_jpg_to_bgr(item["rgb_jpg"])
                        if color_bgr is not None and color_bgr.shape[:2] != depth_u16.shape:
                            color_bgr = cv2.resize(color_bgr, (depth_u16.shape[1], depth_u16.shape[0]), interpolation=cv2.INTER_LINEAR)

                    pts_cam, cols_rgb = backproject_masked_points(
                        depth_m=depth_m,
                        mask_u8=mask_u8,
                        K=Kc,
                        stride=STRIDE,
                        zmin=DEPTH_MIN_M,
                        zmax=DEPTH_MAX_M,
                        max_points=MAX_POINTS_PER_CAM,
                        color_bgr=color_bgr,
                    )
                    if pts_cam.shape[0] == 0:
                        continue

                    Tbc = T_base_from_cam[cam].astype(np.float32)
                    R = Tbc[:3, :3]
                    t = Tbc[:3, 3]
                    pts_base = (pts_cam @ R.T) + t.reshape(1, 3)

                    all_pts.append(pts_base.astype(np.float64))
                    all_cols.append(cols_rgb.astype(np.float64))

                if len(all_pts) > 0:
                    P = np.concatenate(all_pts, axis=0)
                    C = np.concatenate(all_cols, axis=0)
                    pcd.points = o3d.utility.Vector3dVector(P)
                    pcd.colors = o3d.utility.Vector3dVector(C)
                    vis.update_geometry(pcd)

                vis.update_renderer()
            else:
                time.sleep(0.001)

    finally:
        if SHOW_FRONT_CV:
            cv2.destroyAllWindows()
        try:
            sub.close(0)
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass
        try:
            vis.destroy_window()
        except Exception:
            pass
        print("[EXIT] done.")


if __name__ == "__main__":
    main()
