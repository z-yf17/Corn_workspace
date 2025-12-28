#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
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
# 0) 避免 nvdiffrast 编译 compute_89（你的 nvcc 不支持）：
#    - 强制只编译到 8.6 并带 PTX，让驱动在 8.9 上 JIT
#    - 同时临时 monkeypatch torch.cuda.get_device_capability -> (8,6)
# ==============================================================================
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6+PTX"

# 降低 CPU 线程争抢
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

# ===================== 参数配置（可按需改为 argparse 参数） =====================
CAMS = ["front", "left", "right"]          # 融合点云用这三路
FRONT_CAM = "front"                        # FoundationPose 只用 front

# segd 订阅端（你的 worker 是 PUB connect -> SUB bind，所以这里保持 bind）
ZMQ_MODE = "bind"                          # "bind" or "connect"
ZMQ_ADDR = "tcp://0.0.0.0:5556"            # bind 用
TOPIC_PREFIX = "segd/"                     # worker 输出 topic: segd/<cam>

# 发布 refine 点云 & pose 的 PUB
ZMQ_OUT_MODE = "bind"                      # "bind" or "connect"
ZMQ_OUT_ADDR = "tcp://0.0.0.0:5557"
ZMQ_OUT_HWM = 3

# 外参：每个相机一份 npz，里面要有 T_cam_from_base (cam<-base)
EXTR_ROOT = "../calibration/calib_extrinsic"
EXTR_NAME = "fr3_realsense_eye_to_hand.npz"

# FoundationPose mesh
MESH_FILE = "./demo_data/mugblack_002.obj"

# FoundationPose 参数
EST_REFINE_ITER = 5
TRACK_REFINE_ITER = 2
DEBUG = 1
DEBUG_DIR = "./debug_live"
AXIS_SCALE = 0.1

# ====== 自动启动 FoundationPose（默认开启）======
AUTO_REGISTER = True
AUTO_REGISTER_MIN_MASK_PIX = 200          # mask 像素太小不注册
AUTO_REGISTER_COOLDOWN_SEC = 1.0          # 两次注册尝试最小间隔
AUTO_REREGISTER_ON_LOST = True            # track 失败后自动重注册
# ============================================

# mask refine（用于 FRONT 的 pose 注册/追踪）——只做形态学，不做连通域筛选
MASK_REFINE_ENABLE = True
MASK_OPEN_K = 3
MASK_CLOSE_K = 5
MASK_ERODE_ITER = 1

# 点云融合参数
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 2.00
STRIDE = 4
MAX_POINTS_PER_CAM = 12000
STALE_SEC = 1.0

# Open3D 降频
RENDER_HZ = 8.0

# ===================== OpenCV 可视化：三个视角都显示（默认开启） =====================
SHOW_CV_ALL_DEFAULT = True
CV_WIN_PREFIX = "segd_"            # 窗口名 segd_front / segd_left / segd_right
DRAW_POSE_ON_FRONT = True          # front 额外画 3D box/axis（其它视角不画）
CV_SHOW_THUMB = True               # 右上角贴一张小 mask
# ======================================================================

# ====== 本地把 mask 半透明叠加到原图上（所有视角同样风格）======
MASK_OVERLAY_ENABLE_DEFAULT = True
MASK_OVERLAY_ALPHA = 0.35
MASK_OVERLAY_COLOR_BGR = (0, 0, 255)
MASK_OVERLAY_DRAW_CONTOUR = True
MASK_OVERLAY_CONTOUR_COLOR_BGR = (0, 255, 255)
MASK_OVERLAY_CONTOUR_THICKNESS = 2
CV_VIS_REFINE_MASK = True          # 可视化时也做形态学（不做 CC）
# ======================================================================

# ====== 多视角支持率（防止单视角明显错误分割污染融合/均值）======
MULTIVIEW_SUPPORT_ENABLE = True
SUPPORT_VOXEL_M = 0.02
SUPPORT_RADIUS_M = 0.1
SUPPORT_MIN_RATE = 0.15
SUPPORT_MIN_POINTS = 300
SUPPORT_FILTER_POINTS = True       # True：做点级过滤；False：仅整路门控
SUPPORT_LOG = True
# =====================================================================

# 只取最新帧相关参数
POLL_TIMEOUT_MS = 1

# ===================== third-NN 去噪（一次性，不递归） =====================
THIRDNN_REFINE_ENABLE = True
THIRDNN_REMOVE_IF_D3_GT_M = 0.02
THIRDNN_REFINE_MIN_POINTS = 200
THIRDNN_K = 15
# ==========================================================================

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

# ===================== mask refine：只做形态学（不做连通域筛选） =====================
def refine_mask_u8(mask_u8, k_open=3, k_close=5, erode_iter=0):
    m = (mask_u8 > 0).astype(np.uint8) * 255
    k_open = max(1, int(k_open))
    k_close = max(1, int(k_close))

    if k_open > 1:
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k1)
    if k_close > 1:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2)

    if erode_iter > 0:
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.erode(m, ke, iterations=int(erode_iter))

    return m

def overlay_mask_on_bgr(
    image_bgr: np.ndarray,
    mask_u8: np.ndarray,
    alpha: float = 0.35,
    color_bgr: Tuple[int, int, int] = (0, 0, 255),
    draw_contour: bool = True,
    contour_color_bgr: Tuple[int, int, int] = (0, 255, 255),
    contour_thickness: int = 2,
) -> np.ndarray:
    if image_bgr is None or mask_u8 is None:
        return image_bgr
    H, W = image_bgr.shape[:2]
    if mask_u8.shape[:2] != (H, W):
        mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)

    m = (mask_u8 > 0)
    if not np.any(m):
        return image_bgr

    alpha = float(max(0.0, min(1.0, alpha)))
    out = image_bgr.copy()

    overlay = out.copy()
    overlay[m] = np.array(color_bgr, dtype=np.uint8)
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)

    if draw_contour:
        try:
            contours, _ = cv2.findContours((mask_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(out, contours, -1, contour_color_bgr, int(contour_thickness))
        except Exception:
            pass

    return out

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

def third_nn_distance_once(points_xyz: np.ndarray, thr_k: int = 3) -> np.ndarray:
    N = points_xyz.shape[0]
    out = np.full((N,), np.inf, dtype=np.float32)
    if N < (thr_k + 1):
        return out

    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(points_xyz)
    kdtree = o3d.geometry.KDTreeFlann(pcd_tmp)

    knn_total = thr_k + 1
    for i in range(N):
        _, _, dist2 = kdtree.search_knn_vector_3d(pcd_tmp.points[i], knn_total)
        if len(dist2) >= knn_total:
            out[i] = float(np.sqrt(dist2[thr_k]))
        else:
            out[i] = np.inf
    return out

# ------------------ 多视角支持率：体素近似最近距离 ------------------
_STRUCT3I = np.dtype([("x", np.int32), ("y", np.int32), ("z", np.int32)])

def _voxel_keys(P: np.ndarray, voxel: float) -> np.ndarray:
    return np.floor(P / float(voxel)).astype(np.int32)

def _pack_keys(K: np.ndarray) -> np.ndarray:
    Kc = np.ascontiguousarray(K.astype(np.int32))
    return Kc.view(_STRUCT3I).reshape(-1)

def _expand_keys(keys_packed_unique: np.ndarray, rad: int) -> np.ndarray:
    if rad <= 0 or keys_packed_unique.size == 0:
        return keys_packed_unique
    K = np.stack([keys_packed_unique["x"], keys_packed_unique["y"], keys_packed_unique["z"]], axis=1).astype(np.int32)
    offs = np.array(
        [(dx, dy, dz) for dx in range(-rad, rad + 1)
                    for dy in range(-rad, rad + 1)
                    for dz in range(-rad, rad + 1)],
        dtype=np.int32,
    )
    K2 = (K[None, :, :] + offs[:, None, :]).reshape(-1, 3)
    return np.unique(_pack_keys(K2))

def multiview_support_rate_and_keep(
    P_i: np.ndarray,
    P_other: np.ndarray,
    voxel: float,
    r_support: float,
    min_points: int,
) -> Tuple[float, Optional[np.ndarray]]:
    if P_i is None or P_other is None:
        return 1.0, None
    if P_i.shape[0] < int(min_points) or P_other.shape[0] < int(min_points):
        return 1.0, None

    rad = int(np.ceil(float(r_support) / float(voxel)))
    Ki = _pack_keys(_voxel_keys(P_i, voxel))
    Ko = _pack_keys(_voxel_keys(P_other, voxel))
    Ko_u = np.unique(Ko)
    Ko_exp = _expand_keys(Ko_u, rad=rad)

    keep = np.isin(Ki, Ko_exp)
    support = float(np.mean(keep)) if keep.size > 0 else 0.0
    return support, keep

# ------------------ ZMQ 最新帧 drain（输入） ------------------
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

# ------------------ ZMQ 发布（输出） ------------------
def make_pub_socket(ctx: zmq.Context) -> zmq.Socket:
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, int(ZMQ_OUT_HWM))
    pub.setsockopt(zmq.LINGER, 0)
    if ZMQ_OUT_MODE == "bind":
        pub.bind(ZMQ_OUT_ADDR)
        print(f"[ZMQ-OUT] PUB bind {ZMQ_OUT_ADDR}")
    else:
        pub.connect(ZMQ_OUT_ADDR)
        print(f"[ZMQ-OUT] PUB connect {ZMQ_OUT_ADDR}")
    return pub

def pub_pcd_refined(pub: zmq.Socket, ts: float, P_base: np.ndarray, C_rgb: np.ndarray):
    if pub is None:
        return
    if P_base is None or P_base.shape[0] == 0:
        return

    pts = np.ascontiguousarray(P_base.astype(np.float32))
    cols = np.ascontiguousarray(C_rgb.astype(np.float32))

    meta = {
        "frame": "base",
        "dtype": "float32",
        "points_shape": [int(pts.shape[0]), 3],
        "colors_shape": [int(cols.shape[0]), 3],
    }

    try:
        pub.send_multipart(
            [
                b"pcd_refined/base",
                f"{ts:.6f}".encode("utf-8"),
                json.dumps(meta).encode("utf-8"),
                pts.tobytes(),
                cols.tobytes(),
            ],
            flags=zmq.DONTWAIT,
        )
    except zmq.Again:
        pass

def pub_pose(pub: zmq.Socket, ts: float, pose_cam: Optional[np.ndarray], pose_base: Optional[np.ndarray], tracking: bool):
    if pub is None:
        return

    def _send(topic: bytes, mat: Optional[np.ndarray], frame: str):
        meta = {
            "frame": frame,
            "dtype": "float32",
            "shape": [4, 4],
            "tracking": bool(tracking),
            "valid": bool(mat is not None),
        }
        payload = b""
        if mat is not None:
            payload = np.ascontiguousarray(mat.astype(np.float32)).tobytes()
        try:
            pub.send_multipart(
                [topic, f"{ts:.6f}".encode("utf-8"), json.dumps(meta).encode("utf-8"), payload],
                flags=zmq.DONTWAIT,
            )
        except zmq.Again:
            pass

    _send(b"pose/front_cam", pose_cam, "front_cam")
    _send(b"pose/base", pose_base, "base")

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

def _make_cam_vis_bgr(
    cam: str,
    now: float,
    item: dict,
    overlay_enable: bool,
    support_rate: Optional[float],
    tracking: bool,
    frame_idx: int,
    pose: Optional[np.ndarray],
    to_origin: Optional[np.ndarray],
    K_front: Optional[np.ndarray],
    bbox: Optional[np.ndarray],
    draw_posed_3d_box,
    draw_xyz_axis,
) -> Optional[np.ndarray]:
    if item is None or item.get("rgb_jpg", None) is None or item.get("mask_png", None) is None:
        return None

    frame_bgr = decode_jpg_to_bgr(item["rgb_jpg"])
    mask_u8 = decode_png8_gray(item["mask_png"])
    if frame_bgr is None or mask_u8 is None:
        return None

    H, W = frame_bgr.shape[:2]
    if mask_u8.shape[:2] != (H, W):
        mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)

    if CV_VIS_REFINE_MASK:
        mask_u8_vis = refine_mask_u8(mask_u8, k_open=MASK_OPEN_K, k_close=MASK_CLOSE_K, erode_iter=MASK_ERODE_ITER)
    else:
        mask_u8_vis = (mask_u8 > 0).astype(np.uint8) * 255

    vis = frame_bgr.copy()

    if overlay_enable:
        vis = overlay_mask_on_bgr(
            vis,
            mask_u8_vis,
            alpha=MASK_OVERLAY_ALPHA,
            color_bgr=MASK_OVERLAY_COLOR_BGR,
            draw_contour=MASK_OVERLAY_DRAW_CONTOUR,
            contour_color_bgr=MASK_OVERLAY_CONTOUR_COLOR_BGR,
            contour_thickness=MASK_OVERLAY_CONTOUR_THICKNESS,
        )

    if CV_SHOW_THUMB:
        ms = cv2.resize(mask_u8_vis, (max(1, W // 4), max(1, H // 4)), interpolation=cv2.INTER_NEAREST)
        ms_bgr = cv2.cvtColor(ms, cv2.COLOR_GRAY2BGR)
        vis[0:ms_bgr.shape[0], W - ms_bgr.shape[1]:W] = ms_bgr

    if cam == FRONT_CAM and DRAW_POSE_ON_FRONT and pose is not None and to_origin is not None and K_front is not None and bbox is not None:
        try:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            vis_rgb = draw_posed_3d_box(K_front, img=vis_rgb, ob_in_cam=center_pose, bbox=bbox)
            vis_rgb = draw_xyz_axis(
                vis_rgb,
                ob_in_cam=center_pose,
                scale=AXIS_SCALE,
                K=K_front,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            vis = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    mask_pix = int((mask_u8_vis > 0).sum())
    stale = (now - float(item.get("t_recv", 0.0))) > STALE_SEC
    sr_txt = f"{support_rate:.2f}" if (support_rate is not None) else "--"
    st_txt = "STALE" if stale else "LIVE"

    if cam == FRONT_CAM:
        state = "TRACKING" if tracking else "IDLE"
        hud = f"{cam} | {state} | {st_txt} | frame={frame_idx} | mask_pix={mask_pix} | support={sr_txt}"
    else:
        hud = f"{cam} | {st_txt} | mask_pix={mask_pix} | support={sr_txt}"

    cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return vis

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="不创建任何可视化窗口（Open3D/OpenCV 都关闭）")
    ap.add_argument("--no_auto_register", action="store_true", help="禁用 FoundationPose 自动 register/重注册（默认开启）")
    ap.add_argument("--no_support_log", action="store_true", help="关闭 support_rate 打印")
    return ap.parse_args()

def main():
    global STOP
    args = parse_args()

    headless = bool(args.headless)
    auto_register = (not args.no_auto_register)
    support_log = (SUPPORT_LOG and (not args.no_support_log))

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

    # 3) Import FoundationPose stack
    from estimater import (
        set_logging_format, set_seed,
        ScorePredictor, PoseRefinePredictor, FoundationPose,
        draw_posed_3d_box, draw_xyz_axis,
    )
    set_logging_format()
    set_seed(0)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()

    # 4) nvdiffrast
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
    if headless:
        print("[MODE] HEADLESS ON (no Open3D/OpenCV windows).")
    else:
        print("[MODE] Visualization ON (Open3D + OpenCV). Keys: S=register(front) | R=reset | M=toggle overlay | Q/ESC=quit")

    if auto_register:
        print(
            f"[POSE] AUTO_REGISTER=ON min_pix={AUTO_REGISTER_MIN_MASK_PIX} cooldown={AUTO_REGISTER_COOLDOWN_SEC:.1f}s "
            f"reregister_on_lost={AUTO_REREGISTER_ON_LOST}"
        )
    else:
        print("[POSE] AUTO_REGISTER=OFF (manual only; in headless this means it will never register)")

    if MULTIVIEW_SUPPORT_ENABLE:
        print(
            f"[MV] multiview gating ON: voxel={SUPPORT_VOXEL_M:.3f}m r={SUPPORT_RADIUS_M:.3f}m "
            f"min_rate={SUPPORT_MIN_RATE:.2f} min_pts={SUPPORT_MIN_POINTS} filter_points={SUPPORT_FILTER_POINTS}"
        )
    if THIRDNN_REFINE_ENABLE:
        print(f"[PCD] kNN denoise: remove points with d{THIRDNN_K} > {THIRDNN_REMOVE_IF_D3_GT_M:.3f} m (one-shot)")

    print(f"[ZMQ] IN  segd/<cam>  @ {ZMQ_ADDR} (mode={ZMQ_MODE})")
    print(f"[ZMQ] OUT pose+pcd    @ {ZMQ_OUT_ADDR} (mode={ZMQ_OUT_MODE})")

    # 5) ZMQ
    ctx = zmq.Context()
    sub, poller = make_sub_socket(ctx)
    pub = make_pub_socket(ctx)

    # per-cam latest cache
    latest: Dict[str, dict] = {
        cam: {"t_recv": 0.0, "ts": 0.0, "meta": None, "mask_png": None, "depth_png": None, "rgb_jpg": None}
        for cam in CAMS
    }

    # 6) Open3D for fused point cloud (optional)
    vis = None
    pcd = None
    if not headless:
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

        vis.register_key_callback(ord("Q"), _quit)
        vis.register_key_callback(ord("q"), _quit)
        vis.register_key_callback(256, _quit)  # ESC

    # 7) OpenCV windows (optional)
    show_cv = (SHOW_CV_ALL_DEFAULT and (not headless))
    overlay_enable = bool(MASK_OVERLAY_ENABLE_DEFAULT)
    last_vis_bgr: Dict[str, Optional[np.ndarray]] = {cam: None for cam in CAMS}

    if show_cv:
        for cam in CAMS:
            cv2.namedWindow(f"{CV_WIN_PREFIX}{cam}", cv2.WINDOW_NORMAL)

    # pose state (front only)
    tracking = False
    pose = None
    need_register = False
    frame_idx = 0
    last_front_K = None
    last_register_try_t = 0.0

    last_support_rates: Dict[str, float] = {}

    render_dt = 1.0 / max(RENDER_HZ, 1e-6)
    t_last_render = 0.0

    try:
        while not STOP:
            # ---- Open3D event loop (if enabled) ----
            if vis is not None:
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
                            mask_u8_pose = refine_mask_u8(mask_u8, k_open=MASK_OPEN_K, k_close=MASK_CLOSE_K, erode_iter=MASK_ERODE_ITER)
                        else:
                            mask_u8_pose = (mask_u8 > 0).astype(np.uint8) * 255

                        ob_mask = (mask_u8_pose > 0)
                        mask_pix = int(ob_mask.sum())
                        depth_m = np.ascontiguousarray(depth_u16.astype(np.float32) * np.float32(depth_scale), dtype=np.float32)
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                        frame_idx += 1
                        last_front_K = K

                        # --- auto register trigger ---
                        if auto_register:
                            need_auto = False
                            if pose is None:
                                need_auto = True
                            elif (not tracking) and AUTO_REREGISTER_ON_LOST:
                                need_auto = True

                            if need_auto and (now - last_register_try_t) >= float(AUTO_REGISTER_COOLDOWN_SEC):
                                if mask_pix >= int(AUTO_REGISTER_MIN_MASK_PIX):
                                    need_register = True
                                last_register_try_t = now

                        if need_register:
                            if mask_pix < int(AUTO_REGISTER_MIN_MASK_PIX):
                                # 手动触发也尊重 min_pix，避免乱注册
                                need_register = False
                            else:
                                try:
                                    print(f"[POSE] register(FRONT)... frame={frame_idx} mask_pix={mask_pix}")
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

                        # --- 发布 pose（front_cam + base）---
                        pose_cam = None
                        pose_base = None
                        if pose is not None:
                            try:
                                center_pose = pose @ np.linalg.inv(to_origin)
                                pose_cam = center_pose.astype(np.float64)
                                pose_base = (T_base_from_front @ pose_cam).astype(np.float64)
                            except Exception:
                                pose_cam = None
                                pose_base = None
                        pub_pose(pub, ts=now, pose_cam=pose_cam, pose_base=pose_base, tracking=tracking)

            # ---- Fuse point cloud (all cams) at low rate ----
            now = time.time()
            if (now - t_last_render) >= render_dt:
                t_last_render = now

                per_cam_pts: Dict[str, np.ndarray] = {}
                per_cam_cols: Dict[str, np.ndarray] = {}

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

                    per_cam_pts[cam] = pts_base.astype(np.float64)
                    per_cam_cols[cam] = cols_rgb.astype(np.float64)

                kept_pts = []
                kept_cols = []
                support_rates: Dict[str, float] = {}

                if MULTIVIEW_SUPPORT_ENABLE and len(per_cam_pts) >= 2:
                    for cam in CAMS:
                        P = per_cam_pts.get(cam, None)
                        C = per_cam_cols.get(cam, None)
                        if P is None or P.shape[0] == 0:
                            continue

                        others = []
                        for c2 in CAMS:
                            if c2 == cam:
                                continue
                            P2 = per_cam_pts.get(c2, None)
                            if P2 is None or P2.shape[0] == 0:
                                continue
                            others.append(P2)

                        if len(others) == 0:
                            support_rates[cam] = 1.0
                            kept_pts.append(P); kept_cols.append(C)
                            continue

                        P_other = np.concatenate(others, axis=0)
                        rate, keep_mask = multiview_support_rate_and_keep(
                            P_i=P,
                            P_other=P_other,
                            voxel=SUPPORT_VOXEL_M,
                            r_support=SUPPORT_RADIUS_M,
                            min_points=SUPPORT_MIN_POINTS,
                        )
                        support_rates[cam] = float(rate)

                        if rate < float(SUPPORT_MIN_RATE):
                            continue

                        if SUPPORT_FILTER_POINTS and keep_mask is not None:
                            kept_pts.append(P[keep_mask])
                            kept_cols.append(C[keep_mask])
                        else:
                            kept_pts.append(P)
                            kept_cols.append(C)

                    last_support_rates = support_rates.copy()

                    if support_log and support_rates:
                        sr_txt = " | ".join([f"{k}:{support_rates[k]:.2f}" for k in sorted(support_rates.keys())])
                        print(f"[MV] support_rate  {sr_txt}")
                else:
                    for cam, P in per_cam_pts.items():
                        kept_pts.append(P)
                        kept_cols.append(per_cam_cols[cam])

                if len(kept_pts) > 0:
                    P = np.concatenate(kept_pts, axis=0)
                    C = np.concatenate(kept_cols, axis=0)

                    if THIRDNN_REFINE_ENABLE and P.shape[0] >= max(THIRDNN_REFINE_MIN_POINTS, THIRDNN_K + 1):
                        dK = third_nn_distance_once(P, thr_k=THIRDNN_K)
                        keep = dK <= float(THIRDNN_REMOVE_IF_D3_GT_M)
                        P = P[keep]
                        C = C[keep]

                    # 更新 Open3D（如果启用）
                    if (vis is not None) and (pcd is not None):
                        pcd.points = o3d.utility.Vector3dVector(P)
                        pcd.colors = o3d.utility.Vector3dVector(C)
                        vis.update_geometry(pcd)
                        vis.update_renderer()

                    # 发布 refine 点云（base）
                    pub_pcd_refined(pub, ts=now, P_base=P, C_rgb=C)

            # ---- OpenCV visualization (optional) ----
            if show_cv:
                now = time.time()
                for cam in CAMS:
                    item = latest.get(cam, None)
                    sr = last_support_rates.get(cam, None)

                    vis_bgr = _make_cam_vis_bgr(
                        cam=cam,
                        now=now,
                        item=item,
                        overlay_enable=overlay_enable,
                        support_rate=sr,
                        tracking=tracking,
                        frame_idx=frame_idx,
                        pose=pose,
                        to_origin=to_origin,
                        K_front=last_front_K,
                        bbox=bbox,
                        draw_posed_3d_box=draw_posed_3d_box,
                        draw_xyz_axis=draw_xyz_axis,
                    )
                    if vis_bgr is not None:
                        last_vis_bgr[cam] = vis_bgr

                    if last_vis_bgr[cam] is not None:
                        cv2.imshow(f"{CV_WIN_PREFIX}{cam}", last_vis_bgr[cam])

                # 单次 waitKey 处理全局按键
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
                if key in (ord("m"), ord("M")):
                    overlay_enable = not overlay_enable
                    print(f"[UI] mask overlay (all) -> {'ON' if overlay_enable else 'OFF'}")
            else:
                # headless 或禁用可视化时，避免空转满 CPU
                time.sleep(0.001)

    finally:
        if show_cv:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        try:
            sub.close(0)
        except Exception:
            pass
        try:
            pub.close(0)
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass
        try:
            if vis is not None:
                vis.destroy_window()
        except Exception:
            pass
        print("[EXIT] done.")

if __name__ == "__main__":
    main()
