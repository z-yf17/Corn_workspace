#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import signal
from typing import Dict, Optional, Tuple, List
from collections import deque

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
CAMS = ["front", "left", "right"]
FRONT_CAM = "front"

# segd 订阅端（你的 worker 是 PUB connect -> SUB bind，所以这里保持 bind）
ZMQ_MODE = "bind"
ZMQ_ADDR = "tcp://0.0.0.0:5556"
TOPIC_PREFIX = "segd/"

# 发布 refine 点云 & pose 的 PUB
ZMQ_OUT_MODE = "bind"
ZMQ_OUT_ADDR = "tcp://0.0.0.0:5557"
ZMQ_OUT_HWM = 3

# 外参：每个相机一份 npz，里面要有 T_cam_from_base (cam<-base)
EXTR_ROOT = "../calibration_latest/calib_extrinsic"
EXTR_NAME = "fr3_realsense_eye_to_hand.npz"

# FoundationPose mesh
MESH_FILE = "./demo_data/mugblack_002.obj"

# FoundationPose 参数
EST_REFINE_ITER = 5
TRACK_REFINE_ITER = 2
DEBUG = 1
DEBUG_DIR = "./debug_live"
MIN_MASK_PIX_FOR_REGISTER = 200

# ===================== headless：自动注册/重注册 =====================
AUTO_REGISTER = True
AUTO_REGISTER_COOLDOWN_SEC = 1.0   # 两次 register 尝试最小间隔
AUTO_REREGISTER_ON_LOST = True     # tracking 掉了自动重试
AUTO_REGISTER_MIN_MASK_PIX = MIN_MASK_PIX_FOR_REGISTER
# ====================================================================

# mask refine：只做形态学，不做连通域筛选
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

# 融合/发布频率（headless 只控制融合节奏）
RENDER_HZ = 8.0

# ====== 多视角支持率（点级过滤）======
MULTIVIEW_SUPPORT_ENABLE = True
SUPPORT_VOXEL_M = 0.02
SUPPORT_RADIUS_M = 0.06
SUPPORT_MIN_RATE = 0.15
SUPPORT_MIN_POINTS = 300
SUPPORT_FILTER_POINTS = True
SUPPORT_LOG = True
# =====================================================================

# 只取最新帧相关参数
POLL_TIMEOUT_MS = 1

# ====== third-NN 去噪（一次性，不递归链式）======
THIRDNN_REFINE_ENABLE = True
THIRDNN_REMOVE_IF_D3_GT_M = 0.02
THIRDNN_REFINE_MIN_POINTS = 200
THIRDNN_K = 15
# ===================================================================

# ====== center gate（优先 pose/base）======
CENTER_GATE_ENABLE = True
CENTER_GATE_RADIUS_M = 0.20
CENTER_HIST_LEN = 10
CENTER_JUMP_MAX_M = 0.12
CENTER_FALLBACK_MIN_PTS = 600
CENTER_MIN_KEEP_PTS = 200
POSE_CENTER_TTL_SEC = 0.50
# =======================================

# ====== pose 3D box gate（把方框外点裁掉）======
POSE_BOX_GATE_ENABLE = True
POSE_BOX_GATE_MARGIN_M = 0.02
POSE_BOX_GATE_TTL_SEC = 0.50
POSE_BOX_MIN_KEEP_PTS = 300
POSE_BOX_MIN_KEEP_RATE = 0.03
# ============================================

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
    if depth_png_bytes is None:
        return None
    arr = np.frombuffer(depth_png_bytes, dtype=np.uint8)
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        return None
    return depth

def decode_png8_gray(mask_png_bytes: bytes) -> Optional[np.ndarray]:
    if mask_png_bytes is None:
        return None
    arr = np.frombuffer(mask_png_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

def decode_jpg_to_bgr(jpg_bytes: bytes) -> Optional[np.ndarray]:
    if jpg_bytes is None:
        return None
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

def refine_mask_u8(mask_u8, k_open=3, k_close=5, erode_iter=0):
    """只做形态学（open/close + 可选 erode），不做连通域筛选。"""
    if mask_u8 is None:
        return None
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

# ------------------ center gate ------------------
def robust_center_from_points(P: np.ndarray) -> Optional[np.ndarray]:
    if P is None or P.shape[0] == 0:
        return None
    c0 = np.median(P, axis=0)
    d = np.linalg.norm(P - c0[None, :], axis=1)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    sigma = 1.4826 * mad + 1e-6
    thr = med + 2.5 * sigma
    inl = d <= thr
    if int(inl.sum()) < max(50, int(0.10 * P.shape[0])):
        return c0.astype(np.float64)
    c = P[inl].mean(axis=0)
    return c.astype(np.float64)

class CenterGate:
    def __init__(self, hist_len: int, jump_max_m: float, radius_m: float):
        self.hist_len = int(max(1, hist_len))
        self.jump_max_m = float(jump_max_m)
        self.radius_m = float(radius_m)
        self.centers = deque(maxlen=self.hist_len)

    def reset(self):
        self.centers.clear()

    def reference_center(self) -> Optional[np.ndarray]:
        if len(self.centers) == 0:
            return None
        C = np.stack(list(self.centers), axis=0)
        return np.median(C, axis=0).astype(np.float64)

    def update(self, c_new: Optional[np.ndarray]) -> bool:
        if c_new is None:
            return False
        c_new = np.asarray(c_new, dtype=np.float64).reshape(3,)
        c_ref = self.reference_center()
        if c_ref is None:
            self.centers.append(c_new)
            return True
        dist = float(np.linalg.norm(c_new - c_ref))
        if dist > self.jump_max_m:
            return False
        self.centers.append(c_new)
        return True

    def filter_points(self, P: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        c_ref = self.reference_center()
        if c_ref is None or P is None or P.shape[0] == 0:
            return P, C, None
        d = np.linalg.norm(P - c_ref[None, :], axis=1)
        keep = d <= self.radius_m
        rate = float(np.mean(keep)) if keep.size > 0 else 0.0
        return P[keep], C[keep], rate

# ------------------ pose box gate ------------------
def box_gate_by_pose(
    P_base: np.ndarray,
    C_rgb: np.ndarray,
    T_base_from_obj: np.ndarray,
    half_extents: np.ndarray,
    margin: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if P_base is None or P_base.shape[0] == 0:
        return P_base, C_rgb, 0.0
    R = T_base_from_obj[:3, :3].astype(np.float64)
    t = T_base_from_obj[:3, 3].astype(np.float64).reshape(1, 3)

    P_obj = (P_base.astype(np.float64) - t) @ R
    bound = (half_extents.astype(np.float64) + float(margin)).reshape(1, 3)
    keep = (np.abs(P_obj) <= bound).all(axis=1)
    keep_rate = float(np.mean(keep)) if keep.size > 0 else 0.0
    return P_base[keep], C_rgb[keep], keep_rate

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

def main():
    global STOP
    os.makedirs(DEBUG_DIR, exist_ok=True)

    # 1) extrinsics
    T_base_from_cam = load_T_base_from_cam_for_all(CAMS)
    T_base_from_front = T_base_from_cam[FRONT_CAM].astype(np.float64)

    # 2) mesh
    if not os.path.exists(MESH_FILE):
        raise FileNotFoundError(f"MESH_FILE not found: {MESH_FILE}")
    mesh = trimesh.load(MESH_FILE)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    half_extents = (extents.astype(np.float64) * 0.5).reshape(3,)

    # 3) FoundationPose
    from estimater import (
        set_logging_format, set_seed,
        ScorePredictor, PoseRefinePredictor, FoundationPose,
    )
    set_logging_format()
    set_seed(0)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()

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

    print("[INIT] Headless pipeline ready.")
    print(f"[ZMQ] IN  {ZMQ_ADDR}  ({ZMQ_MODE})  topic={TOPIC_PREFIX}<cam>")
    print(f"[ZMQ] OUT {ZMQ_OUT_ADDR} ({ZMQ_OUT_MODE}) topics: pcd_refined/base, pose/front_cam, pose/base")
    print(f"[AUTO] register={AUTO_REGISTER} cooldown={AUTO_REGISTER_COOLDOWN_SEC}s min_mask_pix={AUTO_REGISTER_MIN_MASK_PIX}")
    print(f"[GATE] center={CENTER_GATE_ENABLE} r={CENTER_GATE_RADIUS_M:.3f} hist={CENTER_HIST_LEN} jump={CENTER_JUMP_MAX_M:.3f}")
    print(f"[GATE] box={POSE_BOX_GATE_ENABLE} margin={POSE_BOX_GATE_MARGIN_M:.3f} half_extents={half_extents.tolist()}")

    # 4) ZMQ sockets
    ctx = zmq.Context()
    sub, poller = make_sub_socket(ctx)
    pub = make_pub_socket(ctx)

    latest: Dict[str, dict] = {
        cam: {"t_recv": 0.0, "ts": 0.0, "meta": None, "mask_png": None, "depth_png": None, "rgb_jpg": None}
        for cam in CAMS
    }

    # pose state
    tracking = False
    pose = None
    frame_idx = 0
    last_register_try_t = 0.0

    # pose/base cache（给 gate 用）
    last_pose_base = None
    last_pose_base_time = 0.0

    # multiview HUD/log
    last_support_rates: Dict[str, float] = {}

    # gates
    gate = CenterGate(hist_len=CENTER_HIST_LEN, jump_max_m=CENTER_JUMP_MAX_M, radius_m=CENTER_GATE_RADIUS_M)

    render_dt = 1.0 / max(RENDER_HZ, 1e-6)
    t_last_fuse = 0.0

    try:
        while not STOP:
            now = time.time()

            # ---- drain messages ----
            msgs = drain_all_latest(sub, poller, timeout_ms=POLL_TIMEOUT_MS)
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
                if (now - float(front_item["t_recv"])) <= STALE_SEC:
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
                            mask_u8_pose = refine_mask_u8(mask_u8, MASK_OPEN_K, MASK_CLOSE_K, MASK_ERODE_ITER)
                        else:
                            mask_u8_pose = (mask_u8 > 0).astype(np.uint8) * 255

                        ob_mask = (mask_u8_pose > 0)
                        mask_pix = int(ob_mask.sum())
                        depth_m = np.ascontiguousarray(depth_u16.astype(np.float32) * np.float32(depth_scale), dtype=np.float32)
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                        frame_idx += 1

                        # auto register
                        if AUTO_REGISTER:
                            need_reg = (pose is None) or (not tracking and AUTO_REREGISTER_ON_LOST)
                            if need_reg and (mask_pix >= int(AUTO_REGISTER_MIN_MASK_PIX)):
                                if (now - last_register_try_t) >= float(AUTO_REGISTER_COOLDOWN_SEC):
                                    last_register_try_t = now
                                    try:
                                        print(f"[POSE] auto register... frame={frame_idx} mask_pix={mask_pix}")
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

                        # track
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

                        # publish pose
                        pose_cam = None
                        pose_base = None
                        if pose is not None:
                            try:
                                center_pose = pose @ np.linalg.inv(to_origin)  # object_in_front_cam
                                pose_cam = center_pose.astype(np.float64)
                                pose_base = (T_base_from_front @ pose_cam).astype(np.float64)  # object_in_base
                            except Exception:
                                pose_cam = None
                                pose_base = None

                        pub_pose(pub, ts=now, pose_cam=pose_cam, pose_base=pose_base, tracking=tracking)

                        if tracking and pose_base is not None:
                            last_pose_base = pose_base.copy()
                            last_pose_base_time = now

            # ---- fuse & publish refined pcd ----
            if (now - t_last_fuse) >= render_dt:
                t_last_fuse = now

                per_cam_pts: Dict[str, np.ndarray] = {}
                per_cam_cols: Dict[str, np.ndarray] = {}

                for cam in CAMS:
                    item = latest[cam]
                    if item["mask_png"] is None or item["depth_png"] is None or item["meta"] is None:
                        continue
                    if (now - float(item["t_recv"])) > STALE_SEC:
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
                    if SUPPORT_LOG and support_rates:
                        sr_txt = " | ".join([f"{k}:{support_rates[k]:.2f}" for k in sorted(support_rates.keys())])
                        print(f"[MV] support_rate  {sr_txt}")
                else:
                    for cam, P in per_cam_pts.items():
                        kept_pts.append(P)
                        kept_cols.append(per_cam_cols[cam])

                if len(kept_pts) > 0:
                    P = np.concatenate(kept_pts, axis=0)
                    C = np.concatenate(kept_cols, axis=0)

                    # ---- center gate ----
                    if CENTER_GATE_ENABLE:
                        c_candidate = None
                        if last_pose_base is not None and (now - float(last_pose_base_time)) <= float(POSE_CENTER_TTL_SEC):
                            try:
                                c_candidate = np.asarray(last_pose_base[:3, 3], dtype=np.float64).reshape(3,)
                            except Exception:
                                c_candidate = None
                        if c_candidate is None and P.shape[0] >= int(CENTER_FALLBACK_MIN_PTS):
                            c_candidate = robust_center_from_points(P)

                        gate.update(c_candidate)
                        P2, C2, _ = gate.filter_points(P, C)
                        if P2 is not None and P2.shape[0] >= int(CENTER_MIN_KEEP_PTS):
                            P, C = P2, C2
                        else:
                            P = np.zeros((0, 3), dtype=np.float64)
                            C = np.zeros((0, 3), dtype=np.float64)

                    # ---- pose box gate ----
                    if POSE_BOX_GATE_ENABLE and P.shape[0] > 0:
                        if last_pose_base is not None and (now - float(last_pose_base_time)) <= float(POSE_BOX_GATE_TTL_SEC):
                            P3, C3, keep_rate = box_gate_by_pose(
                                P_base=P,
                                C_rgb=C,
                                T_base_from_obj=last_pose_base,
                                half_extents=half_extents,
                                margin=float(POSE_BOX_GATE_MARGIN_M),
                            )
                            if (P3 is None) or (P3.shape[0] < int(POSE_BOX_MIN_KEEP_PTS)) or (keep_rate < float(POSE_BOX_MIN_KEEP_RATE)):
                                P = np.zeros((0, 3), dtype=np.float64)
                                C = np.zeros((0, 3), dtype=np.float64)
                            else:
                                P, C = P3, C3

                    # ---- thirdNN ----
                    if P.shape[0] > 0 and THIRDNN_REFINE_ENABLE and P.shape[0] >= max(THIRDNN_REFINE_MIN_POINTS, THIRDNN_K + 1):
                        dK = third_nn_distance_once(P, thr_k=THIRDNN_K)
                        keep = dK <= float(THIRDNN_REMOVE_IF_D3_GT_M)
                        P = P[keep]
                        C = C[keep]

                    # ---- publish ----
                    if P.shape[0] > 0:
                        pub_pcd_refined(pub, ts=now, P_base=P, C_rgb=C)

            else:
                time.sleep(0.001)

    finally:
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
        print("[EXIT] done.")

if __name__ == "__main__":
    main()
