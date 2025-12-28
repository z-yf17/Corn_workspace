#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import cv2
import zmq
import trimesh
import torch
import open3d as o3d

# ===================== Params =====================
MESH_FILE = "./demo_data/mugblack_002.obj"

ZMQ_BIND = "tcp://0.0.0.0:5556"
ZMQ_TOPIC = "segd"

EST_REFINE_ITER = 5
TRACK_REFINE_ITER = 2

DEBUG = 1
DEBUG_DIR = "./debug_live"

# ---- depth / render ----
DEPTH_TRUNC = 3.0
VOXEL_RENDER = 0.005
AXIS_SCALE = 0.1

# ---- ZMQ ----
POLL_TIMEOUT_MS = 1

# ---- 2D mask refine ----
MASK_REFINE_ENABLE_DEFAULT = True
MASK_MIN_AREA = 800
MASK_OPEN_K = 3
MASK_CLOSE_K = 5
MASK_ERODE_ITER = 1

# ---- register gate ----
MIN_MASK_PIX_FOR_REGISTER = 200

# ---- depth band filter ----
DEPTH_BAND_ENABLE_DEFAULT = True
DEPTH_BAND_P_LO = 10
DEPTH_BAND_P_HI = 90
DEPTH_BAND_PAD_M = 0.05
DEPTH_BAND_MIN_PIX = 80

# ---- 3D main cluster filter (optional) ----
CLUSTER_FILTER_ENABLE_DEFAULT = False
VOXEL_CLUSTER = 0.01
DBSCAN_EPS = 0.03
DBSCAN_MIN_POINTS = 20
CENTER_DIST_THRESH = 0.25
MIN_POINTS_FOR_CLUSTER = 200

# ---- click-to-normal on OpenCV ----
PICK_REQUIRE_SHIFT_DEFAULT = True   # True: 必须 Shift+左键 才触发（防误触）
PICK_WINDOW = 11                   # 邻域窗口大小(odd)
PICK_MIN_PTS = 50                  # 邻域最少点数
PICK_NORMAL_LEN = 0.06             # 法向量长度(米)，用于2D和3D显示

# ---- Open3D normal thickness (cylinder) ----
NORMAL_RADIUS_3D = 0.004           # Open3D里“加粗”
NORMAL_CYL_RESOLUTION = 24
# =================================================

# ===================== Torch float32 patch (fixed) =====================
torch.set_default_dtype(torch.float32)

_orig_from_numpy = torch.from_numpy
def _from_numpy_float32(x):
    t = _orig_from_numpy(x)
    return t.float() if t.dtype == torch.float64 else t
torch.from_numpy = _from_numpy_float32

_orig_tensor = torch.tensor
def _tensor_float32(*a, **k):
    t = _orig_tensor(*a, **k)
    return t.float() if t.dtype == torch.float64 else t
torch.tensor = _tensor_float32
# ======================================================================

from estimater import (
    set_logging_format, set_seed,
    ScorePredictor, PoseRefinePredictor, FoundationPose,
    draw_posed_3d_box, draw_xyz_axis,
)
import nvdiffrast.torch as dr


# ===================== Utils =====================
def decode_png16(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)

def decode_png8(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)

def decode_jpg(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)

def meta_to_K(meta):
    K = np.array([
        [meta["fx"], 0, meta["ppx"]],
        [0, meta["fy"], meta["ppy"]],
        [0, 0, 1]
    ], dtype=np.float32)
    return K, float(meta.get("depth_scale", 0.001))

def recv_latest(sub, poller):
    if not dict(poller.poll(POLL_TIMEOUT_MS)):
        return None
    latest = None
    while True:
        try:
            latest = sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
    return latest

def put_text(img, text, y, color=(0, 255, 0)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, lineType=cv2.LINE_AA)

def project_point(K, P):
    x, y, z = float(P[0]), float(P[1]), float(P[2])
    if z <= 1e-6:
        return None
    u = (x / z) * float(K[0, 0]) + float(K[0, 2])
    v = (y / z) * float(K[1, 1]) + float(K[1, 2])
    return int(round(u)), int(round(v))


# ===================== Post-processing =====================
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

def depth_band_from_mask(depth_u16, mask_bool, depth_scale,
                         p_lo=10, p_hi=90, pad=0.05, min_pix=80):
    z = depth_u16[mask_bool].astype(np.float32) * float(depth_scale)
    z = z[z > 0]
    if z.size < int(min_pix):
        return None
    lo, hi = np.percentile(z, [float(p_lo), float(p_hi)])
    lo = max(0.0, float(lo) - float(pad))
    hi = float(hi) + float(pad)
    if hi <= lo:
        return None
    return lo, hi

def backproject(depth_u16, mask_bool, K, scale, color_bgr=None, band=None):
    z = depth_u16.astype(np.float32) * float(scale)
    keep = mask_bool & (z > 0) & (z < float(DEPTH_TRUNC))
    if band is not None:
        lo, hi = band
        keep = keep & (z >= float(lo)) & (z <= float(hi))

    if not np.any(keep):
        return np.empty((0, 3), np.float64), np.empty((0, 3), np.float64)

    v, u = np.where(keep)
    z_keep = z[v, u].astype(np.float64)

    x = (u.astype(np.float64) - float(K[0, 2])) / float(K[0, 0]) * z_keep
    y = (v.astype(np.float64) - float(K[1, 2])) / float(K[1, 1]) * z_keep
    pts = np.stack([x, y, z_keep], axis=1)

    if color_bgr is not None and color_bgr.shape[:2] == depth_u16.shape:
        rgb = color_bgr[v, u, ::-1].astype(np.float64) / 255.0
    else:
        rgb = np.full((pts.shape[0], 3), 0.7, dtype=np.float64)

    finite = np.isfinite(pts).all(axis=1)
    return pts[finite], rgb[finite]

def main_cluster_center_dbscan(pts_full):
    if pts_full.shape[0] < MIN_POINTS_FOR_CLUSTER:
        return np.median(pts_full, axis=0), False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_full)
    pcd_ds = pcd.voxel_down_sample(VOXEL_CLUSTER)
    if len(pcd_ds.points) < DBSCAN_MIN_POINTS:
        return np.median(pts_full, axis=0), False

    labels = np.array(pcd_ds.cluster_dbscan(
        eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=False
    ))
    if labels.size == 0:
        return np.median(pts_full, axis=0), False

    valid = labels >= 0
    if not np.any(valid):
        return np.median(pts_full, axis=0), False

    lab = labels[valid]
    pts = np.asarray(pcd_ds.points)[valid]

    uniq, cnt = np.unique(lab, return_counts=True)
    largest = uniq[np.argmax(cnt)]
    main_pts = pts[lab == largest]
    if main_pts.shape[0] == 0:
        return np.median(pts_full, axis=0), False

    return main_pts.mean(axis=0), True


# ===================== Click normal (OpenCV) =====================
def estimate_normal_from_depth(depth_u16, keep_bool, K, scale, u, v,
                               win=11, min_pts=50):
    H, W = depth_u16.shape
    u = int(np.clip(u, 0, W - 1))
    v = int(np.clip(v, 0, H - 1))

    # 如果点击处无效：向周围找最近有效点
    if not keep_bool[v, u]:
        found = False
        for r in range(1, 15):
            v0, v1 = max(0, v - r), min(H, v + r + 1)
            u0, u1 = max(0, u - r), min(W, u + r + 1)
            ys, xs = np.where(keep_bool[v0:v1, u0:u1])
            if ys.size > 0:
                dy = ys - (v - v0)
                dx = xs - (u - u0)
                j = int(np.argmin(dx * dx + dy * dy))
                v = v0 + int(ys[j])
                u = u0 + int(xs[j])
                found = True
                break
        if not found:
            return None, None, "no valid depth/mask near click", (u, v)

    r = int(win) // 2
    v0, v1 = max(0, v - r), min(H, v + r + 1)
    u0, u1 = max(0, u - r), min(W, u + r + 1)

    kb = keep_bool[v0:v1, u0:u1]
    if not np.any(kb):
        return None, None, "empty neighborhood", (u, v)

    ys, xs = np.where(kb)
    uu = (u0 + xs).astype(np.float64)
    vv = (v0 + ys).astype(np.float64)
    z = depth_u16[v0:v1, u0:u1][ys, xs].astype(np.float64) * float(scale)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z
    pts = np.stack([x, y, z], axis=1)

    if pts.shape[0] < int(min_pts):
        return None, None, f"too few points ({pts.shape[0]})", (u, v)

    c = pts.mean(axis=0)
    X = pts - c
    cov = (X.T @ X) / max(1, (X.shape[0] - 1))
    w, V = np.linalg.eigh(cov)
    n = V[:, 0]

    z0 = float(depth_u16[v, u]) * float(scale)
    p = np.array([(u - cx) / fx * z0, (v - cy) / fy * z0, z0], dtype=np.float64)

    # 让法向量朝向相机（相机在原点附近）
    if float(np.dot(n, p)) > 0:
        n = -n
    n /= (np.linalg.norm(n) + 1e-12)

    planarity = float(w[0] / max(1e-12, (w[0] + w[1] + w[2])))
    return p, n, f"planarity={planarity:.3e}, N={pts.shape[0]}", (u, v)


# ===================== Pose direction auto-bind =====================
def bind_anchor_auto(T, p_cam, n_cam):
    """
    自动判断 T 是 obj->cam 还是 cam->obj，并返回绑定到“物体系”的 (p_obj, n_obj) 以及选择的方向。
    返回: (dir_str, p_obj, n_obj, err_obj2cam, err_cam2obj)
    dir_str: "obj2cam" or "cam2obj"
    """
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3].reshape(3)

    p_cam = np.asarray(p_cam, dtype=np.float64).reshape(3)
    n_cam = np.asarray(n_cam, dtype=np.float64).reshape(3)

    # 假设A：T 是 obj->cam
    p_obj_A = R.T @ (p_cam - t)
    p_cam_A = R @ p_obj_A + t
    errA = float(np.linalg.norm(p_cam_A - p_cam))
    n_obj_A = R.T @ n_cam
    n_obj_A = n_obj_A / (np.linalg.norm(n_obj_A) + 1e-12)

    # 假设B：T 是 cam->obj
    # p_obj = R p_cam + t, 反推 p_cam = R^T (p_obj - t)
    p_obj_B = R @ p_cam + t
    p_cam_B = R.T @ (p_obj_B - t)
    errB = float(np.linalg.norm(p_cam_B - p_cam))
    n_obj_B = R @ n_cam
    n_obj_B = n_obj_B / (np.linalg.norm(n_obj_B) + 1e-12)

    if errA <= errB:
        return "obj2cam", p_obj_A, n_obj_A, errA, errB
    else:
        return "cam2obj", p_obj_B, n_obj_B, errA, errB

def obj_to_cam(T, p_obj, n_obj, dir_str):
    """
    根据 dir_str 把物体系点/法向 变到相机系。
    """
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3].reshape(3)
    p_obj = np.asarray(p_obj, dtype=np.float64).reshape(3)
    n_obj = np.asarray(n_obj, dtype=np.float64).reshape(3)

    if dir_str == "obj2cam":
        p_cam = R @ p_obj + t
        n_cam = R @ n_obj
    else:  # "cam2obj"
        p_cam = R.T @ (p_obj - t)
        n_cam = R.T @ n_obj

    n_cam = n_cam / (np.linalg.norm(n_cam) + 1e-12)
    # 保持朝向相机
    if float(np.dot(n_cam, p_cam)) > 0.0:
        n_cam = -n_cam
    return p_cam, n_cam


# ===================== Open3D thick normal geometry =====================
def make_cylinder_between(p, q, radius=0.004, resolution=24):
    p = np.asarray(p, dtype=np.float64).reshape(3)
    q = np.asarray(q, dtype=np.float64).reshape(3)
    v = q - p
    L = float(np.linalg.norm(v))
    if L < 1e-9:
        return o3d.geometry.TriangleMesh()

    cyl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=float(radius), height=L, resolution=int(resolution), split=4
    )
    cyl.compute_vertex_normals()

    cyl.translate((0.0, 0.0, -L / 2.0))

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v_unit = v / L
    axis = np.cross(z_axis, v_unit)
    axis_norm = float(np.linalg.norm(axis))

    if axis_norm < 1e-9:
        if float(np.dot(z_axis, v_unit)) < 0.0:
            Rm = o3d.geometry.get_rotation_matrix_from_axis_angle(
                np.array([np.pi, 0.0, 0.0], dtype=np.float64)
            )
        else:
            Rm = np.eye(3)
    else:
        axis = axis / axis_norm
        angle = float(np.arctan2(axis_norm, np.dot(z_axis, v_unit)))
        Rm = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    cyl.rotate(Rm, center=(0.0, 0.0, 0.0))
    mid = (p + q) / 2.0
    cyl.translate(mid)
    return cyl


# ===================== Main =====================
def main():
    set_logging_format()
    set_seed(0)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    # ---------- mesh / bbox ----------
    mesh = trimesh.load(MESH_FILE)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2]).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices.astype(np.float32),
        model_normals=mesh.vertex_normals.astype(np.float32),
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=DEBUG_DIR,
        debug=DEBUG,
        glctx=glctx,
    )

    # ---------- ZMQ ----------
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.bind(ZMQ_BIND)
    sub.setsockopt_string(zmq.SUBSCRIBE, ZMQ_TOPIC)
    sub.setsockopt(zmq.RCVHWM, 1)
    try:
        sub.setsockopt(zmq.CONFLATE, 1)
    except Exception:
        pass
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    # ---------- Open3D ----------
    vis = o3d.visualization.Visualizer()
    vis.create_window("PointCloud (post-processed)", 960, 720)

    pcd = o3d.geometry.PointCloud()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis.add_geometry(axis)

    normal_mesh = o3d.geometry.TriangleMesh()
    normal_mesh.paint_uniform_color([1.0, 0.0, 0.0])
    vis.add_geometry(normal_mesh)

    def set_normal_mesh_empty():
        normal_mesh.vertices = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        normal_mesh.triangles = o3d.utility.Vector3iVector(np.zeros((0, 3), dtype=np.int32))
        normal_mesh.vertex_normals = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        normal_mesh.vertex_colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

    set_normal_mesh_empty()
    vis.update_geometry(normal_mesh)

    added = False
    view_init = False

    # ---------- OpenCV ----------
    cv2.namedWindow("live_pose", cv2.WINDOW_NORMAL)

    # ---------- runtime state ----------
    pose = None
    tracking = False
    need_register = False

    ctrl = {
        "mask_refine": MASK_REFINE_ENABLE_DEFAULT,
        "depth_band": DEPTH_BAND_ENABLE_DEFAULT,
        "cluster_filter": CLUSTER_FILTER_ENABLE_DEFAULT,

        "pick_require_shift": PICK_REQUIRE_SHIFT_DEFAULT,
        "pick_armed": False,
        "normal_show": True,
    }

    # 绑定：存“物体坐标系”的点/法向，并记录 pose 的方向
    anchor = {
        "valid": False,
        "dir": None,         # "obj2cam" or "cam2obj"
        "p_obj": None,
        "n_obj": None,
        "base_info": "",
        "bind_note": "",
        # 冻结用（pose丢失时）
        "p_cam_last": None,
        "n_cam_last": None,
    }

    # 当前帧用于显示（相机系）
    normal_draw = {
        "has": False,
        "p_cam": None,
        "q_cam": None,
        "p2d": (0, 0),
        "q2d": (0, 0),
        "text": "",
    }

    click = {"pending": False, "uv": (0, 0), "flags": 0}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["pending"] = True
            click["uv"] = (int(x), int(y))
            click["flags"] = int(flags)

    cv2.setMouseCallback("live_pose", on_mouse)

    def update_o3d_normal_mesh_from_draw():
        if ctrl["normal_show"] and normal_draw["has"] and normal_draw["p_cam"] is not None and normal_draw["q_cam"] is not None:
            cyl = make_cylinder_between(
                normal_draw["p_cam"], normal_draw["q_cam"],
                radius=NORMAL_RADIUS_3D,
                resolution=NORMAL_CYL_RESOLUTION
            )
            cyl.paint_uniform_color([1.0, 0.0, 0.0])
            cyl.compute_vertex_normals()

            normal_mesh.vertices = cyl.vertices
            normal_mesh.triangles = cyl.triangles
            normal_mesh.vertex_normals = cyl.vertex_normals
            normal_mesh.vertex_colors = cyl.vertex_colors
        else:
            set_normal_mesh_empty()

        vis.update_geometry(normal_mesh)

    try:
        while True:
            parts = recv_latest(sub, poller)
            if parts is None:
                cv2.waitKey(1)
                vis.poll_events()
                vis.update_renderer()
                continue

            if len(parts) < 5:
                continue

            raw_jpg = None
            overlay_jpg = None

            if len(parts) >= 7:
                _topic, ts, meta_b, mask_b, depth_b, raw_jpg, overlay_jpg = parts[:7]
            elif len(parts) == 6:
                _topic, ts, meta_b, mask_b, depth_b, raw_jpg = parts
            else:
                _topic, ts, meta_b, mask_b, depth_b = parts

            # ---- meta ----
            try:
                meta = json.loads(meta_b.decode("utf-8"))
                K, scale = meta_to_K(meta)
            except Exception:
                continue

            # ---- decode depth/mask ----
            depth = decode_png16(depth_b)
            mask_raw = decode_png8(mask_b)
            if depth is None or mask_raw is None:
                continue
            if depth.dtype != np.uint16:
                continue
            H, W = depth.shape
            if mask_raw.shape != (H, W):
                mask_raw = cv2.resize(mask_raw, (W, H), interpolation=cv2.INTER_NEAREST)

            # ---- decode rgb ----
            rgb_bgr = None
            if raw_jpg is not None:
                rgb_bgr = decode_jpg(raw_jpg)
            if rgb_bgr is None and overlay_jpg is not None:
                rgb_bgr = decode_jpg(overlay_jpg)
            if rgb_bgr is not None and rgb_bgr.shape[:2] != (H, W):
                rgb_bgr = cv2.resize(rgb_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

            # ---- mask refine ----
            if ctrl["mask_refine"]:
                mask_u8 = refine_mask_u8(mask_raw, MASK_MIN_AREA, MASK_OPEN_K, MASK_CLOSE_K, MASK_ERODE_ITER)
            else:
                mask_u8 = (mask_raw > 0).astype(np.uint8) * 255
            mask_bool = mask_u8 > 0

            # ---- depth band ----
            band = None
            if ctrl["depth_band"] and np.any(mask_bool):
                band = depth_band_from_mask(
                    depth_u16=depth,
                    mask_bool=mask_bool,
                    depth_scale=scale,
                    p_lo=DEPTH_BAND_P_LO,
                    p_hi=DEPTH_BAND_P_HI,
                    pad=DEPTH_BAND_PAD_M,
                    min_pix=DEPTH_BAND_MIN_PIX
                )

            # ---- keep_bool ----
            z_m = depth.astype(np.float32) * float(scale)
            keep_bool = mask_bool & (z_m > 0) & (z_m < float(DEPTH_TRUNC))
            if band is not None:
                lo, hi = band
                keep_bool = keep_bool & (z_m >= float(lo)) & (z_m <= float(hi))

            # ---------- Pose ----------
            mask_pose_bool = keep_bool.copy()
            if rgb_bgr is not None:
                rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                depth_m = depth.astype(np.float32) * float(scale)

                if need_register and mask_pose_bool.sum() > MIN_MASK_PIX_FOR_REGISTER:
                    pose = est.register(K, rgb_rgb, depth_m, mask_pose_bool, EST_REFINE_ITER)
                    tracking = True
                    need_register = False

                if tracking and pose is not None:
                    pose = est.track_one(rgb_rgb, depth_m, K, TRACK_REFINE_ITER)

            # ---------- Point Cloud ----------
            pts, colors = backproject(depth, mask_bool, K, scale, color_bgr=rgb_bgr, band=band)
            if ctrl["cluster_filter"] and pts.shape[0] > 0:
                center, _ = main_cluster_center_dbscan(pts)
                d = np.linalg.norm(pts - center.reshape(1, 3), axis=1)
                keep = d < CENTER_DIST_THRESH
                pts, colors = pts[keep], colors[keep]

            if pts.shape[0] > 0:
                pcd_tmp = o3d.geometry.PointCloud()
                pcd_tmp.points = o3d.utility.Vector3dVector(pts)
                pcd_tmp.colors = o3d.utility.Vector3dVector(colors)
                pcd_ds = pcd_tmp.voxel_down_sample(VOXEL_RENDER)

                pcd.points = pcd_ds.points
                pcd.colors = pcd_ds.colors

                if not added:
                    vis.add_geometry(pcd)
                    added = True
                else:
                    vis.update_geometry(pcd)

                if not view_init and len(pcd.points) > 0:
                    vc = vis.get_view_control()
                    bbox_o3d = pcd.get_axis_aligned_bounding_box()
                    vc.set_lookat(bbox_o3d.get_center())
                    vc.set_front([0, 0, -1])
                    vc.set_up([0, -1, 0])
                    vc.set_zoom(0.6)
                    view_init = True

            # ---------- Click: bind anchor ----------
            if click["pending"]:
                click["pending"] = False
                u, v = click["uv"]
                flags = click["flags"]

                shift_down = (flags & cv2.EVENT_FLAG_SHIFTKEY) != 0
                allowed = (ctrl["pick_armed"]) or (shift_down and ctrl["pick_require_shift"]) or (not ctrl["pick_require_shift"])
                if ctrl["pick_armed"] and allowed:
                    ctrl["pick_armed"] = False

                if allowed:
                    p_cam_click, n_cam_click, info, _ = estimate_normal_from_depth(
                        depth_u16=depth,
                        keep_bool=keep_bool,
                        K=K,
                        scale=scale,
                        u=u, v=v,
                        win=PICK_WINDOW,
                        min_pts=PICK_MIN_PTS
                    )

                    if p_cam_click is not None:
                        ctrl["normal_show"] = True

                        if pose is None:
                            # 没有 pose：无法绑定到物体，只能固定在相机系
                            anchor["valid"] = True
                            anchor["dir"] = None
                            anchor["p_obj"] = None
                            anchor["n_obj"] = None
                            anchor["base_info"] = info
                            anchor["bind_note"] = "NO_POSE (fixed in camera)"
                            anchor["p_cam_last"] = p_cam_click
                            anchor["n_cam_last"] = n_cam_click
                        else:
                            dsel, p_obj, n_obj, errA, errB = bind_anchor_auto(pose, p_cam_click, n_cam_click)
                            anchor["valid"] = True
                            anchor["dir"] = dsel
                            anchor["p_obj"] = p_obj
                            anchor["n_obj"] = n_obj
                            anchor["base_info"] = info
                            anchor["bind_note"] = f"BOUND[{dsel}] errA={errA:.3e} errB={errB:.3e}"
                            # 冻结备份
                            anchor["p_cam_last"] = p_cam_click
                            anchor["n_cam_last"] = n_cam_click
                    else:
                        normal_draw["has"] = False
                        normal_draw["text"] = f"pick failed: {info}"

            # ---------- Per-frame: compute followed normal ----------
            normal_draw["has"] = False
            normal_draw["p_cam"] = None
            normal_draw["q_cam"] = None
            normal_draw["text"] = ""

            if anchor["valid"]:
                if pose is not None and anchor["dir"] is not None and anchor["p_obj"] is not None and anchor["n_obj"] is not None:
                    # 正常跟随
                    p_cam, n_cam = obj_to_cam(pose, anchor["p_obj"], anchor["n_obj"], anchor["dir"])
                    anchor["p_cam_last"] = p_cam
                    anchor["n_cam_last"] = n_cam
                    status = "FOLLOW"
                else:
                    # pose 丢失 / 或点击时没有 pose：冻结
                    if anchor["p_cam_last"] is not None and anchor["n_cam_last"] is not None:
                        p_cam = anchor["p_cam_last"]
                        n_cam = anchor["n_cam_last"]
                        status = "FROZEN"
                    else:
                        p_cam = None
                        n_cam = None
                        status = "NONE"

                if p_cam is not None and n_cam is not None:
                    q_cam = p_cam + n_cam * float(PICK_NORMAL_LEN)
                    p2d = project_point(K, p_cam)
                    q2d = project_point(K, q_cam)
                    if p2d is not None and q2d is not None:
                        normal_draw["has"] = True
                        normal_draw["p_cam"] = p_cam
                        normal_draw["q_cam"] = q_cam
                        normal_draw["p2d"] = p2d
                        normal_draw["q2d"] = q2d
                        normal_draw["text"] = f"{status} | {anchor['bind_note']} | {anchor['base_info']}"
                    else:
                        normal_draw["text"] = f"{status} | projection failed"

            # ---------- Update Open3D normal mesh ----------
            update_o3d_normal_mesh_from_draw()

            vis.poll_events()
            vis.update_renderer()

            # ---------- Visualization 2D ----------
            vis_bgr = rgb_bgr.copy() if rgb_bgr is not None else np.zeros((H, W, 3), dtype=np.uint8)

            if pose is not None and rgb_bgr is not None:
                center_pose = pose @ np.linalg.inv(to_origin)
                vis_rgb = draw_posed_3d_box(K, cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB), center_pose, bbox)
                vis_rgb = draw_xyz_axis(vis_rgb, center_pose, AXIS_SCALE, K)
                vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

            band_str = "band" if (band is not None) else "noBand"
            pick_str = "PICK: Shift+Click" if ctrl["pick_require_shift"] else "PICK: Click"
            if ctrl["pick_armed"]:
                pick_str = "PICK ARMED: Click now"

            put_text(
                vis_bgr,
                f"M{int(ctrl['mask_refine'])} B{int(ctrl['depth_band'])} C{int(ctrl['cluster_filter'])} {band_str} | "
                f"reg[S] | {pick_str} | V=show/hide | H=clear | T=shiftReq | Q/ESC",
                30
            )
            put_text(vis_bgr, f"pose={'Y' if pose is not None else 'N'} track={int(tracking)}", 60, color=(255, 255, 255))

            if ctrl["normal_show"] and normal_draw["has"]:
                cv2.circle(vis_bgr, normal_draw["p2d"], 4, (0, 0, 255), -1)
                cv2.arrowedLine(vis_bgr, normal_draw["p2d"], normal_draw["q2d"], (0, 0, 255), 2, tipLength=0.2)
                put_text(vis_bgr, normal_draw["text"], 90, color=(0, 0, 255))
            else:
                if normal_draw["text"]:
                    put_text(vis_bgr, normal_draw["text"], 90, color=(0, 0, 255))

            cv2.imshow("live_pose", vis_bgr)

            # ---------- Keys ----------
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break

            if key in (ord('s'), ord('S')):
                need_register = True
                tracking = False
                pose = None

            if key in (ord('m'), ord('M')):
                ctrl["mask_refine"] = not ctrl["mask_refine"]
            if key in (ord('b'), ord('B')):
                ctrl["depth_band"] = not ctrl["depth_band"]
            if key in (ord('c'), ord('C')):
                ctrl["cluster_filter"] = not ctrl["cluster_filter"]

            if key in (ord('p'), ord('P')):
                ctrl["pick_armed"] = not ctrl["pick_armed"]

            if key in (ord('v'), ord('V')):
                ctrl["normal_show"] = not ctrl["normal_show"]

            if key in (ord('h'), ord('H')):
                anchor["valid"] = False
                anchor["dir"] = None
                anchor["p_obj"] = None
                anchor["n_obj"] = None
                anchor["base_info"] = ""
                anchor["bind_note"] = ""
                anchor["p_cam_last"] = None
                anchor["n_cam_last"] = None

                normal_draw["has"] = False
                normal_draw["text"] = ""
                ctrl["normal_show"] = False

            if key in (ord('t'), ord('T')):
                ctrl["pick_require_shift"] = not ctrl["pick_require_shift"]

    finally:
        cv2.destroyAllWindows()
        vis.destroy_window()
        ctx.term()


if __name__ == "__main__":
    main()
