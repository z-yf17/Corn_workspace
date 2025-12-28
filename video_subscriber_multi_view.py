#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import argparse
from typing import Dict, Optional, Tuple, List

import numpy as np
import cv2
import zmq
import open3d as o3d


# ------------------ Defaults (same spirit as your reference) ------------------
DEPTH_TRUNC = 3.0        # m
DEPTH_VIS_MIN = 0.15     # m
DEPTH_VIS_MAX = 3.0      # m
VOXEL_RENDER = 0.005     # m

# ---- 2D mask refine ----
MASK_REFINE_ENABLE_DEFAULT = True
MASK_MIN_AREA = 800
MASK_OPEN_K = 3
MASK_CLOSE_K = 5
MASK_ERODE_ITER = 1

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

CAM_SET = ["front", "left", "right"]


# ------------------ Helpers (from your reference) ------------------
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


def backproject_masked_points(depth_u16, mask_bool, intr, depth_scale,
                              depth_trunc=3.0, color_bgr=None):
    H, W = depth_u16.shape

    if mask_bool.shape != (H, W):
        mask_bool = cv2.resize(mask_bool.astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST).astype(bool)

    z = depth_u16.astype(np.float32) * float(depth_scale)
    keep = mask_bool & (z > 0) & (z < float(depth_trunc))

    if not np.any(keep):
        return np.empty((0, 3), np.float64), np.empty((0, 3), np.float64)

    v, u = np.where(keep)
    z_keep = z[v, u].astype(np.float64)

    fx = float(intr["fx"]); fy = float(intr["fy"])
    ppx = float(intr["ppx"]); ppy = float(intr["ppy"])

    if fx <= 1e-9 or fy <= 1e-9:
        return np.empty((0, 3), np.float64), np.empty((0, 3), np.float64)

    x = (u.astype(np.float64) - ppx) / fx * z_keep
    y = (v.astype(np.float64) - ppy) / fy * z_keep
    pts = np.stack([x, y, z_keep], axis=1)

    if color_bgr is not None and color_bgr.shape[:2] == (H, W):
        rgb = color_bgr[v, u, ::-1].astype(np.float64) / 255.0
    else:
        rgb = np.full((pts.shape[0], 3), 0.7, dtype=np.float64)

    finite = np.isfinite(pts).all(axis=1)
    return pts[finite], rgb[finite]


def depth_to_colormap(depth_u16, depth_scale, dmin=0.15, dmax=3.0):
    depth_m = depth_u16.astype(np.float32) * float(depth_scale)
    valid = depth_m > 0
    vis = np.clip(depth_m, dmin, dmax)
    vis_u8 = ((vis - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
    vis_u8[~valid] = 0
    depth_col = cv2.applyColorMap(vis_u8, cv2.COLORMAP_JET)
    return depth_col, depth_m


def main_cluster_center_dbscan(pts_full):
    if pts_full.shape[0] < MIN_POINTS_FOR_CLUSTER:
        return np.median(pts_full, axis=0), False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_full)
    pcd_ds = pcd.voxel_down_sample(VOXEL_CLUSTER)
    if len(pcd_ds.points) < DBSCAN_MIN_POINTS:
        return np.median(pts_full, axis=0), False

    labels = np.array(pcd_ds.cluster_dbscan(eps=DBSCAN_EPS,
                                            min_points=DBSCAN_MIN_POINTS,
                                            print_progress=False))
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


# ------------------ ZMQ / UI helpers ------------------
def parse_cam_from_topic(topic_b: bytes) -> Optional[str]:
    try:
        t = topic_b.decode("utf-8", errors="ignore")
    except Exception:
        return None
    # expect: segd/front
    if "/" in t:
        prefix, cam = t.split("/", 1)
        if prefix.strip() == "segd":
            cam = cam.strip().lower()
            return cam
    if t == "segd":
        return "front"
    return None


def to_bgr_gray(mask_u8: np.ndarray) -> np.ndarray:
    if mask_u8 is None:
        return None
    if mask_u8.ndim == 2:
        return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    return mask_u8


def make_grid(tiles: List[Optional[np.ndarray]], labels: List[str], tile_w: int, tile_h: int,
              active_idx: Optional[int] = None) -> np.ndarray:
    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    out_tiles = []
    for i, (img, lab) in enumerate(zip(tiles, labels)):
        if img is None:
            canvas = blank.copy()
            cv2.putText(canvas, f"{lab} (NO DATA)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            canvas = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            cv2.putText(canvas, lab, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        # highlight active cam tile
        if active_idx is not None and i == active_idx:
            cv2.rectangle(canvas, (2, 2), (tile_w - 3, tile_h - 3), (0, 255, 255), 3)

        out_tiles.append(canvas)

    return np.hstack(out_tiles)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bind", default="0.0.0.0", help="viewer bind ip (default 0.0.0.0)")
    p.add_argument("--port", type=int, default=5556, help="viewer bind port (default 5556)")
    p.add_argument("--cams", default="front,left,right", help="cams to visualize: front,left,right or any subset")

    p.add_argument("--tile-w", type=int, default=480)
    p.add_argument("--tile-h", type=int, default=360)
    p.add_argument("--print-fps", action="store_true", default=True)

    # zmq
    p.add_argument("--rcvhwm", type=int, default=2000, help="RCVHWM (default 2000). Higher avoids drops.")
    return p.parse_args()


def main():
    args = parse_args()
    cams = [c.strip().lower() for c in args.cams.split(",") if c.strip()]
    cams = [c for c in cams if c in CAM_SET]
    if not cams:
        cams = CAM_SET.copy()

    # ---------- ZMQ SUB (bind like your reference) ----------
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.bind(f"tcp://{args.bind}:{args.port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "segd")
    sub.setsockopt(zmq.RCVHWM, int(args.rcvhwm))
    # ❌ 多路不要用 CONFLATE，否则会随机只剩一路

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    print(f"[VIEW] bind tcp://{args.bind}:{args.port}, subscribe 'segd'")
    print("[VIEW] expect: [topic=segd/<cam>, ts, meta, mask_png, depth_png16, raw_rgb_jpg, overlay_jpg] (len 5/6/7 supported)")
    print(f"[VIEW] cams={cams}")
    print("[VIEW] Keys (Open3D): Z cycle cam | 7/8/9 select front/left/right | M/B/C toggles | Q quit")

    # ---------- Open3D ----------
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Segmented PointCloud (active cam)", width=960, height=720)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    axis_added = True
    vis.add_geometry(axis)

    pcd = o3d.geometry.PointCloud()
    pcd_added = False

    ctrl = {
        "rot_step": 15.0,
        "zoom_step": 0.08,
        "quit": False,
        "auto_framed_once": False,

        "mask_refine": MASK_REFINE_ENABLE_DEFAULT,
        "depth_band": DEPTH_BAND_ENABLE_DEFAULT,
        "cluster_filter": CLUSTER_FILTER_ENABLE_DEFAULT,
    }

    active_cam_idx = 0
    active_cam = cams[active_cam_idx]

    def _reg(key, cb):
        if hasattr(vis, "register_key_action_callback"):
            vis.register_key_action_callback(key, cb)
        else:
            vis.register_key_callback(key, lambda v: cb(v, None, None))

    def _pressed(action):
        return (action is None) or (action != 0)

    # ---- camera callbacks ----
    def yaw_left(v, action=None, mods=None):
        if not _pressed(action): return False
        v.get_view_control().rotate(+ctrl["rot_step"], 0.0); return False

    def yaw_right(v, action=None, mods=None):
        if not _pressed(action): return False
        v.get_view_control().rotate(-ctrl["rot_step"], 0.0); return False

    def pitch_up(v, action=None, mods=None):
        if not _pressed(action): return False
        v.get_view_control().rotate(0.0, +ctrl["rot_step"]); return False

    def pitch_down(v, action=None, mods=None):
        if not _pressed(action): return False
        v.get_view_control().rotate(0.0, -ctrl["rot_step"]); return False

    def zoom_in(v, action=None, mods=None):
        if not _pressed(action): return False
        vc = v.get_view_control()
        vc.set_zoom(max(0.02, vc.get_zoom() - ctrl["zoom_step"])); return False

    def zoom_out(v, action=None, mods=None):
        if not _pressed(action): return False
        vc = v.get_view_control()
        vc.set_zoom(min(3.0, vc.get_zoom() + ctrl["zoom_step"])); return False

    def frame_pointcloud(v, action=None, mods=None):
        if not _pressed(action): return False
        if len(pcd.points) == 0: return False
        v.reset_view_point(True); return False

    def toggle_axis(v, action=None, mods=None):
        if not _pressed(action): return False
        nonlocal axis_added
        if axis_added:
            vis.remove_geometry(axis, reset_bounding_box=False); axis_added = False
        else:
            vis.add_geometry(axis, reset_bounding_box=False); axis_added = True
        return False

    # ---- steps ----
    def step_small(v, action=None, mods=None):
        if not _pressed(action): return False
        ctrl["rot_step"] = 6.0; print("[O3D] rot_step = 6"); return False

    def step_medium(v, action=None, mods=None):
        if not _pressed(action): return False
        ctrl["rot_step"] = 15.0; print("[O3D] rot_step = 15"); return False

    def step_large(v, action=None, mods=None):
        if not _pressed(action): return False
        ctrl["rot_step"] = 35.0; print("[O3D] rot_step = 35"); return False

    # ---- post toggles ----
    def toggle_mask_refine(v, action=None, mods=None):
        if not _pressed(action): return False
        ctrl["mask_refine"] = not ctrl["mask_refine"]
        print(f"[POST] mask_refine = {ctrl['mask_refine']}"); return False

    def toggle_depth_band(v, action=None, mods=None):
        if not _pressed(action): return False
        ctrl["depth_band"] = not ctrl["depth_band"]
        print(f"[POST] depth_band = {ctrl['depth_band']}"); return False

    def toggle_cluster_filter(v, action=None, mods=None):
        if not _pressed(action): return False
        ctrl["cluster_filter"] = not ctrl["cluster_filter"]
        print(f"[POST] cluster_filter = {ctrl['cluster_filter']}"); return False

    def quit_o3d(v, action=None, mods=None):
        if not _pressed(action): return False
        ctrl["quit"] = True; return False

    # ---- cam select ----
    def cycle_cam(v, action=None, mods=None):
        if not _pressed(action): return False
        nonlocal active_cam_idx, active_cam
        active_cam_idx = (active_cam_idx + 1) % len(cams)
        active_cam = cams[active_cam_idx]
        ctrl["auto_framed_once"] = False
        print(f"[VIEW] active_cam = {active_cam}")
        return False

    def select_front(v, action=None, mods=None):
        if not _pressed(action): return False
        nonlocal active_cam_idx, active_cam
        if "front" in cams:
            active_cam_idx = cams.index("front")
            active_cam = "front"
            ctrl["auto_framed_once"] = False
            print(f"[VIEW] active_cam = {active_cam}")
        return False

    def select_left(v, action=None, mods=None):
        if not _pressed(action): return False
        nonlocal active_cam_idx, active_cam
        if "left" in cams:
            active_cam_idx = cams.index("left")
            active_cam = "left"
            ctrl["auto_framed_once"] = False
            print(f"[VIEW] active_cam = {active_cam}")
        return False

    def select_right(v, action=None, mods=None):
        if not _pressed(action): return False
        nonlocal active_cam_idx, active_cam
        if "right" in cams:
            active_cam_idx = cams.index("right")
            active_cam = "right"
            ctrl["auto_framed_once"] = False
            print(f"[VIEW] active_cam = {active_cam}")
        return False

    # ---- register keys ----
    _reg(ord("J"), yaw_left)
    _reg(ord("L"), yaw_right)
    _reg(ord("I"), pitch_up)
    _reg(ord("K"), pitch_down)
    _reg(ord("+"), zoom_in)
    _reg(ord("="), zoom_in)
    _reg(ord("-"), zoom_out)
    _reg(ord("F"), frame_pointcloud)
    _reg(ord("X"), toggle_axis)
    _reg(ord("1"), step_small)
    _reg(ord("2"), step_medium)
    _reg(ord("3"), step_large)
    _reg(ord("M"), toggle_mask_refine)
    _reg(ord("B"), toggle_depth_band)
    _reg(ord("C"), toggle_cluster_filter)
    _reg(ord("Q"), quit_o3d)

    _reg(ord("Z"), cycle_cam)
    _reg(ord("7"), select_front)
    _reg(ord("8"), select_left)
    _reg(ord("9"), select_right)

    # ---------- OpenCV windows (3-cam grids) ----------
    cv2.namedWindow("RAW Grid (from segd)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Overlay Grid", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask(raw) Grid", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask(refined) Grid", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Grid", cv2.WINDOW_NORMAL)

    # ---------- Per-cam latest decoded cache ----------
    latest: Dict[str, Dict] = {c: {} for c in cams}

    # FPS stats per cam
    recv_cnt = {c: 0 for c in cams}
    fps_val = {c: 0.0 for c in cams}
    drop_cnt = {c: 0 for c in cams}  # (we don't really know drops, but keep placeholder)
    last_fps_t = time.time()

    # viewer loop stats
    last_print = time.time()

    try:
        while True:
            if ctrl["quit"]:
                break

            # poll to keep UI responsive
            socks = dict(poller.poll(10))
            if sub in socks:
                # drain quickly, keep latest per cam
                while True:
                    try:
                        parts = sub.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break

                    if len(parts) not in (5, 6, 7):
                        continue

                    raw_jpg = None
                    overlay_jpg = None

                    if len(parts) == 7:
                        topic_b, ts_b, meta_b, mask_png, depth_png, raw_jpg, overlay_jpg = parts
                    elif len(parts) == 6:
                        topic_b, ts_b, meta_b, mask_png, depth_png, raw_jpg = parts
                        overlay_jpg = None
                    else:
                        topic_b, ts_b, meta_b, mask_png, depth_png = parts
                        raw_jpg = None
                        overlay_jpg = None

                    cam = parse_cam_from_topic(topic_b)

                    # also allow meta["cam"] to override (robust)
                    meta = {}
                    try:
                        meta = json.loads(meta_b.decode("utf-8"))
                        if not isinstance(meta, dict):
                            meta = {}
                    except Exception:
                        meta = {}

                    cam_meta = str(meta.get("cam", "")).strip().lower()
                    if cam_meta in cams:
                        cam = cam_meta

                    if cam not in cams:
                        continue

                    # decode depth
                    depth_u16 = cv2.imdecode(np.frombuffer(depth_png, np.uint8), cv2.IMREAD_UNCHANGED)
                    if depth_u16 is None or depth_u16.dtype != np.uint16:
                        continue

                    H, W = depth_u16.shape

                    # decode mask
                    mask_u8_raw = cv2.imdecode(np.frombuffer(mask_png, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if mask_u8_raw is None:
                        continue
                    if mask_u8_raw.shape != (H, W):
                        mask_u8_raw = cv2.resize(mask_u8_raw, (W, H), interpolation=cv2.INTER_NEAREST)

                    # decode raw/overlay
                    raw_bgr = None
                    if raw_jpg is not None:
                        raw_bgr = cv2.imdecode(np.frombuffer(raw_jpg, np.uint8), cv2.IMREAD_COLOR)
                        if raw_bgr is not None and raw_bgr.shape[:2] != (H, W):
                            raw_bgr = cv2.resize(raw_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

                    overlay_bgr = None
                    if overlay_jpg is not None:
                        overlay_bgr = cv2.imdecode(np.frombuffer(overlay_jpg, np.uint8), cv2.IMREAD_COLOR)
                        if overlay_bgr is not None and overlay_bgr.shape[:2] != (H, W):
                            overlay_bgr = cv2.resize(overlay_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

                    # latency
                    try:
                        ts_cam = float(ts_b.decode("ascii"))
                        latency_ms = (time.time() - ts_cam) * 1000.0
                    except Exception:
                        latency_ms = -1.0

                    # refined mask (apply toggle to all cams consistently)
                    if ctrl["mask_refine"]:
                        mask_u8_ref = refine_mask_u8(mask_u8_raw, MASK_MIN_AREA, MASK_OPEN_K, MASK_CLOSE_K, MASK_ERODE_ITER)
                    else:
                        mask_u8_ref = (mask_u8_raw > 0).astype(np.uint8) * 255

                    # depth colormap
                    depth_scale = float(meta.get("depth_scale", 0.001))
                    depth_col, depth_m = depth_to_colormap(depth_u16, depth_scale, dmin=DEPTH_VIS_MIN, dmax=DEPTH_VIS_MAX)

                    latest[cam] = {
                        "ts_b": ts_b,
                        "meta": meta,
                        "depth_u16": depth_u16,
                        "mask_raw": mask_u8_raw,
                        "mask_ref": mask_u8_ref,
                        "raw_bgr": raw_bgr,
                        "overlay_bgr": overlay_bgr,
                        "depth_col": depth_col,
                        "lat_ms": latency_ms,
                        "shape": (H, W),
                    }
                    recv_cnt[cam] += 1

            # ---- FPS update 1Hz ----
            now = time.time()
            if args.print_fps and (now - last_fps_t >= 1.0):
                dt = now - last_fps_t
                for c in cams:
                    fps_val[c] = recv_cnt[c] / dt
                    recv_cnt[c] = 0
                msg = " | ".join([f"{c}:{fps_val[c]:.1f}fps" for c in cams])
                print(f"[VIEW] {msg}")
                last_fps_t = now

            # ---- Build OpenCV grids ----
            raw_tiles = []
            overlay_tiles = []
            mask_raw_tiles = []
            mask_ref_tiles = []
            depth_tiles = []
            labels = []

            for i, cam in enumerate(cams):
                data = latest.get(cam, {})
                raw_bgr = data.get("raw_bgr", None)
                overlay_bgr = data.get("overlay_bgr", None)
                mask_raw = data.get("mask_raw", None)
                mask_ref = data.get("mask_ref", None)
                depth_col = data.get("depth_col", None)
                lat = data.get("lat_ms", -1.0)

                # prefer raw, fallback overlay, else None
                raw_show = raw_bgr if raw_bgr is not None else overlay_bgr
                ov_show = overlay_bgr if overlay_bgr is not None else raw_bgr

                # add overlay text on tiles
                def _annot(img, text):
                    if img is None:
                        return None
                    out = img.copy()
                    cv2.putText(out, text, (10, out.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    return out

                tag = f"{cam} | {fps_val.get(cam,0.0):.1f}fps | {lat:.1f}ms"
                raw_tiles.append(_annot(raw_show, tag))
                overlay_tiles.append(_annot(ov_show, tag))
                mask_raw_tiles.append(_annot(to_bgr_gray(mask_raw), tag))
                mask_ref_tiles.append(_annot(to_bgr_gray(mask_ref), tag))
                depth_tiles.append(_annot(depth_col, tag))
                labels.append(cam)

            active_idx = cams.index(active_cam) if active_cam in cams else None

            cv2.imshow("RAW Grid (from segd)",
                       make_grid(raw_tiles, labels, args.tile_w, args.tile_h, active_idx))
            cv2.imshow("Overlay Grid",
                       make_grid(overlay_tiles, labels, args.tile_w, args.tile_h, active_idx))
            cv2.imshow("Mask(raw) Grid",
                       make_grid(mask_raw_tiles, labels, args.tile_w, args.tile_h, active_idx))
            cv2.imshow("Mask(refined) Grid",
                       make_grid(mask_ref_tiles, labels, args.tile_w, args.tile_h, active_idx))
            cv2.imshow("Depth Grid",
                       make_grid(depth_tiles, labels, args.tile_w, args.tile_h, active_idx))

            # ---- Update Open3D with active cam ----
            data = latest.get(active_cam, {})
            if data:
                meta = data.get("meta", {})
                depth_u16 = data.get("depth_u16", None)
                mask_u8 = data.get("mask_ref", None)
                raw_bgr = data.get("raw_bgr", None)
                overlay_bgr = data.get("overlay_bgr", None)

                if depth_u16 is not None and mask_u8 is not None:
                    depth_scale = float(meta.get("depth_scale", 0.001))
                    intr = {
                        "fx": float(meta.get("fx", 0.0)),
                        "fy": float(meta.get("fy", 0.0)),
                        "ppx": float(meta.get("ppx", 0.0)),
                        "ppy": float(meta.get("ppy", 0.0)),
                    }

                    mask_bool = (mask_u8 > 0)

                    # depth band (active cam only)
                    band = None
                    if ctrl["depth_band"] and np.any(mask_bool):
                        band = depth_band_from_mask(depth_u16, mask_bool, depth_scale,
                                                    p_lo=DEPTH_BAND_P_LO, p_hi=DEPTH_BAND_P_HI,
                                                    pad=DEPTH_BAND_PAD_M, min_pix=DEPTH_BAND_MIN_PIX)

                    # color source
                    color_src = raw_bgr if raw_bgr is not None else overlay_bgr

                    pts, rgb = backproject_masked_points(
                        depth_u16=depth_u16,
                        mask_bool=mask_bool,
                        intr=intr,
                        depth_scale=depth_scale,
                        depth_trunc=DEPTH_TRUNC,
                        color_bgr=color_src,
                    )

                    if band is not None and pts.shape[0] > 0:
                        lo, hi = band
                        keep = (pts[:, 2] >= lo) & (pts[:, 2] <= hi)
                        pts, rgb = pts[keep], rgb[keep]

                    clustered = False
                    if ctrl["cluster_filter"] and pts.shape[0] > 0:
                        center, clustered = main_cluster_center_dbscan(pts)
                        d = np.linalg.norm(pts - center.reshape(1, 3), axis=1)
                        keep = d < CENTER_DIST_THRESH
                        pts, rgb = pts[keep], rgb[keep]

                    pcd_new = o3d.geometry.PointCloud()
                    pcd_new.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                    pcd_new.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

                    # flip axes (same as your reference)
                    pcd_new.transform([[1, 0, 0, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, -1, 0],
                                       [0, 0, 0, 1]])

                    if len(pcd_new.points) > 0:
                        pcd_new = pcd_new.voxel_down_sample(VOXEL_RENDER)

                    pcd.points = pcd_new.points
                    pcd.colors = pcd_new.colors

                    if not pcd_added:
                        vis.add_geometry(pcd)
                        pcd_added = True
                    else:
                        vis.update_geometry(pcd)

                    if (not ctrl["auto_framed_once"]) and len(pcd.points) > 0:
                        vis.reset_view_point(True)
                        ctrl["auto_framed_once"] = True

            # ---- keep both UIs responsive ----
            vis.poll_events()
            vis.update_renderer()

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

    finally:
        cv2.destroyAllWindows()
        vis.destroy_window()
        sub.close(0)
        ctx.term()
        print("[VIEW] Stopped.")


if __name__ == "__main__":
    main()
