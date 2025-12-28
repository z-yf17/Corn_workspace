#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import numpy as np
import cv2
import zmq
import trimesh
import torch

# ===================== 参数都写在这里 =====================
MESH_FILE = "./demo_data/mugblack_002.obj"

ZMQ_BIND = "tcp://0.0.0.0:5556"
ZMQ_TOPIC = "segd"

EST_REFINE_ITER = 5
TRACK_REFINE_ITER = 2

DEBUG = 1
DEBUG_DIR = "./debug_live"

MASK_REFINE_ENABLE = True
MASK_MIN_AREA = 800
MASK_OPEN_K = 3
MASK_CLOSE_K = 5
MASK_ERODE_ITER = 1

MIN_MASK_PIX_FOR_REGISTER = 200
AXIS_SCALE = 0.1

# 只处理最新帧相关参数
POLL_TIMEOUT_MS = 1   # 等新帧的轮询等待时间（ms），越小越实时但更吃CPU
# =========================================================


# ===================== 关键：彻底拦截 float64 -> torch.double =====================
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


# ========= FoundationPose 工程依赖（放补丁后 import 才能命中） =========
from estimater import (
    set_logging_format, set_seed,
    ScorePredictor, PoseRefinePredictor, FoundationPose,
    draw_posed_3d_box, draw_xyz_axis,
)
import nvdiffrast.torch as dr


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


def decode_png16(depth_png_bytes: bytes):
    arr = np.frombuffer(depth_png_bytes, dtype=np.uint8)
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        return None
    return depth


def decode_png8_gray(mask_png_bytes: bytes):
    arr = np.frombuffer(mask_png_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def decode_jpg_to_bgr(jpg_bytes: bytes):
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def meta_to_K_and_scale(meta: dict):
    fx = np.float32(meta["fx"])
    fy = np.float32(meta["fy"])
    cx = np.float32(meta["ppx"])
    cy = np.float32(meta["ppy"])
    depth_scale = np.float32(meta.get("depth_scale", 0.001))

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K, depth_scale


# ===================== 新增：只取最新帧 =====================
def recv_latest_multipart(sub, poller, poll_timeout_ms=1):
    """
    抽干SUB接收队列，只返回最新的一条multipart。
    没有新消息则返回None。
    """
    socks = dict(poller.poll(poll_timeout_ms))
    if sub not in socks:
        return None

    latest = None
    while True:
        try:
            latest = sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
    return latest
# =========================================================


def main():
    set_logging_format()
    set_seed(0)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    if not os.path.exists(MESH_FILE):
        raise FileNotFoundError(f"MESH_FILE not found: {MESH_FILE}")

    mesh = trimesh.load(MESH_FILE)

    # bbox for visualization
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # quick sanity: patch是否生效（可保留）
    try:
        print("[CHECK] torch.tensor(mesh.vertices).dtype =", torch.tensor(mesh.vertices).dtype)
    except Exception:
        pass

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

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
    print("[UI] Keys: S=register using current mask | R=reset | Q/ESC=quit")

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)

    # 你原来是 bind：保持不动（如果发布端也是bind会连不上，通常PUB bind / SUB connect）
    sub.bind(ZMQ_BIND)

    sub.setsockopt_string(zmq.SUBSCRIBE, ZMQ_TOPIC)
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.setsockopt(zmq.LINGER, 0)  # 退出时不阻塞

    try:
        sub.setsockopt(zmq.CONFLATE, 1)  # 有些平台/版本不支持；不支持也没关系，我们下面会抽干队列
    except Exception:
        pass

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    print(f"[ZMQ] Listening {ZMQ_BIND}, topic='{ZMQ_TOPIC}'")
    print("[ZMQ] Mode: drain-queue -> process latest frame only")

    tracking = False
    pose = None
    need_register = False
    frame_idx = 0

    cv2.namedWindow("live_pose", cv2.WINDOW_NORMAL)

    try:
        while True:
            # ====== 只取最新帧 ======
            parts = recv_latest_multipart(sub, poller, poll_timeout_ms=POLL_TIMEOUT_MS)
            if parts is None:
                # 没新帧也要让GUI响应键盘/刷新
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("s"), ord("S")):
                    need_register = True
                    tracking = False
                    pose = None
                    print("[UI] register requested (use current mask).")
                if key in (ord("r"), ord("R")):
                    need_register = False
                    tracking = False
                    pose = None
                    print("[UI] reset -> IDLE.")
                continue
            # ========================

            raw_jpg = None
            overlay_jpg = None

            if len(parts) == 7:
                _topic, ts_bytes, meta_bytes, mask_png, depth_png, raw_jpg, overlay_jpg = parts
            elif len(parts) == 6:
                _topic, ts_bytes, meta_bytes, mask_png, depth_png, raw_jpg = parts
            elif len(parts) == 5:
                _topic, ts_bytes, meta_bytes, mask_png, depth_png = parts
            else:
                print(f"[WARN] unexpected multipart len={len(parts)}")
                continue

            try:
                meta = json.loads(meta_bytes.decode("utf-8"))
                K, depth_scale32 = meta_to_K_and_scale(meta)
            except Exception as e:
                print("[WARN] meta parse/intrinsics fail:", e)
                continue

            depth_u16 = decode_png16(depth_png)
            if depth_u16 is None:
                continue
            depth_m = np.ascontiguousarray(depth_u16.astype(np.float32) * depth_scale32, dtype=np.float32)

            mask_u8 = decode_png8_gray(mask_png)
            if mask_u8 is None:
                continue
            if mask_u8.shape != depth_u16.shape:
                mask_u8 = cv2.resize(mask_u8, (depth_u16.shape[1], depth_u16.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

            if MASK_REFINE_ENABLE:
                mask_u8 = refine_mask_u8(mask_u8, MASK_MIN_AREA, MASK_OPEN_K, MASK_CLOSE_K, MASK_ERODE_ITER)
            else:
                mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255

            ob_mask = (mask_u8 > 0)
            mask_pix = int(ob_mask.sum())

            frame_bgr = None
            if raw_jpg is not None:
                frame_bgr = decode_jpg_to_bgr(raw_jpg)
            if frame_bgr is None and overlay_jpg is not None:
                frame_bgr = decode_jpg_to_bgr(overlay_jpg)
            if frame_bgr is None:
                continue
            if frame_bgr.shape[:2] != depth_u16.shape:
                frame_bgr = cv2.resize(frame_bgr, (depth_u16.shape[1], depth_u16.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # uint8
            frame_idx += 1

            # ---------- register：按 S 后用当前 mask ----------
            if need_register:
                if mask_pix < MIN_MASK_PIX_FOR_REGISTER:
                    print(f"[POSE] mask too small ({mask_pix} pix), wait a better frame.")
                else:
                    try:
                        print(f"[POSE] register... mask_pix={mask_pix}")
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
                        print("  dtype K:", K.dtype,
                              "depth:", depth_m.dtype,
                              "rgb:", frame_rgb.dtype,
                              "mesh.v:", mesh.vertices.dtype,
                              "mesh.n:", mesh.vertex_normals.dtype)
                        try:
                            print("  check torch.tensor(mesh.v).dtype:", torch.tensor(mesh.vertices).dtype)
                        except Exception:
                            pass
                        tracking = False
                        pose = None

                need_register = False

            # ---------- track ----------
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

            # ---------- visualize ----------
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
            hud2 = "S=register(mask) | R=reset | Q=quit"
            cv2.putText(vis_bgr, f"{hud1} frame={frame_idx} mask_pix={mask_pix}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(vis_bgr, hud2, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("live_pose", vis_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                need_register = True
                tracking = False
                pose = None
                print("[UI] register requested (use current mask).")
            if key in (ord("r"), ord("R")):
                need_register = False
                tracking = False
                pose = None
                print("[UI] reset -> IDLE.")

    finally:
        cv2.destroyAllWindows()
        sub.close(0)
        ctx.term()
        print("[EXIT] done.")


if __name__ == "__main__":
    main()
