#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
from collections import deque
import bisect
import argparse
from typing import Dict, Optional, Tuple, List

import numpy as np
import cv2
import zmq
import pyrealsense2 as rs

# ================== 尽量不影响控制（建议保留） ==================
try:
    os.nice(10)  # 降低本进程优先级
except Exception:
    pass

try:
    cv2.setNumThreads(1)        # 避免 OpenCV 多线程抢 CPU
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass
# =============================================================

# ================== 配置（默认写死，可被命令行覆盖部分） ==================
# 强制 1080p@30
RS_W = 1920
RS_H = 1080
RS_FPS = 30

# 预览检测策略（降低 CPU）
DETECT_EVERY_N = 2
PREVIEW_DETECT_SCALE = 0.5

# 预览窗口显示缩放
SHOW_SCALE = 0.5

# ZMQ 订阅关节角
SUB_ENDPOINTS = [
    "tcp://127.0.0.1:5555",
    "ipc:///tmp/panda_joints.ipc",
]

# 同步策略参数
JOINT_BUFFER_SEC = 3.0
WAIT_JOINT_AFTER_MS = 80
MAX_ALIGN_ERR_MS = 15
MAX_BRACKET_GAP_MS = 50

# 输出根目录（会在其下创建 front/left/right 子目录）
OUTROOT = "./handeye_data_rs_1080p_sync"

# ArUco marker
ARUCO_DICT = cv2.aruco.DICT_ARUCO_ORIGINAL
DICT_NAME = "DICT_ARUCO_ORIGINAL"
MARKER_LENGTH_M = 0.10
TARGET_ID = None

KEY_SAVE = "s"
KEY_QUIT = "q"

CAM_SET = ["front", "left", "right"]
# =============================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", default="front", choices=CAM_SET, help="Select camera role: front/left/right (default front)")
    p.add_argument("--mapping", default="realsense_mapping.json",
                   help="JSON file: {front:serial, left:serial, right:serial}")
    p.add_argument("--serial", default="", help="Override: directly specify RealSense serial (highest priority)")
    p.add_argument("--outroot", default=OUTROOT, help=f"Output root dir (default {OUTROOT})")
    return p.parse_args()


def normalize_cam(cam: str) -> str:
    c = (cam or "").strip().lower()
    if c not in CAM_SET:
        raise ValueError(f"Unknown cam '{cam}', allowed: {CAM_SET}")
    return c


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def now_ns() -> int:
    return time.monotonic_ns()


def load_mapping(path: str) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        if not isinstance(m, dict):
            return {}
        out = {}
        for k, v in m.items():
            if isinstance(k, str) and isinstance(v, str) and k.lower() in CAM_SET and v.strip():
                out[k.lower()] = v.strip()
        return out
    except Exception:
        return {}


def list_connected_devices() -> Dict[str, rs.device]:
    ctx = rs.context()
    devs = ctx.query_devices()
    out = {}
    for dev in devs:
        try:
            serial = dev.get_info(rs.camera_info.serial_number)
            out[serial] = dev
        except Exception:
            pass
    return out


def pick_serial_for_role(role: str, mapping: Dict[str, str], override_serial: str) -> Tuple[str, str]:
    """
    返回 (serial, policy_str)
      policy_str: "override" / "mapping" / "auto"
    """
    role = normalize_cam(role)

    if override_serial and override_serial.strip():
        return override_serial.strip(), "override"

    if mapping and role in mapping:
        return mapping[role], "mapping"

    # auto assign (deterministic): sorted serials -> front/left/right
    connected = list_connected_devices()
    serials = sorted(list(connected.keys()))
    if len(serials) < len(CAM_SET):
        raise RuntimeError(f"[RS] only {len(serials)} device(s) found. Need at least 3 for auto-assign.")
    idx = CAM_SET.index(role)
    return serials[idx], "auto"


def _list_color_profiles(dev: rs.device):
    try:
        sensor = dev.first_color_sensor()
    except Exception:
        return []
    profiles = []
    for p in sensor.get_stream_profiles():
        try:
            v = p.as_video_stream_profile()
        except Exception:
            continue
        if v.stream_type() != rs.stream.color:
            continue
        profiles.append((v.width(), v.height(), v.fps(), v.format()))
    return sorted(set(profiles), key=lambda x: (x[0] * x[1], x[2], str(x[3])))


def setup_realsense_force_1080p_30_by_serial(serial: str):
    connected = list_connected_devices()
    if serial not in connected:
        # 打印可用设备，方便排错
        if connected:
            print("[RS] Available devices:")
            for s, dev in connected.items():
                try:
                    name = dev.get_info(rs.camera_info.name)
                except Exception:
                    name = "Unknown"
                print(f"  S/N: {s} | {name}")
        raise RuntimeError(f"[RS] Required serial '{serial}' not found.")

    dev = connected[serial]
    name = dev.get_info(rs.camera_info.name)
    print(f"[RS] using device: {name} S/N: {serial}")

    profiles = _list_color_profiles(dev)
    if not profiles:
        raise RuntimeError("[RS] No color profiles found (no color sensor?).")

    want = [(w, h, fps, fmt) for (w, h, fps, fmt) in profiles
            if (w, h, fps) == (RS_W, RS_H, RS_FPS) and fmt in (rs.format.bgr8, rs.format.rgb8)]

    if not want:
        print("[RS] Available color profiles:")
        for w, h, fps, fmt in profiles:
            print(f"  {w}x{h} @{fps} fmt={fmt}")
        raise RuntimeError(f"[RS] Required profile {RS_W}x{RS_H} @{RS_FPS} (bgr8/rgb8) not available.")

    # 同一档位可能同时有 rgb8/bgr8：优先 bgr8
    want.sort(key=lambda x: (x[3] == rs.format.bgr8), reverse=True)
    w, h, fps, fmt = want[0]
    need_rgb2bgr = (fmt == rs.format.rgb8)
    print(f"[RS] selected color profile: {w}x{h} @{fps} fmt={fmt}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, w, h, fmt, fps)

    profile = pipeline.start(config)

    # 尽量减少积压帧延迟
    try:
        dev2 = profile.get_device()
        for s in dev2.query_sensors():
            if s.supports(rs.option.frames_queue_size):
                s.set_option(rs.option.frames_queue_size, 1.0)
    except Exception:
        pass

    for _ in range(10):
        pipeline.wait_for_frames()

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    K = np.array([
        [intr.fx, 0.0,     intr.ppx],
        [0.0,     intr.fy, intr.ppy],
        [0.0,     0.0,     1.0],
    ], dtype=np.float64)

    dist = np.array(intr.coeffs, dtype=np.float64).reshape(1, -1)

    print(f"[RS] intr: {intr.width}x{intr.height} fx={intr.fx:.3f} fy={intr.fy:.3f} "
          f"cx={intr.ppx:.3f} cy={intr.ppy:.3f} model={intr.model}")
    return pipeline, (intr.width, intr.height), K, dist, need_rgb2bgr, name


def setup_zmq_sub():
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVHWM, 200)
    sub.setsockopt(zmq.LINGER, 0)

    for ep in SUB_ENDPOINTS:
        sub.connect(ep)
        print("[ZMQ] SUB connect:", ep)

    time.sleep(0.2)
    return ctx, sub


def drain_joint_messages(sub, joint_buf: deque, max_msgs: int = 500):
    n = 0
    while n < max_msgs:
        try:
            s = sub.recv_string(flags=zmq.NOBLOCK)
        except zmq.Again:
            break

        n += 1
        try:
            m = json.loads(s)
            if not isinstance(m, dict) or "q" not in m:
                continue
            q = m["q"]
            if not isinstance(q, list) or len(q) < 7:
                continue

            if "t_meas_ns" in m:
                t_meas_ns = int(m["t_meas_ns"])
            elif "t" in m:
                # 兼容旧版，但注意：t(time.time) 与 monotonic 不同时间基，严格同步会失效
                t_meas_ns = int(float(m["t"]) * 1e9)
            else:
                continue

            joint_buf.append({
                "seq": int(m.get("seq", -1)),
                "t_meas_ns": t_meas_ns,
                "q": np.asarray(q[:7], dtype=np.float64),
            })
        except Exception:
            pass


def prune_joint_buffer(joint_buf: deque, keep_sec: float):
    if not joint_buf:
        return
    newest = joint_buf[-1]["t_meas_ns"]
    keep_ns = int(keep_sec * 1e9)
    while joint_buf and (newest - joint_buf[0]["t_meas_ns"] > keep_ns):
        joint_buf.popleft()


def grab_latest_color_frame(pipeline, need_rgb2bgr: bool):
    frames = pipeline.wait_for_frames()
    t_ns = now_ns()

    while True:
        fs = pipeline.poll_for_frames()
        if not fs:
            break
        frames = fs
        t_ns = now_ns()

    color = frames.get_color_frame()
    if not color:
        return None, None, None, None

    img = np.asanyarray(color.get_data())
    if need_rgb2bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    rs_ts_ms = float(color.get_timestamp())
    ts_domain = color.get_frame_timestamp_domain()
    return img, t_ns, rs_ts_ms, ts_domain


def make_detector(dictionary):
    aruco = cv2.aruco
    params = aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        return lambda gray: detector.detectMarkers(gray)
    else:
        return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=params)


def detect_markers(frame_bgr, detect_fn, scale: float):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    if scale < 1.0:
        gray_s = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        corners_s, ids, rejected = detect_fn(gray_s)
        if ids is None or len(ids) == 0:
            return None, None, rejected
        corners = [c.astype(np.float64) / scale for c in corners_s]
        return corners, ids, rejected

    corners, ids, rejected = detect_fn(gray)
    return corners, ids, rejected


def estimate_pose_ippe_square(one_marker_corners, K, dist, marker_length_m):
    imgp = np.asarray(one_marker_corners, dtype=np.float64).reshape(-1, 2)
    if imgp.shape[0] != 4:
        return None, None

    s = marker_length_m / 2.0
    objp = np.array([
        [-s,  s, 0.0],
        [ s,  s, 0.0],
        [ s, -s, 0.0],
        [-s, -s, 0.0],
    ], dtype=np.float64)

    ok, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
        objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok or rvecs is None or len(rvecs) == 0:
        ok2, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok2:
            return None, None
        return rvec.reshape(3), tvec.reshape(3)

    best_i = None
    best_err = 1e18
    for i in range(len(rvecs)):
        tv = np.asarray(tvecs[i]).reshape(3)
        err = float(np.asarray(reproj[i]).reshape(-1)[0]) if reproj is not None else 0.0
        if tv[2] <= 0:
            continue
        if err < best_err:
            best_err = err
            best_i = i
    if best_i is None:
        best_i = 0

    return np.asarray(rvecs[best_i]).reshape(3), np.asarray(tvecs[best_i]).reshape(3)


def rvec_tvec_to_T_cam_tag(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def match_joint_at_time(t_img_ns: int, joint_buf: deque):
    if len(joint_buf) < 1:
        return None

    t_list = [m["t_meas_ns"] for m in joint_buf]
    i = bisect.bisect_left(t_list, t_img_ns)

    if 0 < i < len(t_list):
        m0 = joint_buf[i - 1]
        m1 = joint_buf[i]
        t0 = m0["t_meas_ns"]
        t1 = m1["t_meas_ns"]
        gap_ns = t1 - t0
        if gap_ns > 0:
            alpha = (t_img_ns - t0) / float(gap_ns)
            q = (1.0 - alpha) * m0["q"] + alpha * m1["q"]
            err_ns = max(abs(t_img_ns - t0), abs(t1 - t_img_ns))
            return {
                "mode": "interp",
                "q": q,
                "err_ns": int(err_ns),
                "t0_ns": int(t0),
                "t1_ns": int(t1),
                "seq0": int(m0["seq"]),
                "seq1": int(m1["seq"]),
                "gap_ns": int(gap_ns),
                "alpha": float(alpha),
            }

    candidates = []
    if i - 1 >= 0:
        candidates.append(joint_buf[i - 1])
    if i < len(t_list):
        candidates.append(joint_buf[i])

    if not candidates:
        return None

    best = min(candidates, key=lambda m: abs(t_img_ns - m["t_meas_ns"]))
    err_ns = abs(t_img_ns - best["t_meas_ns"])
    return {
        "mode": "nearest",
        "q": best["q"].copy(),
        "err_ns": int(err_ns),
        "t_near_ns": int(best["t_meas_ns"]),
        "seq_near": int(best["seq"]),
    }


def wait_until_have_joint_after(t_img_ns: int, sub, joint_buf: deque, timeout_ms: int):
    deadline = now_ns() + int(timeout_ms * 1e6)
    while now_ns() < deadline:
        drain_joint_messages(sub, joint_buf, max_msgs=500)
        prune_joint_buffer(joint_buf, JOINT_BUFFER_SEC)
        if joint_buf and joint_buf[-1]["t_meas_ns"] >= t_img_ns:
            return True
        time.sleep(0.001)
    return False


def next_index_in_dir(outdir: str) -> int:
    """
    找到 sample_XXXX.npz 的下一个 idx，避免覆盖。
    """
    try:
        files = [f for f in os.listdir(outdir) if f.startswith("sample_") and f.endswith(".npz")]
        mx = -1
        for f in files:
            s = f[len("sample_"):-len(".npz")]
            if s.isdigit():
                mx = max(mx, int(s))
        return mx + 1
    except Exception:
        return 0


def main():
    args = parse_args()
    cam_role = normalize_cam(args.cam)
    mapping = load_mapping(args.mapping)

    # --- create output dirs: outroot/{front,left,right}/color ---
    outroot = args.outroot
    for c in CAM_SET:
        ensure_dir(os.path.join(outroot, c))
        ensure_dir(os.path.join(outroot, c, "color"))

    # --- select serial for this role ---
    serial, policy = pick_serial_for_role(cam_role, mapping, args.serial)
    print(f"[CAM] role={cam_role} serial={serial} (policy={policy})")

    # This cam's output dir
    OUTDIR = os.path.join(outroot, cam_role)
    IMGDIR = os.path.join(OUTDIR, "color")
    ensure_dir(OUTDIR)
    ensure_dir(IMGDIR)

    # --- start camera + zmq ---
    pipeline, (W, H), K, dist, need_rgb2bgr, cam_name = setup_realsense_force_1080p_30_by_serial(serial)
    ctx, sub = setup_zmq_sub()

    joint_buf = deque()

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    detect_fn = make_detector(dictionary)

    win_name = f"collect [{cam_role}] (s=save q=quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    print(f"[DICT] locked to {DICT_NAME}")
    print("[KEY] s=save one sample, q=quit")

    idx = next_index_in_dir(OUTDIR)
    t_hb = time.time()
    frame_count = 0

    cached = {"corners": None, "ids": None, "chosen": None}

    try:
        while True:
            drain_joint_messages(sub, joint_buf, max_msgs=300)
            prune_joint_buffer(joint_buf, JOINT_BUFFER_SEC)

            frame, t_img_ns, rs_ts_ms, ts_domain = grab_latest_color_frame(pipeline, need_rgb2bgr)
            if frame is None:
                continue
            frame_count += 1

            vis = frame.copy()

            # ---------- preview detection ----------
            if frame_count % max(1, int(DETECT_EVERY_N)) == 0:
                corners, ids, _ = detect_markers(frame, detect_fn, PREVIEW_DETECT_SCALE)
                cached["corners"] = corners
                cached["ids"] = ids
                cached["chosen"] = None

                if ids is not None and corners is not None and len(ids) > 0:
                    ids_flat = ids.flatten()
                    if TARGET_ID is None:
                        i = 0
                    else:
                        w_ = np.where(ids_flat == TARGET_ID)[0]
                        i = int(w_[0]) if len(w_) > 0 else None

                    if i is not None:
                        mid = int(ids_flat[i])
                        rvec, tvec = estimate_pose_ippe_square(corners[i], K, dist, MARKER_LENGTH_M)
                        if rvec is not None:
                            cached["chosen"] = (mid, rvec, tvec)

            corners = cached["corners"]
            ids = cached["ids"]
            chosen = cached["chosen"]

            if ids is not None and corners is not None and len(ids) > 0:
                aruco.drawDetectedMarkers(vis, corners, ids)
                if chosen is not None:
                    mid, rvec, tvec = chosen
                    cv2.drawFrameAxes(vis, K, dist, rvec.reshape(3, 1), tvec.reshape(3, 1), MARKER_LENGTH_M * 0.5)

            # ---------- sync hint ----------
            match = match_joint_at_time(t_img_ns, joint_buf) if joint_buf else None
            if match is None:
                sync_txt = "joint: NONE"
                col = (0, 0, 255)
            else:
                err_ms = match["err_ns"] / 1e6
                sync_txt = f"joint: {match['mode']} err={err_ms:.1f}ms"
                col = (0, 255, 0) if err_ms <= MAX_ALIGN_ERR_MS else (0, 0, 255)

            cv2.putText(vis, f"cam={cam_role} | {sync_txt} | press '{KEY_SAVE}'",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)

            show = vis
            if SHOW_SCALE < 1.0:
                show = cv2.resize(vis, (0, 0), fx=SHOW_SCALE, fy=SHOW_SCALE, interpolation=cv2.INTER_AREA)
            cv2.imshow(win_name, show)

            key = (cv2.waitKey(1) & 0xFF)

            if time.time() - t_hb > 1.0:
                jb = len(joint_buf)
                last_seq = joint_buf[-1]["seq"] if jb else None
                print(f"[HB] cam={cam_role} idx={idx} joint_buf={jb} last_seq={last_seq}")
                t_hb = time.time()

            if key in (ord(KEY_QUIT), ord(KEY_QUIT.upper()), 27):
                break

            # ================= SAVE (freeze + full-res detect) =================
            if key in (ord(KEY_SAVE), ord(KEY_SAVE.upper())):
                frozen_frame = frame.copy()
                frozen_t_img_ns = int(t_img_ns)
                frozen_rs_ts_ms = float(rs_ts_ms)
                frozen_domain = str(ts_domain)

                corners_s, ids_s, _ = detect_markers(frozen_frame, detect_fn, scale=1.0)
                if ids_s is None or corners_s is None or len(ids_s) == 0:
                    print("[SAVE] 冻结帧未检测到 marker（全分辨率），未保存。")
                    continue

                ids_flat = ids_s.flatten()
                if TARGET_ID is None:
                    i = 0
                else:
                    w_ = np.where(ids_flat == TARGET_ID)[0]
                    i = int(w_[0]) if len(w_) > 0 else None
                if i is None:
                    print("[SAVE] 冻结帧未检测到目标 ID（全分辨率），未保存。")
                    continue

                mid = int(ids_flat[i])
                rvec, tvec = estimate_pose_ippe_square(corners_s[i], K, dist, MARKER_LENGTH_M)
                if rvec is None:
                    print("[SAVE] 冻结帧 solvePnP 失败，未保存。")
                    continue

                wait_until_have_joint_after(frozen_t_img_ns, sub, joint_buf, WAIT_JOINT_AFTER_MS)

                m = match_joint_at_time(frozen_t_img_ns, joint_buf)
                if m is None:
                    print("[SAVE] 关节缓冲为空/无效，未保存。")
                    continue

                err_ms = m["err_ns"] / 1e6
                if err_ms > MAX_ALIGN_ERR_MS:
                    print(f"[SAVE] 对齐误差过大 err={err_ms:.1f}ms > {MAX_ALIGN_ERR_MS}ms，未保存。")
                    continue

                if m["mode"] == "interp":
                    gap_ms = m["gap_ns"] / 1e6
                    if gap_ms > MAX_BRACKET_GAP_MS:
                        print(f"[SAVE] 插值区间过大 gap={gap_ms:.1f}ms > {MAX_BRACKET_GAP_MS}ms，未保存。")
                        continue

                ts_wall = time.time()
                T_cam_tag = rvec_tvec_to_T_cam_tag(rvec, tvec)

                img_path = os.path.join(IMGDIR, f"img_{idx:04d}.png")
                npz_path = os.path.join(OUTDIR, f"sample_{idx:04d}.npz")

                cv2.imwrite(img_path, frozen_frame)

                payload = dict(
                    # --- camera identity ---
                    camera_role=str(cam_role),
                    camera_serial=str(serial),
                    camera_name=str(cam_name),

                    # --- timing ---
                    timestamp_wall_s=float(ts_wall),
                    t_img_ns=int(frozen_t_img_ns),

                    # --- camera model ---
                    capture_resolution=np.array([W, H], dtype=np.int32),
                    K=K,
                    dist=dist,

                    # --- marker ---
                    dict_name=DICT_NAME,
                    marker_id=int(mid),
                    marker_length_m=float(MARKER_LENGTH_M),
                    rvec=np.asarray(rvec).reshape(3),
                    tvec=np.asarray(tvec).reshape(3),
                    T_cam_tag=T_cam_tag,

                    # --- robot ---
                    q=np.asarray(m["q"], dtype=np.float64),
                    match_mode=str(m["mode"]),
                    match_err_ns=int(m["err_ns"]),

                    # --- files ---
                    color_path=img_path,

                    # --- realsense meta ---
                    rs_timestamp_ms=float(frozen_rs_ts_ms),
                    rs_timestamp_domain=str(frozen_domain),
                    rs_width=int(RS_W),
                    rs_height=int(RS_H),
                    rs_fps=int(RS_FPS),

                    # --- preview params for repro ---
                    preview_detect_scale=float(PREVIEW_DETECT_SCALE),
                    detect_every_n=int(DETECT_EVERY_N),
                    show_scale=float(SHOW_SCALE),
                    sub_endpoints=np.array(SUB_ENDPOINTS, dtype=object),
                )

                for k, v in m.items():
                    if k == "q":
                        continue
                    payload[f"match_{k}"] = v

                np.savez_compressed(npz_path, **payload)
                print(f"[SAVE] cam={cam_role} -> {npz_path} (mode={m['mode']} err={err_ms:.1f}ms)")
                idx += 1

    except KeyboardInterrupt:
        print("\n[EXIT] ctrl+c")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        sub.close(0)
        ctx.term()
        cv2.destroyAllWindows()
        print("[DONE] exit.")


if __name__ == "__main__":
    main()
