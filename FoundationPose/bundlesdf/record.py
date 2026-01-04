#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import cv2
import pyrealsense2 as rs


# ======================== 默认参数 ========================
# Color 强制 1080p@30（你说的要求）
COLOR_W, COLOR_H, COLOR_FPS = 1920, 1080, 30

# Depth 分辨率：很多 D4xx 不支持 1080p depth，这里默认 1280x720@30（更通用）
DEPTH_W, DEPTH_H, DEPTH_FPS = 1280, 720, 30

# 预览显示缩放（减少CPU/GPU压力）
SHOW_SCALE = 0.75

# ArUco 参数
ARUCO_DICT = cv2.aruco.DICT_ARUCO_ORIGINAL
DICT_NAME = "DICT_ARUCO_ORIGINAL"
MARKER_LENGTH_M = 0.10  # 你的 marker 边长（米），务必填真实值
TARGET_ID = None        # None=用检测到的第一个；或填具体ID，例如 23

# 键位
KEY_START = "r"  # start recording
KEY_STOP  = "t"  # stop recording
KEY_QUIT  = "q"  # quit

CAM_SET = ["front", "left", "right"]


# ======================== 工具函数 ========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


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


def load_mapping(path: str) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        if not isinstance(m, dict):
            return {}
        out = {}
        for k, v in m.items():
            if isinstance(k, str) and isinstance(v, str):
                kk = k.strip().lower()
                if kk in CAM_SET and v.strip():
                    out[kk] = v.strip()
        return out
    except Exception:
        return {}


def pick_serial_for_role(role: str, mapping: Dict[str, str], override_serial: str) -> Tuple[str, str]:
    role = (role or "").strip().lower()
    if role not in CAM_SET:
        raise ValueError(f"--cam must be in {CAM_SET}, got {role}")

    if override_serial and override_serial.strip():
        return override_serial.strip(), "override"

    if mapping and role in mapping:
        return mapping[role], "mapping"

    connected = list_connected_devices()
    serials = sorted(list(connected.keys()))
    if len(serials) < len(CAM_SET):
        raise RuntimeError(f"[RS] only {len(serials)} device(s) found. Need >= {len(CAM_SET)} for auto-assign.")
    idx = CAM_SET.index(role)
    return serials[idx], "auto"


def _list_stream_profiles(dev: rs.device, stream_type: rs.stream):
    profiles = []
    try:
        # 对 color / depth 都适用：遍历所有 sensor 的 profiles
        for s in dev.query_sensors():
            for p in s.get_stream_profiles():
                try:
                    v = p.as_video_stream_profile()
                except Exception:
                    continue
                if v.stream_type() != stream_type:
                    continue
                profiles.append((v.width(), v.height(), v.fps(), v.format()))
    except Exception:
        pass
    return sorted(set(profiles), key=lambda x: (x[0] * x[1], x[2], str(x[3])))


def setup_realsense(serial: str,
                    color_w=COLOR_W, color_h=COLOR_H, color_fps=COLOR_FPS,
                    depth_w=DEPTH_W, depth_h=DEPTH_H, depth_fps=DEPTH_FPS):
    connected = list_connected_devices()
    if serial not in connected:
        if connected:
            print("[RS] Available devices:")
            for s, dev in connected.items():
                try:
                    name = dev.get_info(rs.camera_info.name)
                except Exception:
                    name = "Unknown"
                print(f"  S/N: {s} | {name}")
        raise RuntimeError(f"[RS] serial '{serial}' not found")

    dev = connected[serial]
    name = dev.get_info(rs.camera_info.name)
    print(f"[RS] Using device: {name} S/N: {serial}")

    # 检查 color profile
    color_profiles = _list_stream_profiles(dev, rs.stream.color)
    ok_color = [(w, h, fps, fmt) for (w, h, fps, fmt) in color_profiles
                if (w, h, fps) == (color_w, color_h, color_fps) and fmt in (rs.format.bgr8, rs.format.rgb8)]
    if not ok_color:
        print("[RS] Available COLOR profiles:")
        for w, h, fps, fmt in color_profiles:
            print(f"  {w}x{h} @{fps} fmt={fmt}")
        raise RuntimeError(f"[RS] Required COLOR profile {color_w}x{color_h}@{color_fps} (bgr8/rgb8) not available")

    ok_color.sort(key=lambda x: (x[3] == rs.format.bgr8), reverse=True)
    _, _, _, color_fmt = ok_color[0]
    need_rgb2bgr = (color_fmt == rs.format.rgb8)
    print(f"[RS] Color profile: {color_w}x{color_h}@{color_fps} fmt={color_fmt}")

    # 检查 depth profile（z16）
    depth_profiles = _list_stream_profiles(dev, rs.stream.depth)
    ok_depth = [(w, h, fps, fmt) for (w, h, fps, fmt) in depth_profiles
                if (w, h, fps) == (depth_w, depth_h, depth_fps) and fmt == rs.format.z16]
    if not ok_depth:
        print("[RS] Available DEPTH profiles:")
        for w, h, fps, fmt in depth_profiles:
            print(f"  {w}x{h} @{fps} fmt={fmt}")
        raise RuntimeError(f"[RS] Required DEPTH profile {depth_w}x{depth_h}@{depth_fps} (z16) not available")

    print(f"[RS] Depth profile: {depth_w}x{depth_h}@{depth_fps} fmt=z16")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, color_w, color_h, color_fmt, color_fps)
    config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, depth_fps)

    profile = pipeline.start(config)

    # 尽量减少延迟积压
    try:
        dev2 = profile.get_device()
        for s in dev2.query_sensors():
            if s.supports(rs.option.frames_queue_size):
                s.set_option(rs.option.frames_queue_size, 1.0)
    except Exception:
        pass

    # depth scale
    depth_scale = None
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
    except Exception:
        depth_scale = None

    # intrinsics (use color stream)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    K = np.array([
        [intr.fx, 0.0,     intr.ppx],
        [0.0,     intr.fy, intr.ppy],
        [0.0,     0.0,     1.0],
    ], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64).reshape(1, -1)

    print(f"[RS] intr: {intr.width}x{intr.height} fx={intr.fx:.3f} fy={intr.fy:.3f} cx={intr.ppx:.3f} cy={intr.ppy:.3f} model={intr.model}")
    if depth_scale is not None:
        print(f"[RS] depth_scale_m: {depth_scale} (meters per depth unit)")

    # alignment: depth -> color
    align = rs.align(rs.stream.color)

    # warm up
    for _ in range(15):
        pipeline.wait_for_frames()

    return pipeline, align, (intr.width, intr.height), K, dist, need_rgb2bgr, name, depth_scale


def make_detector():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    params = aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    # 让它更稳一点（可按需要调）
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        return lambda gray: detector.detectMarkers(gray)
    else:
        return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=params)


def estimate_pose_ippe_square(one_marker_corners, K, dist, marker_length_m):
    imgp = np.asarray(one_marker_corners, dtype=np.float64).reshape(-1, 2)
    if imgp.shape[0] != 4:
        return None, None, None

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
            return None, None, None
        return rvec.reshape(3), tvec.reshape(3), None

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

    return np.asarray(rvecs[best_i]).reshape(3), np.asarray(tvecs[best_i]).reshape(3), float(best_err)


def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def ts_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", default="left", choices=CAM_SET, help="camera role: front/left/right (default left)")
    p.add_argument("--mapping", default="realsense_mapping.json",
                   help="JSON: {front:serial, left:serial, right:serial}")
    p.add_argument("--serial", default="", help="Override serial (highest priority)")
    p.add_argument("--outroot", default="./rs_rgbd_aruco_record", help="Output root directory")
    p.add_argument("--marker_len", type=float, default=MARKER_LENGTH_M, help="Aruco marker length in meters")
    p.add_argument("--target_id", type=int, default=-1, help="Target marker id. -1 means use the first detected marker")
    p.add_argument("--show_scale", type=float, default=SHOW_SCALE, help="Preview display scale (e.g. 0.5~1.0)")
    return p.parse_args()


def main():
    # 尽量不抢系统资源
    try:
        os.nice(10)
    except Exception:
        pass
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    args = parse_args()
    mapping = load_mapping(args.mapping)
    serial, policy = pick_serial_for_role(args.cam, mapping, args.serial)
    target_id = None if args.target_id < 0 else int(args.target_id)

    print(f"[CAM] role={args.cam} serial={serial} (policy={policy})")
    print(f"[DICT] {DICT_NAME} | marker_len={args.marker_len}m | target_id={target_id}")

    pipeline, align, (W, H), K, dist, need_rgb2bgr, cam_name, depth_scale = setup_realsense(serial)

    detect_fn = make_detector()
    aruco = cv2.aruco

    # session dirs
    session_dir = os.path.join(args.outroot, args.cam, f"session_{ts_str()}")
    color_dir = os.path.join(session_dir, "color")
    depth_dir = os.path.join(session_dir, "depth")
    meta_dir  = os.path.join(session_dir, "meta")
    ensure_dir(color_dir); ensure_dir(depth_dir); ensure_dir(meta_dir)

    # save K/dist once
    np.savetxt(os.path.join(session_dir, "K.txt"), K)
    np.savetxt(os.path.join(session_dir, "dist.txt"), dist.reshape(-1))

    session_info = {
        "camera_role": args.cam,
        "camera_serial": serial,
        "camera_name": cam_name,
        "color_resolution": [COLOR_W, COLOR_H, COLOR_FPS],
        "depth_resolution": [DEPTH_W, DEPTH_H, DEPTH_FPS],
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),
        "aruco_dict": DICT_NAME,
        "marker_length_m": float(args.marker_len),
        "target_id": target_id,
        "depth_scale_m": depth_scale,
        "notes": "Depth PNG is Z16 raw. depth(m) = depth_raw * depth_scale_m",
        "keys": {"start": KEY_START, "stop": KEY_STOP, "quit": KEY_QUIT},
    }
    with open(os.path.join(session_dir, "session_info.json"), "w", encoding="utf-8") as f:
        json.dump(session_info, f, indent=2, ensure_ascii=False)

    print(f"[OUT] session_dir = {session_dir}")
    print(f"[KEY] '{KEY_START}' start | '{KEY_STOP}' stop | '{KEY_QUIT}' quit (ESC also quits)")

    # UI
    win = f"RS[{args.cam}] ArUco (start={KEY_START}, stop={KEY_STOP}, quit={KEY_QUIT})"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    recording = False
    idx = 0
    fps_t0 = time.time()
    fps_cnt = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)  # depth aligned to color

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # timestamps
            rs_ts_ms = float(color_frame.get_timestamp())
            ts_domain = str(color_frame.get_frame_timestamp_domain())
            wall_s = time.time()

            # images
            color = np.asanyarray(color_frame.get_data())
            if need_rgb2bgr:
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            depth = np.asanyarray(depth_frame.get_data())  # uint16 aligned to color size

            vis = color.copy()

            # detect
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = detect_fn(gray)

            chosen = None
            if ids is not None and corners is not None and len(ids) > 0:
                aruco.drawDetectedMarkers(vis, corners, ids)
                ids_flat = ids.flatten().astype(int)

                if target_id is None:
                    pick = 0
                else:
                    w = np.where(ids_flat == target_id)[0]
                    pick = int(w[0]) if len(w) > 0 else None

                if pick is not None:
                    mid = int(ids_flat[pick])
                    rvec, tvec, reproj_err = estimate_pose_ippe_square(corners[pick], K, dist, args.marker_len)
                    if rvec is not None:
                        chosen = (mid, rvec, tvec, reproj_err)
                        cv2.drawFrameAxes(vis, K, dist, rvec.reshape(3, 1), tvec.reshape(3, 1), args.marker_len * 0.5)

            # overlay text
            fps_cnt += 1
            if time.time() - fps_t0 >= 1.0:
                fps = fps_cnt / (time.time() - fps_t0)
                fps_t0 = time.time()
                fps_cnt = 0
            else:
                fps = None

            status = "REC" if recording else "PREVIEW"
            txt1 = f"{status} | idx={idx} | serial={serial[-6:]}"
            txt2 = f"Aruco: {'OK' if chosen is not None else 'MISS'}"
            if chosen is not None:
                mid, rvec, tvec, reproj_err = chosen
                z = float(tvec[2])
                err = f"{reproj_err:.2f}px" if reproj_err is not None else "n/a"
                txt2 += f" | id={mid} | z={z:.3f}m | reproj={err}"
            if fps is not None:
                txt1 += f" | fps={fps:.1f}"

            cv2.putText(vis, txt1, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if recording else (255, 255, 255), 2)
            cv2.putText(vis, txt2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(vis, f"keys: {KEY_START}=start {KEY_STOP}=stop {KEY_QUIT}=quit",
                        (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            show = vis
            if args.show_scale != 1.0:
                show = cv2.resize(vis, (0, 0), fx=args.show_scale, fy=args.show_scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(win, show)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord(KEY_QUIT), ord(KEY_QUIT.upper()), 27):
                break
            elif key in (ord(KEY_START), ord(KEY_START.upper())):
                recording = True
                print("[REC] START")
            elif key in (ord(KEY_STOP), ord(KEY_STOP.upper())):
                recording = False
                print("[REC] STOP")

            # save
            if recording:
                color_path = os.path.join(color_dir, f"{idx:06d}.png")
                depth_path = os.path.join(depth_dir, f"{idx:06d}.png")
                meta_path  = os.path.join(meta_dir,  f"{idx:06d}.npz")

                cv2.imwrite(color_path, color)
                cv2.imwrite(depth_path, depth)  # uint16 PNG

                valid = (chosen is not None)
                if valid:
                    mid, rvec, tvec, reproj_err = chosen
                    T_cam_tag = rvec_tvec_to_T(rvec, tvec)
                    T_tag_cam = invert_T(T_cam_tag)  # 以后做 cam_in_obs 就用它（把 tag 当 obs）
                else:
                    mid, rvec, tvec, reproj_err = -1, np.zeros(3), np.zeros(3), None
                    T_cam_tag = np.eye(4, dtype=np.float64)
                    T_tag_cam = np.eye(4, dtype=np.float64)

                payload = dict(
                    idx=int(idx),
                    timestamp_wall_s=float(wall_s),
                    rs_timestamp_ms=float(rs_ts_ms),
                    rs_timestamp_domain=str(ts_domain),

                    camera_role=str(args.cam),
                    camera_serial=str(serial),
                    camera_name=str(cam_name),

                    K=K,
                    dist=dist,
                    depth_scale_m=float(depth_scale) if depth_scale is not None else -1.0,

                    dict_name=DICT_NAME,
                    marker_length_m=float(args.marker_len),
                    target_id=int(target_id) if target_id is not None else -1,

                    detected=bool(valid),
                    marker_id=int(mid),
                    rvec=np.asarray(rvec, dtype=np.float64).reshape(3),
                    tvec=np.asarray(tvec, dtype=np.float64).reshape(3),
                    reproj_err_px=float(reproj_err) if reproj_err is not None else -1.0,

                    T_cam_tag=T_cam_tag,
                    T_tag_cam=T_tag_cam,

                    color_path=color_path,
                    depth_path=depth_path,
                )
                np.savez_compressed(meta_path, **payload)

                idx += 1

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print(f"[DONE] saved to: {session_dir}")


if __name__ == "__main__":
    main()

