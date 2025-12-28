#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import zmq

import pyrealsense2 as rs


CAM_SET = ["front", "left", "right"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bind", default="0.0.0.0", help="PUB bind IP, default 0.0.0.0")
    p.add_argument("--port", type=int, default=5555, help="PUB bind port, default 5555")
    p.add_argument("--cams", default="all", help="front,left,right or all (default all)")
    p.add_argument("--mapping", default="realsense_mapping.json",
                   help="JSON: {front:serial, left:serial, right:serial}")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--align", action="store_true", default=True,
                   help="Align depth to color (default True)")
    p.add_argument("--no-align", dest="align", action="store_false")
    p.add_argument("--print-fps", action="store_true", default=True)
    return p.parse_args()


def normalize_cams(cams_arg: str):
    s = (cams_arg or "").strip().lower()
    if s in ("", "all"):
        return CAM_SET.copy()
    cams = [c.strip() for c in s.split(",") if c.strip()]
    out = []
    for c in cams:
        if c not in CAM_SET:
            raise ValueError(f"Unknown cam '{c}', allowed: {CAM_SET} or all")
        if c not in out:
            out.append(c)
    return out


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


def list_connected_serials() -> Dict[str, rs.device]:
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


def start_pipeline(serial: str, width: int, height: int, fps: int) -> Tuple[rs.pipeline, rs.align]:
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipe.start(cfg)
    align = rs.align(rs.stream.color)
    return pipe, align


def stop_pipeline(pipe: Optional[rs.pipeline]):
    if pipe is None:
        return
    try:
        pipe.stop()
    except Exception:
        pass


def get_intrinsics_and_scale(pipe: rs.pipeline) -> Tuple[Dict[str, float], float]:
    """
    Grab intrinsics from the active profile + depth scale from depth sensor.
    """
    prof = pipe.get_active_profile()
    color_stream = prof.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    intr_dict = {"fx": float(intr.fx), "fy": float(intr.fy), "ppx": float(intr.ppx), "ppy": float(intr.ppy)}

    depth_sensor = prof.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    return intr_dict, depth_scale


def main():
    args = parse_args()
    cams = normalize_cams(args.cams)

    mapping = load_mapping(args.mapping)
    # 如果没有 mapping，就直接用设备枚举顺序（不推荐，但可跑）
    if mapping:
        missing = [c for c in cams if c not in mapping]
        if missing:
            raise SystemExit(f"Mapping file '{args.mapping}' missing keys: {missing}")
    else:
        print(f"[PUB] WARN: mapping '{args.mapping}' not found/invalid. Will publish any first 3 devices found.")

    # --- ZMQ PUB ---
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{args.bind}:{args.port}")
    pub.setsockopt(zmq.SNDHWM, 1000)
    pub.setsockopt(zmq.LINGER, 0)

    print(f"[PUB] Bind tcp://{args.bind}:{args.port}")
    print(f"[PUB] cams={cams} size={args.width}x{args.height}@{args.fps} jpegQ={args.jpeg_quality} align={args.align}")
    print("[PUB] multipart: [topic, ts, meta_json, rgb_jpg, depth_png16]")
    print("[PUB] topics: " + ", ".join([f"rgbd/{c}" for c in cams]))

    # --- Per-cam state ---
    pipes: Dict[str, Optional[rs.pipeline]] = {c: None for c in cams}
    aligns: Dict[str, Optional[rs.align]] = {c: None for c in cams}
    intr_map: Dict[str, Optional[Dict[str, float]]] = {c: None for c in cams}
    depth_scale_map: Dict[str, Optional[float]] = {c: None for c in cams}
    serial_map: Dict[str, Optional[str]] = {c: None for c in cams}

    # If mapping missing, assign first 3 connected serials deterministically at start
    if not mapping:
        connected = list_connected_serials()
        serials = list(connected.keys())
        if len(serials) < len(cams):
            print(f"[PUB] ERROR: only {len(serials)} RealSense device(s) found, need {len(cams)}")
        for i, cam in enumerate(cams):
            if i < len(serials):
                serial_map[cam] = serials[i]
                print(f"[PUB] Auto-assign {cam} <- serial {serials[i]}")
    else:
        for cam in cams:
            serial_map[cam] = mapping[cam]
            print(f"[PUB] Map {cam} <- serial {serial_map[cam]}")

    # Reconnect probe
    last_probe = 0.0
    probe_interval = 1.0

    # FPS stats
    recv_cnt = {c: 0 for c in cams}
    last_fps_t = time.time()

    try:
        while True:
            now = time.time()

            # Probe/reconnect
            if now - last_probe >= probe_interval:
                connected = list_connected_serials()
                for cam in cams:
                    serial = serial_map[cam]
                    if not serial:
                        continue

                    if serial in connected and pipes[cam] is None:
                        try:
                            pipe, al = start_pipeline(serial, args.width, args.height, args.fps)
                            pipes[cam] = pipe
                            aligns[cam] = al if args.align else None
                            intr, dscale = get_intrinsics_and_scale(pipe)
                            intr_map[cam] = intr
                            depth_scale_map[cam] = dscale
                            print(f"[PUB] Started cam={cam} serial={serial} depth_scale={dscale}")
                        except Exception as e:
                            pipes[cam] = None
                            aligns[cam] = None
                            intr_map[cam] = None
                            depth_scale_map[cam] = None
                            print(f"[PUB] WARN start failed cam={cam} serial={serial}: {e}")

                    if serial not in connected and pipes[cam] is not None:
                        print(f"[PUB] Disconnected cam={cam} serial={serial}")
                        stop_pipeline(pipes[cam])
                        pipes[cam] = None
                        aligns[cam] = None
                        intr_map[cam] = None
                        depth_scale_map[cam] = None

                last_probe = now

            # Grab & publish each cam (non-blocking by poll_for_frames)
            for cam in cams:
                pipe = pipes[cam]
                if pipe is None:
                    continue

                try:
                    frames = pipe.poll_for_frames()
                    if not frames:
                        continue
                    if aligns[cam] is not None:
                        frames = aligns[cam].process(frames)

                    cf = frames.get_color_frame()
                    df = frames.get_depth_frame()
                    if (not cf) or (not df):
                        continue

                    color_bgr = np.asanyarray(cf.get_data())  # (H,W,3) BGR8
                    depth_u16 = np.asanyarray(df.get_data())  # (H,W) uint16

                    # Encode
                    ok_c, buf_c = cv2.imencode(".jpg", color_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
                    if not ok_c:
                        continue
                    ok_d, buf_d = cv2.imencode(".png", depth_u16)  # PNG16
                    if not ok_d:
                        continue

                    ts = time.time()
                    ts_bytes = f"{ts:.6f}".encode("ascii")

                    intr = intr_map[cam] or {"fx": 0.0, "fy": 0.0, "ppx": 0.0, "ppy": 0.0}
                    dscale = depth_scale_map[cam] if depth_scale_map[cam] is not None else float(df.get_units())

                    meta = {
                        "cam": cam,
                        "serial": serial_map[cam] or "",
                        "width": int(color_bgr.shape[1]),
                        "height": int(color_bgr.shape[0]),
                        "fx": intr["fx"],
                        "fy": intr["fy"],
                        "ppx": intr["ppx"],
                        "ppy": intr["ppy"],
                        "depth_scale": float(dscale),
                    }
                    meta_bytes = json.dumps(meta).encode("utf-8")

                    topic = f"rgbd/{cam}".encode("utf-8")
                    pub.send_multipart([topic, ts_bytes, meta_bytes, buf_c.tobytes(), buf_d.tobytes()])

                    recv_cnt[cam] += 1

                except Exception as e:
                    # If runtime error, drop this pipeline and let probe reconnect
                    print(f"[PUB] WARN runtime cam={cam}: {e}")
                    stop_pipeline(pipe)
                    pipes[cam] = None
                    aligns[cam] = None
                    intr_map[cam] = None
                    depth_scale_map[cam] = None

            # Print FPS per cam
            if args.print_fps:
                tnow = time.time()
                if tnow - last_fps_t >= 1.0:
                    dt = tnow - last_fps_t
                    msg = " | ".join([f"{c}:{(recv_cnt[c]/dt):.1f}fps" for c in cams])
                    print(f"[PUB] {msg}")
                    for c in cams:
                        recv_cnt[c] = 0
                    last_fps_t = tnow

            time.sleep(0.001)

    finally:
        for cam in cams:
            stop_pipeline(pipes[cam])
        pub.close(0)
        ctx.term()
        print("[PUB] Stopped.")


if __name__ == "__main__":
    main()

