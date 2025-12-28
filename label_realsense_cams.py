#!/usr/bin/env python3
import time
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import pyrealsense2 as rs

# 1) 固定显示顺序：显示名只是提示，不影响最终 front/left/right
ORDERED_CAMS = [
    ("Cam1", "336222074955"),
    ("Cam2", "243722072992"),
    ("Cam3", "337122072591"),
]

# 2) 采样参数（带宽不够就降 FPS 或分辨率）
WIDTH, HEIGHT, FPS = 640, 480, 30

# 3) 拼图 tile 大小（仅影响显示）
TILE_W, TILE_H = 480, 360

# 4) 输出 mapping 文件
OUT_JSON = "realsense_mapping.json"


def enumerate_devices(ctx: rs.context) -> List[Dict[str, str]]:
    devs = ctx.query_devices()
    infos: List[Dict[str, str]] = []
    for dev in devs:
        info: Dict[str, str] = {}
        def gi(k):
            try:
                return dev.get_info(k)
            except Exception:
                return "N/A"

        info["name"] = gi(rs.camera_info.name)
        info["serial"] = gi(rs.camera_info.serial_number)
        info["physical_port"] = gi(rs.camera_info.physical_port)
        info["product_line"] = gi(rs.camera_info.product_line)
        info["firmware_version"] = gi(rs.camera_info.firmware_version)
        info["usb_type"] = gi(rs.camera_info.usb_type_descriptor)
        infos.append(info)
    return infos


def print_device_list(ctx: rs.context):
    infos = enumerate_devices(ctx)
    print(f"Found {len(infos)} devices")
    for i, d in enumerate(infos):
        print(
            f"[{i}] name={d['name']}  serial={d['serial']}  physical_port={d['physical_port']}  "
            f"usb={d['usb_type']}  fw={d['firmware_version']}  line={d['product_line']}"
        )


def list_connected_serials(ctx: rs.context) -> List[str]:
    serials: List[str] = []
    for d in enumerate_devices(ctx):
        sn = d.get("serial", "N/A")
        if sn and sn != "N/A":
            serials.append(sn)
    return serials


def start_pipeline(serial: str) -> rs.pipeline:
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipe.start(cfg)
    return pipe


def stop_pipeline(pipe: Optional[rs.pipeline]):
    if pipe is None:
        return
    try:
        pipe.stop()
    except Exception:
        pass


def draw_active_border(img: np.ndarray):
    h, w = img.shape[:2]
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), (0, 255, 255), 4)


def make_grid(
    images: List[Optional[np.ndarray]],
    labels: List[str],
    tile_size: Tuple[int, int],
    active_idx: int,
) -> np.ndarray:
    tw, th = tile_size
    n = len(images)
    cols = n  # 固定 1x3
    blank = np.zeros((th, tw, 3), dtype=np.uint8)

    tiles = []
    for i in range(cols):
        img = images[i]
        lab = labels[i]

        if img is None:
            canvas = blank.copy()
            cv2.putText(canvas, f"{lab}  (MISSING)", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            canvas = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            cv2.putText(canvas, lab, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if i == active_idx:
            draw_active_border(canvas)
            cv2.putText(canvas, "ACTIVE", (10, th - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        tiles.append(canvas)

    return np.hstack(tiles)


def main():
    ctx = rs.context()

    # ============ 合并点：启动先打印设备列表（代码一能力） ============
    print_device_list(ctx)
    print("\nRequested ORDERED_CAMS:")
    for i, (nm, sn) in enumerate(ORDERED_CAMS):
        print(f"  [{i}] {nm} -> {sn}")
    print("========================================================\n")

    window = "RealSense Labeler (1/2/3 select | f/l/r assign | s save | q ESC quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    serials = [sn for _, sn in ORDERED_CAMS]

    # 管理每个 serial 的 pipeline 与最近一帧（缓存避免闪烁）
    pipes: Dict[str, Optional[rs.pipeline]] = {sn: None for sn in serials}
    last_img: Dict[str, Optional[np.ndarray]] = {sn: None for sn in serials}

    # role -> serial 的最终映射
    mapping: Dict[str, str] = {}

    # 控制重连频率
    last_probe_time = 0.0
    probe_interval = 1.0

    active_idx = 0

    def assign(role: str, serial: str):
        mapping[role] = serial
        print(f"[MAP] {role} <- {serial}")
        print(json.dumps(mapping, ensure_ascii=False, indent=2))

    try:
        while True:
            now = time.time()

            # 1) 定期探测在线设备，离线的重连
            if now - last_probe_time >= probe_interval:
                connected = set(list_connected_serials(ctx))
                for _, sn in ORDERED_CAMS:
                    if sn in connected and pipes[sn] is None:
                        try:
                            pipes[sn] = start_pipeline(sn)
                            print(f"[INFO] Started {sn}")
                        except Exception as e:
                            pipes[sn] = None
                            print(f"[WARN] Start failed {sn}: {e}")
                    if sn not in connected and pipes[sn] is not None:
                        print(f"[INFO] Disconnected {sn}")
                        stop_pipeline(pipes[sn])
                        pipes[sn] = None
                        last_img[sn] = None
                last_probe_time = now

            # 2) 拉帧：poll 不阻塞；拿不到新帧就沿用 last_img（不黑屏闪烁）
            for _, sn in ORDERED_CAMS:
                pipe = pipes[sn]
                if pipe is None:
                    continue
                try:
                    frames = pipe.poll_for_frames()
                    if frames:
                        cf = frames.get_color_frame()
                        if cf:
                            last_img[sn] = np.asanyarray(cf.get_data())
                except Exception as e:
                    print(f"[WARN] Runtime error {sn}: {e}")
                    stop_pipeline(pipe)
                    pipes[sn] = None
                    last_img[sn] = None

            # 3) 组装显示（严格按 ORDERED_CAMS 顺序）
            imgs: List[Optional[np.ndarray]] = []
            labels: List[str] = []
            for idx, (name, sn) in enumerate(ORDERED_CAMS):
                imgs.append(last_img[sn])

                # 反查：这个 serial 被映射成了哪些角色
                roles = [r for r, s in mapping.items() if s == sn]
                role_str = ",".join(roles) if roles else "-"

                labels.append(f"[{idx+1}] {name}  serial:{sn}  role:{role_str}")

            grid = make_grid(imgs, labels, (TILE_W, TILE_H), active_idx)

            # 底部提示
            h, w = grid.shape[:2]
            cv2.putText(grid, "keys: 1/2/3 select | f=front l=left r=right | s=save | q/ESC=quit",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(window, grid)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("[EXIT] Quit (not saved).")
                break

            # 选择 active
            if key in (ord('1'), ord('2'), ord('3')):
                idx = key - ord('1')
                if 0 <= idx < len(ORDERED_CAMS):
                    active_idx = idx

            # 绑定 front/left/right
            if key == ord('f'):
                assign("front", ORDERED_CAMS[active_idx][1])
            if key == ord('l'):
                assign("left", ORDERED_CAMS[active_idx][1])
            if key == ord('r'):
                assign("right", ORDERED_CAMS[active_idx][1])

            # 保存
            if key == ord('s'):
                with open(OUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=2)
                print(f"[SAVED] {OUT_JSON}")
                break

            time.sleep(0.001)

    finally:
        for _, sn in ORDERED_CAMS:
            stop_pipeline(pipes[sn])
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

