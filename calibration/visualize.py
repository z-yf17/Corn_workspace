#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import math
import signal
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import zmq
import xml.etree.ElementTree as ET

# ===================== 配置（稳定优先：保守参数） =====================
CAMERA_ROLE   = "right"
MAPPING_JSON  = "realsense_mapping.json"   # { "front":"serial", "left":"serial", "right":"serial" }

URDF_PATH = "./franka_panda/panda_arm_hand.urdf"
MESH_ROOT = os.path.dirname(URDF_PATH)

# 外参：npz 内应包含 "T_cam_from_base" (cam<-base)
EXTR_NPZ = f"./calib_extrinsic/{CAMERA_ROLE}/fr3_realsense_eye_to_hand.npz"

# ZMQ 关节订阅（你现有系统就用这两个）
ZMQ_ENDPOINTS = [
    "ipc:///tmp/panda_joints.ipc",
    "tcp://127.0.0.1:5555",
]

PANDA_ARM_JOINTS = [
    "panda_joint1","panda_joint2","panda_joint3","panda_joint4",
    "panda_joint5","panda_joint6","panda_joint7"
]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
FINGER_OPEN = 0.02

# RealSense（稳定优先：降低 fps）
RS_W, RS_H, RS_FPS = 640, 480, 15

# 点云：强烈建议 stride + 限点数
PC_STRIDE   = 4
DEPTH_MIN_M = 0.20
DEPTH_MAX_M = 2.00
MAX_POINTS  = 12000

# Open3D：整体降频
RENDER_HZ = 8.0     # Open3D 渲染频率
MESH_HZ   = 10.0    # mesh 更新频率（FK+顶点更新）
NORMALS_EVERY_N_MESH_UPDATES = 5  # 每 N 次 mesh 更新才重算 normals（省 CPU）

# watchdog：相机多久没新帧就重启
CAM_STALE_SEC = 2.0

# 重启策略
RS_START_RETRIES       = 3
RS_HWRESET_AFTER_FAILS = 2
# ================================================================

STOP = False


# ------------------ 信号退出（尽量干净） ------------------
def _on_sig(sig, frame):
    global STOP
    STOP = True


signal.signal(signal.SIGINT, _on_sig)
signal.signal(signal.SIGTERM, _on_sig)


# ------------------ 工具：变换 ------------------
def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], dtype=np.float64)
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=np.float64)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


def xyz_rpy_to_T(xyz, rpy):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_R(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = xyz
    return T


def axis_angle_to_R(axis, angle):
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    a = axis / n
    x,y,z = a
    c = math.cos(angle); s = math.sin(angle); C = 1 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ], dtype=np.float64)


def transform_points(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return (V @ R.T) + t.reshape(1, 3)


# ------------------ RealSense serial mapping ------------------
def load_serial_from_mapping(role: str, mapping_json: str) -> str:
    with open(mapping_json, "r", encoding="utf-8") as f:
        m = json.load(f)
    if not isinstance(m, dict):
        return ""
    v = m.get(role, "")
    return v.strip() if isinstance(v, str) else ""


# ------------------ URDF 解析（minimal） ------------------
def _parse_origin(elem):
    if elem is None:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    xyz = np.fromstring(elem.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
    rpy = np.fromstring(elem.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
    if xyz.size != 3: xyz = np.zeros(3, dtype=np.float64)
    if rpy.size != 3: rpy = np.zeros(3, dtype=np.float64)
    return xyz, rpy


def _resolve_mesh_path(filename: str) -> str:
    if filename.startswith("package://"):
        parts = filename[len("package://"):].split("/", 1)
        filename = parts[1] if len(parts) == 2 else parts[0]
    if not os.path.isabs(filename):
        filename = os.path.join(MESH_ROOT, filename)
    return filename


def load_urdf_minimal(urdf_path: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links: Dict[str, Optional[dict]] = {}
    joints: Dict[str, dict] = {}

    # links: 读 mesh（优先 collision，fallback visual）
    for link in root.findall("link"):
        lname = link.attrib["name"]
        mesh_info = None
        for tag in ["collision", "visual"]:
            node = link.find(tag)
            if node is None:
                continue
            geom = node.find("geometry")
            if geom is None:
                continue
            mesh = geom.find("mesh")
            if mesh is None:
                continue
            filename = mesh.attrib.get("filename", "")
            if not filename:
                continue
            filename = _resolve_mesh_path(filename)

            scale = mesh.attrib.get("scale", None)
            if scale is None:
                scale = np.ones(3, dtype=np.float64)
            else:
                scale = np.fromstring(scale, sep=" ", dtype=np.float64)
                if scale.size != 3:
                    scale = np.ones(3, dtype=np.float64)

            xyz, rpy = _parse_origin(node.find("origin"))
            T_link_mesh = xyz_rpy_to_T(xyz, rpy)
            mesh_info = dict(filename=filename, scale=scale, T_link_mesh=T_link_mesh)
            break
        links[lname] = mesh_info

    # joints
    child_links = set()
    for joint in root.findall("joint"):
        jname = joint.attrib["name"]
        jtype = joint.attrib.get("type", "fixed")
        parent = joint.find("parent").attrib["link"]
        child  = joint.find("child").attrib["link"]
        child_links.add(child)

        xyz, rpy = _parse_origin(joint.find("origin"))
        T_origin = xyz_rpy_to_T(xyz, rpy)

        axis_node = joint.find("axis")
        if axis_node is None:
            axis = np.array([0,0,1], dtype=np.float64)
        else:
            axis = np.fromstring(axis_node.attrib.get("xyz","0 0 1"), sep=" ", dtype=np.float64)
            if axis.size != 3:
                axis = np.array([0,0,1], dtype=np.float64)

        joints[jname] = dict(name=jname, type=jtype, parent=parent, child=child, T_origin=T_origin, axis=axis)

    # root link
    all_links = set(links.keys())
    roots = list(all_links - child_links)
    root_link = roots[0] if roots else "panda_link0"
    return links, joints, root_link


def fk_all_links(root_link: str, joints: Dict[str, dict], q_map: Dict[str, float]) -> Dict[str, np.ndarray]:
    # parent -> joints
    children = {}
    for jname, j in joints.items():
        children.setdefault(j["parent"], []).append(jname)

    T = {root_link: np.eye(4, dtype=np.float64)}
    stack = [root_link]
    while stack:
        parent_link = stack.pop()
        Tp = T[parent_link]
        for jname in children.get(parent_link, []):
            j = joints[jname]
            q = float(q_map.get(jname, 0.0))

            T_motion = np.eye(4, dtype=np.float64)
            if j["type"] in ("revolute", "continuous"):
                T_motion[:3, :3] = axis_angle_to_R(j["axis"], q)
            elif j["type"] == "prismatic":
                a = np.asarray(j["axis"], dtype=np.float64).reshape(3)
                T_motion[:3, 3] = a * q

            Tc = Tp @ j["T_origin"] @ T_motion
            T[j["child"]] = Tc
            stack.append(j["child"])

    return T


# ------------------ ZMQ joints ------------------
def make_zmq_sub(endpoints):
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.setsockopt(zmq.LINGER, 0)
    for ep in endpoints:
        sub.connect(ep)
        print("[ZMQ] SUB connect:", ep)
    return ctx, sub


def try_recv_q(sub) -> Optional[Dict[str, float]]:
    try:
        msg = sub.recv(flags=zmq.NOBLOCK)
    except zmq.Again:
        return None

    try:
        data = json.loads(msg.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    q_map: Dict[str, float] = {}

    q = data.get("q", None)
    if isinstance(q, list) and len(q) >= 7:
        for i, jn in enumerate(PANDA_ARM_JOINTS):
            q_map[jn] = float(q[i])
    elif isinstance(q, dict):
        for k, v in q.items():
            try:
                q_map[str(k)] = float(v)
            except Exception:
                pass

    # fingers（如果发布端有 finger 字段）
    if "finger" in data and isinstance(data["finger"], (int, float)):
        f = float(data["finger"])
        q_map[FINGER_JOINTS[0]] = f
        q_map[FINGER_JOINTS[1]] = f

    return q_map if q_map else None


# ------------------ multiprocessing：只保留最新 ------------------
def safe_put_latest(q: mp.Queue, item):
    try:
        q.put_nowait(item)
        return
    except Exception:
        pass
    try:
        _ = q.get_nowait()
    except Exception:
        pass
    try:
        q.put_nowait(item)
    except Exception:
        pass


@dataclass
class CloudPacket:
    ts: float
    pts_base: np.ndarray  # (N,3) float32
    cols_rgb: np.ndarray  # (N,3) float32


# ------------------ RealSense worker（子进程，不 import open3d） ------------------
def rs_worker(
    out_q: mp.Queue,
    stop_evt: mp.Event,
    serial: str,
    T_base_from_cam: np.ndarray,
    w: int, h: int, fps: int,
    stride: int, zmin: float, zmax: float,
    max_points: int,
):
    import pyrealsense2 as rs

    pipe = None
    align = None
    depth_scale = None
    intr = None
    consecutive_start_fails = 0

    def hardware_reset_local(serial_: str) -> bool:
        if not serial_:
            return False
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                try:
                    s = dev.get_info(rs.camera_info.serial_number)
                except Exception:
                    continue
                if s == serial_:
                    print(f"[RS][RESET] hardware_reset serial={serial_}", flush=True)
                    dev.hardware_reset()
                    return True
        except Exception as e:
            print("[RS][RESET] failed:", e, flush=True)
        return False

    def stop_pipe():
        nonlocal pipe
        try:
            if pipe is not None:
                pipe.stop()
        except Exception:
            pass
        pipe = None

    def start_pipe() -> bool:
        nonlocal pipe, align, depth_scale, intr
        try:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            prof = pipe.start(cfg)

            dev = prof.get_device()
            try:
                print("[RS][WORKER] using", dev.get_info(rs.camera_info.name),
                      "S/N:", dev.get_info(rs.camera_info.serial_number), flush=True)
            except Exception:
                pass

            # 减缓存
            try:
                for s in dev.query_sensors():
                    if s.supports(rs.option.frames_queue_size):
                        s.set_option(rs.option.frames_queue_size, 1.0)
            except Exception:
                pass

            depth_scale = float(dev.first_depth_sensor().get_depth_scale())
            align = rs.align(rs.stream.color)
            intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

            # warmup
            for _ in range(10):
                if stop_evt.is_set():
                    return False
                try:
                    pipe.wait_for_frames(500)
                except Exception:
                    pass

            print(f"[RS][WORKER] started {w}x{h}@{fps} depth_scale={depth_scale}", flush=True)
            return True
        except Exception as e:
            print("[RS][WORKER] start failed:", e, flush=True)
            stop_pipe()
            return False

    def compute(depth_frame, color_frame) -> Tuple[np.ndarray, np.ndarray]:
        nonlocal depth_scale, intr
        depth = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        color = np.asanyarray(color_frame.get_data()).astype(np.uint8)

        z = depth.astype(np.float32) * float(depth_scale)
        H, W = depth.shape

        z_s = z[0:H:stride, 0:W:stride]
        c_s = color[0:H:stride, 0:W:stride, :]

        mask = (z_s > zmin) & (z_s < zmax)
        if not np.any(mask):
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

        fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)

        us = np.arange(0, W, stride, dtype=np.float32)
        vs = np.arange(0, H, stride, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)

        Z = z_s[mask].astype(np.float32)
        U = uu[mask].astype(np.float32)
        V = vv[mask].astype(np.float32)

        X = (U - cx) * Z / fx
        Y = (V - cy) * Z / fy
        pts_cam = np.stack([X, Y, Z], axis=1)  # float32

        rgb = c_s[mask][:, ::-1].astype(np.float32) / 255.0  # BGR->RGB

        # 限点数
        n = pts_cam.shape[0]
        if max_points > 0 and n > max_points:
            idx = np.random.choice(n, size=max_points, replace=False)
            pts_cam = pts_cam[idx]
            rgb = rgb[idx]

        # cam -> base
        Rb = T_base_from_cam[:3, :3].astype(np.float32)
        tb = T_base_from_cam[:3, 3].astype(np.float32)
        pts_base = (pts_cam @ Rb.T) + tb.reshape(1, 3)
        return pts_base.astype(np.float32), rgb.astype(np.float32)

    # 初次启动
    if not start_pipe():
        consecutive_start_fails += 1

    fail_cnt = 0
    while not stop_evt.is_set():
        if pipe is None:
            ok = False
            for _ in range(RS_START_RETRIES):
                if stop_evt.is_set():
                    break
                ok = start_pipe()
                if ok:
                    consecutive_start_fails = 0
                    break
                consecutive_start_fails += 1
                if consecutive_start_fails >= RS_HWRESET_AFTER_FAILS:
                    hardware_reset_local(serial)
                    time.sleep(4.0)
                else:
                    time.sleep(1.0)
            if not ok:
                time.sleep(0.2)
                continue

        try:
            frames = pipe.wait_for_frames(200)
        except Exception:
            fail_cnt += 1
            if fail_cnt >= 5:
                stop_pipe()
                fail_cnt = 0
            continue

        fail_cnt = 0
        try:
            frames = align.process(frames)
            df = frames.get_depth_frame()
            cf = frames.get_color_frame()
            if not df or not cf:
                continue

            pts_base, cols = compute(df, cf)
            safe_put_latest(out_q, CloudPacket(time.time(), pts_base, cols))
        except Exception:
            print("[RS][WORKER] runtime error:\n", traceback.format_exc(), flush=True)
            stop_pipe()
            time.sleep(0.2)

    stop_pipe()
    print("[RS][WORKER] exit", flush=True)


def start_cam_process(cam_q: mp.Queue, stop_evt: mp.Event, serial: str, T_base_from_cam: np.ndarray) -> mp.Process:
    p = mp.Process(
        target=rs_worker,
        args=(cam_q, stop_evt, serial, T_base_from_cam,
              RS_W, RS_H, RS_FPS, PC_STRIDE, DEPTH_MIN_M, DEPTH_MAX_M, MAX_POINTS),
        daemon=True
    )
    p.start()
    return p


def stop_cam_process(p: Optional[mp.Process], stop_evt: mp.Event, timeout_sec: float = 2.0):
    stop_evt.set()
    if p is None:
        return
    p.join(timeout=timeout_sec)
    if p.is_alive():
        try: p.terminate()
        except Exception: pass
        p.join(timeout=timeout_sec)
    if p.is_alive():
        try: p.kill()
        except Exception: pass
        p.join(timeout=timeout_sec)


# ------------------ 主程序（Open3D 单窗口：mesh+点云） ------------------
def main():
    global STOP

    # ✅ 主进程才 import Open3D（避免子进程 OpenGL）
    import open3d as o3d

    # serial
    if not os.path.exists(MAPPING_JSON):
        raise RuntimeError(f"Missing mapping json: {MAPPING_JSON}")
    serial = load_serial_from_mapping(CAMERA_ROLE, MAPPING_JSON)
    if not serial:
        raise RuntimeError(f"Mapping invalid: {MAPPING_JSON}, need key '{CAMERA_ROLE}'")
    print(f"[CFG] role={CAMERA_ROLE} serial={serial}")

    # extrinsic：cam<-base -> base<-cam
    if not os.path.exists(EXTR_NPZ):
        raise RuntimeError(f"Extrinsic npz not found: {EXTR_NPZ}")
    r = np.load(EXTR_NPZ, allow_pickle=True)
    if "T_cam_from_base" not in r:
        raise RuntimeError(f"{EXTR_NPZ} missing key 'T_cam_from_base'")
    T_cam_from_base = r["T_cam_from_base"].astype(np.float64)
    T_base_from_cam = inv_T(T_cam_from_base)
    print("[EXTR] loaded T_cam_from_base; using base<-cam for pointcloud")

    # URDF
    if not os.path.exists(URDF_PATH):
        raise RuntimeError(f"URDF not found: {URDF_PATH}")
    links, joints, root_link = load_urdf_minimal(URDF_PATH)
    print(f"[URDF] root={root_link} links={len(links)} joints={len(joints)}")

    # load meshes
    link_mesh = {}
    mesh_count = 0
    for lname, info in links.items():
        if info is None:
            continue
        path = info["filename"]
        if (not path) or (not os.path.exists(path)):
            continue
        mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
        if mesh.is_empty():
            continue
        mesh.compute_vertex_normals()

        V0 = np.asarray(mesh.vertices).astype(np.float64)
        # apply URDF mesh scale once
        scale = np.asarray(info["scale"], dtype=np.float64).reshape(3)
        V0 = V0 * scale.reshape(1, 3)

        link_mesh[lname] = dict(
            mesh=mesh,
            V0=V0,
            T_link_mesh=info["T_link_mesh"],
        )
        mesh_count += 1

    print(f"[MESH] loaded={mesh_count}")

    # ZMQ joints
    zmq_ctx, sub = make_zmq_sub(ZMQ_ENDPOINTS)
    last_q_map = {fj: float(FINGER_OPEN) for fj in FINGER_JOINTS}
    for jn in PANDA_ARM_JOINTS:
        last_q_map.setdefault(jn, 0.0)

    # Open3D window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Mesh+Cloud ({CAMERA_ROLE})", width=1280, height=720)
    try:
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0, 0, 0], dtype=np.float64)
    except Exception:
        pass

    # axis
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis.add_geometry(axis)

    # add meshes
    for obj in link_mesh.values():
        vis.add_geometry(obj["mesh"])

    # point cloud
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    # multiprocessing camera
    cam_q = mp.Queue(maxsize=1)
    cam_stop = mp.Event()
    cam_proc = start_cam_process(cam_q, cam_stop, serial, T_base_from_cam)

    last_cam_ts = 0.0
    did_reset_view = False

    # rates
    mesh_dt = 1.0 / max(MESH_HZ, 1e-6)
    render_dt = 1.0 / max(RENDER_HZ, 1e-6)
    t_last_mesh = 0.0
    t_last_render = 0.0
    mesh_update_count = 0

    def restart_camera():
        nonlocal cam_proc, cam_stop, cam_q, last_cam_ts, did_reset_view
        print("[WATCHDOG] restarting camera process...")
        stop_cam_process(cam_proc, cam_stop, timeout_sec=2.0)
        cam_stop = mp.Event()
        cam_q = mp.Queue(maxsize=1)
        cam_proc = start_cam_process(cam_q, cam_stop, serial, T_base_from_cam)
        last_cam_ts = 0.0
        did_reset_view = False

    def _quit(_vis):
        global STOP
        STOP = True
        return False


    def _restart(_vis):
        restart_camera()
        return False

    # key: q/ESC exit, r restart cam
    vis.register_key_callback(ord('Q'), _quit)
    vis.register_key_callback(ord('q'), _quit)
    vis.register_key_callback(256, _quit)  # ESC
    vis.register_key_callback(ord('R'), _restart)
    vis.register_key_callback(ord('r'), _restart)

    try:
        while not STOP:
            # window events
            if not vis.poll_events():
                break

            now = time.time()

            # ---- joints：尽量只取最新 ----
            for _ in range(5):
                qm = try_recv_q(sub)
                if qm is None:
                    break
                last_q_map.update(qm)
            for fj in FINGER_JOINTS:
                last_q_map.setdefault(fj, float(FINGER_OPEN))

            # ---- mesh 更新（降频）----
            if (now - t_last_mesh) >= mesh_dt:
                t_last_mesh = now
                mesh_update_count += 1

                T_base_link = fk_all_links(root_link, joints, last_q_map)

                # 更新每个 link mesh 顶点
                for lname, obj in link_mesh.items():
                    if lname not in T_base_link:
                        continue
                    Tb = T_base_link[lname]
                    T_base_mesh = Tb @ obj["T_link_mesh"]
                    Vt = transform_points(obj["V0"], T_base_mesh)

                    m = obj["mesh"]
                    np.asarray(m.vertices)[:] = Vt

                    # normals 低频算（很耗）
                    if NORMALS_EVERY_N_MESH_UPDATES > 0 and (mesh_update_count % NORMALS_EVERY_N_MESH_UPDATES == 0):
                        m.compute_vertex_normals()

                    vis.update_geometry(m)

            # ---- 点云更新：取最新 packet ----
            pkt = None
            try:
                pkt = cam_q.get_nowait()
            except Exception:
                pkt = None

            if pkt is not None:
                last_cam_ts = pkt.ts
                if pkt.pts_base.shape[0] > 0:
                    pcd.points = o3d.utility.Vector3dVector(pkt.pts_base.astype(np.float64))
                    pcd.colors = o3d.utility.Vector3dVector(pkt.cols_rgb.astype(np.float64))
                    vis.update_geometry(pcd)

                    if not did_reset_view:
                        try:
                            vis.reset_view_point(True)
                        except Exception:
                            pass
                        did_reset_view = True

            # ---- watchdog ----
            if last_cam_ts > 0 and (time.time() - last_cam_ts) > CAM_STALE_SEC:
                restart_camera()

            # ---- 渲染降频 ----
            if (now - t_last_render) >= render_dt:
                t_last_render = now
                vis.update_renderer()
            else:
                # 让出 CPU，减少桌面压力
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")

    finally:
        stop_cam_process(cam_proc, cam_stop, timeout_sec=2.0)
        try:
            sub.close(0)
            zmq_ctx.term()
        except Exception:
            pass
        try:
            vis.destroy_window()
        except Exception:
            pass
        print("[DONE]")


if __name__ == "__main__":
    # 必须在启动子进程前设置
    mp.set_start_method("spawn", force=True)
    main()
