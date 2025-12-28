#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import signal
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import zmq
import torch
from polymetis import RobotInterface
import grpc

torch.set_num_threads(1)

# =========================
# Hard-coded parameters
# =========================
ZMQ_ADDR = "tcp://127.0.0.1:5557"
ZMQ_TOPIC = b"pcd_refined/base"

# ---- point cloud -> target logic ----
MIN_POINTS_FOR_VALID = 80
NEAR_XY_RADIUS = 0.03          # m
Z_OFFSET = 0.2                 # m

# ---- target smoothing ----
TARGET_EMA_ALPHA = 0.20
STALE_TARGET_SEC = 1.0

# ---- control loop ----
CONTROL_HZ = 50.0
MAX_SPEED_MPS = 0.05

# ---- controller selection ----
USE_CARTESIAN_IF_AVAILABLE = True
CART_KP = [200.0, 200.0, 200.0, 25.0, 25.0, 25.0]
CART_KD = [20.0,  20.0,  20.0,  5.0,  5.0,  5.0]
JOINT_KP = [300, 300, 300, 200, 150, 120, 80]
JOINT_KD = [10,  10,  10,   6,   5,  3.5, 2.5]

# ---- gRPC anti-freeze ----
GRPC_TIMEOUT_SEC = 0.20            # 单次 RPC 最多等 200ms，避免“卡死”
STATE_REFRESH_SEC = 2.0            # 每 2s 刷一次真实关节位置作为 IK seed（可关掉）
PRINT_IK_FAIL_EVERY = 20           # IK 连续失败时的打印频率

# =========================
# Shared state
# =========================
@dataclass
class Shared:
    lock: threading.Lock = field(default_factory=threading.Lock)

    # latest cloud
    pts_base: Optional[np.ndarray] = None
    cols_rgb: Optional[np.ndarray] = None
    ts_pub: float = 0.0
    ts_recv: float = 0.0

    # target (raw + filtered)
    target_raw: Optional[np.ndarray] = None
    target_filt: Optional[np.ndarray] = None
    target_ts: float = 0.0

    rx_count: int = 0
    rx_bad: int = 0


STOP = False
def _on_sig(sig, frame):
    global STOP
    STOP = True

signal.signal(signal.SIGINT, _on_sig)
signal.signal(signal.SIGTERM, _on_sig)

def now_s() -> float:
    return time.time()

def perf_s() -> float:
    return time.perf_counter()

# =========================
# Decode + target compute
# =========================
def decode_pcd(parts) -> Optional[Tuple[float, dict, np.ndarray, np.ndarray]]:
    """
    Expected multipart:
      [topic, ts_str, meta_json, pts_bytes, cols_bytes]
    """
    if len(parts) < 5:
        return None
    if parts[0] != ZMQ_TOPIC:
        return None

    try:
        ts = float(parts[1].decode("utf-8"))
    except Exception:
        ts = 0.0

    try:
        meta = json.loads(parts[2].decode("utf-8"))
        if not isinstance(meta, dict):
            meta = {}
    except Exception:
        meta = {}

    pts_shape = meta.get("points_shape", None)
    cols_shape = meta.get("colors_shape", None)
    if pts_shape is None or len(pts_shape) != 2:
        return None
    if cols_shape is None or len(cols_shape) != 2:
        return None

    n = int(pts_shape[0])
    if n <= 0 or int(cols_shape[0]) != n:
        return None

    pts = np.frombuffer(parts[3], dtype=np.float32)
    cols = np.frombuffer(parts[4], dtype=np.float32)
    if pts.size != n * 3 or cols.size != n * 3:
        return None

    pts = pts.reshape(n, 3).copy()
    cols = cols.reshape(n, 3).copy()
    return ts, meta, pts, cols


def compute_target_from_cloud(pts_base: np.ndarray) -> Optional[np.ndarray]:
    if pts_base is None or pts_base.shape[0] < MIN_POINTS_FOR_VALID:
        return None

    P = pts_base
    mean_xy = P[:, :2].mean(axis=0)

    dxy = P[:, :2] - mean_xy.reshape(1, 2)
    dist = np.sqrt((dxy * dxy).sum(axis=1))
    near = dist <= float(NEAR_XY_RADIUS)

    if np.any(near):
        z_max = float(P[near, 2].max())
    else:
        z_max = float(P[:, 2].max())

    return np.array([mean_xy[0], mean_xy[1], z_max + float(Z_OFFSET)], dtype=np.float32)


def ema_update(prev: Optional[np.ndarray], new: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return new.copy()
    return (1.0 - alpha) * prev + alpha * new


# =========================
# ZMQ receiver thread
# =========================
def pcd_receiver(shared: Shared):
    global STOP

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.RCVHWM, 2)

    sub.connect(ZMQ_ADDR)
    sub.setsockopt(zmq.SUBSCRIBE, ZMQ_TOPIC)
    print(f"[ZMQ] SUB connect {ZMQ_ADDR} | subscribe {ZMQ_TOPIC.decode('utf-8')}")

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    last_print = now_s()

    try:
        while not STOP:
            socks = dict(poller.poll(timeout=100))
            if sub not in socks:
                continue

            latest_parts = None
            while True:
                try:
                    parts = sub.recv_multipart(flags=zmq.NOBLOCK)
                    latest_parts = parts
                except zmq.Again:
                    break

            if latest_parts is None:
                continue

            decoded = decode_pcd(latest_parts)
            if decoded is None:
                with shared.lock:
                    shared.rx_bad += 1
                continue

            ts_pub, meta, pts, cols = decoded
            t_recv = now_s()

            tgt = compute_target_from_cloud(pts)

            with shared.lock:
                shared.pts_base = pts
                shared.cols_rgb = cols
                shared.ts_pub = float(ts_pub)
                shared.ts_recv = float(t_recv)
                shared.rx_count += 1
                if tgt is not None:
                    shared.target_raw = tgt
                    shared.target_filt = ema_update(shared.target_filt, tgt, float(TARGET_EMA_ALPHA))
                    shared.target_ts = float(t_recv)

            if (t_recv - last_print) >= 1.0:
                with shared.lock:
                    n = 0 if shared.pts_base is None else int(shared.pts_base.shape[0])
                    age = (t_recv - shared.ts_pub) if shared.ts_pub > 0 else 0.0
                    raw = None if shared.target_raw is None else shared.target_raw.copy()
                    fil = None if shared.target_filt is None else shared.target_filt.copy()
                    bad = shared.rx_bad
                    cnt = shared.rx_count
                if raw is None:
                    print(f"[PCD] rx={cnt} bad={bad} N={n} age={age:.3f}s | target=None")
                else:
                    print(f"[PCD] rx={cnt} bad={bad} N={n} age={age:.3f}s | "
                          f"raw=[{raw[0]:+.4f},{raw[1]:+.4f},{raw[2]:+.4f}] "
                          f"filt=[{fil[0]:+.4f},{fil[1]:+.4f},{fil[2]:+.4f}]")
                last_print = t_recv

    finally:
        try:
            poller.unregister(sub)
        except Exception:
            pass
        try:
            sub.close(0)
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass
        print("[ZMQ] receiver exit.")


# =========================
# Polymetis helpers
# =========================
def start_controller(robot: RobotInterface) -> str:
    if USE_CARTESIAN_IF_AVAILABLE:
        try:
            robot.start_cartesian_impedance(kp=CART_KP, kd=CART_KD)
            print("[ROBOT] start_cartesian_impedance OK.")
            return "cartesian"
        except Exception as e:
            print(f"[ROBOT] start_cartesian_impedance failed -> fallback joint. ({e})")

    robot.start_joint_impedance(kp=JOINT_KP, kd=JOINT_KD)
    print("[ROBOT] start_joint_impedance OK.")
    return "joint"


def monkeypatch_grpc_timeouts(robot: RobotInterface, timeout_s: float):
    """
    给 gRPC stub 默认加 deadline，防止无限阻塞导致“卡死”。
    """
    if not hasattr(robot, "grpc_connection"):
        print("[GRPC] robot has no grpc_connection attr; skip timeout patch.")
        return

    conn = robot.grpc_connection

    if hasattr(conn, "UpdateController"):
        _orig = conn.UpdateController
        def _uc(req, timeout=None, **kwargs):
            if timeout is None:
                timeout = timeout_s
            return _orig(req, timeout=timeout, **kwargs)
        conn.UpdateController = _uc
        print(f"[GRPC] Patch UpdateController timeout={timeout_s:.3f}s")

    if hasattr(conn, "GetRobotState"):
        _orig2 = conn.GetRobotState
        def _grs(req, timeout=None, **kwargs):
            if timeout is None:
                timeout = timeout_s
            return _orig2(req, timeout=timeout, **kwargs)
        conn.GetRobotState = _grs
        print(f"[GRPC] Patch GetRobotState timeout={timeout_s:.3f}s")


# =========================
# Control loop (translation only)
# =========================
def control_loop(shared: Shared):
    global STOP

    robot = RobotInterface()
    monkeypatch_grpc_timeouts(robot, GRPC_TIMEOUT_SEC)

    mode = start_controller(robot)

    # 只在开始取一次真实姿态与关节角
    ee_pos_t, ee_quat_t = robot.get_ee_pose()
    ee_pos = ee_pos_t.detach().cpu().numpy().astype(np.float32).reshape(3)
    ee_quat = ee_quat_t.detach().cpu().numpy().astype(np.float32).reshape(4)

    q_seed = robot.get_joint_positions()   # torch.Tensor
    t_last_state_refresh = now_s()

    print(f"[EE] cur_pos=[{ee_pos[0]:+.3f},{ee_pos[1]:+.3f},{ee_pos[2]:+.3f}] "
          f"cur_quat=[{ee_quat[0]:+.3f},{ee_quat[1]:+.3f},{ee_quat[2]:+.3f},{ee_quat[3]:+.3f}]")

    # wait first target
    print("[CTRL] waiting for valid point cloud / target...")
    while not STOP:
        with shared.lock:
            tgt = None if shared.target_filt is None else shared.target_filt.copy()
            n = 0 if shared.pts_base is None else int(shared.pts_base.shape[0])
            ts_pub = shared.ts_pub
        if tgt is not None and n >= MIN_POINTS_FOR_VALID:
            age = (now_s() - ts_pub) if ts_pub > 0 else 0.0
            print(f"[CTRL] got target: N={n} age={age:.3f}s target_filt=[{tgt[0]:+.4f},{tgt[1]:+.4f},{tgt[2]:+.4f}]")
            break
        time.sleep(0.02)
    if STOP:
        return

    dt = 1.0 / float(CONTROL_HZ)
    max_step = float(MAX_SPEED_MPS) * dt

    p_cmd = ee_pos.copy().astype(np.float32)

    slip = 0
    upd_hist = []
    last_stat = now_s()
    next_t = perf_s()

    ik_fail = 0
    restart_count = 0
    MAX_RESTART = 5

    try:
        while not STOP:
            t_loop0 = perf_s()

            # target (filtered). stale -> hold
            with shared.lock:
                tgt = None if shared.target_filt is None else shared.target_filt.copy()
                tgt_ts = shared.target_ts
                n = 0 if shared.pts_base is None else int(shared.pts_base.shape[0])
                age = (now_s() - shared.ts_pub) if shared.ts_pub > 0 else 0.0

            if tgt is None or (now_s() - tgt_ts) > float(STALE_TARGET_SEC):
                tgt = p_cmd.copy()

            # speed-limited step
            d = (tgt.astype(np.float32) - p_cmd).astype(np.float32)
            dist = float(np.linalg.norm(d))
            if dist > 1e-8:
                step = min(dist, max_step)
                p_cmd = p_cmd + (d / dist) * step

            # 可选：每隔一段时间刷新一次真实关节角作为 IK seed（降低漂移）
            if STATE_REFRESH_SEC > 0 and (now_s() - t_last_state_refresh) >= float(STATE_REFRESH_SEC):
                try:
                    q_seed = robot.get_joint_positions()
                except grpc.RpcError as e:
                    # 超时/失败就先不刷新
                    pass
                t_last_state_refresh = now_s()

            # 本地 IK（不再每次取状态） + 单次 RPC 更新关节目标
            try:
                pos_t = torch.from_numpy(p_cmd.astype(np.float32))
                quat_t = torch.from_numpy(ee_quat.astype(np.float32))
                q_des, success = robot.solve_inverse_kinematics(pos_t, quat_t, q_seed)
                if not success:
                    ik_fail += 1
                    if (ik_fail % PRINT_IK_FAIL_EVERY) == 1:
                        print(f"[IK] failed x{ik_fail} (holding last command). p_cmd={p_cmd.tolist()}")
                else:
                    ik_fail = 0
                    t0 = perf_s()
                    robot.update_desired_joint_positions(q_des)
                    upd_hist.append(perf_s() - t0)
                    q_seed = q_des  # 用解出来的关节角做下一次 seed

            except grpc.RpcError as e:
                msg = ""
                try:
                    msg = e.details()
                except Exception:
                    msg = str(e)
                code = getattr(e, "code", lambda: None)()
                print(f"[CTRL-ERR] gRPC error: code={code} msg={msg}")

                # 超时/无控制器：尝试重启 controller
                if restart_count < MAX_RESTART:
                    restart_count += 1
                    print(f"[CTRL] try restart controller ({restart_count}/{MAX_RESTART}) ...")
                    try:
                        mode = start_controller(robot)
                        # 重启后重新取一次 seed
                        q_seed = robot.get_joint_positions()
                        time.sleep(0.05)
                        continue
                    except Exception as re:
                        print(f"[CTRL] restart failed: {re}")
                        break
                else:
                    print("[CTRL] restart limit reached -> exit.")
                    break

            except Exception as e:
                print(f"[CTRL-ERR] exception: {e}")
                break

            # pacing
            nowp = perf_s()
            if nowp > next_t + 2.0 * dt:
                slip += 1
                next_t = nowp
            else:
                sleep_s = next_t - nowp
                if sleep_s > 0:
                    time.sleep(sleep_s)
                next_t += dt

            # stats
            if (now_s() - last_stat) >= 1.0:
                tail = upd_hist[-int(CONTROL_HZ):] if len(upd_hist) > int(CONTROL_HZ) else upd_hist
                if len(tail) > 0:
                    arr = np.array(tail, dtype=np.float32)
                    avg_u = float(arr.mean())
                    max_u = float(arr.max())
                else:
                    avg_u, max_u = 0.0, 0.0

                loop_ms = (perf_s() - t_loop0) * 1000.0
                print(f"[STAT] hz={CONTROL_HZ:.1f} slip={slip} loop={loop_ms:.2f}ms upd_avg={avg_u*1000:.2f}ms upd_max={max_u*1000:.2f}ms | "
                      f"pcdN={n} age={age:.3f}s ik_fail={ik_fail} | "
                      f"cmd=[{p_cmd[0]:+.3f},{p_cmd[1]:+.3f},{p_cmd[2]:+.3f}]")
                last_stat = now_s()

    finally:
        try:
            robot.stop()
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass
        print("[CTRL] exit.")


# =========================
# Main
# =========================
def main():
    global STOP
    shared = Shared()

    th_rx = threading.Thread(target=pcd_receiver, args=(shared,), daemon=True)
    th_ctrl = threading.Thread(target=control_loop, args=(shared,), daemon=True)

    th_rx.start()
    th_ctrl.start()

    while not STOP:
        time.sleep(0.1)

    STOP = True
    th_ctrl.join(timeout=2.0)
    th_rx.join(timeout=2.0)
    print("[EXIT] done.")


if __name__ == "__main__":
    main()
