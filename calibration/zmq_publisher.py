#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import zmq
from polymetis import RobotInterface

# ====== 写死参数 ======
POLY_IP = "127.0.0.1"
POLY_PORT = 50051

PUB_HZ = 120  # 建议提高到 120 或 200（机器扛得住的话），匹配会更准

BIND_ENDPOINTS = [
    "tcp://*:5555",
    "ipc:///tmp/panda_joints.ipc",
]
# =====================

def now_ns() -> int:
    return time.monotonic_ns()

def main():
    robot = RobotInterface(ip_address=POLY_IP, port=POLY_PORT)
    print("[POLY] connected")

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 10)   # 允许一点点缓冲（接收端会自己选对齐）
    pub.setsockopt(zmq.LINGER, 0)

    for ep in BIND_ENDPOINTS:
        pub.bind(ep)
        print("[ZMQ] PUB bind:", ep)

    time.sleep(0.2)  # slow joiner

    dt_ns = int(1e9 / float(PUB_HZ))
    next_ns = now_ns()

    seq = 0
    t_hb_ns = now_ns()

    try:
        while True:
            next_ns += dt_ns

            # 读关节
            q = robot.get_joint_positions()
            t_meas_ns = now_ns()  # 尽量贴近“读到关节”的时刻

            q7 = [float(x) for x in q[:7]]

            msg = {
                "seq": seq,
                "t_meas_ns": int(t_meas_ns),
                "t_pub_ns": int(now_ns()),
                "q": q7,
                "pub_hz": float(PUB_HZ),
            }
            pub.send_string(json.dumps(msg))

            if now_ns() - t_hb_ns > 1_000_000_000:
                print(f"[HB] seq={seq} q0={q7[0]:+.3f}")
                t_hb_ns = now_ns()

            seq += 1

            sleep_ns = next_ns - now_ns()
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1e9)

    except KeyboardInterrupt:
        print("\n[EXIT] ctrl+c")
    finally:
        pub.close(0)
        ctx.term()
        print("[DONE]")

if __name__ == "__main__":
    main()

