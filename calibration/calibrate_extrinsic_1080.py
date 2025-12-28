#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, random
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from collections import deque

# ================== 参数写死在这里 ==================
# 你的新采集数据目录（改成你实际的）
camera = 'left'
SAMPLES_DIR = "./handeye_data_rs_1080p_sync/" + camera   # e.g. ./handeye_data_rs_1080p_sync

OUT_NPZ = "./calib_extrinsic/" + camera +"/fr3_realsense_eye_to_hand_1080.npz"
OUT_TXT = "./calib_extrinsic/" + camera + "/fr3_realsense_eye_to_hand_1080.txt"

# URDF（你解压 franka_panda.zip 后的路径，按需改）
URDF_PATH = "./franka_panda/panda_arm_hand.urdf"

# FK 链的起点/终点（非常关键：要跟你“定义的 ee”一致）
BASE_LINK = "panda_link0"
EE_LINK   = "panda_hand"    # 常用：panda_hand；如果你想用法兰帧可改为 "panda_link8"

# q 的关节顺序（与你保存的 q 顺序一致；你发布端 q[:7] 通常就是这个顺序）
JOINT_ORDER = [f"panda_joint{i}" for i in range(1, 8)]

RANSAC_ITERS = 800
RANSAC_SAMPLE_PAIRS = 30

POOL_NPAIRS = 3000
MIN_REL_ROT_DEG = 5.0
MIN_REL_TRANS_M = 0.01

INLIER_ROT_DEG = 7.0
INLIER_TRANS_M = 0.05

SEED = 1
# ===================================================


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rot_err_deg(R):
    tr = float(np.trace(R))
    c = (tr - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def T_to_rotvec(T):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec.reshape(3)


def rpy_to_R(rpy):
    """URDF rpy: roll,pitch,yaw ; Rot = Rz(yaw) * Ry(pitch) * Rx(roll)"""
    r, p, y = [float(x) for x in rpy]
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]], dtype=np.float64)
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


def axis_angle_to_R(axis, theta):
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-12 or abs(theta) < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / n
    x, y, z = axis
    K = np.array([[0, -z,  y],
                  [z,  0, -x],
                  [-y, x,  0]], dtype=np.float64)
    I = np.eye(3, dtype=np.float64)
    return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def parse_urdf_joints(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints = {}
    for j in root.findall("joint"):
        name = j.get("name")
        jtype = j.get("type")

        parent = j.find("parent").get("link")
        child = j.find("child").get("link")

        origin = j.find("origin")
        if origin is not None:
            xyz = [float(x) for x in (origin.get("xyz") or "0 0 0").split()]
            rpy = [float(x) for x in (origin.get("rpy") or "0 0 0").split()]
        else:
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]

        axis_el = j.find("axis")
        axis = [float(x) for x in (axis_el.get("xyz") if axis_el is not None else "0 0 0").split()]

        joints[name] = dict(
            type=jtype,
            parent=parent,
            child=child,
            xyz=np.asarray(xyz, dtype=np.float64),
            rpy=np.asarray(rpy, dtype=np.float64),
            axis=np.asarray(axis, dtype=np.float64),
        )
    return joints


def find_joint_chain(joints, base_link, ee_link):
    """在 joint 图里找 base_link -> ee_link 的 joint 序列（BFS）"""
    graph = {}
    for jname, info in joints.items():
        graph.setdefault(info["parent"], []).append((info["child"], jname))

    prev = {base_link: (None, None)}
    dq = deque([base_link])

    while dq:
        u = dq.popleft()
        if u == ee_link:
            break
        for v, jname in graph.get(u, []):
            if v not in prev:
                prev[v] = (u, jname)
                dq.append(v)

    if ee_link not in prev:
        raise RuntimeError(f"Cannot find chain from {base_link} to {ee_link} in URDF.")

    chain = []
    cur = ee_link
    while cur != base_link:
        pu, jname = prev[cur]
        chain.append(jname)
        cur = pu
    chain.reverse()
    return chain


def joint_transform(info, q):
    """parent->child transform for this joint"""
    T0 = np.eye(4, dtype=np.float64)
    R0 = rpy_to_R(info["rpy"])
    T0[:3, :3] = R0
    T0[:3, 3] = info["xyz"]

    jtype = info["type"]
    if jtype in ("revolute", "continuous"):
        Rm = axis_angle_to_R(info["axis"], q)
        Tm = np.eye(4, dtype=np.float64)
        Tm[:3, :3] = Rm
        return T0 @ Tm
    elif jtype == "prismatic":
        axis = info["axis"].copy()
        n = float(np.linalg.norm(axis))
        if n > 1e-12:
            axis /= n
        Tm = np.eye(4, dtype=np.float64)
        Tm[:3, 3] = axis * float(q)
        return T0 @ Tm
    else:  # fixed
        return T0


class PandaFK:
    def __init__(self, urdf_path, base_link, ee_link, joint_order):
        self.joints = parse_urdf_joints(urdf_path)
        self.chain = find_joint_chain(self.joints, base_link, ee_link)
        self.joint_order = list(joint_order)
        self.base_link = base_link
        self.ee_link = ee_link

        # sanity: chain should contain panda_joint1..7 + fixed joints
        print("[FK] base:", base_link, "ee:", ee_link)
        print("[FK] chain joints:", " -> ".join(self.chain))

    def fk_T_base_ee(self, q7):
        q7 = np.asarray(q7, dtype=np.float64).reshape(-1)
        if q7.size < 7:
            raise ValueError("q7 must have at least 7 elements")

        qmap = {jn: float(q7[i]) for i, jn in enumerate(self.joint_order)}

        T = np.eye(4, dtype=np.float64)
        for jname in self.chain:
            info = self.joints[jname]
            q = qmap.get(jname, 0.0)  # fixed joints not in qmap
            T = T @ joint_transform(info, q)
        return T


def mean_rotation_markley(R_list):
    def R_to_q_xyzw(R):
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / S
                x = 0.25 * S
                y = (R[0, 1] + R[1, 0]) / S
                z = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / S
                x = (R[0, 1] + R[1, 0]) / S
                y = 0.25 * S
                z = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / S
                x = (R[0, 2] + R[2, 0]) / S
                y = (R[1, 2] + R[2, 1]) / S
                z = 0.25 * S

        q = np.array([x, y, z, w], dtype=np.float64)
        q /= (np.linalg.norm(q) + 1e-12)
        return q

    def q_xyzw_to_R(q):
        x, y, z, w = q
        n = np.linalg.norm(q)
        if n < 1e-12:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = x / n, y / n, z / n, w / n
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    A = np.zeros((4, 4), dtype=np.float64)
    for R in R_list:
        q = R_to_q_xyzw(R)
        if q[3] < 0:
            q = -q
        A += np.outer(q, q)

    vals, vecs = np.linalg.eigh(A)
    q = vecs[:, np.argmax(vals)]
    if q[3] < 0:
        q = -q
    q /= (np.linalg.norm(q) + 1e-12)
    return q_xyzw_to_R(q)


def solve_X_from_rel(A_rel_list, B_rel_list):
    wA = np.array([T_to_rotvec(A) for A in A_rel_list], dtype=np.float64)
    wB = np.array([T_to_rotvec(B) for B in B_rel_list], dtype=np.float64)

    H = np.zeros((3, 3), dtype=np.float64)
    for a, b in zip(wA, wB):
        H += np.outer(a, b)

    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    I = np.eye(3, dtype=np.float64)
    M = []
    y = []
    for A, B in zip(A_rel_list, B_rel_list):
        RA = A[:3, :3]
        tA = A[:3, 3]
        tB = B[:3, 3]
        M.append(RA - I)
        y.append(R @ tB - tA)

    M = np.vstack(M)
    y = np.hstack(y).reshape(-1, 1)
    t, *_ = np.linalg.lstsq(M, y, rcond=None)
    t = t.reshape(3)

    X = np.eye(4, dtype=np.float64)
    X[:3, :3] = R
    X[:3, 3] = t
    return X


def compute_Y(X, A_list, B_list):
    Ys = [inv_T(A) @ inv_T(X) @ B for A, B in zip(A_list, B_list)]
    R = mean_rotation_markley([Y[:3, :3] for Y in Ys])
    t = np.mean(np.stack([Y[:3, 3] for Y in Ys], axis=0), axis=0)

    Y = np.eye(4, dtype=np.float64)
    Y[:3, :3] = R
    Y[:3, 3] = t
    return Y


def eval_errors(X, Y, A_list, B_list):
    rots = []
    trans = []
    for A, B in zip(A_list, B_list):
        B_pred = X @ A @ Y
        dT = inv_T(B) @ B_pred
        rots.append(rot_err_deg(dT[:3, :3]))
        trans.append(float(np.linalg.norm(dT[:3, 3])))
    return np.array(rots, dtype=np.float64), np.array(trans, dtype=np.float64)


def build_pair_pool(A_list, B_list, n_pairs, min_rot_deg, min_trans_m):
    n = len(A_list)
    pairsA, pairsB = [], []
    tries = 0
    while len(pairsA) < n_pairs and tries < n_pairs * 60:
        tries += 1
        i = random.randrange(0, n - 1)
        j = random.randrange(i + 1, n)

        # 相对运动（注意这里保持你原脚本的构造方式）
        A_rel = B_list[j] @ inv_T(B_list[i])  # camera side
        B_rel = A_list[j] @ inv_T(A_list[i])  # robot side

        if rot_err_deg(A_rel[:3, :3]) < min_rot_deg and np.linalg.norm(A_rel[:3, 3]) < min_trans_m:
            continue

        pairsA.append(A_rel)
        pairsB.append(B_rel)

    if len(pairsA) < n_pairs:
        print(f"[POOL] only built {len(pairsA)}/{n_pairs} pairs (motion too small?)")
    return pairsA, pairsB


def stats(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    if not os.path.exists(URDF_PATH):
        raise RuntimeError(f"URDF not found: {URDF_PATH}")

    fk = PandaFK(URDF_PATH, BASE_LINK, EE_LINK, JOINT_ORDER)

    paths = sorted(glob.glob(os.path.join(SAMPLES_DIR, "sample_*.npz")))
    if not paths:
        raise RuntimeError(f"no samples under: {SAMPLES_DIR}")

    A_list, B_list = [], []
    kept = 0
    for p in paths:
        d = np.load(p, allow_pickle=True)

        if "T_cam_tag" not in d:
            raise RuntimeError(f"{p} missing T_cam_tag")
        B = d["T_cam_tag"].astype(np.float64)

        # 优先用文件里已有的 T_base_ee；否则用 q 做 FK
        if "T_base_ee" in d:
            A = d["T_base_ee"].astype(np.float64)
        elif "q" in d:
            A = fk.fk_T_base_ee(d["q"])
        else:
            raise RuntimeError(f"{p} missing both T_base_ee and q")

        A_list.append(A)
        B_list.append(B)
        kept += 1

    print(f"[LOAD] valid={kept}")

    poolA, poolB = build_pair_pool(
        A_list, B_list,
        n_pairs=POOL_NPAIRS,
        min_rot_deg=MIN_REL_ROT_DEG,
        min_trans_m=MIN_REL_TRANS_M,
    )
    print(f"[POOL] pairs={len(poolA)}")

    best = None
    best_inliers = -1
    best_score = 1e18

    for it in range(RANSAC_ITERS):
        idxs = np.random.choice(len(poolA), size=RANSAC_SAMPLE_PAIRS, replace=False)
        A_rel = [poolA[i] for i in idxs]
        B_rel = [poolB[i] for i in idxs]

        X = solve_X_from_rel(A_rel, B_rel)

        # 先粗算一次 Y，再找内点，然后用内点重算一次 Y（更稳）
        Y0 = compute_Y(X, A_list, B_list)
        rots0, trans0 = eval_errors(X, Y0, A_list, B_list)
        in0 = (rots0 < INLIER_ROT_DEG) & (trans0 < INLIER_TRANS_M)
        nin0 = int(in0.sum())
        if nin0 >= 3:
            A_in = [A_list[i] for i in np.where(in0)[0]]
            B_in = [B_list[i] for i in np.where(in0)[0]]
            Y = compute_Y(X, A_in, B_in)
        else:
            Y = Y0

        rots, trans = eval_errors(X, Y, A_list, B_list)
        inliers = (rots < INLIER_ROT_DEG) & (trans < INLIER_TRANS_M)
        nin = int(inliers.sum())

        score = (float(np.median(rots[inliers])) if nin > 0 else 1e9) + \
                100.0 * (float(np.median(trans[inliers])) if nin > 0 else 1e9)

        if nin > best_inliers or (nin == best_inliers and score < best_score):
            best_inliers = nin
            best_score = score
            best = (X, Y, rots, trans, inliers)

    X, Y, rots, trans, inliers = best
    print("\n[BEST]")
    print(f"  inliers={int(inliers.sum())}/{len(A_list)}")
    print("  rot(deg):", stats(rots))
    print("  trans(mm):", stats(trans * 1000.0))

    ensure_dir(os.path.dirname(OUT_NPZ))

    np.savez_compressed(
        OUT_NPZ,
        T_cam_from_base=X.astype(np.float64),   # cam<-base
        T_ee_from_tag=Y.astype(np.float64),     # ee<-tag  (ee 是你上面选的 EE_LINK)
        inliers=inliers.astype(np.uint8),
        cfg=dict(
            samples_dir=SAMPLES_DIR,
            urdf_path=URDF_PATH,
            base_link=BASE_LINK,
            ee_link=EE_LINK,
            joint_order=JOINT_ORDER,
            ransac_iters=RANSAC_ITERS,
            ransac_sample_pairs=RANSAC_SAMPLE_PAIRS,
            pool_npairs=POOL_NPAIRS,
            min_rel_rot_deg=MIN_REL_ROT_DEG,
            min_rel_trans_m=MIN_REL_TRANS_M,
            inlier_rot_deg=INLIER_ROT_DEG,
            inlier_trans_m=INLIER_TRANS_M,
            seed=SEED,
        ),
    )

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"SAMPLES_DIR: {SAMPLES_DIR}\n")
        f.write(f"URDF_PATH: {URDF_PATH}\n")
        f.write(f"BASE_LINK: {BASE_LINK}\n")
        f.write(f"EE_LINK: {EE_LINK}\n")
        f.write(f"inliers: {int(inliers.sum())}/{len(A_list)}\n\n")
        f.write("T_cam_from_base (cam<-base):\n")
        f.write(np.array2string(X, formatter={"float_kind": lambda x: f"{x: .8f}"}))
        f.write("\n\nT_ee_from_tag (ee<-tag):\n")
        f.write(np.array2string(Y, formatter={"float_kind": lambda x: f"{x: .8f}"}))
        f.write("\n")

    print("\n[DONE] saved:")
    print(" ", OUT_NPZ)
    print(" ", OUT_TXT)


if __name__ == "__main__":
    main()

