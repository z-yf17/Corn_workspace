#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import re
import logging
import numpy as np
import cv2
import imageio
import open3d as o3d

os.environ["PYTHONBREAKPOINT"] = "0"
try:
    import pdb
    pdb.set_trace = lambda *a, **k: None
except Exception:
    pass

# =========================
# ✅ 只需要改这里
# =========================
SESSION_DIR = "/home/galbot/ros_noetic_docker/FoundationPose/bundlesdf/rs_rgbd_aruco_record/left/session_20251229_173003"
STRIDE = 2
MAX_FRAMES = -1
FORCE_MARKER_ID = 518

OUT_DIR = ""    # 留空 = SESSION_DIR/nerf_out
OUT_OBJ = ""    # 留空 = OUT_DIR/model.obj

DISABLE_GRIDENCODER = False

# bounds 采样
MAX_PTS_PER_FRAME = 20000
MAX_PTS_TOTAL = 400000
DEPTH_MAX_M = 3.0
ROBUST_PCT = 1.0

# 有效性筛
MIN_MASK_PIX = 800
MIN_VALID_DEPTH_PIX = 800

# mask 严格使用（强烈建议）
HARD_MASK = True
MASK_ERODE_ITERS = 1
MASK_CLOSE_ITERS = 0

# mesh 更粗更稳（碎片更少）
MESH_VOXEL_MULT = 1.5   # 1.5~3.0

# ✅ 新增：利用 tag 平面去掉“底面平台”
# 逻辑：在 tag 坐标系中，平台≈z=0；删除 “靠近 z=0 且法向接近 z轴(水平面)” 的三角形
REMOVE_TAG_PLANE_FACES = True
PLANE_BAND_M = 0.010    # 删除 |z|<band 的水平面三角形（建议 0.006~0.02）
PLANE_COS_TH = 0.95     # |n·z|>cos 认为是水平面（0.90~0.98）
Z_CUT_M = 0.001         # 可选：再把 z<Z_CUT 的东西直接裁掉（0~0.003）
# =========================

logging.basicConfig(level=logging.INFO, format="%(message)s")

CODE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CODE_DIR, ".."))
sys.path.append(ROOT_DIR)

from nerf_runner import NerfRunner

glcam_in_cvcam = np.array([
    [1.0,  0.0,  0.0, 0.0],
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0, -1.0, 0.0],
    [0.0,  0.0,  0.0, 1.0],
], dtype=np.float64)


def load_cfg(cfg_path: str):
    try:
        import yaml as pyyaml
        if hasattr(pyyaml, "safe_load"):
            with open(cfg_path, "r", encoding="utf-8") as f:
                return pyyaml.safe_load(f)
    except Exception:
        pass
    from ruamel.yaml import YAML
    y = YAML(typ="safe")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return y.load(f)


def pick_default_cfg():
    for p in [os.path.join(CODE_DIR, "config_linemod.yml"),
              os.path.join(CODE_DIR, "config_ycbv.yml")]:
        if os.path.exists(p):
            return p
    raise RuntimeError("Cannot find config_linemod.yml or config_ycbv.yml")


def preprocess_data_fallback(rgbs, depths, masks, normal_maps=None, poses=None, sc_factor=1.0, translation=None):
    if translation is None:
        translation = np.zeros(3, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)

    rgbs = np.asarray(rgbs)
    depths = np.asarray(depths)
    masks = np.asarray(masks)

    rgbs_ = rgbs.astype(np.float32)
    if rgbs_.max() > 1.5:
        rgbs_ /= 255.0

    depths_ = depths.astype(np.float32) * float(sc_factor)
    if depths_.ndim == 3:
        depths_ = depths_[..., None]
    elif depths_.ndim == 4 and depths_.shape[-1] != 1:
        depths_ = depths_[..., :1]

    masks_ = (masks > 0).astype(np.uint8)
    if masks_.ndim == 3:
        masks_ = masks_[..., None]
    elif masks_.ndim == 4 and masks_.shape[-1] != 1:
        masks_ = masks_[..., :1]

    poses_ = None
    if poses is not None:
        poses_ = np.asarray(poses, dtype=np.float64).copy()
        poses_[:, :3, 3] = (poses_[:, :3, 3] + translation.reshape(1, 3)) * float(sc_factor)

    return rgbs_, depths_, masks_, normal_maps, poses_


try:
    from datareader import preprocess_data as preprocess_data_impl
    logging.info("[preprocess] using datareader.preprocess_data")
except Exception:
    preprocess_data_impl = preprocess_data_fallback
    logging.info("[preprocess] datareader.preprocess_data not found, using fallback")


_num_pat = re.compile(r"(\d+)")


def extract_key(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    nums = _num_pat.findall(stem)
    return nums[-1] if nums else stem


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def find_dir(session_dir: str, candidates):
    for name in candidates:
        p = os.path.join(session_dir, name)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"None of dirs exist under {session_dir}: {candidates}")


def list_files(d: str, exts):
    out = []
    for ext in exts:
        out += glob.glob(os.path.join(d, f"*{ext}"))
    return sorted(set(out))


def build_key_map(files):
    mp = {}
    for f in files:
        k = extract_key(f)
        mp.setdefault(k, []).append(f)
    return {k: sorted(v, key=lambda x: (len(os.path.basename(x)), x))[0] for k, v in mp.items()}


def resize_to(rgb_hw, depth=None, mask=None):
    H, W = rgb_hw
    if depth is not None:
        if depth.ndim == 2:
            if depth.shape[:2] != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            depth = depth[..., None]
        elif depth.ndim == 3 and depth.shape[-1] == 1:
            if depth.shape[:2] != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
                if depth.ndim == 2:
                    depth = depth[..., None]

    if mask is not None:
        if mask.ndim == 2:
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = mask[..., None]
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                if mask.ndim == 2:
                    mask = mask[..., None]
    return depth, mask


def backproject_masked_points(depth_hw, mask_hw, K, max_pts, depth_max_m):
    mask = (mask_hw > 0)
    valid = mask & np.isfinite(depth_hw) & (depth_hw > 1e-6) & (depth_hw < float(depth_max_m))
    ys, xs = np.where(valid)
    n = xs.size
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if n > max_pts:
        idx = np.random.choice(n, size=max_pts, replace=False)
        xs = xs[idx]
        ys = ys[idx]

    z = depth_hw[ys, xs].astype(np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def transform_points(T_c2w, pts_cam):
    if pts_cam.shape[0] == 0:
        return pts_cam
    R = T_c2w[:3, :3].astype(np.float32)
    t = T_c2w[:3, 3].astype(np.float32)
    return (pts_cam @ R.T) + t.reshape(1, 3)


def compute_scene_bounds_simple(cam_in_obs, K, depths_m, masks):
    all_pts = []
    total = 0
    for i in range(len(depths_m)):
        d = depths_m[i][..., 0].astype(np.float32)
        m = masks[i][..., 0].astype(np.uint8)
        pts_cam = backproject_masked_points(d, m, K, MAX_PTS_PER_FRAME, DEPTH_MAX_M)
        pts_w = transform_points(cam_in_obs[i], pts_cam)
        if pts_w.shape[0] == 0:
            continue
        all_pts.append(pts_w)
        total += pts_w.shape[0]
        if total >= MAX_PTS_TOTAL:
            break

    if not all_pts:
        raise RuntimeError("No valid masked depth points (check depth/mask).")

    pts = np.concatenate(all_pts, axis=0)
    if pts.shape[0] > MAX_PTS_TOTAL:
        idx = np.random.choice(pts.shape[0], size=MAX_PTS_TOTAL, replace=False)
        pts = pts[idx]

    lo = np.percentile(pts, ROBUST_PCT, axis=0)
    hi = np.percentile(pts, 100.0 - ROBUST_PCT, axis=0)
    center = 0.5 * (lo + hi)
    extent = float(np.max(hi - lo))
    extent = max(extent, 1e-6)

    translation = (-center).astype(np.float64)
    sc_factor = float(1.0 / extent)

    pts_norm = (pts + translation.reshape(1, 3).astype(np.float32)) * np.float32(sc_factor)
    pcd_norm = o3d.geometry.PointCloud()
    pcd_norm.points = o3d.utility.Vector3dVector(pts_norm.astype(np.float64))

    logging.info(f"[bounds] pts={pts.shape[0]} extent={extent:.4f} sc_factor={sc_factor:.6f} translation={translation}")
    return sc_factor, translation, pcd_norm


def apply_transform_to_mesh(mesh, T: np.ndarray):
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    if hasattr(mesh, "apply_transform"):
        mesh.apply_transform(T)
        return mesh
    if hasattr(mesh, "transform"):
        mesh.transform(T)
        return mesh
    if hasattr(mesh, "vertices"):
        v = np.asarray(mesh.vertices)
        v_h = np.concatenate([v, np.ones((v.shape[0], 1))], axis=1)
        v2 = (v_h @ T.T)[:, :3]
        try:
            mesh.vertices = v2
        except Exception:
            pass
        return mesh
    raise TypeError("Unknown mesh type; cannot apply transform")


def mesh_to_real_world_local(mesh, translation: np.ndarray, sc_factor: float, pose_offset=None):
    if pose_offset is None:
        pose_offset = np.eye(4, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    sc = float(sc_factor)

    mesh = apply_transform_to_mesh(mesh, pose_offset)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] *= (1.0 / sc)
    T[:3, 3] = -translation
    mesh = apply_transform_to_mesh(mesh, T)
    return mesh


def remove_plane_faces_in_tag(mesh, band_m=0.01, cos_th=0.95, z_cut_m=0.001):
    """
    在 tag 坐标中删除底面平台：
      - 删除靠近 z=0 且法向接近 z轴 的三角形（水平面）
      - 可选：把 z < z_cut 的内容再裁掉
    """
    try:
        import trimesh
    except Exception:
        logging.warning("[plane-cut] trimesh not available, skip plane-cut")
        return mesh

    if not isinstance(mesh, trimesh.Trimesh):
        return mesh

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if V.shape[0] < 1000 or F.shape[0] < 1000:
        return mesh

    # 统一 z 方向：如果主体整体 z 偏负，翻转判定（避免删错面）
    s = 1.0 if np.median(V[:, 2]) >= 0 else -1.0

    # face normals + face mean z
    fn = mesh.face_normals  # (M,3)
    zaxis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = np.abs(fn @ zaxis)  # |n·z|
    z_mean = (V[F].mean(axis=1)[:, 2]) * s

    # 1) 删除靠近 plane 且近似水平面的 faces
    rm_face = (z_mean < float(band_m)) & (dot > float(cos_th))
    keep_face = ~rm_face
    F2 = F[keep_face]
    if F2.size == 0:
        return mesh

    mesh2 = trimesh.Trimesh(vertices=V, faces=F2, process=False)
    mesh2.remove_unreferenced_vertices()

    # 2) 可选：把 z<z_cut 的残留再裁掉（更狠）
    if z_cut_m is not None and float(z_cut_m) > 0:
        V2 = np.asarray(mesh2.vertices)
        F2 = np.asarray(mesh2.faces)
        keep_v = (V2[:, 2] * s) >= float(z_cut_m)
        keep_f = keep_v[F2].all(axis=1)
        F3 = F2[keep_f]
        if F3.size > 0 and int(keep_v.sum()) > 100:
            mesh3 = trimesh.Trimesh(vertices=V2, faces=F3, process=False)
            mesh3.remove_unreferenced_vertices()
            return mesh3

    return mesh2


def load_rs_session(session_dir: str):
    color_dir = find_dir(session_dir, ["color", "rgb", "images"])
    depth_dir = find_dir(session_dir, ["depth", "depth_enhanced", "depth_aligned"])
    mask_dir  = find_dir(session_dir, ["mask", "masks"])
    meta_dir  = os.path.join(session_dir, "meta") if os.path.isdir(os.path.join(session_dir, "meta")) else session_dir

    K_path = os.path.join(session_dir, "K.txt")
    if not os.path.exists(K_path):
        raise FileNotFoundError(f"Missing K.txt: {K_path}")
    K = np.loadtxt(K_path).reshape(3, 3).astype(np.float64)

    c_map = build_key_map(list_files(color_dir, (".png", ".jpg", ".jpeg")))
    d_map = build_key_map(list_files(depth_dir, (".png", ".npy", ".npz")))
    m_map = build_key_map(list_files(mask_dir,  (".png", ".jpg", ".jpeg")))
    z_map = build_key_map(list_files(meta_dir,  (".npz",)))

    keys = sorted(set(c_map.keys()) & set(d_map.keys()) & set(m_map.keys()) & set(z_map.keys()))
    if STRIDE > 1:
        keys = keys[::STRIDE]
    if MAX_FRAMES > 0:
        keys = keys[:MAX_FRAMES]

    rgbs, depths_m, masks, cam_in_obs = [], [], [], []
    for k in keys:
        cf, df, mf, zf = c_map[k], d_map[k], m_map[k], z_map[k]

        bgr = cv2.imread(cf, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        depth_u = cv2.imread(df, cv2.IMREAD_UNCHANGED)
        if depth_u is None:
            continue
        depth_u = depth_u.astype(np.float32)
        nz = depth_u[depth_u > 0]
        depth_m = depth_u * 0.001 if (nz.size > 0 and float(np.mean(nz)) > 20.0) else depth_u

        if depth_m.ndim == 2:
            depth_m = depth_m[..., None]
        elif depth_m.ndim == 3 and depth_m.shape[-1] != 1:
            depth_m = depth_m[..., :1]

        mk = cv2.imread(mf, cv2.IMREAD_UNCHANGED)
        if mk is None:
            continue
        if mk.ndim == 3:
            mk = mk[..., 0]
        mk = (mk > 0).astype(np.uint8) * 255
        if mk.ndim == 2:
            mk = mk[..., None]

        depth_m, mk = resize_to((H, W), depth=depth_m, mask=mk)

        # mask 后处理
        mb = (mk[..., 0] > 0)
        if MASK_CLOSE_ITERS > 0:
            k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mb = cv2.morphologyEx(mb.astype(np.uint8), cv2.MORPH_CLOSE, k3, iterations=MASK_CLOSE_ITERS).astype(bool)
        if MASK_ERODE_ITERS > 0:
            k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mb = cv2.erode(mb.astype(np.uint8), k3, iterations=MASK_ERODE_ITERS).astype(bool)
        mk[..., 0] = (mb.astype(np.uint8) * 255)

        if HARD_MASK:
            depth_m = depth_m.copy()
            depth_m[~mb, 0] = 0.0
            rgb = rgb.copy()
            rgb[~mb] = 0

        if int((mk[..., 0] > 0).sum()) < MIN_MASK_PIX:
            continue
        if int(((depth_m[..., 0] > 1e-6) & (mk[..., 0] > 0)).sum()) < MIN_VALID_DEPTH_PIX:
            continue

        meta = np.load(zf, allow_pickle=True)

        if FORCE_MARKER_ID is not None and ("marker_id" in meta):
            try:
                if int(meta["marker_id"]) != int(FORCE_MARKER_ID):
                    continue
            except Exception:
                pass

        # solvePnP 输出 tag->cam，这里取逆得到 cam->tag（cam_in_obs）
        if "T_cam_tag" in meta:
            cam_pose = invert_T(np.asarray(meta["T_cam_tag"], dtype=np.float64).reshape(4, 4))
        elif "T_tag_cam" in meta:
            cam_pose = invert_T(np.asarray(meta["T_tag_cam"], dtype=np.float64).reshape(4, 4))
        elif ("rvec" in meta) and ("tvec" in meta):
            rvec = np.asarray(meta["rvec"], dtype=np.float64).reshape(3)
            tvec = np.asarray(meta["tvec"], dtype=np.float64).reshape(3)
            R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = tvec
            cam_pose = invert_T(T)
        else:
            continue

        rgbs.append(rgb)
        depths_m.append(depth_m.astype(np.float32))
        masks.append(mk.astype(np.uint8))
        cam_in_obs.append(cam_pose.astype(np.float64))

    if len(rgbs) < 8:
        raise RuntimeError(f"Too few valid paired frames after filtering: {len(rgbs)}")

    logging.info(f"[data] loaded paired frames: {len(rgbs)}")
    return K, rgbs, depths_m, masks, cam_in_obs


def run_neural_object_field(cfg, K, rgbs, depths_m, masks, cam_in_obs, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    cfg["save_dir"] = save_dir

    rgbs = np.asarray(rgbs)
    depths_m = np.asarray(depths_m)
    masks = np.asarray(masks)
    cam_in_obs = np.asarray(cam_in_obs)

    for i in range(min(len(rgbs), 10)):
        imageio.imwrite(os.path.join(save_dir, f"rgb_{i:04d}.png"), rgbs[i])
        imageio.imwrite(os.path.join(save_dir, f"mask_{i:04d}.png"), masks[i][..., 0])

    sc_factor, translation, pcd_norm = compute_scene_bounds_simple(cam_in_obs, K, depths_m, masks)
    cfg["sc_factor"] = float(sc_factor)
    cfg["translation"] = np.asarray(translation, dtype=np.float64)
    o3d.io.write_point_cloud(os.path.join(save_dir, "pcd_normalized.ply"), pcd_norm)

    glcam_in_obs = cam_in_obs @ glcam_in_cvcam

    rgbs_, depths_, masks_, normal_maps, poses = preprocess_data_impl(
        rgbs, depths_m, masks,
        normal_maps=None,
        poses=glcam_in_obs,
        sc_factor=cfg["sc_factor"],
        translation=cfg["translation"],
    )

    # 维度保险
    if depths_.ndim == 3:
        depths_ = depths_[..., None]
    if masks_.ndim == 3:
        masks_ = masks_[..., None]
    if depths_.ndim == 4 and depths_.shape[-1] != 1:
        depths_ = depths_[..., :1]
    if masks_.ndim == 4 and masks_.shape[-1] != 1:
        masks_ = masks_[..., :1]

    if DISABLE_GRIDENCODER:
        cfg["i_embed"] = 0
        cfg.setdefault("multires", 10)
        cfg.setdefault("multires_view", 4)
        logging.warning("[cfg] DISABLE_GRIDENCODER=True -> i_embed=0")

    nerf = NerfRunner(
        cfg,
        rgbs_, depths_, masks_,
        normal_maps=None,
        poses=poses,
        K=K,
        occ_masks=None,
        build_octree_pcd=pcd_norm,
    )
    nerf.train()

    base_voxel = float(cfg.get("mesh_resolution", 0.01))
    voxel = max(base_voxel, base_voxel * float(MESH_VOXEL_MULT))
    logging.info(f"[mesh] mesh_resolution={base_voxel} voxel_size={voxel} (mult={MESH_VOXEL_MULT})")

    mesh = nerf.extract_mesh(isolevel=0, voxel_size=voxel)
    mesh = nerf.mesh_texture_from_train_images(mesh, rgbs_raw=rgbs, tex_res=1028)

    mesh = mesh_to_real_world_local(
        mesh,
        translation=cfg["translation"],
        sc_factor=cfg["sc_factor"],
        pose_offset=np.eye(4, dtype=np.float64),
    )

    # ✅ 关键：用 tag 平面清掉平台底面
    if REMOVE_TAG_PLANE_FACES:
        logging.info(f"[tag-plane] remove faces near z=0 | band={PLANE_BAND_M} cos>{PLANE_COS_TH} z_cut={Z_CUT_M}")
        mesh = remove_plane_faces_in_tag(mesh, band_m=PLANE_BAND_M, cos_th=PLANE_COS_TH, z_cut_m=Z_CUT_M)

    return mesh


def main():
    session_dir = os.path.abspath(SESSION_DIR)
    out_dir = OUT_DIR.strip() or os.path.join(session_dir, "nerf_out")
    out_obj = OUT_OBJ.strip() or os.path.join(out_dir, "model.obj")
    ensure_dir(out_dir)

    cfg_path = pick_default_cfg()
    cfg = load_cfg(cfg_path)

    logging.info(f"[cfg] {cfg_path}")
    logging.info(f"[in ] {session_dir}")
    logging.info(f"[out] {out_dir}")
    logging.info(f"[obj] {out_obj}")
    logging.info(f"[stride] {STRIDE} [max_frames] {MAX_FRAMES} [marker_id] {FORCE_MARKER_ID}")
    logging.info(f"[mask] HARD_MASK={HARD_MASK} ERODE={MASK_ERODE_ITERS} CLOSE={MASK_CLOSE_ITERS}")
    logging.info(f"[mesh] MESH_VOXEL_MULT={MESH_VOXEL_MULT}")
    logging.info(f"[tag-plane] enable={REMOVE_TAG_PLANE_FACES} band={PLANE_BAND_M} cos={PLANE_COS_TH} z_cut={Z_CUT_M}")

    K, rgbs, depths_m, masks, cam_in_obs = load_rs_session(session_dir)
    mesh = run_neural_object_field(cfg, K, rgbs, depths_m, masks, cam_in_obs, save_dir=out_dir)
    mesh.export(out_obj)
    logging.info(f"[DONE] saved mesh: {out_obj}")


if __name__ == "__main__":
    main()
