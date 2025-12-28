#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
import multiprocessing as mp
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import zmq
from PIL import Image

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


# ================== CUDA/torch 设置 ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# 降低 CPU 线程争抢
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)


# ================== 容器内自动取宿主机网关（不依赖 ip 命令） ==================
def get_default_gateway_from_proc() -> Optional[str]:
    try:
        with open("/proc/net/route", "r") as f:
            lines = f.readlines()
        for line in lines[1:]:
            cols = line.strip().split()
            if len(cols) >= 3 and cols[1] == "00000000":
                gw_hex = cols[2]
                import socket, struct
                return socket.inet_ntoa(struct.pack("<L", int(gw_hex, 16)))
    except Exception:
        pass
    return None


# ================== GroundingDINO ==================
class GroundingDinoPredictor:
    def __init__(self, model_id: str, device: str, local_files_only: bool = True):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, local_files_only=local_files_only
        ).to(device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: "PIL.Image.Image", text_prompts: str,
                box_threshold=0.25, text_threshold=0.25):
        inputs = self.processor(images=image, text=text_prompts, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )
        r0 = results[0]
        boxes = r0.get("boxes", None)
        labels = r0.get("labels", None)
        scores = r0.get("scores", None)
        if boxes is None:
            boxes = torch.empty((0, 4), device=self.device)
        return boxes, labels, scores


# ================== SAM2 image segmentor ==================
class SAM2ImageSegmentor:
    def __init__(self, sam_model_cfg: str, sam_model_ckpt: str, device: str):
        self.device = device
        sam_model = build_sam2(sam_model_cfg, sam_model_ckpt, device=device)
        self.predictor = SAM2ImagePredictor(sam_model)

    def set_image(self, image: np.ndarray):
        self.predictor.set_image(image)

    @torch.inference_mode()
    def predict_masks_from_boxes(self, boxes: torch.Tensor) -> np.ndarray:
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        if masks is None:
            return np.empty((0, 1, 1), dtype=np.uint8)
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        if masks.ndim == 2:
            masks = masks[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)
        return masks


# ================== 工具：bbox / ROI ==================
def mask_bbox(mask_u8: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def clamp_roi(x0: int, y0: int, x1: int, y1: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(x0 + 1, min(W, x1))
    y1 = max(y0 + 1, min(H, y1))
    return x0, y0, x1, y1


def make_roi_from_center(cx: float, cy: float, roi_w: int, roi_h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x0 = int(round(cx - roi_w / 2))
    y0 = int(round(cy - roi_h / 2))
    x1 = x0 + roi_w
    y1 = y0 + roi_h
    return clamp_roi(x0, y0, x1, y1, W, H)


def select_topk_boxes(boxes: torch.Tensor, scores: Optional[torch.Tensor], max_boxes: int) -> torch.Tensor:
    if boxes is None or boxes.numel() == 0:
        return boxes
    max_boxes = max(1, int(max_boxes))
    if boxes.shape[0] <= max_boxes:
        return boxes
    if scores is not None and isinstance(scores, torch.Tensor) and scores.numel() == boxes.shape[0]:
        idx = torch.argsort(scores, descending=True)
    else:
        wh = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        idx = torch.argsort(wh, descending=True)
    return boxes[idx[:max_boxes]]


# ================== ROI 追踪器：union mask + 容错 ==================
class ROIUnionMaskTracker:
    def __init__(
        self,
        dino_id: str,
        sam_cfg: str,
        sam_ckpt: str,
        device: str,
        prompt: str,
        interval: int,
        box_th: float,
        text_th: float,
        local_files_only: bool,
        max_boxes: int,
        roi_ratio: float,
        roi_pad: float,
        roi_smooth: float,
        miss_max: int,
        fallback_full_detect: bool,
    ):
        self.device = device
        self.prompt = prompt
        self.interval = max(1, int(interval))
        self.box_th = float(box_th)
        self.text_th = float(text_th)
        self.max_boxes = max(1, int(max_boxes))

        self.roi_ratio = float(roi_ratio)
        self.roi_pad = float(roi_pad)
        self.roi_smooth = float(roi_smooth)
        self.miss_max = max(0, int(miss_max))
        self.fallback_full_detect = bool(fallback_full_detect)

        self.dino = GroundingDinoPredictor(dino_id, device=device, local_files_only=local_files_only)
        self.sam_img = SAM2ImageSegmentor(sam_cfg, sam_ckpt, device=device)
        self.video_pred = build_sam2_video_predictor(sam_cfg, sam_ckpt)
        self.side = int(getattr(self.video_pred, "image_size", 1024))

        self.state = None
        self.total_frames = 0
        self.has_prompt = False

        self.frame_H = None
        self.frame_W = None

        self.roi = None
        self.roi_cx = None
        self.roi_cy = None

        self.last_full_mask = None
        self.miss_cnt = 0
        self.force_detect = True

        self._reset_state_for_roi(None)

    def _reset_state_for_roi(self, roi_hw: Optional[Tuple[int, int]]):
        self.state = self.video_pred.init_state()
        self.state["images"] = torch.empty((0, 3, self.side, self.side), device=self.device)
        if roi_hw is not None:
            h, w = roi_hw
            self.state["video_height"] = int(h)
            self.state["video_width"] = int(w)
        else:
            self.state["video_height"] = None
            self.state["video_width"] = None
        self.has_prompt = False

    def _need_detect(self) -> bool:
        return self.force_detect or ((self.total_frames % self.interval) == 0)

    def _current_roi_or_full(self) -> Tuple[int, int, int, int]:
        assert self.frame_W is not None and self.frame_H is not None
        if self.roi is None:
            return 0, 0, self.frame_W, self.frame_H
        return self.roi

    def _update_roi_from_mask(self, full_mask_u8: np.ndarray):
        H, W = full_mask_u8.shape[:2]
        bb = mask_bbox(full_mask_u8)
        if bb is None:
            return
        x1, y1, x2, y2 = bb
        cx_new = 0.5 * (x1 + x2)
        cy_new = 0.5 * (y1 + y2)

        if self.roi_cx is None or self.roi_cy is None:
            self.roi_cx, self.roi_cy = cx_new, cy_new
        else:
            a = self.roi_smooth
            self.roi_cx = (1 - a) * self.roi_cx + a * cx_new
            self.roi_cy = (1 - a) * self.roi_cy + a * cy_new

        base_w = max(64, int(round(W * self.roi_ratio)))
        base_h = max(64, int(round(H * self.roi_ratio)))

        bbox_w = max(1, x2 - x1)
        bbox_h = max(1, y2 - y1)
        want_w = int(round(bbox_w * (1.0 + self.roi_pad)))
        want_h = int(round(bbox_h * (1.0 + self.roi_pad)))

        roi_w = min(W, max(base_w, want_w))
        roi_h = min(H, max(base_h, want_h))

        self.roi = make_roi_from_center(self.roi_cx, self.roi_cy, roi_w, roi_h, W, H)

    def _paste_crop_mask_to_full(self, crop_mask_u8: np.ndarray, roi_used: Tuple[int, int, int, int]) -> np.ndarray:
        H, W = self.frame_H, self.frame_W
        full = np.zeros((H, W), dtype=np.uint8)
        x0, y0, x1, y1 = roi_used
        ch, cw = crop_mask_u8.shape[:2]
        if (y1 - y0) != ch or (x1 - x0) != cw:
            crop_mask_u8 = cv2.resize(crop_mask_u8, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        full[y0:y1, x0:x1] = crop_mask_u8
        return full

    @torch.inference_mode()
    def step(self, frame_rgb_full: np.ndarray) -> Optional[np.ndarray]:
        H, W = frame_rgb_full.shape[:2]
        if self.frame_H is None or self.frame_W is None:
            self.frame_H, self.frame_W = H, W

        roi_used = self._current_roi_or_full()
        x0, y0, x1, y1 = roi_used
        frame_rgb = frame_rgb_full[y0:y1, x0:x1]

        if self._need_detect():
            self.force_detect = False
            self.miss_cnt = 0

            def _detect_on(rgb_crop: np.ndarray) -> torch.Tensor:
                img_pil = Image.fromarray(rgb_crop)
                boxes, _labels, scores = self.dino.predict(
                    img_pil, self.prompt, box_threshold=self.box_th, text_threshold=self.text_th
                )
                if boxes is None:
                    boxes = torch.empty((0, 4), device=self.device)
                boxes = select_topk_boxes(boxes, scores, self.max_boxes)
                return boxes

            boxes = _detect_on(frame_rgb)

            if boxes is None or boxes.shape[0] == 0:
                if self.fallback_full_detect and self.roi is not None:
                    roi_used = (0, 0, W, H)
                    x0, y0, x1, y1 = roi_used
                    frame_rgb = frame_rgb_full
                    boxes = _detect_on(frame_rgb)

            if boxes is None or boxes.shape[0] == 0:
                self.has_prompt = False
                self.total_frames += 1
                self.miss_cnt += 1
                if self.last_full_mask is not None and self.miss_cnt <= self.miss_max:
                    return self.last_full_mask
                self.roi = None
                self.roi_cx = None
                self.roi_cy = None
                self.force_detect = True
                torch.cuda.empty_cache()  # 清理显存
                return None

            self.sam_img.set_image(frame_rgb)
            masks = self.sam_img.predict_masks_from_boxes(boxes)

            if masks is None or masks.shape[0] == 0:
                self.has_prompt = False
                self.total_frames += 1
                self.miss_cnt += 1
                if self.last_full_mask is not None and self.miss_cnt <= self.miss_max:
                    return self.last_full_mask
                self.force_detect = True
                torch.cuda.empty_cache()  # 清理显存
                return None

            union = None
            for i in range(masks.shape[0]):
                m = masks[i]
                if isinstance(m, torch.Tensor):
                    m = m.detach().cpu().numpy()
                mb = (m > 0)
                union = mb if union is None else (union | mb)

            crop_mask_u8 = (union.astype(np.uint8) * 255) if union is not None else None
            if crop_mask_u8 is None:
                self.has_prompt = False
                self.total_frames += 1
                self.force_detect = True
                torch.cuda.empty_cache()  # 清理显存
                return None

            ch, cw = frame_rgb.shape[:2]
            self._reset_state_for_roi((ch, cw))

            frame_idx = self.video_pred.add_new_frame(self.state, frame_rgb)
            self.video_pred.reset_state(self.state)

            for oid in range(masks.shape[0]):
                m = masks[oid]
                if isinstance(m, np.ndarray):
                    m_in = m.astype(bool)
                else:
                    m_in = (m > 0).bool()
                self.video_pred.add_new_mask(self.state, frame_idx, int(oid + 1), m_in)

            self.has_prompt = True

            full_mask = self._paste_crop_mask_to_full(crop_mask_u8, roi_used)
            self.last_full_mask = full_mask
            self._update_roi_from_mask(full_mask)

            self.total_frames += 1
            torch.cuda.empty_cache()  # 清理显存
            return full_mask

        if not self.has_prompt:
            self.total_frames += 1
            self.miss_cnt += 1
            if self.last_full_mask is not None and self.miss_cnt <= self.miss_max:
                return self.last_full_mask
            self.force_detect = True
            return None

        frame_idx = self.video_pred.add_new_frame(self.state, frame_rgb)
        frame_idx, obj_ids, video_res_masks = self.video_pred.infer_single_frame(
            inference_state=self.state,
            frame_idx=frame_idx,
        )

        self.total_frames += 1

        if obj_ids is None or len(obj_ids) == 0:
            self.miss_cnt += 1
            if self.last_full_mask is not None and self.miss_cnt <= self.miss_max:
                return self.last_full_mask
            self.has_prompt = False
            self.force_detect = True
            torch.cuda.empty_cache()  # 清理显存
            return None

        crop_mask_u8 = None
        try:
            if isinstance(video_res_masks, (list, tuple)) and len(video_res_masks) > 0 and torch.is_tensor(video_res_masks[0]):
                ms = torch.stack([m[0] if m.ndim == 3 else m for m in video_res_masks], dim=0)
                union_t = (ms > 0.0).any(dim=0).to(torch.uint8) * 255
                crop_mask_u8 = union_t.detach().cpu().numpy()
            else:
                union = None
                for i in range(len(obj_ids)):
                    m = video_res_masks[i]
                    if isinstance(m, np.ndarray):
                        mb = (m > 0.0)
                        mb = mb[0] if mb.ndim == 3 else mb
                    else:
                        mb = (m > 0.0)
                        mb = mb[0] if mb.ndim == 3 else mb
                        mb = mb.detach().cpu().numpy()
                    union = mb if union is None else (union | mb)
                crop_mask_u8 = (union.astype(np.uint8) * 255) if union is not None else None
        except Exception:
            crop_mask_u8 = None

        if crop_mask_u8 is None or int(crop_mask_u8.sum()) == 0:
            self.miss_cnt += 1
            if self.last_full_mask is not None and self.miss_cnt <= self.miss_max:
                return self.last_full_mask
            self.has_prompt = False
            self.force_detect = True
            torch.cuda.empty_cache()  # 清理显存
            return None

        self.miss_cnt = 0

        full_mask = self._paste_crop_mask_to_full(crop_mask_u8, roi_used)
        self.last_full_mask = full_mask
        self._update_roi_from_mask(full_mask)
        torch.cuda.empty_cache()  # 清理显存
        return full_mask


# ================== ZMQ per-cam worker（一个进程处理一个 cam） ==================
def worker_one_cam(
    cam: str,
    host_ip: str,
    prompt: str,
    interval: int,
    box_th: float,
    text_th: float,
    sam_cfg: str,
    sam_ckpt: str,
    dino_id: str,
    allow_download: bool,
    overlay_enable: bool,
    overlay_q: int,
    max_boxes: int,
    mask_png_level: int,
    roi_ratio: float,
    roi_pad: float,
    roi_smooth: float,
    miss_max: int,
    fallback_full_detect: bool,
    sub_hwm: int,
):
    local_files_only = (not allow_download)

    print(f"[W:{cam}] start | host={host_ip} prompt='{prompt}' interval={interval} max_boxes={max_boxes} roi_ratio={roi_ratio} device={DEVICE}")
    print(f"[W:{cam}] SUB tcp://{host_ip}:5555 topic='rgbd/{cam}'  ->  PUB tcp://{host_ip}:5556 topic='segd/{cam}'")
    print(f"[W:{cam}] NOTE: CONFLATE disabled; manual-drain latest enabled (multipart-safe).")

    tracker = ROIUnionMaskTracker(
        dino_id=dino_id,
        sam_cfg=sam_cfg,
        sam_ckpt=sam_ckpt,
        device=DEVICE,
        prompt=prompt,
        interval=interval,
        box_th=box_th,
        text_th=text_th,
        local_files_only=local_files_only,
        max_boxes=max_boxes,
        roi_ratio=roi_ratio,
        roi_pad=roi_pad,
        roi_smooth=roi_smooth,
        miss_max=miss_max,
        fallback_full_detect=fallback_full_detect,
    )

    ctx = zmq.Context()

    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{host_ip}:5555")
    sub.setsockopt_string(zmq.SUBSCRIBE, f"rgbd/{cam}")
    sub.setsockopt(zmq.RCVHWM, int(sub_hwm))
    sub.setsockopt(zmq.LINGER, 0)

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    pub = ctx.socket(zmq.PUB)
    pub.connect(f"tcp://{host_ip}:5556")
    pub.setsockopt(zmq.SNDHWM, 1)
    pub.setsockopt(zmq.LINGER, 0)

    time.sleep(0.2)

    last_t = time.time()
    recv_cnt = 0
    proc_cnt = 0
    avg_infer = 0.0
    alpha = 0.1

    png_level = int(mask_png_level)
    png_level = max(0, min(9, png_level))

    try:
        while True:
            socks = dict(poller.poll(1000))
            if sub not in socks:
                continue

            parts = sub.recv_multipart()
            while True:
                try:
                    p2 = sub.recv_multipart(flags=zmq.NOBLOCK)
                    parts = p2
                except zmq.Again:
                    break

            if len(parts) != 5:
                continue

            _topic_b, ts_b, meta_b, rgb_jpg_b, depth_png_b = parts
            recv_cnt += 1

            frame_bgr = cv2.imdecode(np.frombuffer(rgb_jpg_b, np.uint8), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            t0 = time.time()
            full_mask_u8 = tracker.step(frame_rgb)
            t1 = time.time()

            infer_t = (t1 - t0)
            avg_infer = infer_t if avg_infer == 0.0 else (1 - alpha) * avg_infer + alpha * infer_t
            proc_cnt += 1

            if full_mask_u8 is None:
                full_mask_u8 = np.zeros((frame_bgr.shape[0], frame_bgr.shape[1]), dtype=np.uint8)

            ok_m, buf_m = cv2.imencode(".png", full_mask_u8, [int(cv2.IMWRITE_PNG_COMPRESSION), png_level])
            if not ok_m:
                continue
            mask_png_b = buf_m.tobytes()

            try:
                meta = json.loads(meta_b.decode("utf-8"))
                if not isinstance(meta, dict):
                    meta = {}
            except Exception:
                meta = {}
            meta["cam"] = cam
            meta_out_b = json.dumps(meta).encode("utf-8")

            raw_rgb_jpg_b = rgb_jpg_b

            if not overlay_enable:
                overlay_jpg_b = rgb_jpg_b
            else:
                show = frame_bgr.copy()
                txt = f"{cam} | infer {avg_infer*1000.0:5.1f}ms"
                cv2.putText(show, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                ok_o, buf_o = cv2.imencode(".jpg", show, [int(cv2.IMWRITE_JPEG_QUALITY), int(overlay_q)])
                overlay_jpg_b = buf_o.tobytes() if ok_o else rgb_jpg_b

            out_topic = f"segd/{cam}".encode("utf-8")

            try:
                pub.send_multipart(
                    [
                        out_topic,
                        ts_b,
                        meta_out_b,
                        mask_png_b,
                        depth_png_b,
                        raw_rgb_jpg_b,
                        overlay_jpg_b,
                    ],
                    flags=zmq.NOBLOCK,
                )
            except zmq.Again:
                pass

            now = time.time()
            if now - last_t >= 1.0:
                dt = now - last_t
                print(f"[W:{cam}] recv {recv_cnt/dt:5.1f} fps | proc {proc_cnt/dt:5.1f} fps | infer {avg_infer*1000.0:6.1f} ms")
                recv_cnt = 0
                proc_cnt = 0
                last_t = now

    finally:
        try:
            sub.close(0)
        except Exception:
            pass
        try:
            pub.close(0)
        except Exception:
            pass
        ctx.term()
        print(f"[W:{cam}] stopped")


# ================== main：spawn N 个进程 ==================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--host", default=None, help="宿主机地址（容器里建议留空自动取网关）")
    ap.add_argument("--cams", default="front,left,right", help="要处理的相机：front,left,right（逗号分隔）")

    ap.add_argument("--prompt", default="cup.", help="GroundingDINO prompt（小写+句号）")
    ap.add_argument("--interval", type=int, default=10, help="detection_interval（每隔多少帧做一次 detect）")
    ap.add_argument("--box_th", type=float, default=0.25)
    ap.add_argument("--text_th", type=float, default=0.25)

    ap.add_argument("--max_boxes", type=int, default=3, help="DINO 检测最多保留多少个 box（建议 1~3）")

    ap.add_argument("--sam_cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--sam_ckpt", default="./checkpoints/sam2.1_hiera_tiny.pt")
    ap.add_argument("--dino_id", default="IDEA-Research/grounding-dino-tiny")

    ap.add_argument("--allow_download", action="store_true", help="允许 transformers 自动下载（默认 OFF）")

    # mask png 压缩：0 最快、9 最小
    ap.add_argument("--mask_png_level", type=int, default=0, help="PNG compression level 0~9 (0 fastest)")

    # overlay 默认关闭（最快）
    ap.add_argument("--overlay", action="store_true", help="开启 overlay（会额外编码 jpg，稍慢）")
    ap.add_argument("--overlay_q", type=int, default=70, help="overlay jpg quality")

    # ROI
    ap.add_argument("--roi_ratio", type=float, default=0.5, help="ROI 窗口相对全图比例（如 0.5）")
    ap.add_argument("--roi_pad", type=float, default=0.35, help="目标 bbox 膨胀比例（容错）")
    ap.add_argument("--roi_smooth", type=float, default=0.5, help="ROI center EMA (0~1, 越大跟随越快)")
    ap.add_argument("--miss_max", type=int, default=5, help="允许连续丢失多少帧仍沿用上一帧 mask/ROI")
    ap.add_argument("--no_fallback_full_detect", action="store_true",
                    help="禁用：ROI detect 失败时回退全图 detect（更快但更容易丢）")

    # 手动最新帧策略：用 HWM 控制积压上限
    ap.add_argument("--sub_hwm", type=int, default=10, help="SUB RCVHWM (no conflate). small keeps latency low.")

    return ap.parse_args()


def main():
    args = parse_args()

    cams = [c.strip().lower() for c in args.cams.split(",") if c.strip()]
    cams = list(dict.fromkeys(cams))
    if not cams:
        cams = ["front", "left", "right"]

    host_ip = args.host or get_default_gateway_from_proc() or "172.17.0.1"
    fallback_full_detect = (not args.no_fallback_full_detect)

    print(f"[MAIN] host={host_ip} cams={cams} prompt='{args.prompt}' interval={args.interval} max_boxes={args.max_boxes} roi_ratio={args.roi_ratio} device={DEVICE}")
    print("[MAIN] IN : rgbd/<cam> @ tcp://host:5555 (5 parts)")
    print("[MAIN] OUT: segd/<cam> @ tcp://host:5556 (7 parts)")
    print("[MAIN] FIX: multipart-safe latest-frame (manual drain), CONFLATE disabled.")

    mp.set_start_method("spawn", force=True)

    procs = []
    for cam in cams:
        p = mp.Process(
            target=worker_one_cam,
            args=(
                cam,
                host_ip,
                args.prompt,
                args.interval,
                args.box_th,
                args.text_th,
                args.sam_cfg,
                args.sam_ckpt,
                args.dino_id,
                args.allow_download,
                args.overlay,
                args.overlay_q,
                args.max_boxes,
                args.mask_png_level,
                args.roi_ratio,
                args.roi_pad,
                args.roi_smooth,
                args.miss_max,
                fallback_full_detect,
                args.sub_hwm,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("[MAIN] Ctrl-C, terminating ...")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.join(timeout=1.0)
            except Exception:
                pass


if __name__ == "__main__":
    main()
