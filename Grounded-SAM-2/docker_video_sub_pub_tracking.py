#!/usr/bin/env python3
import copy
import time
import json

import cv2
import numpy as np
import supervision as sv
import torch
import zmq
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

# ================== 环境设置 ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# ================== Grounding DINO 封装 ==================
class GroundingDinoPredictor:
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

    def predict(self, image: "PIL.Image.Image", text_prompts: str,
                box_threshold=0.25, text_threshold=0.25):
        inputs = self.processor(images=image, text=text_prompts, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )
        return results[0]["boxes"], results[0]["labels"]


# ================== SAM2 图像分割封装 ==================
class SAM2ImageSegmentor:
    def __init__(self, sam_model_cfg: str, sam_model_ckpt: str, device="cuda"):
        self.device = device
        sam_model = build_sam2(sam_model_cfg, sam_model_ckpt, device=device)
        self.predictor = SAM2ImagePredictor(sam_model)

    def set_image(self, image: np.ndarray):
        self.predictor.set_image(image)

    def predict_masks_from_boxes(self, boxes: torch.Tensor):
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )

        # 统一成 (N, H, W)
        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        return masks, scores, logits


# ================== 增量 tracking + SAM2 video predictor ==================
class IncrementalObjectTracker:
    def __init__(
        self,
        grounding_model_id="IDEA-Research/grounding-dino-tiny",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_ckpt_path="./checkpoints/sam2.1_hiera_tiny.pt",
        device="cuda",
        prompt_text="person.",
        detection_interval=20,
    ):
        self.device = device
        self.detection_interval = detection_interval
        self.prompt_text = prompt_text

        self.grounding_predictor = GroundingDinoPredictor(model_id=grounding_model_id, device=device)
        self.sam2_segmentor = SAM2ImageSegmentor(
            sam_model_cfg=sam2_model_cfg,
            sam_model_ckpt=sam2_ckpt_path,
            device=device,
        )
        self.video_predictor = build_sam2_video_predictor(sam2_model_cfg, sam2_ckpt_path)

        self.inference_state = self.video_predictor.init_state()
        self.inference_state["images"] = torch.empty((0, 3, 1024, 1024), device=device)
        self.inference_state["video_height"] = None
        self.inference_state["video_width"] = None

        self.total_frames = 0
        self.objects_count = 0
        self.frame_cache_limit = detection_interval - 1

        self.last_mask_dict = MaskDictionaryModel()
        self.track_dict = MaskDictionaryModel()
        self.has_prompt = False

    def _reset_inference_state(self, h, w):
        print(f"[Reset] Resetting inference state after {self.frame_cache_limit} frames to free memory.")
        self.inference_state = self.video_predictor.init_state()
        self.inference_state["images"] = torch.empty((0, 3, 1024, 1024), device=self.device)
        self.inference_state["video_height"] = h
        self.inference_state["video_width"] = w
        self.has_prompt = False

    def add_image(self, image_np: np.ndarray):
        """
        输入一帧 RGB (H,W,3)
        返回: (annotated_rgb, mask_array_int32) 或 (None, None)
        mask_array: HxW, 0 表示背景，其它为 instance_id
        """
        img_pil = Image.fromarray(image_np)

        h, w = image_np.shape[:2]
        if self.inference_state["video_height"] is None:
            self.inference_state["video_height"] = h
        if self.inference_state["video_width"] is None:
            self.inference_state["video_width"] = w

        if self.total_frames % self.detection_interval == 0:
            if self.inference_state["images"].shape[0] > self.frame_cache_limit:
                self._reset_inference_state(h, w)

            boxes, labels = self.grounding_predictor.predict(img_pil, self.prompt_text)
            if boxes.shape[0] == 0:
                print("[Tracker] No detections from GroundingDINO on detection frame.")
                self.total_frames += 1
                self.has_prompt = False
                return None, None

            self.sam2_segmentor.set_image(image_np)
            masks, scores, logits = self.sam2_segmentor.predict_masks_from_boxes(boxes)

            mask_dict = MaskDictionaryModel(promote_type="mask", mask_name=f"mask_{self.total_frames:05d}.npy")
            mask_dict.add_new_frame_annotation(
                mask_list=torch.tensor(masks).to(self.device),
                box_list=boxes.to(self.device),
                label_list=labels,
            )

            self.objects_count = mask_dict.update_masks(
                tracking_annotation_dict=self.last_mask_dict,
                iou_threshold=0.3,
                objects_count=self.objects_count,
            )

            frame_idx = self.video_predictor.add_new_frame(self.inference_state, image_np)
            self.video_predictor.reset_state(self.inference_state)

            for object_id, object_info in mask_dict.labels.items():
                frame_idx, _, _ = self.video_predictor.add_new_mask(
                    self.inference_state,
                    frame_idx,
                    object_id,
                    object_info.mask,
                )

            self.track_dict = copy.deepcopy(mask_dict)
            self.last_mask_dict = mask_dict
            self.has_prompt = True

        else:
            if not self.has_prompt:
                self.total_frames += 1
                return None, None
            frame_idx = self.video_predictor.add_new_frame(self.inference_state, image_np)

        frame_idx, obj_ids, video_res_masks = self.video_predictor.infer_single_frame(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
        )

        if len(obj_ids) == 0:
            print("[Tracker] infer_single_frame returned no objects.")
            self.total_frames += 1
            return None, None

        frame_masks = MaskDictionaryModel()
        for i, obj_id in enumerate(obj_ids):
            out_mask = video_res_masks[i] > 0.0
            object_info = ObjectInfo(
                instance_id=obj_id,
                mask=out_mask[0],
                class_name=self.track_dict.get_target_class_name(obj_id),
                logit=self.track_dict.get_target_logit(obj_id),
            )
            object_info.update_box()
            frame_masks.labels[obj_id] = object_info
            frame_masks.mask_name = f"mask_{frame_idx:05d}.npy"
            frame_masks.mask_height = out_mask.shape[-2]
            frame_masks.mask_width = out_mask.shape[-1]

        self.last_mask_dict = copy.deepcopy(frame_masks)

        H, W = image_np.shape[:2]
        mask_img = torch.zeros((H, W), dtype=torch.int32)
        for obj_id, obj_info in self.last_mask_dict.labels.items():
            mask_img[obj_info.mask == True] = obj_id

        mask_array = mask_img.cpu().numpy()

        annotated_frame = self.visualize_frame_with_mask_and_metadata(
            image_np=image_np,
            mask_array=mask_array,
            json_metadata=self.last_mask_dict.to_dict(),
        )

        self.total_frames += 1
        return annotated_frame, mask_array

    def set_prompt(self, new_prompt: str):
        self.prompt_text = new_prompt
        self.total_frames = 0
        h = self.inference_state.get("video_height", 0) or 0
        w = self.inference_state.get("video_width", 0) or 0
        self._reset_inference_state(h, w)
        print(f"[Prompt Updated] New prompt: '{new_prompt}'. Tracker state reset.")

    def visualize_frame_with_mask_and_metadata(self, image_np: np.ndarray, mask_array: np.ndarray, json_metadata: dict):
        image = image_np.copy()
        metadata_lookup = json_metadata.get("labels", {})

        all_object_ids = []
        all_object_boxes = []
        all_object_classes = []
        all_object_masks = []

        for _, obj_info in metadata_lookup.items():
            instance_id = obj_info.get("instance_id")
            if instance_id is None or instance_id == 0:
                continue
            if instance_id not in np.unique(mask_array):
                continue

            object_mask = mask_array == instance_id
            all_object_ids.append(instance_id)
            x1 = obj_info.get("x1", 0)
            y1 = obj_info.get("y1", 0)
            x2 = obj_info.get("x2", 0)
            y2 = obj_info.get("y2", 0)
            all_object_boxes.append([x1, y1, x2, y2])
            all_object_classes.append(obj_info.get("class_name", "unknown"))
            all_object_masks.append(object_mask[None])  # (1,H,W)

        if len(all_object_ids) == 0:
            return image

        paired = list(zip(all_object_ids, all_object_boxes, all_object_masks, all_object_classes))
        paired.sort(key=lambda x: x[0])

        all_object_ids = [p[0] for p in paired]
        all_object_boxes = [p[1] for p in paired]
        all_object_masks = [p[2] for p in paired]
        all_object_classes = [p[3] for p in paired]

        all_object_masks = np.concatenate(all_object_masks, axis=0)
        detections = sv.Detections(
            xyxy=np.array(all_object_boxes),
            mask=all_object_masks,
            class_id=np.array(all_object_ids, dtype=np.int32),
        )
        labels = [f"{instance_id}: {class_name}" for instance_id, class_name in zip(all_object_ids, all_object_classes)]

        annotated_frame = image.copy()
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_frame = mask_annotator.annotate(annotated_frame, detections)
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

        return annotated_frame


# ================== ZMQ tracking worker ==================

HOST_IP = "192.168.1.99"      # TODO: 改成你的宿主机 IP
TEXT_PROMPT = "cup."          # GroundingDINO 的 prompt：必须小写 + 句号
DETECTION_INTERVAL = 10
SHOW_OVERLAY_FPS = True


def main():
    print("[WORKER] Initializing IncrementalObjectTracker (SAM2 tiny, tracking mode)...")
    tracker = IncrementalObjectTracker(
        grounding_model_id="IDEA-Research/grounding-dino-tiny",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_ckpt_path="./checkpoints/sam2.1_hiera_tiny.pt",
        device=DEVICE,
        prompt_text=TEXT_PROMPT,
        detection_interval=DETECTION_INTERVAL,
    )
    tracker.set_prompt(TEXT_PROMPT)
    print("[WORKER] Tracker initialized.")

    ctx = zmq.Context()

    # SUB：收宿主机的 RGBD（rgb+depth 成对）
    # 输入 multipart: [b"rgbd", ts, meta_json, rgb_jpg, depth_png16]
    sub_sock = ctx.socket(zmq.SUB)
    sub_sock.connect(f"tcp://{HOST_IP}:5555")
    sub_sock.setsockopt_string(zmq.SUBSCRIBE, "rgbd")
    # 仍用“手动 drain”只处理最新，保证低延迟
    sub_sock.setsockopt(zmq.RCVHWM, 1000)

    # PUB：发分割 mask + depth 回宿主机（只保留最新）
    # 输出 multipart: [b"segd", ts, meta_json, mask_png, depth_png16, overlay_jpg]
    pub_sock = ctx.socket(zmq.PUB)
    pub_sock.connect(f"tcp://{HOST_IP}:5556")
    pub_sock.setsockopt(zmq.SNDHWM, 1)
    try:
        pub_sock.setsockopt(zmq.CONFLATE, 1)
    except Exception:
        pass
    pub_sock.setsockopt(zmq.LINGER, 0)

    print("[WORKER] Tracking ZMQ worker started.")
    print(f"         SUB from tcp://{HOST_IP}:5555 (topic 'rgbd')")
    print(f"         PUB   to tcp://{HOST_IP}:5556 (topic 'segd', SNDHWM=1)")

    recv_frame_count = 0
    proc_frame_count = 0
    last_recv_time = time.time()
    last_proc_time = time.time()

    avg_infer_time = 0.0
    alpha = 0.1

    overlay_recv_fps = 0.0
    overlay_proc_fps = 0.0
    overlay_infer_ms = 0.0

    frame_idx = 0

    try:
        while True:
            # ===== 1) 至少收一帧，然后 DRain 队列，只保留最后一帧 =====
            latest_parts = sub_sock.recv_multipart()
            recv_frame_count += 1

            while True:
                try:
                    parts = sub_sock.recv_multipart(flags=zmq.NOBLOCK)
                    latest_parts = parts
                    recv_frame_count += 1
                except zmq.Again:
                    break

            if len(latest_parts) != 5:
                print(f"[WORKER] Unexpected multipart length: {len(latest_parts)} (expected 5)")
                continue

            topic, ts_bytes, meta_bytes, rgb_jpg_bytes, depth_png_bytes = latest_parts
            raw_rgb_jpg_bytes = rgb_jpg_bytes

            # 解 RGB（JPEG）
            arr = np.frombuffer(rgb_jpg_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                print("[WORKER] Warning: decoded RGB frame is None")
                continue

            frame_idx += 1
            curr_h, curr_w = frame_bgr.shape[:2]

            # ===== 2) tracker 需要 RGB =====
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            t0 = time.time()
            try:
                annotated_rgb, mask_array = tracker.add_image(frame_rgb)
            except Exception as e:
                print(f"[WORKER] ERROR in tracker.add_image: {e}")
                annotated_rgb, mask_array = None, None
            t1 = time.time()

            infer_time = t1 - t0
            proc_frame_count += 1
            avg_infer_time = infer_time if avg_infer_time == 0.0 else (1 - alpha) * avg_infer_time + alpha * infer_time

            # ===== 3) 准备 overlay（可选）=====
            if annotated_rgb is None or not isinstance(annotated_rgb, np.ndarray):
                overlay_bgr = frame_bgr
            else:
                overlay_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            # ===== 4) 准备 union 二值 mask（用于点云过滤）=====
            if mask_array is None:
                mask_u8 = np.zeros((curr_h, curr_w), dtype=np.uint8)
            else:
                union = (mask_array > 0)
                mask_u8 = (union.astype(np.uint8) * 255)

            ok_m, buf_m = cv2.imencode(".png", mask_u8)  # PNG 更适合 mask
            if not ok_m:
                continue
            mask_png_bytes = buf_m.tobytes()

            # ===== 5) FPS 统计与 overlay =====
            now = time.time()
            if now - last_recv_time >= 1.0:
                overlay_recv_fps = recv_frame_count / (now - last_recv_time)
                recv_frame_count = 0
                last_recv_time = now

            if now - last_proc_time >= 1.0 and proc_frame_count > 0:
                overlay_proc_fps = proc_frame_count / (now - last_proc_time)
                overlay_infer_ms = avg_infer_time * 1000.0

                print(
                    f"[WORKER] frame {frame_idx:5d} | "
                    f"SIZE: {curr_w}x{curr_h} | "
                    f"RECV FPS: {overlay_recv_fps:5.2f} | "
                    f"PROC FPS: {overlay_proc_fps:5.2f} | "
                    f"avg infer: {overlay_infer_ms:6.1f} ms"
                )
                proc_frame_count = 0
                last_proc_time = now

            if SHOW_OVERLAY_FPS:
                text = (
                    f"{curr_w}x{curr_h} | "
                    f"PROC: {overlay_proc_fps:4.1f} FPS  "
                    f"RECV: {overlay_recv_fps:4.1f} FPS  "
                    f"infer: {overlay_infer_ms:5.1f} ms"
                )
                cv2.putText(
                    overlay_bgr,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )

            # ===== 6) 编码 overlay 并发布：segd（mask+depth 成对）=====
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            ok_o, buf_o = cv2.imencode(".jpg", overlay_bgr, encode_params)
            if not ok_o:
                continue
            overlay_jpg_bytes = buf_o.tobytes()

            try:
                pub_sock.send_multipart([
                    b"segd",
                    ts_bytes,
                    meta_bytes,          # 原样透传 meta（包含 intrinsics / depth_scale 等）
                    mask_png_bytes,      # 分割 union mask（PNG）
                    depth_png_bytes,     # 原始 depth（PNG16）—— 与该帧成对
                    raw_rgb_jpg_bytes,
                    overlay_jpg_bytes,   # 叠加可视化（JPG，可选）
                ])
            except zmq.error.Again:
                print("[WORKER] PUB queue full, drop this frame")

    finally:
        ctx.term()
        print("[WORKER] Worker stopped.")


if __name__ == "__main__":
    main()
