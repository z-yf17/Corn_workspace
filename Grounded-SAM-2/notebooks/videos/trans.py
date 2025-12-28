import os
import cv2
import argparse


def extract_frames(
    video_path: str,
    output_dir: str,
    image_format: str = "jpg",
    start_index: int = 0,
    frame_step: int = 1,
    resize_width: int = None,
    resize_height: int = None,
):
    """
    从 MP4 视频中导出图片序列。

    参数：
        video_path   : 输入视频路径（mp4）
        output_dir   : 导出的图片目录
        image_format : 图片格式，"jpg" 或 "png"
        start_index  : 起始帧编号（文件名）
        frame_step   : 抽帧步长，例如 1 表示每帧都导出，5 表示每隔 5 帧导出一张
        resize_width : 目标宽度（像素），None 表示保持原视频宽度
        resize_height: 目标高度（像素），None 表示保持原视频高度
    """

    assert image_format.lower() in ["jpg", "jpeg", "png"], "image_format 必须是 jpg / jpeg / png"

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    frame_idx = 0           # 视频中的原始帧编号
    save_idx = start_index  # 导出图片的编号

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 读到结尾

        # 抽帧：只在 frame_idx % frame_step == 0 时保存
        if frame_idx % frame_step == 0:
            # 可选：缩放
            if resize_width is not None and resize_height is not None:
                frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

            filename = f"{save_idx:05d}.{image_format.lower()}"
            save_path = os.path.join(output_dir, filename)

            # 对 jpg 使用高质量压缩
            if image_format.lower() in ["jpg", "jpeg"]:
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(save_path, frame)

            save_idx += 1

        frame_idx += 1

    cap.release()
    print(f"处理完成：总帧数 {frame_idx}，导出图片 {save_idx - start_index} 张，保存在 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="将 MP4 视频转为图片序列")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径（mp4）")
    parser.add_argument("--out_dir", type=str, required=True, help="输出图片目录")
    parser.add_argument("--format", type=str, default="jpg", help="图片格式：jpg 或 png，默认 jpg")
    parser.add_argument("--start_idx", type=int, default=0, help="输出图片起始编号，默认 0 对应 00000")
    parser.add_argument("--frame_step", type=int, default=1, help="抽帧步长，1 表示每帧都保存")
    parser.add_argument("--width", type=int, default=None, help="缩放后的宽度（像素），默认不缩放")
    parser.add_argument("--height", type=int, default=None, help="缩放后的高度（像素），默认不缩放")

    args = parser.parse_args()

    extract_frames(
        video_path=args.video,
        output_dir=args.out_dir,
        image_format=args.format,
        start_index=args.start_idx,
        frame_step=args.frame_step,
        resize_width=args.width,
        resize_height=args.height,
    )


if __name__ == "__main__":
    main()
