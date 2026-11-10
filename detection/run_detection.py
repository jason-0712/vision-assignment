"""完整流水线：顺序解码 → 二维检测 → PnP → 一帧一行 JSONL。"""

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import cv2
import numpy as np

from detector import Detector
from pose import attach_pose, load_calibration
from visualize import draw_detections


def distribution(values):
    if not values:
        return None
    return {"mean": float(np.mean(values)), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def process_video(args):
    output = Path(args.output)
    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    # 防止误把输入文件用作输出。
    if output.resolve() in {Path(p).resolve() for p in (args.video, args.model, args.calibration)}:
        raise ValueError("输出路径不能覆盖输入文件")
    if args.sample_every < 1 or args.warmup < 0:
        raise ValueError("sample-every 应为正整数，warmup 不能为负")
    calibration = load_calibration(args.calibration)
    start_setup = perf_counter()
    detector = Detector(args.model, args.confidence, args.nms_iou, args.threads)
    detector.warmup(args.warmup)
    setup_s = perf_counter() - start_setup
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"视频无法打开：{video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or not np.isclose(fps, 30, atol=1e-6):
        cap.release()
        raise ValueError(f"本作业要求 30 FPS，实际视频报告 {fps}")
    output.parent.mkdir(parents=True, exist_ok=True)
    visual_dir = output.parent / (output.stem + "_visualizations")
    visual_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    colors, patterns, failures = Counter(), Counter(), Counter()
    timings = {"preprocess_ms": [], "inference_ms": [], "postprocess_ms": [], "pose_ms": []}
    errors = []
    frames = empty_frames = detection_count = candidate_count = pose_successes = 0
    worst = None
    first_empty_saved = first_failure_saved = False
    sample_names = []
    start_loop = perf_counter()

    def render(frame, detections, projections, frame_index, filename):
        title = f"{video_path.name} frame={frame_index} | green dots=observed; cyan crosses=reprojected | 0=UL 1=LL 2=LR 3=UR"
        image = draw_detections(frame, detections, title, projections)
        if not cv2.imwrite(str(visual_dir / filename), image):
            raise RuntimeError(f"可视化保存失败：{filename}")
        sample_names.append(filename)

    try:
        with temporary.open("w", encoding="utf-8") as file:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                if (width, height) != (calibration.width, calibration.height):
                    raise ValueError(f"第 {frames} 帧分辨率 {width}×{height} 与标定不符")
                detections_2d, stats = detector.detect(frame)
                for key in ("preprocess_ms", "inference_ms", "postprocess_ms"):
                    timings[key].append(stats[key])
                candidate_count += stats["above_threshold"]
                before_pose = perf_counter()
                detections, projections = [], []
                has_failure = False
                for detection in detections_2d:
                    result, projected, failure = attach_pose(detection, calibration)
                    detections.append(result)
                    projections.append(projected)
                    colors[result["color"]] += 1
                    patterns[result["pattern"]] += 1
                    if failure is None:
                        pose_successes += 1
                        errors.append(result["reprojection_error_px"])
                    else:
                        failures[failure] += 1
                        has_failure = True
                timings["pose_ms"].append((perf_counter() - before_pose) * 1000)
                detection_count += len(detections)
                record = {"frame_index": frames, "timestamp_s": frames / 30, "detections": detections}
                file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                if frames % args.sample_every == 0:
                    render(frame, detections, projections, frames, f"frame_{frames:06d}.jpg")
                if not detections:
                    empty_frames += 1
                    if not first_empty_saved:
                        render(frame, detections, projections, frames, "first_no_detection.jpg")
                        first_empty_saved = True
                if has_failure and not first_failure_saved:
                    render(frame, detections, projections, frames, "first_pnp_failure.jpg")
                    first_failure_saved = True
                frame_error = max((d["reprojection_error_px"] for d in detections if d["reprojection_error_px"] is not None), default=-1)
                if frame_error >= 0 and (worst is None or frame_error > worst[0]):
                    worst = (frame_error, frames, frame.copy(), detections, projections)
                last = (frame, detections, projections, frames)
                frames += 1
                if frames % 90 == 0:
                    print(f"{video_path.name}: {frames} 帧，{detection_count} 个检测", file=sys.stderr)
        if frames == 0 or (declared_frames > 0 and frames != declared_frames):
            raise RuntimeError(f"视频解码帧数异常：实际 {frames}，容器报告 {declared_frames}")
        if last[3] % args.sample_every != 0:
            render(*last, f"frame_{last[3]:06d}.jpg")
        if worst is not None:
            render(worst[2], worst[3], worst[4], worst[1], "worst_reprojection.jpg")
        # 只有完整处理成功，才把 partial 文件转为正式结果。
        temporary.replace(output)
    finally:
        cap.release()
    elapsed_s = perf_counter() - start_loop
    summary = {
        "video": str(video_path), "output": str(output),
        "decoded_frames": frames, "container_frames": declared_frames,
        "fps": fps, "resolution": [calibration.width, calibration.height],
        "no_detection_frames": empty_frames,
        "candidates_above_threshold": candidate_count,
        "detections_after_nms": detection_count,
        "pnp_success_detections": pose_successes,
        "pnp_failed_detections": sum(failures.values()),
        "pnp_success_rate_per_detection": pose_successes / detection_count if detection_count else None,
        "pnp_failure_reasons": dict(failures),
        "color_counts": dict(colors), "pattern_counts": dict(patterns),
        "reprojection_rms_px_successes_only": distribution(errors),
        "worst_reprojection_frame": None if worst is None else worst[1],
        "timings_ms_per_frame": {key: distribution(values) for key, values in timings.items()},
        "setup_and_warmup_s": setup_s, "processing_including_samples_s": elapsed_s,
        "config": {"confidence": args.confidence, "nms_iou": args.nms_iou, "nms": "class_agnostic_xyxy_objectness", "threads": args.threads, "warmup_runs": args.warmup, "sample_every": args.sample_every, "pose": "IPPE_min_RMS_positive_depth", "pose_error_rejection_threshold": None},
        "model_metadata": detector.metadata,
        "input_sha256": {Path(path).name: sha256(path) for path in (args.model, args.video, args.calibration)},
        "environment": {"python": platform.python_version(), "system": platform.platform(), "processor": platform.processor(), "dependencies": {name: version(name) for name in ("numpy", "opencv-python-headless", "onnxruntime", "PyYAML")}},
        "visualizations": {"directory": str(visual_dir), "files": sample_names},
    }
    write_json(output.with_suffix(".summary.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=90)
    args = parser.parse_args()
    try:
        summary = process_video(args)
    except (OSError, ValueError, RuntimeError, KeyError, cv2.error) as exc:
        parser.exit(1, f"处理失败：{exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
