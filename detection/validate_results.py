"""独立检查 JSONL 的格式、帧完整性、物理尺寸与重投影误差。"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from pose import board_points, load_calibration
from postprocess import COLORS, PATTERNS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def numeric_array(value, shape):
    require(type(value) is list and len(value) == shape[0], f"角点数组应为 {shape}")
    for row in value:
        require(type(row) is list and len(row) == shape[1], f"角点数组应为 {shape}")
        require(all(finite_number(x) for x in row), "角点必须为有限数值")
    return np.asarray(value, dtype=np.float64)


def validate_record(record, index, calibration):
    require(type(record) is dict and set(record) == {"frame_index", "timestamp_s", "detections"}, "每帧字段不正确")
    require(type(record["frame_index"]) is int and record["frame_index"] == index, "帧索引不连续或类型不正确")
    require(finite_number(record["timestamp_s"]) and abs(record["timestamp_s"] - index / 30) < 1e-9, "时间戳不正确")
    require(type(record["detections"]) is list, "detections 必须是数组")
    fields = {"color", "pattern", "confidence", "corners_px", "corners_camera_m", "reprojection_error_px"}
    for detection in record["detections"]:
        require(type(detection) is dict and set(detection) == fields, "检测字段不正确")
        require(type(detection["color"]) is str and detection["color"] in COLORS, "颜色不正确")
        require(type(detection["pattern"]) is str and detection["pattern"] in PATTERNS, "图案必须为约定字符串")
        require(finite_number(detection["confidence"]) and 0 <= detection["confidence"] <= 1, "置信度不正确")
        observed = numeric_array(detection["corners_px"], (4, 2))
        camera, error = detection["corners_camera_m"], detection["reprojection_error_px"]
        require((camera is None) == (error is None), "位姿失败时两个字段必须同时为 null")
        if camera is None:
            continue
        camera = numeric_array(camera, (4, 3))
        require((camera[:, 2] > 0).all(), "三维点必须在相机前方")
        require(finite_number(error) and error >= 0, "重投影误差不正确")
        physical = board_points(detection["pattern"])
        distances = np.linalg.norm(camera[:, None] - camera[None, :], axis=2)
        expected = np.linalg.norm(physical[:, None] - physical[None, :], axis=2)
        require(np.allclose(distances, expected, atol=1e-6, rtol=1e-5), "三维角点尺寸或对应关系错误（应为米）")
        # 序列化的相机点已经位于相机坐标系，使用单位旋转、零平移重新投影。
        projected, _ = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), calibration.matrix, calibration.distortion)
        rms = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - observed) ** 2, axis=1))))
        require(math.isclose(rms, error, abs_tol=1e-5, rel_tol=1e-6), "保存的 RMS 与三维角点重算结果不符")


def reject_constant(value):
    raise ValueError(f"JSON 禁止非有限数值：{value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"JSON 重复字段：{key}")
        result[key] = value
    return result


def count_video_frames(path, calibration):
    cap = cv2.VideoCapture(str(path))
    require(cap.isOpened(), f"无法打开视频：{path}")
    count = 0
    try:
        require(np.isclose(cap.get(cv2.CAP_PROP_FPS), 30, atol=1e-6), "视频必须为 30 FPS")
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            require(frame.shape[:2] == (calibration.height, calibration.width), "视频分辨率与标定不符")
            count += 1
        require(count > 0 and (declared <= 0 or count == declared), "视频解码帧数异常")
    finally:
        cap.release()
    return count


def validate_results(results_path, video_path, calibration_path, expected_frames=None):
    calibration = load_calibration(calibration_path)
    video_frames = count_video_frames(video_path, calibration)
    if expected_frames is not None:
        require(video_frames == expected_frames, f"视频实际 {video_frames} 帧，要求 {expected_frames} 帧")
    lines = detections = failed_poses = empty_frames = 0
    with Path(results_path).open(encoding="utf-8") as file:
        for index, line in enumerate(file):
            try:
                record = json.loads(line, parse_constant=reject_constant, object_pairs_hook=unique_object)
                validate_record(record, index, calibration)
            except (ValueError, TypeError, OverflowError, cv2.error) as exc:
                raise ValueError(f"JSONL 第 {index + 1} 行：{exc}") from exc
            lines += 1
            detections += len(record["detections"])
            failed_poses += sum(d["corners_camera_m"] is None for d in record["detections"])
            empty_frames += not record["detections"]
    require(lines == video_frames, f"JSONL {lines} 行，但视频实际解码 {video_frames} 帧")
    return {"status": "passed", "video_decoded_frames": video_frames, "jsonl_lines": lines, "last_frame_index": lines - 1, "last_timestamp_s": (lines - 1) / 30, "detections": detections, "pnp_failed_detections": failed_poses, "no_detection_frames": empty_frames, "checks": ["strict_json_schema", "frame_indices_and_timestamps", "all_decoded_frames", "finite_values", "null_pairs", "positive_depth", "physical_pairwise_distances", "recomputed_reprojection_rms"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--report")
    args = parser.parse_args()
    try:
        report = validate_results(args.results, args.video, args.calibration, args.expected_frames)
    except (OSError, ValueError, TypeError, KeyError, cv2.error) as exc:
        parser.exit(1, f"验证失败：{exc}\n")
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
