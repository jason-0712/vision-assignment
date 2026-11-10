"""原图像素点 + 标定 + 物理尺寸 → 相机坐标系四角点（米）。"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from postprocess import PATTERNS


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    matrix: np.ndarray
    distortion: np.ndarray


@dataclass(frozen=True)
class PoseResult:
    camera_points: np.ndarray
    projected_points: np.ndarray
    error_px: float


def load_calibration(path) -> Calibration:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    width, height = data["image_width"], data["image_height"]
    matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(data["distort_coeffs"], dtype=np.float64)
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("标定图像尺寸应为正整数")
    if matrix.shape != (3, 3) or distortion.shape != (5,):
        raise ValueError("标定需要 3×3 相机矩阵和 5 个畸变系数")
    if not (np.isfinite(matrix).all() and np.isfinite(distortion).all()):
        raise ValueError("标定包含非有限数值")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError("相机矩阵格式不正确")
    return Calibration(width, height, matrix, distortion)


def board_points(pattern: str) -> np.ndarray:
    if pattern not in PATTERNS:
        raise ValueError(f"未知图案类别：{pattern}")
    # 大小由每个检测的图案决定，与视频文件名无关。
    width = 0.230 if pattern in ("1", "base_big") else 0.135
    height = 0.056
    return np.array([
        [-width / 2, -height / 2, 0],
        [-width / 2, +height / 2, 0],
        [+width / 2, +height / 2, 0],
        [+width / 2, -height / 2, 0],
    ], dtype=np.float64)


def estimate_pose(corners_px, pattern: str, calibration: Calibration):
    """返回 (位姿或 None, 失败原因或 None)；不对观测角点排序或修改。"""
    observed = np.ascontiguousarray(corners_px, dtype=np.float64)
    if observed.shape != (4, 2) or not np.isfinite(observed).all():
        return None, "invalid_image_points"
    contour = observed.astype(np.float32)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 1e-6:
        return None, "nonconvex_or_degenerate_points"
    physical = board_points(pattern)
    try:
        solutions = cv2.solvePnPGeneric(
            physical, observed, calibration.matrix, calibration.distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return None, "opencv_solver_error"
    if not solutions[0]:
        return None, "solver_no_solution"
    candidates = []
    for rvec, tvec in zip(solutions[1], solutions[2]):
        if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        # 每个角点都要旋转再平移。tvec 仅为板中心，不能复制四遍。
        camera_points = physical @ rotation.T + tvec.reshape(1, 3)
        if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
            continue
        projected, _ = cv2.projectPoints(physical, rvec, tvec, calibration.matrix, calibration.distortion)
        projected = projected.reshape(4, 2)
        # 先求每个点的二维欧氏误差，再对四个点求 RMS。
        rms = float(np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=1))))
        if np.isfinite(projected).all() and np.isfinite(rms):
            candidates.append(PoseResult(camera_points, projected, rms))
    if not candidates:
        return None, "no_finite_positive_depth_solution"
    return min(candidates, key=lambda candidate: candidate.error_px), None


def attach_pose(detection: dict, calibration: Calibration):
    pose, failure = estimate_pose(detection["corners_px"], detection["pattern"], calibration)
    result = {
        **detection,
        "corners_camera_m": None if pose is None else pose.camera_points.tolist(),
        "reprojection_error_px": None if pose is None else pose.error_px,
    }
    return result, None if pose is None else pose.projected_points, failure
