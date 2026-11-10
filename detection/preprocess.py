"""把 OpenCV 的 BGR 图像变成模型输入，并记录可逆的像素坐标变换。"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int

    def to_original(self, points: np.ndarray) -> np.ndarray:
        """保持角点原顺序；不裁剪角点，也不按屏幕位置重新排序。"""
        points = np.asarray(points, dtype=np.float64)
        return (points - [self.pad_left, self.pad_top]) / [self.scale_x, self.scale_y]


def prepare_input(frame: np.ndarray, size: int = 640):
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("输入应为 uint8 HWC BGR 图像")
    height, width = frame.shape[:2]
    if min(height, width, size) <= 0:
        raise ValueError("图像和模型尺寸必须大于零")
    ratio = min(size / width, size / height)
    resized_w = max(1, round(width * ratio))
    resized_h = max(1, round(height * ratio))
    left = (size - resized_w) // 2
    top = (size - resized_h) // 2
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_h, left : left + resized_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    # HWC → CHW → NCHW；先用 float32 归一化，再转换到模型声明的 float16。
    tensor = np.ascontiguousarray((rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None], dtype=np.float16)
    transform = LetterboxTransform(resized_w / width, resized_h / height, left, top)
    return tensor, transform
