"""作业模型专用的 22 维解码；不是通用 YOLO anchor/stride 解码。"""

import numpy as np

from preprocess import LetterboxTransform


COLORS = ("blue", "red", "neutral", "purple")
PATTERNS = ("sentry", "1", "2", "3", "4", "5", "outpost", "base_small", "base_big")


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    exp = np.exp(-np.abs(values))
    return np.where(values >= 0, 1 / (1 + exp), exp / (1 + exp))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """类别无关 NMS：保留 objectness 更高的候选，矩形坐标为 xyxy。"""
    order = np.argsort(-scores, kind="stable")
    areas = np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    keep = []
    while len(order):
        best = int(order[0])
        keep.append(best)
        rest = order[1:]
        lo = np.maximum(boxes[best, :2], boxes[rest, :2])
        hi = np.minimum(boxes[best, 2:], boxes[rest, 2:])
        intersection = np.prod(np.maximum(0, hi - lo), axis=1)
        union = areas[best] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
        order = rest[iou <= iou_threshold]
    return keep


def decode_predictions(output, transform: LetterboxTransform, confidence=0.5, nms_iou=0.5):
    if not (0 <= confidence <= 1 and 0 <= nms_iou <= 1):
        raise ValueError("置信度与 NMS 阈值必须在 [0, 1] 内")
    output = np.asarray(output)
    if output.shape != (1, 25200, 22) or not np.isfinite(output).all():
        raise ValueError(f"模型输出形状或数值异常：{output.shape}")
    rows = output[0]
    scores = sigmoid(rows[:, 8])
    mask = scores >= confidence
    rows, scores = rows[mask], scores[mask]
    # 0:8 已经是输入图像上的四角点，不需要再用 anchor 或 stride 解码。
    corners = rows[:, :8].reshape(-1, 4, 2)
    boxes = np.concatenate((corners.min(axis=1), corners.max(axis=1)), axis=1)
    keep = nms(boxes, scores, nms_iou)
    detections = []
    for i in keep:
        detections.append({
            "color": COLORS[int(rows[i, 9:13].argmax())],
            "pattern": PATTERNS[int(rows[i, 13:22].argmax())],
            "confidence": float(scores[i]),
            "corners_px": transform.to_original(corners[i]).tolist(),
        })
    return detections, {"above_threshold": int(mask.sum()), "after_nms": len(keep)}
