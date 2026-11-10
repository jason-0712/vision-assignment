"""只检查本题容易静默出错的索引、坐标、筛选与几何。"""

import sys
import copy
import json
import unittest
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postprocess import COLORS, PATTERNS, decode_predictions, nms, sigmoid
from preprocess import LetterboxTransform, prepare_input
from pose import Calibration, attach_pose, board_points, estimate_pose
from validate_results import reject_constant, unique_object, validate_record


class ImagePipelineTests(unittest.TestCase):
    def test_letterbox_and_rgb_tensor(self):
        frame = np.full((1080, 1440, 3), [10, 20, 30], dtype=np.uint8)
        tensor, transform = prepare_input(frame)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float16)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual((transform.pad_left, transform.pad_top), (0, 80))
        np.testing.assert_allclose(tensor[0, :, 80, 0], np.array([30, 20, 10]) / 255, atol=1e-4)
        np.testing.assert_allclose(tensor[0, :, 0, 0], 114 / 255, atol=3e-4)
        np.testing.assert_allclose(transform.to_original([[0, 80], [640, 560]]), [[0, 0], [1440, 1080]])

    def test_odd_padding_and_rounding(self):
        _, transform = prepare_input(np.zeros((777, 1001, 3), dtype=np.uint8))
        # 777 * 640 / 1001 = 496.78...，实际缩放高 497，上补边 71，下补边 72。
        self.assertEqual(transform.pad_top, 71)
        np.testing.assert_allclose(transform.to_original([[0, 71], [640, 568]]), [[0, 0], [1001, 777]])
        self.assertNotEqual(transform.scale_x, transform.scale_y)

    def test_sigmoid_and_nms(self):
        np.testing.assert_allclose(sigmoid([-1000, 0, 1000]), [0, 0.5, 1])
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [30, 30, 40, 40]], dtype=float)
        self.assertEqual(nms(boxes, np.array([0.8, 0.9, 0.7]), 0.5), [1, 2])

    def test_decode_all_classes_and_objectness(self):
        output = np.full((1, 25200, 22), -100, dtype=np.float32)
        for i in range(9):
            x = i * 50
            output[0, i, :8] = [x, 20, x, 40, x + 20, 40, x + 20, 20]
            output[0, i, 8] = 2  # objectness 固定，不与类别 logit 相乘。
            output[0, i, 9 + i % 4] = i + 1
            output[0, i, 13 + i] = 10 + i
        detections, counts = decode_predictions(output, LetterboxTransform(0.5, 0.5, 0, 10))
        self.assertEqual(counts, {"above_threshold": 9, "after_nms": 9})
        self.assertEqual([d["pattern"] for d in detections], list(PATTERNS))
        self.assertEqual([d["color"] for d in detections], [COLORS[i % 4] for i in range(9)])
        for d in detections:
            self.assertAlmostEqual(d["confidence"], float(sigmoid(2)))
        np.testing.assert_allclose(detections[0]["corners_px"], [[0, 20], [0, 60], [40, 60], [40, 20]])
        empty, _ = decode_predictions(np.full_like(output, -100), LetterboxTransform(1, 1, 0, 0))
        self.assertEqual(empty, [])


class PoseTests(unittest.TestCase):
    def setUp(self):
        self.calibration = Calibration(1440, 1080, np.array([[1200., 0, 720], [0, 1200, 540], [0, 0, 1]]), np.array([-0.08, 0.12, -0.0003, -0.0007, 0.]))

    def test_synthetic_pose_for_small_and_large(self):
        for pattern, width in (("3", 0.135), ("1", 0.230), ("base_big", 0.230)):
            physical = board_points(pattern)
            rvec = np.array([0.18, -0.32, 0.09])
            tvec = np.array([0.15, -0.07, 2.8])
            observed, _ = cv2.projectPoints(physical, rvec, tvec, self.calibration.matrix, self.calibration.distortion)
            pose, failure = estimate_pose(observed.reshape(4, 2), pattern, self.calibration)
            self.assertIsNone(failure)
            self.assertIsNotNone(pose)
            self.assertLess(pose.error_px, 1e-6)
            self.assertTrue((pose.camera_points[:, 2] > 0).all())
            self.assertAlmostEqual(np.linalg.norm(pose.camera_points[2] - pose.camera_points[1]), width)
            self.assertAlmostEqual(np.linalg.norm(pose.camera_points[1] - pose.camera_points[0]), 0.056)
            # 对这一非退化合成视角，最低误差解应还原生成时的三维角点。
            rotation, _ = cv2.Rodrigues(rvec)
            np.testing.assert_allclose(pose.camera_points, physical @ rotation.T + tvec, atol=1e-6)

    def test_failure_preserves_2d_detection(self):
        detection = {"color": "purple", "pattern": "2", "confidence": 0.9, "corners_px": [[1, 1]] * 4}
        result, projected, failure = attach_pose(detection, self.calibration)
        self.assertIsNotNone(failure)
        self.assertIsNone(projected)
        self.assertEqual(result["corners_px"], detection["corners_px"])
        self.assertIsNone(result["corners_camera_m"])
        self.assertIsNone(result["reprojection_error_px"])


class OutputValidationTests(unittest.TestCase):
    setUp = PoseTests.setUp
    # 复用合成标定；通过序列化后的三维点检查单位与 RMS，而非只检查 solver 返回值。
    def test_serialized_geometry_and_rms(self):
        physical = board_points("1")
        observed, _ = cv2.projectPoints(physical, np.array([0.1, -0.2, 0.05]), np.array([0.1, 0.05, 3.]), self.calibration.matrix, self.calibration.distortion)
        detection = {"color": "red", "pattern": "1", "confidence": 0.8, "corners_px": observed.reshape(4, 2).tolist()}
        result, _, _ = attach_pose(detection, self.calibration)
        record = json.loads(json.dumps({"frame_index": 0, "timestamp_s": 0.0, "detections": [result]}, allow_nan=False))
        validate_record(record, 0, self.calibration)
        corrupted = copy.deepcopy(record)
        corrupted["detections"][0]["corners_camera_m"] = (np.array(result["corners_camera_m"]) * 1000).tolist()
        with self.assertRaisesRegex(ValueError, "尺寸"):
            validate_record(corrupted, 0, self.calibration)
        corrupted = copy.deepcopy(record)
        corrupted["detections"][0]["reprojection_error_px"] = 100
        with self.assertRaisesRegex(ValueError, "RMS"):
            validate_record(corrupted, 0, self.calibration)

    def test_empty_frame_and_null_pair(self):
        validate_record({"frame_index": 3, "timestamp_s": 0.1, "detections": []}, 3, self.calibration)
        invalid = {"frame_index": 3, "timestamp_s": 0.2, "detections": []}
        with self.assertRaises(ValueError):
            validate_record(invalid, 3, self.calibration)
        detection = {"color": "neutral", "pattern": "5", "confidence": 0.7, "corners_px": [[1, 1]] * 4, "corners_camera_m": None, "reprojection_error_px": None}
        record = {"frame_index": 0, "timestamp_s": 0.0, "detections": [detection]}
        validate_record(record, 0, self.calibration)
        detection["reprojection_error_px"] = 0
        with self.assertRaisesRegex(ValueError, "null"):
            validate_record(record, 0, self.calibration)

    def test_strict_json(self):
        for source in ('{"x": NaN}', '{"x": Infinity}', '{"x": 1, "x": 2}'):
            with self.assertRaises(ValueError):
                json.loads(source, parse_constant=reject_constant, object_pairs_hook=unique_object)


if __name__ == "__main__":
    unittest.main()
