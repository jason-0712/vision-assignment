"""ONNX Runtime CPU 推理；始终使用给定的原始 ONNX 文件。"""

from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

from postprocess import decode_predictions
from preprocess import prepare_input


class Detector:
    def __init__(self, model_path, confidence=0.5, nms_iou=0.5, threads=2):
        if not Path(model_path).is_file():
            raise FileNotFoundError(model_path)
        if threads < 1:
            raise ValueError("threads 必须为正整数")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or (inputs[0].name, inputs[0].shape, inputs[0].type) != ("images", [1, 3, 640, 640], "tensor(float16)"):
            raise ValueError("模型输入不符合本作业接口")
        if len(outputs) != 1 or (outputs[0].name, outputs[0].shape, outputs[0].type) != ("output", [1, 25200, 22], "tensor(float)"):
            raise ValueError("模型输出不符合本作业接口")
        self.confidence, self.nms_iou = confidence, nms_iou
        self.metadata = {
            "providers": self.session.get_providers(),
            "input": {"name": inputs[0].name, "shape": inputs[0].shape, "type": inputs[0].type},
            "output": {"name": outputs[0].name, "shape": outputs[0].shape, "type": outputs[0].type},
        }

    def warmup(self, count=1):
        tensor = np.full((1, 3, 640, 640), np.float16(114 / 255), dtype=np.float16)
        for _ in range(count):
            self.session.run(["output"], {"images": tensor})

    def detect(self, frame):
        start = perf_counter()
        tensor, transform = prepare_input(frame)
        before_infer = perf_counter()
        output = self.session.run(["output"], {"images": tensor})[0]
        after_infer = perf_counter()
        detections, counts = decode_predictions(output, transform, self.confidence, self.nms_iou)
        return detections, {
            **counts,
            "preprocess_ms": (before_infer - start) * 1000,
            "inference_ms": (after_infer - before_infer) * 1000,
            "postprocess_ms": (perf_counter() - after_infer) * 1000,
        }
