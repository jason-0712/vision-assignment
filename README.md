# RoboMaster 视觉作业：路线 A

使用给定 ONNX 模型检测装甲板，按每个检测的图案选择物理尺寸，以 PnP 求四角点的相机坐标（米），逐帧输出 JSONL。全部颜色均参与检测，不训练模型。

## 环境

实测 Python 3.13.9、macOS arm64、ONNX Runtime CPUExecutionProvider。以下命令均从仓库根目录执行，先确保 `python --version` 为 3.13.x：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r detection/requirements.txt
```

Windows 激活命令为 `.venv\Scripts\activate`。

## 运行

```bash
python detection/run_detection.py \
  --model detection/assets/0526.onnx \
  --video detection/assets/small.mp4 \
  --calibration detection/assets/calibration.yaml \
  --output detection/outputs/small.jsonl

python detection/run_detection.py \
  --model detection/assets/0526.onnx \
  --video detection/assets/large.mp4 \
  --calibration detection/assets/calibration.yaml \
  --output detection/outputs/large.jsonl
```

默认置信度阈值 0.5、NMS IoU 阈值 0.5，类别无关 NMS 按 sigmoid(objectness) 排序；2 个 CPU 推理线程，预热 1 次。使用原始 float16 ONNX，letterbox 后推理，将角点恢复到原图再应用原始标定。PnP 使用 IPPE，选择四角点正深度且重投影 RMS 最低的有限候选。求解失败时保留二维检测，位姿相关字段同时为 null。

程序还会生成运行统计和抽样图片；这些辅助文件不纳入提交。

## 验证

```bash
python -m unittest discover -s detection/tests -v

python detection/validate_results.py \
  --results detection/outputs/small.jsonl \
  --video detection/assets/small.mp4 \
  --calibration detection/assets/calibration.yaml \
  --expected-frames 360 \
  --report detection/outputs/small.validation.json

python detection/validate_results.py \
  --results detection/outputs/large.jsonl \
  --video detection/assets/large.mp4 \
  --calibration detection/assets/calibration.yaml \
  --expected-frames 360 \
  --report detection/outputs/large.validation.json
```

9 项关键测试通过；独立验证器检查全部解码帧、严格 JSON、连续索引、时间戳、有限值、null 配对、物理尺寸、正深度及重算的重投影误差。

| 视频 | JSONL 帧数 | 检测数 | 无检测帧数 | PnP 失败检测数 | RMS 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 360 | 395 | 162 | 0 | 0.1182 px |
| large | 360 | 150 | 260 | 0 | 0.3396 px |

large 第 137 帧最大 RMS 为 11.09 px，装甲板在画面边缘截断。未设置额外的 RMS 拒绝阈值，保留原始观测及求解结果。没有人工真值，不将 PnP 成功率或重投影误差称为检测准确率或真实距离精度；平面多解和类别错误导致的尺寸误选仍可能影响深度。

## 来源

基于 [官方模板](https://github.com/HKUAstar/vision-training-assignment)，资源基线为 `c9fa5e9b66b61f5edcc23341ac7dac83a840bcc3`，模型、录像与标定保持原样。参考[指定模型实现](https://github.com/broalantaps/RobotDetectionModel/blob/babaebd6c8b3aeceda0ca924772b2d70d51801b4/OpenvinoInfer.cpp)及 [OpenCV PnP 文档](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)。本题颜色映射、置信度及角点对应关系以[作业要求](https://training.herkules.dev/learn/vision-assignment)为准。使用 AI 辅助编写和排错。
