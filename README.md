# RoboMaster 视觉组作业：路线 A 与路线 B

贾涵宇 UID：3036662505。

本仓库完成了检测和瞄准两条路线，二者可独立运行。

| 路线 | 内容 | 代码与说明 |
| --- | --- | --- |
| 路线 A | 从视频检测装甲板，用 PnP 恢复四角点的三维坐标，逐帧导出 JSONL | [detection](detection/README.md) |
| 路线 B | 根据延迟观测预测运动装甲板的位置，求解弹道并判断是否开火 | [aiming](aiming/README.md) |

## 路线 A：装甲板检测与位姿估计

使用给定 ONNX 模型检测装甲板，按每个检测的图案选择物理尺寸，以 PnP 求四角点的相机坐标（单位为米），逐帧输出 JSONL。全部颜色均参与检测，不训练模型。

### 环境

实测 Python 3.13.9、macOS arm64、ONNX Runtime CPUExecutionProvider。以下命令均从仓库根目录执行，先确保 `python --version` 为 3.13.x：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r detection/requirements.txt
```

Windows 激活命令为 `.venv\Scripts\activate`。

### 运行

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

### 验证

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

| 视频 | JSONL 帧数 | 检测数 | 无检测帧数 | PnP 失败检测数 | RMS 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 360 | 395 | 162 | 0 | 0.1182 px |
| large | 360 | 150 | 260 | 0 | 0.3396 px |

large 第 137 帧最大 RMS 为 11.09 px，装甲板在画面边缘截断。未设置额外的 RMS 拒绝阈值，保留原始观测及求解结果。没有人工真值，不将 PnP 成功率或重投影误差称为检测准确率或真实距离精度；平面多解和类别错误导致的尺寸误选仍可能影响深度。

## 路线 B：运动目标瞄准

实现位于 [aiming/src/lib.rs](aiming/src/lib.rs)。先补偿观测年龄和发射延迟，在粗估命中时刻选板，再固定装甲板索引迭代求解未来位置与飞行时间。最终根据完整三维法线和弹丸相对装甲板的速度判断是否从正面入射。无观测、目标不可达或迭代不收敛时不开火。

### 环境与运行

需要 Rust 1.98.1 或更高版本，以及本机 C/C++ 编译器和链接器。3D 调试器还需要桌面环境和支持 wgpu 的图形驱动；无界面测试无需 GPU。

从仓库根目录进入 `aiming` 后执行：

```sh
cd aiming

# 无界面测试。
cargo test --no-default-features --locked

# 界面与回放相关的单元测试。
cargo test --locked --lib

# 三个官方场景的仿真统计。
cargo run --no-default-features --locked --example evaluate

# 打开移动与自旋场景的 3D 调试器，并保存录制。
cargo run --locked -- --config scenarios/moving.json --record target/moving.mcap
```

录制文件已存在时请更换文件名。更多操作见 [aiming/README.md](aiming/README.md)。

### 验证结果

23 项无界面测试和另外 2 项界面/回放测试通过，Clippy 检查通过，3D 版本编译和 GPU 预览导出成功。仿真器、碰撞检测、固定物理参数、场景与依赖锁文件保持原样。

| 官方场景 | 发数 | 命中 | 未命中 | 待结算 | 命中率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 静止 | 80 | 80 | 0 | 0 | 100% |
| 自旋 | 80 | 80 | 0 | 0 | 100% |
| 平移与自旋 | 80 | 80 | 0 | 0 | 100% |

每个场景运行 20 秒，再继续仿真到 22.02 秒以结算全部弹丸。完整统计见 [aiming/results/evaluation.json](aiming/results/evaluation.json)。以上成绩仅对应这些确定性仿真场景，不代表所有场景或真实硬件的表现。

