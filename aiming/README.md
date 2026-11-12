# 瞄准作业 贾涵宇 3036662505

## 实现说明

路线 B 已完成，核心逻辑在 [src/lib.rs](src/lib.rs)，仿真器与固定物理参数保持原样。

实现先根据观测年龄、发射延迟与粗估飞行时间选出预计正面朝向炮口的装甲板，
再固定该板索引，迭代求解未来板中心和低弹道。迭代最多 32 次，飞行时间收敛容差为
`1e-8 s`；最终位置残差不得超过 `1e-5 m`。最终检查完整三维法线与弹丸相对板速度，
朝向和入射余弦均至少为 0.5 才开火。缺失/无效观测、不可达或不收敛时不提交射击指令。

在原始仿真器中，三个官方场景均运行 20 秒，再继续仿真至 22.02 秒以结算所有弹丸：

| 场景 | 接受发数 | 命中 | 未命中 | 待结算 | 命中率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 静止 `stationary` | 80 | 80 | 0 | 0 | 100% |
| 自旋 `spinning` | 80 | 80 | 0 | 0 | 100% |
| 平移与自旋 `moving` | 80 | 80 | 0 | 0 | 100% |

这是上述确定性仿真的成绩，不代表所有运动状态或真实硬件的命中率。
验证环境为 Rust 1.98.1；23 项无界面测试及另外 2 项界面/回放测试通过。新增控制器测试涵盖启动无观测、无效输入、不可达、
背面禁射、绝对时间平移、额外观测延迟及不同距离、速度、自旋与高度。
原有两项针对“空模板必须不开火”的断言，已更新为完成后应输出指令和记录命中的断言；
原有碰撞、物理、时序测试保留。

在本目录复现：

```sh
cargo test --no-default-features --locked
cargo test --locked --lib
cargo run --no-default-features --locked --example evaluate
cargo run --locked -- --config scenarios/moving.json --record target/moving.mcap
```

第三条命令会输出逐场景 JSON 统计。第四条打开 3D 调试器；录制文件已存在时请换一个新文件名。
完整成绩见 [results/evaluation.json](results/evaluation.json)。
实现参考本仓库的 `Target`、`Plate`、`Trajectory` 接口及下文作业要求，使用 AI 辅助编写与验证。

## 任务

机器人会平移和自旋，你需要瞄准装甲板并决定何时开火。仿真器提供精确的目标模型，
但观测描述的是过去的状态。你需要预测弹丸到达时装甲板的位置。

只需修改 [src/lib.rs](src/lib.rs)，其中包含 `aim` 和相关的类型定义。
绘图接口位于 [src/ui.rs](src/ui.rs)。`src/sim/` 包含仿真、界面和 MCAP 功能，请勿修改。
本实现会绘制观测板、预计命中板、弹道和预计命中点，并显示命中计数、飞行时间及入射余弦。

## 参考效果

https://github.com/user-attachments/assets/2a08b028-4c0a-400b-9102-b1cc2989b7e8

[观看参考视频](assets/tracking-reference.mp4)（4K、60 FPS、22 秒）。
视频展示了完成作业后的跟踪和射击效果。绿色轮廓表示预计命中时的装甲板姿态，
橙色曲线表示规划弹道。视频供实现时对照，仓库中不包含完整瞄准解答。

## 环境与硬件

- 安装 [Rust / rustup](https://www.rust-lang.org/tools/install)，Rust 版本至少为 **1.98.1**。
- 需要本机 C/C++ 编译器和链接器。macOS 使用 Xcode Command Line Tools；
  Windows 使用 MSVC C++ Build Tools；Linux 需要编译工具和窗口后端的开发库。
- 3D 界面需要桌面环境和支持 wgpu 的图形驱动，例如 Metal、Vulkan 或 DX12，不需要独立显卡。
- 编译约需 **1 GB 内存**，运行约需 **150 MB 内存**，实际用量因环境而异。
- 只有导出 MP4 才需要 `ffmpeg`。

## 运行

在 `aiming` 目录执行：

```sh
cargo run --locked
cargo run --locked -- --config scenarios/stationary.json
cargo run --locked -- --config scenarios/spinning.json
cargo run --locked -- --config scenarios/moving.json
```

先做静止目标，再做自旋，最后处理平移与自旋同时发生的情况。
这个 crate 独立编译，不依赖 OpenCV 或相机 SDK；运行器始终调用你实现的 `aim`。

## 需要实现的接口

```rust
pub fn aim(input: &Input, debug: &mut Gizmos) -> Option<Command>
```

仿真器每 10 ms 调用一次 `aim`。`Input` 包含当前仿真时间、延迟送达的观测、已知几何尺寸，
以及已经结算的射击计数。每次观测包含采集时间和
`x_m, y_m, theta_rad, vx_m_s, vy_m_s, omega_rad_s`。
最初的观测延迟尚未结束时没有观测；接口不会提供目标当前时刻的真值。

返回 `Some(Command { yaw_rad, pitch_rad, fire })`。`fire: false` 表示只瞄准不开火；
`None` 表示不开火且不提交瞄准指令。被接受的射击指令会锁定角度，在发射延迟后沿固定弹道飞行，
后续指令不能改变已经发射的弹丸。持续请求开火会受到射速限制。
命中率按 `hits / (hits + misses)` 计算，尚在飞行的弹丸未结算。

你需要处理缺失观测和不可达弹道，补偿延迟，预测平移和旋转，并选择能够从正面命中的装甲板。

## 固定物理参数

所有发射物理量和时序都在 [src/lib.rs](src/lib.rs) 的 `physics` 中定义为常量。
`Input` 和场景 JSON 不含可配置物理参数。弹道工具、绘图、界面和仿真统一使用这些常量：

| 常量 | 数值 |
| --- | --- |
| `MUZZLE_M` | `[0.0, 0.0, 0.3]` m |
| `MUZZLE_SPEED_M_S` | 25 m/s |
| `GRAVITY_M_S2` | 9.81 m/s² |
| `OBSERVATION_DELAY_S` | 0.05 s |
| `LAUNCH_DELAY_S` | 0.02 s |
| `FIRE_PERIOD_S` | 0.25 s |
| `MAX_FLIGHT_S` | 发射后 2 s |
| `CONTROL_PERIOD_S` | 0.01 s |

调用工具函数时不需要额外传入物理参数。

## 第一步：确定延迟

先检查 `input.observation` 是否存在。观测在采集时刻是精确的，但并不代表当前状态。
用 `input.now_s - observation.time_s` 求出观测年龄，再考虑从指令被接受到实际发射之间的
`physics::LAUNCH_DELAY_S`。

目标在这两段延迟和弹丸飞行期间都会运动。把观测年龄、发射延迟和飞行时间分别处理，
避免漏算或重复相加。缺少观测时不开火；本题不要求跟踪器或状态估计。

## 第二步：选择装甲板

先选装甲板，再解弹道。可以先用距离除以弹速粗估飞行时间，预测在
“观测年龄 + 发射延迟 + 粗估飞行时间”之后的四块装甲板姿态，
比较外法线与指向炮口的方向，选择正面朝向炮口的装甲板。当前最近的装甲板到命中时可能已经转到背面。

在第四步的迭代中固定选中的装甲板索引，避免为每个候选都完整求解弹道，
也避免在迭代中反复更换装甲板。
初选只是近似，发射前仍需检查最终朝向；所选装甲板不可达时不开火。

装甲板顶部向内倾斜 15°，请使用完整的三维法线。命中必须从正面进入
0.135 × 0.056 m 的计分矩形。外壳比计分区域更大，击中外壳不算命中。
`Plate::point` 以装甲板的右、上和外法线方向为坐标轴，把装甲板坐标转换为世界坐标。

## 第三步：求解弹道

世界坐标为 +X 向前、+Y 向左、+Z 向上。yaw 从 +X 向 +Y 为正，pitch 向上为正。
角度用弧度、距离用米、时间用秒。

设炮口位置为 `m`、弹速为 `v`、yaw 为 `ψ`、pitch 为 `φ`、重力加速度为 `g`。
弹丸在**发射后**经过 `t` 的位置为：

$$
\begin{aligned}
x(t) &= m_x + v\cos\phi\cos\psi\;t,\\
y(t) &= m_y + v\cos\phi\sin\psi\;t,\\
z(t) &= m_z + v\sin\phi\;t - \tfrac12 gt^2.
\end{aligned}
$$

水平速度不变，竖直速度以 `g` 的速率减小。在射击所在的竖直平面内，以水平距离 `r` 表示：

$$
z(r) = m_z + r\tan\phi - \frac{g r^2}{2v^2\cos^2\phi}.
$$

这是一条抛物线，重力为零时退化为直线。仿真没有空气阻力或云台转动耗时，不需要更复杂的弹道模型。

`Trajectory` 包含 `yaw_rad`、`pitch_rad` 和 `flight_time_s`。
`Trajectory::solve(point_m)` 按已知弹速求解到**固定世界坐标点的低弹道**。
目标不可达、坐标非有限值或飞行时间超过 `MAX_FLIGHT_S` 时返回 `None`。
它不预测运动，也不添加延迟。

```rust
let trajectory = Trajectory::solve(aim_point_m)?;
debug.trajectory(trajectory, Color::Orange);
let impact_m = trajectory.position(trajectory.flight_time_s);

Some(Command {
    yaw_rad: trajectory.yaw_rad,
    pitch_rad: trajectory.pitch_rad,
    fire: true, // Decide whether this is a good shot before setting this.
})
```

`position(t)` 和 `velocity(t)` 中的时间都从发射开始计。飞行时间不是观测年龄，也不是绝对命中时刻。

## 第四步：瞄准未来位置

弹道工具只解算到固定点，但选中的装甲板仍在运动。用观测年龄、发射延迟和飞行时间预测命中时的姿态。
未来位置取决于飞行时间，而飞行时间又取决于未来位置，可以用不动点迭代处理：

1. 用距离除以弹速得到初始飞行时间。
2. 预测“观测年龄 + 发射延迟 + 估计飞行时间”后的目标状态。
3. 取预测目标上选中装甲板的位置，保持第二步选定的索引。
4. 求解到该位置的弹道，用新飞行时间更新估计。
5. 变化足够小时停止，并设置最大迭代次数；失败时不开火。

从观测采集时刻算起的预测间隔是：

$$
\Delta t = (t_{\mathrm{now}} - t_{\mathrm{capture}})
           + t_{\mathrm{launch\ delay}} + t_{\mathrm{flight}}.
$$

绝对命中时刻是 `now + launch_delay + flight_time`，计算它时不要再加观测年龄。
`Target::at(dt)` 预测匀速平移和自旋，`Target::plates(size)` 返回该时刻的四块装甲板姿态。
索引随目标一起转动；只预测机器人中心的运动，会漏掉自旋带来的装甲板位移。

用秒作为收敛容差的单位，并限制迭代次数。收敛后弹道终点应接近预测装甲板中心，可以同时画出两者来检查。

还要考虑预计外法线和弹丸相对装甲板的入射速度。
`trajectory.velocity(trajectory.flight_time_s)` 给出命中时弹丸速度，
`Plate::velocity_m_s` 同时包含平移和旋转速度。可以参考下面的朝向评分：

$$
s = -\frac{\mathbf n\cdot(\mathbf v_{\mathrm{projectile}}-
                         \mathbf v_{\mathrm{plate}})}
               {\lVert\mathbf v_{\mathrm{projectile}}-\mathbf v_{\mathrm{plate}}\rVert}.
$$

较大的正值表示更接近正面入射，负值表示从背面接近。排除不可达目标，入射角很偏时可以选择不开火。
开火阈值决定射击频率和对预测误差的敏感程度。

## 调试

在自己的 `aim` 中调用绘图接口。绘图只持续一个 tick，并与产生它的指令一起存入 MCAP：

```rust
use aiming_sim::ui::{Color, Gizmos};

debug.plate(plate, input.size, Color::Cyan);
debug.plate(predicted_plate, input.size, Color::Green);
debug.trajectory(trajectory, Color::Orange);
debug.line(from_m, to_m, Color::Purple);
debug.point("Impact", impact_m, Color::Green);
debug.overlay("Flight", format!("{:.0} ms", trajectory.flight_time_s * 1000.0));
```

`debug.trajectory()` 自动采样炮口到命中时刻的抛物线，不需要自己写绘图循环。
`debug.plate()` 画计分矩形和外法线，不添加标签；需要文字标记时使用 `debug.point()`。
工具会忽略非有限坐标并限制绘图数量。

左键拖动环绕视角；右键、中键或 Shift+左键拖动平移；滚轮缩放。
底部工具栏支持播放/暂停、单步、重置、时间轴和显示/隐藏叠加信息。
Tools 包含场景、绘图图层、速度、相机、初始状态、射击检查和 MCAP 保存/加载。
Inspect shot 会冻结该次射击的原始指令和绘图，便于逐步检查飞行过程；
Current tick 返回当前 tick 的诊断信息。

## 录制、回放与导出

```sh
cargo run --no-default-features --locked -- --offline --record target/run.mcap
cargo run --locked -- --replay target/run.mcap --at 3.65
```

录制包含 `/sim/config` 和 `/sim/frame`，保存精确输入、指令、状态、弹丸、计数和绘图。
回放直接恢复这些内容，不重新运行你的代码。新录制不会覆盖已有文件。
仿真时间以固定 10 ms 步长推进，不受界面播放速度影响。

PNG 和 MP4 使用同一个 GPU 场景渲染器。MP4 需要 `ffmpeg`。
导出视角跟随目标，视频为 3840 × 2160、60 FPS，不包含原生界面的文字标签、叠加信息和控件。

```sh
cargo run --locked -- --replay target/run.mcap --at 3.65 --preview target/preview.png
cargo run --locked -- --replay target/run.mcap --export-video target/run.mp4
```

