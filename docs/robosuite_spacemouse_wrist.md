# robosuite SpaceMouse 主线技术说明

[README](../README.md) 给出当前命令和状态。本页说明当前 robosuite 主线的边界，避免把冻结的 PyBullet 参考代码当成入口。

## 控制边界

三个入口都把 SpaceMouse 映射后的六维物理增量交给世界坐标系 `OSC_POSE`：`Tx Ty Tz Rx Ry Rz`。死区、比例、方向、平滑、限幅和超时都由 [spacemouse_wrist.json](../configs/spacemouse_wrist.json) 统一定义。

| 模型 | 总动作维度 | 机械臂切片 | 手部或夹爪切片 |
| --- | ---: | --- | --- |
| Panda + Allegro | 22 | `[0:6]` | `[6:22]`，16 个绝对关节目标。 |
| Panda + PandaGripper | 7 | `[0:6]` | `[6:7]`，一个策略维度驱动两个指头。 |
| Jaco + JacoThreeFingerGripper | 7 | `[0:6]` | `[6:7]`，一个策略维度同步三个手指。 |

Panda + Allegro 入口的 `OSC_POSE` 在 robosuite 默认 Panda 配置上加了 `uncouple_pos_ori=False`（[allegro_model.py](../src/wrist_teleop/allegro_model.py)），LIBERO 入口同样设置（[libero_spatial_teleop.py](../src/wrist_teleop/libero_spatial_teleop.py) 的 `_controller`）：用完整的 6 维任务空间惯量，而不是位置、姿态两块分开算。分开算时，纯旋转会把手横向推开（2026-11-05 真人会话回放：中位数 28 mm、最多 101 mm），松手后 2 秒内最多还会爬 20 mm / 2.7°；改成整体算后同样的输入位置偏移中位数 1.7 mm、最多 8.5 mm，松手后 ≤ 0.5 mm / 0.07°。只改了这一个控制器参数，增益、阻尼、仿真频率不变。

动作切片每次运行都从本地 robosuite composite controller 查询；新夹爪入口不复制 Allegro 的索引。HID 无效时机械臂和夹爪均输出零动作。`R` 清空状态并重建环境；`Q`、`Ctrl+C` 先发送最后一个零机械臂动作，再关闭资源。

## 同步三视图和激光

[visualization.py](../src/wrist_teleop/visualization.py) 在一个已经步进的 MuJoCo 状态上连续离屏渲染：`frontview`、`robot0_eye_in_hand`、`sideview`。它们合成为一个窗口，不会启动三个控制循环。离屏缓冲在这里统一翻转为正常的窗口坐标；首帧 PNG 写到每个运行目录。

固定主视图与侧视图的位置、四元数和视场角写在 `tri_view` 配置中。腕部相机保留附着在 wrist 的性质，只配置视场角。关闭合成窗口会通过同一键盘路由触发安全退出。

`LaserPointer` 从真实 `grip_site` 读取当前位置和旋转，用局部 `+Z` 得到世界方向。它用 `mj_ray` 跳过机器人自己的几何体，命中桌面或场景后把既有透明 cylinder site 画成红线。运行时只修改该 site 的可视字段，关闭时精确恢复；测试比较了同种子动作下有无激光的 `qpos`、`qvel` 和接触数。

## Panda + Allegro 手部输入

[hand_input.py](../src/wrist_teleop/hand_input.py) 保留摄像头工作线程，但不允许它写入机械臂动作。按钮手势也只写 16 维 Allegro 目标。

正式模式 `--hand-control dex` 使用 [dex_retargeting.py](../src/wrist_teleop/dex_retargeting.py)：MediaPipe 21 个世界坐标关键点先转成官方 MANO 坐标，再按官方 vector 配置中 `[[0,0,0,0],[4,8,12,16]]` 生成四条腕到指尖向量。官方求解器输出通过关节名称而非位置索引映射到 `joint_0.0` 至 `joint_15.0`，并检查 16 项完整性、有限性与关节限位。

官方配置来自 dex-retargeting 0.4.6，未经语义修改；来源、许可证和 SHA-256 在 [PROVENANCE.md](../configs/dex_retargeting/PROVENANCE.md)。当前 Windows Python 3.10 环境没有官方 `pin>=2.7` 依赖，因而 dex 初始化会明确失败。`legacy-curl` 只能显式选择，不能作为 dex 失败时的自动回退。

进入求解器前有三点（2026-11-05 修正）：摄像头画面默认先水平翻转再给 MediaPipe，翻转后的 3-D 关键点是镜像的（形状是左手、标签仍是 Right），所以 `DexHandInputAdapter(unmirror_world_x=True)` 先把世界坐标 x 取反（与 Shadow 路径的 `ShadowDexHandInput` 相同；`--camera-no-mirror` 时不取反）。人手关键点按原尺寸送入（`reference_landmark_scale` = 1.0）：求解器用的是原尺寸 Allegro URDF，关节角与 MuJoCo 模型的 2/3 缩放无关。默认是绝对目标（`dex_hand.relative_motion: false`），求解器输出直接作为 16 个关节目标；`true` 时改成相对第一帧（或按 `C` 时）的人手姿态，`C` 只在相对模式下起作用。

dex 摄像头短时丢失保持最后一个有效关节目标，默认最长 0.5 秒；超时后 hand gesture resolver 使用安全张手。`R` 会重置求解器和官方滤波器对象。

## 计时与节拍（Panda + Allegro integrated）

仿真频率仍是 `control_hz`（20 Hz），物理参数不变。[loop_timing.py](../src/wrist_teleop/loop_timing.py) 提供四样东西：

- `FixedRateClock`：主循环按固定网格 `next_tick += period` 等待，不用"本轮开始 + 周期"，所以每轮工作时长和 Windows 15.6 ms 睡眠取整都不会累积成漂移。运行期间用 `winmm.timeBeginPeriod(1)` 申请 1 ms 定时器精度，退出时释放；最后约 1.5 ms 用 `sleep(0)` 让出。某一轮超时不满一个周期时下一拍立即补上、仍回到原网格；超时一个周期以上（重置、GC 等）时丢掉错过的拍、从当前时刻重新起网格，不连发补帧，`overruns_dropped_ticks` 记录次数。`R` 重置后网格重新起算。
- `LatestSlot` + 取帧线程：摄像头只有一个 `VideoCapture`，只由取帧线程读取，每帧覆盖上一帧；处理线程（MediaPipe → 适配器 → dex 子进程）每次只拿最新一帧，处理慢时旧帧被丢弃而不是在驱动缓冲里排队，`frames_dropped_latest_only` 记录丢弃数。主循环照旧只读适配器的最新结果，不等待摄像头。
- `FixedDelayResampler`（固定延迟取样）：只作用于摄像头给出的手部关节目标（Allegro 16 个；LIBERO SpaceMouse 入口里 Shadow 20 个），在 `gestures.update` 之前。摄像头线程把每一帧被适配器接受的结果连同这一帧的取图时刻（`perf_counter`）存下来（`CameraHandWorker.samples_since`），主循环每拍取"本拍时刻 − `hand_target_delay_s`"那一刻的手：用这一刻前后两帧按取图时刻线性插值。于是每拍看到的都是同样延迟之前的手，均匀的拍看到均匀间隔的手。下一帧还没到（摄像头晚了）时保持最新一帧，不外推，记为 `held_newest`；第一帧不够老之前用最早一帧（`before_oldest`）。输出永远是相邻两帧摄像头结果的混合，不混入任何其他来源；有效 / 保持 / 安全张手的判断不延迟，照旧由手势层决定；机械臂 `[0:6]` 不经过它。`R` 重置和 dex 的 `C` 重新取中立位时清空，并丢掉重置前读到的帧。延迟默认 65 ms（`configs/spacemouse_wrist.json` 的 `hand_target_delay_s`，命令行 `--hand-delay-ms` 可改，0 = 直接用最新一帧）。每个 step 的 `camera.resampling` 记录当拍状态和"最新一帧领先查询时刻多少毫秒"，`timing_summary.hand_resampling` 记录各状态累计拍数，控制台计时行显示保持最新帧的拍数比例。
- `StageProfiler`：每个 step 事件带 `timing_ms`（SpaceMouse 读取、手部结果年龄、env.step、渲染、日志等），每 2 秒和退出时各记一条 `timing_summary`（主循环、时钟、摄像头线程、dex 往返的频率 / 延迟 / p99 抖动）。摄像头线程的 `result_latency` 是一帧从取图到结果可用的时间，固定延迟至少要盖住"一个帧间隔 + 它"。

## 抓取遥测

[grasp_telemetry.py](../src/wrist_teleop/grasp_telemetry.py) 是三个入口共用的只读观察模块。它读取 `cube_main`、`table_collision` 和当前接触：方块抬升至少 0.03 米、脱离桌面、保持手部接触至少 3 秒且接触穿透不超过 4 毫米时记录通过。它不控制方块，也不替代 Gate 2 的静态抓取验证。

## 依赖边界

当前主线锁定 robosuite 1.5.2、MuJoCo 3.3.7、NumPy 1.26.4、hidapi 0.15.0、OpenCV 4.11.0.86 和 MediaPipe 0.10.21。`dex-retargeting==0.4.6` 作为带 Windows 平台标记的可选官方依赖写入 [requirements-robosuite-wrist.lock](../requirements-robosuite-wrist.lock)，不会在本机安装时破坏已验证的 NumPy / MediaPipe 组合。
