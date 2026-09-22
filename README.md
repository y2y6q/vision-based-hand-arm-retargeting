# 手臂与灵巧手遥操作

本项目在 Windows 上使用 robosuite 和 MuJoCo 运行 Panda、Allegro、Panda 平行夹爪和 Jaco 三指夹爪。SpaceMouse 六轴只控制机械臂末端；手部、夹爪、摄像头和按钮不会改写机械臂六维动作。

旧 PyBullet / panda-gym 目录是冻结参考基线，不是当前入口。

## 当前状态

| 项目 | 状态 | 已有证据 |
| --- | --- | --- |
| Panda + Allegro、相机坐标映射后的世界坐标 OSC 控制 | 已验证 | 22 维动作、无界面烟雾测试、三视图烟雾测试；机械臂 OSC 仍接收世界坐标增量。 |
| TCP 激光和同步三视图 | 已验证 | 三个离屏相机同一仿真时间渲染；1920×1080 合成图由原生渲染画面组成；激光只修改无碰撞的可视 site。 |
| Panda + PandaGripper | 已验证 | 实际 7 维动作、开合、无界面和三视图烟雾测试。 |
| Jaco + JacoThreeFingerGripper | 已验证 | 实际 7 维动作、三指整体开合、无界面和三视图烟雾测试。 |
| Panda + Shadow Hand | 仅预检 | 第四入口只能写本地模型和 robosuite 工厂审计；当前拒绝启动，不会把 Allegro 冒充为 Shadow Hand。 |
| Gate 2 固定 Panda 静态抓取 | 已验证 | 锁定配置的独立回归继续通过。 |
| Panda + Allegro 实时摄像头状态窗口 | 已人工验证 | 用户于 2026-09-17 确认三视图与真人摄像头窗口同时显示，且摄像头已显示手部关键点；该确认不代表 native dex solver 已运行。 |
| SpaceMouse 中位安全零 | 已验证 | 自动记录证明 stale/无输入路径严格为零；用户于 2026-09-17 人工确认静止漂移可接受。 |
| 官方 dex-retargeting solver 与 Panda + Allegro 动作链路 | 已验证 | 隔离 sidecar 实际加载官方 `dex-retargeting==0.4.6`、构建 `SeqRetargeting`、执行 `retarget/reset`，并将有限的 16 关节目标写入真实 action `[6:22]`。 |
| 真实 SpaceMouse 方向、摄像头、人手动作和人工抓取 | 待人工验证 | 本轮自动化使用合成输入，不能替代硬件方向、相机画面或人工抓取验证。 |

## 当前任务进度 / Current task progress — 2026-09-22

| 内容 / Item | 状态 / Status | 已有证据 / Evidence |
| --- | --- | --- |
| 官方 dex solver / Official dex solver | **已验证 / Verified** | 隔离 sidecar 实际加载官方 `dex-retargeting==0.4.6` wheel，构建官方 Allegro vector solver，并实际执行 `retarget()` 与 `reset()`。wheel SHA-256 为 `cf08b93e…621bb62f7`，配置 SHA-256 为 `23cdb538…555a5284`。<br>The isolated sidecar loads the official wheel, builds the official Allegro vector solver, and executes `retarget()` and `reset()`. |
| Panda + Allegro 动作接入 / Panda + Allegro action integration | **已验证 / Verified** | 实际环境为 22 维 action：Panda OSC 是 `[0:6]`，Allegro 16 关节是 `[6:22]`。真实 solver 输出按 joint name 重排、限位后仅写入 hand slice。<br>The live action contract is 22-D; named, bounded solver output writes only the 16-D hand slice. |
| 默认 dex 启动 / Default dex startup | **已验证 / Verified** | `integrated --headless --dex-smoke --duration 1` 自动选择 `runtime=sidecar` 并安全退出；本地日志位于 `outputs/spacemouse_wrist/20260922T022502Z_integrated`。<br>The smoke command automatically selected the sidecar and exited cleanly; its local log is excluded from the public repository. |
| 回归测试 / Regression tests | **已验证 / Verified** | worker 协议、真实 solver、真实 Panda step、slice 隔离、R/Q/Ctrl+C 清理共 23 项定向测试通过。<br>Twenty-three focused tests passed, including worker protocol, real solver, Panda step, slice isolation, and shutdown. |
| 真实硬件输入 / Live hardware input | **待人工验证 / Manual verification required** | 最近一次 smoke 未访问到 SpaceMouse HID，摄像头也没有收到帧；自动化不能代替真人手、真实六轴输入或抓取验证。<br>The latest smoke test did not access SpaceMouse HID or receive camera frames; automated tests cannot replace live operation. |
| 冻结旧基线 / Frozen legacy baseline | **有历史差异 / Existing mismatch** | 哈希检查为 172/173；不一致项是旧的 `src/teleop/teleop_factory.py`，不属于本轮 dex 改动。报告保存在本地 `outputs/frozen_baseline_hash_comparison_current.json`。<br>The hash audit reports 172/173 matches; the old `teleop_factory.py` mismatch is outside this dex change. |

### 待人工验证 / Manual verification required

1. **启动真实交互 / Start the live session**：运行下方 Panda + Allegro 默认命令。确认终端显示 `hand mode=dex`、`runtime=sidecar` 与 `dex-retargeting=0.4.6`。<br>Run the default Panda + Allegro command below and confirm those three startup messages.
2. **摄像头与关键点 / Camera and landmarks**：确认真人画面、右手关键点、`DEX ACTIVE` 或 dex 更新状态正常显示。右手活动时，Allegro 16 关节应更新；摄像头输入不得改变 Panda `[0:6]`。<br>Verify the live image, right-hand landmarks, and dex update state. Hand motion must update Allegro only, never Panda's `[0:6]` slice.
3. **SpaceMouse 隔离 / SpaceMouse isolation**：确认六轴只移动 Panda wrist；左右按钮只切换 Allegro 手势，不能改变 arm slice。确认松手后 wrist 命令归零且无持续漂移。<br>Verify that six axes move only the Panda wrist, buttons change only Allegro gestures, and neutral input produces zero wrist command without drift.
4. **安全操作 / Safety controls**：在仿真窗口或摄像头窗口分别验证 `R` 完整重置、`Q` 安全退出，以及 `Ctrl+C` 清理 viewer、HID、camera 和 dex sidecar。<br>Verify `R`, `Q`, and `Ctrl+C` from both windows, including cleanup of viewer, HID, camera, and the dex sidecar.
5. **运行时风险 / Runtime risk**：当前 sidecar 仅在子进程使用 `KMP_DUPLICATE_LIB_OK=TRUE` 处理 Conda 的 OpenMP DLL 冲突。确认长时间真人运行稳定后，再考虑消除该环境级兼容设置。<br>The sidecar alone uses `KMP_DUPLICATE_LIB_OK=TRUE` for the current Conda OpenMP conflict. Confirm long live-session stability before removing this compatibility setting.

当前解释器为 `E:\hand_arm_retargeting\.venv-robosuite\Scripts\python.exe`。实际版本见 [依赖锁定文件](requirements-robosuite-wrist.lock)。

## 三个程序

| 程序 | 机器人和夹爪 | 动作布局 | 输入 |
| --- | --- | --- | --- |
| [run_spacemouse_panda_wrist_teleop.py](scripts/run_spacemouse_panda_wrist_teleop.py) | Panda + Allegro 右手 | `[0:6]` 机械臂，`[6:22]` Allegro 16 关节 | SpaceMouse、摄像头、按钮手势。 |
| [run_spacemouse_panda_jaw_gripper_teleop.py](scripts/run_spacemouse_panda_jaw_gripper_teleop.py) | Panda + PandaGripper | `[0:6]` 机械臂，`[6:7]` 平行夹爪 | SpaceMouse；HID 位 0 闭合，位 1 张开。 |
| [run_spacemouse_jaco_three_finger_teleop.py](scripts/run_spacemouse_jaco_three_finger_teleop.py) | Jaco + JacoThreeFingerGripper | `[0:6]` 机械臂，`[6:7]` 三指整体开合 | SpaceMouse；按住 HID 位 0 连续闭合，按住位 1 连续张开。 |
| [run_spacemouse_panda_shadow_hand_teleop.py](scripts/run_spacemouse_panda_shadow_hand_teleop.py) | Panda + Shadow Hand | 无运行时动作布局 | 仅支持 `--audit` 预检；普通启动会明确拒绝。 |

本机实际动作范围均为 `[-1, 1]`。三套控制器最终都向 `OSC_POSE` 发送世界坐标增量 `Tx Ty Tz Rx Ry Rz`。输入先按主相机实时外参的基向量 `[屏幕右方, 观察深度, 世界上方]` 转为世界坐标；这是 `control_frame=camera`，不是对某一台机器人的固定轴交换。Panda 平行夹爪的一个策略维度驱动两个物理手指，Jaco 的一个策略维度同步驱动三个物理手指。

## 画面、激光和抓取记录

默认界面把肩后方主视图放在主画面，把 `robot0_eye_in_hand` 和 `sideview` 放在两个小画面。合成窗口为原生 `1920×1080`：主画面 `1530×1080`，两个辅助画面各 `380×535`。三个画面都在同一个 `env.step()` 后渲染，不从小图重复放大；关闭合成窗口会安全退出。固定相机位置、姿态、视场角、分辨率和刷新率都在 [configs/spacemouse_wrist.json](configs/spacemouse_wrist.json) 的 `tri_view` 中。

Panda + Allegro 默认还会打开一个实时摄像头状态窗口。它复用同一套 camera capture 和检测线程，显示彩色图、手部关键点和连线、检测到的左右手、`dex` / `legacy-curl` / `safe-open` / 按钮手势模式、检测状态、帧率及 dex 是否更新。三视图窗口和摄像头窗口中的 `R`、`Q` 都生效；关闭任一窗口会触发安全退出。`--headless` 不创建窗口，`--no-camera-display` 只关闭摄像头状态窗口。

Panda + Allegro 的激光从真实 `palm_laser_site` 出发，始终沿世界坐标 `-Z` 垂直向下。程序跳过机器人自身几何体后寻找桌面或场景命中点，并复用透明的 `grip_site_cylinder` 绘制红线。它不创建质量、碰撞、接触、执行器或约束，也不写入 `qpos`、`qvel` 或动作。Panda 与 Jaco 入口继续使用各自的激光起点和方向，但共用同一可视化组件。

三个程序都会把方块高度、桌面接触、手部接触、穿透、无支撑保持时间和结果写入日志。互动抓取记录的通过条件为：方块离初始桌面高度至少 0.03 米、离桌面且仍接触手至少 3 秒、方块接触穿透不超过 4 毫米。它只是遥操作证据，不是自动抓取器；Gate 2 仍使用它自己的锁定静态抓取协议。

## SpaceMouse 与按键

六轴先经过启动中位偏置、进入/退出死区、轴顺序、反向、指数平滑、比例和限幅，再按主相机坐标映射写入机械臂切片。启动时只在手柄保持中位的前 8 帧估计零偏，之后不会在正常移动中学习零偏。进入死区为平移/旋转 `0.08`，退出滞回为 `0.06`；连续中位 3 帧会清空 EMA 并锁定精确零。过期或断开输入也输出精确零机械臂增量。其余主要参数为：每步平移 `0.003 m`、每步旋转 `0.025 rad`、最大平移 `0.004 m`、最大旋转 `0.040 rad`、20 Hz 控制和 `0.35` 平滑系数。

当前 30 秒无人值守记录没有收到 SpaceMouse 运动报告，因此只能证明 stale/无输入路径的零动作稳定性：最终 OSC 六维动作严格为零，EE 位移 `1.55e-10 m`，旋转 `4.68e-6°`。真实手柄居中噪声、物理推拉方向和三个旋转轴仍需人工诊断确认。

| 操作 | Panda + Allegro | Panda / Jaco 夹爪 |
| --- | --- | --- |
| SpaceMouse 左键，HID 位 0 | 上升沿切换拇指—食指捏合 | Panda：闭合；Jaco：按住时连续闭合 |
| SpaceMouse 右键，HID 位 1 | 上升沿切换四指捏合 | Panda：张开；Jaco：按住时连续张开 |
| `B` | 将当前交互 Lift 方块单次放大到默认尺寸的 1.5 倍 | 无作用 |
| `R` | 重置环境、OSC、输入状态、手势、摄像头缓存和 dex 历史 | 重置环境、OSC、输入状态和夹爪状态 |
| `Q` 或 `Ctrl+C` | 发送最后一个零机械臂动作后安全退出 | 同左 |

Panda + Allegro 的手势按钮只在上升沿生效。`B/b` 只在 Panda + Allegro integrated 中生效：每轮环境只放大一次交互方块，`R/r` 会恢复默认尺寸并完整重置。Jaco 在按住期间按实际 `dt` 连续变化，松开立即发送零夹爪速度；两个按钮同时按下时保持不动并记日志。按钮不会改变机械臂切片。HID 位与设备实体左右键的对应仍需通过诊断程序人工确认。

## dex-retargeting

Panda + Allegro 的正式默认手指模式是 `--hand-control dex`：

`MediaPipe 21 个三维世界关键点 → 官方坐标转换 → 官方 Allegro vector 配置 → 16 个按名称映射的 MuJoCo 关节目标`

官方配置副本位于 [configs/dex_retargeting/allegro_hand_right.yml](configs/dex_retargeting/allegro_hand_right.yml)，来源、许可证和 SHA-256 位于 [PROVENANCE.md](configs/dex_retargeting/PROVENANCE.md)。它使用右手、腕部到拇指/食指/中指/无名指四条向量；小指不进入这份官方 Allegro 配置。

两个手同时出现时，dex 会按检测到的右手索引选择同一条世界关键点，避免误把第 0 条手部数据送给右手链路。主 robosuite 进程不导入 Pinocchio 或 dex 原生库；默认 `dex_hand.runtime=auto` 使用隔离 sidecar。sidecar 加载项目内保存的官方 [0.4.6 wheel](third_party/dex_retargeting_0_4_6/dex_retargeting-0.4.6-py3-none-any.whl)，会核对 wheel 和配置 SHA-256，并把 solver 输出按名称重排到 Allegro 的 16 个 MuJoCo 关节。当前机器默认会发现同级 Conda 环境中的 `dexretarget\python.exe`；其他位置可设置 `DEX_RETARGETING_PYTHON`。程序不能使用官方 runtime 时会明确退出，绝不会静默回退到规则式映射。需要旧映射时必须显式传入 `--hand-control legacy-curl`；不控制手指时传入 `--hand-control off`。摄像头瞬时丢失时，dex 目标最多保持 0.5 秒，超时后手势层进入安全张手。

当前 Conda 环境同时加载 Pinocchio 与 Torch 时存在 OpenMP DLL 冲突。程序只给 sidecar 子进程设置 `KMP_DUPLICATE_LIB_OK=TRUE`，不会影响 MuJoCo 主进程；这仍是需要后续消除的运行时风险。真实摄像头帧、真人手部动作与实际抓取尚未在本轮 dex runtime 下验证。

## 常用命令

完整命令、环境同步、三条烟雾测试、Gate 2 和冻结基线检查见 [Run Instructions.txt](Run%20Instructions.txt)。

最常用的交互启动命令如下：

```powershell
Set-Location "E:\hand_arm_retargeting"
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_wrist_teleop.py integrated
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_wrist_teleop.py integrated --headless --dex-smoke --duration 2
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_wrist_teleop.py integrated --hand-control legacy-curl
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_jaw_gripper_teleop.py
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_jaco_three_finger_teleop.py
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_shadow_hand_teleop.py --audit
```

第一条是完整交互入口，默认启动官方 dex sidecar。第二条是不需要 SpaceMouse 的 dex 无界面 smoke。第三条是显式旧映射入口。最后一条只写 Shadow Hand 可用性审计；去掉 `--audit` 的普通启动会明确拒绝，不会启动仿真或替换为 Allegro。

## 代码和日志

| 位置 | 作用 |
| --- | --- |
| [src/wrist_teleop/visualization.py](src/wrist_teleop/visualization.py) | 共享三视图、截图和非物理激光。 |
| [src/wrist_teleop/hand_input.py](src/wrist_teleop/hand_input.py) | Panda + Allegro 的唯一摄像头采集、检测、覆盖层和输入状态。 |
| [src/wrist_teleop/gripper_teleop.py](src/wrist_teleop/gripper_teleop.py) | Panda / Jaco 单自由度夹爪的共享控制循环。 |
| [src/wrist_teleop/grasp_telemetry.py](src/wrist_teleop/grasp_telemetry.py) | 三条入口共用的只读抓取遥测。 |
| [src/wrist_teleop/dex_retargeting.py](src/wrist_teleop/dex_retargeting.py) | 官方 dex 配置、名称映射、隔离 sidecar client 与运行时完整性检查。 |
| [scripts/dex_retargeting_worker.py](scripts/dex_retargeting_worker.py) | 独立解释器中的官方 dex 0.4.6 solver；只使用 JSON 行协议。 |
| [third_party/dex_retargeting_0_4_6](third_party/dex_retargeting_0_4_6) | 已核验 SHA-256 的官方 wheel 及来源说明。 |
| [configs/spacemouse_wrist.json](configs/spacemouse_wrist.json) | 控制、相机、激光、dex 丢帧和抓取遥测的唯一配置来源。 |
| `outputs/spacemouse_wrist/<时间戳>_integrated` | Panda + Allegro 的元数据、逐步日志和三视图截图。 |
| `outputs/gripper_teleop/<时间戳>_*` | Panda / Jaco 夹爪的元数据、逐步日志和三视图截图。 |
| `outputs/weekly_grasp_ux/<时间戳>` | 本轮修改前后基线、测试、Gate 2 和哈希证据。当前中位记录在 `20260917T034805Z_ux_fixes`。 |

更细的 API、动作边界和配置说明见 [robosuite_spacemouse_wrist.md](docs/robosuite_spacemouse_wrist.md)。

## 回退

本轮修改前文件副本和 SHA-256 清单位于 `outputs/weekly_grasp_ux/20260916T153432Z/prechange/source_backup` 与 `source_manifest_sha256.json`。恢复这些文件即可回到修改前；新模块可单独删除。将配置中的 `laser.enabled` 或 `tri_view.enabled` 设为 `false` 可以关闭对应功能；传入 `--hand-control legacy-curl` 可以显式切回旧手指映射。

不要修改 [third_party/panda-gym](third_party/panda-gym)、[src/teleop](src/teleop) 或旧的 [run_camera_panda_allegro_teleop.py](scripts/run_camera_panda_allegro_teleop.py)。
