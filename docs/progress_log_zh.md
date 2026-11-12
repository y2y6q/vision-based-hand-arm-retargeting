# 开发日志（中文）

本文件保存 2026-10-15 起的中文开发记录，原先附在 README.md 末尾。README.md 现为面向 GitHub 的英文说明；运行命令以 `Run Instructions.txt` 和 README 为准。

## 2026-10-15｜PyCharm 运行步骤（最新）

请在 PyCharm 中运行，不要双击 `.py` 文件，也不要点击顶部当前显示的 `python -m pip install -e .\third_party\panda-gym` 绿色三角；那是安装依赖的旧运行配置，不会启动仿真。

### 第一次使用：设置 PyCharm 解释器

1. 在左侧 **Project** 面板确认已打开 `E:\hand_arm_retargeting`。
2. 选择 **File → Settings → Project: hand_arm_retargeting → Python Interpreter**。
3. 对 Panda、Allegro、四种夹爪、Shadow、Gate 2 和 SpaceMouse diagnostic，选择 Existing environment：`E:\hand_arm_retargeting\.venv-robosuite\Scripts\python.exe`。
4. 只有运行 LIBERO 时，临时切换为：`E:\hand_arm_retargeting\.venv-libero\Scripts\python.exe`。
5. 打开底部 **Terminal**（`Alt+F12`），先运行一次下方的目录命令。之后点击左侧 `scripts` 下对应文件查看代码，并在这个 Terminal 粘贴同一行的启动命令。

~~~powershell
Set-Location "E:\hand_arm_retargeting"
~~~

正常交互需要 SpaceMouse；Panda + Allegro 和 Shadow 的相机手势还需要摄像头。下表中的“点击文件”只用于在 PyCharm 中定位入口；实际启动以 Terminal 命令为准。

| 脚本                                                            | 命令 | 对应实验场景 | 用途 |
|---------------------------------------------------------------| --- | --- | --- |
| `scripts/run_spacemouse_panda_wrist_teleop.py`                | `& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_wrist_teleop.py integrated --record .\outputs\spacemouse_wrist\panda_allegro.mp4` | Panda + Allegro 右手 + Lift 方块 | 主程序。SpaceMouse 控制 Panda 手腕；摄像头默认用与 Shadow 相同的规则映射（`curl-pinch`：手指弯曲 + 拇指捏合对掌）控制 Allegro，`--hand-control dex` 可切回 dex。静止时手掌平行桌面。按 `B` 将方块放大 1.5 倍，`R` 重置，`Q` 退出。 |
| `scripts/run_spacemouse_panda_jaw_gripper_teleop.py`          | `& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_jaw_gripper_teleop.py --record .\outputs\gripper_teleop\panda2.mp4` | Panda + PandaGripper 两指夹爪 + Lift 方块 | 两指夹爪对照实验。左/右 SpaceMouse 按钮控制闭合/张开。 |
| `scripts/run_spacemouse_panda_robotiq85_teleop.py`            | `& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_robotiq85_teleop.py --record .\outputs\gripper_teleop\robotiq85.mp4` | Panda + Robotiq85 两指夹爪 + Lift 方块 | 第二个两指夹爪对照实验。 |
| `scripts/run_spacemouse_jaco_three_finger_teleop.py`          | `& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_jaco_three_finger_teleop.py --record .\outputs\gripper_teleop\jaco3.mp4` | Jaco + JacoThreeFingerGripper 三指夹爪 + Lift 方块 | 三指夹爪对照实验。按住左/右按钮会持续闭合/张开；松开即保持。2026-10-22 起按住约 0.7 s 走完全行程（原 robosuite 每步固定 0.01，全行程约 10 s）；可用 `--gripper-rate` 调整起步斜率。若 Windows HID 按住时不重复发包，已确认的按住状态最多维持 6 秒，随后安全归零。 |
| `scripts/run_spacemouse_panda_robotiq_three_finger_teleop.py` | `& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_robotiq_three_finger_teleop.py --record .\outputs\gripper_teleop\robotiq3.mp4` | Panda + RobotiqThreeFingerGripper 三指夹爪 + Lift 方块 | 长按连续开合；默认以 12 policy-units/s 快速达到合法满速。可用 `--gripper-rate 6` 降低速度；仍保留 6 秒受限按住保护。 |
| `scripts/run_spacemouse_panda_shadow_hand_teleop.py`          | `& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_shadow_hand_teleop.py --record .\outputs\spacemouse_wrist\panda_shadow.mp4` | Panda + Shadow Hand（24 个物理关节、20 个执行器）+ Lift 方块 | Shadow 实际仿真入口。当前使用规则式相机 curl，尚未接入 Shadow 的 dex-retargeting。仅检查模型可用性时加 `--audit`。 |
| `scripts/run_grasp_validation.py`                             | `& .\.venv-robosuite\Scripts\python.exe .\scripts\run_grasp_validation.py --gate 2 --visualize` | 固定 Panda 的 Gate 2 静态抓取 | 回放锁定的 Allegro 静态抓取流程；不是实时 SpaceMouse 实验。 |
| `scripts/run_libero_task_preview.py`                          | `& .\.venv-libero\Scripts\python.exe .\scripts\run_libero_task_preview.py --suite libero_spatial --task-id 0 --record .\outputs\libero\libero_spatial_task0_preview.mp4` | LIBERO Spatial 第 0 个官方场景 | 仅加载场景、初始状态并录制零动作预览。 |
| `scripts/run_libero_spatial_panda_shadow_teleop.py` | `& .\.venv-libero\Scripts\python.exe .\scripts\run_libero_spatial_panda_shadow_teleop.py` | LIBERO Spatial 官方场景 + Panda + Shadow Hand | SpaceMouse 只控制末端；摄像头默认经官方 dex-retargeting **DexPilot** 控制 Shadow（拇指—各指捏合向量，拇指对掌更灵活），`--hand-control legacy-curl` 为旧规则映射。默认机械臂速度 2.5×（`--arm-speed` / `--rot-speed` 可调）。官方初始状态按同名关节迁移。 |
| `scripts/run_libero_spatial_panda_robotiq_three_finger_teleop.py` | `& .\.venv-libero\Scripts\python.exe .\scripts\run_libero_spatial_panda_robotiq_three_finger_teleop.py` | LIBERO Spatial 官方场景 + Panda + Robotiq 三指夹爪 | 复用 SpaceMouse 六维腕部控制（默认 2.5× 速度，`--arm-speed` 可调）；左键长按关闭、右键长按打开，默认 12 policy-units/s。 |

### 用绿色三角启动（可选）

如果不想每次粘贴命令：右键点击上表对应的脚本 → **Modify Run Configuration**。把 **Script path** 指向该脚本，**Working directory** 设为 `E:\hand_arm_retargeting`，把上表命令中脚本名之后的内容填入 **Parameters**。保存后，绿色三角才会启动该实验。PyCharm 通常共用项目解释器：除 LIBERO 外保持项目解释器为 `.venv-robosuite`；运行 LIBERO 前临时切换项目解释器为 `.venv-libero`，完成后再切回。

开始真实 SpaceMouse 测试前，先点开 `scripts/run_spacemouse_panda_wrist_teleop.py`，然后在 Terminal 运行：

~~~powershell
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_wrist_teleop.py diagnostic --duration 10
~~~

确认六轴和两个按钮显示正确后，再启动 Panda + Allegro 主程序。输出视频和日志会写入命令中指定的 `outputs\...` 路径。

### 无硬件时的安全检查

没有连接 SpaceMouse 或摄像头时，使用下列命令检查仿真和录像链路。它们使用合成输入，不能替代真实硬件验证。

~~~powershell
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_jaw_gripper_teleop.py --headless --headless-smoke --duration 1.1 --record .\outputs\gripper_teleop\panda2_smoke.mp4
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_robotiq85_teleop.py --headless --headless-smoke --duration 1.2 --record .\outputs\gripper_teleop\robotiq85_smoke.mp4
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_jaco_three_finger_teleop.py --headless --headless-smoke --duration 1.1 --record .\outputs\gripper_teleop\jaco3_smoke.mp4
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_robotiq_three_finger_teleop.py --headless --headless-smoke --duration 1.2 --record .\outputs\gripper_teleop\robotiq3_smoke.mp4
~~~

录像写入 `outputs/gripper_teleop/`，Panda + Allegro 遥测写入 `outputs/spacemouse_wrist/`，LIBERO 预览写入 `outputs/libero/`。在恢复 SpaceMouse 与摄像头之前，以上正常交互命令应只视为待人工硬件验证，不能把仿真 smoke 当成真实遥操作成功。

## 2026-10-15｜

实际完成的内容。

- 主程序 Panda + Allegro 现在默认使用 dex-retargeting。SpaceMouse 只控制 Panda 手腕，摄像头手部输入只控制 Allegro 手指；两条控制不会互相覆盖。
- `B` 可以把 Lift 方块放大到 1.5 倍。系统会同步更新方块大小、质量、惯量和离桌高度，不会把大方块直接塞进桌面。
- Allegro 的红色激光现在从实体掌心中心沿掌心局部 `+Z` 工作法向射出，会随手腕转动。默认桌面姿态下它朝向桌面；右上掌心第一视角位于掌心外 25 mm，与该射线同轴，并显示红色十字准星。该视图只临时隐藏机器人自身的渲染几何和实体圆柱，避免手指、手腕或圆柱遮住方块；它不改变物理、碰撞或控制。
- Panda + Allegro、PandaGripper、Robotiq85、Jaco 三指、Robotiq 三指和 Shadow 都可以录制 MP4。三指夹爪改为按住按钮持续开合，闭合速度已提高。
- Shadow Hand 已能在真实 MuJoCo 场景运行，包含 20 个手部执行器；它暂时使用规则式手指 curl，尚未接入 Shadow 的 dex-retargeting。
- Gate 2 固定 Panda 静态抓取在本机重新跑过：单次 1/1、qualification 3/3、formal 10/10 通过。该结果说明锁定的仿真抓取稳定，但不代表真人遥操作已经验证。
- LIBERO 的 Spatial、Object、Goal、LIBERO-10 和 LIBERO-90 五类官方场景都能加载并录制预览视频。Spatial 现在新增 Panda + Shadow、Panda + Robotiq 三指两条实时控制入口：两者均实测官方第 0 场景 reset、官方状态的同名关节迁移、action step、安全零动作和三视图渲染。它们尚未完成真实 SpaceMouse / 摄像头人工验证，也没有把“进入场景”表述为“完成官方任务”。
- 摄像头窗口的 OpenCV 显示已移到与三视图相同的主线程，避免 Windows 上两个 GUI 线程互相导致黑屏。摄像头送入 MediaPipe 前会按配置镜像；只有检测为 `Right` 的帧才进入 Allegro 的 dex 流程，实际镜像方向仍需用真人右手确认。
- SpaceMouse 的平移和旋转 HID 分包现分别超时；新平移包不能继续重放旧旋转值。该规则同时用于 Allegro 和 Shadow。

还需要人工完成的事情只有真实设备相关部分：确认 SpaceMouse 六轴方向和左右按钮，确认摄像头关键点与真人手势能驱动 Allegro，分别录制四种夹爪的真实抓取视频。夹爪碰到桌面后不会自动移动机械臂重新规划抓取；测试时应先让手指和方块离桌面留出余量。

完整的测试结果、画面证据和明日测试顺序见 `outputs/acceptance_audit/20261014_screenshot_task_readiness.md`。

### 需要人工验证

1. 启动 Panda + Allegro 后，确认红线从掌心侧射出；右上掌心第一视角中央应有红色十字准星，且它指向红线的落点。
2. 用 SpaceMouse 确认六轴、两个按钮、`R` 和 `Q` 的实际行为；松手后 Panda 必须停止。重点检查 Shadow：静止时不应再有周期性扭转或抽搐。
3. 确认摄像头窗口显示关键点和 `detected hand: Right`，再确认 `DEX ACTIVE: retarget() updated` 出现；物理右手运动只改变 Allegro 手指。程序默认镜像普通 Windows 摄像头帧；若驱动已经镜像画面，再加 `--camera-no-mirror`，不能把 `Left` 的被拒绝帧当 dex 成功。

## 2026-10-19｜代码一致性检查与最新进度

本次检查覆盖四个实时入口、共享运行时、Allegro/Shadow 模型、dex sidecar、三视图和 README 指令。已修正唯一发现的中断遗留冲突：主程序的激光注释仍写旧的局部 `-X`，现已与真实配置统一为掌心局部 `+Z`。未发现半改名模块、断开的导入、两套相互竞争的 Allegro 控制路径，或由本轮中断造成的控制切片混用。

- 【已验证】Allegro 运行时模型等比缩小为原尺寸的 `2/3`；外观、碰撞、质量、惯量、掌心 TCP 和掌心视图同步缩放。Panda 动作仍为 `[0:6]`，Allegro 动作为 `[6:22]`。
- 【已验证】默认 Allegro 路径启动官方 dex-retargeting 0.4.6 sidecar；相机关键点先转到与官方检测器一致的 wrist frame，再转成按名字映射的 16 个 Allegro 目标。由于 MuJoCo Allegro 已缩小为 `2/3`，送入未改动的官方求解器前会将 wrist-relative 人手参考点同步缩放为 `0.6667`，避免大手模型的几何目标直接驱动小手。合成开手/弯手输入使 14 个关节输出发生变化；`legacy-curl` 只能显式选择。
- 【已验证】最新真实摄像头短测中，识别到物理 `Right` 手的 282 个控制周期里有 210 个有效 dex 更新，15 个关节的目标变化超过 `0.05 rad`，最大单关节范围为 `1.806 rad`。检测到左手或无手的帧会安全保持/张开，不会改用 legacy curl。
- 【已验证】红色定位线和右上掌心视图均沿 Allegro 掌心局部 `+Z`；相机位于手指外侧，视图中保留红色准星和目标方块，不再让手指或激光圆柱遮挡。
- 【已验证】Shadow 运行时模型去除了 Panda 与手掌之间约 213 mm 的黑色前臂段，保留直接腕部安装和完整 Shadow 手。Shadow 目前明确使用规则式 curl，尚未接入 dex；这不是 Allegro dex 的回退。
- 【已验证】最近一次聚焦回归为 `48 passed`：三视图、Allegro 模型、主程序安全退出、dex 几何/sidecar、Shadow 模型和可用性均通过。Python 编译检查也通过。
- 【已验证】冻结清单检查为 `172/173` 相符；唯一旧差异是 `src/teleop/teleop_factory.py`，不属于本轮改动。`third_party/panda-gym` 冻结基线文件均未被修改。

### 仍需真人设备验证

1. 用真实右手运行 Allegro 主程序：摄像头窗口应显示 `detected hand: Right` 和 `DEX ACTIVE: retarget() updated`，随后确认张手、握拳和拇指动作能让 16 个 Allegro 关节自然跟随。最近一次 25 秒相机 smoke 虽成功处理 598 帧，但未检测到手，因此不能据此声称真人重定向已通过。
2. 在已连接的 SpaceMouse 上运行 `diagnostic --duration 10`，再确认 Allegro 与 Shadow 静止时没有漂移、六轴方向正确、两个按钮和 `R/Q` 正常。最近一次常规入口在本进程中没有找到匹配 HID 接口，因此硬件链路仍标为待验证。
3. 检查右上掌心视图的红色准星与主画面红线落点一致，并以实际抓取姿态确认 2/3 尺寸 Allegro 和缩短后的 Shadow 都不会先碰桌面或方块。

## 2026-10-20｜Allegro dex 相对中位与掌心视图调整

- 【已验证】真人右手日志 `outputs/spacemouse_wrist/20261020T225259Z_integrated` 中，`camera` 来源的 781 帧实际改变了 MuJoCo Allegro 的全部 16 个关节；最大关节轨迹范围为 `1.92 rad`。因此官方 dex sidecar、右手筛选、命名映射和手部执行器连接均已工作，不会回退到 legacy curl。
- 【已实现未完成真人效果验收】默认 dex 现在在首个有效物理右手帧捕获相对中位，并把后续官方 dex 解算的变化映射到 Allegro 的安全张手姿态。`C` 可在相机窗口或三视图窗口随时重新取中位；`R` 会清除中位并重新开始。该设计解决真人自然起始手型与官方源模型绝对姿态不同导致的“不像跟手”的第一版问题。
- 【已验证】掌心相机从掌心工作面外 `25 mm` 处沿与激光相同的局部 `+Z` 法线观察；右上角只显示桌面、方块和红色准星，暂时隐藏机器人自身视觉几何。`52 passed` 的回归涵盖激光方向、相机与激光同轴、渲染恢复、dex 中位、相机输入和安全退出。

下一次真人测试请先在相机中举起张开的**右手**，看到 `DEX RELATIVE: neutral captured` 后再弯曲手指；若机器人中位姿态不舒适，保持想要的中位手势并按 `C`。右上红色十字就是掌心激光的落点方向；移动 Panda 让方块靠近十字后再闭合手指。

## 2026-10-22｜LIBERO Spatial：Shadow dex、漂移修复、提速与画面

两个 LIBERO 入口（`run_libero_spatial_panda_shadow_teleop.py`、`run_libero_spatial_panda_robotiq_three_finger_teleop.py`）共用 `src/wrist_teleop/libero_spatial_teleop.py`，以下修改两者同时生效。

- 【已验证·根因】robosuite 1.4 的 OSC 每一步都把位置目标重设为“当前末端位置 + 增量”，零输入时不保持位置；同时 Shadow 的 OSC 控制点 `grip_site` 挂在 `rh_palm` 上，位于 Shadow 自身两个弱驱动腕关节 WRJ2/WRJ1 之后。手指一动，腕关节晃，OSC 用机械臂去追这个点，形成放大的振荡。无 SpaceMouse 输入、只开合手指 6 s，末端漂移 **109 mm / 17.6°**。
- 【已验证·修复】LIBERO 适配器自己保存期望位姿（等价于 robosuite 1.5 的 `goal_update_mode="desired"`），带 5 cm / 0.5 rad 牵引上限，松手瞬间锁定到当前位姿；OSC `kp=300`。LIBERO Shadow 使用 `ShadowHandRightRigidTcp`：同一模型与 20 维动作，只把 OSC 控制点放到与法兰刚性相连的 `rh_mount` 上（WRJ=0 时与原位置一致）。同一测试漂移降为 **0.26 mm / <1°**。主线 Shadow 入口未改动。
- 【已验证】机械臂提速：默认平移 2.5×、旋转 1.6×，满偏实测 **0.18 m/s**（原名义上限 0.08 m/s，且实际只跑 8 Hz）；松手后停止误差 1–6 mm。控制周期若被拖慢，按真实周期补偿（最多 2×），手感速度不随帧率下降。
- 【已验证】帧率：去掉每步无用的策略相机观测渲染（39→25 ms/步）；Windows 计时器精度提到 1 ms（Python 3.8 的 `sleep` 原本每次多睡约 14 ms）；三视图 1:1 原生 1920×1080 显示并用 `cvtColor`（显示耗时 21→6 ms，且不再缩放发糊）。可见窗口实测 **21 Hz**（此前真实运行 8 Hz）。
- 【已验证】修复按 `R` 后崩溃 `'MjSim' object has no attribute 'model'`：robosuite 1.4 硬重置会销毁旧 sim，激光现在随重置重新绑定。
- 【已实现·待真人验证】Shadow 默认 `--hand-control dex`：官方 dex-retargeting 0.4.6 **DexPilot**（`configs/dex_retargeting/shadow_hand_right_dexpilot.yml`，在官方配置基础上只把 WRJ2/WRJ1 排除出优化，由 SpaceMouse 负责腕部）。URDF 来自 dex-urdf（`third_party/dex_urdf/shadow_hand`，Apache-2.0）；与 MuJoCo Shadow 指尖几何误差 0.1 mm，关节符号/限位一致；远端腱驱动取 J2+J1。摄像头画面在 MediaPipe 前做了镜像，因此送入 dex 前把 3-D 关键点 x 轴翻回（与官方检测器的未镜像输入一致）。sidecar 首次启动约 60 s。
- 【已验证】Jaco 三指夹爪：robosuite `JacoThreeFingerGripper` 每个策略步只走 0.01 且只看符号，全行程约 10 s；Jaco 入口改为每步 0.15 并按指令幅值，1 s 内手指平均行程 0.57 rad（原 0.07 rad）。

```powershell
& .\.venv-libero\Scripts\python.exe .\scripts\run_libero_spatial_panda_shadow_teleop.py
& .\.venv-libero\Scripts\python.exe .\scripts\run_libero_spatial_panda_shadow_teleop.py --arm-speed 3.5
& .\.venv-libero\Scripts\python.exe .\scripts\run_libero_spatial_panda_robotiq_three_finger_teleop.py --viewer-smoke
```

退出时终端会打印 `median control/render loop: xx Hz`。真人测试请先张开右手面对摄像头，再做拇指—食指、拇指—中指捏合，看 Shadow 拇指是否对掌并靠近对应指尖。

## 2026-10-22｜Allegro 映射与构型、LIBERO 视图布局

- 【已验证·仿真】Panda + Allegro 默认手部映射改为 `curl-pinch`，沿用 Shadow 入口的规则式逻辑：MediaPipe 关节角估计每指弯曲（与摄像头镜像和首帧中位无关），再按拇指尖到各指尖距离给出捏合强度；捏合时伙伴手指至少弯到捏合位，拇指混合到在 2/3 尺寸 MuJoCo Allegro 上用正运动学搜索得到的对掌姿态（食指 / 中指 / 无名指各一套）。仿真中拇指—食指指尖间距：旧拇指弯曲 30.5 mm，新食指捏合 17.2 mm，中指捏合 16.0 mm（指尖接触距离 16 mm）。`--hand-control dex` 与 `legacy-curl` 仍可用；`--dex-smoke` 自动使用 dex。
- 【已验证·仿真】遥操作入口使用 `AllegroRightHandPalmDown`：同一 16 关节与动作契约，手绕自身 Y 轴转 −90°，手背贴在法兰（3 mm 间隙），静止时手掌平行桌面、手指朝前，与 Shadow 一致。OSC 控制点仍在法兰原位；激光与掌心视图改沿真实掌心法向（手局部 +X），静止时垂直指向桌面；与机械臂无接触。Gate 2 和默认模型不变。
- 【已验证】两个 LIBERO 入口的三视图：主视图 1270 px，右侧掌心视图与侧视图各 640×535，占合成画面宽度的 1/3；仍为 1920×1080 原生显示，实测 21 Hz。
- 【已知·既有】`run_grasp_validation.py --gate 2` 当前为 `failed_gate2`；最后一次通过是 2026-09-17，2026-10-20（Allegro 改为 2/3 尺寸之后）的 Gate 2 可视化运行已经失败，并非本次修改引起。`test_dex_sidecar_runtime.py` 中开/合手对比测试因 2026-10-20 的相对中位默认值而失效，按要求暂不修改。

```powershell
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_wrist_teleop.py integrated
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_wrist_teleop.py integrated --hand-control dex
```

## 2026-10-22｜Gate 2 恢复通过（2/3 尺寸 Allegro）

- 【已验证·根因】Gate 2 失败（fixture smoke `object_ejected`，接触后峰值速度 0.888 m/s，限值 0.20）来自 2026-10-19 把运行时 Allegro 缩为 `2/3`：v9 的方块位置 `[0.085, 0.030, 0.115]` 与握合 curl `[0.50, 0.50, 0.50, 0.85]` 按原尺寸手标定，方块落在小手的拇指—手指包络之外。仅把 `ALLEGRO_GEOMETRY_SCALE` 临时强制为 1.0（不改仓库），Gate 2 即 1/1、3/3、10/10 通过，fixture 指标与 2026-09-17 逐位相同。
- 【已验证·修复】`src/wrist_teleop/grasp_validation.py` 配置升为 v10：方块 TCP 位置 `[0.046, 0.017, 0.088]`，curl `[0.45, 0.45, 0.45, 0.90]`，拇指先单独闭合 0.60 s（`static_thumb_lead_s`）再让食指/中指压住方块远端面；搜索网格改为 x 42–50、y 13–21、z 86–90 mm，第 3 阶段候选同样拇指先行。若各指同时闭合，缩小后的拇指会扫到方块侧面，法向点积约为 0，不构成对握。方块、物理参数、fixture、法向对握判据和 1/3/10 验收策略均未改；无 weld、mocap、qpos 改写，也未改重力或碰撞。默认 `AllegroRightHand` 安装和遥操作 palm-down 变体未改。
- 【已验证】`run_grasp_validation.py --gate 2` 为 `passed_gate2`：fixture smoke 通过，筛选 16 个候选，单次 1/1、qualification 3/3、formal 10/10，锁定 `outputs/grasp_validation/20261022T182301Z_gate2`。`tests/test_grasp_validation.py` 通过；冻结清单仍为 `172/173`，唯一差异仍是 `src/teleop/teleop_factory.py`。上文“【已知·既有】Gate 2 当前为 failed_gate2”一条已不再适用。
- 【未验证】Gate 3 同样读取 `static_cube_center_tcp_m` / `static_close_curls`，本次未运行 Gate 3。`--visualize` 已能解析到新锁，但本次未打开查看器。

```powershell
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_grasp_validation.py --gate 2
```

## 2026-10-22｜SpaceMouse 漂移 / 方向映射与拇指逻辑（所有入口）

根据四段测试视频和对应日志（Allegro 616 s、Jaco 173 s 会话）定位的问题与修改：

- 【已验证·根因】平移和旋转互相串扰：日志中 67–72% 的有效输入同时带有平移和旋转，推帽时帽子会被带歪，手就跟着转。`SixDofMapper` 新增 `motion_coupling="dominant"`（默认）：每一刻只保留较强的平移组或旋转组（`dominant_switch_ratio=1.5` 迟滞），并用 `axis_snap_ratio=0.3` 去掉组内弱分量。用真实日志回放，平移时带出的意外旋转从累计 1718°（Allegro 会话）/ 448°（Jaco 会话）降为 0。需要同时平移和旋转时，把配置改成 `"motion_coupling": "combined"`。
- 【已实现·待实物确认】方向映射：旧配置 `axis_order=[1,0,2,…]` 是早期世界坐标映射的遗留，在相机坐标下会把“前后推”变成“左右”、把“左右推”变成“前后”（相差 90°）。默认值改为 3Dconnexion HID 约定（帽子 logo 朝向自己：+X 右、+Y 朝向自己、+Z 向下）：`axis_order=[0,1,2,3,4,5]`、`axis_inversion=[1,-1,-1,1,-1,-1]`，即前推→画面纵深、右推→画面右、上提→上；旋转为“手跟着帽子转”。旋转符号来自约定推导，请先运行下方的引导式标定，用你的设备写入实测结果（旧配置自动备份）。
- 【已验证·仿真】主线 robosuite 1.5 入口的 OSC 期望位姿保护（`DesiredGoalGuard`）：松手瞬间锁定到实测位姿，期望位姿最多领先 5 cm / 0.5 rad，防止推到桌面或物体时目标越积越远、松手后继续动。主线 Shadow 改用刚性 TCP（与 LIBERO 相同根因）：零输入开合手指 6 s 的漂移从 3.7 mm（最大 11 mm）降到 0.2 mm（最大 1.3 mm）。
- 【已验证·仿真】Allegro 掌心朝下版本的 OSC 控制点移到手掌前方抓取中心（手局部 `(0.04, 0, 0.03)` m），旋转时绕抓取点转，不再让手指绕法兰大幅甩动。Gate 2 使用的默认模型不变，Gate 2 仍通过。
- 【已验证·仿真】Allegro 拇指：新增连续“对掌度”（拇指尖到无名指根部的距离 / 掌宽），对掌度驱动关节 12/13（掌下摆动、指腹转向手指），拇指弯曲驱动 14/15，捏合吸附仍叠加在上面。拇指尖到中指根部：张开 136 mm → 贴食指 106 mm → 横跨掌心 37 mm；握拳时拇指尖距中指尖 27 mm，可以包握。

首次使用新映射前，请运行一次标定（约 40 秒，需要 SpaceMouse；按提示依次做 6 个动作，每个保持约 2.5 秒）：

```powershell
& .\.venv-robosuite\Scripts\python.exe .\scripts\calibrate_spacemouse_axes.py
& .\.venv-robosuite\Scripts\python.exe .\scripts\calibrate_spacemouse_axes.py --dry-run
```

## 2026-10-22｜LIBERO Spatial 纯摄像头遥操作（Panda + Shadow，无 SpaceMouse）

新入口 `scripts/run_libero_spatial_panda_shadow_camera_teleop.py`（只用 `.venv-libero`）：一个普通摄像头里的**右手**同时控制机械臂末端和 Shadow 手指（默认 DexPilot）。系统**不含任何自动/智能抓取辅助**：所有运动都来自操作者的手，保护层只会减速或停止，不会自己移动机械臂。

**怎么用**：启动后把张开的右手放在摄像头前，手掌朝向摄像头。约 1.5 秒后终端显示 `arm: tracking`，此后（2026-10-23 起为增量映射，像鼠标一样）：

| 你的手 | 机器人 |
| --- | --- |
| 向你的右 / 左、向上 / 向下移动 | 主视图里向右 / 左、向上 / 向下。慢动精细（约 0.5×），快动走得远（最高约 1.8×，“指针加速”） |
| 停在比起始位置**更靠近**摄像头处 / 更远处 | 末端以与偏离量成正比的速度远离观察者 / 靠近（速率控制）；±3 cm 死区内不动，回到中性位置即停 |
| 转手：拧转、前倾、侧倾 | Shadow 跟随手掌的全部转动（拧转 = 偏航，前倾 / 侧倾 = 倾斜）；手在平移时旋转被冻结 |
| 手指弯曲、捏合 | Shadow 手指（DexPilot） |
| **左手**伸进画面（与右手同时出现） | 离合：机械臂原地保持，右手可挪回舒服位置；左手离开后从当前位姿继续，不跳 |

按键（仿真窗口）：`C` 暂停/恢复；`-` / `=` 增益；`R` 重置；`Q` 退出。2026-10-23 起去掉了精细模式和旋转模式切换（它们会改变操作者到机器人的映射，违背“不使用 shared control”的原则），旋转固定为全部跟随。按键有 0.35 s 防连发。手离开画面时机械臂原地保持，回来后从当前位姿继续。

**画面**：主视图（左）+ 两个与操作方向对齐的**正交**视图（右）：右上为俯视平面图（画面右 = 你的右，画面上 = 远离），手掌在碗正上方就是真的在正上方，不受高度视差影响；右下为从左侧看的立面图（画面上 = 上），红色竖线和黄色数字显示掌心离下方表面的高度。主视图有竖直向下的掌心激光；左下角嵌入摄像头预览：方框为安全区（绿 / 黄 = 接近边缘 / 红 = 被裁切），右侧竖条为纵深量规（灰带为死区，标记在灰带外即在移动），预览上方显示 `CLUTCH`、`PAUSED` 等状态。

**映射与关键设计**（`src/wrist_teleop/camera_arm_control.py`，参数集中在 `CameraArmSettings`）：
- 【已验证·合成数据】手腕位置透视精确：掌部 6 个刚性关键点的 3-D 形状（MediaPipe 世界坐标，按时间平均成掌形模板）必须落在各自像素的视线上，解小线性最小二乘得掌心深度，再换算到手腕；偏心、转手时误差为 0，握拳不影响深度。边界检查只要求手腕 + 4 个指根在画面内（拇指根被裁不再算丢失），并把离边缘距离送到预览安全区。
- 【已验证·合成数据】防抖：One Euro 滤波 + 横向 4 mm 死区（运动时 0.6 s 精细跟随）+ 拧转 2.9° / 倾斜 5.7° 死区；纵深用 3 cm 死区的速率控制，静止的深度噪声不会推动机械臂。1–2 像素噪声下手静止目标完全不动；单帧误检被忽略。
- 【已验证·合成数据】增量映射：慢动增益 0.5、快动 1.8（手速 2 → 12 cm/s 平滑过渡，`--gain` 为总倍数）；纵深 1.5 (m/s)/m，最高 0.12 m/s（`--depth-gain`）；平移速度 > 3 cm/s 时及其后 0.3 s 旋转冻结。被工作空间、牵引或保护拒绝的指令直接丢弃（不累积），往回动机械臂立即响应；目标最多领先当前指令 0.25 s 的行程，手停机械臂就停。
- 【已验证·LIBERO 闭环，仿真时间】合成手右移 8 cm → 末端右移 104 mm；靠近摄像头 7 cm 保持约 1.7 s → 远离 98 mm，回中性后 1 s 内漂移 0.0 mm；上抬 6 cm → 59 mm；拧 30° → 偏航 29.5°；静止 1.5 s 末端波动 0.0 mm。离合时手移 10 cm，末端变化 < 0.3 mm（实时冒烟）。
- 【已验证·LIBERO 闭环】保护（只减速/停止，不主动移动）：① 工作空间 = 场景物体与固定件的水平包围盒 + 15 cm、与初始位姿 ±35/±40 cm 盒取交集，桌面以上 2 cm、肩部可达 0.80 m、离基座轴 ≥ 0.28 m、偏航 ±90°、倾斜 ±30°；② `ArmSafetyGuard`：雅可比预测（含 OSC 领先量、前瞻 3 步），拦截压向关节限位或可操作度跌破初始 25% 的运动；**肘关节保持弯曲 q4 ≤ −0.82 rad**——Panda 因 0.0825 m 连杆偏置，真正的伸直奇异点在 q4 ≈ −0.47 rad（可操作度≈0），而不是限位 −0.07 rad，旧版 0.45 rad 裕度只离奇异点 0.05 rad，所以以前需要“主动退出”；停住后保持当前目标让 OSC 稳住位姿，而不是每步把目标挪到实测位姿（那样会随动力学漂入奇异）；③ 虚拟墙：被拦后去掉朝墙分量（可沿墙滑动），操作者一往回动或每 1 s（仿真时间）严格复查确认安全时自动解除；④ 目标 1 s 到不了（被物体挡住）就原地保持。全速外推 250 步：停在 q4 = −0.74、可操作度比 0.63 并保持，可退回。
- 【已实现·待真人验证】真实摄像头下的手感（增益曲线、纵深死区、离合检测稳定性、`--camera-fov`）需要你实测。

```powershell
& .\.venv-libero\Scripts\python.exe .\scripts\run_libero_spatial_panda_shadow_camera_teleop.py
& .\.venv-libero\Scripts\python.exe .\scripts\run_libero_spatial_panda_shadow_camera_teleop.py --gain 0.8 --depth-gain 1.0
& .\.venv-libero\Scripts\python.exe .\scripts\run_libero_spatial_panda_shadow_camera_teleop.py --viewer-smoke
& .\.venv-libero\Scripts\python.exe -m pytest -q .\tests\test_camera_arm_control.py .\tests\test_libero_camera_teleop.py
```

## 2026-10-27｜纯摄像头 v2 第一次真手测试：分析（修改见 2026-10-28）

数据：录屏 `C:\Users\lenovo\Desktop\屏幕录制 2026-10-27 205230.mp4`（1:19）、`205457.mp4`（0:59）；日志 `outputs\libero\20261028T004810Z_panda_shadow_camera_teleop\`（20:48–20:55，录屏 1 的视频时间 ≈ 日志 monotonic − 15060.5 s）。全程增益被 `=` 调到 ×3。

- 【已验证·根因】"手没动，机械臂还在加速往前"（录屏 1 约 24 s）：不是延迟，是纵深速率控制。滤波后的纵深偏移从 1 cm 涨到 5.8 cm（One Euro 0.3 Hz，手停后还在慢慢涨），机械臂前进速度涨到 11 cm/s；之后手基本不动（纵深速度 0.2–0.4 cm/s），偏移停在 4.2 cm，机械臂继续以 5 cm/s 前进，直到 reach_limit。屏幕上显示 `depth +5 cm/s`。增益 ×3 同时把 depth_gain 从 1.5 放大到 4.0。
- 【已验证】延迟：两段录屏一致，摄像头预览里手动作 → 俯视图红十字（实测手掌）滞后 300–330 ms（不含摄像头采集延迟）。日志里"映射目标 → 实测手掌"约 280–290 ms，"滤波后的手 → 目标"为 0，说明大部分延迟在目标之后（TargetFollower 0.2 m/s 限速 / leash / OSC kp=300），明天要把 composer 的 goal 也记进日志再拆分。循环 16–18 Hz，没有卡顿。
- 【已验证】暂停 / 重新锚定：每次手丢失后重新锚定（15103、15157、15195、15282 s），纵深中位都是在手还在移动时取的，1 s 内偏移达到 ±4–25 cm，机械臂自己沿纵深走了 6–13 cm。C 暂停 / 继续（15075 s）的日志看起来手和机械臂比例正常（约 1.5 倍，增益 ×3），需要再问用户具体哪里不准。整场左手离合一次都没触发（没有 clutched 状态）；手经常走到画面边缘（edge < 0.07）而丢失，手到摄像头距离 28–73 cm。
- 【说明】20:50:03–20:51:37 我同时在跑冒烟测试（与录屏 1 的前 30 s 重叠），这段时间循环仍是 16.6–18 Hz，未见影响。

当时的计划（已于 2026-10-28 完成，见下一节）：
1. 纵深改成和左右上下一样的增量位置控制（手不动机械臂就不动），单独的纵深死区和更轻的滤波；去掉速率控制，更新画面文字、深度条、`--depth-gain` 含义和文档。
2. 继续 / 重新锚定时，要求手稳定（速度低于阈值）后再锚定。
3. 延迟：拆分 follower / OSC 的份额，考虑缩短指针加速的测速窗口、调 One Euro；全部用测试和闭环仿真验证。
4. 更新测试、跑两套测试、合成手闭环、更新 Run Instructions 5.2 / RUN_COMMANDS.md / README。

## 2026-10-28｜纯摄像头 v2：纵深改位置控制、静止后才重新锚定、降延迟

针对 10-27 真手测试的两个问题，只改了纯摄像头入口（`camera_arm_control.py`、`libero_camera_teleop.py`、`operator_views.py`、入口脚本）；SpaceMouse 的 LIBERO 入口不变。

- 【已验证·仿真】纵深改成和左右上下一样的增量位置控制（指针加速、手不动机械臂就不动）。去掉 `depth_rate` / `depth_deadzone_m` / `depth_max_speed_m_s`，`--depth-gain` 改为"纵深相对其他方向的额外倍数"（默认 1.0），`-`/`=` 只调整体增益。纵深单独的死区 10 mm、运动阈值 12 mm、One Euro 0.5 Hz / beta 5（在 2 px / 4 mm 的严苛合成噪声下静止漂移 0）。用 10-27 日志里的真实手部轨迹回放（增益 ×3）：录屏 22–24 s 手停住时，旧逻辑机械臂以 5–10 cm/s 一直前进到 26–28 cm，新逻辑停在 8 cm、速度 0。
- 【已验证·仿真】暂停后继续 / 手丢失后找回 / 卡住恢复：手必须先稳定（3-D 速度 < 3 cm/s，且已有 15 帧稳定检测）才重新锚定，期间显示 HOLD STILL，机械臂不动；手挪回来的过程不会带动机械臂。新增单元测试 `test_resume_waits_for_the_hand_to_settle`。左手离合松开时仍立即接续（与鼠标抬起放下一致）。
- 【已验证·仿真】降延迟：左右方向 One Euro beta 15 → 60（映射这一段延迟约 100 → 67 ms，静止漂移仍为 0）；本入口 OSC kp 300 → 500（`CAMERA_OSC_KP`，跟踪正弦目标的滞后 100 → 50 ms 仿真时间，无过冲，手指开合时末端漂移 0.53 → 0.37 mm）；末端速度上限 0.20 → 0.30 m/s（6 cm / 0.4 s 快速一挥后，机械臂停稳的拖尾 400 → 300 ms 仿真时间）。慢速移动后的停止拖尾 100 → 0 ms。
- 【已验证】仍存在：循环 16 Hz（`env.step` 约 25 ms + 三视图渲染约 20 ms），仿真实时系数约 0.8；起步时的死区和慢速 0.5 倍增益是有意保留的（保证手不动机械臂不动）。
- 【已验证】深度条改为显示"手相对开始跟踪时近 / 远多少"（±15 cm，接近两端变黄，提示用离合），状态行去掉 `depth cm/s`。
- 【未验证】以上都还没有用真手复测。

测试：`test_camera_arm_control.py` 19 通过；LIBERO + 纯摄像头 28 通过（闭环测试改用 kp 500）；主线 201 通过、9 跳过、1 个已知失败（`test_dex_sidecar_runtime` 开/合手）；入口 `--headless --headless-smoke` 正常结束。

## 2026-10-28｜LIBERO 遥操作入口支持其他任务套件（--suite）

- 【已验证·仿真】三个 LIBERO 遥操作入口（Shadow、Robotiq 三指、纯摄像头）新增 `--suite`（`libero_spatial` 默认、`libero_object`、`libero_goal`、`libero_10`、`libero_90`），`--task-id` 在所选套件内编号，越界时报出有效范围。日志 metadata 里记录 suite。
- 【已验证·仿真】主视图和侧视图的固定机位原本按 Spatial 桌面场景摆放；libero_object 的机械臂底座在地面上（底座 z = 0 而不是 0.912），画面里只剩左下角一小块。现在机位按"当前底座 − Spatial 底座"整体平移（`follow_robot_base`），各套件看到同样的机械臂左后方视角；Spatial 机位不变（仍是 -1.55, -1.10, 1.52）。各套件里物体相对底座的位置基本一致（前方 0.4–0.75 m、左右 ±0.27 m）。
- 【已验证·仿真】libero_object 任务 0（字母汤）、4（番茄酱）、7（牛奶）：Shadow 入口无头冒烟正常；纯摄像头入口任务 4、Robotiq 三指入口任务 7 冒烟正常；查看了开场画面。LIBERO + 纯摄像头测试 28 通过。
- 【未验证】其他套件还没有真人遥操作过；libero_goal / libero_10 / libero_90 只确认了底座位置，未逐个看画面。
- `RUN_COMMANDS.md` 增加 Spatial 与 Object 两套 `--task-id` 0–9 对照表和 libero_object 命令。

## 2026-10-28｜LIBERO：三视图重做、帧率、手朝下构型、物体质量、拇指 / 多指抓取

起因：用户 libero_10 一段 18 分钟的会话控制循环只有 10.6 Hz；右上角手掌视角在抓取时被物体完全挡住；右下角侧视图没有空间感；希望 libero_object 的物体更重、Shadow 有手朝下的构型、拇指在三指 / 全手 / 铲 / 侧握时更自然、机械臂更快更稳更流畅、画面更清楚帧率更高。

- 【已验证·仿真】计时修正：所有 LIBERO 循环原来用 `time.monotonic()`，Windows 上它 15.6 ms 才跳一次，测得的周期只有 47 / 62 / 94 ms 几档，每步的速度补偿系数在 ×1.0 / ×1.25 / ×1.88 之间乱跳（机械臂一顿一顿的主因之一）。循环计时和等待改用 `perf_counter`；手部时间戳仍与摄像头线程共用 monotonic。
- 【已验证·仿真】新显示模块 `libero_views.py`（三个 LIBERO 入口共用）：主线程每步只做 `mjv_updateScene` 快照（三视图约 1 ms），另一线程用独立 OpenGL 上下文渲染、合成、显示；每个控制步额外在步中间拍一次快照，显示帧率约为控制频率的两倍。进程设为 DPI-aware，窗口按主显示器工作区 1:1 显示（2560x1440 屏上为 2544x1334，原来 1920x1080 被 Windows 放大 1.2 倍导致发糊），`--view-size WxH` 可改。实测（webcam + dex + 窗口，libero_10 任务 9）：控制循环 19.8 Hz（p5 19.4），显示约 36–38 fps，原渲染器下同场景用户会话为 10.6 Hz。窗口客户区 2544x1334、DPI 115，已用 Win32 API 查过。
- 【已验证·仿真】视图：主视图 fovy 45 → 38（略放大）；右上改为跟随手的特写（从主相机一侧低角度看，手指 / 拇指 / 物体之间的间隙清楚，`V` 键绕手转 90°）；右下改为跟随手的"右前方 3D 视角"（与主视角视线约成直角，补深度；机械臂在手后面，不会挡住）。试过从上往下的俯视跟随视角：手朝下时完全被 Panda 手腕挡住，弃用。摄像头预览画中画放在主视图左下角。纯摄像头入口保留原来的正交俯视 / 侧视，只换成新的线程渲染和 1:1 窗口，叠加文字按窗口高度缩放。
- 【已验证·仿真】机械臂：SpaceMouse 的 LIBERO 入口 OSC kp 300 → 500（0.15 m/s 匀速指令下滞后 18.5 → 14.4 mm，松手后停稳 118 → 94 ms，无过冲），默认速度倍数 2.5 / 1.6 → 3.0 / 2.0。测过 OSC 目标线性插值：速度波动本来只有 3–5 %，插值反而增加滞后，未采用。
- 【已验证·仿真】`--hand-mount down`（Shadow 两个入口）：同一只手在法兰上绕夹爪 y 轴转 90°，起始姿态手指竖直朝下、掌心朝机器人；TCP、激光（改沿手指方向）随之变化；libero_object 起始时手最低点离地 12 cm、无接触。注意：robosuite 的模型基类自带 `mount` 属性，类属性因此改名为 `hand_mount`。
- 【已验证·仿真】`--object-mass-scale N`（三个入口）：所有 BDDL 物体的质量和惯量 ×N（保持密度分布），每次复位后对新加载的模型应用一次、不会累乘；启动时打印每个物体的质量。libero_object 官方质量 5–136 g（篮子 136 g、汤罐 26 g），×10 后汤罐 259 g。
- 【已验证·合成数据】拇指 / 多指：在 dex 子进程里给官方 DexPilot 加"抓取项"（`configs/dex_retargeting/shadow_grasp_terms.json`）：① 人手拇指三段、四指近 / 中节的方向作为姿态目标（按机器人指节长度）；② 只有人手拇指尖落在该手指远节上时才做 DexPilot 的捏合吸附。用合成人手（捏、三指、全手包握、侧握、紧侧握、铲、五指撮）评估：捏 / 三指 / 五指接触不变；四指近节方向误差 10–20° → 2–10°（小指 21–29° → 3–8°）；侧握拇指不再被吸到食指尖（3 mm 关键点噪声下官方 70 % 帧吸附，新版 0 %），拇指贴在食指侧面，距离与人手一致；侧握 / 包握时拇指远节仍有约 25° 偏差（Shadow 拇指运动学限制）。每帧求解 9 → 16 ms。全部只读人手关键点，没有物体信息、没有抓取预设。`--dex-official` 可切回原版对比。
- 【未验证】真人：新视图、手朝下构型、加重物体、新拇指映射都还没有真人遥操作测试。

测试：LIBERO + 纯摄像头 28 通过；新增 `test_libero_views_and_options.py` 5 通过（布局、质量缩放不累乘、手朝下、线程渲染）；dex worker 新增 3 个单元测试；主线 201 通过、9 跳过、1 个已知失败（`test_dex_sidecar_runtime`）；三个入口无头冒烟 19.8–19.9 Hz。

## 2026-10-29｜精简入口脚本和运行命令

项目目标收敛到"在 LIBERO 里用 Shadow 流畅抓取"，把不再用的入口删除（先移到 `archive\` 确认，随后按用户要求整个删掉；GitHub 仓库里还有它们 10-28 之前的版本）：四个夹爪对照（平行夹爪、Robotiq 85、Robotiq 三指、Jaco 三指）、LIBERO Robotiq 三指入口、robosuite 1.5 的 Shadow 抓方块入口、Allegro Gate 2 静态抓取回归、旧版 PyBullet 静态 IK 采样。`src\` 下的模块和对应测试都没动，从 GitHub 取回脚本即可恢复。

保留：两个 LIBERO Shadow 入口（SpaceMouse / 纯摄像头）、dex 子进程、SpaceMouse 标定和 3DxWare 配置、`run_spacemouse_panda_wrist_teleop.py`（提供 SpaceMouse 硬件诊断）、LIBERO 场景预览、冻结旧版及其哈希检查（AGENTS.md 规定不改）。`RUN_COMMANDS.md` 从 51 条减到 20 条；`Run Instructions.txt` 第 3–5 节改为退役说明，删掉 Robotiq / Gate 2 命令；README 入口表同步。

## 2026-11-04｜Shadow 映射：拇指抬起增益、每指分段缩放、连续捏合

起因（用户真人测试后提出）：① 侧握竖直杯子时要先把拇指抬到四指前方，仿真拇指 THJ4 太不灵敏；② 确认"捏合吸附"不是 share control，项目只允许调摄像头到仿真的转换权重 / 倍数；③ 人手伸平时仿真小指仍然弯曲，要求严格按摄像头映射修正，不能写死"开始是平的"或做吸附。

分析（改代码前）：
- 【已验证·仿真】THJ4 电机跟随正常：指令 0.3 / 0.6 / 0.9 / 1.2 rad → 实际 0.33 / 0.62 / 0.92 / 1.21（手朝前）、0.29 / 0.58 / 0.88 / 1.18（手朝下）。问题在映射给的指令太小。
- 【已验证·真人日志】用户多段会话里 THJ5 有 11–22 % 的帧顶在 60° 上限，这些帧里 THJ4 中位数只有 8–34°（官方版 52–56°）：拇指靠 THJ5 贴着手掌横扫，而不是用 THJ4 抬到四指前面。
- 【已验证·合成数据】摄像头正对手掌时"拇指抬离手掌"就是深度方向。把合成人手的深度压到 60 %，人手拇指抬起 30° / 45° 时 THJ4 从 35° / 49° 降到 24° / 36°，THJ5 被推向上限，和真人日志一致。
- 【已验证·真人日志】手最张开的 10 % 帧里，机器人小指只伸到伸直长度的 0.91–0.97（食指 / 中指 1.00）。原因：DexPilot 所有手指同一个倍数 1.2，Shadow 五指几乎一样长，人的小指短得多。
- 捏合吸附是官方 DexPilot 自带的（10-22 起就有），只读人手关键点，人权重为 1，不属于 share control；但它是 3 cm 阈值 + 5 cm 迟滞，不是倍数。用户选择换成连续映射。

改动（只在默认 dex 映射里；`--dex-official` 仍是原版）：
- 【已验证·合成数据】拇指抬起增益：拇指四个关键点离人手掌平面（腕 + 四个 MCP 拟合）的距离 ×k（`thumb_lift_gain`，默认 1.5，`--thumb-lift-gain` 可调，范围 0.5–4）。手转动时同样按手掌平面算；拇指平躺在手掌平面内时不受影响。深度压到 60 % 时，人手抬起 15 / 30 / 45 / 60° → THJ4 18 / 24 / 36 / 52° 变为 21 / 32 / 45 / 61°（深度如实、无增益时为 23 / 35 / 49 / 64°；k = 1.67 可完全恢复）。
- 【已验证·合成数据】每指分段缩放（`finger_length_scaling`）：每根手指的手掌段（腕→MCP）和手指段（MCP→指尖）分别按 Shadow 对应长度缩放，长度每帧从关键点量出。手指段弦长 = 机器人手指长 × 人手这根手指自身的伸直度，所以伸直对伸直、弯一半对弯一半，与人和 Shadow 掌骨角度的差异无关。拇指保持官方 1.2。合成伸平手小指伸直度 0.97 → 1.01（深度压缩 60 % 时同样 1.01）。最初只按"腕→指尖"整体缩放，人造手形上远端关节仍被弯 12°，所以改成分段。长度测量噪声在"机器人长度 / 测量长度 × 测量向量"里相互抵消，3 mm 关键点噪声下指尖抖动与官方版相当，没有另加平滑（官方输出低通仍在）。
- 【已验证·合成数据】连续捏合：机器人拇指—各指尖距离 = 1.2 × max(0, 人手距离 − 1 cm)，方向取人手方向，没有阈值和迟滞（1 cm 是两指指腹贴紧时两个指尖关键点之间的间隙，`pinch_gap_m`）。权重在人手距离 5 → 3 cm 之间从 1 平滑升到官方的 200；两根手指同时靠近拇指时，官方的 3 cm 机器人指尖间距按同一系数平滑加入。捏合按摄像头测得的原始距离判断（不受拇指增益影响），否则抬起增益会把捏紧的两指拉开。原来的"指尖段"门控删除。合成结果：指尖捏 4–5 mm 闭合（官方 3 mm）；三指捏 10 / 17 mm（人手 10 / 17）；紧侧握拇指—食指尖 18 mm（人手 25 mm，目标 18 mm；官方吸到 2 mm）。
- 每帧日志：`camera_details` 里新增 `mano_landmarks`（送入求解器的 21 个人手关键点）和 `dex_mapping`（各指缩放、拇指增益、捏合系数、拇指—各指距离）；纯摄像头入口的 step 事件也开始记录 `camera_details`。下一次真人测试后可以用数据标定增益和 1 cm 间隙。
- 【已验证】真实 dex 子进程端到端（LIBERO 解释器）：默认、`--thumb-lift-gain 2.5`、`--dex-official` 三种都正常求解，元数据里增益生效；`--dex-official` 与 `--thumb-lift-gain` 同时给出、或增益超出 0.5–4 时，命令行直接报错。
- 【未验证】真人：拇指抬起、伸平小指、连续捏合都还没有真人测试；拇指在手平放时的抖动（3 mm 噪声下拇指—食指距离标准差 4 mm，官方 1.4 mm）主要来自 10-28 的拇指姿态项，需要真人数据确认是否明显。

测试：dex worker / Shadow dex 单元测试 20 通过（新增：拇指增益只改拇指离掌距离、伸直手指对应伸直、捏合映射连续无跳变、日志字段、增益范围与 worker 一致）；LIBERO + 纯摄像头 33 通过；主线 210 通过、12 跳过、1 个已知失败（`test_dex_sidecar_runtime`）；冻结清单仍只有 1 个已知差异；主入口无头冒烟 19.9 Hz。

## 2026-11-04（续）｜输入端与 20 Hz 仿真对齐：计时 profiling、固定节拍、最新帧、关节插值

目标（用户提出）：仿真控制频率保持 20 Hz 不变，让摄像头 / MediaPipe / dex 子进程 / SpaceMouse 的更新节奏与仿真对齐；不改仿真频率和物理参数，不改冻结旧版，保持机械臂 `[0:6]` / 手 `[6:22]` 写入隔离和 R / Q / Ctrl+C 行为。范围是 robosuite Panda + Allegro integrated 入口（`run_spacemouse_panda_wrist_teleop.py integrated`）。用户提到的 `HAND_ARM_RETARGETING_PROJECT_CONTINUITY.md` 在仓库里不存在，按 AGENTS.md 执行。

第一步，先加计时（[loop_timing.py](../src/wrist_teleop/loop_timing.py) 的 `StageProfiler`，全部用 `perf_counter`）：主循环每个 step 记 `timing_ms`（SpaceMouse 读取、手部结果年龄、手部合成、env.step、抓取 / 激光、三视图渲染、摄像头窗口、日志写入、本轮工作总时长、周期）；摄像头线程记取帧、MediaPipe、手部更新（含 dex 子进程往返）；dex 子进程记 retarget 往返。每 2 秒和退出时各写一条 `timing_summary`（频率、均值 / p50 / p99 / 最大延迟、周期 p99、p99 抖动 = |周期 − 平均周期| 的 p99）。

第二步，测出的瓶颈（改动前，同一台机器）：
- 【已验证·仿真】主循环真正的瓶颈是节拍本身：每轮工作只有 9–10 ms，但旧的 `_sleep_to_rate` 用 `time.monotonic`（Windows 上 15.6 ms 一跳）计时、`time.sleep` 也按 15.6 ms 取整，周期在 47 / 63 ms 两档之间跳，实际 18.5–19.6 Hz，p99 抖动 9.5–14 ms。
- 【已验证·仿真】开三视图窗口时，主线程同步离屏渲染平均 23.8 ms、p99 50 ms、偶发 279 ms，把循环压到 17.9 Hz、p99 抖动 22 ms。
- 【已验证·真实摄像头】摄像头线程受摄像头本身 30 fps 限制：取帧平均阻塞 24 ms（在等下一帧，说明当时没有排队），MediaPipe 10 ms，处理一帧约 10 ms；仿真用到的手部结果平均已有 17 ms、p99 43 ms。
- 【已验证·真实 dex 子进程，合成关键点】dex 往返平均 4.6 ms、p99 8 ms、最大 23 ms。
- 【已验证·真实 HID】SpaceMouse 非阻塞读取 0.05–0.12 ms，不是瓶颈（设备已连接但无人操作，只读到空闲状态）。

第三步，最小改动：
- 固定节拍 `FixedRateClock`：`next_tick += period`，不用"本轮开始 + 周期"，工作时长和睡眠取整都不累积漂移；运行期间 `timeBeginPeriod(1)` 申请 1 ms 定时器精度、退出释放，最后约 1.5 ms 用 `sleep(0)` 让出。迟到不满一个周期时下一拍立即补上、仍回原网格；超过一个周期（重置、首帧渲染等）丢掉错过的拍、从当前时刻重起网格，不连发补帧，次数记在 `overruns_dropped_ticks`。`R` 重置后网格重新起算。
- 摄像头只保留最新帧：同一个 `VideoCapture` 只由新的取帧线程读取，每帧覆盖上一帧（`LatestSlot`）；处理线程（MediaPipe → 适配器 → dex）每次只拿最新一帧，处理慢时旧帧被丢弃而不是在驱动缓冲里排队，丢弃数记在 `frames_dropped_latest_only`。dex 原本就在这个后台线程里；主循环照旧只读最新结果、不等摄像头。LIBERO 的两个入口用的是同一个 `CameraHandWorker`，也一起受益。
- 关节插值 `TargetInterpolator`：只作用于摄像头给出的 16 个 Allegro 关节目标，在按钮手势之前。新增 `HandCommand.sample_timestamp_s`（产生这组目标的帧时间；原来的 `timestamp_s` 是查询时间），帧时间变了才算新样本，从当前输出线性走到新目标，用 `ceil(样本间隔均值 / 50 ms − 0.1)` 拍走完；输入不慢于 20 Hz 时就是直通。无效样本直接透传，保持 / 安全张手仍由原逻辑决定。它只改变摄像头目标落到仿真节拍的时间，不混入任何其他来源，机械臂 `[0:6]` 不经过它。
- 没改：仿真频率、物理参数、OSC 控制器、`composer.compose` 的切片写入、R / Q / Ctrl+C 流程、冻结旧版。三视图渲染仍在主线程（挪到后台线程要改 OpenGL 上下文归属，不属于最小改动）。

前后对比（同样的命令，周期 = 相邻两拍开始时间差）：

| 场景 | 改动前 频率 / 周期 p50 / p99 / 最大 / p99 抖动 | 改动后 |
| --- | --- | --- |
| A 无头、合成输入 `--headless-smoke` 20 s | 19.1 Hz / 47.1 / 63.5 / 64.5 / 11.0 ms | 20.0 Hz / 50.0 / 50.3 / 50.6 / 0.4 ms |
| B 三视图窗口、合成输入 `--viewer-smoke` 20 s | 17.9 Hz / 59.8 / 78.1 / 366 / 22.1 ms | 20.0 Hz / 50.0 / 50.4 / 344 / 1.5 ms（1 次丢拍，在启动首帧渲染） |
| C 无头、真实摄像头 + MediaPipe + dex 子进程 `--dex-smoke` 25 s | 19.6 Hz / 46.4 / 67.4 / 98.0 / 14.1 ms | 20.0 Hz / 50.0 / 50.5 / 73.3 / 5.5 ms |
| D 无头、真实 SpaceMouse HID、手部关闭 15 s | 18.5 Hz / 47.8 / 63.5 / 64.0 / 9.5 ms | 20.0 Hz / 50.0 / 50.2 / 50.3 / 0.2 ms |

C 的摄像头线程（改动后）：取帧约 29 Hz，处理 29.6 Hz，拿到手时最新帧平均只"老" 0.1 ms（p99 0.6 ms），丢帧 0（处理 10 ms 小于 33 ms 帧间隔，所以平时本来就不排队；最新帧机制保证处理变慢时也不排队）；仿真用到的手部结果年龄平均 17 ms、p99 44 ms，和改动前相同，因为它由 30 fps 帧间隔决定。

- 【已验证·仿真】A / B / D 三种情况 20.0 Hz、周期 p99 ≤ 50.5 ms。
- 【已验证·真实摄像头，画面里没有手】C：主循环 20.0 Hz；摄像头线程无排队。因为画面里没有手，dex 子进程在这次运行中没有被调用；dex 往返只用合成关键点单独测过。
- 【已实现未验证】关节插值只在摄像头慢于约 18 Hz 时起作用；本机摄像头 30 fps，A–D 里都是直通，插值行为只有单元测试覆盖。
- 【未验证·实机】真人手在画面里时的 dex 闭环、SpaceMouse 实际推动、真人按 R / Q / Ctrl+C（只有单元测试和代码路径），都还没有真人测试。

测试：新增 `tests/test_loop_timing.py` 15 个（统计、最新帧丢弃计数、固定节拍不漂移 / 短迟到补拍 / 长迟到丢拍、插值直通 / 爬坡 / 不过冲 / 无效透传 / 长间隔不拉大估计）；`test_hand_input.py` 新增 3 个（取帧线程只交最新帧、读帧异常关闭并报错、帧时间与查询时间分开）；`test_integrated_entry_shutdown.py` 新增 1 个（R 重置后 Q 仍正常退出，退出计时摘要含时钟信息）。全量测试中途发现退出计时摘要直接调用 `retargeter.timing_snapshot()`，假求解器没有这个方法时会打断 finally 里的退出日志，已改为安全调用。主线 229 通过、12 跳过、1 个已知失败（`test_dex_sidecar_runtime`，与本次无关）；LIBERO + 纯摄像头 33 通过（Python 3.8）；LIBERO SpaceMouse 入口无头 + 真实摄像头 + dex 12 秒正常运行（画面无手，循环中位数 19.8 Hz）；冻结清单仍只有 1 个已知差异（`src/teleop/teleop_factory.py`）。

## 2026-11-05｜固定延迟取样：手部目标每拍延迟相同（仿真仍 20 Hz）

为什么：上一轮之后，每拍用的是"最新算完的那一帧"。摄像头 30 帧模式下帧间隔只有 32 / 48 ms 两种（驱动按 16 ms 格子交付，开 1 ms 系统计时器也一样），和 50 ms 的仿真节拍互相错开，所以每拍用到的手"老" 0–46 ms，每拍不一样：同样匀速的手，在仿真里每拍有时走 1 帧、有时走 2 帧。把仿真改成 49 ms 不能解决（结果本来约 33 ms 出一次；两块表会漂；LIBERO 只接受整数 Hz；导师要求仿真频率不变），所以改为让真实世界这边配合 20 Hz。

改了什么：
- [loop_timing.py](../src/wrist_teleop/loop_timing.py) 新增 `FixedDelayResampler`，替换掉上一轮的 `TargetInterpolator`。每拍取"本拍时刻 − D"那一刻的手，用这一刻前后两帧按取图时刻线性插值；下一帧还没到时保持最新一帧，不外推；输出只会是相邻两帧摄像头结果的混合。
- [hand_input.py](../src/wrist_teleop/hand_input.py)：`CameraHandWorker` 把每一帧被适配器接受的结果连同取图时刻（`perf_counter`）存下来，`samples_since` 给主循环取；新增计时项 `result_latency`（取图到结果可用）。被拒绝或保持的帧不算新样本。
- D 集中在配置 `hand_target_delay_s`（默认 0.065 s，`configs/spacemouse_wrist.json`），两个入口都可以用 `--hand-delay-ms` 临时改。
- 接入：Panda + Allegro integrated 入口、LIBERO SpaceMouse + Shadow 入口。有效 / 保持 / 安全张手的判断不延迟；`R` 重置（以及 Allegro dex 的 `C`）清空并丢掉之前读到的帧。纯摄像头 LIBERO 入口没有接：那里机械臂也来自同一帧摄像头，只延迟手指会让手和臂错开。
- 没改：仿真频率、物理参数、OSC、动作 `[0:6]` / `[6:22]` 的写入、R / Q / Ctrl+C 流程、冻结旧版。

D 怎么定的（【已验证·真实摄像头，画面无手，curl-pinch，20 秒 600 帧】）：取图到结果可用平均 10.0 ms、p99 11.8 ms；每拍"要插值至少需要的 D"中位数 27 ms、p90 40 ms、p99 55 ms。D = 50 / 55 / 60 / 65 / 70 ms 时，需要保持最新帧的拍分别占 2.4% / 1.0% / 0.16% / 0.13% / 0.11%。dex 每帧还要多 5–8 ms，所以默认取 65 ms。代价：手部平均延迟从约 27 ms（从取图算起）变成固定 65 ms，多约 38 ms。

结果：
- 【已验证·真实摄像头 + 真实线程 + 20 Hz 节拍，手的数值是合成的】让适配器把每一帧都当成"匀速移动的手"（数值 = 帧序号），20 秒 400 拍：改前每拍前进 1 帧或 2 帧各约一半（标准差 0.50）；现在大多数拍前进 1.56 帧（标准差 0.14），400 拍里 1 拍保持最新帧。剩下的不均匀来自驱动的 16 ms 交付格子：帧的时间戳本身有最多约 16 ms 的误差。
- 【已验证·仿真】`--headless-smoke`、`--dex-smoke`（真实摄像头，画面无手）、LIBERO `--headless-smoke`（hand off / legacy-curl + 真实摄像头 + `--hand-delay-ms 60`）都正常跑完；日志里有 `resampling` 字段。画面无手时没有样本，按设计直接用适配器结果。
- 【已知·既有】`--dex-smoke` 启动后约 2.6 秒（摄像头和 MediaPipe 初始化）有一次 20–39 ms 的迟到唤醒或 70 ms 的 env.step，改动前的日志（11-05 04:10）同样有；之后每拍迟到 ≤ 0.7 ms。
- 【未验证·实机】真人手在画面里时的效果（curl-pinch / dex）、手感上延迟是否可以接受，都还没有真人测试。

测试：`tests/test_loop_timing.py` 用 9 个固定延迟测试替换了 6 个插值测试（按实测 32 / 48 ms 帧间隔模拟时每拍恰好是 D 之前的手、拍间隔均匀；对照"取最新帧"每拍年龄相差 > 30 ms；晚帧保持不外推；第一帧不够老时用最早一帧；输出夹在相邻两帧之间；D = 0；重置丢帧；乱序 / 重复帧忽略；参数检查）；`test_hand_input.py` 新增 1 个（只有被接受的帧才是样本）。主线 212 通过、1 个已知失败（`test_dex_sidecar_runtime`，与本次无关）；LIBERO 环境（Python 3.8）69 通过；冻结清单 172/173，唯一差异仍是 `src/teleop/teleop_factory.py`。

## 2026-11-05（续）｜Shadow 捏合：机器人侧改用指肚点，人手侧间隙按实测 2.2 cm

起因（用户真人测试截图）：LIBERO libero_spatial 0、Panda + Shadow、dex 映射下比 OK 手势，人手拇指和食指已经捏上，Shadow 的拇指和食指之间还留着缝。用户判断：摄像头识别点在指肚，仿真映射点在 Shadow 指尖，要求把 Shadow 这边的点挪到指肚，或调整距离。

查到的情况（改代码前）：
- 【已验证·模型几何】DexPilot 用的 Shadow `fftip` / `thtip` 等指尖坐标系在远节指骨里面：指腹表面在它掌侧约 5 mm 处（MuJoCo 碰撞用远节网格的凸包）。所以两个指尖坐标系重合 = 两个指肚互相压进去约 1 cm；指肚刚贴上时，两个指尖坐标系相距约 1 cm。
- 【已验证·真人日志】MediaPipe 的 4 / 8 号点在指尖端，不在指肚上；但手指有厚度，人手指肚贴紧时两个指尖关键点之间的 3-D 距离实测约 2.2 cm（140843Z 捏合帧中位数 2.16 cm；142042Z 最近的 1% 帧 2.20 cm、干净捏合段 2.3–2.45 cm）。原来按 1 cm 当"贴上"，偏小。

改了什么（只在默认 dex 映射里；`--dex-official` 仍是原版）：
- [dex_retargeting_worker.py](../scripts/dex_retargeting_worker.py) 新增 `pinch_points`（"tips" / "pads"，默认 "tips" = 原行为）。"pads" 时在求解器的 Pinocchio 模型上给五个远节指骨各加一个指肚点（固定在远节上，四指 z = 22 mm、y = −6.1 mm，拇指 z = 24 mm、y = −7.0 mm），拇指—四指这 4 条 DexPilot 向量改成指肚到指肚；腕到指尖、指尖之间的向量和每指长度缩放仍用原来的指尖坐标系。距离 0 = 两个指肚表面贴上，和人手捏合的含义一致。
- [shadow_grasp_terms.json](../configs/dex_retargeting/shadow_grasp_terms.json) 升到 v3：`pinch_points: "pads"`，`pinch_gap_m` 0.01 → 0.022（人手指肚贴紧时的指尖关键点距离，取自上面的实测）。机器人指肚距离 = 1.2 × max(0, 人手距离 − 2.2 cm)，仍是连续映射、无阈值、无吸附，只读当前帧的人手关键点。
- [shadow_dex.py](../src/wrist_teleop/shadow_dex.py)：元数据的 algorithm 字段写明 `continuous pinch between pads`。

验证：
- 【已验证·模型几何】指肚点就在 MuJoCo 远节碰撞面上：从掌侧 1 cm 外朝指肚打射线，命中点与指肚点相差拇指 0.06 mm、四指 0.01 mm（新单元测试）。
- 【已验证·日志回放，真实 dex 子进程 + MuJoCo 正运动学】把用户两段真人会话的 21 点关键点按顺序重新送入 dex 子进程，用 Shadow MJCF 摆出姿态，量拇指远节和各指远节碰撞体之间的有符号距离（只算运动学，负值 = 两个碰撞体重叠，真实仿真里会变成接触后互相顶住）。比较了 tips / 1 cm（原来）和 pads 搭配 1.0 / 1.8 / 2.0 / 2.2 / 2.4 cm：
  - 140843Z（干净的拇指—食指捏合，拇指增益 1.5）：人手 2.0–2.6 cm 的帧，原来就已经碰上（−2.7 ~ −3.3 mm），pads + 2.2 cm 变成 −6.2 ~ −7.1 mm（捏得更紧）；人手 2.6–3.0 / 3.0–4.0 cm 时机器人间隙 11.1 / 10.7 mm → 5.9 / 7.0 mm；张手（≥ 6 cm）57.5 → 57.0 mm，不变；人手 ≥ 3 cm 时机器人却碰上的帧 9 → 13（共 2354 帧）。
  - 142042Z（截图那一段，拇指增益 1.8）：截图里的 OK 手势是会话最后约 80 帧。这些帧里摄像头 3-D 关键点给出的拇指尖—食指尖距离中位数 3.84 cm（画面里两点只隔约 1.7 cm），机器人拇指—食指间隙中位数 8.9 mm → 7.8 mm，仍然没碰上。同一会话里干净的捏合段（1846–1858 帧，人手 2.3–2.45 cm）原来和现在都能碰上。616–700 帧摄像头把中指也测成离拇指 1.6–2.3 cm（中指伸直度 0.80，平时 0.94），求解器把拇指送进中指，食指差 6–8 mm，原来和现在一样。
- 结论：指肚点 + 2.2 cm 让"贴上"的含义对齐了，正常捏合会贴得更实，2.6–4 cm 区间机器人整体靠近约 4–5 mm；但截图里 OK 手势没碰上的主要原因不是指尖 / 指肚，而是那一刻 MediaPipe 的 3-D 深度把两个指尖拉开到约 3.8 cm。把间隙直接调到 3.8 cm 会让人手实际相隔 1.5–2 cm 的很多帧也合上（3–4 cm 区间每段会话有 370–490 帧），所以这次没有这样做。
- 【已验证·仿真】LIBERO SpaceMouse + Shadow 入口无头 + 真实摄像头 + dex 子进程 20 s：子进程以 `pads` / 0.022 启动，循环中位数 19.8 Hz（画面无手）。
- 【未验证·实机】真人再测：干净捏合、OK 手势、三指捏是否贴上，以及捏紧时会不会把物体挤飞。

测试：`test_dex_sidecar_worker.py` 的设置校验加上 `pinch_points`（项目配置为 pads、默认仍是 tips、非法值报错）；`test_shadow_dex.py` 新增指肚点落在 MuJoCo 碰撞面上的测试。主线 234 通过、12 跳过、1 个已知失败（`test_dex_sidecar_runtime`，与本次无关）；LIBERO 环境 54 通过；冻结清单 172/173，唯一差异仍是 `src/teleop/teleop_factory.py`。

## 2026-11-05（续）｜LIBERO SpaceMouse + Shadow 入口改用固定节拍

为什么：11-04 的固定节拍只改在 Panda + Allegro 主入口（当时任务点名的文件）；11-05 的固定延迟取样接进了 LIBERO SpaceMouse 入口，但它的节拍还是"干完活再睡掉本拍剩余时间"（`_sleep`），某一拍晚了，后面每一拍都跟着往后推。日志：用户 11-05 真人会话 142042Z（开三视图）平均 19.46 Hz，周期 p99 68.6 ms、最慢 100 ms，10.4% 的拍晚 2 ms 以上；无头冒烟 19.79 Hz。固定延迟取样要求拍点均匀，拍点乱跳会把取样时刻一起带乱。

改了什么：[libero_spatial_teleop.py](../src/wrist_teleop/libero_spatial_teleop.py) 的主循环改用主入口同一个 `FixedRateClock`（`next_tick += 50 ms`，短迟到下一拍追回，超过一整拍就丢拍重排，不连发）；`R` 重置后重新排拍；1 ms 系统定时器仍由原来的 `_WindowsTimerResolution` 持有；退出事件新增 `clock`（周期、拍数、丢拍数）。纯摄像头 LIBERO 入口没改，仍用 `_sleep`。没改：仿真频率、物理参数、动作写入、R / Q / Ctrl+C。

验证：按用户要求没有做改动前后对比。【已验证·仿真】LIBERO 单元测试 5 通过；无头 3 秒冒烟（手部关闭）正常结束，循环中位数 20.0 Hz，44 拍里启动时丢 2 拍。【未验证·实机】开三视图、真人手在画面里的实际频率和抖动。

## 2026-11-05（续）｜机械臂旋转漂移（OSC 位置 / 姿态耦合）与 Allegro dex 手势不跟手

用户真人测试（录屏 + 日志 `20261105T170317Z_integrated`、LIBERO `20261105T171014Z_panda_shadow_teleop`）：
- integrated `--hand-control dex`：SpaceMouse 左右旋转、左右偏时机械臂有较大漂移，其余还好；dex 手部不能正确反映姿势，curl-pinch 正常。
- LIBERO Shadow：左右偏航略有轻微漂移。
- 用户决定：纯摄像头版本这条线先不做。

**机械臂漂移：查到的原因**
- 不是输入问题：日志里松手后每拍命令全是 0，纯旋转时平移分量也是 0。
- 用户的真实 SpaceMouse 输入在无头环境里按原样重放（同一控制器、同一保护逻辑），逐层排查：保护绳（leash）、松手锁定（latch）、原版 desired 模式都不是根因。根因是 robosuite 默认 Panda 控制器配置 `uncouple_pos_ori: true`：位置和姿态的任务空间惯量分开算，纯旋转产生的关节力矩在手端带出横向加速度，手被推开，松手锁定把这个偏移当成新的目标，之后还慢慢爬。加上 6 号关节撞上限位（3.752 rad）、6 / 7 号关节力矩饱和（±12 Nm），会更明显。

**机械臂：改了什么**（只改这一个控制器参数；增益、阻尼、仿真频率、物理参数、松手锁定和保护绳都没改）
- [allegro_model.py](../src/wrist_teleop/allegro_model.py) `make_allegro_panda_env`：机械臂控制器加 `uncouple_pos_ori=False`，用完整的 6 维任务空间惯量。
- [libero_spatial_teleop.py](../src/wrist_teleop/libero_spatial_teleop.py) `_controller`：同样加 `uncouple_pos_ori=False`。纯摄像头 LIBERO 入口共用这个函数，所以它也带上了这个改动（那条线本次没做其它改动，也没测）。

**机械臂：结果**
- 【已验证·回放用户真实输入】`20261105T170317Z` 共 2861 拍：

  | 指标 | 改前 | 改后 |
  | --- | --- | --- |
  | 纯旋转段手的位置偏移（中位数 / 最大） | 28.4 / 101.2 mm | 1.7 / 8.5 mm |
  | 松手后 2 秒内的移动（最大） | 20.0 mm / 2.73° | 0.5 mm / 0.07° |
  | 6 号关节贴限位的拍数 | 103 | 0 |

  回放是开环的（用户当时看着画面纠正，重放时不会），所以轨迹和原日志不完全一样；改后这次回放中 5 号关节有几拍到了限位附近。
- 【已验证·合成探针】绕世界 z、视线深度方向、屏幕右方向各正负旋转 30 拍（0.5 rad/s）后松手 2 秒：
  - robosuite：旋转中位置偏移 ≤ 1.8 mm，松手后 ≤ 0.2 mm / 0.08°，最后 0.5 秒 ≤ 0.02 mm；
  - LIBERO：旋转中 ≤ 0.1 mm，松手后 0.0 mm / 0.01°。改前 LIBERO 松手 2 秒后每 0.5 秒还在爬约 4 mm / 0.4°。
- 【未验证·实机】真实 SpaceMouse 的手感（旋转时手还会不会跑、跟手程度）。

**dex 手部：查到的原因**（三个叠在一起）
1. 镜像：摄像头画面默认先水平翻转再给 MediaPipe（自拍视角），翻转后的 3-D 世界关键点是镜像的，形状是左手、标签仍是 Right。右手求解器拿到左手形状，手指往反方向弯，卡在一个弯曲姿态不动。Shadow 路径早就有 `unmirror_world_x`，Allegro dex 路径一直没有。
2. 缩放：Allegro dex 入口给求解器的人手关键点乘了 2/3（`ALLEGRO_GEOMETRY_SCALE`，MuJoCo 里 Allegro 是 2/3 尺寸模型）。但求解器用的是原尺寸 Allegro URDF，关节角和模型尺寸无关，缩小输入只会让手指看起来一直半弯。
3. 相对模式：`relative_motion: true` 把第一帧当中立位，之后输出"相对第一帧的变化"。前两条让求解器输出几乎不随人手变化，减掉第一帧后就基本不动。

【已验证·日志回放，真实 dex 子进程】把会话里 344 个不同的手部关键点帧重新送入子进程，三根手指 MCP + PIP + DIP 弯曲角之和取平均：

| | 人手 | 改前（镜像 + 2/3） | 改后（去镜像 + 原尺寸） |
| --- | --- | --- | --- |
| 最张开的 20% 帧 | 51° | 272° | 30° |
| 最握紧的 20% 帧 | 259° | 271° | 177° |
| 与人手的相关系数 | | 0.15 | 0.88 |

另外验证了"世界坐标 x 取反"和回放里用的"MANO 坐标 x 取反"在 200 个随机手上结果完全一样（差 0.0），所以回放结论直接适用于新代码。

**dex 手部：改了什么**
- [hand_input.py](../src/wrist_teleop/hand_input.py) `DexHandInputAdapter`：
  - 新增 `unmirror_world_x`：进入 MANO 转换前把世界坐标 x 取反，和 `ShadowDexHandInput` 相同；每帧 details 记录这个开关；
  - `relative_motion` 默认改为 False。
- [run_spacemouse_panda_wrist_teleop.py](../scripts/run_spacemouse_panda_wrist_teleop.py) `_make_hand_input`：
  - 画面镜像时（默认）打开 `unmirror_world_x`，`--camera-no-mirror` 时关闭；
  - 去掉 `reference_landmark_scale=ALLEGRO_GEOMETRY_SCALE`（回到默认 1.0）。
- [config.py](../src/wrist_teleop/config.py) `DexHandConfig.relative_motion` 默认改为 False；[spacemouse_wrist.json](../configs/spacemouse_wrist.json) 同步改为 `false`（仓库约定配置文件与默认值一致）。
- 没改：curl-pinch（用户确认正常）、MANO 转换本身、官方 dex 配置、Shadow 路径、手部 `[6:22]` / 机械臂 `[0:6]` 写入隔离、R / Q / Ctrl+C。

**已知限制**
- 官方 Allegro 配置只用腕到指尖 4 条向量，握拳时主要弯 PIP / DIP（回放里握紧比张开多 70–83°），MCP 基本不动（−19° ~ +4°）。这是官方 vector 配置本身的特点，本次没改。
- 绝对模式下 `C`（重新取中立位）不起作用，画面提示行仍显示 "C re-centre dex"。

**测试**
- 新增 5 个：
  - `test_integrated_composer.py`：Panda + Allegro 纯旋转三个轴 × 正负，检查控制器为耦合、位置偏移 < 5 mm、松手后 < 1 mm、最后 0.5 s < 0.1 mm / 0.05°；
  - `test_libero_spatial_teleop.py`：LIBERO 偏航同样的检查；
  - `test_dex_retargeting.py`：翻转后的关键点经去镜像与原图一致、不去镜像则不同；仓库配置为绝对模式；
  - `test_integrated_entry_shutdown.py`：入口 dex 接线（镜像 ↔ 去镜像、不再传 2/3 缩放、绝对模式）。
- 结果：
  - 主线 239 通过、13 跳过、0 失败。之前一直失败的 `test_dex_sidecar_runtime` 开 / 合手测试现在通过：它没改，原来失败就是因为相对模式在两帧之间重置后两次都输出中立位；
  - LIBERO 环境（Python 3.8）的 LIBERO / Shadow 测试 27 通过；
  - integrated `--headless --headless-smoke --duration 20`：20.0 Hz、0 次丢拍、正常退出；
  - 冻结清单 172/173，与之前相同。
- 【未验证·实机】真人手在 dex 模式下的效果、SpaceMouse 旋转的实际手感；`--dex-smoke` 本次没跑（需要打开摄像头）。

## 2026-11-05（续）｜LIBERO 单个物体质量参数 `--object-mass`、fork 改动核对、真机试跑

**目的**
- 用户希望在命令里直接改单个物体的质量。之前只有整体倍数 `--object-mass-scale`。
- 同时逐条核对 fork 会话（机械臂旋转漂移 + Allegro dex）的改动，并用真机跑一次。

**改动**
- [libero_spatial_teleop.py](../src/wrist_teleop/libero_spatial_teleop.py)：
  - `parse_object_mass_overrides`：把 `NAME=GRAMS` 解析成 kg，范围 1–20000 g；缺 `=`、不是数字、nan、越界都报错；
  - `scale_object_masses` 新增 `overrides_kg`：
    - 名字统一转小写、只留字母和数字；先按实例名匹配（`butter_2`），再按去掉末尾 `_数字` 的类名匹配（`AkitaBlackBowl` / `akita_black_bowl`）；
    - 点名的物体质量设为给定值，惯量按同一倍数缩放；没点名的物体仍按 `--object-mass-scale`；
    - 有名字匹配不到本任务的任何物体时报错，报错信息里列出本任务的物体；
    - 每次 reset 都从官方质量重新计算，不会叠加；
  - `OfficialLiberoSpatialSession` / `run_libero_spatial_teleop` 新增 `object_masses_kg`；`describe()` 记录 `object_mass_overrides_g`；
  - 启动时打印的质量里，点名设置的物体带 `*`。
- [run_libero_spatial_panda_shadow_teleop.py](../scripts/run_libero_spatial_panda_shadow_teleop.py)：新增 `--object-mass NAME=GRAMS ...`（一次可写多个，也可以重复写这个参数）。
- [RUN_COMMANDS.md](../RUN_COMMANDS.md)：后缀表加一行，组合示例改成带 `--object-mass` 的写法，并注明 `--` 后面不能有空格。
- 纯摄像头 LIBERO 入口没有加这个参数。
- 没改：仿真频率、timestep、重力、摩擦等仿真参数、冻结 baseline、arm `[0:6]` / hand `[6:22]` 写入隔离、R / Q / Ctrl+C。不加参数时物体质量和官方一致。

**fork 改动逐条核对**
- OSC `uncouple_pos_ori=False`（[allegro_model.py](../src/wrist_teleop/allegro_model.py) 和 LIBERO `_controller`）：
  - 本机 robosuite 1.5.2（`controllers/parts/arm/osc.py`）和 1.4（`controllers/osc.py`）都接受这个参数，并设到 `self.uncoupling`；
  - 它是控制器参数，不涉及仿真频率或物理参数；
  - `_controller` 是共用的，所以纯摄像头 LIBERO 入口也一起换成了耦合 OSC，那条入口没有单独测过。
- Allegro dex：`unmirror_world_x`（镜像画面时打开）、关键点回到原尺寸（`reference_landmark_scale` 默认 1.0）、`relative_motion` 默认 False。config.py、spacemouse_wrist.json、入口、测试、文档彼此一致。
- 遗留：绝对模式下 `C` 不起作用，提示行仍显示 "C re-centre dex"。fork 自己已列出，本次没改。

**测试**
- 新增 3 个（[test_libero_views_and_options.py](../tests/test_libero_views_and_options.py)）：
  - 解析：正确写法，以及缺 `=`、缺名字、非数字、0、超上限、nan；
  - libero_object 任务 0、整体 ×2、`AlphabetSoup=400 g`、`basket_1=500 g`：点名的等于给定值，其余 ×2；reset 后不叠加；惯量与官方相比是同一倍数；
  - 名字不在任务里报错（错误信息带本任务物体名）；0 g 在构造时被拒。
- 结果：
  - LIBERO 环境 58 通过；
  - 主线 239 通过、16 跳过、0 失败；
  - 冻结清单 172/173，与之前相同。

**真机试跑**
- SpaceMouse 诊断 5 s（`outputs/spacemouse_wrist/20261105T180108Z_diagnostic`）：
  - 识别到设备，6 轴都有读数（rx 到 350 满量程，rz 到 293）；
  - 映射后的旋转增量被限幅到每步约 0.0145 / 0.0204 rad，松开后回到 0；
  - 记录里 `buttons_valid` 为 false。
- LIBERO 窗口 40 s（`outputs/libero/20261105T180311Z_panda_shadow_down_teleop`）：
  - 命令：`--thumb-lift-gain 2.0 --hand-mount down --suite libero_90 --task-id 1 --object-mass-scale 10 --object-mass AkitaBlackBowl=300 butter_2=110 --duration 40`；
  - 质量生效：`akita_black_bowl_1* 300 g`、`butter_2* 110 g`，其余 ×10（`butter_1` 52 g、`chocolate_pudding_1` 102 g）；
  - 固定节拍：775 拍，平均周期 51.64 ms（19.37 Hz），p50 50.0 ms、p99 59.0 ms，最大一次 739 ms，丢拍 6 次；
  - `env.step` 平均 39.4 ms、p99 56.2 ms：libero_90 这个场景加手指朝下时，一次仿真步就接近 50 ms 的预算，是超时的主要来源；三视图 2.5 ms；
  - 这 40 s 内没人操作：SpaceMouse 有效 0%、按键全为 false；摄像头一直有帧，但没检测到手（`lost_hold`，无错误），Shadow 保持安全张开；任务未完成。
- 【未验证·实机】真人用 SpaceMouse + 手在 LIBERO 里完成任务；fork 的旋转漂移修复和 Allegro dex 修复的真实手感。

## 2026-11-06｜LIBERO 规划器版：仿真慢放、数字键接近目标（OMPL）、指尖射线

**目的**：本周老师布置了三件事，做在一个新入口里，原 SpaceMouse 入口和主循环一行没改。
1. 放慢仿真（slow down the simulation）。
2. 用运动规划器把手送到物体附近。实验室认为这不算共享控制：同一时刻只有一方控制机械臂，不混合输入。
3. 把掌心红线换成每根手指一条射线。

**讨论后定下的做法**
- **慢放 `--sim-speed`（1 / 0.8 / 0.6 / 0.4）：**
  - 现实节拍仍是 20 Hz，画面和读手也还是 20 Hz；
  - 每拍仿真只走 `s × 50 ms`，即 robosuite `control_freq = 20 / s`，2 ms 物理步长不变；
  - SpaceMouse 每拍的增量同步乘 s，机械臂在仿真时间里的速度不变；
  - 取值要让每拍的仿真时长是 2 ms 的整数倍，所以没有 0.5。
- **数字键 1–9：** 编号自动从任务里读，先列物体、再列固定装置，按 BDDL 的顺序。130 个任务里物体加固定装置最多 9 个，数字键够用。
- **对准用的点：** "抓取中心"，即拇指尖与食指、中指两个指尖平均位置之间的中点。二指、三指、五指抓都会用到拇指和食指，所以三种抓法都适用；它在按键那一刻按当前手型计算。
  - 指尖取远节指骨碰撞网格最远的那一端。
- **到位位置：**
  - 水平方向：抓取中心对准目标碰撞外形包围盒的中心；
  - 高度：最低指尖比目标顶面高 2 cm；
  - 手腕朝向不变，转角由人来调（用户选的）；
  - 手里拿着东西时（被至少两个手部部件接触的物体），改成让该物体的中心对准目标，物体底部高出目标顶面 2 cm。
- **规划器：** OMPL 2.0.2（conda-forge Windows 版），装在新建的 Conda 环境 `ompl_planner` 里（Python 3.11、MuJoCo 3.2.3，和 LIBERO 环境是同一个 MuJoCo 版本）。
  - 用法和 dex 一样，作为子进程运行：[ompl_planner_worker.py](../scripts/ompl_planner_worker.py)。
  - 主程序用 `mj_saveModel` 把当前场景存成 MJB 文件交给它，按 R 复位后重新存一份。
  - 子进程先用阻尼最小二乘做逆运动学（多个起点，要求解无碰撞），再在 7 个关节空间里用 RRT-Connect 搜路径，然后简化路径。
  - 碰撞检测用 MuJoCo 的 `mj_collision`：机器人（加手里的物体）和场景之间穿透超过 3 mm 算碰撞；起始时已经存在的接触整段允许；不检查机器人自身碰撞。
  - 返回的关节路径用正运动学换算成末端路点，主循环用最小加加速度时间律逐拍交给 OSC 去跟：峰值 0.15 m/s、0.8 rad/s（按仿真时间计）。
  - 规划在后台线程里跑，主循环照常走拍。
- **对照用的 `--planner straight`：** 先抬高（越过路线下方最高的东西再加 5 cm），再平移，最后下降。不做碰撞检测。
- **交接：**
  - 规划或移动过程中，SpaceMouse 只要有输入（过死区之后）就立刻取消，期望位姿设为当前位姿，当拍就交给 SpaceMouse；`0` 键同样取消；
  - 到位的判断：离终点 8 mm 和 0.05 rad 以内，或终点之后 1.5 s（仿真时间）。到位后同样交回 SpaceMouse；
  - 手指一直由摄像头控制。
- **指尖射线：** 五根，竖直向下（世界坐标 −z），碰到第一个非机器人表面就停，红色，三个视图都画；`L` 开关；新入口不画掌心射线。
  - 射线和编号都是渲染时临时加进场景的几何体，不改模型、不影响物理。

**新增文件**（原入口和共用模块都没改）
- [libero_motion_planning.py](../src/wrist_teleop/libero_motion_planning.py)：几何计算（指尖、抓取中心、包围盒、接近目标）、三段直线、最小加加速度路径、规划状态机、OMPL 子进程客户端。
- [libero_planner_teleop.py](../src/wrist_teleop/libero_planner_teleop.py)：由原主循环复制后修改：慢放、数字键、交接、射线和编号的绘制。视图窗口用子类重写了 `_capture`。
- [run_libero_panda_shadow_planner_teleop.py](../scripts/run_libero_panda_shadow_planner_teleop.py)：新入口，`--suite` 只能选 libero_spatial / libero_object，默认手指朝下。
- [ompl_planner_worker.py](../scripts/ompl_planner_worker.py)：OMPL 子进程。
- [test_libero_planner_teleop.py](../tests/test_libero_planner_teleop.py)：13 个测试。
- 运行日志文件夹名为 `*_panda_shadow_down_planner_teleop`，日志里有 `planner_request` / `planner_result` / `planner_cancel` / `planner_failed` / `planner_arrived` 事件，每拍记录 `arm_owner`（planner / spacemouse）。

**测试**
- 新测试 13 个，在 LIBERO 环境里全部通过：
  - 几何：抓取中心、接近目标、手里有物体时的接近目标、三段直线的抬高高度；
  - 最小加加速度路径：起点和终点速度为 0，速度不超限；
  - 状态机：到位后交回；规划中被取消后，晚到的结果被丢弃；
  - 慢放：每拍的仿真时长和每拍增量的缩放；
  - 场景：编号顺序、指尖在远节关节之外、射线竖直落到桌面、`L` 关闭；
  - 三段直线到位后，抓取中心离物体中心 < 5 mm，最低指尖离物体顶面 2 cm ± 5 mm；
  - 移动中途取消：滑行 < 2 cm 后停住；
  - OMPL：libero_spatial 任务 0 去盘子上方，执行过程中机器人和场景之间没有超过 3 mm 的穿透，并且到位。
- 无窗口冒烟测试（SpaceMouse 输入为全零，没有摄像头）：
  - libero_object 任务 0，目标 1：三段直线到位误差 0.2 mm；OMPL 规划共 737 ms（逆运动学 361 ms，搜索 30 ms），到位误差 0.2 mm；
  - libero_spatial 任务 0，目标 5（盘子），`--sim-speed 0.6`：每拍 15 个物理步（30 ms），现实 15.0 s 里仿真走了 8.64 s，规划 338 ms，到位误差 0.1 mm。
- 手里有物体时的规划：把罐头当作已经抓住，规划到篮子上方，成功（48 个路点，359 ms）。用预设手势从上往下没能真正抓起物体（四指捏、拇指食指捏都试了），所以"真实抓起后再搬运"这一整段没有在仿真里跑通过，只验证了规划这一步。
- 截图确认：三个视图里都画出了五根射线和黄色编号。编号文字偏小。
- 回归：LIBERO 环境 71 个测试全部通过（新测试 + 原 LIBERO / 摄像头 / Shadow / dex 测试）；主线环境 246 通过、21 跳过、1 失败——`test_loop_timing` 的 30 ms 睡眠计时用例，两套测试同时跑时机器负载高导致超时，单独重跑 5 次全部通过，和本次改动无关；冻结 baseline 哈希 172/173 不变。
- 【未验证·实机】真人操作 SpaceMouse 打断规划、到位后微调、用摄像头手抓取、搬运放置，以及慢放的手感。

## 2026-11-06（续）｜规划器版实机后的修改：先查直线路线、留安全距离、提速、小编号、M 键换构型

**实机反馈**（当晚第一次实机。日志 `outputs/libero/20261107T011100Z`、`011642Z`、`013035Z` 用 OMPL，`012424Z` 用 straight；文件夹时间戳是 UTC）
1. 很多次规划失败；
2. 拿着东西去篮子时，手有时整个转一大圈；
3. 去黄油的路上碰倒了牛奶；
4. 手指朝下的初始构型不好用：手腕能转的范围小，一转手腕很多关节一起快速动、不稳；手掌平行桌面会更好；
5. 机械臂太慢，sim speed 1 也慢；
6. 物体编号和名字太大。

**日志里的原因**
- OMPL 共 72 次请求，37 次失败：
  - 报"(1 checks)"的：起点被判为无效，只检查了 1 个状态就放弃。起始接触本来就允许，所以只可能是关节限位：旧代码要求每个关节离限位至少 0.01 rad，而操作者转过手腕后，有关节停在了限位上。
  - 报"no collision-free IK (24 seeds, 0 in collision)"的：保持当前手腕朝向时，目标位置解不出逆运动学，24 个起点全部失败，和碰撞无关。
- 手转一大圈：当前关节角附近的起点解不出逆运动学时（第 13–22 个起点才成功），用的是随机起点，解落在另一种手臂构型上，中间的关节路径就转得很大（6–8 rad），例如"0.52 m、仿真 14.9 s"那一次。
- 碰倒牛奶：旧版没有安全距离，只允许 3 mm 以内的穿透；OSC 又只管末端位姿，手臂其他部分不受约束。
- 日志里所有"SpaceMouse 接管"都是真的有输入（连续 7–20 拍非零），不是误触发。

**构型分析**（libero_object 任务 0，从复位时的关节角出发做局部逆运动学，桌面上方 90 个"掌心朝下"的位姿）

| 构型 | 能到的位姿（共 90） | 关节离限位的中位数 | 其他 |
| --- | --- | --- | --- |
| down（手指朝下） | 16–68 | 0–0.1 rad | 从复位姿态要转 96–124° 才能掌心朝下 |
| forward（手掌平行桌面） | 71–84 | 0.57–0.73 rad | 复位时已经是掌心朝下（只差 5.8°） |

手指朝下时，想让掌心朝下就得大幅转手腕，转完关节紧贴限位，这就是"一转手腕很多关节快速晃动"的原因。

**改动**（只改规划器版的新文件；原入口、共用模块、冻结 baseline 都没动）
- 规划顺序（`--planner ompl`）：
  1. 先试几条直线路线，依次是：目标在 5 cm 以内时直接连线；"抬高 → 平移 → 下降"，平移高度比路线下方最高的物体高 5 / 10 / 16 cm；"斜线直接到目标上方 → 下降"，同样三个高度。
  2. 每条路线每 1 cm 检查一次。手的朝向固定；逆运动学从上一步的关节角接着解，任何关节一步变化超过 0.25 rad 就算"构型跳变"、放弃这条路线，所以手臂不会换构型，手也不会翻。整个机器人（加上手里的物体）离场景至少 1.5 cm；目标正上方半径 2 cm 的柱子里放宽到 0.5 cm；起始时已经很近的部件，只要求不再靠近超过 2 mm。最后竖直下降时如果会碰到下面的东西，就停在上方，最多高 4 cm。
  3. 所有直线都不行，才用 OMPL RRT-Connect。目标的逆运动学只在当前关节角附近找（12 个起点，σ = 0.15 rad），取离当前构型最近的解；整条路径上手的朝向和目标朝向相差不超过 0.5 rad（约 29°）；搜索时间 2 s。
  4. 起点先夹回关节范围以内，不会再因为关节压在限位上而整次失败。
- 安全距离：给机器人部件设 2 cm 的接触余量（geom margin）后，`mj_collision` 给出带符号距离，和 `mj_geomDistance` 的结果一致。
- 直线路线按段执行：每段一个最小加加速度时间律，在拐角处停一下，手沿着检查过的直线走，不切角；总时间和一个时间律走完全程相同。实测 0.30 m/s 时，末端离规划直线最远 4–7 mm，小于 1.5 cm 的安全距离。
- 规划器速度：峰值 0.15 → 0.30 m/s，角速度按同样比例提到 1.6 rad/s；新后缀 `--planner-speed`（0.05–0.5 m/s）。去篮子的那一段从 5.3 s 缩短到 2.7 s。
- 对照组 `--planner straight` 改用同样的分段时间律，只是什么都不检查，只走第一条"抬高 → 平移 → 下降"。两者的差别只剩：检查与否、换路线、OMPL 兜底。
- 编号：去掉名字，只在主视图和 3D 视图里画小的黄色圆形编号（用渲染相机的参数把物体顶面中心投影到图像上，再用 cv2 画，和渲染位置差 1–3 px）。名字只在启动时打印在控制台。
- 新按键 `M`：手指朝下 ↔ 手掌平行桌面。会关掉当前环境、用另一种构型重建，任务回到初始状态（约 4 s）。状态栏显示当前构型，每拍日志记 `hand_mount`，切换时记 `hand_mount_switch` 事件。没有用 `C`：摄像头小窗里共用模块固定印着"C re-centre dex"。
- 交给子进程的 MJB 场景文件（每个约 250 MB）改放在系统临时目录 `libero_planner_scenes` 里：每次重存都删掉上一个，退出时删除，下次启动时清掉 12 小时以前残留的。之前的版本存在每次运行的日志文件夹里（`outputs/libero/*/planner_scene_*.mjb`，12 个，约 3.1 GB；旧版测试还在 `outputs/test_planner` 留了 2 个，约 0.5 GB）。11-07 经用户确认，这 14 个文件（共 3.58 GB）已删除，日志文件夹里的其他文件保留。

**手掌平行桌面构型的局限**（这次在仿真里查出来的）
- 这个构型的手伸在手腕前方约 12 cm。目标离机器人底座近时（libero_object 的黄油、奶油奶酪，libero_spatial 的炉子），手腕要贴近肩部，肘关节（关节 4）弯到限位（−3.055 rad，限位 −3.072 rad），保持朝向的直线路线走不通，只能由 OMPL 换一种手臂构型过去（手的朝向变化 5–21°）。
- 手指朝下时，libero_spatial 木柜的顶面太高，保持当前朝向够不着，会报 out of reach。

**测试**
- `test_libero_planner_teleop.py` 从 13 个增加到 24 个，LIBERO 环境全部通过。新增内容：
  - 斜线路线、路线顺序和去重；
  - 分段时间律：拐角处速度为 0，所有采样点都在直线上，总时间不变；
  - `--planner-speed` 同时缩放线速度和角速度；
  - 投影公式、状态文字带编号、构型辅助函数；
  - 场景：编号的位置；编号只出现在主视图和 3D 视图；
  - 子进程：检查过的直线路线全程手的朝向变化 < 3°、穿透不超过 3 mm，并且到位；只用 OMPL 时穿透不超过 3 mm、朝向不超出 0.5 rad，并且到位；关节超出限位 2 mm 的起点也能规划；穿过牛奶盒的路线被拒绝，由 OMPL 兜底；
  - 入口：按 `M` 切到手掌平行桌面后，状态和日志正确，切换后规划器照常到位。
- 无窗口冒烟（SpaceMouse 输入全零，没有摄像头）：
  - libero_object 任务 0，手指朝下：去字母汤罐头共 234 ms（子进程规划 29 ms，其余是第一次加载场景），到位误差 6 mm；去黄油走直线路线 1，到位误差 7.1 mm；按 `M` 后 3.8 s 重建完成；
  - 切到手掌平行桌面后：去黄油、去牛奶都由 OMPL 兜底（直线路线因肘关节限位走不通），都到位；
  - straight 对照组从 forward 起步，再按 `M`：正常；
  - libero_spatial 任务 0，`--sim-speed 0.6`：去盘子走直线路线 1，规划 248 ms，到位；
  - 退出后临时目录里的场景文件已删除。
- 回归：LIBERO 环境 61 个测试全部通过（LIBERO / 摄像头 / 视图 / 规划器测试）；主线环境 254 通过、25 跳过、0 失败；冻结 baseline 哈希 172/173，与之前相同。`test_shadow_gripper.py` 在 LIBERO 环境（Python 3.8）里会因为 `str.removeprefix` 报错，它属于主线环境（Python 3.10），在主线里通过。
- 【未验证·实机】以上所有修改都还没有真人实机试过。

## 2026-11-08｜复位时再按 R 不再多复位一次，画面显示 RESETTING

**问题**（11-07 实机测试 F 发现）
- LIBERO 的复位是 robosuite 1.4 的硬复位，要重新建模型：主线程卡约 1.9 s，复位后第一步 `env.step` 再卡约 0.5 s（那次日志里 7 次复位都是这样）。这段时间画面停在最后一帧，没有任何提示。
- 画面窗口有自己的线程，主线程卡住时照常收键。卡住时按的 R 被记下来，复位一结束马上又复位一次。G 视频开始前连续复位了 4 次，卡了 7.6 s。

**改动**
- `src/wrist_teleop/runtime_controls.py`
  - `KeyboardCommandRouter` 新增 `hold_resets()` / `release_resets(grace_s)`。hold 期间收到的 R 直接丢掉并计数；release 后再过 0.5 s（`RESET_RELEASE_GRACE_S`）才重新接受 R，卡顿期间排队、晚到的按键也挡住。Q 不受影响。不调用这两个方法的入口（robosuite 1.5 那几个）行为不变。
  - 新增 `ResetFeedback`：入口取到复位后先 `begin`（挡住 R、画面显示提示、控制台打印），复位后第一步走完再 `settle`（去掉提示、放开 R，写一条 `reset_settled` 日志事件：`held_s` 是卡了多久，`ignored_r_presses` 是丢掉了几次 R）。
- `src/wrist_teleop/libero_views.py`：新增 `LiberoViewWindow.show_notice(text, wait_s)`。主线程卡住时渲染线程照样运行，提示直接盖在最后一帧上：主视图中间一条暗色横条，琥珀色字 `RESETTING - please wait` / `R is ignored until the scene is back`（OpenCV 只能画英文）。`begin` 最多等 0.15 s，确认提示已经上屏才开始复位。提示只画在显示的那一份上，`last_frame`（截图用）不带提示。
- 三个 LIBERO 入口都接上：
  - `libero_spatial_teleop.py`（主命令 1）；
  - `libero_planner_teleop.py`（主命令 2）。按 `M` 换构型重建时同样处理，提示 `SWITCHING HAND MOUNT`；
  - `libero_camera_teleop.py`（纯摄像头版）。
- `RUN_COMMANDS.md` 按键表的 `R`、`M` 两行更新。
- 没改：仿真频率、物理参数、复位过程本身、arm [0:6] / hand [6:22] 写入、Q / Ctrl+C、冻结 baseline。

**测试**
- 新增 `tests/test_reset_feedback.py`，12 个：
  - 按键路由：hold 期间的 R 被丢掉并计数，grace 内仍丢，之后新按的 R 正常复位；hold 期间 Q 照常退出；一直按住 R（自动重复）跨过 release 也不会再复位；不用 hold 时和以前一样；`take_reset` 和 hold 之间漏进来的 R 也被丢掉；
  - `ResetFeedback`：begin / settle 的提示和日志记录；没有画面窗口时也能用；
  - 画面：提示只画在主视图里；窄窗口自动缩小字号；只画在显示的那一份上；`show_notice` 会等渲染线程把提示画上屏；
  - 入口（LIBERO 环境，真实 libero_object 任务 0，无窗口，SpaceMouse 输入全零）：按 R 复位，复位过程中再按两次 R。结果只复位 1 次，`reset_settled` 记录丢掉 2 次、卡 2.52 s，之后正常控制 50 步；提示先出现、后消失。
- 规划器入口测试（按 `M` 切换构型）加了一条断言：切换后有一条 `what = "hand mount switch"` 的 `reset_settled`（这次卡 5.44 s）。
- 回归：LIBERO 环境 31 个通过（新测试 + 视图 / 主入口 / 纯摄像头版 / 规划器入口测试）；主线环境 40 通过、1 跳过（`test_integrated_runtime.py`、`test_integrated_entry_shutdown.py`、`test_hand_input.py` 和新测试，跳过的是只在 LIBERO 环境跑的入口测试）；冻结 baseline 哈希 172/173，与之前相同。
- 横幅的样子：用 11-07 那次运行的截图合成了一张预览，确认在主视图中间、不挡右侧两个小视图、字能看清。
- 【未验证·实机】还没在真人实机上按过 R。

## 2026-11-08（续）｜LIBERO 数据采集入口（主命令 3）：每个任务的成功检查、自动保存、随机摆放、视频和数据集

按用户 11-08 的决定做：先 libero_object，再 libero_spatial；每个任务 10 条；只存成功的；只计时不设超时；右上角特写固定成截图里的视角并左右镜像；按 M 换构型（和规划器版一样）；分辨率没影响就用高的；视频自动命名，放到云端同步文件夹里分享。完整的流程、结构和数据格式见 `docs/libero_data_collection_zh.md`。

**做了什么**
- 新入口 `scripts/run_libero_collect.py`（主命令 3）：纯人工遥操作，控制和主命令 1 一样（SpaceMouse 控臂，摄像头 dex 控 Shadow 手指）。按 M 换构型的逻辑在采集入口里重写，没有 import 规划器模块，采集入口里没有任何规划器代码。
- 每个任务一个成功检查器，由这个任务自己的 BDDL `(:goal ...)` 生成：
  - 逐条条件实时显示在主视图右下角，满足是绿色 `[OK]`，不满足是灰色 `[--]`；
  - 最终判定直接调用官方 `env._check_success()`，连续 10 步（0.5 s）成立才算成功；
  - 五个套件的 6 个场景类里 `_check_success` 写法相同（逐个看过源码），所以不用手写 130 个函数。
- 只存成功的 demo，每一轮都写进 `attempts.jsonl`：
  - 成功率 = 成功 / (成功 + 失败)；
  - 动过 SpaceMouse 之后按 R 或 M 记失败；中途退出、还没动就按键都不计入。
- 计时从第一次动 SpaceMouse 开始，不设超时。成功后自动保存、换随机摆放；一个任务满 10 条自动加载下一个任务；中途退出后再运行同一条命令就接着采。
- 随机摆放：`env.reset()` 之前设 `np.random` 种子，由 LIBERO 自己的摆放采样器摆（和官方初始状态同分布；官方 50 个初始状态留给评测）。验证过：不同编号摆放不同（变化几厘米），同一编号完全一样。
- 等物体落稳：随机摆放时物体从几厘米高处放下，头 0.3–1 s 会下落、滑动 5–10 cm（libero_spatial 有一个碗滑了 1 s）。每轮开始前机械臂保持不动（零增量，手张开），等所有物体速度连续 3 步低于 5 mm/s，最多 60 步。libero_object 实测约 10 步（0.5 s）。这几步不记录，期间显示复位提示。
- 碰倒检测：不在成功规则里的物体，从开始计时起移动超过 3 cm 或倾斜超过 30°，主视图显示 `MOVED: 物体名`。demo 照存，记下 `disturbed_objects`；生成数据集时可以用 `--exclude-disturbed` 排除。
- 右上角特写固定在方位角 270°（机器人左侧，主命令 1 里按两次 V 的视角），左右镜像，按 V 不转。只在采集入口里改。
- 按 M：同一任务、同一摆放，换另一种构型重建（无窗口实测约 2 s）。
- 复位、换构型、保存后换摆放都用 `ResetFeedback`，期间按 R 不起作用。
- 退出后自动运行 `scripts/build_libero_dataset.py`，从保存的仿真状态离线渲染，生成：
  - 每条 demo 一个视频，自动命名，如 `libero_object_t00_demo003_forward_12.4s__<任务名>.mp4`；
  - LIBERO 格式数据集，每个任务、每种构型一个文件，图像 256×256；
  - `SUMMARY.md`、`summary.csv`、`share_message.txt`（可以直接贴到 Slack 的一段话）。
- 设置了 `--share-dir`（或环境变量 `LIBERO_SHARE_DIR`），就把视频和汇总复制过去，由 OneDrive 同步上传；共享链接用户在网页上建一次，用 `--share-url` / `LIBERO_SHARE_URL` 告诉脚本。脚本不调用云盘或 Slack 的接口。
- 文档：新 `docs/libero_data_collection_zh.md`；`RUN_COMMANDS.md` 加主命令 3、按键、后缀和分享文件夹的设置；README 加一节。
- 没改：仿真频率、物理参数、现有脚本和共用模块、arm [0:6] / hand [6:26] 写入、R / Q / Ctrl+C 行为、冻结 baseline。

**离线渲染**
- 用保存的模型 XML 重建模型，按每步的状态设 qpos / qvel，`mj_forward` 之后渲染。要隐藏碰撞几何（group 0，绿色），否则机器人是绿的。
- 和 robosuite 自己渲染的图比，平均每像素差 0.1–0.2（满分 255）。
- 数据集图像沿用 LIBERO 的上下颠倒约定（和环境自带的相机观测一样）；视频是正的。

**实测大小**（一条 10.5 s 的模拟 demo，机械臂一直在动）
- 原始 demo 0.25 MB，视频 1.2 MB，256×256 数据集 42.8 MB（约 4 MB/s）；
- 生成视频和数据集用 4.4 s；
- 照这个算，每个任务 10 条、每条 15 s，一个套件约 6 GB 数据集。数据集随时可以从原始 demo 重新生成。

**测试**
- 新增 `tests/test_libero_collection.py`，14 个，全部通过（71 s）：
  - 纯逻辑 10 个、特写镜像 1 个；
  - 20 个任务的成功规则 2 个（libero_object + libero_spatial）：刚摆好时不成功，把物体放进篮子 / 盘子后官方判定和逐条判定都为真；
  - 采集主循环 + 离线处理 1 个：R、M、碰倒标记、自动保存、换摆放、采满结束；视频帧数、数据集形状、数据集图像和 robosuite 渲染的差 < 1/255、分享文件夹、重复运行不重做。
- 第一次跑主循环测试时，没人碰的牛奶和沙拉酱被标成移动了 4–5 cm。查下来是随机摆放后物体还在下落，于是加了等物体落稳，之后通过。
- 回归：LIBERO 环境 54 个通过（视图 8、复位提示 12、主入口 5、规划器 24、纯摄像头 5）；冻结 baseline 哈希 172/173，和之前相同。
- 【未验证·实机】还没有用真 SpaceMouse 和摄像头采过。

## 2026-11-09｜采集入口改为建在规划器版（主命令 2）上；特写 = 规划器版按两次 V 的视角，固定，只做镜像

**起因**（用户 11-09 指出 11-08 版的三个问题）
- 最新的版本是规划器版（主命令 2），11-08 的采集入口却是照主命令 1 做的：没有规划器，全程要自己控制。
- 射线：11-08 版是主命令 1 的掌心射线，最新版是五根指尖射线。
- 右上角特写：要的是规划器版的特写按两次 V 之后的视角（相机在手的左边照着手），固定住，然后只把相机画面左右镜像，方便操作，其他什么都不改。

**改动**
- `src/wrist_teleop/libero_collect_teleop.py` 重写：主循环照主命令 2 的主循环复制，import 规划器版的 `PlannerActionComposer`、`PlannerScene`（编号）、`FingertipRays`、`PlannerViewWindow`、`ApproachPlanner`、OMPL 侧车（这些文件都不改），再把 11-08 的采集功能（成功检查、自动保存、随机摆放和等落稳、attempts.jsonl、碰倒标记、M 换构型、复位提示）加上去。
  - 按键和命令 2 一样：1–9 接近目标，0 取消，L 开关射线，M 换构型；一动 SpaceMouse，规划器当拍停下，由 SpaceMouse 接管。
  - 每一拍手臂只听一方。每一步记在 demo 的 `control/arm_owner`（0 = SpaceMouse，1 = 规划器）；规划器事件（请求、结果、到位、取消及原因、失败）带步号存进 demo 属性 `planner_events`；每一轮的规划器计数写进 attempts.jsonl。
  - 计时从手臂第一次动开始（动 SpaceMouse，或者规划器开始带手臂走）。按下数字键到规划出路线这段手臂不动，不算。
  - 和命令 2 一样：默认构型 down，只支持 libero_spatial / libero_object。仿真速度固定 1×（数据按 20 Hz 记，没有 `--sim-speed`）。
  - 换摆放、换构型、保存 demo、退出之前，先取消正在进行的规划。
  - OMPL 场景文件：模型 XML 没变就沿用。libero_object 换摆放时 XML 不变（实测），整个运行只生成一个文件，规划器进程也不用重新加载；libero_spatial 每次重新生成。
- 特写：`CollectionViewWindow` 是 `PlannerViewWindow` 的子类。相机参数就是命令 2 的特写参数，方位角加 2 × 90°（= 270°），不转。整帧画好后只把特写那一块左右翻转，然后再画标签和编号，所以字是正的。标签文字和命令 2 一样（`CLOSE-UP (follows hand)`）。主视图、右下 3D 视图、射线、编号、SpaceMouse 映射都和命令 2 一样。
- `libero_collection.py`：demo 格式改为 v2，加 `control/arm_owner`；`append` 先检查参数再写入（以前 `goal_status` 长度不对时会留下半行）；面板的按键行改成规划器版的按键。
- `libero_dataset_build.py`：数据集每条 demo 加 `arm_owner`；视频里规划器在动的帧，时间后面标 `planner`；summary.csv 加 `planner_requests`、`planner_steps`；SUMMARY.md 每条 demo 写规划器动手臂的步数占比。
- `scripts/run_libero_collect.py`：加规划器后缀（`--planner`、`--planner-speed`、`--planner-python`、`--clearance-cm`、`--no-rays`）和测试用的 `--smoke-keys`；`--hand-mount` 默认 down。
- 文档：`RUN_COMMANDS.md` 主命令 3、按键表、后缀表；`docs/libero_data_collection_zh.md`；README。
- 没改：仿真频率、物理参数、现有脚本和共用模块（规划器版只 import 不改）、arm [0:6] / hand [6:26] 写入、R / Q / Ctrl+C 行为、冻结 baseline。

**验证**
- 特写逐像素核对：同一场景、1600×900 窗口，采集入口的特写和命令 2 按两次 V 的特写左右翻转后一样（标签那一条除外，平均差 0.000；不翻转时 34.25）。截图看过：主视图和右下视图有编号 1–7，五根红色射线，目标面板在主视图右下角。
- `tests/test_libero_collection.py` 14 个全部通过（68 s）。新增覆盖：SpaceMouse 还在推时按数字键，规划器当拍被收回、没动手臂；SpaceMouse 不动时按 1，规划器连续带一段，`0` 取消（或已到位）；`arm_owner`、`planner_events`、attempts.jsonl 互相对得上；数据集里的 `arm_owner`；summary.csv 的规划器列。
- OMPL 冒烟（无窗口，真 OMPL）：按 1 走检查过的直线路线，规划 361 ms，到位误差 3.5 mm；R 后再按 1 规划 23 ms（场景文件只生成 1 个）；控制循环 20.0 Hz。
- 命令行：`--list-tasks` 正常；`--headless --headless-smoke --smoke-keys 1:1,5:r,6.5:2`（默认 OMPL）：按 1 规划 209 ms、到位 3.5 mm；R 记失败，复位 1.7 s；再按 2 规划 24 ms；运行结束时还在走的规划被取消，这一轮记为中途退出。
- 一条"先按 1 再用 SpaceMouse"的模拟 demo：规划器在第 10–54 步（45 步），之后 SpaceMouse 接手；计时和数据集都从第 10 步开始。原始 0.26 MB，视频 1.26 MB，256² 数据集 38.9 MB（10.0 s），离线处理 4.2 s。
- 回归：LIBERO 环境 54 个全部通过（视图 8、复位提示 12、主命令 1 的 5、规划器 24、纯摄像头 5）；冻结 baseline 哈希 172/173，和之前相同。
- 另查了随机摆放的变化：libero_object 任务 0 的汤罐、沙拉酱、番茄沙司随机摆放时位置不变（0.0 cm），篮子 ≤ 3 cm，其他物体 ≤ 1 cm；官方 50 个初始状态也一样。原因是 LIBERO 的摆放区域减去物体半径后几乎没有余量。libero_spatial 约 3 cm，也和官方一致。所以同一个任务每次规划出的路线长度几乎一样，是正常的。
- 【未验证·实机】规划器版的采集入口还没在实机上跑过。

## 2026-11-09｜采集输出简化：只留 attempts.csv 和每一轮的录屏视频

**起因**（用户 11-09 实际采了三轮，看了输出文件夹和视频；那三轮用的还是 11-08 的版本）
- 输出的东西太多太杂，现阶段用不上，只要一张 CSV 表和视频文件。
- 视频换了视角，也没有摄像头画面。要求视频里的三视图和操作时看到的一样（摄像头画面不用打码），重点突出时间和这一轮成功还是失败。
- 用户已经把那次的输出文件夹删到回收站，没有动它；`outputs/libero_collect/` 现在是空的。

**改动**
- 新 `src/wrist_teleop/libero_attempt_video.py`：每一轮一个视频。画面是三视图窗口每一帧画好的图（`last_frame`：三个视图、摄像头小窗、状态栏、目标面板、射线和编号都在；RESETTING 之类的大字提示只画在显示的那一份上，不进视频），缩到 1920 宽，上方加一条标题栏：任务、套件 / 第几轮 / 构型、大号计时（开始前黄色 READY）。结尾停 2 s：标题栏变成绿色 SUCCESS / 红色 FAILED / 灰色 QUIT 加用时和结束原因，画面加同色边框。
  - 每个控制拍录一帧（20 fps，实时速度）；保留开始前 1 s；上一个摆放的旧画面不录；手臂没动过的轮次不留文件。
  - 编码在单独的线程里，用 imageio 自带的 ffmpeg（低优先级进程，libx264 veryfast，CRF 23）。控制循环这边不阻塞，队列满了就丢帧并计数。先写临时文件，结束时改成正式文件名。
- `libero_collection.py`：
  - 新 `AttemptTable`（`attempts.csv`，12 列）。新文件带 BOM，Excel 直接打开不乱码。断点续采、成功次数、成功率、下一个轮次号和摆放编号都从这张表算。最后一行被截断也能读，下一行另起一行。Excel 锁住文件时行先留在内存里，之后补写；退出时还锁着，就另存 `attempts_unsaved_<时间>.csv`。
  - 新 `attempt_video_name`、`ended_by_text`、`share_collection`（只复制 CSV 和写好的视频）。
  - 面板的 "saved" 改成 "successes"，最后一行加上 `Q: quit`；终端汇总表的列名也改成 successes。
- `libero_collect_teleop.py`：
  - 每一轮开始时开一个视频，每拍把窗口画面交给它；结束时（成功 / R / M / 退出）写上结果，再往 CSV 加一行。手臂没动过的轮次不留行也不留视频。
  - 原始 demo 只在 `save_demos`（`--save-demos`）时存，放到 `outputs/libero_collect_demos/<套件>/`。运行日志移到 `outputs/libero/<时间>_collect_<套件>/`。
  - 成功的用时改成和计时器一样的真实时间。以前按步数算，启动时窗口创建卡一下就会差 0.3 s。原始 demo 里两个都存。
- `scripts/run_libero_collect.py`：去掉结束后自动运行离线处理（连同 `--no-build`、`--image-size`），加 `--save-demos`；`--share-dir` 改为复制 CSV 和视频；`--list-tasks` 从 CSV 读。
- `scripts/build_libero_dataset.py`：默认读 `outputs/libero_collect_demos`，采集入口不再自动调用它。
- 文档：`RUN_COMMANDS.md` 主命令 3、按键表、后缀表；`docs/libero_data_collection_zh.md` 重写流程和输出部分；README。
- 没改：仿真频率、物理参数、现有脚本和共用模块、arm [0:6] / hand [6:26] 写入、R / Q / Ctrl+C 行为、冻结 baseline；特写和窗口内容也没动（按 11-09 上一轮的要求）。

**验证**
- `tests/test_libero_collection.py` 21 个全部通过（82 s）。新增和改动：
  - attempts.csv 3 个：读写、续采、BOM、截断行、文件被锁；
  - 视频 2 个；
  - 默认模式 1 个：只有 CSV 和视频，quit 的视频，再运行时编号接着数，手臂没动过的运行不留东西；
  - 完整主循环测试改为检查 CSV 的四行和 4 个视频，原始 demo 和数据集的检查照旧。
- 实测条件：2544×1334（用户实际的窗口大小），无窗口模式，摄像头小窗用合成画面（没有开摄像头），straight 规划器，脚本输入，各 20 s。
  - 录视频：20.00 Hz（最慢 5% 是 19.98 Hz），每拍耗时中位数 20.6 ms、p95 23.2 ms，丢帧 0；
  - 不录：20.00 Hz（19.99 Hz），20.0 / 21.8 ms；
  - 视频 1920×1088，约 0.6 MB / 10 s。
  - 抽帧看过：三视图、摄像头小窗、计时，结尾的 FAILED / QUIT / SUCCESS 和边框都对。
- 回归：LIBERO 环境 54 个全部通过；冻结 baseline 哈希 172/173，和之前相同。
- 【未验证·实机】新的输出还没在实机上采过；用真实摄像头画面时视频多大也没测。

## 2026-11-10｜规划器对准点改为三指指根（掌心朝下时）；正在录的视频不再放进 videos/

**起因**：用户 11-10 第一次用 11-09 版实采（libero_object 任务 0）。第一次运行 8 轮、3 次成功；第二次运行的第 9 轮挂了 17 分钟，之后按 Q，记为 quit。然后提了三件事：
- 视频在哪里，"如果没有输出视频就先不要添加"。8 个视频都在 `outputs/libero_collect/libero_object/videos/`。当时第二次运行正在录第 9 轮，录制中的临时文件 `_recording_<pid>_<n>.mp4` 也在 `videos/` 里。ffmpeg 还开着它，资源管理器显示 0 KB（用句柄读到的实际大小是 14.4 MB），看起来像一个没有内容的视频。
- 现在还能做哪些任务，命令是什么。在对话里回答了：libero_object 和 libero_spatial 各 10 个任务，`--tasks`、`--suite`、`--list-tasks`、`--hand-mount forward`。
- 规划器到位后物体在指尖下面。用户每次还要沿手指方向（截图右下角的箭头）把手往前推一点才好抓，希望物体在食指、中指、无名指的指根下面。

**改动**
- `libero_motion_planning.py`：
  - 新增 `ROOT_FINGERS`（ff / mf / rf）、`PALM_DOWN_MAX_TILT_DEG = 45`、`finger_root_center`、`palm_faces_down`。
  - 新增 `FingerRootLocator`：取三根手指近节的原点，也就是 MCP 屈曲关节 J3，和 knuckle 原点重合；掌心法向取手掌局部 −y，来自 `shadow_gripper.SHADOW_PALMAR_NORMAL_LOCAL`。
  - `HandState` 加 `roots` 和 `palm_normal`，新增 `aim_point` / `aim`：
    - 掌心法向离竖直向下不到 45°（forward 构型）时，对准三指指根的中心；
    - 否则（down 构型，手指朝下）还是原来的抓取中心；
    - 拿着东西时对准物体，和以前一样。
  - 高度规则不变：最低指尖高出目标顶面 2 cm。`planner_request` 事件多记一个 `aim`。
- `libero_planner_teleop.py`：`PlannerScene` 建 `FingerRootLocator`，`hand_state()` 带上指根和掌心法向。主命令 2 和 3 都用它，所以两个入口一起改了。
- 录制文件：
  - `libero_attempt_video.py`：`AttemptVideo` 新参数 `work_dir`。录制中的文件写在这里，结束时才改名移到 `videos/`（跨盘时用 `shutil.move`）。
  - `libero_collect_teleop.py`：`work_dir` 传本次运行的日志文件夹 `outputs/libero/<时间>_collect_<套件>/`。
  - 现在 `videos/` 里只有录完的视频；程序意外中断时，没录完的文件留在日志文件夹里。
- 测试，以及文档：RUN_COMMANDS（主命令 3 说明、按键表）、采集文档 5.3 / 7 / 8、README、桌面交接文档。
- 没改：仿真频率、物理参数、arm [0:6] / hand [6:26] 写入、R / Q / Ctrl+C 行为、冻结 baseline、规划器的高度规则和路线检查、手指映射。

**为什么 down 构型不改**：down 构型手指朝下，指根就在指尖正上方。让指根对准物体，等于让手指落在物体顶上。静态测量里，down 构型的抓取中心比指根中心偏向拇指一侧 5.4 cm。

**验证**
- 静态几何（libero_object 任务 0，初始手型，forward 构型）：
  - 三指指根中心在旧抓取中心后面 5.3 cm（沿手指方向，往手腕那边），偏小指一侧 5.4 cm，水平共 7.6 cm；
  - 旧点和拇指张开多少有关，新点不随手型变；
  - 近节原点、knuckle 原点、J3 关节锚点三者重合（0.0 mm）。
- 规划实测，条件和实际采集一样：
  - OMPL 后端（先检查直线路线，再 RRT-Connect），采集入口的随机摆放（基础种子 7）。
  - forward 构型 20 个摆放：libero_object 任务 0 摆放 0–7（就是用户 11-10 采的那几个摆放）、任务 1–9 摆放 0、libero_spatial 任务 0–2 摆放 0。down 构型 1 个：任务 0 摆放 0。
  - 每个摆放从同一个状态出发，旧、新对准点各规划一次，走到位后再停 10 步。
- 新对准点（forward）：
  - 20/20 规划成功并到位；
  - 指根中心离物体中心 ≤ 0.1 mm，最低指尖高出顶面 19.1–19.3 mm；
  - 黄油那个摆放例外：规划器为了离旁边远一点，把终点抬了 9.9 mm；
  - 手离目标物体最近 13.4–18.6 mm。
- 旧对准点：
  - 19/20 成功。任务 1 奶油奶酪那个摆放报 blocked：手会碰到旁边的番茄沙司；
  - 到位后指根中心离物体中心 76 mm；
  - 有 2 个摆放的终点被抬高 29.8 / 39.9 mm。
- 所有移动过程中机器人和场景都没有接触（MuJoCo 接触深度 0）。down 构型新旧完全一样，还是抓取中心。
- 用户 11-10 的 9 个视频（只读长度和帧数，没看画面）：
  - 都能正常读，1920×1088、20 fps；
  - 长度 = 用时 + 开始前 1 s + 结尾 2 s，长的几段短约 1%。原因是那次运行控制循环平均只有 19.74 Hz：中位数 20.5 Hz，每拍耗时中位数 41 ms、p95 51 ms，6.5% 的拍超过 50 ms，共掉了 195 拍。视频队列丢帧 0；
  - 画面上的计时和 CSV 的用时都是真实时间，不受影响；
  - 真实摄像头下视频约 1 MB / 10 s（46 s 的成功那轮 4.95 MB）。
- 新增和改动的测试：
  - `tests/test_libero_planner_teleop.py` 新增 2 项：
    - 几何：掌心朝下对准指根，侧着对准抓取中心，拿着东西对准物体，45° 分界；
    - 仿真：forward 构型三段直线到位后，指根中心离物体中心 < 5 mm，最低指尖 2 cm ± 5 mm，请求事件里记了 `aim`。
  - 原来 down 构型那一项加了一条"还是抓取中心"。
  - `tests/test_libero_collection.py` 新增 1 项：录制中的文件在工作文件夹里，`videos/` 不存在；结束后移进去，工作文件夹为空，帧数对。默认模式那一项加上"日志文件夹里没有残留视频"。
- 测试结果：
  - `test_libero_collection.py` 22 个全部通过（94 s）；
  - 回归 LIBERO 环境 75 个全部通过（视图 8、复位提示 12、主命令 1 的 5、规划器 26、纯摄像头 5、摄像头控臂 19；158 s）。这次把 `test_camera_arm_control.py` 也一起跑了，以前的 54 个不含它；
  - 端到端冒烟（采集主循环，OMPL，forward 构型，无窗口，1 s 时按 1，输出写在临时文件夹）：请求的 `aim` 是 finger roots，检查过的直线路线一次找到，到位误差 0.9 mm；录制中的文件只在运行日志文件夹里，`videos/` 里只出现录完的视频，结束后日志文件夹没有残留；CSV 一行（quit，time limit）；
  - 冻结基线哈希 172/173，和之前相同。
- 【未验证·实机】新的对准点还没在实机上试；指根对准对 libero_spatial 的碗（要抓碗沿）是否合适也没试。

## 2026-11-10（晚）｜采集入口：按 S 才保存这一轮的视频

**起因**：用户要求"新增一个功能，按键才正式保存视频，不按就不录视频，其余的什么任务成功检测和成功率还保留"。

**做法**
- 新键 `S`（只在采集入口，主命令 3）：保存这一轮的视频，再按一次取消。每一轮开始时都是不存，R、M 之后的新一轮也是。
- 每一轮照旧在后台录，写在运行日志文件夹的 `_recording_*.mp4`。这一轮结束时：
  - 按过 `S`：改名移进 `videos/`，CSV 的 `video` 写文件名；
  - 没按：ffmpeg 停下，文件删掉，`video` 为空。
  - 所以存下来的总是整轮（第一次动手臂前 1 s 开始，到结尾 2 s 的结果），`S` 按得早晚都一样。
- 为什么不做成"按 S 才开始录"：那样按键之前的部分就没了，而想存哪一轮往往要做到一半才知道。代价是没存的轮次也在后台编码，CPU 占用和以前一样。
- 主视图面板：
  - 计时那一行从 `recording` 改成 `running`，因为现在不一定录；
  - 下面多一行：灰字 `video not saved   S: save this attempt's video`，或红字 `REC: this attempt's video will be saved   S: undo`；
  - 终端打印每次切换，存下时打印视频路径。
- 按住 `S`（键盘自动重复）只切换一次：和上一次按键隔不到 0.3 s 的不算（`SAVE_VIDEO_REPEAT_S`）。
- 复位、切换构型期间按的 `S` 和其他键一样丢掉。成功判定一成立这一轮就结束，之后不能补存。
- 运行日志：
  - `video_mark` 事件：第几轮、存不存；
  - `exit` 事件里的 `recordings_not_saved`；
  - `metadata.json` 的 `video.kept_only_after_key`。
- 退出时等没存的录制删完再结束，日志文件夹里不留文件。
- 没改：成功判定、CSV 的行和列、成功率、计时、R / M / Q、仿真频率、物理参数、arm [0:6] / hand [6:26] 写入、冻结 baseline、主命令 1 / 2。

**改动的文件**
- `libero_collect_teleop.py`：按键、切换、结束时存或删、退出时等删除完成；
- `libero_collection.py`：面板的一行、颜色 `RED`；
- `run_libero_collect.py`：说明文字；
- `tests/test_libero_collection.py`。

**验证**
- `test_libero_collection.py` 23 个全部通过（90 s）：
  - 新增 1 项：开始录之后放弃，工作文件夹里的文件被删，`videos/` 不出现；
  - 面板：两种提示的文字和颜色；
  - 主循环 ①：第 1 轮按 `S`，第 3 轮按住（两次隔 50 ms 只算一次），第 4 轮按了又取消，第 2 轮不按。结果只有第 1、3 轮有视频，第 2、4 轮 CSV 的 `video` 为空，日志里的 `S` 记录是 1 存、3 存、4 存、4 不存，日志文件夹里没有残留视频；
  - 主循环 ②：不按 `S` 的运行只有 CSV 一行；1.5 s 时按 `S` 的运行，视频帧数和这一轮的步数对得上，说明是整轮，不是从按键开始。
- 回归 LIBERO 环境 75 个全部通过（152 s）；冻结基线哈希 172/173 不变。
- 端到端冒烟（无窗口，合成输入）：OMPL、forward 构型，1 s 时按 1 让规划器带手臂走。第一次运行 4 s 时按 `S`：录制中的文件只出现在运行日志文件夹里，`videos/` 里没有；到时结束后视频移进 `videos/`（QUIT 8.7 s，231 帧；按整轮算应有 233 帧，如果从按键那一刻才开始录只有约 159 帧）。第二次运行不按 `S`：CSV 照样有第 2 行，`video` 为空；录制在运行日志里出现过，结束时被删掉。两次控制循环中位数都是 20.00 Hz，视频丢帧 0，日志文件夹里没有残留视频。
- 【未验证·实机】还没在实机上按过 `S`。

## 2026-11-10（晚 2）｜S 改成按下才开始录

**起因**：上一版是每轮都在后台录、按过 S 才留下。做完问用户"S 键这样用可以吗"，用户选了"改成按 S 才开始录"。

**做法**
- 不按 `S`：完全不录，不开编码线程和 ffmpeg，也不写文件。
- 按 `S`：从这一拍开始录，录到这一轮结束，结束时移进 `videos/`。在手臂第一次动之前按的，照旧从第一次动之前 1 s 开始。
- 录的时候再按 `S`：取消，ffmpeg 停下、文件删掉；再按又从那一刻重新开始录。
- 每一轮开始时都不录。面板提示改成：
  - 灰字 `not recording   S: record this attempt from now`；
  - 红字 `REC  recording, saved when this attempt ends   S: cancel`。
- 中途按的，视频第一帧的计时就是按下时的用时。
- 运行日志：
  - `video_record` 事件：第几轮、开始还是取消、第几步；
  - `exit` 事件里的 `recordings_cancelled`；
  - `metadata.json` 的 `video.recorded_from_key`。
- 常量改名为 `RECORD_KEY = "s"`、`RECORD_REPEAT_S = 0.3`。
- 没改：成功判定、CSV、成功率、计时、R / M / Q、仿真和物理参数、写入隔离、冻结 baseline。

**验证**
- `test_libero_collection.py` 23 个全部通过（98 s）：
  - 主循环 ①：第 3 轮成功视频的帧数 = 从第 4 步（按键）到结束的步数 + 结尾 2 s；第 4 轮取消的录制被删，退出时记 1 个；
  - 主循环 ②：不按 `S` 什么都不录；1.5 s 时按 `S` 的视频帧数 = 按键之后的步数 + 2 s。
- 回归 LIBERO 环境 75 个全部通过（166 s）；冻结基线哈希 172/173 不变。
- 端到端冒烟（无窗口，合成输入）：OMPL、forward 构型，1 s 时按 1 让规划器带手臂走，每次运行 10 s。第一次运行 4 s 时按 `S`：第 74 步开始录，录制中的文件只在运行日志文件夹里；结束后视频移进 `videos/`，159 帧 = 193 步 - 74 + 结尾 40 帧，正好是从按键开始（整轮应有 232 帧）。第二次运行不按 `S`：CSV 有这一行、`video` 为空，运行日志文件夹里从头到尾没出现过视频文件。第三次运行 3 s 按 `S`、5 s 再按：第 53 步开始录、第 93 步取消，录制文件出现过、结束时已删掉，`videos/` 没有新文件，退出时记 1 个取消的录制。三次视频丢帧都是 0，平均 20 Hz（第一次 8.6 s 走了 172 步）。
- 【未验证·实机】还没在实机上按过 `S`。

## 2026-11-10（晚 3）｜采集成功一次就退出：WPS 锁着 CSV + stderr 句柄失效

**起因**：用户实采时第一轮成功后程序就退出了（终端停在 `success: please wait, R is ignored until the scene is back ...`，没有汇总），要求"跟之前一样，一直做，每个任务10次"。

**排查**
- 两次运行（17:40、18:06）的 telemetry 一样：`attempt_started`、规划器两次到位、`exception: OSError: [WinError 6] 句柄无效`、`safe_zero`，没有 `attempt_end` 和 `exit`；采集文件夹里各多了一个 `attempts_unsaved_<时间>.csv`，成功那一行在里面。
- `attempts.csv` 当时开在 WPS 里（WPS 打开后别的程序只能读）。成功那一行写不进去，按设计留在内存里，然后往 stderr 打一句提示，就是这句报了 `WinError 6`。异常处理里的 stderr 打印、退出前"还锁着，已另存"的 stderr 打印又各报一次，所以终端里什么都没有。
- stderr 为什么坏：在新开的控制台窗口里试，`import mediapipe` 之后 fd 2 的句柄变了（240 → 1688），`print(..., file=sys.stderr)` 就报 `WinError 6`，stdout 不受影响。mediapipe 的音频模块导入 sounddevice，sounddevice 启动 PortAudio 时先把 fd 2 指到 NUL，再换回一个新句柄；原来的控制台句柄被关掉，Python 3.8（这里是 3.8.0）控制台里的 `sys.stderr` 还拿着它。只导入 sounddevice 也一样。管道和测试里 stderr 不是控制台，不会出现，所以测试一直没发现。
- 11-10 第一次运行（8 轮、3 次成功）没出事，是因为那时 WPS 没开着这张表，成功时不往 stderr 打印。

**改动**（只改采集入口自己的文件）
- `scripts/run_libero_collect.py`：最前面（导入 robosuite 等之前）调用 `reopen_stderr_after_sounddevice()`：先导入 sounddevice（约 0.4 s），fd 2 的句柄变了而且 `sys.stderr` 是控制台流，就在 fd 2 上重新打开 `sys.stderr`（编码、错误处理、行缓冲照旧）。之后摄像头线程导入 mediapipe 时 sounddevice 已经加载，不会再动 fd 2。管道、文件、测试里什么都不改。
- 没改：采集主循环、CSV 被锁时的处理（行留在内存，解锁后补写，退出时还锁着就另存）、成功判定、计时、R / M / Q / S、仿真和物理参数、写入隔离、冻结 baseline、共享模块。

**验证**
- 真实控制台（新开的窗口）里端到端：先导入 mediapipe，`attempts.csv` 用只读、只许别人读的方式锁住（和 WPS 一样），OMPL，forward 构型，无窗口，脚本 SpaceMouse，第 15 步把汤罐放进篮子。
  - 修之前：第 1 轮成功后退出，日志和实采一样，留下 `attempts_unsaved_*.csv`，退出码 120。
  - 修之后：第 1 轮成功后接着换摆放；第 2 轮第 3 步解锁，按 1 让 OMPL 规划，成功时取消还在走的规划，两行一起写进表；第 3 轮成功后 3/3，`all tasks done`，退出码 0，没有另存的表。
- `test_libero_collection.py` 25 个全部通过（101 s），新加 2 个：CSV 被锁时成功后接着采（解锁后补写）；隐藏的新控制台里，只导入 mediapipe 时 stderr 打印报 `WinError 6`（退出码 120），先导入入口脚本时正常（退出码 0）。
- 回归 LIBERO 环境 75 个全部通过（150 s）；冻结基线哈希 172/173 不变。
- 【未验证·实机】修好后还没在实机上重跑。

**两次没写进表的成功**：17:40 那次（第 10 轮，摆放 8，14.1 s，有视频 `libero_object_t00_a010_SUCCESS_14.1s__...mp4`）在 `attempts_unsaved_20261110T224004Z.csv`；18:06 那次（也记成第 10 轮，摆放 8，20.7 s，没录视频）在 `attempts_unsaved_20261110T230623Z.csv`。`attempts.csv` 里还是 9 行（3 次成功），补不补等用户决定。

## 2026-11-11｜采集入口：换摆放后 SpaceMouse 要等三四秒才能接管

**起因**：用户每个任务要做 10 次，每次换摆放后按 1 让规划器把手腕带到番茄酱上方，想马上用 SpaceMouse 微调去抓，但要等三四秒 SpaceMouse 才有反应。

**原因**
- 每次换摆放（以及 M、换任务）时 `rebind()` 调 `SpaceMouseHid.reset_input_state()`。它除了清掉旧的动作和按键，还把启动时学到的零点也清掉、取消就绪。
- 重新学零点要连续 8 个落在死区（量程的 8%）里的 HID 报告，学好之前所有输入都当成 0。
- 这台 SpaceMouse Wireless 不碰时一个报告都不发（2026-09-17 的诊断日志：不碰的 10 s 里 `valid` 一直是 false）；一推，第一两个报告就超出死区，计数清零。所以只有手轻轻搭着、碰巧连着出 8 个小报告时才学好，平均要三四秒。
- 映射层（平滑 0.35、3 拍归零）只差一两拍，不是原因；规划器到位后也立刻把手臂交还 SpaceMouse，也不是原因。

**改动**（只改采集入口，共享模块不动）
- 新文件 `src/wrist_teleop/libero_collect_device.py`：`KeepNeutralSpaceMouse`（`SpaceMouseHid` 的子类）。`reset_input_state()` 照旧清掉旧输入、取消就绪，但已经学好的零点放回去。零点是设备本身的偏置，学好后本来就不会再改。
- 就绪的条件不变（复位 0.3 s 后、看到摇杆在中间）：不碰时不发报告，算在中间，0.3 s 后就绪；复位时还按着的，一直在发偏离的报告，要松开才就绪。
- 打开（或断开后重新打开）设备时照旧重新学零点。
- `libero_collect_teleop.py`：真实设备改用 `KeepNeutralSpaceMouse`；每一拍的日志加 `spacemouse_armed`，退出记录加 `spacemouse_kept_neutral_resets`，下次实采能直接从日志看到换摆放后什么时候就绪。
- 没改：`device.py`、映射、死区、平滑、规划器、R / M / Q / S、仿真和物理参数、写入隔离、冻结 baseline。命令 1、2 还是原来的行为。

**验证**
- 新的 `tests/test_libero_collect_device.py` 9 个全部通过（29 s）。用假 HID 复现：原来的读取换摆放后连推 4 下都不动；新的读取推的下一拍就动。真实采集主循环里：换摆放后静置 1.5 s 再推，新的在第 32 步以内开始计时，旧的推了 1 s 这一轮还没开始。
- 采集 25 项 + 新的 9 项 + LIBERO 回归，共 126 个全部通过（290 s）；冻结基线 172/173 不变。
- 【未验证·实机】还没用真的 SpaceMouse 试。

## 2026-11-11（续）｜实采 47 轮 44 次成功；SpaceMouse 和 stderr 修复的实机数据；演示片段；README 和 GitHub

**实采**（01:10–01:40，主命令 3；dex 手：DexPilot + 项目映射；forward 构型；OMPL；SpaceMouse Wireless）
- libero_object：字母汤罐头（任务 0）3 轮 2 次成功、1 次 Q；番茄酱（任务 4）11 轮 10 次成功、1 次 Q；番茄沙司（任务 5）10/10；黄油（任务 6）10/10。
- libero_spatial：桌子中间的黑碗放到盘子上（任务 2）10/10；炉子上的碗（任务 7）1/1；木柜顶上的碗（任务 9）2 轮 1 次成功、1 次 Q。
- 这一晚 9 次运行、47 轮、44 次成功，约 30 分钟。两张 `attempts.csv` 合计 56 轮 47 次成功（加上 11-10 下午的 9 轮 3 次），成功用时中位数 13.5 s（p10 8.7 s、p90 24.3 s、最长 46.9 s）。各任务成功用时中位数：字母汤罐头 12.2 s、番茄酱 14.0 s、番茄沙司 9.4 s、黄油 16.4 s、桌子中间的碗 14.0 s。
- 凑满 10 次成功的任务：番茄酱、番茄沙司、黄油（libero_object）和桌子中间的碗（libero_spatial）。
- libero_object 表里有 5 行标了 `moved_objects`（别的物体被挪动或碰倒）。

**SpaceMouse 修复（上一条）在实机上成立**
- 看的是：规划器到物体上方（`planner_arrived`）之后，同一轮里 SpaceMouse 第一次有输入（`wrist_norm` > 0）用了多久。放到篮子或盘子上方的到位不算，那里只需要张手。
- 修之前（11-10 晚，15 次到位）：中位数 3.16 s，53% 超过 2 s。
- 修之后（11-11，40 次到位）：中位数 0.34 s，p75 0.73 s，12% 超过 2 s。
- 换摆放后到 `spacemouse_armed` 为真：中位数 0.59 s（48 次）。9 次运行的退出记录里 `spacemouse_kept_neutral_resets` 合计 41。
- 上一条最后的【未验证·实机】由这里的实机数据补上。

**stderr 修复（11-10 晚 3）也在实机上成立**：修好后 11-10 晚上 18:34–18:47 又开了 6 次，`attempts.csv` 一直开在 WPS 里。6 次都是按 Q 正常退出，没有异常。其中 3 次各成功 2 次，第一次成功后程序没退出，接着采。有记录的 5 次退出时按设计把行另存进 `attempts_unsaved_*.csv`，另 1 次没开始采就按了 Q。11-11 的 9 次运行也都正常退出。11-10（晚 3）那条最后的【未验证·实机】由这里补上。

**规划器（数字键）**
- 11-10：55 次请求，48 次有路径（直线 41、OMPL 7），35 次到位，19 次取消（14 次是 SpaceMouse 接管），1 次失败；规划用时中位数 94 ms。
- 11-11：120 次请求，116 次有路径（直线 84、OMPL 32），83 次到位，34 次取消（SpaceMouse 接管 26、换目标 5、成功时 3），3 次失败（1 次没找到路径、2 次起点被挡住）；规划用时中位数 92 ms。
- 演示片段里快的几轮（8 s 左右）：规划器走 4–6 s，SpaceMouse 动 1 s 左右；手指一直跟着我的手。

**要注意的**
- 所有实采（11-10、11-11）都没加 `--save-demos`，没有 `outputs/libero_collect_demos/`。这 47 次成功只有 CSV 里的一行（按了 S 的还有视频），没有能拿来训练的轨迹。要训练数据，下次采集要加 `--save-demos`。
- 7 个 `attempts_unsaved_*.csv`（都是 11-10 晚上的）里有 11 行不在 `attempts.csv`：字母汤罐头 5 行（4 次成功、1 次失败），番茄酱 6 行（4 次成功、2 次 Q）。补不补等用户决定，补之前先关掉 WPS。
- 摄像头小窗写的是 `hand mode: legacy-curl | active: legacy-curl`，但这些运行的元数据里手是 dex（DexPilot sidecar）。`hand_input.py` 的标签在运行状态里没有 `hand_mode`、`algorithm` 又不等于 `dex-retargeting` 时退回 legacy-curl，Shadow 这条路径的 `algorithm` 是 "official DexPilot …"。只是标签不对，没改（共享模块）。

**演示片段**（`docs/media/`，去掉了声音，摄像头小窗里的人脸打了马赛克，每个都看过）
- `libero_spatial_bowl_on_plate.mp4`：OBS 录屏（01:35:33 那段）20.85 s 起 40.2 s，1280x720 30 fps；任务 2 第 3–5 轮连续成功（7.7、13.5、8.3 s）。GIF 只放第 3 轮，2 倍速，800 宽，README 首页用它。
- `libero_object_tomato_sauce_x3.mp4`：OBS 录屏（01:22:05 那段）88.85 s 起 35.8 s，1280x720；番茄沙司第 8–10 次成功（8.7、8.6、8.4 s），最后是 `ALL TASKS DONE`。GIF 4 倍速，480 宽。
- `libero_object_soup_collect.mp4`：采集入口自己录的视频（11-10，第 10 轮，按了 S），1920x1088 20 fps，17.1 s，上方有计时，14.1 s 成功。GIF 2 倍速，480 宽。
- 11-11 没按 S，入口自己没录视频，所以用 OBS 录屏。黄油那段 OBS（01:25:05，4 分钟）没用；11-05 的屏幕录制里没有完整的成功抓取，也没用。

**测试**
- 主测试集（`.venv-robosuite`）：285 通过、41 跳过、1 失败。失败的是 `test_loop_timing.py::FixedRateClockTest::test_short_overrun_catches_up_on_the_same_grid`（睡 30 ms，要求 40 ms 内醒）：`test_allegro_hand_input.py`、`test_grasp_validation.py`、`test_gripper_teleop.py` 任意一个在同一进程里先跑才失败，单独跑 5/5 通过。原因还没查到，没改。
- LIBERO 环境 8 个文件：109 个全部通过，没有跳过（规划器 worker 的测试用到 `ompl_planner` Conda 环境），6 分 30 秒。凌晨那次没跑完（日志停在 72/109），这次是重跑。
- 冻结基线哈希 172/173 不变。

**README 和 GitHub**
- README：首页动图换成 11-11 的碗放盘子；演示分成"11 月：采集"和"10 月：第一版"两组；加了两条教训（换摆放后 SpaceMouse 不理人、打开 CSV 让程序退出）；"还不行的地方"、运行命令、测试、文件位置、开发日志改到 11-11。
- GitHub 镜像同步本地：新加规划器、采集、视频、设备、计时模块和对应的脚本、测试、文档和演示片段，删掉已退役的 8 个脚本；`RUN_COMMANDS.md` 第一次放上去（README 一直链接它）。
