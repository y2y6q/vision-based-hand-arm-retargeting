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
