# robosuite SpaceMouse 主线技术说明

[README](../README.md) 给出当前命令和状态。本页说明当前 robosuite 主线的边界，避免把冻结的 PyBullet 参考代码当成入口。

## 控制边界

三个入口都把 SpaceMouse 映射后的六维物理增量交给世界坐标系 `OSC_POSE`：`Tx Ty Tz Rx Ry Rz`。死区、比例、方向、平滑、限幅和超时都由 [spacemouse_wrist.json](../configs/spacemouse_wrist.json) 统一定义。

| 模型 | 总动作维度 | 机械臂切片 | 手部或夹爪切片 |
| --- | ---: | --- | --- |
| Panda + Allegro | 22 | `[0:6]` | `[6:22]`，16 个绝对关节目标。 |
| Panda + PandaGripper | 7 | `[0:6]` | `[6:7]`，一个策略维度驱动两个指头。 |
| Jaco + JacoThreeFingerGripper | 7 | `[0:6]` | `[6:7]`，一个策略维度同步三个手指。 |

动作切片每次运行都从本地 robosuite composite controller 查询；新夹爪入口不复制 Allegro 的索引。HID 无效时机械臂和夹爪均输出零动作。`R` 清空状态并重建环境；`Q`、`Ctrl+C` 先发送最后一个零机械臂动作，再关闭资源。

## 同步三视图和激光

[visualization.py](../src/wrist_teleop/visualization.py) 在一个已经步进的 MuJoCo 状态上连续离屏渲染：`frontview`、`robot0_eye_in_hand`、`sideview`。它们合成为一个窗口，不会启动三个控制循环。离屏缓冲在这里统一翻转为正常的窗口坐标；首帧 PNG 写到每个运行目录。

固定主视图与侧视图的位置、四元数和视场角写在 `tri_view` 配置中。腕部相机保留附着在 wrist 的性质，只配置视场角。关闭合成窗口会通过同一键盘路由触发安全退出。

`LaserPointer` 从真实 `grip_site` 读取当前位置和旋转，用局部 `+Z` 得到世界方向。它用 `mj_ray` 跳过机器人自己的几何体，命中桌面或场景后把既有透明 cylinder site 画成红线。运行时只修改该 site 的可视字段，关闭时精确恢复；测试比较了同种子动作下有无激光的 `qpos`、`qvel` 和接触数。

## Panda + Allegro 手部输入

[hand_input.py](../src/wrist_teleop/hand_input.py) 保留摄像头工作线程，但不允许它写入机械臂动作。按钮手势也只写 16 维 Allegro 目标。

正式模式 `--hand-control dex` 使用 [dex_retargeting.py](../src/wrist_teleop/dex_retargeting.py)：MediaPipe 21 个世界坐标关键点先转成官方 MANO 坐标，再按官方 vector 配置中 `[[0,0,0,0],[4,8,12,16]]` 生成四条腕到指尖向量。官方求解器输出通过关节名称而非位置索引映射到 `joint_0.0` 至 `joint_15.0`，并检查 16 项完整性、有限性与关节限位。

官方配置来自 dex-retargeting 0.4.6，未经语义修改；来源、许可证和 SHA-256 在 [PROVENANCE.md](../configs/dex_retargeting/PROVENANCE.md)。当前 Windows Python 3.10 环境没有官方 `pin>=2.7` 依赖，因而 dex 初始化会明确失败。`legacy-curl` 只能显式选择，不能作为 dex 失败时的自动回退。

dex 摄像头短时丢失保持最后一个有效关节目标，默认最长 0.5 秒；超时后 hand gesture resolver 使用安全张手。`R` 会重置求解器和官方滤波器对象。

## 抓取遥测

[grasp_telemetry.py](../src/wrist_teleop/grasp_telemetry.py) 是三个入口共用的只读观察模块。它读取 `cube_main`、`table_collision` 和当前接触：方块抬升至少 0.03 米、脱离桌面、保持手部接触至少 3 秒且接触穿透不超过 4 毫米时记录通过。它不控制方块，也不替代 Gate 2 的静态抓取验证。

## 依赖边界

当前主线锁定 robosuite 1.5.2、MuJoCo 3.3.7、NumPy 1.26.4、hidapi 0.15.0、OpenCV 4.11.0.86 和 MediaPipe 0.10.21。`dex-retargeting==0.4.6` 作为带 Windows 平台标记的可选官方依赖写入 [requirements-robosuite-wrist.lock](../requirements-robosuite-wrist.lock)，不会在本机安装时破坏已验证的 NumPy / MediaPipe 组合。
