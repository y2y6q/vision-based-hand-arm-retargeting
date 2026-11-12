# 运行命令

整行复制到 PowerShell 运行。只有三个主命令，其他需求都是在主命令末尾加后缀（可以组合；`--` 和参数名之间不能有空格）。SpaceMouse 标定 / 诊断、3DxWare 配置、纯摄像头版本、场景预览这些一次性或备用的命令见 `Run Instructions.txt`。

## 主命令 1：SpaceMouse 遥操作（主线）

Shadow 手：SpaceMouse 控机械臂，摄像头控手指。默认 libero_spatial 任务 0，手掌平行桌面。
```powershell
cd E:\hand_arm_retargeting; .\.venv-libero\Scripts\python.exe .\scripts\run_libero_spatial_panda_shadow_teleop.py --thumb-lift-gain 1.8
```

## 主命令 2：规划器版（本周实验）

在主命令 1 的基础上，按数字键选物体，规划器把手送到物体上方，之后由你接手。只支持 libero_spatial 和 libero_object。默认手指朝下，运行中按 `M` 换成手掌平行桌面。
```powershell
cd E:\hand_arm_retargeting; .\.venv-libero\Scripts\python.exe .\scripts\run_libero_panda_shadow_planner_teleop.py --suite libero_object
```

## 主命令 3：数据采集

在主命令 2（规划器版）上加了采集功能，操作和命令 2 一样：数字键 1–9 让规划器把手送到物体上方，动 SpaceMouse 立刻接管，`0` 取消，`L` 开 / 关五根指尖射线，`M` 换手的构型（摆放不变）。默认手指朝下（加 `--hand-mount forward` 就直接从手掌平行桌面开始，不用每次按 `M`），只支持 libero_spatial 和 libero_object，固定实时（不能慢放）。右上角特写就是命令 2 的特写按两次 `V` 的视角（相机在机器人左侧照着手），固定不转，只把画面左右镜像。每个任务采 10 条：窗口右下角逐条显示这个任务的成功条件，官方成功判定连续 0.5 s 成立就算成功，换一个随机摆放接着采；一个任务成功 10 次后自动换下一个任务。输出只有两样，都在 `outputs\libero_collect\<套件>\`：`attempts.csv`（每一轮一行：成功 / 失败 / 中途退出、用时、摆放编号、手的构型、怎么结束的、视频文件名）和 `videos\`（只有按过 `S` 的那几轮有视频，就是操作时看到的整个窗口，包括摄像头小窗；上方一条标题栏显示计时，结尾 2 s 显示 SUCCESS / FAILED / QUIT 和用时）。按 `S` 才开始录，从按下那一刻录到这一轮结束（在手臂第一次动之前按的，从第一次动之前 1 s 开始）；不按就完全不录。录的时候先写在这次运行的日志文件夹 `outputs\libero\<时间>_collect_<套件>\` 里，这一轮结束才移进 `videos\`；录到一半再按 `S` 就取消，已录的部分删掉。CSV 照样每一轮一行，成功判定和成功率都不变，没录视频的那一行 `video` 一栏是空的。手臂还没动就按 `R` 不算一轮，不留记录。中途退出后再运行同一条命令就是接着采。采集时先在 Excel / WPS 里关掉 `attempts.csv`：开着会锁住文件，结果先存在内存里，关掉后下一轮结束时补写；退出时还开着就另存成 `attempts_unsaved_<时间>.csv`，下次运行接不上。要训练用的原始 demo 时加 `--save-demos`。说明见 [docs/libero_data_collection_zh.md](docs/libero_data_collection_zh.md)。
```powershell
cd E:\hand_arm_retargeting; .\.venv-libero\Scripts\python.exe .\scripts\run_libero_collect.py --suite libero_object --thumb-lift-gain 1.8
```

只看任务列表、成功规则和已成功次数 / 成功率（不启动仿真）：
```powershell
cd E:\hand_arm_retargeting; .\.venv-libero\Scripts\python.exe .\scripts\run_libero_collect.py --suite libero_object --list-tasks
```

把 `--save-demos` 存下的原始 demo 做成 LIBERO 格式数据集（命令 3 结束时不再自动运行，需要时手动运行）：
```powershell
cd E:\hand_arm_retargeting; .\.venv-libero\Scripts\python.exe .\scripts\build_libero_dataset.py --suite libero_object
```

## 按键（在仿真窗口里按）

| 按键 | 作用 | 命令 |
| --- | --- | --- |
| `V` | 转动特写视角（命令 3 的特写固定为命令 2 按两次 `V` 的视角并左右镜像，按 `V` 不转） | 1、2 |
| `R` | 复位，任务回到初始状态。画面会停约 2.5 s，主视图中间显示 `RESETTING - please wait`；这段时间里和画面恢复后 0.5 s 内再按 `R` 都不起作用（以前会再复位一次） | 1、2 |
| `R` | 放弃这一轮，换一个新的随机摆放。手臂已经动过（SpaceMouse 或规划器）的话，这一轮记为失败（存了视频的话结尾标 FAILED）；还没动就不算一轮，不留记录和视频。复位期间按 `R` 同样不起作用 | 3 |
| `Q` / 关窗口 / Ctrl+C | 安全退出。命令 3 里还在进行的一轮记为中途退出（存了视频的话结尾标 QUIT），退出时打印每个任务的成功次数和成功率 | 1、2、3 |
| `1`–`9` | 规划器把手送到这个编号的物体上方：最低指尖高出物体顶面 2 cm，手的朝向不变。水平位置：手掌平行桌面（掌心朝下）时，食指、中指、无名指三个指根的中心对准物体中心，物体在指根下面；手指朝下时，抓取中心（拇指尖和食指、中指指尖平均位置的中点）对准物体中心。手里拿着东西时，改为让物体对准目标、底部高出目标 2 cm | 2、3 |
| 动 SpaceMouse | 规划器立刻停下，SpaceMouse 接管 | 2、3 |
| `0` | 取消规划 | 2、3 |
| `L` | 开 / 关五根指尖的红色射线 | 2、3 |
| `M` | 切换手的构型：手指朝下 ↔ 手掌平行桌面。会重建场景（约 4 s），期间显示 `SWITCHING HAND MOUNT`，按 `R` 不起作用。命令 2 里任务从头开始；命令 3 里摆放不变，这一轮手臂动过的话记为失败 | 2、3 |
| `S` | 开始录这一轮的视频：从按下那一刻录到这一轮结束，结束时存进 `videos\`。录的时候再按一次就取消（已录的删掉），再按又从那一刻重新开始录。每一轮开始时都不录；想录整轮，就在手臂第一次动之前按（视频从第一次动之前 1 s 开始）。主视图右下角面板里，红字 `REC  recording` 表示正在录，灰字 `not recording` 表示没在录 | 3 |

## 后缀

### 三个命令都能用

| 后缀 | 意思 |
| --- | --- |
| `--suite libero_object` | 换套件。命令 1 有 libero_spatial / libero_object / libero_goal / libero_10 / libero_90，命令 2、3 只有前两个 |
| `--hand-mount down` | 手的构型：`forward` = 手掌平行桌面、手指朝前；`down` = 手指朝下。命令 1 默认 forward，命令 2、3 默认 down |
| `--arm-speed 4 --rot-speed 2.5` | SpaceMouse 控臂的平移 / 旋转速度倍数（0.1–5，默认 3.0 / 2.0） |
| `--object-mass-scale 10` | 所有任务物体质量乘以这个数（0.1–100） |
| `--object-mass AkitaBlackBowl=300 butter_2=110` | 单独设某几个物体的质量（克，1–20000）。名字可以写类名（见 [libero_tasks_object_masses.md](docs/libero_tasks_object_masses.md)），也可以写实例名（如 `butter_2`，同时写时实例名优先）；没点名的按 `--object-mass-scale`；名字不在本任务里会报错，并列出本任务的物体 |
| `--thumb-lift-gain 1.8` | 拇指抬离手掌（朝摄像头）的放大倍数（0.5–4，默认 1.5，1 = 按摄像头原样） |
| `--dex-official` | 手指用未改动的官方 DexPilot，对比用（不能和 `--thumb-lift-gain` 同时用） |
| `--hand-delay-ms 80` | 每拍用多少毫秒之前的手（0–300，默认 65；0 = 直接用最新一帧） |
| `--hand-control legacy-curl` | 手指控制方式：`dex`（默认）/ `legacy-curl`（旧的弯曲映射）/ `off`（手不动） |
| `--camera-index 1` | 换摄像头（默认 0） |
| `--camera-no-mirror` | 摄像头画面不做镜像 |
| `--no-camera-display` | 仿真窗口里不显示摄像头小窗 |
| `--view-size 1920x1080` | 仿真窗口大小（默认铺满主屏） |
| `--duration 120` | 运行这么多秒后自动退出 |

### 只对命令 1、2

| 后缀 | 意思 |
| --- | --- |
| `--task-id 4` | 换任务，任务号见下表（命令 3 用 `--tasks`） |
| `--init-state-id 3` | 同一任务换物体的初始摆放（0–49）。命令 3 不用这 50 个官方摆放，留给以后评测 |

### 命令 2、3 的规划器后缀

| 后缀 | 意思 |
| --- | --- |
| `--planner-speed 0.4` | 规划器移动的峰值速度（m/s，0.05–0.5，默认 0.30；按仿真时间算，慢放时现实里也跟着变慢） |
| `--sim-speed 0.6` | 只对命令 2。仿真慢放：现实里每 50 ms 一拍，仿真只走 30 ms。可选 1 / 0.8 / 0.6 / 0.4，默认 1 = 实时。命令 3 固定实时（数据按 LIBERO 的 20 Hz 记） |
| `--planner straight` | 对照组：直接走"抬高 → 平移 → 下降"三段直线，不检查碰撞。默认 `ompl`：先检查几条直线路线，都走不通时才用 OMPL 搜索 |
| `--clearance-cm 3` | 到位时最低指尖离物体顶面的高度（cm，0–20，默认 2） |
| `--no-rays` | 启动时不显示指尖射线（`L` 可以再打开） |

### 只对命令 3（数据采集）

| 后缀 | 意思 |
| --- | --- |
| `--tasks 0-4` | 只采这些任务：`all`（默认）、`3`、`0-4`、`1,3,7-9`。任务号见下表 |
| `--demos-per-task 5` | 每个任务成功几次后换下一个任务（默认 10） |
| `--list-tasks` | 只打印任务、成功规则、已成功次数和成功率，不启动 |
| `--seed 8` | 随机摆放的基础种子（默认 7）。同一个种子、同一个编号的摆放总是一样的 |
| `--operator 名字` | 写进 CSV 每一行的操作者名字（可选） |
| `--save-demos` | 另外把每次成功的原始 demo（仿真状态和动作，训练用）存到 `outputs\libero_collect_demos\<套件>\`。默认不存 |
| `--share-dir C:\Users\lenovo\OneDrive\libero_collect` | 结束后把 `attempts.csv` 和视频复制到这个会自动同步到云端的文件夹（默认读环境变量 `LIBERO_SHARE_DIR`） |
| `--share-url 链接` | 这个文件夹的共享链接，结束时打印出来（默认读环境变量 `LIBERO_SHARE_URL`） |

`build_libero_dataset.py`（只处理 `--save-demos` 存下的数据）的后缀：`--image-size 128`（换分辨率，数据集按新尺寸重做，覆盖旧的）、`--force`（视频和数据集全部重做）、`--partial`（没采满的任务也生成数据集）、`--exclude-disturbed`（数据集里不要碰倒过其他物体的 demo）、`--share-datasets`（数据集也复制到分享文件夹）、`--share-dir` / `--share-url`（同上）。

### task-id 对照

| task-id | libero_spatial（黑碗放到盘子上，碗的起始位置） | libero_object（放进篮子） |
| --- | --- | --- |
| 0 | 盘子和小烤碗之间 | 字母汤罐头 |
| 1 | 小烤碗旁边 | 奶油奶酪 |
| 2 | 桌子中间 | 沙拉酱 |
| 3 | 饼干盒上 | 烧烤酱 |
| 4 | 木柜最上层抽屉里 | 番茄酱 |
| 5 | 小烤碗上 | 番茄沙司 |
| 6 | 饼干盒旁边 | 黄油 |
| 7 | 炉子上 | 牛奶 |
| 8 | 盘子旁边 | 巧克力布丁 |
| 9 | 木柜顶上 | 橙汁 |

## 分享文件夹（命令 3，一次设置）

设一次就一直有效（设完要新开一个 PowerShell 窗口）。第二行的链接在 OneDrive 网页上对这个文件夹点"共享"后复制：
```powershell
setx LIBERO_SHARE_DIR "C:\Users\lenovo\OneDrive\libero_collect"
```
```powershell
setx LIBERO_SHARE_URL "这里换成复制的共享链接"
```

## 找不到外部环境时

只对当前终端有效。OMPL（命令 2、3 的规划器）：
```powershell
$env:OMPL_PLANNER_PYTHON = "E:\Anaconda3-2024.10-1-Windows-x86_64\envs\ompl_planner\python.exe"
```

dex 手指映射：
```powershell
$env:DEX_RETARGETING_PYTHON = "E:\Anaconda3-2024.10-1-Windows-x86_64\envs\dexretarget\python.exe"
```
