# LIBERO 数据采集入口：流程、输出和结构

2026-11-08 建立；2026-11-09 改为建在规划器版（主命令 2）上，同一天按要求把输出简化成一张 CSV 表加每一轮的视频。2026-11-10：正在录的视频先写在运行日志文件夹，录完才移进 `videos/`；规划器在手掌平行桌面时改为让食指、中指、无名指的指根对准物体（主命令 2、3 一起改）。2026-11-10 晚：按 `S` 才开始录这一轮的视频，不按就不录；CSV、成功判定和成功率不变。命令见 [RUN_COMMANDS.md](../RUN_COMMANDS.md) 主命令 3。

这个入口建在主命令 2（规划器版）上，操作和命令 2 一样：SpaceMouse 控机械臂，摄像头控 Shadow 手指；按数字键 1–9，规划器把手送到这个编号的物体上方，一动 SpaceMouse 规划器就当拍停下、由 SpaceMouse 接管，`0` 取消；五根指尖红色射线（`L` 开关）；视图里有物体编号；`M` 切换手的构型。采集功能是在这上面加的。每一拍手臂只听一方（SpaceMouse 或规划器）；手指始终来自摄像头。

输出只有两样，都在 `outputs/libero_collect/<套件>/`：`attempts.csv`（每一轮一行）和 `videos/`（按过 `S` 的那几轮各一个视频，从按下 `S` 录到这一轮结束，就是操作时看到的整个窗口，上方显示计时，结尾显示成功 / 失败和用时）。详见第 5 节。

## 1. 整体流程

一次运行（例如 `--suite libero_object`）按下面的顺序走：

1. **找任务。** 读 `outputs/libero_collect/<套件>/attempts.csv`，数出每个任务已经成功了几次（默认每个任务要 10 次），从第一个还没满的任务开始。已经满的任务跳过，所以中途退出后再运行同一条命令就是接着采，轮次编号和摆放编号也接着往下数。
2. **摆场景。** 打开这个任务（Panda + Shadow，默认手指朝下，和命令 2 一样），用一个随机种子摆放物体。摆放规则是 LIBERO 自己的（BDDL 里写的摆放区域，大多数物体在几厘米范围内变化，libero_object 有几个物体位置固定，见第 8 节），和官方 50 个评测初始状态同分布，但不是那 50 个。这样那 50 个以后还能留作评测。同一个种子、同一个编号的摆放总是一样的。
3. **等物体落稳。** LIBERO 摆放时物体是从几厘米高处放下的，刚摆好的 0.3–1 s 里会下落、滑动 5–10 cm。所以摆好后机械臂先保持不动（手张开），等所有物体的速度连续 3 步低于 5 mm/s（最多等 3 s 仿真时间），这一轮才开始。LIBERO 官方评测在复位后也会这样空等几步。这几步不记录，期间主视图显示复位提示。
4. **等开始。** 屏幕显示 `waiting for the SpaceMouse / 1-9`。手臂第一次动起来时开始计时：动 SpaceMouse，或者按数字键后规划器开始带着手臂走。从按下数字键到规划出路线（一般零点几秒）手臂不动，这段不算开始。只计时，不设超时。视频从开始前 1 s 录起，这 1 s 标题栏显示黄色 `READY`。
5. **每一步（20 Hz）。**
   - 窗口当前的画面（三个视图、摄像头小窗、状态栏和目标面板，和你看到的一样）录进这一轮的视频，上方标题栏显示计时。
   - 按这个任务的官方成功规则检查，窗口里逐条显示是否满足。
   - 检查不是目标的物体有没有被碰倒或推开。被碰的话显示警告，但不影响成功判定。
6. **判成功。** 成功规则连续成立 10 步（0.5 s）就算成功，然后：
   - 视频结尾停 2 s：标题栏变成绿色 `SUCCESS 用时`，画面加绿框；CSV 加一行；
   - 屏幕显示 `SUCCESS - k/10 for this task (用时)`；
   - 自动换一个新的随机摆放，开始下一轮。
7. **换任务。** 一个任务成功满 10 次后，自动加载下一个任务。全部满了就自动退出。
8. **收尾。** 退出时（全部完成或按 Q），还在进行的一轮记为中途退出，视频结尾标 `QUIT`。等最后的视频写完（一般不到 1 s），打印每个任务的成功次数、成功率和用时中位数。设置了分享文件夹的话，把 `attempts.csv` 和视频复制进去。

```text
启动 ─> 找下一个没满的任务 ─> 打开场景 ─> 随机摆放，等物体落稳 ─> 等手臂第一次动（SpaceMouse 或 1–9，开始计时）
                ^                                ^                             │
                │                                │                             v
                │                                │        每步：画面录进视频，检查成功，检查碰倒
                │                                │                    │
                │         R / M：这一轮记失败（视频标 FAILED）────────┤
                │                                                     │ 连续 10 步成功
                │                                                     v
                └─── 成功满 10 次：下一个任务 <─── 视频标 SUCCESS，CSV 记一行，自动换摆放
Q / 全部完成 ─> 还在进行的一轮记 QUIT ─> 打印汇总 ─> 复制到分享文件夹
```

### 按键（在仿真窗口里按）

| 键 | 作用 | 这一轮怎么算 |
| --- | --- | --- |
| `1`–`9` | 规划器把手送到这个编号的物体上方（和命令 2 一样：最低指尖高出物体顶面 2 cm，手的朝向不变） | 规划器开始带手臂走，就算开始 |
| 动 SpaceMouse | 规划器当拍停下，SpaceMouse 接管 | — |
| `0` | 取消规划 | — |
| `L` | 开 / 关五根指尖射线 | — |
| `S` | 开始录这一轮的视频（从按下那一刻到这一轮结束）；录的时候再按一次取消，已录的删掉。每一轮开始时都不录，主视图面板里有一行显示在不在录（见 5.3） | 不影响结果，只决定有没有视频 |
| `R` | 放弃这一轮，换一个新的随机摆放 | 手臂已经动过：记为失败，视频标 FAILED；还没动：不算一轮，不留记录和视频 |
| `M` | 切换手的构型（手指朝下 ↔ 手掌平行桌面），重建场景，摆放不变（约 4 s） | 同上 |
| `Q` / 关窗口 / Ctrl+C | 退出，打印汇总 | 已经动过：记为中途退出，视频标 QUIT，不计入成功率 |
| `V` | 不起作用（特写固定为命令 2 按两次 `V` 的视角，见第 3 节） | — |

复位和切换构型期间按 `R` 无效，主视图会显示提示（和主线一样）；这期间按的数字键、`L`、`M`、`S` 也会丢掉。换摆放、切换构型、判定成功和退出之前，正在进行的规划都会先取消。

### 一轮的结果和成功率

| 结果 | 什么时候 | CSV 和视频 | 计入成功率 |
| --- | --- | --- | --- |
| `success` | 成功规则连续 10 步成立 | 有 CSV 行；按过 `S` 才有视频，结尾绿色 `SUCCESS` | 计入 |
| `failed` | 手臂动过之后按了 R 或 M | 有 CSV 行；按过 `S` 才有视频，结尾红色 `FAILED` | 计入 |
| `quit` | 手臂动过之后退出（Q、关窗口、Ctrl+C、`--duration` 到时） | 有 CSV 行；按过 `S` 才有视频，结尾灰色 `QUIT` | 不计入 |
| 没开始 | 手臂还没动就按了 R / M / Q | 没有 | 不计入 |

成功率 = 成功 / (成功 + 失败)，按 `attempts.csv` 里这个任务的所有行算，所以跨多次运行累计。窗口里实时显示当前任务的成功次数和成功率。

## 2. 成功判断：每个任务一个检查器

### 2.1 规则从哪来

LIBERO 每个任务的成功条件写在它的 BDDL 文件的 `(:goal ...)` 里。官方的 `env._check_success()` 就是把这些条件逐条检查，全部满足才算成功。五个套件用到的 6 个场景类里，这个函数的写法完全一样（已逐个核对源码）。

所以采集入口不手写判断函数。每次加载任务时，从这个任务自己的 BDDL 目标生成它的检查器：

- 每条条件单独检查，用来实时显示：每条一行，满足是绿色 `[OK]`，不满足是灰色 `[--]`。
- 最终判定直接调用官方的 `env._check_success()`，和 LIBERO 评测完全一致。
- 逐条结果和官方结果不一致时写日志。理论上不会发生，测试里逐个任务核对过。

不手写 130 个函数，是因为手写容易和官方评测对不上，那样采到的"成功"在评测里可能不算。

### 2.2 条件的意思（按源码）

| 条件 | 成立的意思 |
| --- | --- |
| `in(A, 区域)` | A 的中心点在这个区域的盒子范围内。区域是挂在容器上的一个空间范围，比如 `basket_1_contain_region` 是篮子里面靠底的一块 |
| `on(A, 物体B)` | A 的中心不低于 B 的中心，A 和 B 有接触，两者水平距离小于 3 cm |
| `on(A, 区域)` | A 在这个区域上方（区域自己的判断），并且 A 和区域所属的物体有接触 |
| `open` / `close` | 抽屉、柜门或微波炉门的关节转到打开 / 关闭的阈值 |
| `turnon` / `turnoff` | 炉子旋钮转到开 / 关的阈值 |

### 2.3 推荐的两个套件，每个任务的规则

**libero_object**（地面场景，6 个日用品加一个篮子）：

| 任务 | 指令 | 成功规则 |
| --- | --- | --- |
| 0 | pick up the alphabet soup and place it in the basket | `in(alphabet_soup_1, basket_1_contain_region)` |
| 1 | pick up the cream cheese and place it in the basket | `in(cream_cheese_1, basket_1_contain_region)` |
| 2 | pick up the salad dressing and place it in the basket | `in(salad_dressing_1, basket_1_contain_region)` |
| 3 | pick up the bbq sauce and place it in the basket | `in(bbq_sauce_1, basket_1_contain_region)` |
| 4 | pick up the ketchup and place it in the basket | `in(ketchup_1, basket_1_contain_region)` |
| 5 | pick up the tomato sauce and place it in the basket | `in(tomato_sauce_1, basket_1_contain_region)` |
| 6 | pick up the butter and place it in the basket | `in(butter_1, basket_1_contain_region)` |
| 7 | pick up the milk and place it in the basket | `in(milk_1, basket_1_contain_region)` |
| 8 | pick up the chocolate pudding and place it in the basket | `in(chocolate_pudding_1, basket_1_contain_region)` |
| 9 | pick up the orange juice and place it in the basket | `in(orange_juice_1, basket_1_contain_region)` |

**libero_spatial**（桌面场景，两个黑碗、盘子、饼干盒、小烤碗、木柜、炉子）：10 个任务的规则都是 `on(akita_black_bowl_1, plate_1)`，即"指定的那个黑碗放到盘子上"。区别只在这个碗一开始放在哪：盘子和小烤碗之间、小烤碗旁边、桌子中间、饼干盒上、抽屉里、小烤碗上、饼干盒旁边、炉子上、盘子旁边、木柜上。

**libero_goal**（以后再采；含开抽屉、开炉子）：

| 任务 | 指令 | 成功规则 |
| --- | --- | --- |
| 0 | open the middle drawer of the cabinet | `open(wooden_cabinet_1_middle_region)` |
| 1 | put the bowl on the stove | `on(akita_black_bowl_1, flat_stove_1_cook_region)` |
| 2 | put the wine bottle on top of the cabinet | `on(wine_bottle_1, wooden_cabinet_1_top_side)` |
| 3 | open the top drawer and put the bowl inside | `in(akita_black_bowl_1, wooden_cabinet_1_top_region)` |
| 4 | put the bowl on top of the cabinet | `on(akita_black_bowl_1, wooden_cabinet_1_top_side)` |
| 5 | push the plate to the front of the stove | `on(plate_1, main_table_stove_front_region)` |
| 6 | put the cream cheese in the bowl | `on(cream_cheese_1, akita_black_bowl_1)` |
| 7 | turn on the stove | `turnon(flat_stove_1)` |
| 8 | put the bowl on the plate | `on(akita_black_bowl_1, plate_1)` |
| 9 | put the wine bottle on the rack | `on(wine_bottle_1, wine_rack_1_top_region)` |

**libero_10**（长任务）的 10 个任务大多要同时满足两条，例如任务 0 是汤罐和番茄酱都在篮子里，任务 8 是两个摩卡壶都在炉子上并且炉子打开。libero_90 有 90 个任务，不在这里列。任何套件都可以用 `--list-tasks` 打印全部任务和规则。

### 2.4 为什么要连续 0.5 s

物体刚落进篮子时可能弹出来，碗放到盘子上时可能滑开。要求连续 10 步都满足，可以避开"碰一下就算成功"。LIBERO 官方的采集脚本也是连续 10 步。这 0.5 s 算在用时里。

### 2.5 碰倒其他物体

BDDL 只看目标物体，把旁边的瓶子碰倒也算成功（11-07 实测，4 次里有 2 次碰倒了沙拉酱瓶）。采集入口额外检查：没出现在成功规则里的物体，从开始计时那一刻算起（这时物体已经落稳，见第 1 节），只要移动超过 3 cm 或倾斜超过 30°，窗口里就显示 `MOVED: 物体名`。

这一轮照样算成功，CSV 的 `moved_objects` 列会写上被碰的物体，例如 `salad_dressing_1 (6 cm, tilted 85 deg)`。想要干净的数据，可以看到警告就按 R 重来（这一轮记为失败）。

## 3. 右上角特写：命令 2 按两次 V 的视角，固定，只做镜像

特写就是主命令 2 的特写按两次 `V` 之后的样子：相机在机器人左侧照着手（方位角 90° + 2 × 90° = 270°），距离、俯角、视野和跟随都和命令 2 一样。采集入口把它固定在这个角度（按 `V` 不转），然后只把相机画面左右镜像，方便操作。指尖射线也在特写里，跟着画面一起镜像；标签在镜像之后才写上去，所以是正的，文字也和命令 2 一样（`CLOSE-UP (follows hand)`）。其他都没改：相机、SpaceMouse 映射、主视图和右下 3D 视图都和命令 2 一样。

逐像素核对过：同一个场景里，采集入口的特写和命令 2 按两次 `V` 的特写左右翻转后完全一样（标签那一条除外，平均差 0.000）；不翻转时平均差 34（满分 255）。

镜像后，特写里的运动方向（按实际的相机位姿和 SpaceMouse 映射算出来的）：

| SpaceMouse | 主视图 | 右上角特写（镜像后） |
| --- | --- | --- |
| 往左 / 往右 | 往左 / 往右 | 往左 / 往右，但更多是靠近 / 远离镜头 |
| 往上 / 往下 | 往上 / 往下 | 往上 / 往下 |
| 往前（推离自己）/ 往后 | 往画面里 / 往画面外 | 往右 / 往左（同时有一部分是靠近 / 远离镜头） |

特写是从侧面看的，所以"前后"在这个视角里只能显示成左右。

这个改动只在采集入口里，主命令 1 和 2 的特写不变。视频里录的就是这个镜像后的特写。

## 4. 工程结构

新文件（现有脚本和共用模块都不改）：

| 文件 | 内容 |
| --- | --- |
| `src/wrist_teleop/libero_collection.py` | 纯逻辑，不依赖仿真：成功保持计数、每轮结果和成功率、`attempts.csv` 的读写（断点续采；Excel 开着文件时行也不会丢）、摆放种子、文件命名、碰倒检测、原始 demo 的记录和写文件 |
| `src/wrist_teleop/libero_collect_device.py` | 采集入口用的 SpaceMouse（`SpaceMouseHid` 的子类）：换摆放、按 M、换任务时照样清掉旧输入，但保留启动时学到的零点，新场景出来后一推就能接管（11-11） |
| `src/wrist_teleop/libero_attempt_video.py` | 每一轮的视频：窗口画面加上标题栏（任务、计时、结果），在单独的线程里交给 ffmpeg（低优先级进程）编码，不拖慢控制循环 |
| `src/wrist_teleop/libero_collect_teleop.py` | 采集主循环（照主命令 2 规划器版的主循环复制，再加上采集功能）、随机摆放和等物体落稳、带固定镜像特写和目标面板的窗口 |
| `scripts/run_libero_collect.py` | 采集的命令行入口 |
| `src/wrist_teleop/libero_dataset_build.py` | 离线处理 `--save-demos` 存下的原始 demo：从仿真状态重新渲染图像，生成 LIBERO 格式数据集、离线视频和汇总表 |
| `scripts/build_libero_dataset.py` | 离线处理的命令行入口，需要时手动运行（采集结束时不再自动运行） |
| `tests/test_libero_collection.py` | 测试，见第 7 节 |
| `tests/test_libero_collect_device.py` | 换摆放后 SpaceMouse 接管的测试，见第 7 节 |

复用、不改的现有部分：

- `OfficialLiberoSpatialSession`（采集入口里用它的子类，加了随机摆放）；
- 规划器版的 `PlannerActionComposer`、`PlannerScene`（物体编号）、`FingertipRays`（五根指尖射线）、`PlannerViewWindow`（采集入口里用它的子类，加了特写镜像和目标面板）、`ApproachPlanner` 和 OMPL 侧车、M 切换构型的提示；
- 三视图窗口（`LiberoViewWindow`：视频直接用它每一帧画好的画面，摄像头小窗就在里面）、复位提示（`ResetFeedback`）、SpaceMouse 映射、摄像头手部输入、dex 侧车。

规划器只在按数字键之后接管手臂，任何 SpaceMouse 输入当拍收回（和命令 2 一样）。窗口顶上的状态栏显示规划器的状态，视频里也看得到；加 `--save-demos` 时，每一步是谁在动手臂还会记在原始 demo 里（`control/arm_owner`）。11-08 的第一版是照主命令 1 做的、没有规划器，11-09 按要求改成了现在这样。

**换摆放后 SpaceMouse 马上能用（11-11 改）**。每次换摆放、按 M、换任务时，SpaceMouse 的输入都会清零，复位前的推动不会带到新场景里，这一点没变。以前清零时连程序启动时学到的零点也一起清掉了，要重新学：需要连续 8 个落在死区里的 HID 报告。可是这台 SpaceMouse Wireless 不碰时一个报告都不发，一推又马上超出死区，计数又清零。所以每换一个摆放都要轻碰几下、等三四秒，手臂才肯动（规划器带到位后想接着微调就卡在这里）。零点是设备本身的偏置，和摆放无关，现在 `libero_collect_device.py` 换摆放时把它留着：新场景出来 0.3 s 后，SpaceMouse 一推手臂就动，规划器在走的话当拍收回。复位时还按着没松手的，要先松开一次才接管。程序刚启动时还是要先学一次零点（手轻放在 SpaceMouse 上一下就行），断开重连后也会重新学。

动作的写入隔离不变：手臂 `[0:6]`，Shadow 手 `[6:26]`。控制频率还是 20 Hz，仿真速度固定 1×（命令 2 的 `--sim-speed` 慢放在采集入口里没有，因为要按 LIBERO 的 20 Hz 记），物理参数不变。冻结基线不动。

OMPL 后端每换一次模型，就要把场景编译成一个约 250 MB 的临时文件交给规划器进程。libero_object 换随机摆放时模型不变（实测同一个任务各个摆放的模型 XML 完全一样），所以沿用同一个文件，规划器也不用重新加载；libero_spatial 的柜子、炉子等固定件每次摆放位置不同，每次重新生成。退出时删掉。

## 5. 输出

### 5.1 目录

```text
outputs/libero_collect/
  libero_object/
    attempts.csv     每一轮一行（成功、失败、中途退出）
    videos/          按过 S 的每一轮一个 MP4，文件名和 CSV 的 video 列一样
  libero_spatial/
    …
```

采集文件夹里只有这两样。另外两处平时不用看：

- 每次运行的日志（`metadata.json`、`telemetry.jsonl`，排查问题用）在 `outputs/libero/<时间>_collect_<套件>/`，和其他入口的日志放在一起；
- 加了 `--save-demos` 才会有 `outputs/libero_collect_demos/<套件>/`，见 5.4。

### 5.2 `attempts.csv`

用 Excel 直接打开就行（UTF-8 带 BOM，不会乱码）。手臂动过的每一轮一行，列：

| 列 | 内容 |
| --- | --- |
| `attempt` | 第几轮。同一个套件里从 1 往上数，跨运行连续，和视频文件名里的 `a012` 对应 |
| `started_at` | 这一轮场景摆好的时间（本机时间） |
| `task_id` | 任务号 |
| `task` | 任务指令，例如 `pick up the alphabet soup and place it in the basket` |
| `hand_mount` | 手的构型：`down`（手指朝下）/ `forward`（手掌平行桌面） |
| `layout` | 随机摆放的编号。同一个任务里从 0 往上数；同一个种子、同一个编号的摆放总是一样的；按 M 换构型时不变 |
| `result` | `success` / `failed` / `quit` |
| `time_s` | 用时（秒）：从手臂第一次动开始，到判定成功（含最后保持的 0.5 s）、按 R / M 或退出为止。就是视频里计时器的数 |
| `ended_by` | 怎么结束的：`goal held 0.5 s`（成功）、`R: new layout`、`M: hand mount switch`、`Q: quit`、`window closed`、`Ctrl+C`、`time limit`（`--duration` 到时） |
| `moved_objects` | 被碰倒 / 推开的其他物体（见 2.5），没有就空着 |
| `video` | 这一轮的视频文件名（在 `videos/` 里）；没按 `S`、没录视频的轮次是空的 |
| `operator` | `--operator` 写的名字，没写就空着 |

实际输出的样子（无窗口模拟运行，输入是脚本给的）。一次运行里先按 R 放弃一轮，下一轮到 `--duration` 时间结束：

```text
attempt,started_at,task_id,task,hand_mount,layout,result,time_s,ended_by,moved_objects,video,operator
1,2026-11-09 01:36:28,0,pick up the alphabet soup and place it in the basket,down,0,failed,9.07,R: new layout,,libero_object_t00_a001_FAILED_9.1s__pick_up_the_alphabet_soup_and_place_it_in_the_basket.mp4,
2,2026-11-09 01:36:39,0,pick up the alphabet soup and place it in the basket,down,1,quit,9.68,time limit,,libero_object_t00_a002_QUIT_9.7s__pick_up_the_alphabet_soup_and_place_it_in_the_basket.mp4,
```

另一次运行，把汤罐直接放进篮子触发成功：

```text
attempt,started_at,task_id,task,hand_mount,layout,result,time_s,ended_by,moved_objects,video,operator
1,2026-11-09 01:44:37,0,pick up the alphabet soup and place it in the basket,down,0,success,2.32,goal held 0.5 s,,libero_object_t00_a001_SUCCESS_2.3s__pick_up_the_alphabet_soup_and_place_it_in_the_basket.mp4,sim-test
```

成功次数和成功率都按表里的行算，所以手动删掉一行，那一轮就不算了。程序自己只往后追加，不改已有的行。

Excel 或 WPS 打开 CSV 时会锁住文件。这时结束的一轮照样计数（窗口里的成功次数和成功率照常更新），这一行先留在内存里，文件关掉之后，下一轮结束时连同之前没写进去的一起写；退出时还锁着的话，就把整张表另存成 `attempts_unsaved_<时间>.csv`，终端里会提示。另存的那些行不在 `attempts.csv` 里，下次运行的成功次数和轮次编号就接不上了，所以采集时先关掉 Excel / WPS，要看结果等退出之后再开。

2026-11-10 晚的两次实采都是第一轮成功后就退出了，原因就在这里：`attempts.csv` 开在 WPS 里，成功那一行按上面的办法留在内存里，但提示这件事的那句话是往 stderr 打的，它报了 `OSError: [WinError 6] 句柄无效`，采集循环就结束了；退出前的两句 stderr 提示又各报一次，所以终端里什么都没有（成功那一行在 `attempts_unsaved_*.csv` 里）。stderr 坏掉的原因：摄像头线程导入 mediapipe 时会导入 sounddevice，它启动 PortAudio 时先把 fd 2 指到 NUL，再换回一个新句柄，原来的控制台句柄被关掉了，而 Python 3.8 控制台里的 `sys.stderr` 还拿着旧句柄。只在真实控制台里出现（测试和管道里 stderr 不是控制台）。修法：入口脚本 `run_libero_collect.py` 一启动就先导入 sounddevice，fd 2 变了就把 `sys.stderr` 重新打开到新句柄上；之后摄像头线程再导入时什么也不会变。

### 5.3 视频

文件名例如：

`libero_object_t00_a012_SUCCESS_16.4s__pick_up_the_alphabet_soup_and_place_it_in_the_basket.mp4`

依次是：套件、任务号、第几轮（和 CSV 的 `attempt` 一样）、结果（SUCCESS / FAILED / QUIT）、用时、任务名（太长时截断）。

画面就是你操作时看到的整个窗口：主视图、右上特写（镜像后的）、右下 3D 视图、左下摄像头小窗（不打码）、状态栏、目标面板、射线和编号都在，只是缩小到 1920 宽（窗口 2544×1334 时视频是 1920×1088）。窗口中间临时出现的大字提示（RESETTING、SUCCESS 等）不进视频。

窗口上方加了一条标题栏：

- 左边是任务（`TASK 0: pick up ...`），下面一行是套件、第几轮、手的构型；
- 右边是大号计时，从手臂第一次动开始；开始前显示黄色 `READY`；
- 这一轮结束时，标题栏变成绿色 `SUCCESS 16.4 s`、红色 `FAILED 7.3 s` 或灰色 `QUIT 3.0 s`，后面写上怎么结束的（和 CSV 的 `ended_by` 一样），画面加同色边框，停 2 s。

每个控制拍（20 Hz）录一帧，所以是实时速度；H.264，20 fps。从手臂第一次动之前 1 s 录到这一轮结束。手臂还没动就按 R / M / Q 的轮次没有视频。换摆放后新场景刚出来的约 0.5 s 里，窗口还停在旧画面上，这段不录。

2026-11-10 晚起，按 `S` 才开始录：从按下那一刻录到这一轮结束；在手臂第一次动之前按的，从第一次动之前 1 s 开始（和以前每轮都录时一样）。不按就完全不录，不开 ffmpeg，也不写文件。录的时候写在这次运行的日志文件夹里（`outputs/libero/<时间>_collect_<套件>/_recording_<进程号>_<n>.mp4`），这一轮结束（成功、R、M、Q）时改成正式文件名移进 `videos/`，所以 `videos/` 里只有录完的视频。录的时候再按一次 `S` 就取消：ffmpeg 停下、文件删掉；再按又从那一刻重新开始录。没录的轮次，CSV 那一行的 `video` 一栏空着。

每一轮开始时都不录（R、M 之后的新一轮也是）。主视图右下角的面板里，计时那一行下面：灰字 `not recording   S: record this attempt from now` 表示没在录，红字 `REC  recording, saved when this attempt ends   S: cancel` 表示正在录；终端也会打印一行，结束时打印视频路径。这一行在窗口里，所以录下的视频里每一帧都是这行红字。计时那一行原来写 `recording`，现在不一定录，改成了 `running`。中途按 `S` 的话，视频第一帧的计时就是按下时的用时（比如 12.3 s），不是 0。成功判定一成立这一轮就结束了，之后按的 `S` 不算这一轮（复位期间按的丢掉，复位完按的算下一轮）。按住 `S` 不放（键盘自动重复）只算一次。

同一天傍晚先做过另一种：每一轮都在后台录，结束前按过 `S` 才留下，没按就删掉，好处是留下的总是整轮。用户看过后选了"改成按 S 才开始录"。

11-10 下午以前，录制中的文件直接放在 `videos/` 里；ffmpeg 还开着它的时候，资源管理器显示 0 KB，看起来像一个空视频。程序意外中断时，没录完的文件留在日志文件夹里，不进 `videos/`。

视频边录边编码：控制循环只是把窗口这一帧交出去（不到 1 ms），编码在单独的线程和低优先级的 ffmpeg 进程里做，每一轮结束后 1 s 左右文件就写好。编码跟不上时宁可丢帧也不拖慢控制（实测没丢过，见第 7 节）。大小：合成摄像头画面时约 0.6 MB / 10 s；11-10 实采（真实摄像头）约 1 MB / 10 s（46 s 的那轮 4.95 MB，挂了 17 分钟的那轮 30.9 MB）。

### 5.4 原始 demo 和 LIBERO 格式数据集（只在 `--save-demos` 时）

以后要训练时才需要。加 `--save-demos` 后，每次成功另外存一个原始 demo：

```text
outputs/libero_collect_demos/
  libero_object/
    task00_pick_up_the_alphabet_soup_and_place_it_in_the_basket/
      demo_000.hdf5 …        每次成功一个文件
    attempts.jsonl            每一轮一行，字段比 CSV 多（摆放种子、规划器使用情况、碰倒的数值等），给离线处理用
```

之后运行 `scripts/build_libero_dataset.py`（默认就读这个文件夹），在同一个文件夹里生成 `datasets/`（LIBERO 格式数据集）、离线渲染的视频（1024×640，从仿真状态重新渲染，没有摄像头画面）和 `SUMMARY.md` / `summary.csv`。

**原始 demo `demo_NNN.hdf5`** 只存仿真状态、动作、人的输入和每步谁在动手臂，不存图像，图像之后从状态重新渲染。实测一条 10.5 s 的 demo 是 0.25 MB。T 是步数（20 步 = 1 s）。

| 字段 | 形状 | 内容 |
| --- | --- | --- |
| `actions` | (T, 26) float64 | 每步发给环境的动作。`[0:6]` 是手臂 OSC 增量（米 / 弧度，世界坐标），`[6:26]` 是 Shadow 手 20 个驱动器的归一化指令 |
| `states` | (T, S) float64 | 第 t 步动作之前的完整仿真状态 `[时间, qpos, qvel]`，和 LIBERO 官方 demo 的 `states` 一样。S 随场景不同，libero_object 任务 0 是 154 = 1 + 80 + 73 |
| `final_state` | (S,) float64 | 最后一步之后的状态（成功的那一刻） |
| `success` | (T,) | 第 t 步之后官方 `_check_success()` 的结果 |
| `goal_status` | (T, K) | 第 t 步之后每条成功条件的结果（K = 条件数） |
| `human/spacemouse_axes` | (T, 6) | SpaceMouse 校准后的 6 个轴 |
| `human/spacemouse_buttons` | (T, 8) | 按键 |
| `human/arm_delta` | (T, 6) | SpaceMouse 映射出的手臂增量（滤波、死区、限幅之后） |
| `human/hand_targets` | (T, 20) | 摄像头算出的 Shadow 手目标关节角 |
| `human/hand_valid` | (T,) | 这一步摄像头手部跟踪是否有效 |
| `human/mano_landmarks` | (T, 21, 3) | MediaPipe 手部 21 个关键点（米，跟踪无效时为 NaN） |
| `control/arm_owner` | (T,) uint8 | 第 t 步的手臂动作是谁给的：0 = SpaceMouse，1 = 规划器（按数字键后，从路线规划好开始，到到位、取消或被 SpaceMouse 收回为止） |
| `time/wall_s`, `time/sim_s` | (T,) | 从摆好场景开始的真实时间 / 仿真时间 |
| `model_xml` | (约 85000,) uint8 | 这个场景的 MuJoCo 模型 XML（之后离线渲染要用） |

文件属性（attrs）里有：格式版本 `format`（`libero_shadow_teleop_demo_v2`）、套件、任务号、任务名、语言指令、BDDL 文件、手的构型、摆放方式和种子、动作各段的含义、控制频率、`state_layout`（S 怎么拆）、成功规则和保持步数、开始计时的步号 `first_input_step`、按步数算的用时 `duration_s`、真实时间的用时 `wall_duration_s`（和 CSV 的 `time_s` 一样）、`total_s`（从摆好场景开始）、落稳用了几步 `settle`、碰倒的物体、规划器的使用情况 `planner` 和 `planner_events`、`arm_owner_codes`、操作者主视图相机位姿、各关节在 qpos 里的位置、软件版本、采集时间。

**LIBERO 格式数据集 `datasets/<任务名>__<构型>_demo.hdf5`** 和 LIBERO 官方数据集的层级一样（`create_dataset.py` 生成的那种）：

```text
data            attrs: env_name, problem_info, bddl_file_name, env_args, num_demos, total, …
  demo_0        attrs: num_samples, model_file, init_state, source_demo, duration_s, …
    actions          (T, 26)
    states           (T, S)
    rewards, dones   (T,)        最后一步为 1，其余为 0
    robot_states     (T, 31)     手指关节(24) + 末端位置(3) + 末端四元数(4)
    arm_owner        (T,)        uint8，0 = SpaceMouse，1 = 规划器
    obs/agentview_rgb    (T, 256, 256, 3)  uint8
    obs/eye_in_hand_rgb  (T, 256, 256, 3)  uint8
    obs/joint_states     (T, 7)
    obs/gripper_states   (T, 24)   Shadow 手 24 个关节
    obs/ee_pos (T, 3), obs/ee_ori (T, 3), obs/ee_states (T, 6)
  demo_1 …
```

- 每条从手臂第一次动的那一步开始（开头没动的部分去掉），到成功判定为止。
- 图像方向沿用 LIBERO 官方数据的约定：上下颠倒，和环境自己的相机观测一样，现成的 LIBERO 读取 / 转换脚本可以直接用。
- 和官方数据的区别：动作是 26 维（官方 7 维），`gripper_states` 是 24 维（官方 2 维）。所以不能和官方 Panda 夹爪的数据直接混在一起训练。
- 大小：256×256 时每秒 demo 约 4 MB，一个套件 100 条约 6 GB；128×128 大约是它的 1/4（`--image-size 128`）。数据集随时可以从原始 demo 重新生成。图像是离线渲染的，所以分辨率对实时遥操作没有影响；离线渲染的图和环境自带相机拍的图平均每个像素只差 0.1–0.2（满分 255）。

## 6. 云端分享

视频和 CSV 都在本地。云端分享用"同步文件夹"的办法，不需要 API 密钥：

1. 指定一个会自动同步到云端的文件夹：
   - 这台电脑的 OneDrive（个人账户）已经登录并在运行，可以直接用 `C:\Users\lenovo\OneDrive\libero_collect`；
   - 想用 Google Drive，需要先安装 Google Drive for desktop 并登录（这台电脑还没装），然后用 `G:\My Drive\libero_collect` 这样的路径。
2. 采集结束后，脚本把 `attempts.csv` 和 `videos/` 复制到 `<这个文件夹>/<套件>/`（已经复制过、大小没变的视频跳过），同步客户端自动上传。
3. 在网页上对这个文件夹点一次"共享 → 任何知道链接的人可查看"，复制链接。之后把链接用 `--share-url` 或环境变量 `LIBERO_SHARE_URL` 告诉脚本，采集结束时会打印出来。

以后每次采集结束，新视频和更新后的 CSV 会出现在同一个链接下，Slack 里只需要发一次链接。脚本自己不发 Slack 消息。分享文件夹和链接可以用一次 `setx` 永久设置（见 RUN_COMMANDS.md）。

## 7. 测试

`tests/test_libero_collection.py` 共 25 项，2026-11-09 改成"CSV + 视频"之后全部通过（82 s）；11-10 加了 1 项（录制中的文件不在 `videos/` 里），22 项全部通过（94 s）；11-10 晚加了 `S` 键，又加 1 项（取消的录制被删掉），改成按 `S` 才开始录之后 23 项全部通过（98 s）；11-10 晚修了成功后就退出的问题（见 5.2），又加 2 项，25 项全部通过（101 s）：

| 测试 | 内容 |
| --- | --- |
| 纯逻辑（14 项） | 连续 10 步计数（中途断开要重数；开始前就成立的不算）；结果分类和成功率；`attempts.csv`：写入读出、续采的轮次和摆放编号、Excel 认得的 BOM、最后一行写坏了也能读而且下一行另起一行、文件被锁时行不丢（另存 `attempts_unsaved_*.csv`，解锁后补写）；分享只复制 CSV 和写好的视频；种子不重复；文件命名和"怎么结束的"文字；碰倒检测；面板文字（包括在录 / 没在录两种提示）；原始 demo 写入读出 |
| 视频（4 项） | 用合成画面录一轮：开始前只留最后一段、上一个摆放的旧画面不录、帧数 = 开始前 + 录到的 + 结尾停留、标题栏颜色（录制时深灰，结尾绿色 SUCCESS 和边框）、画面没有被翻转、没有残留临时文件；放弃的一轮不留文件；第一帧还没送到编码线程就结束时也能写出视频；录制中的文件在工作文件夹（采集时是运行日志文件夹）里，`videos/` 里没有，结束后才移进去；取消的录制：录到一半停掉，工作文件夹里的文件被删，`videos/` 不出现 |
| 特写（1 项） | 特写的相机就是命令 2 的窗口按两次 `V` 之后的相机（方位角 270°，其余参数不变），固定不转；画面确实左右翻转；窗口是命令 2 的窗口类（有射线和编号）；编号标签和目标面板画在主视图里 |
| 每个任务的成功规则（2 项） | libero_object 和 libero_spatial 共 20 个任务逐个加载：检查器的条件和 BDDL 一致；刚摆好时不成功；把目标物体直接放进篮子 / 盘子上，物理稳定后，官方 `_check_success()` 和逐条结果都为真，两者没有出现不一致 |
| 采集主循环（3 项） | ① 无窗口运行真实主循环（规划器用 straight 后端，不需要 OMPL 环境），用脚本模拟 SpaceMouse 和按键，直接把物体放进篮子来触发成功，加 `--save-demos`。覆盖：R 记失败；M 切换构型且摆放不变；SpaceMouse 还在推的时候按数字键，规划器当拍就被收回；SpaceMouse 不动时按 1，规划器连续带了一段，`0` 取消（或已到位）；碰倒的物体被标出，照样算成功；成功后自动换摆放；满了自动结束。`S`：第 1 轮按，第 3 轮按住（两次隔 50 ms，只算一次），第 4 轮按了又取消，第 2 轮不按。检查：采集文件夹里只有 `attempts.csv` 和 `videos/`；CSV 四行（R 失败、M 失败、成功、成功）的结果、怎么结束的、构型、摆放编号、碰倒的物体都对；只有第 1、3 轮有视频，和 CSV 的 `video` 列一一对应，第 2、4 轮 `video` 为空，运行日志里 `S` 的记录是第 1 轮第 2 步开始录、第 3 轮第 4 步开始录、第 4 轮第 2 步开始录、第 13 步取消，退出时记 1 个取消的录制，日志文件夹里没有残留视频；第 3 轮成功视频的帧数 = 从第 4 步到结束的步数 + 结尾 2 s（从按键开始，不是整轮）；视频大小是窗口加标题栏（960×540 → 960×576），结尾颜色对；原始 demo（规划器步数、`arm_owner` 等）和离线数据集（形状、图像和 robosuite 自己渲染的一致）照旧。② 默认模式（不加 `--save-demos`），没有 demo 文件夹：第一次运行不按 `S`，`--duration` 到时记为 quit / time limit，CSV 有这一行但 `video` 为空，采集文件夹里只有 `attempts.csv`，什么都没录；再运行一次、1.5 s 时按 `S`，轮次和摆放编号接着往下数，视频结尾灰色 QUIT，帧数 = 按键之后的步数 + 结尾 2 s；手臂一直没动的运行不留行也不留视频；运行日志文件夹里没有残留的视频文件。③ `attempts.csv` 被锁着时成功（像 WPS 那样只读打开、别的程序只能读）：第 1 轮的成功行留在内存里，提示只打一次，接着换摆放；第 2 轮中途解锁，成功后两行一起写进表；采满 2 次正常结束，没有 `attempts_unsaved_*.csv`，运行日志里没有异常 |
| 控制台 stderr（1 项） | 在隐藏的新控制台窗口里起子进程（stderr 是真正的控制台）：只导入 mediapipe，之后 `print(..., file=sys.stderr)` 报 `WinError 6`、退出码 120，就是 11-10 的问题本身；先导入采集入口脚本再导入 mediapipe，打印正常、退出码 0 |

真实窗口大小下的实测（不在测试文件里）：2544×1334（就是你实际用的窗口大小），无窗口模式，摄像头小窗用合成画面代替（没有打开摄像头），straight 规划器，脚本输入（按 1、0、R，再按 1），各 20 s：

| | 控制循环中位数 | 最慢的 5% | 每拍耗时中位数 / p95 | 视频丢帧 |
| --- | --- | --- | --- | --- |
| 录视频 | 20.00 Hz | 19.98 Hz | 20.6 / 23.2 ms | 0 |
| 不录（对照） | 20.00 Hz | 19.99 Hz | 20.0 / 21.8 ms | — |

两次都只有 2 拍超时（启动和换摆放时，和录不录无关）。抽帧看过：三个视图、摄像头小窗、标题栏计时、结尾的 FAILED / QUIT / SUCCESS 和边框都对。

回归（11-10 改完重跑）：LIBERO 环境 75 个全部通过（视图 8、复位提示 12、主命令 1 的 5、规划器 26、纯摄像头 5、摄像头控臂 19；158 s）。这次把 `test_camera_arm_control.py` 也一起跑了，以前的 54 个不含它。11-10 晚加 `S` 键之后重跑：75 个全部通过（152 s）；改成按 `S` 才开始录之后再跑：75 个全部通过（166 s）；11-10 晚修 stderr 之后再跑：75 个全部通过（150 s）。冻结基线哈希：172 个文件不变、1 个有变化，和之前一样。

11-11 修换摆放后 SpaceMouse 接管慢：新加 `tests/test_libero_collect_device.py` 9 项，全部通过（29 s）。假 HID 设备照这台 SpaceMouse Wireless 的样子：不碰时不发报告，推的时候每 10 ms 一对报告（平移、旋转分开发），松开时最后发一个零。

| 测试 | 内容 |
| --- | --- |
| 复现（1 项） | 原来的 `SpaceMouseHid`：启动时学好零点，换摆放，静置 2 s（规划器在走），然后连推 4 下（每下 0.6 s）：手臂一下都没动，零点一直没重新学好，就是用户遇到的情况 |
| 新的读取（6 项） | 换摆放后零点不变；静置 2 s 后一推，下一拍就有动作（不超过 2 拍）；复位后 0.3 s 内推不算；复位时一直按着的，新场景里不动，松开后再推才动；复位期间（没在读）推过又松开的报告不会在新场景里重放；启动时还没学到零点的，照旧要先学（推不动，手轻放一下学好后能推）；关掉再打开的设备重新学零点 |
| 采集主循环（2 项） | 真实主循环（无窗口，straight 规划器），假 HID 接在新 / 旧两种读取上：第 1 个摆放手轻放、推一下、按 R；第 2 个摆放静置 1.5 s 后推 1 s。新的：第 2 个摆放推的下一拍就开始计时（第 32 步以内），退出时日志记录保留零点的次数；旧的：同样推了 1 s，这一轮一直没开始（not_started） |

改完重跑：采集 25 项、新的 9 项、SpaceMouse 读取和映射、复位提示、视图、主命令 1、规划器、纯摄像头，共 126 个全部通过（290 s）；冻结基线 172 个不变、1 个有变化。实机还没试。

实机：2026-11-10 用户用 11-09 版实采了 libero_object 任务 0：第一次运行 8 轮、3 次成功；第二次运行第 9 轮挂了 17 分钟后按 Q，记为 quit。CSV 9 行、9 个视频都正常生成（读过长度和帧数，没看画面：1920×1088、20 fps，长度 = 用时 + 开始前 1 s + 结尾 2 s，长的几段短约 1%，见第 8 节）。11-10 的三处改动（录制中的文件位置、规划器的指根对准、`S` 键存视频）还没在实机上跑过。端到端冒烟（无窗口，合成输入）：OMPL、forward 构型，1 s 时按 1 让规划器带手臂走，每次运行 10 s。第一次运行 4 s 时按 `S`：第 74 步开始录，录制中的文件只在运行日志文件夹里；结束后视频移进 `videos/`，159 帧 = 193 步 - 74 + 结尾 40 帧，正好是从按键开始（整轮应有 232 帧）。第二次运行不按 `S`：CSV 有这一行、`video` 为空，运行日志文件夹里从头到尾没出现过视频文件。第三次运行 3 s 按 `S`、5 s 再按：第 53 步开始录、第 93 步取消，录制文件出现过、结束时已删掉，`videos/` 没有新文件，退出时记 1 个取消的录制。三次视频丢帧都是 0，平均 20 Hz（第一次 8.6 s 走了 172 步）。

11-10 晚用户又实采了两次（17:40 和 18:06，forward 构型，OMPL）：两次都是第一轮就成功了（规划器指根对准到位误差 2.4 / 0.8 mm，拿着汤罐到篮子上方 3.5 / 7.4 mm），然后程序没有任何输出就退出了，原因见 5.2。修之前在真实控制台里（新开的控制台窗口，先导入 mediapipe，`attempts.csv` 被锁着，OMPL，无窗口，脚本输入，脚本把汤罐放进篮子）复现出来：第 1 轮成功后退出，日志和实采的一样（`attempt_started` 之后只有 `exception`，没有 `attempt_end` 和 `exit`，留下 `attempts_unsaved_*.csv`）。修之后同样的步骤：第 1 轮成功后接着换摆放；第 2 轮解锁，按 1 让 OMPL 规划，成功时取消还在走的规划，两行一起写进表；第 3 轮成功后 3/3 正常结束，退出码 0。修好后还没在实机上重跑。

## 8. 已知限制

- 视频是窗口缩小到 1920 宽的画面，窗口里的小字会小一些。
- 换摆放 / 切换构型后新场景出来前的约 0.5 s 不录（窗口那时还停在旧画面上）。
- 控制循环超时的那几拍没有画面（11-10 实采那次平均 19.74 Hz，6.5% 的拍超过 50 ms），所以长视频比真实时间短约 1%。每一帧标题栏上的计时和 CSV 的用时都是真实时间，不受影响。
- 成功只按 BDDL 判断；碰倒其他物体只做标记，不判失败。
- 视频从按 `S` 那一刻开始录，按之前的部分没有；想录整轮要在手臂第一次动之前按。一轮结束后不能补录。不录的轮次不开 ffmpeg，省下编码占的 CPU。
- 采集时 Excel / WPS 开着 `attempts.csv` 的话，行会先存在内存里，关掉后补写；退出时还开着就另存成 `attempts_unsaved_*.csv`，下次运行接不上，见 5.2。
- stderr 的修复只加在采集入口（命令 3）。命令 1、2 等入口开了摄像头之后，在真实控制台里往 stderr 打印同样会报错；它们只在出错退出时往 stderr 打一句原因，那句会看不到。
- 换摆放后保留 SpaceMouse 零点只改在采集入口（命令 3）。命令 1、2 换摆放后还是要先轻碰几下 SpaceMouse 才能接管。
- 分享链接要手动在网页上创建一次，脚本不调用云盘或 Slack 的接口。
- 规划器默认用 OMPL 后端，需要 `ompl_planner` Conda 环境（找不到时见 RUN_COMMANDS.md 最后一节，或者用 `--planner straight`）。
- libero_object 里有几个物体随机摆放时位置是固定的（例如任务 0 的汤罐、沙拉酱和番茄沙司），篮子变化不到 3 cm，其他物体不到 1 cm。这是 LIBERO 自己的设计：BDDL 摆放区域减去物体半径后几乎没有余量。官方 50 个初始状态也一样（11-09 对比过），所以采集的分布和官方一致。libero_spatial 每个物体都有约 3 cm 的变化。
- `--save-demos` 的数据：26 维动作和官方 LIBERO 的 7 维不兼容（见 5.4）；模型 XML 里引用的是这台电脑上网格文件的绝对路径，在别的电脑上重新渲染需要同样的项目目录和 LIBERO。
