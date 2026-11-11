# LIBERO 套件、任务和物体质量

从 `.venv-libero` 里逐个加载官方任务（LIBERO 自带的 robosuite 1.4，不开渲染），读编译后 MuJoCo 模型里每个可移动物体的质量（`model.body_mass`，物体所有刚体相加）。只读，没有改任何东西。2026-11-05 生成。

- 一共 130 个任务，全部是控制频率 20 Hz、物理步长 2 ms（每个控制步算 25 次物理）。
- 同一种物体在所有任务里质量相同，所以质量只跟物体种类有关。
- 只统计任务里能拿起来的物体。柜子、炉子、微波炉、酒架、书架、笔筒这类固定家具不算。
- 项目入口的 `--object-mass-scale k` 会把任务里所有可移动物体的质量和转动惯量一起乘以 k（范围 0.1–100），家具、机器人和手不变。

## 1. 五个套件

| 套件 | 任务数 | 场景 | 考什么 |
| --- | --- | --- | --- |
| `libero_spatial` | 10 | 桌面 | 同一组物体，只换黑碗的起始位置：考空间关系 |
| `libero_object` | 10 | 地面 + 篮子 | 每个任务抓一种不同的日用品放进篮子：考物体种类 |
| `libero_goal` | 10 | 桌面 | 同一个场景、同一组物体，目标不同（开抽屉、开炉子、放瓶子等）：考任务目标 |
| `libero_10` | 10 | 厨房 / 客厅 / 书房 | 长任务，一个任务里有两步：考长序列 |
| `libero_90` | 90 | 厨房 / 客厅 / 书房 | 90 个短任务，场景和物体最多 |

## 2. 物体质量（26 种，从轻到重）

右边一列是同类实物的大概质量，只是常识粗估，还没有按仿真里物体的尺寸核对。

| 物体（类名） | LIBERO 质量 | 出现在 | 同类实物粗估 |
| --- | --- | --- | --- |
| 白碗（`WhiteBowl`） | 2.1 g | libero_90 | 250–400 g |
| 黄油（`Butter`） | 5.2 g | libero_object、libero_10、libero_90 | 110–450 g |
| 黑碗（`AkitaBlackBowl`） | 5.6 g | libero_spatial、libero_goal、libero_10、libero_90 | 250–400 g |
| 奶油奶酪盒（`CreamCheese`） | 6.2 g | libero_object、libero_goal、libero_10、libero_90 | 约 230 g |
| 饼干盒（`Cookies`） | 9.6 g | libero_spatial | 300–450 g |
| 小烤碗（`GlazedRimPorcelainRamekin`） | 9.7 g | libero_spatial | 150–250 g |
| 巧克力布丁（`ChocolatePudding`） | 10.2 g | libero_object、libero_10、libero_90 | 100–450 g |
| 盘子（`Plate`） | 11.5 g | libero_spatial、libero_goal、libero_10、libero_90 | 350–600 g |
| 烧烤酱瓶（`BbqSauce`） | 12.2 g | libero_object | 500–650 g |
| 葡萄酒瓶（`WineBottle`） | 15.4 g | libero_goal、libero_10、libero_90 | 1200–1500 g（满瓶） |
| 白瓷杯（`PorcelainMug`） | 16.5 g | libero_10、libero_90 | 300–400 g |
| 沙拉酱瓶（`SaladDressing`） | 21.8 g | libero_object | 450–550 g |
| 沙拉酱瓶（新款）（`NewSaladDressing`） | 21.8 g | libero_90 | 450–550 g |
| 番茄酱瓶（`Ketchup`） | 23.2 g | libero_object、libero_10、libero_90 | 450–650 g |
| 字母汤罐头（`AlphabetSoup`） | 25.9 g | libero_object、libero_10、libero_90 | 350–450 g |
| 番茄沙司罐头（`TomatoSauce`） | 26.6 g | libero_object、libero_10、libero_90 | 250–450 g |
| 黄书（`YellowBook`） | 28.1 g | libero_90 | 300–800 g |
| 红咖啡杯（`RedCoffeeMug`） | 29.4 g | libero_10、libero_90 | 300–400 g |
| 黄白杯（`WhiteYellowMug`） | 30.9 g | libero_10、libero_90 | 300–400 g |
| 牛奶盒（`Milk`） | 31.7 g | libero_object、libero_10、libero_90 | 250–1100 g（看盒子大小） |
| 橙汁盒（`OrangeJuice`） | 31.7 g | libero_object、libero_10、libero_90 | 250–1100 g（看盒子大小） |
| 黑书（`BlackBook`） | 42.5 g | libero_10、libero_90 | 300–800 g |
| 8 寸平底锅（`Chefmate8Frypan`） | 50.2 g | libero_10、libero_90 | 700–1100 g |
| 木托盘（`WoodenTray`） | 57.2 g | libero_90 | 300–800 g |
| 摩卡壶（`MokaPot`） | 68.2 g | libero_10、libero_90 | 400–700 g（空壶） |
| 篮子（`Basket`） | 136.1 g | libero_object、libero_10、libero_90 | 300–800 g |

## 3. 每个任务

任务号就是入口的 `--suite <套件> --task-id <号>`。任务说明是官方英文原文。

### `libero_spatial`（10 个）

| task-id | 任务 | 场景里的可移动物体和质量 |
| --- | --- | --- |
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 黑碗 ×2 5.6 g、饼干盒 9.6 g、小烤碗 9.7 g、盘子 11.5 g |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 黑碗 ×2 5.6 g、饼干盒 9.6 g、小烤碗 9.7 g、盘子 11.5 g |
| 2 | pick up the black bowl from table center and place it on the plate | 黑碗 ×2 5.6 g、饼干盒 9.6 g、小烤碗 9.7 g、盘子 11.5 g |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 黑碗 ×2 5.6 g、饼干盒 9.6 g、小烤碗 9.7 g、盘子 11.5 g |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 黑碗 ×2 5.6 g、饼干盒 9.6 g、小烤碗 9.7 g、盘子 11.5 g |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 黑碗 ×2 5.6 g、饼干盒 9.6 g、小烤碗 9.7 g、盘子 11.5 g |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 黑碗 ×2 5.6 g、饼干盒 9.6 g、小烤碗 9.7 g、盘子 11.5 g |
| 7 | pick up the black bowl on the stove and place it on the plate | 黑碗 ×2 5.6 g、饼干盒 9.6 g、小烤碗 9.7 g、盘子 11.5 g |
| 8 | pick up the black bowl next to the plate and place it on the plate | 黑碗 ×2 5.6 g、饼干盒 9.6 g、小烤碗 9.7 g、盘子 11.5 g |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 黑碗 ×2 5.6 g、饼干盒 9.6 g、小烤碗 9.7 g、盘子 11.5 g |

### `libero_object`（10 个）

| task-id | 任务 | 场景里的可移动物体和质量 |
| --- | --- | --- |
| 0 | pick up the alphabet soup and place it in the basket | 字母汤罐头 25.9 g、篮子 136.1 g、沙拉酱瓶 21.8 g、奶油奶酪盒 6.2 g、牛奶盒 31.7 g、番茄沙司罐头 26.6 g、黄油 5.2 g |
| 1 | pick up the cream cheese and place it in the basket | 奶油奶酪盒 6.2 g、篮子 136.1 g、字母汤罐头 25.9 g、牛奶盒 31.7 g、番茄沙司罐头 26.6 g、黄油 5.2 g、橙汁盒 31.7 g |
| 2 | pick up the salad dressing and place it in the basket | 沙拉酱瓶 21.8 g、篮子 136.1 g、番茄酱瓶 23.2 g、字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、牛奶盒 31.7 g、番茄沙司罐头 26.6 g |
| 3 | pick up the bbq sauce and place it in the basket | 烧烤酱瓶 12.2 g、篮子 136.1 g、巧克力布丁 10.2 g、番茄酱瓶 23.2 g、沙拉酱瓶 21.8 g、字母汤罐头 25.9 g、奶油奶酪盒 6.2 g |
| 4 | pick up the ketchup and place it in the basket | 番茄酱瓶 23.2 g、篮子 136.1 g、烧烤酱瓶 12.2 g、沙拉酱瓶 21.8 g、字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、牛奶盒 31.7 g |
| 5 | pick up the tomato sauce and place it in the basket | 番茄沙司罐头 26.6 g、篮子 136.1 g、牛奶盒 31.7 g、黄油 5.2 g、橙汁盒 31.7 g、巧克力布丁 10.2 g、烧烤酱瓶 12.2 g |
| 6 | pick up the butter and place it in the basket | 黄油 5.2 g、篮子 136.1 g、番茄沙司罐头 26.6 g、橙汁盒 31.7 g、巧克力布丁 10.2 g、烧烤酱瓶 12.2 g、番茄酱瓶 23.2 g |
| 7 | pick up the milk and place it in the basket | 牛奶盒 31.7 g、篮子 136.1 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、黄油 5.2 g、橙汁盒 31.7 g、巧克力布丁 10.2 g |
| 8 | pick up the chocolate pudding and place it in the basket | 巧克力布丁 10.2 g、篮子 136.1 g、橙汁盒 31.7 g、烧烤酱瓶 12.2 g、番茄酱瓶 23.2 g、沙拉酱瓶 21.8 g、字母汤罐头 25.9 g |
| 9 | pick up the orange juice and place it in the basket | 橙汁盒 31.7 g、篮子 136.1 g、黄油 5.2 g、巧克力布丁 10.2 g、烧烤酱瓶 12.2 g、番茄酱瓶 23.2 g、沙拉酱瓶 21.8 g |

### `libero_goal`（10 个）

| task-id | 任务 | 场景里的可移动物体和质量 |
| --- | --- | --- |
| 0 | open the middle drawer of the cabinet | 黑碗 5.6 g、奶油奶酪盒 6.2 g、葡萄酒瓶 15.4 g、盘子 11.5 g |
| 1 | put the bowl on the stove | 黑碗 5.6 g、奶油奶酪盒 6.2 g、葡萄酒瓶 15.4 g、盘子 11.5 g |
| 2 | put the wine bottle on top of the cabinet | 黑碗 5.6 g、奶油奶酪盒 6.2 g、葡萄酒瓶 15.4 g、盘子 11.5 g |
| 3 | open the top drawer and put the bowl inside | 黑碗 5.6 g、奶油奶酪盒 6.2 g、葡萄酒瓶 15.4 g、盘子 11.5 g |
| 4 | put the bowl on top of the cabinet | 黑碗 5.6 g、奶油奶酪盒 6.2 g、葡萄酒瓶 15.4 g、盘子 11.5 g |
| 5 | push the plate to the front of the stove | 黑碗 5.6 g、奶油奶酪盒 6.2 g、葡萄酒瓶 15.4 g、盘子 11.5 g |
| 6 | put the cream cheese in the bowl | 黑碗 5.6 g、奶油奶酪盒 6.2 g、葡萄酒瓶 15.4 g、盘子 11.5 g |
| 7 | turn on the stove | 黑碗 5.6 g、奶油奶酪盒 6.2 g、葡萄酒瓶 15.4 g、盘子 11.5 g |
| 8 | put the bowl on the plate | 黑碗 5.6 g、奶油奶酪盒 6.2 g、葡萄酒瓶 15.4 g、盘子 11.5 g |
| 9 | put the wine bottle on the rack | 黑碗 5.6 g、奶油奶酪盒 6.2 g、葡萄酒瓶 15.4 g、盘子 11.5 g |

### `libero_10`（10 个）

| task-id | 任务 | 场景里的可移动物体和质量 |
| --- | --- | --- |
| 0 | put both the alphabet soup and the tomato sauce in the basket | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、橙汁盒 31.7 g、牛奶盒 31.7 g、黄油 5.2 g、篮子 136.1 g |
| 1 | put both the cream cheese box and the butter in the basket | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、橙汁盒 31.7 g、牛奶盒 31.7 g、黄油 5.2 g、篮子 136.1 g |
| 2 | turn on the stove and put the moka pot on it | 8 寸平底锅 50.2 g、摩卡壶 68.2 g |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 黑碗 5.6 g、葡萄酒瓶 15.4 g |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 白瓷杯 16.5 g、红咖啡杯 29.4 g、黄白杯 30.9 g、盘子 ×2 11.5 g |
| 5 | pick up the book and place it in the back compartment of the caddy | 黑书 42.5 g、黄白杯 30.9 g |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 白瓷杯 16.5 g、红咖啡杯 29.4 g、盘子 11.5 g、巧克力布丁 10.2 g |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、篮子 136.1 g |
| 8 | put both moka pots on the stove | 摩卡壶 ×2 68.2 g |
| 9 | put the yellow and white mug in the microwave and close it | 白瓷杯 16.5 g、黄白杯 30.9 g |

### `libero_90`（90 个）

| task-id | 任务 | 场景里的可移动物体和质量 |
| --- | --- | --- |
| 0 | close the top drawer of the cabinet | 黑碗 5.6 g、黄油 ×2 5.2 g、巧克力布丁 10.2 g |
| 1 | close the top drawer of the cabinet and put the black bowl on top of it | 黑碗 5.6 g、黄油 ×2 5.2 g、巧克力布丁 10.2 g |
| 2 | put the black bowl in the top drawer of the cabinet | 黑碗 5.6 g、黄油 ×2 5.2 g、巧克力布丁 10.2 g |
| 3 | put the butter at the back in the top drawer of the cabinet and close it | 黑碗 5.6 g、黄油 ×2 5.2 g、巧克力布丁 10.2 g |
| 4 | put the butter at the front in the top drawer of the cabinet and close it | 黑碗 5.6 g、黄油 ×2 5.2 g、巧克力布丁 10.2 g |
| 5 | put the chocolate pudding in the top drawer of the cabinet and close it | 黑碗 5.6 g、黄油 ×2 5.2 g、巧克力布丁 10.2 g |
| 6 | open the bottom drawer of the cabinet | 黑碗 5.6 g、盘子 11.5 g |
| 7 | open the top drawer of the cabinet | 黑碗 5.6 g、盘子 11.5 g |
| 8 | open the top drawer of the cabinet and put the bowl in it | 黑碗 5.6 g、盘子 11.5 g |
| 9 | put the black bowl on the plate | 黑碗 5.6 g、盘子 11.5 g |
| 10 | put the black bowl on top of the cabinet | 黑碗 5.6 g、盘子 11.5 g |
| 11 | open the top drawer of the cabinet | 黑碗 ×3 5.6 g、盘子 11.5 g |
| 12 | put the black bowl at the back on the plate | 黑碗 ×3 5.6 g、盘子 11.5 g |
| 13 | put the black bowl at the front on the plate | 黑碗 ×3 5.6 g、盘子 11.5 g |
| 14 | put the middle black bowl on the plate | 黑碗 ×3 5.6 g、盘子 11.5 g |
| 15 | put the middle black bowl on top of the cabinet | 黑碗 ×3 5.6 g、盘子 11.5 g |
| 16 | stack the black bowl at the front on the black bowl in the middle | 黑碗 ×3 5.6 g、盘子 11.5 g |
| 17 | stack the middle black bowl on the back black bowl | 黑碗 ×3 5.6 g、盘子 11.5 g |
| 18 | put the frying pan on the stove | 8 寸平底锅 50.2 g、摩卡壶 68.2 g |
| 19 | put the moka pot on the stove | 8 寸平底锅 50.2 g、摩卡壶 68.2 g |
| 20 | turn on the stove | 8 寸平底锅 50.2 g、摩卡壶 68.2 g |
| 21 | turn on the stove and put the frying pan on it | 8 寸平底锅 50.2 g、摩卡壶 68.2 g |
| 22 | close the bottom drawer of the cabinet | 黑碗 5.6 g、葡萄酒瓶 15.4 g |
| 23 | close the bottom drawer of the cabinet and open the top drawer | 黑碗 5.6 g、葡萄酒瓶 15.4 g |
| 24 | put the black bowl in the bottom drawer of the cabinet | 黑碗 5.6 g、葡萄酒瓶 15.4 g |
| 25 | put the black bowl on top of the cabinet | 黑碗 5.6 g、葡萄酒瓶 15.4 g |
| 26 | put the wine bottle in the bottom drawer of the cabinet | 黑碗 5.6 g、葡萄酒瓶 15.4 g |
| 27 | put the wine bottle on the wine rack | 黑碗 5.6 g、葡萄酒瓶 15.4 g |
| 28 | close the top drawer of the cabinet | 黑碗 5.6 g、盘子 11.5 g、番茄酱瓶 23.2 g |
| 29 | put the black bowl in the top drawer of the cabinet | 黑碗 5.6 g、盘子 11.5 g、番茄酱瓶 23.2 g |
| 30 | put the black bowl on the plate | 黑碗 5.6 g、盘子 11.5 g、番茄酱瓶 23.2 g |
| 31 | put the black bowl on top of the cabinet | 黑碗 5.6 g、盘子 11.5 g、番茄酱瓶 23.2 g |
| 32 | put the ketchup in the top drawer of the cabinet | 黑碗 5.6 g、盘子 11.5 g、番茄酱瓶 23.2 g |
| 33 | close the microwave | 白瓷杯 16.5 g、黄白杯 30.9 g |
| 34 | put the yellow and white mug to the front of the white mug | 白瓷杯 16.5 g、黄白杯 30.9 g |
| 35 | open the microwave | 白碗 2.1 g、盘子 11.5 g |
| 36 | put the white bowl on the plate | 白碗 2.1 g、盘子 11.5 g |
| 37 | put the white bowl to the right of the plate | 白碗 2.1 g、盘子 11.5 g |
| 38 | put the right moka pot on the stove | 摩卡壶 ×2 68.2 g |
| 39 | turn off the stove | 摩卡壶 ×2 68.2 g |
| 40 | put the frying pan on the cabinet shelf | 白碗 2.1 g、8 寸平底锅 50.2 g |
| 41 | put the frying pan on top of the cabinet | 白碗 2.1 g、8 寸平底锅 50.2 g |
| 42 | put the frying pan under the cabinet shelf | 白碗 2.1 g、8 寸平底锅 50.2 g |
| 43 | put the white bowl on top of the cabinet | 白碗 2.1 g、8 寸平底锅 50.2 g |
| 44 | turn on the stove | 白碗 2.1 g、8 寸平底锅 50.2 g |
| 45 | turn on the stove and put the frying pan on it | 白碗 2.1 g、8 寸平底锅 50.2 g |
| 46 | pick up the alphabet soup and put it in the basket | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、篮子 136.1 g |
| 47 | pick up the cream cheese box and put it in the basket | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、篮子 136.1 g |
| 48 | pick up the ketchup and put it in the basket | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、篮子 136.1 g |
| 49 | pick up the tomato sauce and put it in the basket | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、篮子 136.1 g |
| 50 | pick up the alphabet soup and put it in the basket | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、橙汁盒 31.7 g、牛奶盒 31.7 g、黄油 5.2 g、篮子 136.1 g |
| 51 | pick up the butter and put it in the basket | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、橙汁盒 31.7 g、牛奶盒 31.7 g、黄油 5.2 g、篮子 136.1 g |
| 52 | pick up the milk and put it in the basket | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、橙汁盒 31.7 g、牛奶盒 31.7 g、黄油 5.2 g、篮子 136.1 g |
| 53 | pick up the orange juice and put it in the basket | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、橙汁盒 31.7 g、牛奶盒 31.7 g、黄油 5.2 g、篮子 136.1 g |
| 54 | pick up the tomato sauce and put it in the basket | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、橙汁盒 31.7 g、牛奶盒 31.7 g、黄油 5.2 g、篮子 136.1 g |
| 55 | pick up the alphabet soup and put it in the tray | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、黄油 5.2 g、木托盘 57.2 g |
| 56 | pick up the butter and put it in the tray | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、黄油 5.2 g、木托盘 57.2 g |
| 57 | pick up the cream cheese and put it in the tray | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、黄油 5.2 g、木托盘 57.2 g |
| 58 | pick up the ketchup and put it in the tray | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、黄油 5.2 g、木托盘 57.2 g |
| 59 | pick up the tomato sauce and put it in the tray | 字母汤罐头 25.9 g、奶油奶酪盒 6.2 g、番茄沙司罐头 26.6 g、番茄酱瓶 23.2 g、黄油 5.2 g、木托盘 57.2 g |
| 60 | pick up the black bowl on the left and put it in the tray | 黑碗 ×2 5.6 g、沙拉酱瓶（新款） 21.8 g、巧克力布丁 10.2 g、木托盘 57.2 g |
| 61 | pick up the chocolate pudding and put it in the tray | 黑碗 ×2 5.6 g、沙拉酱瓶（新款） 21.8 g、巧克力布丁 10.2 g、木托盘 57.2 g |
| 62 | pick up the salad dressing and put it in the tray | 黑碗 ×2 5.6 g、沙拉酱瓶（新款） 21.8 g、巧克力布丁 10.2 g、木托盘 57.2 g |
| 63 | stack the left bowl on the right bowl and place them in the tray | 黑碗 ×2 5.6 g、沙拉酱瓶（新款） 21.8 g、巧克力布丁 10.2 g、木托盘 57.2 g |
| 64 | stack the right bowl on the left bowl and place them in the tray | 黑碗 ×2 5.6 g、沙拉酱瓶（新款） 21.8 g、巧克力布丁 10.2 g、木托盘 57.2 g |
| 65 | put the red mug on the left plate | 白瓷杯 16.5 g、红咖啡杯 29.4 g、黄白杯 30.9 g、盘子 ×2 11.5 g |
| 66 | put the red mug on the right plate | 白瓷杯 16.5 g、红咖啡杯 29.4 g、黄白杯 30.9 g、盘子 ×2 11.5 g |
| 67 | put the white mug on the left plate | 白瓷杯 16.5 g、红咖啡杯 29.4 g、黄白杯 30.9 g、盘子 ×2 11.5 g |
| 68 | put the yellow and white mug on the right plate | 白瓷杯 16.5 g、红咖啡杯 29.4 g、黄白杯 30.9 g、盘子 ×2 11.5 g |
| 69 | put the chocolate pudding to the left of the plate | 白瓷杯 16.5 g、红咖啡杯 29.4 g、盘子 11.5 g、巧克力布丁 10.2 g |
| 70 | put the chocolate pudding to the right of the plate | 白瓷杯 16.5 g、红咖啡杯 29.4 g、盘子 11.5 g、巧克力布丁 10.2 g |
| 71 | put the red mug on the plate | 白瓷杯 16.5 g、红咖啡杯 29.4 g、盘子 11.5 g、巧克力布丁 10.2 g |
| 72 | put the white mug on the plate | 白瓷杯 16.5 g、红咖啡杯 29.4 g、盘子 11.5 g、巧克力布丁 10.2 g |
| 73 | pick up the book and place it in the front compartment of the caddy | 黑书 42.5 g、黄白杯 30.9 g |
| 74 | pick up the book and place it in the left compartment of the caddy | 黑书 42.5 g、黄白杯 30.9 g |
| 75 | pick up the book and place it in the right compartment of the caddy | 黑书 42.5 g、黄白杯 30.9 g |
| 76 | pick up the yellow and white mug and place it to the right of the caddy | 黑书 42.5 g、黄白杯 30.9 g |
| 77 | pick up the book and place it in the back compartment of the caddy | 黑书 42.5 g、红咖啡杯 29.4 g |
| 78 | pick up the book and place it in the front compartment of the caddy | 黑书 42.5 g、红咖啡杯 29.4 g |
| 79 | pick up the book and place it in the left compartment of the caddy | 黑书 42.5 g、红咖啡杯 29.4 g |
| 80 | pick up the book and place it in the right compartment of the caddy | 黑书 42.5 g、红咖啡杯 29.4 g |
| 81 | pick up the book and place it in the front compartment of the caddy | 黑书 42.5 g、红咖啡杯 29.4 g、白瓷杯 16.5 g |
| 82 | pick up the book and place it in the left compartment of the caddy | 黑书 42.5 g、红咖啡杯 29.4 g、白瓷杯 16.5 g |
| 83 | pick up the book and place it in the right compartment of the caddy | 黑书 42.5 g、红咖啡杯 29.4 g、白瓷杯 16.5 g |
| 84 | pick up the red mug and place it to the right of the caddy | 黑书 42.5 g、红咖啡杯 29.4 g、白瓷杯 16.5 g |
| 85 | pick up the white mug and place it to the right of the caddy | 黑书 42.5 g、红咖啡杯 29.4 g、白瓷杯 16.5 g |
| 86 | pick up the book in the middle and place it on the cabinet shelf | 黑书 42.5 g、黄书 ×2 28.1 g |
| 87 | pick up the book on the left and place it on top of the shelf | 黑书 42.5 g、黄书 ×2 28.1 g |
| 88 | pick up the book on the right and place it on the cabinet shelf | 黑书 42.5 g、黄书 ×2 28.1 g |
| 89 | pick up the book on the right and place it under the cabinet shelf | 黑书 42.5 g、黄书 ×2 28.1 g |
