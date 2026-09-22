# dex-retargeting 配置来源

- 上游项目：dexsuite/dex-retargeting
- 固定版本：0.4.6
- 原始路径：`dex_retargeting/configs/teleop/allegro_hand_right.yml`
- 许可证：MIT，Copyright 2023 Yuzhe Qin
- 原始配置 SHA-256：`23cdb5385137dec4174f1f9211ffe4884a9222f5d3a98fcf7218c715555a5284`
- 本地文件保持上游语义不变；运行时通过 `RetargetingConfig.set_default_urdf_dir()`
  将相对 URDF 路径解析到现有冻结 Allegro 资产目录。

当前 Windows `.venv-robosuite` 缺少 dex-retargeting 所需的 Pinocchio 二进制依赖，因此
`--hand-control dex` 会明确报错，不会退回到规则式 curl。
