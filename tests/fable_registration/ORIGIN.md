# 配准诊断回归来源

2026-09-16，从 Fable `registration_diagnostics_20260916/delivery/` 接收纯诊断模块后，产品线独立复核并修正。原交付保持不变。

- `test_delivery_contract.py` 来自交付的 `test_registration_diagnostics.py`，原 SHA-256 为 `59c55c389c658f919ec5c39c4e266fafd43418bec05910ea2ce4c249cbfabcb9`。仅修改模块导入、CLI 源码路径和本地 example fixture 路径，断言保持原样。
- `test_independent_contract.py` 来自产品线另写的解析真值及反例，不是 Fable 原交付测试；只将动态审查目标改为产品模块。原审查测试 SHA-256 为 `e06e201f24e7fae1e6fa4728efdeb69e2d66cc22d45381ff4d0080f61dfb853c`。
- `example_input.json` 从任务包复制，SHA-256 `201ae8befff1fbd54c6c4bff478ebccf1064ff2e1e408b524498e159a5c40ec7`。

修正候选与生产模块仅版本字符串不同（候选 `1.0.1-review-candidate`，生产 `1.0.1`）；验收与逐文件哈希见 `outputs/fable-registration-acceptance-20260916/product-integration.json`。模块没有实际相机数据、星点提取、Siril 变换解析或自动剔帧能力。
