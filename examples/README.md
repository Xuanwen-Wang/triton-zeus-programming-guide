# Triton Zeus 随书示例

从指南的 PDF 附件提取 ZIP，或从[指南仓库](https://github.com/Xuanwen-Wang/triton-zeus-programming-guide)取得与 PDF 构建记录匹配的源码。从解包目录运行以下命令。

```bash
python examples/check_environment.py
python examples/vector_add.py --numel 2500
python examples/ldg_layout.py
```

默认命令运行 CPU 参考或宿主布局检查。运行 Zeus 代码前，按交付材料配置匹配的 SDK 和驱动或模拟器，再检查 JIT 前置条件。

```bash
python examples/check_environment.py --require-jit
ZECC_KERNEL_MODE=jit python examples/vector_add.py --numel 2500 --run-device
```

向量、RMSNorm、Softmax 和融合示例的 `--run-device` 提交到 SDK 配置的后端。Matmul、量化、Graph 与 KV Cache 的专用条件见对应章节及附录 D。保留运行后端、SDK 版本、命令和输出，以便对照同一构建。

© 2026 北极雄芯信息科技（西安）有限公司。示例代码及正文片段的使用、复制、改编、项目集成和分发，适用随附许可或客户书面授权。其他权利和责任约定见本指南版权声明。
