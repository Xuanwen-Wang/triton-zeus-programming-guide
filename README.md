# Triton Zeus 编程指南

面向 Zeus V3 的中文编程指南，介绍 LDG 3D DRAM 近存架构、数据布局、Triton 语言接口、算子开发与性能优化。当前版本 **v0.0.4**，包括 14 章、附录 A–D、31 个语言 API 条目和 8 个完整 kernel 示例。

**[在线阅读](https://xuanwen-wang.github.io/triton-zeus-programming-guide/)** · **[下载 PDF](https://xuanwen-wang.github.io/triton-zeus-programming-guide/Triton-Zeus-Programming-Guide.pdf)** · [章节索引](OUTLINE.md) · [版本记录](CHANGELOG.md)

网页版提供章节导航、全文搜索、代码复制和插图放大。网页版与 PDF 共用 `chapters/` 下的 Markdown 正文；`book.yaml` 管理版本、日期和章节顺序。示例程序位于 `examples/`，插图由 `scripts/draw_*.py` 生成，可继续编辑。

## 阅读路线

| 开发任务 | 建议入口 |
|---|---|
| 第一次编写算子 | 第 2 章：环境检查、向量加法与运行 |
| 理解 Zeus V3 | 第 3 章：执行单元、容量、算力与带宽 |
| 使用权重和 KV Cache | 第 4 章：LDG 数据布局与访问 |
| 查询语言接口 | 第 6、7 章及附录 A |
| 接入框架、调试与优化 | 第 8–10 章 |
| 开发 Matmul、归一化、Attention | 第 11–13 章 |
| 迁移已有 GPU Triton 算子 | 第 14 章 |

## 运行示例

安装配套的 Zeus SDK、PyTorch 和 Triton 后，向量加法可以从普通 Python 入口执行：

```bash
python examples/vector_add.py --numel 2500 --run-device
```

`--run-device` 调用已安装的 Zeus 运行时；实际连接功能模拟还是真实设备，取决于该环境的运行时配置。不加此参数时只运行 CPU 数学参考。用户无需自行编写内部仿真测试脚本；输出与 PyTorch 参考的比较由示例完成。环境配置和常见错误见第 2、8 章。

LDG 矩阵乘和 KV Cache 示例还需要配套的资源分配与 pack 接口，不能用普通 Tensor 地址替代 LDG 地址表。第 4、11–13 章说明相关宿主代码与资源生命周期。

## 构建 PDF 与网站

PDF 使用 Pandoc **3.9.0.2**、Typst **0.14.2** 与 Python。所需字体为 Noto Sans CJK SC、Noto Sans Mono CJK SC 和 Liberation Mono；数学字体由 Typst 提供。Python 渲染与 PDF 依赖见 `requirements-review.txt`。

```bash
python3 -m pip install -r requirements-review.txt
python3 scripts/build_guide.py
python3 scripts/review_pdf.py
python3 scripts/build_site.py
python3 scripts/check_site.py
python3 scripts/check_public_artifact.py
```

如 Typst 不在 `PATH`，设置 `TYPST` 或使用 `--typst` 指定可执行文件。PDF 默认生成在工程根目录；本地网站生成在 `_site/`：

```bash
python3 -m http.server 8000 --directory _site
```

`pdf-build.json` 记录 PDF 与全部输入的 SHA-256。网站构建检查正文、PDF、跨章节链接、锚点、插图和搜索索引的一致性。推送 `main` 后，GitHub Actions 使用固定版本的 Pandoc 构建网站并部署 GitHub Pages；如果 PDF 与正文不匹配，检查会阻止部署。更新正文时应先重建 PDF，将正文、PDF 和 `pdf-build.json` 一同提交。

## 验证范围

[验证摘要](validation-summary.json)保留示例编译、CPU 数学参考、宿主布局与 Python 功能仿真的结果及输入身份，具体范围见[本版审阅记录](RELEASE-REVIEW.md)。本轮修订未新增物理设备、CycleSim、VCS/RTL、性能或整模型执行结果。理论带宽和算力的适用条件在正文中单独说明。

## 反馈与权利声明

欢迎提交 [issue](https://github.com/Xuanwen-Wang/triton-zeus-programming-guide/issues) 或通过 fork 发起 Pull Request。主分支由仓库所有者审核和合并，协作方法见 [CONTRIBUTING.md](CONTRIBUTING.md)。第三方参考手册、内部取证资料和工具链二进制不包含在公开仓库中。

Copyright © 2026 Polar Bear Tech (Xi'an) Co., Ltd. All rights reserved.

© 2026 北极雄芯信息科技（西安）有限公司。详见[版权声明](chapters/00-copyright.md)。
