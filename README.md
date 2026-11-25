# Triton Zeus 编程指南

面向 Zeus V3 的中文编程指南，介绍 LDG 3D DRAM 近存架构、数据布局、Zeus 编程接口、算子开发与性能优化。当前版本 **v0.0.6**，共 182 页，包括 14 章、附录 A–D、33 张正文矢量图和 8 个完整 kernel 示例。第四章先给出布局总览，再分别介绍 Weight 与分页 KV Cache。

**[在线阅读 / 下载 PDF](https://xuanwen-wang.github.io/triton-zeus-programming-guide/Triton-Zeus-Programming-Guide.pdf)** · [章节索引](OUTLINE.md) · [版本记录](CHANGELOG.md)

指南统一以 PDF 发布，可在浏览器或 PDF 阅读器中阅读，使用书签和文内链接导航。网站首页直接打开 PDF，原 HTML 章节页面已下线。

正文位于 `chapters/`，`book.yaml` 管理版本、日期和章节顺序。示例程序位于 `examples/`，正文示意图由 `scripts/draw_*.py` 生成，可继续编辑。封面采用 3D DRAM 封装的铜版线刻插画，标题、公司名、版本与日期均由 Typst 排为矢量文字。[插画生成说明](assets/cover-art-v005.prompt.txt)

## 阅读路线

| 开发任务 | 建议入口 |
|---|---|
| 第一次编写算子 | 第 2 章：环境检查、向量加法与运行 |
| 理解 Zeus V3 | 第 3 章：执行单元、容量、算力与带宽 |
| 使用权重和 KV Cache | 第 4 章：LDG 数据布局与访问 |
| 查询语言规则与平台专用接口 | 第 6 章语言规则、第 7 章 Zeus 编程接口及附录 A |
| 接入框架、调试与优化 | 第 8–10 章 |
| 开发 Matmul、归一化、Attention | 第 11–13 章 |
| 迁移已有 GPU Triton 算子 | 第 14 章 |

## 运行示例

PDF 内嵌同版 `Triton-Zeus-Examples-v0.0.6.zip`。用支持 PDF 附件的阅读器保存并解压，或使用仓库的 [`examples/`](examples/) 目录。设备执行所需的 Zeus SDK、PyTorch、Triton 和配套组件版本以产品交付清单为准。安装并完成后端配置后，从随书目录运行环境检查和向量加法：

```bash
python examples/check_environment.py --require-jit
ZECC_KERNEL_MODE=jit python examples/vector_add.py --numel 2500 --run-device
```

`--run-device` 调用已安装的 Zeus 运行时并与 PyTorch 参考比较，实际后端由 SDK 配置决定。默认命令只运行 CPU 数学参考。环境配置和常见错误见第 2、8 章，各示例的输入与执行范围见[随书示例说明](examples/README.md)。

LDG 矩阵乘和 KV Cache 示例还需要配套的资源分配与 pack 接口，不能用普通 Tensor 地址替代 LDG 地址表。第 4、11–13 章说明相关宿主代码与资源生命周期。

## 构建 PDF 与发布目录

PDF 使用 Pandoc **3.9.0.2**、Typst **0.14.2** 与 Python。系统字体需要 Noto Sans CJK SC、Noto Sans Mono CJK SC、Noto Serif CJK SC、Liberation Mono 和 FreeSans。封面使用的 Cormorant Garamond 随仓库提供，数学字体由 Typst 提供。Python 渲染与 PDF 依赖见 `requirements-review.txt`。

```bash
python3 -m pip install -r requirements-review.txt
python3 scripts/build_guide.py
python3 scripts/review_pdf.py
python3 scripts/build_site.py
python3 scripts/check_site.py
python3 scripts/check_public_artifact.py
```

如 Typst 不在 `PATH`，设置 `TYPST` 或使用 `--typst` 指定可执行文件。PDF 默认生成在工程根目录；`_site/` 只包含 PDF、版本信息和首页入口，可在本地预览：

```bash
python3 -m http.server 8000 --directory _site
```

`pdf-build.json` 记录 PDF 与全部输入的 SHA-256。推送 `main` 后，GitHub Actions 核对 PDF 与正文身份，并部署 PDF 发布目录。如果 PDF 与正文不匹配，检查会阻止部署。更新正文时应先重建 PDF，将正文、PDF 和 `pdf-build.json` 一同提交。

`review_pdf.py` 将全部页面渲染到 `review/render-<PDF哈希前缀>/`，并记录书签、链接和页面边界检查。发布前还需实际查看页面；结构检查通过不等于版式审阅完成。只修改不参与 PDF 构建的维护文档时，无需重建 PDF。

## 维护入口

开始修改前阅读 [AGENTS.md](AGENTS.md)，其中约定技术取证、图表与提示框、版本同步、逐页审阅和发布验证。贡献者入口见 [CONTRIBUTING.md](CONTRIBUTING.md)。

| 文件或目录 | 内容 |
|---|---|
| `book.yaml`、`chapters/` | 版本与章节顺序、客户正文 |
| `layout/` | Typst 模板与 Pandoc 过滤器 |
| `scripts/draw_*.py`、`assets/` | 图表生成器及资产 |
| `examples/` | 示例程序和宿主布局检查 |
| `OUTLINE.md`、`CHANGELOG.md` | 章节索引和逐版变更 |
| `RELEASE-REVIEW.md`、`pdf-build.json` | 审阅结论、PDF 与输入身份 |
| `reference/`、`review/` | 本地参考和审阅材料，不进入公开仓库 |

布局检查可独立在 Host 上运行：

```bash
python3 examples/ldg_layout.py
```

该脚本检查逻辑坐标、物理布局、索引和追加写入的地址关系，不调用设备 kernel。

## 验证范围

本轮修订通过 8 项离线编译、15 项 CPU 数学参考、11 项 Host 布局参考和 16 项 Python 功能仿真。验证结果及输入身份见[验证摘要](validation-summary.json)的 `customer_review_update`，测试条件见[本版审阅记录](RELEASE-REVIEW.md)。客户 SDK 的 JIT、AOT、Graph 与物理设备执行待在配套环境验证。理论带宽和算力的适用条件在正文中单独说明。

## 反馈与权利声明

欢迎提交 [issue](https://github.com/Xuanwen-Wang/triton-zeus-programming-guide/issues) 或通过 fork 发起 Pull Request。主分支由仓库所有者审核和合并，协作方法见 [CONTRIBUTING.md](CONTRIBUTING.md)。第三方参考手册、内部取证资料和工具链二进制不包含在公开仓库中。

Copyright © 2026 Polar Bear Tech (Xi'an) Co., Ltd. All rights reserved.

© 2026 北极雄芯信息科技（西安）有限公司。详见[版权声明](chapters/00-copyright.md)。
