# 前言 {#preface .unnumbered}

Triton Zeus 是面向 Zeus V3 的张量编程环境。开发者用 Python 描述分块、索引、运算和依赖，编译器据此安排核内执行。程序设计还须明确数据位置、物理布局，以及多个执行单元使用同一份数据的顺序。

Zeus V3 采用 3D DRAM 近存架构。LDG 存放权重与 KV Cache，矩阵单元通过专用高带宽通路访问；GDG 和片上 L2 Buffer 承担激活、输出和计算工作集等职责。本书据此讲解硬件模型与算子实例。

本书面向算子开发、框架集成和性能分析人员。开始前建议熟悉 Python、PyTorch 张量及 Triton 的基本分块概念。

## 阅读方法 {.unnumbered}

| 开发任务 | 建议阅读内容 |
|:---|:---|
| 第一次编写算子 | [快速上手](#quick-start) → [编程模型](#kernel-launch) → [语言规则](#language-and-api) |
| 使用权重或 KV Cache | [硬件与存储概览](#hardware-model) → [LDG 数据布局](#ldg-memory) → [Attention 实例](#attention-cache) |
| 查询平台接口与配套工具 | [Zeus 编程接口](#api-reference)与[附录支持表](#api-support) |
| 集成、调试或迁移 | [编译运行](#compile-and-run)、[调试](#debugging)、[性能分析](#performance)、[迁移](#migration) |

## 符号与代码约定 {.unnumbered}

`kernel` 指设备侧程序；`program` 指 Triton 逻辑程序实例；`tile` 指一次处理的数据块。矩阵维度使用 `M`、`N`、`K`，分块尺寸使用 `BLOCK` 或 `BLOCK_M` 等名称。区间采用左闭右开形式，例如 `[0, 1024)` 包含 1024 个元素。容量采用二进制单位：1 KiB = 1024 B，1 MiB = 1024 KiB。

正文统一使用 **LDG** 表示近存数据存储。代码中的 `memory_type="weight"`、`"kcache"`、`"vcache"` 以及 SDK 的 `kind` 参数保留接口原名。它们描述具体访问或布局契约，不应仅凭名字判断数据能否修改。

代码采用英文标识符与注释。完整程序在 [PDF 随书示例](#example-download)中，解包后即可按正文的相对路径执行命令。标为“片段”的代码说明局部操作。调整公式或例子的分块尺寸时，重新检查索引、存储预算和接口约束。

全书使用三类提示框：橙色“注意”标出正确性或接口约束，青色“提示”给出操作建议，蓝色“注解”解释架构差异、实现范围或版本条件。框内保留文字标签，黑白打印时也可辨认其用途。

## 适用范围 {.unnumbered}

本版适用于 Zeus V3 单设备编程。硬件配置以配套产品规格为准，环境检查和 SDK 获取方式见[安装和检查环境](#setup-environment)，示例验证范围见[附录 D](#versions-and-examples)。文档问题可在[公开仓库的反馈入口](https://github.com/Xuanwen-Wang/triton-zeus-programming-guide/issues)报告，产品或客户环境问题请通过既有交付或技术支持渠道处理。
