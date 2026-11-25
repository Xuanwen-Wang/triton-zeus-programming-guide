# Triton Zeus 编程指南 v0.0.6 章节索引

PDF 使用以下正文。章节顺序由 `book.yaml` 管理。

- [版权声明](chapters/00-copyright.md)
- [前言](chapters/00-preface.md)

## 1 概述

源文件：[01-overview.md](chapters/01-overview.md)

- 适用范围与阅读路线
- 一个算子由什么组成
- 软件栈与版本配套

## 2 快速上手

源文件：[02-quick-start.md](chapters/02-quick-start.md)

- 安装和检查环境
- 第一个向量加法
- 编译、运行与检查输出

## 3 Zeus 硬件与近存架构

源文件：[03-hardware-model.md](chapters/03-hardware-model.md)

- 系统、计算核与任务
- GDG、L2 Buffer、LDG 与 WL1
- 计算、变换与搬运单元
- Zeus V3 的容量、算力、吞吐与带宽

## 4 LDG 数据布局与访问

源文件：[03a-ldg-memory.md](chapters/03a-ldg-memory.md)

- LDG 用途与布局概览
- miniblock 与对齐
- Weight 布局：Dense、Blocked 紧排、Blocked 带 padding，以及 UScale
- 分页 KV Cache 的离散布局
- 数据准备、更新与读取
- 布局选择与错误定位

## 5 Triton Zeus 编程模型

源文件：[04-programming-model.md](chapters/04-programming-model.md)

- Kernel 启动与程序实例
- Tile、形状、步长与边界
- 任务分配与累加
- 控制流、状态和数值类型
- 存算流水与执行单元映射

## 6 数据类型与语言规则

源文件：[05-language-and-api.md](chapters/05-language-and-api.md)

- 数据类型、转换与累加精度
- 张量、指针与访存
- 算术、归约与矩阵乘
- 控制流、特化与 Zeus 扩展

## 7 Zeus 编程接口

源文件：[05a-api-reference.md](chapters/05a-api-reference.md)

- 接口入口与资源契约
- LDG 特殊读取与量化、分页参数
- LW 布局写入与原样 ST
- 矩阵计算与专用数学接口
- 执行单元与 kernel 策略提示
- 提前退出与输出状态

## 8 编译、执行与框架接入

源文件：[06-compile-and-run.md](chapters/06-compile-and-run.md)

- Python 接入与运行时 JIT
- ZECC 离线编译与 AOT
- Stream 与 Graph 捕获和重放
- 查看 IR、汇编与二进制
- 算子封装与集成

## 9 调试与正确性验证

源文件：[07-debugging.md](chapters/07-debugging.md)

- 编译问题定位
- 地址、布局与写回问题
- 资源与执行依赖问题
- 精度与边界测试
- 重放失败与验证修复

## 10 性能分析与调优

源文件：[08-performance.md](chapters/08-performance.md)

- 基线、计时范围与瓶颈
- 任务切分与访存效率
- L2 Buffer 工作集与 tile 选择
- LDG 访问、布局与打包
- 存算重叠、流水与双缓冲
- 执行单元并行与算子融合

## 11 算子开发实例

源文件：[09-examples.md](chapters/09-examples.md)

- 向量加法调优
- 归约与 RMSNorm
- Softmax：归约、中间精度与有效域
- Matmul
- Fused Add RMSNorm

## 12 量化矩阵乘与 UScale

源文件：[09b-quantized-matmul.md](chapters/09b-quantized-matmul.md)

- 量化后的数学定义
- INT4 的物理字节排列
- UScale 的逻辑形状与物理页
- 相邻与独立的 scale 存储
- 完整实例的开发过程
- 数据量、LDG 带宽与性能

## 13 分页 KV Cache 与 Attention

源文件：[09a-attention-and-cache.md](chapters/09a-attention-and-cache.md)

- 先固定一次 decode 的数学定义
- 从逻辑 token 到物理页
- 普通 weight 布局如何保存 K 和 V
- Host 准备与缓存生命周期
- QK、在线 softmax 与 PV
- 分片结果如何合并
- 从缓存到输出逐步验证

## 14 从 GPU Triton 迁移

源文件：[10-migration.md](chapters/10-migration.md)

- 迁移前检查
- 最小改写与回归
- 访存表达的迁移
- 矩阵乘与缓存的迁移
- 数值与归约结构的迁移
- 宿主与框架接入
- 平台差异与排障索引

## 附录

源文件：[11-appendices.md](chapters/11-appendices.md)

- A API 与数据类型支持表
- B 编译选项与环境变量
- C 工具、术语与产物速查
- D 示例、验证与版本记录
