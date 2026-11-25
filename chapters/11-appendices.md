# 附录 {#appendices .unnumbered}

## A API 与数据类型支持表 {#api-support .unnumbered}

本附录按接口、类型和配置汇总正文的适用范围。基础工作集覆盖本书算例使用的计算方式；专用接口带有独立的输入域或资源要求；未列入工作集的接口需要另行确认用途与配套条件。通用语法见[数据类型与语言规则](#language-and-api)，平台特有签名和参数组合见[Zeus 编程接口](#api-reference)。

### A.1 基础工作集 {.unnumbered}

| 接口组 | 本书使用的接口 | 主要条件 |
|---|---|---|
| 定义与启动 | `triton.jit`、kernel launch | 配套 Triton 与运行时；核数一致 |
| 实例与常量 | `program_id`、`constexpr` | 基础路径用一维 grid 和静态 tile |
| 索引 | `arange`、`cdiv` | 区分有效长度与静态 tile shape |
| 普通访存 | `load`、`store` | 有效地址 mask；明确填充值 |
| 块访存 | `make_block_ptr`、`advance` | shape、stride、order 和边界维度正确 |
| 形状操作 | 广播、reshape、transpose | 元素关系与实际布局保持一致 |
| 逐元素计算 | 加减乘、比较、`where` | 广播相容；分支表达式本身合法 |
| 行归约 | `sum`、`max`、`min` | 正确的轴、填充值与累加类型 |
| 矩阵乘 | `dot` | 收缩维匹配；累加类型与输入布局一致 |

Table: 基础 API 工作集

表中的名称默认属于 `triton.language`，以 `tl` 为别名。语言层的数据类型和广播规则见[数据类型与语言规则](#language-and-api)。Softmax 等专用操作使用 `tl.zeus`；执行提示由 `triton.backends.triton_shared.language` 提供，正文记为 `ztl`。启动时的 `zeus_mode` 等关键字属于宿主适配器，不是设备语言函数。

| 开发问题 | 查阅位置 |
|---|---|
| 分配核与遍历逻辑 tile | [Kernel 启动与程序实例](#kernel-launch) |
| 普通索引、块指针与边界 | [张量、指针与访存](#memory-api) |
| 广播、变换、类型和通用语法 | [数据类型与语言规则](#language-and-api) |
| 装载 LDG 权重、scale 与缓存 | [Zeus load](#api-load) |
| LW 布局写入与原样 ST | [Zeus store](#api-store) |
| 矩阵计算的 Zeus 约束 | [dot](#api-dot) |
| 专用 Softmax 与数学接口 | [Zeus 计算扩展](#api-zeus-extensions) |
| 多 VP/DT 与独立操作并行 | [执行单元提示](#api-executor-hint)、[kernel 策略](#api-kernel-hint) |
| 提前退出与输出状态 | [early_exit_if](#api-early-exit) |
| 运行时参数与编译期检查 | [控制流与特化](#control-and-extensions) |

Table: 按开发问题检索接口

### A.2 类型和数值路径 {.unnumbered}

| 输入与计算路径 | 本书采用的输出 | 应记录的数值条件 |
|---|---|---|
| BF16 装载后转 FP32 | BF16 或 FP32 | 转换节点与输出舍入 |
| FP32 逐元素计算 | FP32 | 有限值域、溢出与特殊值策略 |
| BF16 平方和、RMSNorm | BF16 | FP32 统计、eps、归约顺序 |
| BF16 × BF16 `tl.dot` | BF16 | FP32 累加、K 分块、输出转换 |
| INT8 矩阵输入 | 显式选择累加类型 | 量化缩放与 FP32 / INT32 契约 |
| FP8 E4M3FN 数据 | 由具体算子确定 | 编码、缩放及对应实现范围 |
| FP8 E4M3FN × INT4 | FP32 | N256 行配对、BF16 UScale、组长 128 |

Table: 数据类型组合的核对项

低精度类型转换和近似函数应使用独立数值测试。特别是 NaN、无穷大、次正规数、带符号零以及接近饱和边界的输入，不宜只混在随机大样本中；使用定向样例更容易识别具体行为。

INT4 的完整最小实例与布局见[量化矩阵乘与 UScale](#quantized-matmul)。设备间通信未纳入本书算例。GPU 内联汇编、CUDA 专属同步与 warp 特有语义不适用于本书的 Zeus 编程模型。原子更新等未覆盖接口应查相应专用接口和发行说明，不能仅由符号存在推断所有组合均可使用。

类型组合的验证还应固定 shape、stride、输入域和转换位置。例如“BF16 输入、FP32 累加、BF16 输出”不能被简写成“BF16 算子”后丢弃累加精度；INT8 dot 的输出类型也不能仅由输入位宽推断。发布一个算子时，至少保留一组常规值、边界值和重复调用的结果，并明确其运行环境。

### A.3 Zeus 扩展 {.unnumbered}

| 扩展 | 输入条件 | 结果或约束 |
|---|---|---|
| `vp_hint`、`dt_hint` | 目标操作可对应执行单元 | 局部精确策略或自动策略 |
| `parallel_executor_hint` | kernel 顶层标记 | 开启独立操作并行分析 |
| `double_buffer_hint` | kernel 顶层标记 | 实验性、不完善，暂不推荐使用 |
| `softmax` | FP32/BF16，rank 1–4 | 最内维归约，前导维均为 1 |
| `early_exit_if` | 单元素 FP32 整数标志，推荐 0/1 | 顶层调用；后续输出保持原值 |
| `exp_with_lut` | FP32 输入与完整系数表 | 调用方负责系数精度 |
| `sigmoid_with_exp_lut` | FP32 输入与两组表参数 | 按指定 exp 表构造路径 |
| `positive_normal_reciprocal` | 正、有限、正常 FP32 | 调用方负责近似误差界 |
| `positive_normal_rsqrt` | 正、有限、正常 FP32 | 表的输入域必须匹配 |

Table: 平台扩展的契约速查

执行提示通常在编译期被消费。解释器可能把提示当作恒等操作或空操作，因此解释器跑通不能证明执行单元并行、双缓冲或提前退出已经在设备路径中实现。

### A.4 访存方式与资源的配对 {.unnumbered}

普通指针 tile 使用逐元素 mask，块指针使用整维 `boundary_check`。二者的有效区域表达不同，不在同一调用中混用。矩形块的 `padding_option` 决定越界 load 的数值；store 没有通过填充值写入越界区域的语义。

LDG 访问还需要匹配物理组织。`memory_type="weight"` 是保留的接口字符串；KCache 与 VCache 使用各自布局契约。`store_mode`、`blocksize` 等专用参数按相应接口解释，不能推断为普通线性数组的通用选项。

地址有效性与所有权是独立条件。某个地址落在分配范围内，不代表两个核可以同时改写它所在的外部字。调试输出损坏时，应同时检查元素范围、扩展访存范围、实际写入者和跨 stream 顺序，参见[参数 ABI 与资源信息](#argument-resources)。

## B 编译选项与环境变量 {#options .unnumbered}

同名设置出现在不同入口时，应先判断由谁读取。ZECC 驱动、`zeus-compiler` 和运行时适配器有各自的参数范围；单次 Python 启动参数与进程环境变量也有不同作用域。只设置某个名称相近的变量，不保证所有入口都会采用它。

### B.1 离线编译入口 {.unnumbered}

完整编译与产物示例见[ZECC 离线编译与 AOT](#zecc-and-aot)，本节用于查询选项含义。

| 入口与选项 | 作用 | 默认或注意事项 |
|---|---|---|
| `zecc -c`、`-o` | 指定输入与输出 | 两者都需要提供 |
| `zecc --kernel-name` | 选择入口函数 | 多入口模块显式选择 |
| `zecc --compile-only-zbin` | 输出独立设备二进制 | 当前要求 Triton `.py` 输入 |
| `zecc --signature` | 指定运行时参数类型 | 可重复使用，格式为名称=类型 |
| `zecc --constant` | 指定 constexpr 值 | raw zbin 路径需完整集合 |
| `zecc --gen-include-dir` | 生成启动头文件 | 用于对象文件集成 |
| `zecc --loc=source` | 保留源码位置 | 独立 `.zbin` 路径 |
| `zecc --loc=full` | 源位置及入口 TTIR | 不带值的 `--loc` 等价于 full |

Table: ZECC 常用参数

```bash
zecc --compile-only-zbin \
  -c examples/vector_add.py -o build/vector-add.zbin \
  --kernel-name add_bf16 \
  --signature 'out_ptr=*bf16' \
  --signature 'in1_ptr=*bf16' \
  --signature 'in2_ptr=*bf16' \
  --signature numel=i32 --signature alpha=fp32 \
  --constant BLOCK=1024 --constant CORE_NUM=2
```

如果使用 shell，指针类型字符串应加引号，避免 `*` 被解释为文件名通配符。命令中的参数名必须与 kernel 定义完全一致，不能只保持顺序相似而更改名称。

**输入和输出。** `-c` 与 `-o` 均必需。`--compile-only-zbin` 当前只接受 Triton `.py` 源码，输出独立设备程序；常规 `.py` 对象路径将设备程序嵌入宿主对象。`.c`、`.zelir` 使用各自入口处理，不能给它们添加 raw Triton 输出选项后假定语义相同。

**入口与签名。** `--kernel-name NAME` 选择 kernel。`--signature name=type` 可重复出现，每项描述一个非 constexpr 参数；指针类型如 `*bf16`，标量类型如 `i32`、`fp32`。名称必须对应形式参数，不能使用宿主调用中的局部变量名替代。

**常量。** `--constant name=value` 可重复出现。独立 `.zbin` 路径把显式集合视为完整 constants；对象路径允许使用模块默认值并合并显式覆盖。若 kernel 需要 `BLOCK` 与 `CORE_NUM`，raw 路径只传 `BLOCK` 会遗漏另一个必需常量。

**目标与元数据。** `--target zeus-v3` 指定本书目标，也是当前默认值。`--meta-out FILE` 保存编译元信息，`--ttir-out FILE` 保存入口 TTIR。它们帮助核对身份和前端结果，不替代数值验证。

**位置。** 推荐使用 `--loc=source` 或 `--loc=full` 的等号形式。不带值的 `--loc` 等价于 `full`；不要把 `source` 写成一个没有选项名的独立词。位置调试输出需要独立二进制文件，不能假定 `.o` 旁边也存在同样的 sidecar。

**宿主集成。** `--gen-include-dir DIR` 用于生成启动头文件。使用 `.c` 宿主封装时，`--cc` 选择 C 编译器，默认 `gcc`；`--cflags` 传递宿主编译参数，其当前空格拆分方式不等价于完整 shell 解析。含空格路径或复杂引号的参数应先在最小构建中核对，不把设备编译选项混入宿主编译参数。

### B.2 编译策略 {.unnumbered}

下表列出本书涉及的 `zeus-compiler` 策略。ZECC 与 Python JIT 可通过 `ZEUS_COMPILER_EXTRA_FLAGS` 传递有效编译选项；Python 单次启动还可使用 `zeus_compiler_extra_flags`。先确认配套编译器支持相应策略，再进行正确性与性能对照。

| 选项 | 默认行为 | 调整前提 |
|---|---|---|
| `--num-vp=1` | 单 VP 请求 | Multi-VP 需同条件 A/B |
| `--num-dt=1` | 单 DT 请求 | 验证实际拆分与搬运收益 |
| `--enable-parallel-executor` | 关闭 | 存在可证明独立的工作 |
| `--parallel-executor-width` | 0，由硬件上限约束 | 核对资源及编译器接口 |
| `--enable-double-buffer` | 关闭 | 实验性、不完善，暂不推荐使用 |
| `--enable-loop-scheduling` | 关闭 | 保持循环依赖与数学语义 |
| `--rsqrt-scsq=1` | 使用该编码的迭代配置 | 同时复核精度和性能 |

Table: 编译策略的默认行为与使用前提

需要固定基础策略时，显式传入 `--num-vp=1 --num-dt=1`。执行单元数量与 Parallel 宽度分别设置，不把增加 VP 数量当作启用独立操作并行的同义操作。配置以本次实际编译选项为准。

Parallel-Executor 与 Multi-Executor 的分工见[执行单元并行与算子融合](#performance)。只有 trace 与产物确认了实际映射，才能把某次测量归因于某项策略。

### B.3 JIT 缓存和诊断 {.unnumbered}

| 环境变量 | 含义 | 默认行为 |
|---|---|---|
| `ZECC_JIT_CACHE_DIR` | 持久缓存目录 | XDG 缓存目录下的 `zecc/jit` |
| `ZECC_JIT_CACHE_DISABLE` | 关闭持久缓存 | 默认启用持久缓存 |
| `ZECC_JIT_LOG` | 输出 JIT 日志 | 默认不启用详细日志 |
| `ZECC_JIT_DUMP_DIR` | 诊断产物目录 | 调试时显式指定 |
| `ZECC_JIT_LOC` | `source` 或 `full` | 未设置或 `off` 时关闭 |
| `ZECC_PYTHON_EXECUTABLE` | 编译前端的 Python | 由当前入口解析 |

Table: JIT 缓存与诊断配置

`ZECC_JIT_LOC` 的有效值应使用小写 `source`、`full` 或 `off`。开启后仍需配套编译器支持；不能用一个只支持普通编译的旧工具生成看似完整的调试包。

切换工具链时启动新进程。`ZECC` 和 `ZEUS_COMPILER` 可用于选择实际工具，但自定义脚本包装器不一定满足运行时的身份解析要求。诊断过程中优先记录解析后的真实二进制路径，而不是只记录 PATH 中的命令名称。

**`ZECC_JIT_CACHE_DIR`。** 指定持久缓存根目录。未设置时先使用 `$XDG_CACHE_HOME/zecc/jit`，再回退到 `$HOME/.cache/zecc/jit`。目录应允许创建文件及锁；用独立目录区分实验可避免把他人的缓存状态混入本次观测。该设置不改变已经加载的进程内 kernel。

**`ZECC_JIT_CACHE_DISABLE`。** 启用时跳过持久缓存，进程内缓存仍可命中。推荐用 `1` 开启，未设置时保持持久缓存。适合检查一次新进程的编译行为，不应用它代替正常部署的缓存目录管理。

**`ZECC_JIT_LOG`。** 启用 JIT 路径和身份相关日志；接受的真值包括 `1`、`true`、`on`、`yes`，大小写不敏感，其他值视为未启用。Python 单次启动的 `zeus_log` 或低层 `log` 可覆盖日志开关，不修改其他线程的进程环境。

**`ZECC_JIT_DUMP_DIR`。** 显式指定诊断产物目录，运行时按 key 建立子目录。它用于保留可查看的二进制和 sidecar，与可执行持久缓存是两个位置。请求源位置时同时设置它，便于将该次产物完整归档。

**`ZECC_JIT_LOC`。** `source` 保留源码映射，`full` 还保存入口 TTIR；未设置、空值、`off` 或 `0` 关闭。字符串大小写会被归一化，推荐统一使用小写。其他值直接报错，避免误以为已经开启定位。若缓存条目缺少当前要求的 sidecar，运行时会补齐所需编译产物；因此开启调试后的首次调用可能比普通缓存命中慢。

**`ZECC_PYTHON_EXECUTABLE`。** 指定编译前端使用的 Python，未指定时 Python 启动器使用当前 `sys.executable`。它应能够导入配套 Triton 和 kernel 模块的依赖。只修改 PATH 不保证长期运行的服务切换到了新的 Python 或原生扩展。

**`ZECC`、`ZEUS_COMPILER`。** 选择编译驱动和后端工具时，应指向可识别的实际工具。标准 wheel 的 console launcher 可解析到其配套二进制；任意自定义包装脚本不能提供可靠的工具身份。更新工具或原生库后重启进程，再记录本次实际解析的路径与摘要。

### B.4 部署模式与单次启动 {.unnumbered}

| 设置 | 作用 | 使用方式 |
|---|---|---|
| `ZECC_KERNEL_MODE` | 选择 jit / aot / hybrid | 默认 `jit` |
| `ZECC_AOT_RECORD_FILE` | 记录运行时特化 | JIT 代表性输入采集 |
| `ZECC_AOT_BUNDLE` | 指定待加载 bundle | 在首次启动前设置 |
| `ZEUS_COMPILER_EXTRA_FLAGS` | 向编译器传递选项 | 必须纳入复现记录 |
| `zeus_mode` | 单次启动的部署模式 | 覆盖模式选择 |
| `zeus_compiler_extra_flags` | 单特化的编译选项 | 空字符串清除环境默认值 |

Table: 模式与启动配置

环境变量与 Python 关键字属于不同作用域。比如 `zeus_compiler_extra_flags` 只替换该次编译子进程的选项，不应改写其他同时运行任务的进程环境。AOT 发布时，完整保留有效编译选项与特化身份。

**模式选择。** 优先使用单次 `zeus_mode`，否则读取 `ZECC_KERNEL_MODE`，缺省为 `jit`。值归一化为小写后必须是 `jit`、`aot`、`hybrid` 之一。`zeus_enable_jit=False` 在此基础上收紧为 AOT；设为 `True` 不能扩大显式 AOT 模式。

**特化记录。** `ZECC_AOT_RECORD_FILE` 指向 JSONL 文件，每行记录一次调用的源码身份与编译元数据。写入采用文件锁，适合多个进程采集到同一个文件。记录实参数值的完整数据集仍由测试程序负责，JSONL 不包含用于数值重放的全部输入张量。

**bundle 位置。** `ZECC_AOT_BUNDLE` 在首次解析前设置，可以指向 bundle 目录或 manifest 文件；多个位置使用平台路径分隔符连接。已按路径加载的内容不会因磁盘文件被覆盖而自动更新。部署切换应使用新进程与经过检查的新目录。

**额外编译选项。** 未提供 `zeus_compiler_extra_flags` 时继承 `ZEUS_COMPILER_EXTRA_FLAGS`；显式空字符串清除本次继承；非空字符串替换本次编译子进程的值。它不是追加操作。schema 2 的 AOT 条目保留有效选项，确保编译结果与记录身份一致。

### B.5 前端形状与诊断配置 {.unnumbered}

`TRITON_BLOCK_SHAPE_POWER_OF_TWO=1` 对前端 tile 维度启用 2 的幂约束；默认关闭这一兼容性检查。关闭只表示前端不施加该条统一限制，不保证所有非 2 的幂形状都能映射到目标布局。`arange` 的边界仍必须是编译期整数，长度为正，后端资源和算子布局条件仍适用。

下列配置通过普通 Python JIT 入口保留诊断产物，并固定基础编译策略。执行前完成[运行后端](#runtime-backends)配置，从随书目录启动新进程：

```bash
ZECC_JIT_LOG=1 ZECC_JIT_LOC=full \
ZECC_JIT_DUMP_DIR=build/diagnostic \
ZEUS_COMPILER_EXTRA_FLAGS="--num-vp=1 --num-dt=1" \
python examples/vector_add.py --numel 2500 --run-device
```

此命令会提交计算并比较输出；若只需离线编译，使用 B.1 的 ZECC 入口。诊断目录帮助定位特化和源码位置，失败输入、PyTorch 参考及运行后端仍需由调用者单独保存。它们共同组成可复现的问题报告。

## C 工具、术语与产物速查 {#glossary .unnumbered}

### C.1 工具与文件 {.unnumbered}

| 名称 | 职责 | 常见输入或输出 |
|---|---|---|
| `zecc` | 编译驱动与宿主对象封装 | `.py` / `.c` / `.zelir` → `.o` 或 `.zbin` |
| `zeus-compiler` | Zeus 编译管线 | IR → 设备程序 |
| `zecc_jit` | Python 启动适配与 bundle 加载 | kernel、实参和部署模式 |
| `libzecc_jit` | 原生特化缓存与解析 | 特化记录、编译器、`.zbin` |
| `.zbin.debug` | 源位置映射 | 与对应 `.zbin` 配套使用 |

Table: 工具和产物

`.S` 是用于阅读与诊断的汇编文本；`.zbin` 是设备二进制；`.o` 是供宿主链接的对象，内部可以嵌入 `.zbin`。三种文件不能仅靠改扩展名互换。

### C.2 编程术语 {.unnumbered}

| 术语 | 含义 |
|---|---|
| Host | 准备数据、提交任务并消费结果的宿主 |
| Device | 由运行时管理的 Zeus 执行设备 |
| Core | 设备内承担任务的计算核 |
| Grid | Kernel 启动所使用的实例维度描述 |
| Program instance | 一次逻辑 kernel 实例，拥有自己的索引与任务区间 |
| Tile | 一次处理的静态形状数据块 |
| Tail | 输入范围不能填满整个 tile 时的有效尾部 |
| Stride | 在某一维前进一步所跨过的元素数量 |
| Specialization | 按类型、常量和目标等条件生成的特化版本 |
| Accumulator | 保存跨多次运算累计结果的中间值 |

Table: 编程模型术语

### C.3 硬件和编译术语 {.unnumbered}

| 术语 | 含义 |
|---|---|
| VP | 承担向量计算的执行单元 |
| PE | 承担矩阵类计算的执行单元 |
| DT | 承担数据变换的执行单元 |
| L2 Buffer | 编译器管理的片上工作存储，保存 tile 与中间结果 |
| LDG | 承载权重和 KV cache 的 3D DRAM |
| TTIR | Triton 前端的中间表示 |
| Linalg / MemRef | 表达结构化计算与存储访问的中间表示 |
| ZeusHigh | Zeus 操作与资源组织层的中间表示 |
| ZeusLow | 设备配置、命令和同步层的中间表示 |
| ABI | 调用双方对参数布局、类型和产物的约定 |
| Stream | 运行时管理的有序操作流 |
| Sidecar | 与主产物配套保存的辅助文件 |

Table: 硬件与编译术语

DT、外部搬运与 L2 Buffer 是不同概念。L2 Buffer 不等同于 PE 内部的 AL1、OL1；物理实现中的 SRAM 也不都属于程序可用的工作存储。阅读性能图时，应使用实际 trace 中的单元与命令名称，不能把所有数据移动都笼统归为 DT。硬件资源关系见[Zeus 硬件与近存架构](#hardware-model)。

## D 版本记录与示例索引 {#versions-and-examples .unnumbered}

### D.1 文档版本 {.unnumbered}

#### v0.0.6 修订稿 {.unnumbered}

更新日期：2026-11-25。

- 硬件章节：补充 GDG、LDG 的单组、单核及整设备设计容量，并区分物理容量与运行时可分配空间；GDG 与 L2 Buffer–LDG 读写采用 51.2 GB/s 口径。
- LDG 带宽：分别说明 Zeus V3 的 15 credit / 23 拍模型和未来 Zeus V3 PRO 的两拍 miniblock 模型，移除 WL1 到 PE 端口容量叠加项。
- Zeus 编程接口：替换通用语言 API 罗列，集中说明特殊 load/store、量化与分页参数、专用计算、执行提示和提前退出约束。
- 性能与实例：将自动双缓冲标为实验性、不完善、暂不推荐；说明在 kernel 中显式组织预取、双工作区及复用依赖的方法。
- 量化与缓存：明确静态权重和 scale 在 launch 前完成布局准备、LDG 写入及同步；补充动态 KV 的 LW 布局写入与分段 ST 路径。
- 全书版式：加入“注意、提示、注解”三类彩色框，突出 Grid、LDG 参数 ABI、数据准备和迁移约束；沿用仓库最新封面。

本稿先完成第四章以外的修订，第四章正文保留上一版，待章节结构确认后统一修订和发行。

#### v0.0.5 {.unnumbered}

更新日期：2026-11-25。

- 封面与导航：重新设计近存计算主题封面，加入 Polar Bear Tech、公司全称、版本和日期；增加“目录”“插图目录”“表格目录”三个独立 PDF 书签。
- 全书版式：增加目录条目间距，减淡引导点线；分别设置列表的行距和条目间距，改善附录更新记录及参考资料的阅读体验。
- 快速上手：取消独立的边界排查小节，只保留可选的单元素尾块检查；完整排查方法统一指向调试章节。
- 硬件章节：新增单 VP 内两组 32 lane 的对照图，分别说明宽模式和级联模式，以及 BF16、FP32 的有效元素范围。
- LDG 章节：从数学矩阵、miniblock 切分和坐标出发重组布局说明，分开介绍 Dense/Blocked、UScale 和页表寻址；补充具体元素到物理地址的逐步推导。
- 量化与 KV Cache 章节：补充 UScale 的 lane、页内与跨页坐标，以及 K/V 的槽位和字节偏移；区分 USaddr 系数地址与 KV 索引表。
- 发布方式：统一提供 PDF 在线阅读与下载，网站首页直接打开 PDF；原 HTML 章节页面下线。

#### v0.0.4 {.unnumbered}

更新日期：2026-11-25。

- 硬件、编程模型与性能章节：统一程序工作存储名称为 L2 Buffer，区分其容量预算、PE 内部缓冲和 LDG 读取量；将笼统的数据描述改为具体的访问、带宽和输入就绪条件。
- 硬件章节：将带宽讨论合并到“Zeus V3 的算力、吞吐与带宽”，取消原有独立带宽及资源对照小节，集中说明数据方向、接口宽度与条件换算。
- 编程模型章节：增加 GPU 与 Zeus 的 Grid 对照，区分 GPU 的 tile 实例数、Zeus 核数与核内循环的分工。
- 快速上手与编译运行章节：补充 SDK 执行后端配置，说明 `--run-device` 经 Triton JIT 提交到所配置的后端；区分 CPU 参考、功能模拟和物理设备运行。
- 调试与算子实例章节：以公开 JIT 入口和 PyTorch 结果比较组织操作步骤；增加 Matmul 的 LocalMem 权重准备、单核启动、同步和结果比较示例。
- 量化与 Attention 章节：明确量化加载器的格式责任，分别计算 L2 Buffer 中间状态和 K/V 的 LDG 读取量，保留普通权重与专用缓存的布局边界。
- 附录：展开各版本的逐章节变更，更新工具索引、示例运行方式和发行范围；明确宿主接入片段尚未覆盖的物理设备验证。
- 全书版式：更新中英文版权声明与页脚，调整章节书签和参考资料列表间距，复核代码、公式与图表的分页。
- 交付形式：整理公开仓库内容，增加 PDF 与在线阅读双输出，保留章节导航和示例入口。

#### v0.0.3 {.unnumbered}

::: {.keep}

更新日期：2026-11-25。

- 硬件与近存架构章节：扩充双核、每核多 VP/DT、PE 数据路径和理论吞吐换算，增加硬件总图。
- LDG 近存编程章节：新增 Dense/Blocked 坐标解释、LocalMem 权重准备、KCache/VCache 追加和读取契约，区分普通页池与专用索引表。
- Triton 语言与 API 章节：补充参数、返回值、类型与布局条件、常见错误及关联操作，扩展控制流和 Zeus 专用接口。
- 编译与框架接入章节：扩充 JIT、ZECC、AOT、Stream、Graph 与资源生命周期，说明编译特化和参数 ABI。
- 调试、性能与完整实例章节：补充工作集计算、流水依赖、独立操作并行、Softmax 有效域和 BF16 融合语义。
- 量化与 Attention 章节：新增 INT4 配对、UScale 页、分页 KV 更新、QK/PV 布局和在线 Softmax 推导；配套增加布局检查及示例。

:::

#### v0.0.2 {.unnumbered}

更新日期：2026-11-24。

- 概述至迁移章节：形成第一份完整正文，覆盖编程模型、数据类型、访存、编译执行、调试与优化。
- 算子实例章节：加入 BF16 向量加法、RMSNorm、Matmul 和 Fused Add RMSNorm，说明 FP32 累加与 BF16 输出转换。
- 全书版式：加入矢量示意图、代码块、表格和交叉引用，完成初稿逐页审阅。

#### v0.0.1 {.unnumbered}

::: {.keep}

更新日期：2026-11-24。

- 全书结构：确定概述、上手、硬件、编程模型、语言、运行、调试、性能、实例、迁移与附录的大纲。
- 版式样张：建立 A4 页面、标题层级、中文正文、代码与表格样式，确认 PDF 构建方式。
:::

本版面向 Zeus V3 单设备算子开发。安装时使用 SDK 发行说明列出的配套 Triton、PyTorch 后端、编译器和运行时；接口支持范围同时受 dtype、shape、布局及目标配置约束。正文中的带宽与算力换算都附有时钟和数据类型等条件，理论值不代表应用实测性能。

### D.2 示例索引 {.unnumbered}

| 示例文件 | 主要内容 | 对应章节 |
|---|---|---|
| `examples/check_environment.py` | 检查 Python、工具与可选运行时 | 快速上手 |
| `examples/vector_add.py` | BF16 加法、尾块与核内循环 | 上手、模型、编译与调优 |
| `examples/rmsnorm.py` | 行归约、FP32 统计与缩放 | 算子开发实例 |
| `examples/matmul.py` | M/N/K 分块与 `tl.dot` | 算子开发实例 |
| `examples/fused_add_rmsnorm.py` | 残差加法与归一化融合 | 算子开发实例 |
| `examples/append_kcache.py` | 单 token 的 KCache 页写入 | LDG 与 KV cache |
| `examples/append_vcache.py` | 单 token 的 VCache 页写入 | LDG 与 KV cache |
| `examples/ldg_layout.py` | 完整矩阵的 LDG 地址映射验证 | LDG 与 KV cache |
| `examples/softmax.py` | 固定行块的 FP32 Softmax | 算子开发实例 |
| `examples/quantized_matmul.py` | FP8 × INT4、BF16 UScale、FP32 输出 | 量化矩阵乘与 UScale |

Table: 随书示例

运行示例前先阅读其输入契约。向量加法、RMSNorm、Softmax 和融合模块提供 `--run-device`，经配套运行时执行并与 PyTorch 参考比较；省略该参数只计算 CPU 参考。Matmul 的 Host 接入在正文单独给出。量化与 KV 追加模块需要符合各自格式的 LDG 分配和宿主参数，不能向这些模块添加不存在的命令行选项。布局检查只验证坐标与地址公式，不执行设备 kernel。

修改案例时，从最小的整块输入开始，再覆盖尾块、多块、动态形状和算法专有边界。保留原始参考函数，使优化前后的实现能够使用同一输入进行比较。

### D.3 验证结果的阅读方式 {.unnumbered}

源码与命令检查用于确认接口存在、参数规则和调用关系；编译结果用于确认一组具体源码、类型、常量和工具链能够产生设备程序；CPU 参考用于确认数学定义或地址公式；功能仿真用于检查指定配置下的结果；硬件运行与性能测量还需要对应设备、SDK 和实际执行记录。这些结果分别回答不同的问题，不相互替代。

手册的 API 条目描述调用契约，不把某个单点结果扩大为全部 dtype、shape 和布局的笛卡尔积。特别是本书的 Matmul、Softmax 和 KV cache 示例，均在正文中给出具体布局与输入条件。对新组合，应沿用现有参考方法加入新样例，而不只检查编译是否成功。

v0.0.3 随书示例已完成 8 个完整 kernel 的编译、15 组 CPU 数学参考、11 组宿主布局检查与 16 组功能仿真。布局检查覆盖完整矩阵地址映射和 KV 追加边界；功能仿真覆盖向量、BF16/量化矩阵乘、KV 追加和 Softmax，后者另含非整块宽度下的极端有限输入。另检查向量加法宿主参数在 i32 溢出边界处的拒绝行为。

v0.0.5 扩展了宿主布局检查，逐坐标核对 Dense、Blocked、UScale 及专用 K/V 页，覆盖 731666 个元素位置和 16 个索引副本，并与配套 pack 实现做对照。该检查只在 Host 上执行，不调用设备 kernel；历史编译和仿真记录仍保留原被测源码身份。

这些记录限定于相应的类型、shape、布局和输入条件。RMSNorm 与融合归一化完成编译及 CPU 参考；Stream、Graph、AOT 和 Matmul 宿主接入片段完成接口核查。本版没有新增通过 JIT 路径执行的验证记录；随书检查未包含物理设备执行、整模型回归或硬件性能测量。

### D.4 参考资料 {.unnumbered}

本指南的章节组织与手册体例参考以下资料，技术接口以 Zeus 配套实现与实际验证为准。

1. 《Cambricon BANG C/C++ 编程指南》v1.9.0。用于参考硬件与编程模型的讲解顺序、性能分析到优化案例的组织方式，以及 A4 技术手册版式。
2. [《TritonGCU 编程指南》EFC3072，TopsRider 3.8 Update 1](https://support.enflame-tech.com/onlinedoc_dev_3.8/4-develop/triton_gcu_programming_guide/content/source/index.html)。用于参考执行模型、类型接口、调试、限制和迁移章节的划分。
3. Zeus 配套编译器的架构、ZECC、Executor 使用说明及随版本交付的算例。用于核对本书中的具体入口、参数和平台行为。

参考厂商的芯片规格、API 限制和性能数据不构成 Zeus 的规格。本文图示为针对 Zeus 编程过程重新绘制的示意图；图中的计算顺序、任务区间和数据归属由对应章节给出的条件定义。
