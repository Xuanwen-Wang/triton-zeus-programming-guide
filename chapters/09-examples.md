# 算子开发实例 {#examples}

本章把前面的模型、接口与调优方法连成完整算子。示例覆盖元素级分工、行归约、softmax、矩阵分块和保持数值契约的融合。每个示例先定义输入输出，再解释 tile 内部计算及全局地址，最后给出验证与优化顺序。

随书 `examples` 目录保存完整 Python 模块，正文选取决定语义的片段。直接运行模块会生成 CPU 参考；向量加法、RMSNorm、Softmax 和融合示例还提供显式的 `--run-device` 入口。该入口要求[快速上手](#quick-start)与[编译与运行](#compile-and-run)所述的配套 Triton、`torch_zeus` 和 SDK 环境，通过 JIT 提交到所配置的执行后端。Matmul 需要额外准备 LDG 权重，本节给出相应 Host 封装。后文的算法改写片段用于解释数据流变化，除非包含完整的参数和宿主准备，不作为独立可运行程序。

## 向量加法调优 {#vector-add-tuning}

### 从同一份输入建立基线

复用[第一个向量加法](#vector-add)，定义 $y_i=x_i+\alpha z_i$。X、Z、Y 都是连续 BF16 向量，输入提升到 FP32 后进行计算，最后写回 BF16。常规用例取 $\alpha=1$；同时保留非一标量用于验证运行时参数传递。

主线输入为 `numel=2500`，tile 大小为 `1024`，核数为 `2`。三个 tile 的逻辑区间分别为 `[0,1024)`、`[1024,2048)`、`[2048,2500)`。最后一个 tile 中只有 452 个元素有效。块指针的 shape 记录真实长度 2500，边界装载用零填充，边界写回只更新有效区域。

任务数为 $T=\lceil N/B\rceil$，核数为 $C$。令 $q=\lfloor T/C\rfloor$、$r=T\bmod C$，编号为 $c$ 的核处理 $q+[c<r]$ 个 tile，起始 tile 为 $cq+\min(c,r)$。这里的方括号表示条件成立时取一。这种分配使每个核得到连续 tile 区间，且任意两个区间不重叠。

| 核 | 起始 tile | tile 数 | 有效元素 |
|---|---:|---:|---:|
| core 0 | 0 | 2 | 2048 |
| core 1 | 2 | 1 | 452 |

Table: 2500 元素向量在两个核上的工作分配

这张表也解释了为什么“启动两个核”不等于“时间减半”。尾块较小时，一个核会更早结束。对 2048 元素整块输入，两核各处理一个 tile；对小于 1024 元素的输入，只有一个核有工作。分析性能时必须同时保留这些情形。

### 第一轮只调整 tile

先保留算法、输入、参考与核数，比较不同 `BLOCK`。新的 BLOCK 必须满足相关接口对 tile 形状的要求。较小 tile 增加分工粒度，较大 tile 减少循环次数；二者都可能改变搬运、L2 Buffer 占用和编译结果。对每个候选检查输出、两核时长和实际搬运区间，不从源码长度推断性能。

建议用 `1024` 作为基线，再选择经过编译器支持范围确认的较小与较大候选。每轮只改变一个常量，保存独立产物。若要将某个配置用于一组 shape，至少覆盖单 tile、整块、尾块和多轮循环，不能只按 2500 元素这一点选择全局默认值。

可以先算工作分配，再决定哪些候选值得编译。下面仍使用 2500 个元素和连续均分策略，列出的仅是逻辑元素数量。

| `BLOCK` | 总 tile 数 | core 0 有效元素 | core 1 有效元素 |
|---:|---:|---:|---:|
| 256 | 10 | 1280 | 1220 |
| 512 | 5 | 1536 | 964 |
| 1024 | 3 | 2048 | 452 |
| 2048 | 2 | 2048 | 452 |

Table: 改变 tile 后的分工推导，不是性能结果

256 的分配更均衡，但总共要处理十个 tile；2048 并未改善这个输入上的有效元素均衡。两者谁更快，要看小块固定开销、实际搬运粒度和并行资源。还可以用 2048 元素整块用例对照：同一组候选会形成另一种分布，因此不能把专门针对 2500 的选择直接推广到所有长度。

对于很长的向量，连续分配让每个核处理一段连续地址，便于描述符逐块前进。轮转分配则使 core c 处理 `c,c+C,c+2C,...`，可能改变尾块分工和读请求分布。两种算法都可以覆盖所有 tile，但起点和步进必须配套；不能保留连续分配的起点，再把循环步长单独改为核数。

### 第二轮检查表达式与搬运

核心计算保持为

```python
x32 = in1.to(tl.float32)
z32 = in2.to(tl.float32)
out = (x32 + alpha * z32).to(tl.bfloat16)
```

把中间结果提前转成 BF16 会改变数值行为；将 `alpha` 改成编译期常量会改变特化与缓存行为。二者都应作为独立改动说明。若希望合并逐元素操作，先查看生成 IR 是否已将它们安排为合适的计算序列，再检查是否仍有不必要的中间存储。

针对连续输入，检查每个 tile 的输入和输出偏移是否都由同一个 `block_id * BLOCK` 得到。如果输出总长度正确但部分区域重复，优先检查起始 tile 与循环步进。对无工作核，不应出现对有效输出区间的写入。

### 第三轮验证重叠与配置选择

在基础版本稳定后，才在 Triton kernel 中显式组织预取、两组工作区和可用的并行配置。向量加法的工作依赖较短，小输入通常受启动与调度成本影响较大；不能预设双缓冲或增加 lane 一定获益。保留同源 A/B 和 trace，用[性能分析与调优](#performance)的方法区分实际重叠、资源空闲与尾部工作量。

最终配置表应列出每种 shape 范围对应的 tile、核数、精度条件与测量环境。若采用动态选择，边界两侧都应回归。例如在 `N=1024` 附近切换配置，需要检查 `1023`、`1024`、`1025`。配置变化不能改变输出长度、stride 契约或数值参考。

### 从失败位置反推分工错误

如果 `[0,2048)` 正确、`[2048,2500)` 全部保持初始化值，先检查 core 1 是否得到一个 tile，以及起点是否为 2。如果 `[1024,2048)` 被重复写入而尾块遗漏，检查是否把 `start_block` 误写为 `core_id`。如果仅索引 2499 以后的保护区改变，检查输出边界；如果前面的结果也在错误，继续看真实 shape 与地址类型。

可以再用 `alpha=0` 检查输入 X 的搬运与输出转换，用 `alpha=1` 检查加法，用 `alpha=0.5` 检查标量 ABI 和乘法。这里仍从同一份 BF16 输入计算 FP32 表达式，不能用一种不同的中间舍入方式作为参考。`alpha=0` 不是性能基线，它可能触发不同的特化和化简，只作为定向诊断。

## 归约与 RMSNorm {#rmsnorm-example}

### 定义一行的计算

输入 X 的形状为 $[M,H]$，缩放向量 W 的长度为 H，输出 Y 与 X 同形状。对第 m 行，先计算平方均值，再计算逆均方根，最后逐元素缩放。

$$
s_m=\frac{1}{H}\sum_{h=0}^{H-1}x_{m,h}^2,
\quad q_m=\frac{1}{\sqrt{s_m+\varepsilon}},
\quad y_{m,h}=x_{m,h}q_m w_h.
$$

本例 X、W、Y 使用 BF16，平方、归约、逆平方根与缩放使用 FP32，最终结果转换为 BF16。$\varepsilon>0$ 是接口参数，属于数学定义的一部分。将其从 $10^{-5}$ 改为 $10^{-6}$，或把 W 的乘法提前进行，都需要重新确认参考计算。

先用一行六个数理解 padding。若 `x=[1,2,3,4,5,6]`、W 全一，平方和为 91，平方均值是 $91/6$。把这行补成八个位置后，补入的两个零不会改变平方和；分母仍然是六，不能使用 tile 宽度八。图中的小尺寸仅用于说明索引与数学关系。

![RMSNorm 的零填充、归约和广播](assets/rmsnorm.svg){width=100%}

### 单遍 tile 覆盖整行

完整程序为 `examples/rmsnorm.py`，采用以下编译期配置。

```python
BLOCK_TOKENS = 4
BLOCK_HIDDEN = 256
CORE_NUM = 2
```

测试输入使用 65 行、每行 256 列。一个 tile 覆盖四行，每行都在同一个 tile 内完成归约。65 行产生 17 个行块，其中最后一块只有一行有效。

列宽改为 250 时，tile 的另外六列用零填充；每行平方和仍除以 250。宿主封装拒绝 `hidden_size>256`，因为这个单遍实现没有列方向循环。把 `BLOCK_HIDDEN` 调大是另一份编译配置，需要重新验证资源、编译结果与代表形状。

```python
x = tl.load(
    input_block_ptr,
    boundary_check=(0, 1), padding_option="zero",
).to(tl.float32)
square_sum = tl.sum(x * x, axis=1)
inverse_rms = tl.rsqrt(square_sum / hidden_size + eps)
output = x * inverse_rms[:, None] * weight_f32
```

`axis=1` 沿列归约，将形状 `[BLOCK_TOKENS,BLOCK_HIDDEN]` 变成 `[BLOCK_TOKENS]`。增加 `[:,None]` 后，逆均方根变为列向量，沿每一行广播。W 则沿行方向广播。两个广播方向不同，用形状标注可以避免把某一行的统计量误用于其他行。

全局输入和输出采用连续行主序，地址为 `base + row * H + column`。两个核按照行块分工，一个核持有某行的完整归约域，因而不需要跨核合并平方和。边界检查分别保护无效行与无效列；输出写回不应修改补齐区域。

### 行很宽时需要改变算法

如果整行 tile 的 L2 Buffer 开销过大，可将一行拆成多个列块。第一遍遍历列块，累加 FP32 平方和；得到整行的逆均方根后，第二遍重新读取各列块，完成归一化与缩放。列尾块仍补零，除数仍使用真实行宽。

两遍算法增加了一次输入读取，但降低单个 tile 的宽度及同时活跃的数据量。单遍算法保留更多数据，可能减少外存流量。比较时应固定 FP32 累加与输出转换位置，同时记录列块之间的累加顺序；不能拿 BF16 中间计算的两遍版本与 FP32 单遍版本直接归因于流量差异。

### 两遍归约的完整数据流

设 `H=1000`、列块宽度 `B_H=256`，一行分为四块：`[0,256)`、`[256,512)`、`[512,768)`、`[768,1000)`。最后一块有 232 个有效值。第一遍每个块先得到局部平方和，再把四个局部和相加，最后只除一次真实行宽 1000。每个块各算均值再无权平均是错误的，因为尾块的有效元素数与前面三块不同。

下面是两遍实现的第一部分。`x_template` 的 shape 为完整 `(M,H)`，block shape 为 `(B_R,B_H)`；`row0` 是当前行块起点。代码展示算法结构，Host 分工和描述符创建沿用本节规则。

```python
sum_square = tl.zeros((B_R, 1), dtype=tl.float32)
for col0 in range(0, H, B_H):
    x = tl.load(
        tl.advance(x_template, (row0, col0)),
        boundary_check=(0, 1), padding_option="zero",
    ).to(tl.float32)
    part = tl.sum(x * x, axis=1, keep_dims=True)
    sum_square += part
inverse_rms = tl.rsqrt(sum_square / H + eps)
```

`sum_square` 的形状始终是 `[B_R,1]`。保留这个单例维度后，第二遍可以直接与 `[B_R,B_H]` 的 X 广播相乘，不需要把统计量先展平再重新推断轴。第一遍结束后，每行已经得到覆盖全部 H 列的统计量，第二遍才能写出最终结果。

```python
for col0 in range(0, H, B_H):
    x = tl.load(
        tl.advance(x_template, (row0, col0)),
        boundary_check=(0, 1), padding_option="zero",
    ).to(tl.float32)
    w = tl.load(
        tl.advance(w_template, (col0,)),
        boundary_check=(0,), padding_option="zero",
    ).to(tl.float32)
    y = x * inverse_rms * w[None, :]
    tl.store(
        tl.advance(y_template, (row0, col0)),
        y.to(tl.bfloat16), boundary_check=(0, 1),
    )
```

W 的模板覆盖完整 H，第二遍使用与 X 相同的列起点。若 W 总是从第 0 列读取，则第一块正确、后续块使用了错误权重；W 全一的测试发现不了该问题，应增加随列变化的 W。

这一实现每行读取 X 两次、写 Y 一次。只计 BF16 的 X/Y，流量为每行 `6H` 字节，整行单遍的相应流量为 `4H` 字节；W、对齐和中间存储另计。两遍的好处是每个 X tile 在局部和计算完成后可以被释放，不需要把整行保留到最终缩放。若同时融合了 residual 更新，则第一遍写回更新后的 residual，第二遍必须读取这份已经按契约舍入的值。

### 行分工和列分工的取舍

按行块切分时，每个核持有完整的平方和，只有行尾的边界问题。按列切分时，两个核各得到局部平方和，必须合并后才能计算同一个逆均方根；这又要求把结果分发给两个核。对普通行宽和足够多的行，按行切分通常是更直接的起点。

如果只有一行、行宽很大，按行策略无法同时利用两个核。此时是否值得按列拆分，要比较新增同步、部分和存储、统计量分发与减少计算的成本。不能只把列宽除以二，继续各自使用自己的局部平方和；那会把一个 RMSNorm 变成两个独立归一化。

对于单遍版本，减少 `BLOCK_TOKENS` 主要降低同时存活的行数；改变 `BLOCK_HIDDEN` 则同时改变列 padding 与归约结构。先试行块大小，再决定是否更换为两遍算法，可以让资源变化的原因更清楚。

### 验证与运行

先运行 CPU 参考，检查输出形状与有限性，再通过配套 SDK 的 JIT 路径执行并比较输出。

```bash
python examples/rmsnorm.py --rows 65 --width 256
python examples/rmsnorm.py --rows 65 --width 250
python examples/rmsnorm.py --rows 65 --width 250 --run-device
```

执行模式对这组有限随机输入使用 `atol=1e-2`、`rtol=1e-2`，并打印最大绝对误差和均方根误差。这是随书示例的检查条件，业务接入应冻结自己的误差预算。补充测试包括全零行、常数行、正负对称行，以及行数为 1、4、5 的边界。全零输入应输出零；常数输入可直接从公式计算参考。接近零的行能检查 $\varepsilon$ 的作用，较大输入则用于单独考察平方溢出的处理约定。

对两遍算法，还应增加 `H=B_H-1,B_H,B_H+1,4B_H-24`，检查列块边界。用一行只有最后一个元素非零的输入，可以直接发现漏读最后一块；用每个块不同的常数，可以发现局部和被重置或覆盖。把第二遍输出与第一遍保存的整行平方和一起检查，能把归约错误与权重/写回错误分开。

## Softmax：归约、中间精度与有效域 {#softmax-example}

### 稳定形式及两种不同的尾部

对一行长度 D 的输入，softmax 为 $p_j=\exp(x_j)/\sum_k\exp(x_k)$。直接求指数可能溢出，稳定实现先减去该行最大值：

$$
m=\max_j x_j,\qquad e_j=\exp(x_j-m),
\qquad p_j=e_j/\sum_k e_k.
$$

例如输入 `[1000,1001,1002]`，先减去 1002，指数变为 `[exp(-2),exp(-1),1]`，归一化约为 `[0.09003,0.24473,0.66524]`。这些数值只是数学算例。所有输入加上同一个常数，理想 softmax 应保持不变，这也是很有用的定向测试。

Softmax 的 padding 与 RMSNorm 不同。平方和中的无效元素填零不会增加结果；softmax 若将无效 score 填零再参加指数，则 `exp(0)=1` 会进入分母。以有效输入 `[-2,-1]` 为例，补两个零后执行四元素 softmax，会明显改变前两个元素的概率，即使最后只写回两个元素也无法修复。

需要区分两种有效域。行尾由真实 D 限定；attention 的有效 key 还可能由序列长度和因果条件限定。不能只保护最终输出的 shape，就认为 score 中的无效 key 已被排除。通用数学形式通常把无效 score 视为负无穷，并使其指数贡献为零；设备实现采用有限哨兵或专用有效长度机制时，要额外检查其数值范围和空行行为。

### 普通表达式与 Zeus 专用接口

下面的片段说明 FP32 softmax 的计算依赖，假定 `x` 已经是一个完整的有效行块，且每行至少有一个有效元素。它不是包含任意 mask 和异常值处理的完整 kernel。

```python
row_max = tl.max(x, axis=1, keep_dims=True)
exp_values = tl.exp(x - row_max)
row_sum = tl.sum(exp_values, axis=1, keep_dims=True)
probability = exp_values / row_sum
```

配套源码还提供 `tl.zeus.softmax(x, axis=1)`，用于相应支持形态的 Softmax_ATT 路径。例如完整 FP32 示例采用两核、每块一行、静态宽度 1024，并要求一条操作覆盖整个逻辑行。专用接口的有效长度和 lowering 与普通 `max/exp/sum/div` 组合不同，不能把普通表达式的补零方式直接当作专用接口的实现规范。

::: {.caution}
专用 `tl.zeus.softmax` 一次处理一行：输入 rank 为 1–4，只沿最内维归约，前导维必须静态为 1。多行输入应在 kernel 中逐行遍历；不能把随书的 `[1,1024]` tile 直接改成 `[M,1024]` 作为多行批处理。
:::

若要开发新的封装，可以先选 `D=1024` 的完整行，直接复用对应源码模板；确认正确后，再检查 `D<1024` 时生成描述符携带的有效范围和尾部结果。输出保护区测试能检查写回越界，概率和接近 1 的测试则检查分母是否错误地包含 padding。两种测试必须同时存在。

随书 `examples/softmax.py` 保留这一完整行模板，输入与输出均为 FP32，支持 `1<=width<=1024`，不包含因果 mask。Host 封装检查有限、连续的二维输入，空行数直接返回。以下命令分别生成 CPU 参考，以及通过 JIT 执行并与该参考比较。

```bash
python examples/softmax.py --rows 5 --width 1000
python examples/softmax.py --rows 5 --width 1000 --run-device
```

完整行 1024、尾宽 1000 和单元素行分别覆盖正常块、有效域与极小输入。执行模式采用 `atol=1e-5,rtol=1e-3`，这组阈值用于示例的有限随机输入，不替代业务数据的精度预算。`--run-device` 选择运行时执行路径，模拟后端与物理设备的配置区别见[运行后端](#runtime-backends)；随书已完成的验证范围见附录。

### 宽行与在线归约

把长行拆成多块后，每块独立做 softmax 再拼接不等于完整行 softmax，因为各块使用了不同的分母。可以使用两遍算法：第一遍计算全局最大值，第二遍计算指数及总和，再决定是否保留指数供最终归一化；也可以在扫描块时维护运行最大值和指数和。后者在 attention 中还能同步累计加权 V，避免写出完整概率矩阵。

设已有最大值 m 和指数和 l，新块最大值为 $\hat m$。合并时使用 $m'=\max(m,\hat m)$，旧和要乘 $\exp(m-m')$，新块的指数也必须在 $m'$ 下计算。省略旧和的缩放，会使每当新块出现更大 score 时结果错误。这一机制在[在线 Attention](#attention-online)中继续推导。

### 用不变量检查结果

有限输入、至少一个有效元素的行，概率应非负且总和接近 1；全相等输入应接近均匀分布；单个有效元素的概率应为 1；两行不同数据不得相互影响。再加入同常数平移、很负的有限输入和非整块长度。对全 mask 行必须在接口中另行规定输出，不能把 `0/0` 的结果当作自然定义。

输出 BF16 时，最终概率求和未必严格等于 1；检查时应在 FP32 中统计并使用预先定义的误差预算。还应保存最大概率、最小非零概率和出现差异的位置，区分最后的量化误差、指数近似与归约域错误。

## Matmul {#matmul-example}

### 明确逻辑矩阵与权重存储

本例计算 $C=A B$，其中 A 为 $[M,K]$，B 为 $[K,N]$，C 为 $[M,N]$。A 和 C 使用行主序。权重在示例接口中以 $W[N,K]$ 表示，逻辑上满足 $B=W^T$，因此参考计算是 `A @ W.T`。

Zeus 的权重存储路径与普通 GDG 张量指针不同。完整示例的 `w_ptr_list` 指向核对应的权重地址表，kernel 读取地址后，以 `memory_type="weight"` 装载。该地址由配套运行时或仿真框架按权重布局创建。普通张量的 `data_ptr()` 不能仅因数据同样是 BF16，就直接当作这个接口的权重地址。

为集中说明分块，`examples/matmul.py` 固定单核；`CORE_NUM=1` 是这个示例的接口条件。它仍保留核地址表形式，便于与工具链的 GEMM 示例一致。扩展到双核时，应同时设计权重分区和输出分区，不能只把启动 grid 从一改成二。

### 一个输出 tile 沿 K 累加

教学形状取 `M=256,N=256,K=384`，三个 tile 维度都取 128。输出包含四个 $128\times128$ tile，每个输出 tile 需要三个 K 块。对于输出左上角，依次读取 A 的 `[0:128,0:128]`、`[0:128,128:256]`、`[0:128,256:384]`，以及 W 中对应列区间的三个块。

![输出 tile 与三个 K 分块的对应关系](assets/matmul-tiles.svg){width=100%}

每轮装载得到 A tile 的形状 `[BLOCK_M,BLOCK_K]` 和 W tile 的形状 `[BLOCK_N,BLOCK_K]`。逻辑转置后，W tile 以 `[BLOCK_K,BLOCK_N]` 参与 `tl.dot`。FP32 累加器在进入 K 循环之前清零，在所有 K 块结束后才转换为 BF16。

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for ki in range(tl.cdiv(K, BLOCK_K)):
    k = ki * BLOCK_K
    a = tl.load(
        tl.advance(a_template, (m, k)),
        boundary_check=(0, 1), padding_option="zero",
    )
    w_nk = tl.load(
        tl.advance(w_template, (n, k)),
        memory_type="weight",
        boundary_check=(0, 1), padding_option="zero",
    )
    acc += tl.dot(a, tl.trans(w_nk))
```

这里使用普通 `tl.dot`，由编译器根据形状与 dtype 选择 PE 实现。`tl.trans` 表达逻辑维度交换，最终是否需要显式数据搬运，应看生成的布局和指令。调优时不在 kernel 中固定 PE 模式或 minitile。

累加器生命周期必须覆盖完整 K 循环。如果把清零放到循环内部，只会保留最后一个 K 块；如果每次都转回 BF16，则每轮都引入一次额外的低精度转换。这两种错误在 `K=128` 的单块用例上可能难以暴露，因此必须保留多个 K 块的测试。

### M、N、K 尾部各有职责

M 与 N 的尾部影响输出有效区域，存储时只写合法元素。K 的尾部影响乘积求和，A 和 W 的无效 K 元素必须填零。输出的边界检查不能补救错误的 K 输入；同样，K 填零也不能阻止越界输出写入。

在逻辑模型中，可把最后一个 A 或 W tile 看作补零后的矩形，但权重实际布局仍由配套工具按其边界规则准备。编译通过不能代替权重 packing 的验证。首先核对逻辑 W 与装载布局的对应，再比较 Matmul 输出；若采用离线打包，保存打包参数和产物标识。

一个具体的尾部算例是 `M=130,N=129,K=257`，tile 仍为 128。输出需要两个 M 块和两个 N 块；右下角输出 tile 只有 `2×1` 个有效元素。K 分三轮，最后一轮只有一个有效乘积项。此时 A 的最后块有效范围为 `2×1`，W 的最后块有效范围为 `1×1`，累加器仍要保留前两轮结果。

对于右下角的 `C[129,128]`，正确公式仍是

$$
C_{129,128}=Q\left(\sum_{k=0}^{256}
\operatorname{FP32}(A_{129,k})
\operatorname{FP32}(W_{128,k})\right).
$$

不能只因为最后一轮有一个有效 K 就把整个 dot 结果乘以 `1/128`，也不能按 tile 宽度对 Matmul 求平均。零填充的作用是让无效乘积贡献为零，真实乘积的权重不变。该算例用于推导索引；具体尾部形状能否走某个 LDG 打包和编译路径，应按对应配置验证。

### 通过循环次序获得输入复用

基础版本的循环顺序是 M 块、N 块、K 块，每个输出 tile 有一份贯穿 K 的累加器。若把相邻两个 N 块放在同一 K 循环中，就可以读取一次 A，同时更新两个累加器。以下片段省略边界装载的模板定义，保留了复用的核心依赖。

```python
acc0 = tl.zeros((B_M, B_N), dtype=tl.float32)
acc1 = tl.zeros((B_M, B_N), dtype=tl.float32)
for k0 in range(0, K, B_K):
    a = tl.load(tl.advance(a_tpl, (m0, k0)),
                boundary_check=(0, 1), padding_option="zero")
    w0 = tl.load(tl.advance(w_tpl, (n0, k0)),
                 memory_type="weight", boundary_check=(0, 1),
                 padding_option="zero")
    w1 = tl.load(tl.advance(w_tpl, (n0 + B_N, k0)),
                 memory_type="weight", boundary_check=(0, 1),
                 padding_option="zero")
    acc0 += tl.dot(a, tl.trans(w0))
    acc1 += tl.dot(a, tl.trans(w1))
```

写回时 `acc0` 对应 `(m0,n0)`，`acc1` 对应 `(m0,n0+B_N)`，各自保护 M/N 边界。如果输出只剩一个 N 块，选择只包含一个累加器的尾部路径或经验证的受保护实现，避免无意义地为第二个输出保留整块资源。此改写对源码装载数的影响与工作集计算见[LDG 访问、布局与打包](#ldg-performance)。

另一种复用是让一个 W tile 服务多个 M 块，它对较大的 M 更有吸引力。但要同时保存多个 M 块的累加器，也可能增加 A 的驻留。不要仅根据权重更大就固定跨 M 复用：短 batch decode 与大 batch prefill 的工作量形状不同，适合的循环次序也可能不同。

### 从单核扩展到双核

若按 N 分工，每个核拥有一部分 W 的行，以及 C 的对应列；A 可以由两核读取同一个逻辑输入。一个核内部仍保留完整 K 累加，不需要跨核加和。Host 必须将 `W[N,K]` 按同一 N 分区打包，并生成与核编号对应的 LDG 基址对象。

例如 `N=256`、两核平均切分，core 0 的本地 W 是全局 `[0:128,:]`，core 1 是 `[128:256,:]`。本地权重描述符的 N 起点从 0 开始，而输出 C 的列起点分别是 0 和 128。把全局输出列偏移再次加到 core 1 的本地 W 上，会重复偏移，导致读到错误区域。

N 不能整除核数或分区对齐时，有两种清楚的接口：拒绝这组形状，或显式提供每核起点和有效列数。简单使用 `N//CORE_NUM` 会丢掉余数。按 K 分工则需要另一个阶段合并 FP32 部分和；合并的运算顺序、临时存储和同步均应纳入实现与性能比较。单核随书示例没有暗含这种合并机制。

### 用结构化输入检查矩阵乘

随机矩阵适合观察总体误差，以下三类结构化输入更适合定位实现错误。A 的某一行只有一个位置为 1，可以检查 W 某列的读取；W 的某一行只有一个位置为 1，可以检查 A 某列与输出 N 的对应；A/W 按 K 块取不同小常数，可以检查是否漏掉前几轮、重复某轮或在循环内清零。

FP32 累加也会因为乘积和归约顺序而出现不同舍入。先分别比较每个 K 块的部分积，再比较最终 FP32 和，最后比较 BF16 转换。对于最后一步才产生的小差异，应检查误差是否位于 BF16 舍入中点附近；对于整行或整列大范围错位，应先查布局，不要把它归为数值误差。

### 宿主封装与结果比较

CPU 参考使用已经转换为 BF16 的 A、W，提升到 FP32 做矩阵乘，最后转换为 BF16。先运行默认形状，再按 M、N、K 分别构造尾部情况。

```bash
python examples/matmul.py --m 256 --n 256 --k 384
```

这个模块的命令行只计算 CPU 参考，没有 `--run-device` 参数。下面的 SDK 宿主示例展示完整调用：将两段代码顺序保存为脚本，从随书根目录运行。先建立 BF16 输入，再按本核的普通 weight 格式准备 W；其中 `packed` 是拥有存储的 LocalMem 对象。

```python
import torch
import torch_zeus
from examples.matmul import matmul_bf16, constants, reference

m, n, k = 256, 256, 384
gen = torch.Generator().manual_seed(20261124)
a_host = torch.randn(m, k, generator=gen).to(torch.bfloat16)
w_host = torch.randn(n, k, generator=gen).to(torch.bfloat16)
a = a_host.to("zeus:0")
out = torch.empty((m, n), dtype=torch.bfloat16, device="zeus:0")
packed = torch.zeus.local_memory.from_tensor(
    w_host, kind="weight", Tr=1, Tc=1, core_id=0,
)
```

```python
matmul_bf16[(1,)](
    a, packed, out, m, k, n,
    **constants, zeus_enable_jit=True,
)
torch.zeus.synchronize(0)
actual = out.cpu()
expected = reference(a_host, w_host)
torch.testing.assert_close(
    actual, expected, rtol=1e-2, atol=1e-2,
)
print("max_abs_error=", (actual.float() - expected.float()).abs().max())
```

启动 grid 和 `constants["CORE_NUM"]` 都为 1；`packed`、A 与输出必须保持有效直到执行完成。参考直接使用同一份 BF16 输入，执行 FP32 矩阵乘后转换为 BF16。上述容差是有限随机输入的初始检查条件，不保证任意数据范围都满足；业务接入应按误差预算固定阈值，并保留最坏误差位置。若要求按位一致，还需对齐 dot 与归约顺序。该宿主封装的接口与布局已核对，随书检查不包含它在物理设备上的执行结果。

### 优化顺序

优化按三个层次进行。第一层调整 M/N/K 分块，观察资源和重复搬运；第二层调整循环与复用，确认某个 A 或权重 tile 是否被重复读取；第三层在 kernel 中显式组织预取和双缓冲，验证 K 块之间是否真正重叠。每一层都保留前一层的数值契约与性能基线。

::: {.tip}
先用单缓冲获得正确、可复现的基线，再改写预取和缓冲复用。`--enable-double-buffer` 与 `double_buffer_hint()` 是尚不完善的实验性自动功能，暂不推荐作为本例的优化入口。
:::

::: {.keep}
## Fused Add RMSNorm {#fused-add-rmsnorm-example}

### 先固定两个可见输出

输入 X 与 residual R 均为 `[M,H]` 的 BF16 连续张量，W 长度为 H。独立算子链先更新 residual，再对更新后的 residual 执行 RMSNorm。定义 $Q$ 为 BF16 转换，则本例的契约为

$$
R'=Q\bigl(\operatorname{FP32}(X)+\operatorname{FP32}(R)\bigr),
\qquad Y=Q\bigl(\operatorname{RMSNorm}_{\mathrm{FP32}}(R',W)\bigr).
$$
:::

Y 和 $R'$ 都是可见输出。完整实现将 Y 写回 X 的存储，将 $R'$ 写回 residual 的存储。宿主示例为两个输入分别创建设备副本，使两块输出存储互不重叠。调用者若需要保留原始 X 或 R，也应先保留副本。

这里的转换位置决定融合是否等价。独立链把 residual 写成 BF16，下一次 RMSNorm 读到的是这个 BF16 值；融合版本必须让归一化使用同一个已转换的值。直接用未转换的 FP32 和做平方和，即使误差较小，也改变了计算过程。

![保留 residual 输出与舍入边界的算子融合](assets/fusion.svg){width=100%}

### 在同一个行 tile 内完成计算

与 RMSNorm 一样，每个 tile 覆盖若干完整行。先把 X 与 R 都读入，再计算更新 residual；之后写回 residual，并保留它的值用于归一化。FP32 平方和、逆平方根和缩放与独立 RMSNorm 的定义一致。

```python
updated = (x_f32 + residual_f32).to(tl.bfloat16)
tl.store(residual_block_ptr, updated, boundary_check=(0, 1))

value = updated.to(tl.float32)
square_sum = tl.sum(value * value, axis=1)
inverse_rms = tl.rsqrt(square_sum / hidden_size + eps)
output = (value * inverse_rms[:, None] * weight_f32)
tl.store(input_block_ptr, output.to(tl.bfloat16),
         boundary_check=(0, 1))
```

源码中 `updated.to(tl.float32)` 使用已有中间值，不要求再从全局 residual 读取。但编译后的缓冲和搬运仍需查看实际产物。两个输入都在覆盖前完成读取，且不同 tile 处理不同的行；这些条件保证原地更新不会破坏尚未使用的输入。

### 计算实际节省的流量

只考虑 X、R、Y 的 BF16 数据，忽略 W 的复用、对齐与缓存，独立链首先读取 X、R 并写回 $R'$，随后读取 $R'$ 并写出 Y，共产生每元素 10 字节的逻辑流量。融合版本读取 X、R，写回 $R'$ 与 Y，共每元素 8 字节，少了一次 residual 读取。

这些数是流量推导，不是耗时结果。融合还把两次 kernel 启动变成一次，同时增加一个 kernel 内部的活跃状态。实际收益要比较同样的输入、输出、转换规则与计时范围。若去掉 residual 输出，得到更小流量的同时也改变了接口，不能用于这一对照。

### 宽行融合仍然需要两遍

如果整行无法驻留，可以在第一遍读取 X 与 R，计算 BF16 的 $R'$，写回 residual，同时累加 $\operatorname{FP32}(R')^2$；第二遍读取 $R'$ 与 W，完成归一化并写 Y。这条路径保留两个输出与中间转换契约，但不再消除归一化阶段的 residual 读取。

在相同的 X/R/Y 流量口径下，宽行两遍融合仍是每元素 10 字节：读 X、读 R、写 $R'$、再读 $R'$、写 Y。它相对分离 kernel 的优势主要可能来自少一次启动及内部组织，而不应继续宣称“每元素 8 字节”。是否保留单遍或两遍，应由工作集和完整测量决定。

第一遍不能在计算平方和之后再把 residual 转 BF16。否则平方和使用的是尚未舍入的和，第二遍读取的却是 BF16 residual，两个阶段内部已经不一致。正确顺序是“加法→BF16 转换→写回与平方和”，两个消费者看到同一个值。

### 原地更新的别名边界

两个输入使用不同变量名不等于不同存储。若 X 与 R 指向同一块内存，先写 residual 再写归一化输出会让最终可见的两个输出重叠；接口也无法同时保留 Y 和 $R'$。随书封装要求两份独立存储。部分重叠的切片更难处理，应同样拒绝，除非实现明确支持并验证了该别名模式。

W 也不能位于将被 X/R 覆盖的区域。否则前面行块的更新可能改变后面行块使用的权重，造成对处理顺序敏感的结果。调用者应在入口确定只读权重、原地输出以及仍被其他消费者需要的旧输入；融合只消除多余的读写，不改变这些数据的所有权。

### 同时验证输出、残差与重复调用

完整模块为 `examples/fused_add_rmsnorm.py`，默认配置与 RMSNorm 相同。

```bash
python examples/fused_add_rmsnorm.py --rows 65 --width 250
python examples/fused_add_rmsnorm.py \
  --rows 65 --width 250 --run-device
```

对 `(output, residual)` 分别报告最大误差：归一化沿用 RMSNorm 对有限随机输入的容差，residual 与独立 BF16 Add 逐元素比较，`atol=rtol=0`。补测全零、相消及正负混合的 X/R，检查残差转换位置。性能对照保持相同环境、向量加法与 FP32 RMSNorm；每轮恢复原始 X/R，kernel 计时排除恢复副本和结果回读，端到端计时按业务边界确定。
