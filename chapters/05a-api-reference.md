# Zeus 编程接口 {#api-reference}

Zeus 在 Triton 的张量表达式上增加了专用存储路径、计算操作和执行策略。本章说明这些接口如何与 LDG 布局、量化参数和核内执行资源配合。索引、广播、类型转换和普通归约的通用规则见[数据类型与语言规则](#language-and-api)；LDG 的字节排列与资源准备见[LDG 数据布局与访问](#ldg-memory)。

接口片段运行在 `@triton.jit` 函数中，使用以下两个入口。`tl.zeus` 提供专用计算；`ztl` 提供执行提示、提前退出及显式查表计算。调用形式省略编译器内部的 `_semantic` 参数，大写尺寸及策略参数在编译期确定。

~~~python
import triton
import triton.language as tl
from triton.backends.triton_shared import language as ztl
~~~

## 接口与资源契约

选择一个 Zeus 接口时，应先确定三件事：指针代表哪种资源、数据按什么格式组织、操作会产生什么效果。把地址转换成某种指针类型，只改变编译器对它的解释，不会分配 LDG、搬运数据或重新排列字节。

| 入口 | 主要用途 | 调用前需要确定 |
|---|---|---|
| `tl.load` 的扩展参数 | LDG 矩阵、量化权重及 K/V | 布局、描述符及页组织 |
| `tl.store` 的扩展参数 | LDG 原始传输或 LW 格式写入 | 目标分配、元素类型及写入区间 |
| `tl.zeus` | 专用 Softmax、分段函数等 | 输入域、形状与数值误差 |
| `ztl` 的执行提示 | 单操作拆分或独立工作调度 | 操作类型、依赖及资源容量 |
| `ztl.early_exit_if` | 设备端跳过后续工作 | 标志值与输出保留语义 |

Table: Zeus 接口的使用前提

本章给出可用于组织 kernel 的接口契约，不把模块中的全部符号视为可互换的算子库。未列出的接口不应仅凭同名函数直接替换已有实现；例如 RMSNorm 可按[归约实例](#rmsnorm-example)明确表达累加精度、归约长度和输出类型。原子操作、GPU 专属异步指令等也不能从上游 Triton 的接口列表直接推定为 Zeus 功能。

::: {.note}
`memory_type="weight"` 是历史保留的接口字符串，表示相应的 LDG 访问路径。LDG 可以承载权重、KV Cache 和 UScale；接口名称不限制其中的数据用途，也不替调用者完成格式转换。
:::

## LDG 装载接口 {#api-memory}

### load：参数与合法组合 {#api-load}

`tl.load` 保留 Triton 的普通指针规则，并增加 LDG 访问参数。公开调用形式为：

~~~python
tl.load(
    pointer, mask=None, other=None, boundary_check=(),
    padding_option="", cache_modifier="",
    eviction_policy="", volatile=False,
    memory_type="default", weight_dtype=None,
    ugroup=None, page_size=None, uscale=None,
)
~~~

普通指针和指针张量使用 `mask` 与 `other`；块指针使用 `boundary_check` 和 `padding_option`，不能同时传入 `mask` 或 `other`。以下专用矩阵访问以 `tl.make_block_ptr` 建立的块指针为基础，描述的是逻辑矩阵；其物理字节仍须符合相应的 LDG 格式。

| `memory_type` | 访问对象 | 扩展参数 |
|---|---|---|
| `"default"` | 普通设备数据 | 保持专用参数为空 |
| `"weight"` | LDG 矩阵及量化权重 | 可配 `weight_dtype`、`ugroup`、`uscale` |
| `"kcache"` | 专用 K 索引及 payload 布局 | 必须指定 `page_size` |
| `"vcache"` | 专用 V 索引及 payload 布局 | 必须指定 `page_size` |
| `"uscale"` | UScale 页描述路径 | 普通量化调用使用 `uscale=` 绑定 |

Table: load 的公开 memory_type 与参数配对

`"uscale"` 不表示“把任意 scale 张量读出来再逐元素相乘”。量化 kernel 使用 `memory_type="weight"`，通过 `ugroup` 指定分组，通过可选的 `uscale` 指针绑定物理页。内部组合字符串不属于应用接口；不要自行拼写带冒号的 memory_type，也不要把 `"key"`、`"value"` 当成 `"kcache"`、`"vcache"` 的别名。

缓存修饰符与 `volatile` 沿用函数签名，不提供跨 stream 的完成顺序。读侧所依赖的搬运、LW 写入或 scale 更新必须先完成；所有资源的 owner 应持续存活到最后一次消费结束。

### 普通 LDG 矩阵

矩阵乘 $A[M,K]B[K,N]$ 的右操作数在 LDG 接口中使用逻辑视图 $W[N,K]=B^T$。`shape`、`strides` 和 `offsets` 描述 W 的逻辑坐标，不能用于把已 pack 的 LDG 字节解释为普通行主序数组。

下面读取完整的 BF16 tile。`weight_ptr_list` 是运行时准备的每核 64 位地址表；`N`、`K` 和 tile 起点须与该分配一致。

~~~python
core = tl.program_id(0)
address = tl.load(weight_ptr_list + core)
base = address.to(tl.pointer_type(tl.bfloat16))
w_desc = tl.make_block_ptr(
    base, (N, K), (K, 1), (n0, k0),
    (BLOCK_N, BLOCK_K), (1, 0),
)
w = tl.load(w_desc, memory_type="weight")
~~~

片段假定当前 tile 完整落在逻辑矩阵内。处理尾块时，需要同时满足描述符边界、pack 补齐和后续矩阵乘的形状要求；不能仅在 store 端增加 mask。多个独立 packed block 还须按它们各自的基址和完整形状建描述符，不能用一套 Dense stride 推断所有分块位置。

`weight_dtype` 默认是 `None`，普通 BF16、FP8 或 INT8 路径由指针元素类型决定。当前非空值只接受 `"int4"`；传入 `"bf16"`、`"fp8"` 或 `"int8"` 不会选择对应格式，而会违反接口条件。

### INT4、ugroup 与 UScale {#api-quantized-load}

`weight_dtype="int4"` 表示右操作数使用 packed INT4 权重。它不会对一份浮点矩阵即时量化，也不会把 FP8 指针背后的每个字节自动转换为两个合法权重。应用必须先按约定产生 INT4 码、排列 nibble，并准备配套 scale。

`ugroup` 表示沿收缩维 K 的量化分组长度，单位是逻辑权重元素。它必须与 pack 时的分组一致。改变 `ugroup` 会改变 scale 与权重的对应关系，不是只影响性能的 tile 选项。具体分组、页坐标和计算示例见[量化矩阵乘与 UScale](#quantized-matmul)。

~~~python
w = tl.load(
    w4_desc, memory_type="weight",
    weight_dtype="int4", ugroup=128,
)
c = tl.dot(a_fp8, tl.trans(w), out_dtype=tl.float32)
~~~

此处 `w4_desc` 使用逻辑 $[N,K]$ 视图及 FP8 E4M3FN 指针类型，物理内容按 INT4 格式组织；`a_fp8` 是配套的 FP8 E4M3FN 激活 tile。随书例固定 $M=4,K=128,N=256$，一个 K 组含 128 个元素。这里的类型组合不能推广为任意激活类型与 INT4 权重都可相乘。

省略 `uscale` 时，量化路径按与权重相邻的布局寻找 scale。若 scale 独立存放，则把物理页首传入 `uscale`：

~~~python
scale_address = tl.load(scale_ptr_list + core)
scale_base = scale_address.to(
    tl.pointer_type(tl.bfloat16),
)
scale_page = scale_base + page_id * 8192

w = tl.load(
    w4_desc, memory_type="weight",
    weight_dtype="int4", ugroup=128,
    uscale=scale_page,
)
~~~

`8192` 是一个 16 KiB 页中的 BF16 元素数。`scale_page` 必须是指向 LDG 页首的**标量 BF16 指针**，不能是整数地址、指针张量或块指针。weight 端必须是块指针，且调用须显式指定 `ugroup`。传入的页需要已经按 UScale 格式排列；普通 GDG 中的二维系数矩阵不符合此契约。

::: {.caution}
`uscale` 绑定的是 16 KiB 对齐的物理页，不是某个 scale 元素。页内定位由分组与矩阵坐标共同决定。把逻辑组号作为地址，或用普通系数张量替代物理页，即使元素类型相同，也会读取错误位置。
:::

UScale 也可用于带分组缩放的 FP8 权重：此时保留 `ugroup` 及必要的 `uscale`，省略 `weight_dtype`。是否使用 INT4 与是否使用分组 scale 是两个独立问题，不能把二者视为同一个开关。

### 专用 K/V 装载与 page_size {#api-cache-load}

专用 K/V load 通过索引组织定位 payload。描述符基址代表该核的专用索引表，不能换成普通软件页表，也不能直接换成已解析的 payload 地址。表的编码和页内排列见[专用 K/V 页与索引表](#ldg-memory)。

`page_size` 是物理页可容纳的逻辑 token 数，必须显式给出不小于 128 的 2 次幂。它不等于当前有效序列长度，也不等于一次读取的计算 tile 长度。除 `"kcache"`、`"vcache"` 外，其他 memory_type 不接受该参数。

下面固定 BF16、每头 128 维、物理页长 1024，一次读取 128 个 token。`k_index_base` 指向按此格式准备的索引表，`seq_start` 是当前 tile 的逻辑起点。

~~~python
k_desc = tl.make_block_ptr(
    k_index_base, (1024, 128), (128, 1),
    (seq_start, 0), (128, 128), (1, 0),
)
k_tile = tl.load(
    k_desc, memory_type="kcache", page_size=1024,
    boundary_check=(0, 1), padding_option="zero",
)
scores = tl.dot(q_tile, tl.trans(k_tile))
~~~

`page_size=1024` 与 `block_shape=(128,128)` 同时出现是正常的：前者选择页格式，后者选择一次计算的逻辑范围。用于 PV 的 V 描述符以 $[D,S]$ 解释 PE 右操作数；其物理排列仍是专用 V 格式，不能由普通 `pack(V.T)` 替代。

::: {.caution}
页容量和有效 token 数必须分别维护。K 的无效位置即使读成零，也可能在 Softmax 中产生非零概率。注意力还需显式处理有效长度与因果 mask；专用页访问不会自动替应用完成这些判断。
:::

普通 `memory_type="weight"` 承载 KV 的路径采用软件页表和普通 packed 池，是另一套完整契约。更换读取模式时，必须同时更换相应的资源组织，不能只改一处字符串。

## LDG 写入接口 {#api-store}

### store：原始传输与 LW 转换

~~~python
tl.store(
    pointer, value, mask=None, boundary_check=(),
    cache_modifier="", eviction_policy="",
    memory_type="default", store_mode="default",
    blocksize=None,
)
~~~

`store_mode="default"` 保持普通存储语义；`store_mode="lw"` 请求从 L2 Buffer 中的逻辑矩阵写成对应的 LDG 物理排列。两种模式不能互相替代。

| `memory_type` | `store_mode` | `blocksize` 与效果 |
|---|---|---|
| `"default"` | `"default"` | 不传；普通设备写入 |
| `"weight"` | `"default"` | 不传；LDG 原始传输，不做 pack |
| `"weight"` | `"lw"` | 必传；转换为普通 LDG 格式 |
| `"kcache"` | `"lw"` | 必传；写专用 K payload |
| `"vcache"` | `"lw"` | 必传；写专用 V payload |

Table: store 的合法模式组合

`"kcache"`、`"vcache"` 不接受默认 store 模式，`"default"` 不接受 LW 模式；store 不提供 `memory_type="uscale"`。默认模式传入 `blocksize` 也会被拒绝。更新 UScale 时，应按其物理页契约准备正确的字节与传输，不能另造一种 scale store 组合。

普通块 store 的 value shape 和元素类型应与目标块指针一致，类型转换在写入前显式完成。LW 进一步要求源与目的元素类型一致，只接受对应的 INT8、FP8 和 BF16 路径；FP16、FP32 和 INT4 不属于 LW 的输入类型。INT4 权重应使用预先打包的上传路径。

### blocksize 与目标描述符 {#api-lw-store}

LW 的 `blocksize` 必须是不小于 128 的 2 次幂。它描述布局中的逻辑分组粒度；在专用 K/V 路径中，对应页的 token 容量。它不是字节数、16 KiB miniblock 的大小，也不是强制每次写入那么多行。

以下片段把一个完整 $[128,128]$ BF16 tile 写成普通 LDG 布局。`ldg_base` 来自该核的 LDG 地址表，`values` 是同形状、同元素类型的逻辑矩阵。

~~~python
dst = tl.make_block_ptr(
    ldg_base, (128, 128), (128, 1),
    (0, 0), (128, 128), (1, 0),
)
tl.store(
    dst, values, memory_type="weight",
    store_mode="lw", blocksize=128,
)
~~~

LW 的有效数据 tile 至少有两个维度；如有额外前导维，它们必须为 1。一个调用不能把多份独立矩阵藏在非单例前导维中，期望 LW 自动逐份选择分配。批量数据应逐个建立相应的目标视图。

LW 的目标必须能追溯到真实的 LDG 分配。类型转换和存储模式参数都不会创建这份分配，也不会将 GDG 指针变成合法的 LW 目的地。读写同一资源时，应分别建立读侧和写侧块指针，不要让同一块指针视图同时携带互相冲突的 weight 读取与 LW 写入模式。

::: {.tip}
先固定数据格式，再选择写入粒度。对完整矩阵更新检查读回值；对追加更新还要检查未写区域保持原值。只比较整页平均误差，容易遗漏占比很小的 token 写入错误。
:::

### 直接追加 K/V payload

专用 K/V 写端与读端的地址含义不同：LW 追加接收**已解析的 payload 页地址**，不再执行索引表查找。下面写入 BF16 的一个 token，页容量为 1024，head dimension 为 128。

~~~python
dst = tl.make_block_ptr(
    payload_base, (1024, 128), (128, 1),
    (token_offset, 0), (1, 128), (1, 0),
)
tl.store(
    dst, token, memory_type="kcache",
    store_mode="lw", blocksize=1024,
)
~~~

`token` 的形状为 $[1,128]$，`token_offset` 应在 $[0,1024)$ 内；宿主在调用前验证此范围。目标页分配、未写部分的内容和 owner 生命周期都由调用者维护。下一页的追加需要重新选择 payload，并从页内位置零开始。

写 V 时，输入仍为 token-major 的 $[1,128]$，目标逻辑描述仍为 $[1024,128]$，改用独立 V payload 及 `memory_type="vcache"`。LW 完成对应的排列转换，不需要先对 token 做软件转置。

::: {.caution}
专用 K/V 的 load 接收索引表，LW store 接收 payload。两者即使使用同一个 memory_type 字符串，基址也不是同一种对象。不能把读侧索引表直接作为追加目标，否则会破坏后续寻址。
:::

完整追加 kernel 及页边界检查见 `examples/append_kcache.py` 和 `examples/append_vcache.py`。普通页池的 token 追加则按[分页 KV Cache 与 Attention](#attention-cache)的 slot 和普通布局规则处理。

## 专用计算与数值契约 {#api-zeus-extensions}

### dot：LDG 右操作数与累加 {#api-dot}

矩阵乘继续使用 `tl.dot`，由编译器选择 PE 实现和内部 tile 参数。调用者描述逻辑矩阵及数值类型，不手动指定 PE 模式。

~~~python
product = tl.dot(a, tl.trans(w), out_dtype=tl.float32)
acc = acc + product
~~~

`a` 的形状为 $[M,K]$，`w` 是从 LDG 读取的逻辑 $[N,K]$ tile，结果为 $[M,N]$。代码中的转置表达数学方向，不表示宿主还需对 packed LDG 字节再次转置。K 循环每步沿 A 与 W 的第二维推进，并把同形状的部分结果加到循环外初始化的 accumulator。

Zeus PE 没有通用的非零 C 输入端口。跨 K 块累加应写成上面的独立 FP32 加法，不使用非零第三参数来要求原生 `dot(a,b,c)`。FP32 accumulator 也不会恢复输入量化前的精度：BF16、FP8 或 INT4 的误差已进入乘法操作数。

`out_dtype` 是输出类型要求，不是任意混合 dtype 的许可。普通 BF16 矩阵乘与 FP8×INT4 量化矩阵乘分别采用[矩阵乘实例](#matmul-example)和[量化实例](#quantized-matmul)的类型、布局及 scale 契约。TF32 等其他设备的精度模式不能作为 Zeus 的通用选项套用。

### softmax：单行专用归约 {#api-zeus-softmax}

~~~python
tl.zeus.softmax(x, axis=-1)
~~~

该接口计算数值稳定的 Softmax，输入与输出 shape、dtype 相同。Zeus 专用路径接受 FP32 或 BF16；rank 为 1 至 4，归约轴必须是最内维，其他前导维必须静态为 1，最内维长度必须静态为正。因此 $[1,N]$ 可以表示一行，$[M,N]$ 在 $M>1$ 时不能一次调用后期待独立处理 M 行。

下面延续随书 FP32 示例，一次处理一行，静态宽度为 1024。`row` 是有效行号，`width` 在 1 至 1024 之间，描述符保留真实的矩阵宽度。

~~~python
src = tl.make_block_ptr(
    input_ptr, (rows, width), (width, 1),
    (row, 0), (1, 1024), (1, 0),
)
dst = tl.make_block_ptr(
    output_ptr, (rows, width), (width, 1),
    (row, 0), (1, 1024), (1, 0),
)
x = tl.load(
    src, boundary_check=(0, 1), padding_option="zero",
)
y = tl.zeus.softmax(x, axis=1)
tl.store(dst, y, boundary_check=(0, 1))
~~~

这个写法将真实边界交给专用归约路径；不要先把 source 的逻辑宽度也伪装成 1024，再期待算子猜出有效列数。完整 kernel 通过循环分配多行，不能把 `block_shape` 的第一维增大来代替该循环。

::: {.caution}
这里的零填充属于专用 Softmax 的描述符有效域契约。改写为普通 `max/exp/sum` 时，零填充列会参与归约；全负输入尤其容易出错。还应检查有效列行和、尾部保护及全负输入，不能只查越界。
:::

专用 Softmax 不是任意长度的跨 tile 在线归约。单行超过一个 tile 时，应使用具有显式状态更新的算法；其状态契约见下一节。`tile_softmax` 也不应仅凭相近的名称与本接口互换。

### 在线 Softmax 状态更新 {#api-online-softmax}

~~~python
tl.zeus.fa2_online_softmax_update(
    scores, m_state, l_state, axis=-1,
)
~~~

该接口完成一次 FlashAttention 在线 Softmax 的状态更新。`scores` 必须是静态 FP32 矩阵 $[H,S]$，两维均为正；`m_state` 和 `l_state` 都是静态 FP32 $[H,1]$，不能简写为 $[H]$。只支持沿最后一维更新，`axis` 可写 1 或 -1。

返回值依次为新的运行最大值、运行指数和、旧状态缩放系数、当前指数 tile 和 LSE：

~~~python
m_new, l_new, alpha, p, lse = (
    tl.zeus.fa2_online_softmax_update(
        scores, m_state, l_state, axis=1,
    )
)
~~~

设当前 tile 的行最大值为 $m_t$，其状态关系为

$$
m_{\mathrm{new}}=\max(m_{\mathrm{state}},m_t),\qquad
\alpha=\exp(m_{\mathrm{state}}-m_{\mathrm{new}}),
$$

$$
p=\operatorname{ReLU}
  \bigl(\exp(\mathrm{scores}-m_{\mathrm{new}})-10^{-30}\bigr),
\qquad
l_{\mathrm{new}}=\alpha l_{\mathrm{state}}+\sum p.
$$

减去微小常数并截断到零，用于抑制指数近似的残留。返回的 `p` 尚未除以运行指数和；它不是一份行和为 1 的最终概率矩阵。LSE 按 $m_{\mathrm{new}}+\log(\max(l_{\mathrm{new}},1))$ 形成。

调用者负责沿历史 tile 循环传递状态、处理有效长度与因果 mask，以及使用 `alpha` 缩放已有输出 accumulator 后再合并当前 `pV`。此接口不替调用者读取 V、不更新最终输出，也不完成最后的归一化。状态初始化和全 mask 行的处理必须与整个注意力算法一致，不能只把普通 Softmax 调用换成这个多返回值接口。

### sigmoid、log 与 softplus {#api-activation}

这些入口选择 Zeus 的专用近似路径。它们保持输入输出形状；使用 FP32 输入建立数值基线，最终低精度输出在计算后显式转换。方法名称用于选择实现，不承诺对任意输入域、dtype 或模型误差预算都适用。

~~~python
tl.zeus.sigmoid(x, approx="fast", seg_num=64)
tl.zeus.log(x, approx="accurate", seg_num=64)
tl.zeus.softplus(x, approx="fast", seg_num=64)
~~~

`sigmoid` 的 `approx` 接受四种值：

| 方法 | 计算组织 | 使用时关注 |
|---|---|---|
| `"fast"` | 直接分段线性拟合 | 输入范围及拟合误差 |
| `"balanced"` | 两级查表级联 | 固定系数布局 |
| `"accurate"` | 指数与倒数组合 | 中间舍入及最终误差 |
| `"cascade_add"` | 两级分段结果显式相加 | 中间值和额外操作 |

Table: sigmoid 的专用近似方法

`seg_num` 是编译期参数，默认 64。`"balanced"` 和 `"accurate"` 使用固定查表布局，不通过改变此参数自由增加近似精度。应用应先保留默认值；若调整其他方法的分段数，须重新核对其系数、输入范围与计算误差。

`log` 当前只接受 `approx="accurate"`，采用尾数与指数分解。调用者应提供正的有限正规数；零、负数、次正规数和非有限值不能按宿主数学库的特殊值处理方式直接外推。参数中的 `seg_num` 也不改变该方法固定的查表组织。

`softplus` 接受 `"fast"`、`"decomp_w_abs"`、`"decomp_wo_abs"`。默认直接拟合的区间为 $[-10,10]$。

::: {.keep}
两种分解方法利用下列关系：

$$
\operatorname{softplus}(x)
=\operatorname{ReLU}(x)+\log(1+\exp(-|x|)).
$$
:::

其中前一种将绝对值关系包含在查表函数中，后一种显式计算绝对值。分解使用的拟合范围也有限，不能因公式数学上成立就忽略近似表的适用范围。

~~~python
x32 = x.to(tl.float32)
gate = tl.zeus.sigmoid(x32, approx="accurate")
y32 = x32 * gate
y = y32.to(tl.bfloat16)
~~~

这个片段明确表达 SiLU 的乘法和输出转换。其他激活也应先按模型定义明确数学表达式，再选择误差满足要求的近似路径。验证时至少包含实际输入分布、范围边缘和接近零的位置；不要把方法名中的 `accurate` 当作逐位正确舍入保证。

### make_pwlf 与 pwlf：自定义分段线性函数 {#api-pwlf}

~~~python
tl.zeus.make_pwlf(bp, k, b)
tl.zeus.pwlf(x, lut)
~~~

`make_pwlf` 将断点 `bp`、斜率 `k` 和截距 `b` 组成 kernel 内的查表句柄；`pwlf` 对每个输入选择对应分段，并计算该段的线性函数。句柄不是宿主数组，也不是独立设备内存的 owner。

系数必须是 FP32 张量。以一维表示时，`k`、`b` 的长度同为分段数 S，`bp` 的长度至少为 S+1，且 $1\le S\le64$；也可使用等价的 $[1,1,1,L]$ 系数形状。分段数不超过 32 时按一组查表组织，大于 32 时按两组组织。断点必须单调，补齐项也不能破坏查表范围。

下面构造在 $[-1,1)$ 上表示 ReLU 的两段线性函数。额外断点用于满足系数张量长度，示例只声明这一输入区间的行为。

~~~python
bp = tl.arange(0, 4).to(tl.float32) - 1.0
segment = tl.arange(0, 2)
slope = tl.where(segment == 0, 0.0, 1.0)
bias = tl.full((2,), 0.0, tl.float32)
lut = tl.zeus.make_pwlf(bp, slope, bias)
y32 = tl.zeus.pwlf(x32, lut)
~~~

实际应用可以从系数缓冲区装载 `bp/k/b`，并在循环外创建句柄，供多个 tile 复用。输入为整数时先显式转为 FP32；采用 BF16 输入输出时，还要把端点转换误差纳入评估。FP16 形式不作为本书的设备使用基线。

::: {.tip}
查表函数的质量由系数和输入域共同决定。记录拟合区间、断点约定、最大绝对误差和输出转换方式；对分段交界两侧分别取样。增加分段数前先确认当前误差来自拟合，还是来自低精度输入输出。
:::

若误差要求覆盖全域，不应让超出拟合区间的数据直接落入未定义的使用范围。可以约束模型输入、选用具有明确边界处理的组合表达式，或重新准备覆盖目标范围的系数；每种选择都要保持与参考计算相同的数学语义。

### 归一化查表计算 {#api-normalized-lut}

`ztl` 还提供四个显式查表接口，便于在已知输入域上控制系数。它们均返回与输入相同形状的 FP32 值。

~~~python
ztl.positive_normal_reciprocal(x, lut)
ztl.positive_normal_rsqrt(x, lut)
ztl.exp_with_lut(x, lut)
ztl.sigmoid_with_exp_lut(x, lut)
~~~

前两个接口要求输入是**正的、有限的 FP32 正规数**。倒数先把输入写成 $m2^e$，在 $m\in[1,2)$ 上拟合 $1/m$，再恢复指数缩放；倒平方根把指数调整为偶数，在 $m\in[1,4)$ 上拟合 $1/\sqrt m$。它们接收 `make_pwlf` 生成的一组或两组系数句柄。

~~~python
lut = tl.zeus.make_pwlf(bp, slope, bias)
inv = ztl.positive_normal_reciprocal(x32, lut)
~~~

片段中的系数必须拟合归一化后的尾数函数，不能直接拿原始输入区间上的倒数拟合表替代。若一张表只针对 BF16 可表示值验证过，即使调用时转成 FP32，也不能把它的误差结论推广到所有 FP32 输入。

`exp_with_lut` 与 `sigmoid_with_exp_lut` 要求 FP32 输入和两组、共六个系数张量；对应 64 段的 $2^f$ 查表，分数域为 $[0,1)$。第二个接口使用这份指数表组织 Sigmoid，不接受一张直接拟合 Sigmoid 的表。输入范围还须使指数缩放处于所用数值路径的有效域。

::: {.caution}
归一化查表接口不是 IEEE 除法或完整数学库的替代品。零、负数、次正规数及非有限值不属于正正规数接口的输入域；指数路径的上溢、下溢也需单独处理。系数的误差边界由使用方负责建立。
:::

### floor：保留元素类型的向下取整 {#api-zeus-floor}

~~~python
tl.zeus.floor(x)
~~~

接口对每个元素向负无穷取整，保持 shape 和 dtype。它接受 FP32、BF16、FP8 E4M3FN 和有符号 INT8；INT8 已经是整数，按原值返回。FP64、其他 FP8 格式及其他整数类型不在此接口范围内。

~~~python
x32 = x.to(tl.float32)
rounded = tl.zeus.floor(x32)
index = rounded.to(tl.int32)
~~~

`floor` 返回浮点值时，不会隐式变成整数索引。若后续需要 INT32，先确保数值落在可表示范围，再执行类型转换。浮点计算中的零规范化为正零，不承诺保留 NaN payload；本接口也不表示整数的向下取整除法。

## 执行策略提示

### vp_hint 与 dt_hint：标注一个操作 {#api-executor-hint}

~~~python
ztl.vp_hint(value, num_vp="auto")
ztl.dt_hint(value, num_dt="auto")
~~~

这两个接口标注产生 `value` 的操作，分别请求 VP 或 DT 执行策略。返回值与输入 shape、dtype 相同，应让后续消费者使用这个返回值。忽略它而继续使用未标注的原值，可能使提示被消除。

`num_vp`、`num_dt` 必须在编译期确定，合法值为 `"auto"` 或 1 至 4 的整数，不接受布尔值。`"auto"` 让编译器在合法拆分范围内选择；整数表示明确的执行单元数量要求。对无法满足的多执行器精确要求，编译会报错，不会悄悄降为一个执行器。

~~~python
x32 = x.to(tl.float32)
y32 = ztl.vp_hint(x32 + 1.0, num_vp="auto")
tl.store(out_ptr + offsets, y32, mask=valid)
~~~

VP 提示适用于可映射的向量计算。DT 提示应用于转置等数据变换，例如下面把一个逻辑转置分给两个 DT；完整 tile 为 $[64,64]$ BF16，目标块指针描述转置后的布局。

~~~python
x = tl.load(input_block)
y = ztl.dt_hint(tl.trans(x), num_dt=2)
tl.store(output_block, y)
~~~

操作的 dtype、形状、连续方向和依赖决定能否拆分。不能因为一个 BF16 转置可以使用两个 DT，就把相同提示移到任意 gather、归约或尾块上。精确策略失败时，应先核对报错指出的拆分条件，再考虑保持默认或使用 `"auto"`。

::: {.note}
`num_vp=2` 表示请求一个操作使用两个 VP，不表示增加两个程序实例，也不表示每个元素执行两次。提示没有运行时屏障语义，不改变输出归属；多个物理执行器仍共同完成原来的逻辑操作。
:::

### parallel_executor_hint：调度独立工作 {#api-kernel-hint}

~~~python
ztl.parallel_executor_hint()
~~~

该提示作用于整个 kernel，允许编译器寻找可分配给不同执行器的独立操作组或循环。它与把一个 tile 拆给多个 VP/DT 的值提示不同；应用不必把每条表达式都包上一层 `vp_hint`。

调用必须放在 kernel 顶层，不能位于 `for` 或 `if` 内，通常写成第一条语句。下面的两个向量分支具有独立输入和输出；N 在一个静态 tile 范围内，宿主须保证四个缓冲区之间没有破坏独立性的重叠。

~~~python
@triton.jit
def two_outputs(x_ptr, y_ptr, p_ptr, q_ptr, N,
                BLOCK: tl.constexpr):
    ztl.parallel_executor_hint()
    i = tl.arange(0, BLOCK)
    valid = i < N
    x = tl.load(x_ptr + i, valid, other=0.0)
    y = tl.load(y_ptr + i, valid, other=0.0)
    p = x + 1.0
    q = y * 2.0
    tl.store(p_ptr + i, p, valid)
    tl.store(q_ptr + i, q, valid)
~~~

源代码表达了两个可供分析的分支，不承诺它们一定形成并行指令。编译器还会检查访存独立性、别名、工作集和支持的操作组合；没有合适的组时可以保持单执行器，并给出提示未生效的诊断。

这一接口与对单个 kernel 启用 `--enable-parallel-executor` 具有相同的分析范围。它既不创建 Host stream，也不建立核间通信。性能判断应查看实际生成的执行器分配与依赖，再比较相同输入下的总时间，不能按提示条数估算并发度。

### 自动双缓冲的使用边界 {#api-double-buffer}

::: {.caution}
自动 double buffer、`--enable-double-buffer` 和 `ztl.double_buffer_hint()` 均属于实验性能力，目前不完善，暂不推荐使用。不要把启用选项视为正确重叠、资源容量或性能收益的保证。
:::

`ztl.double_buffer_hint()` 是无参数的 kernel 级编译提示，对应自动双缓冲分析；它不是一条设备同步指令，也不直接分配两份用户可控制的缓冲。形式上要求位于 kernel 顶层，但满足这一语法条件并不使任意循环成为可用的流水线。

手写预取与计算流水需要另行证明：预取地址在有效范围内，生产者完成后消费者才读取，最后一次消费完成后缓冲才复用，循环尾部不会多读下一块。代码中出现两个变量、两份 buffer 或交替下标，都不等于最终指令一定重叠。

正式实现应先建立顺序执行且可验证的基线，再按[性能分析与调优](#performance)检查真实的数据依赖和执行时间。不能通过提前复用 buffer、删去必要依赖或扩大未验证输入域来制造表面上的重叠。

## 设备端提前退出 {#api-early-exit}

~~~python
ztl.early_exit_if(flag, nonzero=True)
~~~

该接口在设备侧判断一个单元素 FP32 标志，并按条件跳过后续工作。`nonzero` 是编译期布尔策略：为真时，非零标志触发退出；为假时，零标志触发退出。标志应使用小的非负整数值，推荐只采用 `0.0` 和 `1.0`。

下例由一个实例处理不超过 BLOCK 个元素。宿主先准备 flag，输出缓冲必须具有调用者希望在退出时保留的内容。

~~~python
@triton.jit
def guarded_copy(out_ptr, in_ptr, flag_ptr, N,
                 BLOCK: tl.constexpr):
    flag = tl.load(flag_ptr + tl.arange(0, 1))
    ztl.early_exit_if(flag, nonzero=True)

    i = tl.arange(0, BLOCK)
    valid = i < N
    x = tl.load(in_ptr + i, valid, other=0.0)
    tl.store(out_ptr + i, x, valid)
~~~

`flag_ptr` 指向 FP32 数据。flag 为 1 时，复制部分不执行，输出保留此前内容；flag 为 0 时，执行正常复制。提前退出不会替输出写零，也不会撤销调用前已经完成的工作。

::: {.caution}
不要把任意浮点数当作布尔标志。当前判断路径先转换为整数，`0.5` 不具有“非零必退出”的语义。将标志限定为 `0.0/1.0`，并在其写入完成后再启动消费者。
:::

为了表达整个 kernel 的后续退出，调用应处于顶层、位于需要保护的操作之前。嵌入某个循环体或分支时，其作用只覆盖所在块的后续部分，不能把它理解为 Python 的函数级 `return`。

这是一次已启动 kernel 内的设备判断，不会取消 Host 上的提交，也不会自动跳过其他核的工作。若算法含跨核等待，各核的退出条件必须与通信顺序兼容，避免一核退出而另一核继续等待。与输出初始化、stream 顺序及 owner 生命周期的配合，仍遵循[编译、装载与运行](#compile-and-run)中的规则。
