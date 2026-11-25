# 分页 KV Cache 与 Attention {#attention-cache}

Attention 的实现把矩阵乘、归约、逐元素运算和持续更新的缓存连接在一起。一个步骤出错，往往只在最后的输出中出现偏差；一个步骤布局不合适，也可能让 PE 等待数据。本章从一个具体的 decode GQA 实现出发，依次解释缓存坐标、Host 准备、更新、QK、在线 softmax、PV 和结果合并。

这里分析配套算子库中 `common.gqa` 的普通 BF16 weight 路径。该路径将 LDG 的普通 weight 布局用于存储 K/V，不使用专用 `kcache/vcache` 的页表和格式。代码片段展示已有接口及其算法结构；它们需要相应算子库和设备环境，执行覆盖范围见随书验证说明。框架中的其他 attention 路径应保留各自布局契约，不能仅按名称互换。

## 先固定一次 decode 的数学定义

### GQA 的 head 对应关系

一次 decode 为每个请求增加一个 token，查询为 `Q[B,Hq,D]`，新 K/V 为 `[B,Hkv,D]`。GQA 让多个 query head 共享一个 KV head。若 $H_q$ 能被 $H_{kv}$ 整除，组大小为 $G=H_q/H_{kv}$，query head h 使用的 KV head 为 $\lfloor h/G\rfloor$。

本章取 `B=1,Hq=16,Hkv=2,D=256`。query head 0–7 使用 KV head 0，head 8–15 使用 KV head 1；两个核各拥有一个 KV head 及对应的八个 query head。这个分工让两核读取各自的 LDG 池，并写不同 head 的输出，在该 kernel 内不需要合并两核的 head 结果。

对一个 query head，设当前可见序列长度为 S，数学定义为

$$
s_j=\frac{q\cdot k_j}{\sqrt D},\qquad
p_j=\frac{\exp(s_j)}{\sum_{t=0}^{S-1}\exp(s_t)},\qquad
o=\sum_{j=0}^{S-1}p_jv_j.
$$

缓存中的 K 应与 Q 使用配套的位置编码和归一化约定。如果上游已经完成 RoPE，写入缓存的就是该结果，读取端不应再旋转一次。缩放因子本例为 $1/\sqrt{256}=1/16$，应在 QK score 上使用一次。重复缩放或把未处理的 K 写入已有缓存，属于输入契约错误，softmax 不能修复。

### Decode 的可见长度

新 token 先写入缓存，随后 attention 读取包含它的有效序列。若历史长度为 389，本次新增 token 的逻辑位置是 389，更新后的有效长度是 390。把长度仍传为 389，会忽略当前 token；先把长度增加到 390，却让 attention 在写入完成前读取，也可能读到旧值。

单 query decode 的因果性可以通过“只读取当前已经生成的 S 个 token”表达。prefill 同时处理多个 query 时，每个 query 的可见前缀不同，必须使用 query 位置对应的因果条件。本章的 decode 模板没有自动包含这一扩展，不能直接将 B 维换成 query 序列维就得到正确的 prefill attention。

## 从逻辑 token 到物理页

### 三个量分别承担什么职责

设页长 P=256，池中有四个物理页，因此每个 KV head 可容纳 `R=4*256=1024` 个 token slot。请求使用的页表为 `[3,1]`：逻辑页 0 存放于物理页 3，逻辑页 1 存放于物理页 1。页表元素是页号，不是字节地址，也不是 LDG 基址。

这里的“页”是缓存管理的一组 P 个 token slot，与 LDG 的 16 KiB miniblock 不同。地址转换分两步：先由请求页表得到池中的 slot，再按 K/V 矩阵布局把 `(slot,d)` 转成字节位置。尤其是 V，一个 token 页在 LDG 中可能跨多个行块；页号不表示一段可以按普通行主序读取的连续字节。

对逻辑 token t，其逻辑页号、页内偏移和物理 slot 为

$$
\ell=\lfloor t/P\rfloor,\qquad
u=t\bmod P,\qquad
\mathrm{slot}=\mathrm{block\_table}[\ell]P+u.
$$

| 逻辑 token t | 逻辑页 | 物理页 | 页内偏移 | 物理 slot |
|---:|---:|---:|---:|---:|
| 0 | 0 | 3 | 0 | 768 |
| 255 | 0 | 3 | 255 | 1023 |
| 256 | 1 | 1 | 0 | 256 |
| 389 | 1 | 1 | 133 | 389 |

Table: 页表 `[3,1]` 下的 token 地址转换

![逻辑序列通过页表访问不连续的物理页](assets/paged-attention.svg){width=100%}

最后一行的逻辑位置和物理 slot 恰好相等，只是这组页表的巧合。不能据此省略页表。第 0 个 token 对应 slot 768，已足以发现把逻辑位置直接当作物理位置的错误。

`slot_idx[B]` 告诉更新 kernel 把本次 token 写到哪里；`block_table[B,max_pages]` 告诉读取 kernel 按什么顺序找到历史数据；`seq_lens[B]` 告诉读取 kernel 有多少 token 参与计算。三个张量应属于同一次调度结果。仅更新 slot、不更新页表或长度，会让写入与读取指向不同的序列。

### 跨页块必须拆开

读取 kernel 使用 `BLOCK_S=256`，要求页长是该块宽的整数倍。第 s 块的逻辑起点为 `g0=s*BLOCK_S`，先查询 `g0//P` 对应的物理页，再从 `physical_page*P + g0%P` 开始读一个规则块。由于块不跨页，一次页表查询就足够。

若把 `BLOCK_S` 改为 384 而页长仍为 256，第一块会跨越逻辑页 0 和 1；这两页分别位于物理页 3 和 1，地址并不连续。仅根据第一个页号连续读取 384 个 token 会读错后半段。应保持整除约束，或重新设计能分段读取跨页数据的算法。

本例 `S=390` 时共两块，第一块 256 个 token 全部有效，第二块只有 134 个有效 token。物理池中其余 slot 可能保存别的请求或旧数据。尾部 mask 必须根据逻辑有效长度计算，而不能依据池容量判断所有已分配 slot 都有效。

## 普通 weight 布局如何保存 K 和 V

### K 的行与 V 的列

每个核的 K 池逻辑形状为 `[R,D]`，一行对应一个物理 token slot；V 池逻辑形状为 `[D,R]`，一列对应一个物理 slot。Host 可以用同样的 token-major `[R,D]` 张量描述 K 和 V，但打包 V 前需要转置，变成 dim-major 的 `[D,R]`。

这两个方向服务于不同的 dot：Q 形状为 `[H,D]`，读入 K tile `[S_b,D]` 后计算 `Q @ K.T`，得到 score `[H,S_b]`；概率 P 形状为 `[H,S_b]`，读入 V tile `[D,S_b]` 后计算 `P @ V.T`，得到输出部分和 `[H,D]`。两次都使用普通 `tl.dot`，由编译器选择 PE 模式。

创建两个池时，`kind` 均为 `"weight"`；kernel 读写的 `memory_type` 同样使用 `"weight"`。本章 V 的准备过程是“逻辑转置，再按普通 weight 排列”。专用 `vcache` 还涉及其独立的物理列组交错与索引格式，不能用这次转置概括；两条路径的对照见 [LDG 数据布局与访问](#ldg-memory)。完整池形状、dtype、patch 参数、核所属关系都是格式的一部分，K/V 的数学含义相同并不保证存储对象可互换。

### 展开 BF16 的物理坐标

在本章 `Tr=Tc=1` 的普通 weight 布局中，一个物理块是 `128×64` 个 BF16 元素，即 16 KiB。块按逻辑行块、列块的顺序排列，块内部行主序。设矩阵列数为 C，元素 `(r,c)` 的物理块编号为

$$
b=\left\lfloor\frac r{128}\right\rfloor
\left\lceil\frac C{64}\right\rceil
+\left\lfloor\frac c{64}\right\rfloor.
$$

相对于该核池基址的 BF16 元素偏移为

$$
e=8192b+64(r\bmod128)+(c\bmod64),
\qquad \mathrm{byte\_offset}=2e.
$$

该公式只用于本章普通 BF16 weight、`Tr=Tc=1` 的布局。更改 dtype、patch 几何或缓存种类后，应重新使用对应格式公式。

写入物理 slot 389 时，K 的行块为 3、块内行为 5。D=256 分成四个 64 元素行段，因此 K 写入物理块 12、13、14、15 的第 5 行。它们不是四段首尾相接的普通连续数组，中间相隔一个物理块。

V 的逻辑形状是 `[256,1024]`。slot 389 位于列块 6、块内列 5，D 方向有两个 128 行块，因此 V 写入物理块 6 和 22 的第 5 列。两个列段分别保存 `d=0..127` 和 `d=128..255`。

![同一个 token 的 K 行段与 V 列段写入坐标](assets/attention-slots.svg){width=100%}

取这个 token 的维度索引 d=130，可以把两次写入完整算到字节。下面的偏移分别相对当前核的 K 池或 V 池基址：

| 输入元素 | 池内逻辑坐标 | 物理块 / 块内坐标 | 字节偏移 |
|---|---|---|---:|
| K 的 d=130 | `(389,130)` | 14 / `(5,2)` | 230020 |
| V 的 d=130 | `(130,389)` | 22 / `(2,5)` | 360714 |

Table: slot 389 的两个缓存目标

例如 K 的偏移为 $14\times16384+5\times128+2\times2$；V 为 $22\times16384+2\times128+5\times2$。两者从同样长度的输入向量出发，目标块、块内方向和字节位置都不同。

只看 `D=256` 个 BF16 值，K 和 V 的输入大小相同；看物理目标时，K 是四个连续行段，V 是两个带跨度的列段。把 K 的写入方法复制给 V，会把同一份值写到错误坐标，即使没有越过池的总范围也会破坏缓存。

### 逻辑读取与物理写入的描述符

读取时，weight 路径使用完整逻辑矩阵描述符。K 的 shape 始终是 `(R,D)`，V 始终是 `(D,R)`；当前请求长度 S 只决定循环次数和尾部有效域。若把 V 的 shape 改成 `(D,S)`，列块数可能改变，后续行块的地址也随之改变。

本例 V 的完整列数是 1024，每个行块对应 16 个列块。若误用有效长度 390，就只剩 $\lceil390/64\rceil=7$ 个列块；上表中的 `(130,389)` 会被解释成物理块 13、偏移 213258 B，而正确位置是物理块 22。这个错误可能仍在池内，却读到了另一处数据。因此，池容量决定布局，序列长度决定有效域。

更新 kernel 则把池描述为物理 `[num_blocks,128,64]` 数组。K 每次写一个 `[1,1,64]` 行段，V 每次写一个 `[1,128,1]` 列段。所有偏移通过描述符的 `tl.advance` 表达，保持池基址不变。这一写法与直接对 weight base 做普通线性指针算术不同，应沿用配套路径的实现方式。

以下片段展示 K 的物理目标，假定 `CB_K=D//64`，`pool_row_blocks=R//128`，输入 `k_segment` 对应一个 64 元素行段。

```python
k_target = tl.make_block_ptr(
    base=k_pool,
    shape=(pool_row_blocks * CB_K, 128, 64),
    strides=(8192, 64, 1), offsets=(0, 0, 0),
    block_shape=(1, 1, 64), order=(2, 1, 0),
)
block = (slot // 128) * CB_K + column_block
position = (block, slot % 128, 0)
tl.store(tl.advance(k_target, position), k_segment,
         boundary_check=(0, 1, 2), memory_type="weight")
```

V 的目标使用 `pool_col_blocks=R//64`；`row_block` 沿 D 的 128 行块递增。

```python
v_target = tl.make_block_ptr(
    base=v_pool,
    shape=((D // 128) * pool_col_blocks, 128, 64),
    strides=(8192, 64, 1), offsets=(0, 0, 0),
    block_shape=(1, 128, 1), order=(2, 1, 0),
)
block = row_block * pool_col_blocks + slot // 64
position = (block, 0, slot % 64)
tl.store(tl.advance(v_target, position), v_segment,
         boundary_check=(0, 1, 2), memory_type="weight")
```

这些片段说明目标坐标。完整实现还包括每核 LDG 基址读取、按 batch 遍历，以及从 `[B,Hkv,D]` 中选择本核 head 的输入段。把一次大范围散写拆成一个物理块内的规则写入，也能降低地址表达与资源分配的复杂度。

## Host 准备与缓存生命周期

### 先检查接口条件

本章选用的封装要求 `Hkv=CORE_NUM=2`；D 为 128 的倍数；page size 为 256 的倍数，以满足读取的 `BLOCK_S=256`。Q/K/V 是连续 BF16 张量，页表、slot 和序列长度使用连续 int32 张量。每个核的 query head 数还需要由封装选定的完整 head tile 覆盖。

除了封装中的类型检查，调用者还要验证：所有物理页号位于 `[0,num_pages)`；所有 slot 位于 `[0,R)`；有效长度不大于页表可表达的 token 数；同一批新增 token 的目标 slot 不发生非法重叠。边界检查可以保护某次搬运范围，不会替调用者验证“两个请求误用了同一物理页”的业务所有权。

本例选择两核是该 GQA 封装的 head 分工契约。不要在不修改 head 映射、池构造和输出分区的情况下，把 `CORE_NUM` 改成其他值。空 batch、空序列和非本分片 token 也不同：空 batch 可以不启动；空序列需要规定零输出；不归本分片拥有的 token 如采用 dummy slot，必须分配并隔离该 slot，不能把 -1 直接交给物理缓存写入。

### 通过已有 helper 创建配套池

以下调用片段使用算子库提供的 helper，演示如何保留 Host 逻辑数据并生成配套的 K/V 池。随机数在 CPU 生成，输入本身先转换为 BF16；后续数学参考使用同一份数据。

```python
import torch
import torch_zeus
from sgl_kernel_zeus_triton.common.gqa import (
    sgl_gqa_flash_attn_partial_triton as fa,
    sgl_gqa_kv_cache_store_triton as cache,
)

torch.manual_seed(7)
k_host = torch.randn(2, 1024, 256).to(torch.bfloat16)
v_host = torch.randn(2, 1024, 256).to(torch.bfloat16)
pools = fa.fill_gqa_kv_pools(
    4, 256, 256, k_host, v_host,
)
q_host = torch.randn(1, 16, 256).to(torch.bfloat16)
q = q_host.to("zeus")
```

`fill_gqa_kv_pools` 接受 K/V 都为 token-major 的 CPU 数据，内部把 V 转置并按相同 weight 格式打包。返回对象包含核对应的池描述符及保持其存储存活的对象。调用者应保留整个池 owner，直到所有引用它的执行完成；不能只取一个裸地址，然后丢弃拥有实际分配的对象。

继续准备一次更新。此片段不负责框架页分配，它使用本章已经确定的页表和 slot。

```python
k_new_host = torch.randn(1, 2, 256).to(torch.bfloat16)
v_new_host = torch.randn(1, 2, 256).to(torch.bfloat16)
slot = torch.tensor([389], dtype=torch.int32, device="zeus")
table = torch.tensor([[3, 1]], dtype=torch.int32,
                     device="zeus")
lengths = torch.tensor([390], dtype=torch.int32,
                       device="zeus")
k_new = k_new_host.to("zeus")
v_new = v_new_host.to("zeus")
cache.gqa_kv_cache_store(
    k_new, v_new,
    pools["k_pool"], pools["v_pool"], slot, 4, 256,
)
partial, lse = fa.gqa_flash_attn_partial(
    q, pools["k_pool"], pools["v_pool"], table, lengths,
    1.0 / 16.0, 4, 256,
)
torch.zeus.synchronize()
```

更新和读取要由配套运行时维持正确的执行顺序；若把它们放到不同执行流，需要建立明确的依赖，不能仅依靠 Python 源码先后。上例在取结果之前同步，Host 参考则相应更新 `k_host[:,389,:]` 和 `v_host[:,389,:]` 后再计算。

这个兼容 helper 以未包装的核地址对象提交池参数，不能据此假定运行时已自动把它标为可写。若此前通过 `.cpu()` 或 `to_gdg()` 建立过 LocalMem 的线性视图缓存，调试读回前应在写入完成后明确使缓存失效。

```python
torch.zeus.synchronize()
pools["k_pool"].invalidate_cache()
pools["v_pool"].invalidate_cache()
```

`invalidate_cache()` 使下一次线性化重新读取已更新的 LDG，双核池会向各核分配传播失效。它不替代同步；同步也不自动等同于失效。直接由 PE 从 LDG 消费 K/V 不经过该 Host 线性视图，但调试时读到旧副本会误导正确性判断。新封装若使用显式资源接口，应按[资源契约](#argument-resources)准确声明读写，并保持所需地址表 ABI。

### Graph 中更新内容而不是悄悄替换地址

模型通常在每一步改变 K/V、slot、页表与长度，同时复用已捕获的执行图。图中的参数可能绑定稳定地址，重新创建一个同形状张量并赋给同名 Python 变量，不会自动更换已捕获的指针。应在捕获前准备长期存活的输入和元数据缓冲，replay 时按支持的路径更新内容。

另一个容易忽略的来源是 QKV 投影后的切片。`split` 得到的 K/V 可能是非连续视图，而缓存更新 kernel 要求连续数据。框架应为需要的转换准备由缓存池持有的稳定缓冲，不能在捕获过程中创建临时连续副本、把地址交给后续 kernel 后立即释放。数据刷新本身也必须被捕获，不能由图外 CPU fallback 完成一次后就一直使用旧值。

这些要求不只影响内存安全，也影响结果：eager 模式每次重新准备数据可能正确，Graph replay 却一直写入捕获时的 K/V。定位时检查多步更新后的缓存内容，不能只比较第一次执行或只确认指针地址没有变化。

## QK、在线 softmax 与 PV {#attention-online}

### 一个序列块包含哪些张量

每核处理八个 query head，取 `BLOCK_H=8,BLOCK_S=256,D=256`。Q tile 为 `[8,256]` BF16；K tile 的逻辑读取形状为 `[256,256]` BF16；score 为 `[8,256]` FP32；V tile 的逻辑读取形状为 `[256,256]` BF16；输出累加器为 `[8,256]` FP32。

K 和 V 来自 LDG，其逻辑读取块分别是 128 KiB，不能直接把这两项相加当作 L2 Buffer 分配。L2 Buffer 的主要状态包括 Q 的 4 KiB、score 的 8 KiB、FP32 概率的 8 KiB、输出累加器的 8 KiB，以及每 head 的统计量；实际是否还有 BF16 概率副本、转换和布局临时量，查看生成的分配结果。增大 `BLOCK_S` 会同时增大 score/概率及每轮 K/V 读取量，增大 `BLOCK_H` 则增加 Q、统计量和输出状态。

每个序列块依次形成 QK score、更新 softmax 统计量、执行 PV 并更新输出状态。完整的 `[H,S]` score 或 probability 不写回外存，循环中只保留当前块与跨块状态。这样既限制了中间存储规模，也让序列长度增长时不必分配一个同长度的完整概率矩阵。

### 为什么旧状态需要重新缩放

::: {.keep}
维护三个 FP32 状态：每 head 的最大值 m、指数和 l、未归一化的加权和 O。初始在数学上取 $m=-\infty,l=0,O=0$。新块的 score 为 S，取该块与历史共同的最大值：

$$
m'=\max(m,\max_j S_j),\qquad
\alpha=\exp(m-m'),\qquad
P_j=\exp(S_j-m').
$$

然后更新

$$
l'=\alpha l+\sum_jP_j,\qquad
O'=\alpha O+PV,\qquad m\leftarrow m'.
$$
:::

所有块结束后输出 $O/l$，同时可输出 `LSE=m+log(l)`。旧的 l 与 O 原来采用 m 为指数基准，必须乘以相同的 $\alpha$，才能与新基准下的 P 合并。只缩放 l、不缩放 O，或者直接相加两个块的独立 softmax 输出，都会改变权重。

![在线 softmax 的最大值、分母与输出状态共同更新](assets/online-softmax.svg){width=100%}

用两个标量 token 手算可以验证递推。第一块 score 为 0、value 为 1，状态是 `m=0,l=1,O=1`。第二块 score 为 `log(2)`、value 为 3，新最大值为 `log(2)`，旧状态缩放因子为 1/2，新概率未归一化值为 1。于是 `l=1.5,O=3.5`，最终输出 `7/3`，与完整 softmax 权重 `1/3,2/3` 一致。这个小算例仅用于数学推导，不是设备合法 tile 的规格。

### 实现中的精度边界

配套 BF16 GQA 路径的 Q/K/V 为 BF16，QK 和 PV 使用 FP32 累加；m、l、O 保留 FP32。为了执行 BF16 的 PV，当前块概率在进入 dot 前转换为 BF16，l 的更新仍使用 FP32 概率。最后输出 partial O 转成 BF16，LSE 保持 FP32。因此该实现与“所有概率及乘积均使用 FP32”的数学参考存在可解释的精度差异。

以下片段概括非空有效块的计算顺序。`score` 已经包含缩放与有效域处理，`v_tile` 按 `[D,BLOCK_S]` 从 LDG 读取；单例列维度始终保留。

```python
row_max = tl.max(score, axis=1, keep_dims=True)
m_new = tl.maximum(m_state, row_max)
alpha = tl.exp(m_state - m_new)
p = tl.exp(score - m_new)
l_state = alpha * l_state + tl.sum(
    p, axis=1, keep_dims=True,
)
o_state = alpha * o_state + tl.dot(
    p.to(tl.bfloat16), tl.trans(v_tile),
    out_dtype=tl.float32,
)
m_state = m_new
```

完整配套实现还把概率计算为 `max(exp(score-m_new)-1e-30,0)`，使足够小的指数结果归零；它没有把概率抬高到一个正下限。初始 m 使用 `-1e9`，无效 score 使用 `-2e9`，最终分母取 `max(l,1)`。这些有限保护属于具体模板的数值契约，不能省略后声称得到与源码相同的异常值行为。

评估精度时，先用完整 FP32 数学参考检查目标，再用包含上述处理、概率转换、相同分块和最终 BF16 转换的参考解释实现差异。若某次调优改变 `BLOCK_S`，也会改变概率转换与跨块累加的过程，需要重新比较。

### 尾块 mask 和空序列

本例第二个块从逻辑位置 256 开始，有效长度 390，因此块内 `j<134` 才参与 softmax。普通数学 mask 可以在 score 上排除其余位置；配套实现采用有限大负值，并以相应指数范围处理让无效位置贡献为零。无论采用哪种表达，都必须检查“所有有效 score 都是负数”的输入，以发现补零参与最大值或分母的问题。

该源码路径的有限 score 哨兵依赖输入范围，不能把它描述成对任意 FP32、NaN 或 Inf 都等价于负无穷。非空行的最大有效 score 应不低于初始 m 哨兵；否则指数和可能小于一，最终分母保护会改变归一化结果。特别是有 Inf 时，先乘零再加哨兵并不能得到合法的有限 mask，因为 `Inf*0` 会产生 NaN。接口应限定并检查正常推理范围，特殊值另行定义和测试。

当 `seq_len=0` 时，序列循环不执行。配套路径保持 O=0，使用受保护分母输出零，并返回约定的有限 LSE 哨兵。消费者必须认识这一空分片约定，不能把它当作一个真实、概率为一的 token。数学公式中的 `-inf` 初始化与设备使用的有限哨兵是两个层次，不应混用而遗漏空状态处理。

## 分片结果如何合并

### 归一化输出必须配合 LSE

如果不同 context-parallel 分片各自处理一段 token，分片 c 返回已经归一化的 $o_c$ 和 $L_c=\log\sum_j\exp(s_{c,j})$。直接对 $o_c$ 求平均不正确，因为不同分片的概率总质量不相同。令 $L_{\max}=\max_cL_c$，则

$$
w_c=\exp(L_c-L_{\max}),\qquad
o=\frac{\sum_cw_co_c}{\sum_cw_c}.
$$

这个公式解释了为什么 partial 输出和 LSE 必须成对传递，也解释了为什么一个分片的序列长度改变后，需要重新计算它的 LSE。通信必须保持 batch、head、分片顺序一致；把正确的输出与另一个请求的 LSE 配对，会得到看似平滑但错误的结果。

源实现为避免极端指数范围还会采用有限截断，其结果应按对应数值契约检查。若所有分片为空，先采用接口定义的空输出，不把一般非空公式直接用于 `0/0`。若仅一个分片非空，合并结果应接近该分片的结果，可用作定向检查。

### 输出投影的分工改变了同步需求

attention 的输出按 head 拼接后，逻辑宽度为 `Hq*D`。若输出投影 `O @ W_o.T` 按输出列分核，每个核可以读取全部 head 的 partial 结果，在本核完成完整收缩维度，并写独立输出列；无需再合并两核的部分积。

若沿 head 切分输出投影，切开的恰好是收缩维度，每核只得到一个部分和，需要后续合并。前一种方式可能重复少量 softmax 结果合并工作，后一种方式需要额外的部分和与同步。选择取决于模型形状、权重所在核和数据规模，不能仅因 attention 本身按 head 分核，就强制输出投影也按相同方式划分。

## 从缓存到输出逐步验证

### 先验证更新不破坏邻居

先让缓存使用可辨认的有限值，选一个 slot 更新 K/V，然后读回目标及其前后邻居。K 要检查 64 列段和 128 行块边界，V 要检查 64 token 列块边界。对本章布局，优先选择 slot `0,63,64,127,128,255,256,389,1023`，分别覆盖块内、换块和池末尾；这些是测试建议，执行覆盖以验证记录为准。

单次写入通过后，再连续更新两个相邻 slot，检查后一次没有破坏前一次。随后在同一物理页内改变请求的有效长度，确保 attention 只读取规定前缀。最后再把页表换为非连续映射，例如 `[3,1]`。始终使用 `[0,1]` 的连续页表，无法暴露逻辑 token 与物理 slot 混用的问题。

### 建立不依赖物理打包的 Host 参考

Host 参考保留 `[Hkv,R,D]` 的逻辑 K/V，按页表生成 slot 序列，再逐 head 计算。下面先把 slot 389 替换成新值，再对应本章一次更新后的请求求参考。

```python
k_host[:, 389, :] = k_new_host[0]
v_host[:, 389, :] = v_new_host[0]
slots = torch.cat((torch.arange(768, 1024),
                   torch.arange(256, 390)))
reference = torch.empty(16, 256, dtype=torch.float32)
for head in range(16):
    kv_head = head // 8
    keys = k_host[kv_head].index_select(0, slots).float()
    values = v_host[kv_head].index_select(0, slots).float()
    query = q_host[0, head].float()
    scores = (query @ keys.T) / 16.0
    probability = torch.softmax(scores, dim=-1)
    reference[head] = probability @ values
```

这份参考不使用 LDG 物理公式，因而能独立检查页表、head 映射和数学目标。另写一份按 `BLOCK_S=256` 递推的参考，加入概率 BF16 转换，便于定位实现精度。不要让 Host 参考复制同一套错误的物理写入公式，再据此宣布缓存正确。

### 按症状缩小到一个阶段

| 现象 | 首要检查 | 定向用例 |
|---|---|---|
| 一页正确，跨页后错 | 页号与 slot 换算 | 非连续页表 `[3,1]` |
| K 正确、V 错位 | V 列写入及转置布局 | 单 slot、单维度标记 |
| 整块正确，尾块偏差大 | score 有效域 | S=255、256、257、390 |
| 增加新块后历史贡献错 | m 改变时的双重缩放 | 后一块出现更大 score |
| head 0 正确，其余 head 异常 | head 所属核与广播维度 | 每 head 使用不同 Q |
| eager 正确，replay 重复旧结果 | 稳定地址及数据刷新 | 连续两步不同 K/V |

Table: Attention 的分阶段诊断

性能分析也沿相同分段进行。先确认页表查询是否过于频繁、序列块是否规则；再看 K/V 的 LDG 读取是否延迟 PE 计算；随后看 VP 的 max/exp/sum 及状态缩放是否位于关键路径。增大序列块会减少循环，但增加 score/概率的 L2 Buffer 占用与每轮读取量；增大 head 块能复用 K/V，却增加 Q 和输出累加器。用真实分配和时间线选择平衡点，不能从一张完整 attention 流程图直接推断最优配置。
