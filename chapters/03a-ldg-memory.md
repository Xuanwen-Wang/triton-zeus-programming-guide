# LDG 数据布局与访问 {#ldg-memory}

LDG 是 Zeus 的 3D DRAM 近存数据空间，主要承载矩阵权重与 KV Cache。选择 LDG 的编程方式时，需要先确定数据用途及布局，再准备相配套的物理内容、地址和读取描述符。本章先介绍布局类别与共同的 miniblock 粒度，随后分别展开 Weight 与分页 KV Cache。

## LDG 用途与布局概览 {#ldg-layout-overview}

### 先区分 Weight 与 KV Cache

Weight 通常在算子运行前准备，随后由多个 kernel 或多个 token 的计算复用。KV Cache 则随请求增长，需要在运行中追加、读取和回收。两者都位于 LDG，但其页组织、更新方式和读取参数不同。

Weight 主要采用三种块组织：Dense 对完整矩阵统一排列；Blocked 先切出独立计算块，再将各块紧密拼接；带 padding 的 Blocked 为这些块预留指定的对齐空间。分页 KV Cache 通过索引将逻辑连续的 token 映射到可以离散放置的物理页，各页内部仍有确定的 K/V 排列。

![LDG 布局概览：Weight 的三种块组织与 KV Cache 的分页组织](assets/ldg-layout-overview.svg){width=100%}

| 数据用途与布局 | 物理组织 | 编程时首先确定 |
|---|---|---|
| Weight：Dense | 整个矩阵统一 pack，miniblock 连续排列 | 完整矩阵形状与 tile 起点 |
| Weight：Blocked 紧排 | 独立块分别 pack，块间紧密相接 | 块形状、遍历顺序与块基址 |
| Weight：Blocked 带 padding | packed block 放入带空隙的对齐空间 | 有效载荷、槽位容量与地址步长 |
| KV Cache：软件分页 | 普通 Weight 组织 K 行与 V 列，上层页表定位 slot | 完整池形状、页表和有效 token 范围 |
| KV Cache：专用分页 | 专用索引定位 payload 页，页内采用配套 K/V 布局 | 专用页长、索引编码与 K/V 格式 |

Table: LDG 数据用途与布局选择

### 带 scale 的 Weight 与分页 KV 的边界

UScale 是配套量化 Weight 的系数资源。在支持的组合中，权重及其 scale 按约定绑定；scale 可以与权重相邻，也可以单独存放。它作为 Weight 布局的变体在本章后文说明，具体量化计算见[量化矩阵乘与 UScale](#quantized-matmul)。

::: {.caution}
专用 `memory_type="kcache"` 和 `"vcache"` 不支持 UScale 反量化参数，不能传入 `weight_dtype`、`ugroup` 或 `uscale`。它们必须指定与物理布局一致的 `page_size`；这些参数不能从 Weight 示例直接移植。
:::

本章的“离散”描述页与页之间的位置。一次页内读取仍须遵守 miniblock 和 K/V 布局，不能把一个页内的元素任意分散。普通 Weight 形式承载 K/V 是另一种已有实现方式，后文会与专用分页路径对照说明。

## miniblock 与对齐 {#ldg-miniblocks}

### 八路 WL1 共同使用的读取粒度

矩阵操作数从 LDG 经八路 WL1 进入 PE。与这一路径配套的物理基本单元称为 **miniblock**：一个完整 miniblock 为 16 KiB，按八路划分为八个 2 KiB 片段。每路 WL1 负责其中对应的一片，八片共同组成矩阵读取所需的数据。

因此，权重准备至少要形成完整 miniblock 所要求的物理组织。一个计算 tile 可以包含多个 miniblock；其有效元素也可以少于分配的存储范围，此时要同时处理布局补齐和计算边界。2 KiB 是每路在该 miniblock 中的数据份额，不是 WL1 的总容量或每拍吞吐；八个片段也不是八个可独立调度的 program。

![一个 16 KiB miniblock 的八路划分及其与 WL1 的对应关系](assets/ldg-miniblock-wl1.svg){width=100%}

::: {.note}
16 KiB 是这里的矩阵布局粒度，不表示每一次 LD/ST 都必须搬运 16 KiB。LDG 的矩阵读取与 L2 Buffer 读写使用不同路径，带宽与传输口径见[Zeus V3 的容量、算力、吞吐与带宽](#hardware-throughput)。
:::

### 字节粒度相同，逻辑元素数不同

一个 miniblock 能容纳多少矩阵元素，取决于其编码。采用存储视图 W[N,K] 时，普通格式的逻辑覆盖为：

| 权重格式 | 一个 miniblock 的逻辑范围 | 物理字节量 |
|---|---|---:|
| BF16 | 128 × 64 个元素，每元素 2 B | 16 KiB |
| INT8 / FP8 | 128 × 128 个元素，每元素 1 B | 16 KiB |
| INT4 | 256 × 128 个元素，每两个编码共用 1 B | 16 KiB |

Table: 不同数据类型对应的 miniblock 逻辑范围

后文先用 BF16 推导坐标与字节位置，再介绍量化 Weight 的系数页。INT4 的四位配对方向另有规则，不能直接把 BF16 地址除以四；专用 V Cache 也有自己的页内排列。

### 分开理解三种补齐和对齐

| 要求 | 处理的对象 | 对程序的影响 |
|---|---|---|
| 矩阵尾部补齐 | 未填满 miniblock 的逻辑边缘 | 增加分配量，计算仍使用真实 N/K |
| Blocked 块间 padding | packed block 所在的对齐空间 | 改变块基址步长和驻留容量 |
| 分配基址对齐 | 整份 LDG 分配的起始地址 | 限定起点，不自动改变内部排列 |

Table: 尾部补齐、块间 padding 与基址对齐

Dense 和紧排 Blocked 都可能需要矩阵尾部补齐。“紧排”仅表示块间没有额外空隙。Host pack、kernel 的地址推进与读取描述符必须使用同一套规则，否则第一块正确也不能保证后续块正确。

## Weight 布局 {#ldg-block-organization}

下面从同一组矩阵元素出发，先说明块内排列，再比较三种块组织。Weight 的准备与写入过程见[数据准备、更新与读取](#ldg-preparation)。

### 逻辑矩阵与共同的块内组织 {#ldg-weight-coordinates}


本书使用 $W[N,K]=B[K,N]^T$ 作为权重准备与 LDG 描述符的逻辑视图，因此

$$
C[m,n]=\sum_{k=0}^{K-1}A[m,k]W[n,k].
$$

数学 B 的列 n，对应 W 的行 n；收缩维 K 在 B 中是行，在 W 中是列。后续公式中，n 始终表示输出通道，k 始终表示收缩维位置，不随画图方向改变。

对于 `nn.Linear`，传入的权重本来就是 `[N,K]`，直接使用即可。若应用最初持有数学方向的 B，先显式取得 `W = B.T.contiguous()`。当前权重 pack 接口保持所传矩阵的行列含义，不会自动把一个 `[K,N]` 输入识别成需要转置的权重。

```python
# B is the mathematical right operand, shape [K, N].
B = torch.arange(128 * 256).reshape(128, 256)
W = B.T.contiguous()
assert W.shape == (256, 128)
assert W[145, 70] == B[70, 145]
```

这段代码只演示坐标对应。实际 BF16 输入应按所需数值范围生成；不要用 BF16 保存上述所有整数编号，否则较大的相邻编号可能被舍入为相同值。随书布局检查使用整数标签跟踪元素，不把标签当作 BF16 数值精度测试。

#### 一个 BF16 miniblock 内的八个片段

在 W 视图中，将一个 $128\times64$ miniblock 的 128 行再分成 8 组，每组 16 行。每个 $16\times64$ BF16 片段占 2 KiB；片段内按行存放，每行 64 个 BF16，占 128 字节。这 8 个片段对应八路 WL1 的数据组织。

令 $\ell=\lfloor(n\bmod128)/16\rfloor$ 表示片段编号，$r=n\bmod16$ 表示片段内行号，$c=k\bmod64$ 表示行内元素号。一个元素在 miniblock 内的字节偏移为

$$
o_{\mathrm{mini}}=\ell\times2048+r\times128+c\times2.
$$

这里的 16 KiB miniblock 与 2 KiB 片段不是同一级“块”。前者是完整矩阵读取的物理基本单元，后者解释它在八路中的分布。把 2 KiB 片段误当成一个完整 miniblock，会使块号、字节量与地址推进全部相差 8 倍。

### Dense：完整矩阵统一排列 {#ldg-dense}

Dense 将完整 W[N,K] 按 miniblock 顺序存放。kernel 保持同一个矩阵基址与完整 shape，在循环中更新 tile 起点。`BLOCK_N`、`BLOCK_K` 是一次计算的范围，改变它们不会重新排列已经写入 LDG 的矩阵。

#### 从一个 K×N 矩阵划分 miniblock

普通 BF16 布局的一个 miniblock 为 16 KiB。它覆盖 W 的 128 行、64 列，即 $128\times64\times2$ 字节；回到数学 B 视图，同一块覆盖 64 行、128 列。两个 shape 描述同一组元素，字节量相同。

对 $B[K=128,N=256]$，沿 K 切成两个 64 行区间，沿 N 切成两个 128 列区间，得到四个 miniblock。定义块坐标

$$
b_N=\lfloor n/128\rfloor,\qquad
b_K=\lfloor k/64\rfloor.
$$

采用普通 Dense 排列时，固定一个 $b_N$，先依次放完该 N 分组的全部 $b_K$，再进入下一个 N 分组。由于本例 K 方向有两块，物理块编号为 $j=2b_N+b_K$。

| 物理块 j | 数学 B 的范围 | 存储 W 的范围 | 字节区间 |
|---:|---|---|---|
| 0 | K 0–63，N 0–127 | N 0–127，K 0–63 | [0,16384) |
| 1 | K 64–127，N 0–127 | N 0–127，K 64–127 | [16384,32768) |
| 2 | K 0–63，N 128–255 | N 128–255，K 0–63 | [32768,49152) |
| 3 | K 64–127，N 128–255 | N 128–255，K 64–127 | [49152,65536) |

Table: B[128,256] 的四个 BF16 miniblock

![从数学 B[K,N] 到存储 W[N,K]：四个 miniblock 的覆盖范围与物理顺序](assets/ldg-v005-miniblocks.svg){width=100% alt="权重转为 W 的输出通道行、收缩维列。先沿同一行块的两个 K 列块排列，再进入下一行块。每块占 16 KiB。"}

在 B 图中按行从左到右看，块号是 0、2、1、3；字节流却按 0、1、2、3 排列。转为 W 视图后，Dense 的 miniblock 按块行优先存放，再结合前述八路片段规则即可定位到具体元素。

#### 完整追踪 B[70,145] 的地址

该元素在 W 中是 W[145,70]。它位于第二个 N 分组、第二个 K 分组，因此 $b_N=1,b_K=1$，miniblock 编号 $j=3$。块内行索引 17（从 0 计数）属于片段 1，片段内行号为 1，列号为 6。

相对于整个 payload 基址，其字节偏移为

$$
o=3\times16384+1\times2048+1\times128+6\times2
=51340.
$$

若本核分配的 LDG 相对基址为 1048576，最终相对地址就是 $1048576+51340=1099916$。数学 B 的普通行优先偏移则为 $(70\times256+145)\times2=36130$；W 的普通行优先偏移为 $(145\times128+70)\times2=37260$。三个偏移不同，但指向的是同一个数学元素在三种表示中的位置。

对任意 Dense BF16 W[N,K]，可以不显式展开七个轴，直接计算

$$
j=\left\lfloor n/128\right\rfloor
  \left\lceil K/64\right\rceil+\left\lfloor k/64\right\rfloor,
\qquad
o=16384j+128(n\bmod128)+2(k\bmod64).
$$

| 数学坐标 | W 坐标 | miniblock | LDG 字节偏移 |
|---|---|---:|---:|
| B[0,0] | W[0,0] | 0 | 0 |
| B[0,16] | W[16,0] | 0 | 2048 |
| B[64,0] | W[0,64] | 1 | 16384 |
| B[0,128] | W[128,0] | 2 | 32768 |
| B[70,145] | W[145,70] | 3 | 51340 |
| B[127,255] | W[255,127] | 3 | 65534 |

Table: 同一 B[128,256] 中的坐标检查点

#### 七维变换是同一排列的另一种写法 {#ldg-seven-dim}

仍取 $W[N=256,K=128]$，先将 N 拆成 $2\times1\times8\times16$，将 K 拆成 $2\times1\times64$。第一步 reshape 得到

$$
[2,1,8,16,\;2,1,64].
$$

为了避免编译器内部 K/C 名称与数学 K 混淆，先用面向本例的轴名来读这个形状：

| 原轴号 | 教学记号 | 本例大小 | 解释 |
|---:|---|---:|---|
| 0 | $n_b$ | 2 | N 方向的外层 block |
| 1 | $n_g$ | 1 | 一个外层 block 内的 128 行分组 |
| 2 | $n_\ell$ | 8 | 每个分组内的 16 行片段 |
| 3 | $n_i$ | 16 | 片段内行位置 |
| 4 | $k_g$ | 2 | K 方向的 64 列分组 |
| 5 | $k_\ell$ | 1 | 本 BF16 normal 格式中退化为 1 的轴 |
| 6 | $k_i$ | 64 | 分组内列位置 |

Table: 七维 reshape 的每一维来自哪里

reshape 后仍按原来的 N 行、K 列顺序解释元素。要让一个 $128\times64$ miniblock 的所有元素连续，需把 $k_g$ 提到片段轴前面。实际排列采用

$$
(0,1,4,2,5,3,6),
$$

因此物理 shape 为

$$
[n_b,n_g,k_g,n_\ell,k_\ell,n_i,k_i]
=[2,1,2,8,1,16,64].
$$

最右边的 $k_i$ 变化最快，然后依次是 $n_i$、$k_\ell$、$n_\ell$、$k_g$、$n_g$、$n_b$。阅读七维形状的关键，是把它还原成这一层层循环顺序，而不是记住七个数字。

```python
# W contains coordinate labels in [N, K] order.
split = W.reshape(2, 1, 8, 16, 2, 1, 64)
packed = split.permute(0, 1, 4, 2, 5, 3, 6)
packed = packed.contiguous()
```

最后的 `contiguous()` 将新轴顺序物化为字节顺序。只对一个张量调用 `reshape`，或者只建立转置视图而仍拷贝旧底层字节，都不能代替 pack。

![七维变换的轴顺序，以及 miniblock 3 内元素 B[70,145] 的分解](assets/ldg-v005-seven-dim.svg){width=100%}

#### blocksize 与完整矩阵的分组

配套布局实现将原始二维输入称为 `[K,C]`，这里的源码 K 对应本章 N，源码 C 对应本章收缩维 K。其物理七维名称依次为 `k_block, k_group, c_group, k1, c1, k0, c0`，对应上表重排后的七个教学记号。

对于 normal BF16，固定的内层大小是 $k_1=8,c_1=1,k_0=16,c_0=64$。设输入为 W[N,K]，`blocksize` 为外层 N block 的范围，则

$$
G_N=\left\lceil\frac{\mathrm{blocksize}}{128}\right\rceil,\quad
B_N=\left\lceil\frac{N}{128G_N}\right\rceil,\quad
G_K=\left\lceil\frac{K}{64}\right\rceil.
$$

先补齐为 $[128B_NG_N,64G_K]$，再 reshape 为 $[B_N,G_N,8,16,G_K,1,64]$，最后按同一置换排列。普通 Dense 采用 `blocksize=128`，故 $G_N=1$。源码保留这个大小为 1 的轴，是为了与其他 blocksize 和专用格式共用维度结构；它不表示额外数据。

注意：在 normal 排列中，外层 $B_N$ 与 $G_N$ 重排后仍相邻。在无需额外补齐的条件下，只把这两个轴重新分组，并不会自动改变最终字节顺序。独立 Blocked 布局的变化来自“先切出子矩阵、对子矩阵分别 pack、再拼接”，不能仅通过修改 `blocksize` 来推断。

#### 尾部补齐与有效矩阵

若 B 为 $[K=70,N=130]$，对应 W[130,70]，普通 BF16 格式将 N 补到 256、K 补到 128。有效数据为 18200 字节，物理排列为 65536 字节。分配量约为有效数据的 3.60 倍，但数学输出仍只有 130 列。

补齐区用于满足布局，原始 N/K 用于边界与数学含义。一次 PE 是否读取全部分配，还取决于当前 tile、有效范围与执行方式；分配容量不能直接充当每次计算的实际流量。跨核复制、索引表、UScale 和对齐间隔应另外计入容量。

### Blocked：独立块紧密排列 {#ldg-blocked-compact}

用 $W[256,256]$ 举例，划成四个 $128\times128$ 子区：W 的左上为 A、右上为 B、左下为 C、右下为 D。每个子区包含两个 BF16 miniblock，占 32 KiB。完整 Dense pack 的顺序为 A、B、C、D。回到数学 $B[256,256]$ 图，四个子区的位置则是左上 A、右上 C、左下 B、右下 D。

现在沿收缩维 K 切成两个独立块，每块 W 的 shape 为 $[B_N=256,B_K=128]$。第一块包含 A、C，第二块包含 B、D。分别 pack 后按块号拼接，物理顺序变成 A、C、B、D；两块各占 64 KiB，整个矩阵仍为 128 KiB。

| 逻辑元素 | Dense 字节偏移 | 本例 Blocked 字节偏移 |
|---|---:|---:|
| W[0,0] | 0 | 0 |
| W[0,128] | 32768 | 65536 |
| W[128,0] | 65536 | 32768 |
| W[128,128] | 98304 | 98304 |

Table: BN=256、BK=128 时的独立块地址

Blocked 的当前块基址为 `base + block_id * 65536` 字节，块内 shape 为 `(256,128)`，局部 K 从 0 开始。Dense 则保持完整 `(256,256)` 视图，用逻辑 K 起点访问。混用“Blocked 块基址”和“Dense 完整行跨度”，会让第二块读错位置。

若独立块改为 W[128,128]，并按 N 块在外、K 块在内拼接，此例的字节顺序恰好与 Dense 相同。因此，判断布局是否相同，要比较实际块大小、遍历顺序与补齐；不能只比较 Dense/Blocked 名称。

### Blocked：带 padding 的对齐空间 {#ldg-blocked-padding}

沿用前一小节的两个独立 W[256,128] 块，每块的 packed 载荷为 64 KiB。若每个块放入一个 96 KiB 槽位，则前 64 KiB 保存矩阵字节，后 32 KiB 留作 padding。第二个块从 96 KiB 开始，两个槽位共占 192 KiB；矩阵有效载荷仍为 128 KiB。

![同一矩阵的 Dense、Blocked 紧排与 96 KiB 槽位 padding 对照](assets/ldg-blocked-padding.svg){width=100% alt="Dense 先沿矩阵行块内的列块排列。Blocked 将独立 PATCH 连续排列，每块 64 KiB。96 KiB 槽位在块后预留 32 KiB。"}

| 项目 | Blocked 紧排 | 本例带 padding |
|---|---:|---:|
| 块 0 的基址偏移 | 0 | 0 |
| 块 1 的基址偏移 | 64 KiB | 96 KiB |
| 每块 packed 载荷 | 64 KiB | 64 KiB |
| 两块的分配量 | 128 KiB | 192 KiB |

Table: 相同独立块在紧排与带 padding 时的地址和容量

padding 不改变块内元素的对应关系，但会改变下一块从哪里开始。读侧先选择块基址，再在该块内使用局部 shape 与坐标；不能继续按紧排的 64 KiB 步长推进。

一般情况下，一个对齐空间可以容纳多个 PATCH。设空间大小为 S、PATCH 大小为 P，且 $S\ge P$，则每个空间容纳 $q=\lfloor S/P\rfloor$ 个 PATCH，编号 j 的起点为

$$
o_j=\left\lfloor\frac{j}{q}\right\rfloor S+(j\bmod q)P.
$$

例如 P=64 KiB、S=128 KiB 时，q=2，两块正好填满空间，没有块间空隙；S=96 KiB 时 q=1，才得到图中的每块 32 KiB padding。这里的 S 与 P 均以字节量参与地址计算。

::: {.caution}
先计算实际 PATCH 载荷和每个对齐空间能容纳的 PATCH 数，再确定地址步长。`aligned_size` 为正不一定产生空隙；它也不等同于“在每个块后固定插入 aligned_size 字节”。
:::

### 带 UScale 的 Weight 布局 {#ldg-uscale}

带 scale 的 Weight 沿用所选块组织，同时增加与量化分组对应的系数页。准备时须把权重编码、块形状、scale 格式和读取参数一起确定；不能只在已有 BF16 载荷上增加一个量化参数。普通 LocalMem 的 Tr/Tc 与 aligned_size 只组织对应分配的 PATCH，不会自动生成 UScale；权重、系数与槽位的组合需使用匹配的量化 pack 契约。

INT4 不是简单把 BF16 字节偏移除以 4。以一个 W[256,128] 的 INT4 miniblock 为例，行 0–127 放低四位，行 128–255 放高四位；同列的两行共享一个字节。形成 [128,128] 的字节矩阵后，再执行相应的一字节格式 pack。数值量化与累加类型见[量化矩阵乘与 UScale](#quantized-matmul)。



#### UScale 保存什么，为什么另有页布局

若每 G 个收缩维元素共用一个系数，则逻辑 scale 为 $S[N,q]$，其中 $q=\lceil K/G\rceil$。权重元素 W[n,k] 使用 $S[n,\lfloor k/G\rfloor]$。scale 的行仍对应 N，列表示量化组，不是原矩阵的 K 列。

本章采用的 UScale 格式以 BF16 系数存放，每个物理页为 16 KiB，划成 8 个 2 KiB lane。每个 lane 负责一个 128 行 N 分组中的 16 行。一个 slot 保存“某个 N 分组、某个量化组”的这 16 个系数，占 32 字节，所以每 lane 每页可放 64 个 slot。

对 scale 坐标 (n,g)，设

$$
\ell=\lfloor(n\bmod128)/16\rfloor,\quad
r=n\bmod16,\quad
s=\lfloor n/128\rfloor q+g.
$$

则页号为 $p=\lfloor s/64\rfloor$，页内 slot 为 $s\bmod64$，字节偏移为

$$
o_S=16384p+2048\ell+32(s\bmod64)+2r.
$$

这一定义允许一个页容纳多个 N 分组的 scale。它不是对 S 的行优先字节每 16 KiB 截成一页；逻辑相邻的不同 N 行可能位于不同 lane。

#### 一个 256×128 INT4 块及其 scale

取 N=256、K=128、G=32，则 $q=4$。INT4 权重的有效载荷为 $256\times128/2=16384$ 字节，正好一个权重 miniblock。scale 有 $256\times4=1024$ 个 BF16，有效数据为 2048 字节。

每 lane 包含两个 N 分组、每组四个 slot，共 $2\times4\times32=256$ 字节。每 lane 的剩余 1792 字节为页内补齐；8 lane 合起来仍需一个 16 KiB UScale 页。两者合计分配 32 KiB，其中 18 KiB 是权重与有效 scale，14 KiB 是 scale 页补齐。

以 $S[145,2]$ 为例，$\ell=1,r=1,s=1\times4+2=6$，因此位于第 0 页、lane 1 的 slot 6，偏移为 $2048+6\times32+2=2242$ 字节。它缩放 W 第 145 行、K 位置 64–95 的 32 个 INT4 权重。

![UScale 页中的八路布局与 S[145,2]：有效 slot、补齐区和字节偏移](assets/ldg-v005-uscale.svg){width=100%}

::: {.keep}

一般所需页数为

$$
P_S=\left\lceil
\frac{\lceil N/128\rceil\,q}{64}
\right\rceil.
$$

:::

这里 N 先按 128 行分组，不能先计算未补齐的 `N*q*2` 再直接除以 16 KiB。多页时各页连续存放，每页都包含完整 8 lane；起始地址指向第一页，后续页位置由对应分组推导。跨页坐标与量化舍入的具体例子见量化章节。

#### USaddr 与相邻、独立 scale 存储

UScale 起始地址在指令与描述符中称为 `USaddr`；部分汇编辅助接口使用 `uaddr` 这个名字。它表示以 16 KiB 为单位的 scale 起点，不是 KV Cache 页表。以本核 LDG 相对字节地址 1048576 存放上述 INT4 权重为例，若 scale 紧随权重，则 scale 地址为 1064960，编码为 $1064960/16384=65$。

若同一份 weight 绑定到另一份 scale，scale 也可放在独立 LDG 分配中。例如其起点为 2097152，地址编码为 128。权重布局与权重基址不变，改变的是该次 PE 消费的 scale 资源。kernel 通过真实 LDG 指针显式绑定，由编译器纳入运行时基址：

```python
scale_address = tl.load(scale_ptr_list + core_id)
scale_base = scale_address.to(tl.pointer_type(tl.bfloat16))
scale_page = scale_base + page_id * 8192

w = tl.load(
    w_desc, memory_type="weight",
    weight_dtype="int4", ugroup=32,
    uscale=scale_page,
)
```

`scale_page` 是指向页首的标量 BF16 指针，`8192` 是一页的 BF16 元素数；它不是字节数，也不是软件逻辑页号。只把裸整数 65 或 128 作为 `uscale` 参数传入，无法保留 LDG 分配、基址和依赖关系。独立 scale 的写后读顺序、临时页复用和数值融合详见量化章节。

### PATCH 参数与容量对照

LocalMem 用 PATCH 表达若干 miniblock 的组合。在 BF16 下，一个 miniblock 是 W 的 $128\times64$；`Tr` 是 PATCH 沿 W 行方向的 miniblock 数，`Tc` 是列方向的数量。故 PATCH 的逻辑范围为 $[128Tr,64Tc]$。

前述独立 W[256,128] 对应 `Tr=2,Tc=2`。在 PATCH 内，先遍历列 miniblock，再遍历行 miniblock；PATCH 之间也按行优先排列。这与上述 A、C、B、D 的独立块组织相符。

`aligned_size` 同时决定 PATCH 是否生效及空间如何排列：

| 参数 | 实际组织 | 本例的结果 |
|---|---|---|
| `aligned_size=-1` | Dense，运行时将 Tr/Tc 视为 1/1 | 四子区 A、B、C、D |
| `aligned_size=0`、Tr=2、Tc=2 | 每个 PATCH 紧密相接 | A、C、B、D，无块间空洞 |
| `aligned_size=96`、Tr=2、Tc=2 | 96 KiB 空间中容纳一个 64 KiB PATCH | 每个空间末尾空出 32 KiB |

Table: LocalMem 中 Dense 与 PATCH 的参数关系

Tr、Tc 须为正整数，乘积须为不超过 4096 的 2 次幂；Dense 模式仍需通过此检查。`aligned_size` 为正时，取值范围为 16–1024 KiB，且至少容纳一个 PATCH。

此参数的正值单位是 KiB。例如 96 表示 98304 字节，不是 96 字节。对齐空间容纳多个 PATCH 时，先按 $q=\lfloor S/P\rfloor$ 计算每槽块数，再使用[块间 padding](#ldg-blocked-padding)中的地址公式。

`base_alignment_bytes` 则只约束整份分配的基址，不改变内部块排列。增大基址对齐不等于在每个块后插入间隔；同样，kernel 改了块跨度，Host pack 也必须提供对应间隔。零填充容量、读取 payload 和地址跨越的空洞应分别计算。

基址对齐须为 0，或不小于 16 KiB 的 2 次幂。需要额外基址对齐时，先用 `supports_base_alignment` 查询配套运行时是否支持。

## 分页 KV Cache 的离散布局 {#ldg-paged-kv}

分页 KV Cache 将历史 token 分成逻辑页，由映射关系选择各页的实际存放位置。页 0、页 1 在逻辑序列中相邻，在 LDG 中可以隔开，也可以按相反顺序放置。读取先找到正确的 payload 页，再在页内按 K/V 布局定位元素。

::: {.caution}
本节的专用 `kcache/vcache` 路径不支持 `weight_dtype`、`ugroup`、`uscale` 反量化参数。页映射与 `page_size` 描述地址和布局，不能代替 Weight 的量化分组与 scale 绑定。
:::

### 软件分页与专用索引两种路径

K/V 是算法中的用途，`kcache`、`vcache` 是具体接口中的布局与寻址方式。两者不能只按名字对应。普通 LDG 权重布局也可以保存 K/V，由 kernel 根据软件页表计算逻辑池中的位置，再以 `memory_type="weight"` 读取。

本书注意力案例采用这种普通布局：每核 K 池为 $[R,D]$，V 池为 $[D,R]$；软件页表保存逻辑页到物理页的映射，slot 表示物理 token 位置。生产者和消费者共同维护普通布局的地址关系。其具体实例见 [注意力与 KV Cache](#attention-cache)。

专用 `memory_type="kcache"` 或 `"vcache"` 的 PE 读取使用硬件索引表和对应布局。读取基址指向索引组织，硬件根据逻辑页查找 payload；直接追加的 LW 写入则接收已经解析好的 payload 页地址，不再查一次索引表。

| 比较项 | 普通 LDG 布局承载 KV | 专用 K/V 模式 |
|---|---|---|
| 读侧接口 | `memory_type="weight"` | `"kcache"` / `"vcache"` |
| 页映射 | kernel 读取软件页表 | PE 解释专用索引表 |
| 表的内容 | 逻辑页到池内物理页号 | LDG payload 地址编码 |
| 数据描述 | K 池 $[R,D]$、V 池 $[D,R]$ | 专用 K/V payload 排列 |
| 追加目标 | 由 slot 解析出的物理位置 | 已解析的 payload 页及页内位置 |
| 必须配套 | pool shape、slot、普通 pack | page_size、索引编码、K/V 布局 |

Table: 两种 KV Cache 路径的接口差异

不要把普通软件 `block_table` 复制到专用 32 KiB 索引表的位置，也不要用 `kind="weight"` 生成的 V 池直接配专用 `memory_type="vcache"` 读取。类似地，名为 `allocate_kv_cache` 的上层分配函数不自动承诺生成这一专用索引格式；仍须核对它返回对象的 shape、pack 和调用方约定。

### K 与 V 的页内排列

K 的每行对应一个 token，列对应 head dimension。就页内 payload 而言，在本章相同页形状、补齐与块组织下，BF16 专用 K Cache 与相同 $[T,D]$ 的普通 normal 权重排列一致；区别在于专用读取还带有页索引和地址解析。因而可以用“页内 K 与普通 weight 一样”理解数据排列，但不能把索引表也当作普通权重数据。

V 在 $PV$ 中沿历史 token 维收缩。若统一使用普通 weight 路径，K 保存为 $[T,D]$，V 保存为 $[D,T]$，随后都使用普通 normal pack；此时两者在逻辑层确实主要相差一次转置。专用 `memory_type="vcache"` 则在转置方向之外，还按列组组织物理数据，不能直接使用普通 `pack(V.T)` 的字节替代。

具体地，专用 BF16 V 的写端输入仍是 token-major `[T,D]`。其七维内层从 normal 的 `[8,1,16,64]` 改为 `[1,8,64,16]`，再采用同一轴置换。一个 16 KiB miniblock 现在覆盖写端输入的 64 个 token、128 个维度，分成 8 个“64 token × 16 dim”的 2 KiB 片段。每片内，同一 token 的 16 个维度连续，占 32 字节。

对一页 T=1024、D=128，设 token 为 t、维度为 d，页内偏移为

$$
o_V(t,d)=16384\lfloor t/64\rfloor
+2048\lfloor d/16\rfloor+32(t\bmod64)+2(d\bmod16).
$$

用一页 $T=1024$、$D=128$ 的 BF16 数据说明：同一个算法元素 `(token=1, dim=0)` 在 K 页内偏移 128 B；在普通 weight 路径的转置 V 中偏移 2 B；在专用 V Cache 页内偏移 32 B。三者都保存同一类型的标量，地址不同来自排列方式。

两种 V 表示都把 token 用作 PV 的收缩维，却以不同的小片组织字节。应先选择读写契约，再确定转置和 pack；不能由“V 需要转置”直接跳到某一种物理格式。

| 使用方式 | 逻辑矩阵方向 | 页内或矩阵内排列 |
|---|---|---|
| 普通 K / 专用 K payload | token × dimension | normal 权重排列 |
| 普通 weight 路径的 V | dimension × token | 逻辑转置后做 normal pack |
| 专用 vcache payload | 读取沿 token 维收缩 | 专用列组排列 |

Table: K/V 的逻辑方向与物理排列需要分别确定

LocalMem 的 `kind` 和 Triton 的 `memory_type` 属于不同接口。不要把 `kind="kv_value"` 的产物当作任意专用 `vcache` 页。本书普通注意力池使用 `kind="weight"`；专用页追加使用已匹配的 payload pack 与 LW 模式。改变其中一层时，应同时核对生产者、读取描述符和页内字节映射。

### 专用页、索引表与离散地址 {#ldg-kv-index}

#### 容量、页长和有效长度

以下固定一个核、一个头、BF16、$D=128$、`PAGE_SIZE=1024`。一个 K payload 页可容纳 $1024\times128$ 个元素，占 262144 字节，即 256 KiB；V payload 页也占 256 KiB，但排列不同。

假设页内已有 token 0–999，本次在 `token_offset=1000` 写入新 token。写入完成后，有效长度由 1000 变为 1001，页容量仍为 1024。位置 1001–1023 不应参与本次注意力。容量、已写入范围与因果可见范围是三个独立概念。

`PAGE_SIZE` 决定物理页布局，计算用的 `BLOCK_N` 或 `BLOCK_K` 决定一次 PE 处理多少元素。页长 1024 可以由多次 128 元素计算消费，不必把 PE tile 也设成 1024。反过来，只把消费者的 tile 改大，不会改变已有页的排列。

#### 专用索引表的物理组织

本章专用格式使用 32 KiB 索引表，物理 shape 为 $[2,8,512]$，元素为 32 位整数。它可表示 1024 个逻辑条目，每个条目在 8 个 WL1 位置各有一份副本。因此，一个逻辑条目的更新不能只改其中一份。

对逻辑条目 $j$ 和 WL1 编号 $w$，其表内字节偏移为

$$
\operatorname{index\_off}(j,w)
=16384\left\lfloor j/512\right\rfloor
+2048w+4(j\bmod512).
$$

例如条目 513 在每个 WL1 副本中的偏移为 `16384 + 2048*w + 4`。这里 512 是索引表每组副本包含的条目数，与一个 payload 页容纳多少 token 无关。

条目值使用 16 KiB 为单位的 LDG payload 地址编码，而不是 Host 指针或普通逻辑页号。地址重定位由对应运行时/准备工具负责。对于基址为零、表后立即放一个 payload 页的教学布局，索引表占前两个 16 KiB 单位，首 payload 的编码为 2。把这套数据迁移到别处时，必须重新满足实际地址编码契约，不能机械保留数值 2。

#### 两个逻辑页可以按相反顺序放置

取两个 K payload 页，每页 1024 个 token、D=128，每页 256 KiB。为看清索引的作用，令本核 LDG 相对地址 0 处放 32 KiB 索引表，紧接着先放逻辑页 1，再放逻辑页 0：

| 内容 | 本核 LDG 相对字节区间 | 页首的 16 KiB 地址编码 |
|---|---|---:|
| 索引表 | [0,32768) | 不作为 payload |
| 逻辑页 1 的 payload | [32768,294912) | 2 |
| 逻辑页 0 的 payload | [294912,557056) | 18 |

Table: 逻辑页顺序与实际 payload 顺序分离的例子

此时索引条目 0 填 18，条目 1 填 2，并各写入八个副本。读取全局 token 1025 时，逻辑页号为 $1025//1024=1$，页内位置为 1；索引解析得到 payload 起点 $2\times16384=32768$。维度 0 在 K 页内的偏移为 128 字节，因此最终地址为 32896。

![两个专用 K 页的乱序放置：从 token 1025 经索引到 payload 字节 32896](assets/ldg-v005-index-pages.svg){width=100% alt="token 1025 落在逻辑页 1 的第 1 行。索引编码 2 定位本核 LDG 基址加 32768 B，再加页内 128 B 得 32896 B。"}

这条地址链与 UScale 地址不同。这里先用逻辑页号查表，再在 payload 内做矩阵布局寻址；`USaddr` 直接给出 scale 第一页的起点，没有上述逻辑页到 payload 页的查表过程。二者都使用 16 KiB 地址单位，不能据此将两种资源或表内容互换。

一个 1024-token K 页包含 16 个 16 KiB miniblock。修改 `page_size` 会改变一个索引条目覆盖的逻辑范围；修改计算 tile 则只改变一次 PE 读取的范围。把这两个参数都叫“块大小”会掩盖它们不同的职责。

### 从索引页读取 QK 与 PV

下面回到单个 payload 页的例子：页容量为 1024 个 token，完成追加后的有效长度为 1001。它用于说明一次页内读取；前面的两页历史还需要按页和 tile 分别遍历，不能直接套用这里的单页 shape。

#### QK 的逻辑矩阵和读取模式

QK 计算为 $Q[M,D]K[S,D]^T$，输出 $[M,S]$。在本例中 D 为 128，S 的物理页容量为 1024。K 描述符以 token 为行，head dimension 为列，读取使用专用 K 模式：

```python
k_desc = tl.make_block_ptr(
    base=k_index_base,
    shape=(1024, 128),
    strides=(128, 1),
    offsets=(seq_start, 0),
    block_shape=(128, 128),
    order=(1, 0),
)
k_tile = tl.load(
    k_desc, memory_type="kcache", page_size=1024,
    boundary_check=(0, 1), padding_option="zero",
)
scores = tl.dot(q_tile, tl.trans(k_tile))
```

其中 `k_index_base` 来自该核专用索引表的基址，`q_tile` 已按对应 query tile 准备。该片段展示一块 QK 的读取和计算关系；完整注意力还需遍历历史、缩放、mask、softmax 及输出组织。

#### 有效长度需要单独进入注意力 mask

追加完成后有效长度是 1001。以 128 个历史位置为一个计算 tile 时，最后一块从 896 开始，有 105 个有效位置和 23 个未生效位置。PE 描述符中的容量边界不会自动表达“只有前 1001 个 token 对当前 query 可见”。

QK 的未生效位置必须被注意力 mask 排除。仅把 K 的无效位置补零，会得到零分数；如果有效分数为负，零分数仍可能在 softmax 中获得概率。应按所用数值路径建立有效长度与因果 mask，PV 再只消费被允许的位置。关于稳定 softmax 与在线累加，参见注意力实例。

::: {.keep}

#### PV 的矩阵方向

PV 计算为 $P[M,S]V[S,D]$，输出 $[M,D]$。PE 的右操作数统一按 $[N,K]$ 理解，所以此时 N 为 D、K 为 S，描述符逻辑形状是 $[128,1024]$：

```python
v_desc = tl.make_block_ptr(
    base=v_index_base,
    shape=(128, 1024),
    strides=(1024, 1),
    offsets=(0, seq_start),
    block_shape=(128, 128),
    order=(1, 0),
)
v_nk = tl.load(
    v_desc, memory_type="vcache", page_size=1024,
    boundary_check=(0, 1), padding_option="zero",
)
partial = tl.dot(p_tile, tl.trans(v_nk))
```

:::

写 V 时以 `[token,dim]` 描述新 token，读 PV 时以 `[dim,token]` 描述 PE 的右操作数，这两种视图由专用 V 布局连接。不能仅凭两个维度都等于 128 的小例子判断方向正确；验证时应让序列长度与头维度不同，并采用位置相关数据。

QK 和 PV 之间还存在数值中间结果，`p_tile` 的 dtype、softmax 精度和跨历史块累加方式都属于算法契约。布局正确只能保证读到正确元素，不能替代数值与注意力 mask 的验证。

## 数据准备、更新与读取 {#ldg-preparation}

### 静态 Weight：先准备并写入，再启动消费者

静态权重应在消费它的 kernel launch 之前完成以下工作：确定完整布局与量化资源，生成相应物理内容，分配并写入目标核的 LDG，再建立写入完成到消费者执行的顺序。Host 上生成 packed 张量只完成了格式准备，尚不代表数据已驻留 LDG。

::: {.caution}
kernel 中的 Weight load 读取已经准备好的 LDG 内容，不负责把 Host 权重即时打包上传。weight、UScale 与读取描述符必须配套，且所有写入均须在消费者使用前完成。
:::



配套 Python 运行时提供 LocalMem 封装。下面从 CPU BF16 逻辑矩阵建立普通 Weight LDG 对象，配套入口负责布局准备与复制：

```python
import torch
import torch_zeus

weight = torch.arange(256 * 128, dtype=torch.float32)
weight = (weight.reshape(256, 128) % 31 - 15).to(
    torch.bfloat16
)
packed = torch.zeus.local_memory.from_tensor(
    weight, kind="weight", Tr=1, Tc=1, aligned_size=-1,
    core_id=0,
)
```

输入采用 $[N=256,K=128]$ 顺序，与 Dense 小节的数学 B[128,256] 相对应，也与 `nn.Linear` 的权重方向一致。这里显式选择 Dense，得到四个紧排的 miniblock，共 64 KiB。`Tr`、`Tc` 是 PATCH 参数，不是原始行列数，也不是 kernel 的 `BLOCK_M`、`BLOCK_N`；Dense 下运行时按 1/1 处理。

准备好的 `packed` 可作为 LDG 参数传给相配套的 JIT kernel，适配器按其每核存储组织地址表。上例明确放在核 0；使用两个核时，应按对应接口准备分片或复制，不能仅改启动 grid。读取完成前保留 `packed` 对象；修改原 CPU 的 `weight` 不会自动更新已经准备好的 LDG 内容。

若权重多次复用，应把 pack 和装载放在首次计算之前。每次 PE 仍会读取 LDG，但不需要为每个 token 重做 Host pack。KV Cache 或跨步状态则在运行中变化，应分开管理“已分配容量”“已完成写入范围”和“本次可见范围”，并在下一次消费前完成相应同步。

若采用异步复制，按配套运行时接口保留输入和 LocalMem owner，并在 launch 前建立完成依赖。已经由发行包提供的 packed 权重按其格式装载，不重复执行另一套 pack。

### 地址表、逻辑描述符与每核归属

#### 每核地址表不是矩阵数据

常见 LDG kernel 参数采用 `*i64` 类型的每核地址表。kernel 先用核编号选择条目，再把其中的地址值转换为相应数据类型的指针：

```python
core_id = tl.program_id(0)
ldg_address = tl.load(ldg_ptr_list + core_id)
ldg_base = ldg_address.to(tl.pointer_type(tl.bfloat16))
```

这里存在两次不同含义的访问。`ldg_ptr_list + core_id` 指向一个 64 位地址条目；`ldg_base` 才指向该核的 LDG 数据。表中每个条目占 8 字节，但 BF16 指针加 1 前进 2 字节。把地址表直接当成 BF16 矩阵读取，会把地址位模式解释成数据。

在 Python JIT 接口中，配套的 LocalMem 对象可以由适配器转换为这类参数。程序应传递对象本身，使运行时保留其每核存储、大小和生命周期信息。不要以一个普通 Tensor 的 `data_ptr()` 整数替代整个 LocalMem 对象，也不要把 Host 虚拟地址手工填入设备地址表。

#### 同一个数值地址不代表同一块数据

核 0 和核 1 的 LDG 归属不同。一个地址值必须结合所选核及其地址空间解释。两核表项数值相等，可能对应各自局部空间中的相同偏移，而非共享同一个 payload。反之，两个不同逻辑视图也可能引用同一块分配。

如果核 0 保存 W 的前半行、核 1 保存后半行，kernel 必须同时使用各核的基址和各自的逻辑 shape。只把 `CORE_NUM` 从 1 改成 2，却继续按完整 W 的 shape 解释半份数据，会越界或读取错误行。分片方案、每核地址表和输出分区应一起定义。

::: {.keep}

#### 逻辑描述符的六个参数

用 `base`、`shape`、`strides` 和 `order` 描述逻辑矩阵 W，再用 `offsets` 和 `block_shape` 选出当前 tile。

```python
b_desc = tl.make_block_ptr(
    base=ldg_base,
    shape=(N, K),
    strides=(K, 1),
    offsets=(n0, k0),
    block_shape=(BLOCK_N, BLOCK_K),
    order=(1, 0),
)
b_nk = tl.load(
    b_desc, memory_type="weight",
    boundary_check=(0, 1), padding_option="zero",
)
```

:::

当这个读取被 PE 矩阵运算消费时，编译器据此建立 LDG 操作数描述符。这里的 `strides=(K,1)` 描述原逻辑矩阵；它不是声明物理 LDG 字节已经按普通行优先排列。物理排列由 pack 契约解释。

同样的 `memory_type="weight"` 读取若用于普通读回而非 PE，可以走不同降级路径。因此，使用哪一种 shape 还与目的有关：矩阵计算使用逻辑矩阵视图，检查原始字节使用物理视图。不能把读回的 packed 字节直接与原始行优先张量逐元素比较。

### 动态写入：LW 转换与分段 ST

#### 矩阵读取、原样复制与布局写入

PE 从 LDG 经 WL1 读取右操作数，当前激活经 AL1 进入 PE，结果经 OL1 返回 L2 Buffer。已排好的大矩阵不必先完整搬进 L2 Buffer；相关单元与核内连接见[Zeus 硬件与近存架构](#hardware-model)。

LD/ST 通过 DMA、互连及 WA/WL1 相关接口访问原始字节，保持既有字节次序；LW 从 L2 Buffer 的逻辑矩阵生成目标布局并写入 LDG。两者即使源、目的区域相同，处理的视图仍可能不同。

| 使用目的 | 主要路径 | 程序需要控制的内容 |
|---|---|---|
| 矩阵右操作数读取 | LDG → WL1 → PE | 逻辑 shape、pack、tile |
| 原始字节读回或写入 | LD/ST ↔ 互连、WA/WL1 ↔ LDG | 字节范围、对齐、完成顺序 |
| 逻辑矩阵或缓存追加 | L2 Buffer → LW → WA/WL1 → LDG | 目标布局、有效位置、写后可见性 |

Table: 按数据视图区分三种 LDG 访问路径

PE 矩阵读取与原始字节访问具有不同的并行度，不能套用同一带宽。带宽口径、部分写和拷贝的计数方法见[Zeus V3 的容量、算力、吞吐与带宽](#hardware-throughput)。

#### 从设备产生的矩阵写入 LDG

若矩阵已经在 GDG，先用普通 load 得到 L2 Buffer tile，再以 LW 写入目标 LDG。以下为完整算子中的核心片段，假设 `ldg_base` 来自有效的每核地址表，且已分配至少 32 KiB：

```python
src = tl.make_block_ptr(
    base=src_ptr,
    shape=(128, 128),
    strides=(128, 1),
    offsets=(0, 0),
    block_shape=(128, 128),
    order=(1, 0),
)
values = tl.load(src)
dst = tl.make_block_ptr(
    base=ldg_base,
    shape=(128, 128),
    strides=(128, 1),
    offsets=(0, 0),
    block_shape=(128, 128),
    order=(1, 0),
)
tl.store(
    dst, values, memory_type="weight",
    store_mode="lw", blocksize=128,
)
```

`values` 表示逻辑 $[128,128]$ BF16 矩阵；LW 负责把它写成相应 7D 布局。`blocksize` 必须与读侧和预期物理排列一致。本接口要求显式提供不小于 128 的 2 次幂 blocksize。这个要求属于 LW/页布局，不能泛化为所有 Triton 张量维度都必须是 2 次幂。

LW 当前接受相应的 INT8、FP8 和 BF16 数据路径；不能因为 FP16 也占 2 字节，就把它当作 BF16 输入。目标必须能追溯到 LDG 地址表所代表的分配，普通 GDG 指针加上一个 `memory_type` 字符串不能替代 LDG 分配。

#### 原样搬运与布局写入的选择

若源数据已经是 packed 字节，使用原样搬运保留字节序；若源数据是逻辑矩阵，使用匹配布局的 LW 或先显式完成 pack 再搬运。对原始字节进行 load/store 与对逻辑矩阵进行矩阵运算，虽然都可能出现 `memory_type="weight"`，但读写视图不同。

| 源内容 | 所需目标内容 | 合适的处理 |
|---|---|---|
| Host 逻辑矩阵 | 普通 LDG 矩阵 | 配套 LocalMem pack 与装载 |
| L2 Buffer 逻辑矩阵 | 普通 LDG 矩阵 | LW 普通布局写入 |
| L2 Buffer 新 K/V token | 专用 K/V payload | 对应 cache 布局的 LW 写入 |
| 已正确 pack 的字节流 | 相同 LDG 字节流 | 原样复制 |
| LDG packed 内容 | GDG 调试输出 | 物理视图读回，再独立解码 |

Table: 根据源格式选择装载或写入方式

原样读回是定位 pack 问题的有效办法。普通 $128\times128$ BF16 样例可以把 32 KiB packed 内容看成 $[256,64]$ 的物理视图，读回后与 CPU pack 结果比较。这里的 256 行不表示原矩阵有 256 个输出通道。

手动更新 KV 时，也可以在 kernel 中用索引和循环形成目标物理片段，再通过普通 ST 分段写入。此时格式转换由调用方完成，ST 只保留给定字节次序。下一小节的 token 1000 地址例给出需要更新的实际片段；采用这条路线还须保护片段之外已有 token 的内容。



### 追加一个 BF16 token {#ldg-append-token}

#### 输入与目标页

新 token 是连续 128 个 BF16 值，共 256 字节。Host 或上游阶段需先确定当前请求、头、物理页、页内 token 位置，并保证目标页仍由该请求拥有。下面两个核心片段来自同一种接口契约：页长为 1024、头维度为 128、单核，且 $0\le\text{token_offset}<1024$。

先从每核 K 页地址表取得已解析的 K payload 基址，再读取新 token 的 K 向量。

```python
core_id = tl.program_id(0)
k_address = tl.load(k_page_ptr_list + core_id)
k_page_base = k_address.to(tl.pointer_type(tl.bfloat16))

src = tl.make_block_ptr(
    base=new_token_ptr,
    shape=(1, 128),
    strides=(128, 1),
    offsets=(0, 0),
    block_shape=(1, 128),
    order=(1, 0),
)
token = tl.load(src)
```

这里不使用专用索引表首地址。若应用持有的只有页表，需要先解析出本次写入的 payload 地址，再按约定组织参数。把索引表地址误当成 payload 地址，会破坏索引本身或写到错误区域。

::: {.keep}

#### 写入 K payload

K 的逻辑页表示为 $[1024,128]$，行是 token，列是 head dimension。目的块只覆盖当前 token 的一行：

```python
k_dst = tl.make_block_ptr(
    base=k_page_base,
    shape=(1024, 128),
    strides=(128, 1),
    offsets=(token_offset, 0),
    block_shape=(1, 128),
    order=(1, 0),
)
tl.store(
    k_dst, token,
    memory_type="kcache", store_mode="lw",
    blocksize=1024,
)
```

:::

LW 使用 K payload 所需的普通排列，不进行索引表查找。源码中的 `offsets` 是页内逻辑行位置，不是把 token 当成连续 256 字节写入 `base + token_offset * 256`。

#### 写入 V payload

追加 V 时，按相同的 `src` 模板从 `new_token_ptr` 读取新 token 的 V 向量。用 V 的每核页地址表建立写入描述符，逻辑页仍按 token 行描述。

```python
v_address = tl.load(v_page_ptr_list + core_id)
v_page_base = v_address.to(tl.pointer_type(tl.bfloat16))
v_dst = tl.make_block_ptr(
    base=v_page_base,
    shape=(1024, 128),
    strides=(128, 1),
    offsets=(token_offset, 0),
    block_shape=(1, 128),
    order=(1, 0),
)
tl.store(
    v_dst, token,
    memory_type="vcache", store_mode="lw",
    blocksize=1024,
)
```

两张页地址表分别来自已解析的 K 和 V payload。写入 V 时直接传入 token-major 输入，由 LW 完成 K/C 交换的物理排列。

读写模式也不能混同：专用 PE 读取需要索引寻址，直接 LW 写入已经解析了 payload。V 的读侧模式包含“布局交换”和“索引查找”两项语义，直接写入只需要前者。算子代码使用上面的接口，具体位域由编译器设置。

#### token 1000 到底写了哪些字节

K payload 的 128 个值分为两个 64 元素段，物理偏移分别为 242688 和 259072；每段 128 字节。V payload 则分为 8 个 16 元素段，每段 32 字节，第 $h$ 段的起点为 $247040+2048h$，$0\le h<8$。

| 数据 | token 1000 的写入段 | 每段字节数 | 逻辑总量 |
|---|---|---:|---:|
| K | 起点 242688、259072 | 128 | 256 B |
| V | 起点 $247040+2048h$，$h=0\ldots7$ | 32 | 256 B |

Table: 相对于各自 payload 页基址的写入偏移

这些偏移来自固定的 BF16 页布局，不适用于其他 dtype、头维度或不同页面组织。它们说明：相同的逻辑一行，在 K/V 物理页里具有不同分散方式。直接逐字节覆盖一个连续的 256 字节区域不能同时实现这两个追加操作。

![专用 K/V 物理页中 token 1000 的写入位置与字节段](assets/kv-cache-update.svg){width=100% alt="一次追加每页只写 256 B。K 页写两个 128 B 段，V 页以 2048 B 为间隔写八个 32 B 段。其余 token 保持原值。"}

#### 追加正确性要检查整页

每页有 131072 个 BF16 元素，单次只改变 128 个，占约 0.098%。即使写入完全丢失，整页余弦相似度也可能很高。追加检查应比较预期位置的全部新值，并确认其余位置保持原内容。

测试位置应覆盖布局分组边界。K 可检查 0、127、128、1000、1023；V 可检查 0、63、64、1000、1023。还应在宿主拒绝负位置和 1024，不能把越界写当作下一页追加。下一页需要重新选择 payload，并把页内位置归零。

随书提供 `examples/append_kcache.py` 和 `examples/append_vcache.py` 两个完整 kernel。它们的默认命令执行 Host 地址检查，打印当前 token 对应的物理偏移；功能仿真使用文件中的参数与参考实现声明执行设备程序。

```bash
python3 examples/append_kcache.py --token-offset 1000
python3 examples/append_vcache.py --token-offset 1000
```

Host 检查应分别得到表中的两个 128 字节段和八个 32 字节段。两个示例都使用已经解析的 payload 页，不能把普通应用软件页表直接作为其 `cache_page_ptr_list` 参数。执行后端与逐页 raw 比较结果可在附录的示例记录中核对。

### 完成顺序与资源复用 {#ldg-lifetime}

一个完整更新步骤应依次满足：新 K/V 已生成，payload 写入完成，必要索引更新完成，新的有效长度才对消费者可见，QK/PV 按同一请求和同一版本读取。若是在已有页中追加，索引条目通常不需改变；若分配了新页，则要在消费者使用前建立映射。

正文中的页长 1024 追加片段，应与同样页长、同样布局的消费者配对。配套源码还有页长 128 的独立 GEMM 测例，用来检查专用读取模式；它不能直接消费按页长 1024 写好的页面。修改参数时必须重新核对物理 pack、索引和描述符，而不是只改一处 `page_size`。

#### 将“可分配”与“可覆盖”分开

一个页被请求 A 使用后，即使 A 已经提交最后一次 kernel，也不能立即分配给请求 B 覆盖。必须等 A 的最后一个读取者完成。否则 B 的追加可能与 A 的 QK/PV 读取重叠，错误表现与请求长度、运行时机有关，单次测试不一定出现。

有效长度也不应先于数据发布。若先把长度从 1000 更新为 1001，消费者可能读取尚未写完的 token 1000。无论是同核融合、多个 kernel 还是多条执行流，都要保留数据写入到可见范围更新之间的依赖。

#### 写入完成后再消费新数据

LW 已提交不等于 LDG 内容已经更新。读取新状态的 PE 操作必须发生在对应写入完成之后；覆盖旧状态前，则必须等旧数据的最后一个读者结束。同一 kernel 内应表达真实的数据流，保留可追踪的每核地址、分配身份和访问范围，使编译器能够维护所需顺序。把同一物理区域伪装为两个无关参数，会隐藏实际的读写关系。

访问范围应同时考虑读者和写者。两个追加操作写入不同 token，可能互不重叠，但都可能与读取整页的 QK/PV 操作冲突。若两个操作更新同一 token，还必须定义哪次更新最终生效。循环复用同一页时，同样需要等上一轮的读者结束，不能仅依赖一次测试中偶然出现的执行先后。

核间也需要明确的交接协议。两个核在同一次启动中分别执行 store 和 load，并不会自动形成可用的生产消费关系。基础方案可以让各核写独立结果，再由有完成边界的后续阶段消费；共享 KV 页时，还要同时核对所属核、每核地址和页的访问协议。

#### 多个 kernel 与多条流

基础闭环先运行写入任务，用 `torch.zeus.synchronize(device)` 等待目标设备完成，再运行消费任务。缩小等待范围时，可用同一流顺序或流间等待建立依赖。用法见[Stream 与 Graph](#streams-and-graph)。

输入、输出、LocalMem 对象和索引表必须存活到设备最后一次访问结束。异步提交后立刻释放 owner，或把页交给其他请求覆盖，都可能使仍在运行的消费者读到另一请求的数据。完成等待保证设备工作结束；若宿主封装另行缓存了 LDG 的线性视图，还需使对应缓存失效，再读取更新后的内容。

图捕获和复用还要求参数及存储在重放期间保持有效。更换页表内容、复用 slot 或改变有效长度，应区分值更新与地址变化；缓存的二进制不等于缓存的输入内容。每次重放都要得到本次请求的数据，而不是上一次状态的偶然残留。

#### 计算驻留量时包含元数据与补齐

本例 K/V payload 合计 512 KiB。如果 K 和 V 各带一份 32 KiB 专用索引表，合计为 576 KiB，尚未包含分配对齐、额外页或其他元数据。追加一个 token 共写入 512 字节，应与驻留容量分开统计。

多头、多层和多请求场景按各自归属累计。先确定一个物理页是否被多请求共享，再统计唯一分配的页；不要把多个引用各算成一份容量，也不要忽略因核间复制产生的真实额外存储。长期容量、某次更新字节数与某次计算读取量应分别列出。

## 布局选择与错误定位 {#ldg-layout-checks}

### 从用途选择布局与访问方式

| 当前需求 | 选择与准备 | 配套核对 |
|---|---|---|
| 完整常驻权重，按 tile 访问 | Dense；上传完整 pack 结果 | 完整 shape、尾部补齐与 tile 起点 |
| 独立权重块逐块消费 | Blocked 紧排 | 块顺序、块内 shape 与基址步长 |
| 独立块需要规定的对齐空间 | Blocked 带 padding | PATCH 数量、槽位容量与有效载荷 |
| Weight 使用分组 scale | 配套 Weight 编码与 UScale 页 | `ugroup`、scale 页及完成顺序 |
| 历史 KV 分页增长 | 配套的软件页表或专用索引 | K/V 页内布局、有效长度与页归属 |

Table: LDG 布局与准备方式速查

确定布局后，分别计算长期驻留容量、一次更新字节量与一次矩阵读取量。padding、索引和 scale 都可能占用容量，但不一定在每次计算中全部读取。性能分析需要与实际消费范围一致。

### 地址与单位速查

每核地址表、专用索引表和应用软件页表解决三个不同问题。每核地址表决定当前核使用哪份存储；专用索引表由 PE 定位 LDG payload；软件页表则由 kernel 或宿主实现应用级映射。它们的元素类型、数量、所处空间和编码含义都不相同。

| 对象 | 谁解释它 | 一个值表示什么 |
|---|---|---|
| 每核 `*i64` 地址表 | kernel 与运行时适配 | 当前核分配或 payload 页的地址 |
| 应用 `block_table` | kernel 或宿主 | 应用逻辑页对应的池内物理页号 |
| 专用 K/V 索引表 | PE 的索引访问路径 | payload 起点的 16 KiB 地址编码 |
| UScale 起点 `USaddr` | PE 的 scale 访问路径 | scale 第一页的 16 KiB 地址编码 |

Table: 地址表、页号与 scale 起点的单位和消费者

直接追加还可能接收“每核 payload 页地址表”。它在 ABI 上仍表现为每核地址条目，但条目指向已选定的页，而不是专用索引表首地址。读侧与写侧参数都叫 `cache_ptr` 容易掩盖这个区别，应使用 `index_base`、`page_payload` 等能表达用途的名字。

### 从逻辑元素定位到物理字节

发现输出错误后，先选一个能够手算的逻辑元素，追踪它的原值、pack 后偏移、当前核基址、描述符起点和最终消费位置。对布局错位，使用行列都不同的输入通常比全 1 矩阵有效；对 KV 追加，旧页与新 token 应使用不同模式的数据。

可以先验证原样搬运，再验证 pack 字节，再验证 PE 消费，最后验证注意力结果。每一步都检查同一输入，避免多个错误在“pack 后再反变换”的过程中互相抵消。追加时保留旧页副本，并检查其余位置不变。

| 现象 | 优先检查 | 修正方向 |
|---|---|---|
| 全部元素错位，数值范围正常 | 行列方向、7D 次序、dtype | 对照坐标到字节偏移 |
| 一部分列正确，其他列错误 | 列分组、stride、补齐 | 核对逻辑 K 与物理列组 |
| 第一块正确，后续块错位 | 紧排或槽位步长、PATCH 数量 | 核对 padding 与实际块基址 |
| 单核正确、双核错误 | 每核地址表和分片 shape | 配套修改 pack 与输出归属 |
| 追加前正常，追加后全页异常 | payload 地址是否误用索引基址 | 分开命名与传参 |
| 新 token 不生效 | token_offset、有效长度、写完成 | 检查更新位置及可见顺序 |
| 只有边界 token 错误 | 63/64、127/128、页尾位置 | 检查布局分组和换页 |
| K 正常、V 错误 | V 布局、矩阵方向、读取模式 | 配对写端和 PV 描述符 |
| 多请求偶发错误 | 页复用、跨流依赖、slot 冲突 | 延迟覆盖至最后读者结束 |

Table: LDG 与 KV Cache 错误的定位入口

### 用具体坐标复核布局

随书脚本同时检查二维坐标公式、七维变换、逆变换和全部有效元素的位置：

```bash
python3 examples/ldg_layout.py
```

输出 `execution=host_layout_only`。脚本只需 NumPy，不加载 Zeus 运行时；它覆盖本章的 Dense/Blocked、UScale 和专用 K/V 地址例子。它检查 Host 上的排列与公式，不证明设备读取、同步或矩阵计算已经通过。

最小复现应保存逻辑输入、当前页的更新前内容、写入 token、页内位置、有效长度、每核分片和布局参数。若包含索引表，还应保存其逻辑条目与编码后的副本。只有输出差异，没有这些状态，通常无法区分地址错误、布局错误和并发覆盖。

编译成功后也应检查产物使用的是预期路径：普通原样复制、LW 布局写入、普通 PE 读取或专用 K/V 索引读取。随后分别比较新位置、未变化区域和最终矩阵结果。这样得到的结论才能说明整个生产消费契约，而不只是某一段代码可以运行。
