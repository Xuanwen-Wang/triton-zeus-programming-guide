# 数据类型与语言规则 {#language-and-api}

一个 Triton kernel 同时描述三件事：每个程序实例负责哪些元素，怎样从存储中取得这些元素，以及用什么类型计算结果。接口名称相同，并不意味着不同设备会采用相同的数据布局、执行单元或近似算法。本章建立贯穿算子的语言规则；平台特有的访存参数、计算接口及执行提示见[Zeus 编程接口](#api-reference)。

阅读示例时，应把**存储类型、计算类型和输出类型**分别标出来。例如，BF16 张量可以在装载后转换成 FP32，完成乘加后再转换回 BF16。转换发生的位置决定了数值语义，也决定了需要保存的中间结果。

## 数据类型、转换与累加精度 {#types-and-precision}

### 存储类型和计算类型

指针的元素类型决定一次索引增加所跨越的元素大小。对于 `*bf16` 指针，`ptr + 1` 指向下一个 BF16 元素；它不是将地址增加一个字节。张量装载后的 dtype 由指针类型决定，后续 `.to(...)` 才改变计算中的数值类型。

本书基本算例使用下列组合。表中的用途是接口的使用边界，不能据此把某一种 dtype 扩展成任意算子的支持承诺。

| 类型 | 本书中的用途 | 需要检查的条件 |
|---|---|---|
| `tl.bfloat16` | 向量、归一化和矩阵输入输出 | 转换位置、舍入、溢出范围 |
| `tl.float32` | 向量计算、归约与矩阵累加 | 运算顺序、近似函数和中间物化 |
| `tl.int32`、`tl.int64` | 索引、长度、步长与标量参数 | 数值范围、地址计算和 ABI 位宽 |
| `tl.int8` | 具备相应契约的量化数据 | 缩放、饱和、矩阵累加类型 |
| `tl.float8e4nv` | E4M3FN 格式的进阶数据路径 | 格式一致性及对应算子支持范围 |

Table: 主要类型及其使用条件

BF16 与 FP32 的指数宽度相同，BF16 的有效精度更低。提高中间计算精度有助于减少累计误差，但无法恢复输入在 BF16 存储时已经丢失的信息。FP8 还需要明确具体编码；E4M3FN、E5M2 和 FNUZ 不能仅因同为 8 位而互换。

整数类型首先服务于精确的索引与控制。一个 `i32` 长度能够进入 kernel，并不代表任意 INT32 张量运算都能映射到 VP。张量整数计算采用对应算子的类型契约；地址计算则应在宿主端检查长度与步长乘积是否超出所用整数类型的范围。

### 标量、tile 和指针是不同的对象

设备函数中的 `numel` 是运行时标量，`BLOCK` 是编译期值，`tl.arange(0, BLOCK)` 是包含 BLOCK 个元素的 tile。它们虽然都可能参与整数算术，但决定代码的方式不同。运行时标量每次启动都能改变；编译期值用于决定静态形状和被保留的代码；tile 中的每个元素代表一个逻辑位置。

指针还携带被指向元素的类型。`*bf16` 与 `*i64` 不只是整数地址的不同拼写：前者通常指向 BF16 数据，后者可能指向存放地址的整数表。LDG 权重入口常先从地址表中取出某个核的地址，再构造数据指针。不能把一个普通 BF16 Tensor 的地址强制解释为地址表，以为两者都只是“传一个指针”。参数传递的具体关系见[参数 ABI 与资源信息](#argument-resources)。

```python
core = tl.program_id(0)
lane = tl.arange(0, BLOCK)
offsets = core * BLOCK + lane
valid = offsets < numel
```

此处 core 与 numel 是标量，lane、offsets、valid 是静态长度 tile，BLOCK 在编译时已知。调试 shape 错误时先给每个中间值写出这类角色，比只看变量名称更有效。

### 类型提升与字面量

二元表达式先对齐形状，再确定共同的计算类型。两个 BF16 tile 的加、减、乘可能保持 BF16 类型；其中一侧明确转成 FP32 时，另一侧会参与 FP32 表达式。因此，“输入是 BF16，表达式看起来像普通 Python”并不意味着所有中间结果自动成为 FP32。

Python 字面量与运行时标量的角色也不同。一个浮点字面量参与 BF16 tile 运算时，未必要求把整个 tile 提升到 FP32；而显式的 FP32 张量或运行时参数会形成不同的类型关系。数值精度有要求时，应在代码中明确转换，避免让字面量推导承担契约。

```python
# Preserve an explicit FP32 compute boundary.
x32 = x.to(tl.float32)
y32 = x32 * 0.5 + bias32
y = y32.to(tl.bfloat16)
```

混合整数的有符号性同样重要。长度和正向偏移通常使用非负有符号整数；同时使用无符号上界与有符号偏移时，应先明确比较和转换规则。整数除法在设备张量表达式中朝零截断，负数行为与 Python 宿主的 `//` 不同；索引代码应避免依靠负数商余数实现环绕访问。

类型提升只解释表达式的数值语义，不会为任意类型组合创造硬件实现。基础数据路径保持 BF16 存储与 FP32 计算，可以使输入量化、转换、累计和输出四个阶段都便于核对。

### 显式表达转换边界

向量加法的数学定义为 $y_i=x_i+\alpha z_i$。当输入输出都是 BF16、计算使用 FP32 时，数值路径应写成下面的形式。

```python
x32 = x.to(tl.float32)
z32 = z.to(tl.float32)
y32 = x32 + alpha * z32
y = y32.to(tl.bfloat16)
```

这里的 `alpha` 是运行时 FP32 标量。若先把 `alpha * z32` 转成 BF16，再与 `x32` 相加，就多了一次低精度物化，得到的是另一套舍入边界。融合算子尤其需要保留这些边界：减少一次 kernel 启动，不应悄悄改变参考算法要求保留的 BF16 中间结果。

浮点向低精度类型转换时，Triton 的默认语义采用最近偶数舍入，即 RTNE。当前 Zeus 编译路径会为 FP32 到 BF16 的显式转换设置相应的舍入行为。与它对拍的参考计算也应采用相同语义，不能直接截取 FP32 的高 16 位来代替舍入。

普通向量转换路径对溢出、无穷大、次正规数和带符号零还存在平台条件。需要这些数值的算子应单独制定输入域和测试集，按具体转换路径核对结果；不能从有限正常数样例推导出完整 IEEE 754 行为。本书的入门加法首先使用有限、可精确表示的 BF16 值，使索引错误与舍入误差能够分开定位。

`bitcast=True` 表达的是保持位模式的类型重解释。它与数值转换有根本区别。例如，将 FP32 的位模式看成 32 位整数不会得到该浮点数的整数部分。只有在编写明确的编码处理逻辑时才使用 bitcast，并同时约束源类型、目标类型和位宽。

对于量化输出，至少需要分别定义缩放、舍入和饱和。设量化步骤为 $q=\operatorname{clip}(\operatorname{round}(x/s),q_{min},q_{max})$，单独一次 `.to(tl.int8)` 不应被理解为同时实现这三件事。整数缩窄、浮点转整数和浮点缩窄的规则不同；误差测试也应针对阈值两侧构造输入，而不是只抽取远离边界的随机数。

FP8 更需要完整格式标识。保存 FP8 数据时，宿主 Tensor dtype、指针签名、块访问格式、缩放张量和参考解码必须互相对应。读取一个字节的能力并不证明其位模式已经按正确的 FP8 格式解释。有关具体量化算子的格式与打包，采用该算子的专门接口，避免在基础 kernel 中自行拼接猜测的存储结构。

### 归约和矩阵乘的累加

对于长度为 $D$ 的一行，平方和为

$$
s=\sum_{j=0}^{D-1}x_j^2.
$$

BF16 输入应先转成 FP32 再平方和归约。否则每个乘积都可能先落到低精度类型中，改变累计误差。RMSNorm 中的平方和、均值和倒平方根也应在相同的 FP32 数值路径中比较。

```python
x32 = x.to(tl.float32)
sum_sq = tl.sum(x32 * x32, axis=0)
mean_sq = sum_sq / D
inv_rms = tl.rsqrt(mean_sq + eps)
```

矩阵乘的累加类型要显式表达。对于本书 BF16 Matmul，使用 FP32 accumulator，遍历 K 方向的 tile，最后再转换输出。INT8 矩阵乘还需要特别检查 `out_dtype`；Zeus 前端具备原生 INT8 输入、FP32 累加路径，显式要求 INT32 输出则属于另一套数值契约。不能把 GPU 上常见的 INT32 中间累加假设直接套到所有 Zeus 路径。

容差应服务于已经确定的计算语义。先固定输入量化、运算顺序、累加类型和输出舍入，再选择绝对误差与相对误差阈值。改变这些条件后仍沿用旧阈值，可能把语义变化当作普通误差接受。

## 张量、指针与访存 {#memory-api}

### 索引、广播与地址

`tl.arange` 生成 tile 内的逻辑索引。下面的片段适用于一维连续张量，`start` 是当前 tile 的全局起点，`BLOCK` 为编译期常量。

```python
offsets = start + tl.arange(0, BLOCK)
valid = offsets < numel
x = tl.load(in_ptr + offsets, mask=valid, other=0.0)
tl.store(out_ptr + offsets, x, mask=valid)
```

`offsets` 描述位置，`valid` 描述哪些位置属于有效输入。改变 `BLOCK` 会改变编译期张量形状，因此也会形成新的编译特化。本书的一维 `tl.arange` 示例使用 2 的幂长度，便于比较整块和尾块；Zeus 前端默认也允许非 2 次幂静态长度。静态长度条件仍然存在，不能把任意运行时长度直接放进 `tl.arange` 来生成动态形状 tile。

二维张量需要把逻辑坐标和存储步长同时写入地址表达式。若张量 $X$ 的元素步长为 $s_0,s_1$，则位置 $(r,c)$ 的元素偏移为 $rs_0+cs_1$。连续行优先存储只是 $s_1=1$ 的特例。

```python
rows = row_start + tl.arange(0, BLOCK_M)
cols = col_start + tl.arange(0, BLOCK_N)
offsets = rows[:, None] * stride_m
offsets += cols[None, :] * stride_n
valid = (rows[:, None] < M) & (cols[None, :] < N)
tile = tl.load(x_ptr + offsets, mask=valid, other=0.0)
```

`rows[:, None]` 的形状为 `[BLOCK_M, 1]`，`cols[None, :]` 的形状为 `[1, BLOCK_N]`。广播组合后得到二维地址 tile。这个表达式没有把原张量复制成二维索引数组；它描述的是编译器要生成的访问关系。

宿主端传入的是元素步长，不能未经换算传入字节步长。对于转置视图和切片，先打印实际 shape 与 stride，再确定 kernel 支持直接访问还是需要显式连续化。`reshape` 不等于任意数据重排，逻辑转置也不意味着搬运成本为零。

![行缩放与通道缩放的广播轴](assets/broadcast-axes.svg){#fig-broadcast-axes width=100%}

### 逻辑顺序与实际存储

一个 `[2,3]` 张量的行优先元素顺序可以写为 `[a,b,c,d,e,f]`。reshape 为 `[3,2]` 后各行为 `[a,b]`、`[c,d]`、`[e,f]`；原矩阵转置后各行为 `[a,d]`、`[b,e]`、`[c,f]`。两者输出 shape 相同，但元素对应关系不同。

对切片也应做同样的区分。若宿主选择原矩阵每隔一列的数据，切片的列 stride 可能为 2。把切片 shape 传给 kernel，却仍以 stride=1 读取，会得到相邻列而不是所选列。mask 只能限制访问范围，无法修复这个地址关系。

有两种清楚的宿主接口：一种限定连续输入，遇到非连续视图时明确拒绝或先连续化；另一种把受支持的元素 stride 作为参数传入，并测试这些布局。不能把“接受 Tensor 对象”写成默认接受其所有视图。输出如果要求连续，也应单独保证，而不是从输入的连续性推导。

### Mask 必须控制访存

无效地址必须在 `tl.load` 或 `tl.store` 上受到保护。`tl.where` 选择计算结果，不能代替访存 mask。下面的写法在执行选择前就需要取得 `tl.load` 的值，因此不适合保护越界地址。

```python
# Incorrect boundary protection: the load is unmasked.
value = tl.where(valid, tl.load(ptr + offsets), 0.0)
```

应将条件放在装载接口上。

```python
value = tl.load(ptr + offsets, mask=valid, other=0.0)
```

`other` 决定未装载位置进入后续计算的值。求和时常用零；最大值归约应使用与算法输入域一致的下界；最小值归约应使用相应上界。选择错误的填充值会使内存访问合法，但结果仍然错误。

Store 的 mask 只允许有效元素写回。填充位置可以参与中间向量计算，但不能因此扩大输出张量的逻辑范围。对于多个程序实例共同写一个输出，每个实例还需要有独立的输出区间；所有实例都满足 `offsets < numel`，仍然可能写入相同元素。

### 块指针

块指针将 base、完整 shape、stride、tile 起点和 tile shape 组织成一个对象，适合表达规则的块状访问。贯穿案例使用一维块指针，随后通过 `tl.advance` 推进 tile。

```python
x_block = tl.make_block_ptr(
    base=in_ptr,
    shape=(numel,),
    strides=(1,),
    offsets=(start,),
    block_shape=(BLOCK,),
    order=(0,),
)
x = tl.load(
    x_block,
    boundary_check=(0,),
    padding_option="zero",
)
next_block = tl.advance(x_block, (BLOCK,))
```

`shape` 是整个被访问张量的边界，`block_shape` 是当前 tile 的静态尺寸。`offsets` 表示 tile 的起点，`strides` 表示各维步长。`order` 描述块访问的维度顺序，不会替读者转置宿主数据或修正错误 stride。

块指针的边界通过 `boundary_check` 指定需要检查的维度。若尾块数据还要参与求和或矩阵乘，应明确使用零填充；不要依赖无效位置的未定义内容。块指针访问与普通指针张量的 mask 写法是两套接口，不能把两套边界参数混在同一次 load/store 中。

`tl.advance` 返回推进后的块指针，推进单位是该维的元素个数。对以 `[M,K]` 和 `[K,N]` 存储的普通矩阵，A 与 B 分别推进第二维和第一维；本书 Matmul 使用 LDG 权重的逻辑 `[N,K]` 视图，因此 A 与 W 都推进第二维，装载后再将 W 转置。先确定访存描述的轴，再决定推进方向。

![块指针的完整边界、访问窗口与推进](assets/block-pointer-window.svg){#fig-block-pointer-window width=100%}

块指针还可以用来逐项审查边界。先检查 `base` 是否指向预期张量起点，再把 offsets 代入 stride 得到第一个元素偏移，最后计算整个窗口中的最大有效偏移。以行优先二维张量为例，起点为 `(r,c)`，最大逻辑坐标为 `(r+BM-1,c+BN-1)`；该坐标可能超出完整 shape，因此应由相应维度的 boundary_check 保护。只检查最内维 N，不能保护已经越过 M 的整行。

对于已经指向某个子区域的 base，shape 和 offsets 必须采用同一个坐标系。若 base 已向后偏移一行，offsets 又包含这一行的偏移，就会把起点移动两次。这类错误往往在第一个 tile 正常、后续 tile 错误时暴露。

### 对齐、可读范围与输出归属

逻辑上的元素边界并不总等于一次物理搬运的边界。某些低精度、非对齐外部访问需要扩大到外部存储字，或者使用读改写处理部分字。相关分配范围和共享写入规则由宿主与运行时共同保证。

高级接口允许源文件声明 `arg_attrs`，例如完整外部字可读、外部字独占读改写、或运行时浮点缩窄安全。

::: {.caution}
`arg_attrs` 声明的是调用方已保证的条件。它不会自动扩充分配、补齐对齐或建立独占关系；宿主没有满足相应条件时，不能用该属性绕过编译诊断。
:::

从接口使用者角度，最稳妥的顺序是先使用独立、连续、配套分配器产生的输入输出，跑通整块和尾块；再逐项加入切片、原地更新或跨实例写回，并为每项新增条件补充测试。

## 算术、归约与矩阵乘 {#arithmetic-api}

### 逐元素运算

加、减、乘和比较通常作用于形状相容的 tile。广播必须能从维度关系解释，不能仅以编译成功作为形状正确的依据。例如 `[R,1]` 的行缩放与 `[1,D]` 的通道缩放都可以作用于 `[R,D]`，但两者的数学含义不同。

除法、指数、倒平方根和激活函数需要额外关注输入域与近似路径。对于 `tl.rsqrt(x)`，先检查算法是否保证输入为正；RMSNorm 的 `eps` 应在平方均值之后加入，不能改到平方之前。`eps` 的值和类型也是算子数值契约的一部分。

`tl.where` 适合表达已经安全计算出的分支结果。它不能保证只计算被选中的分支，也不能用来保护另一分支中的无效访存。对于涉及除零、非法对数或昂贵计算的分支，应结合输入域与控制流设计，而不是只在最后选择一个结果。

### 归约轴与填充

`tl.sum(x, axis=...)` 的 axis 是当前 tile 的逻辑轴。`[R,D]` tile 沿 `axis=1` 求和后得到 `[R]`，每个元素对应一行。之后使用 `[:, None]` 恢复广播维度，才能把每行的缩放应用到原 tile。

```python
x32 = x.to(tl.float32)
sum_sq = tl.sum(x32 * x32, axis=1)
inv_rms = tl.rsqrt(sum_sq / D + eps)
normalized = x32 * inv_rms[:, None]
```

如果实际行宽 $D$ 小于 tile 宽度，尾部补零可以保持平方和不变，但均值的分母必须仍是 $D$。例如实际宽度为 96、tile 宽度为 128，求和覆盖 128 个槽位，其中后 32 个为零，除数仍为 96。把除数写成 128 会稳定地产生错误缩放。

当一行分成多个 tile，必须先在 FP32 中累计各 tile 的局部平方和，再计算整行的倒平方根。每个 tile 各自归一化会改变算法。若一行由多个核处理，还需要明确局部统计量的汇总与结果广播；本书的首个 RMSNorm 实例采用整行归属单核，保持输出与统计量的归属一致。

### 使用普通 `tl.dot`

对二维输入，`tl.dot(a,b)` 的收缩维度由 A 的最后一维与 B 的倒数第二维给出。若 A tile 为 `[BM,BK]`，逻辑 B tile 为 `[BK,BN]`，结果 tile 为 `[BM,BN]`。M、N 决定输出归属，K 决定累加长度。本书 Matmul 的 LDG 权重按逻辑 `[N,K]` 描述，实际字节遵循 LDG pack；装载后再转成 `tl.dot` 所需的逻辑视图。

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
for k_tile in range(tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_block, boundary_check=(0, 1),
                padding_option="zero")
    b_nk = tl.load(b_block, memory_type="weight",
                   boundary_check=(0, 1), padding_option="zero")
    acc += tl.dot(a, tl.trans(b_nk))
    a_block = tl.advance(a_block, (0, BLOCK_K))
    b_block = tl.advance(b_block, (0, BLOCK_K))
```

这段代码强调输入块、K 循环和 accumulator 的生命周期。`b_block` 描述 LDG 权重的逻辑 `[N,K]` 视图，因此沿 K 推进时偏移在第二维。权重地址表、A/B 块的初始化和输出边界见[Matmul 实例](#matmul-example)。K 尾块两侧都必须按零扩展；否则无效的乘积会进入有效输出。`memory_type="weight"` 选择 LDG 相关的权重访问路径，是保留的历史接口命名。LDG 是承载权重与 KV cache 等数据的 3D DRAM，不因接口字符串而被限制为某一种数据用途。调用方必须提供相应存储对象，不能把普通张量指针无条件传入。

使用普通 `tl.dot`，由编译器根据 shape 和 dtype 选择 PE 模式及 minitile。Kernel 作者应控制的是算法分块、操作数布局、累加类型和可复用的数据；不要把 PE 内部模式参数固化进通用算子。改变 K 的分块还可能改变浮点累加次序，应在性能对照中同时复核误差。

GPU 上的 TF32、Tensor Core 配置和 warp 数不能作为 Zeus 矩阵乘的通用参数。面向 BF16 的例子保留普通 `tl.dot` 默认路径；需要额外精度选项时，先核对实际后端接受的值。

## 控制流、特化与 Zeus 扩展 {#control-and-extensions}

### 运行时参数与 `tl.constexpr`

`tl.constexpr` 在编译期确定。它适合描述 tile shape、静态算法选择和核数量等影响代码结构的值。普通标量参数在每次 launch 时传入，适合描述有效长度或运行时缩放。

贯穿案例把 `BLOCK` 与 `CORE_NUM` 放在编译常量中，把 `numel` 与 `alpha` 放在运行时参数中。于是同一份 `BLOCK=1024, CORE_NUM=2` 的特化可以处理多个合法长度；更换 `BLOCK` 或 `CORE_NUM` 则需要另一份特化。若把每个输入长度都声明成 constexpr，AOT 包的条目数和首次编译成本都会随长度组合增加。

编译期条件可以移除不需要的分支；运行时条件则需要设备执行。循环的 trip count 可以由运行时长度计算，但循环体内 tile 的形状仍然固定。设计动态形状 kernel 时，应把“循环次数可变”和“tile 形状可变”分开理解。

`tl.static_assert` 适合表达编译期形状关系，例如 tile 参数必须为正。宿主端仍需检查每次调用的 dtype、device、长度、stride 和输出容量。编译期断言无法验证后来传入的每个运行时值。

### 条件分支与值的合流

编译期条件适合在一个 kernel 中保留少量互斥实现，例如是否添加 bias。条件为 constexpr 时，未选择的分支不会成为该特化的设备工作。运行时标量条件则会形成设备控制流；两条路径重新合流时，后续需要使用的值必须在所有路径上有定义，并具有相容的类型与形状。

```python
if HAS_BIAS:
    value = x32 + bias32
else:
    value = x32
```

这里 `HAS_BIAS` 为 constexpr。若条件来自运行时参数，不能让一个分支产生 `[BLOCK]` 而另一个分支产生 `[2*BLOCK]`，再把它们当作同一个 value 使用。动态控制流决定执行哪些操作，不改变一个已编译 tile 的基本类型。

逐元素条件是另一种情况。`offsets < N` 产生一个布尔 tile，通常应该用于 mask 或 where，而不是作为单个 if 的条件。只有一个分支有效时，where 仍不能保证跳过另一分支的非法访存；条件选值与条件执行必须分别设计。

### 设备辅助函数与模块边界

较复杂 kernel 可以拆成多个 `@triton.jit` 辅助函数，复用数值表达式与静态规则。辅助函数接受 tile 或编译期参数，并返回计算值；宿主分配、文件读取、普通 PyTorch 操作和运行时初始化仍留在 kernel 外。

```python
@triton.jit
def affine_fp32(x, scale, bias):
    return x.to(tl.float32) * scale + bias
```

调用辅助函数不会产生第二次宿主 kernel launch。它与把两个独立 kernel 在 Python 中连续调用不同，后者存在两个独立提交及其各自的输出边界。封装时应明确这一点，避免在融合说明中混淆源代码拆分与设备任务拆分。

辅助函数的输入输出仍需写清 shape、dtype 与数值范围。若函数内完成 BF16 缩窄，调用者再提升回 FP32 也无法消除该处舍入。将转换留在明确的算子边界通常更容易审查；确需低精度中间值时，应将其作为函数契约的一部分。

### 执行单元提示

Zeus 专属接口位于后端语言模块。

```python
from triton.backends.triton_shared import language as ztl
```

平台扩展宜集中封装，保留清晰的调用契约。普通向量、归约和矩阵乘优先使用标准 Triton 表达式，先建立正确性基线，再按[性能分析与调优](#performance)评估扩展。

| 接口 | 作用 | 使用要点 |
|---|---|---|
| `vp_hint(value, num_vp=...)` | 请求一个逻辑 VP 操作的拆分策略 | 继续使用返回值；整数为精确要求 |
| `dt_hint(value, num_dt=...)` | 请求一个逻辑 DT 操作的拆分策略 | 合法性取决于实际操作和布局 |
| `parallel_executor_hint()` | 启用独立操作的并行分析 | 仅限 kernel 顶层 |
| `double_buffer_hint()` | 请求自动双缓冲分析 | 实验性且不完善，暂不推荐使用 |

Table: 执行单元扩展接口

局部 `vp_hint` 和 `dt_hint` 接受整数 1–4 或字符串 `"auto"`。整数表示精确要求，不能满足时应诊断失败；`"auto"` 允许按该操作的条件选择。整 kernel 的 `--num-vp` 或 `--num-dt` 是另一层请求，不能用它推断每一个操作最终都占用了相同数量的执行单元。

Multi-Executor 把一个逻辑操作拆成多个 sibling，Parallel-Executor 把多个独立逻辑操作分配到不同 lane。两者的工作分解不同。优先评估独立工作所带来的 Parallel-Executor 收益；Multi-VP 保持默认关闭，只有同条件 A/B 证明整体收益时再采用。

::: {.note}
`double_buffer_hint()` 与 `--enable-double-buffer` 对应实验性的自动双缓冲功能，目前不完善，暂不推荐使用。需要存算重叠时，优先在 Triton kernel 中显式组织预取、两组工作区和缓冲复用，方法见[性能分析与调优](#performance)。
:::

![拆分一个操作与并行多个独立操作](assets/executor-models.svg){#fig-executor-models width=100%}

Hint 不改变数学结果，也不能取消真实的数据依赖。若后续操作没有使用 hint 的返回值，标记可能在优化过程中消失。确认扩展是否生效，应查看编译产物与执行 trace，而不是只看源代码是否出现函数名。

### 专用数值接口与提前退出

Zeus 扩展包含分段线性函数、归一化倒数与倒平方根、查表指数等专用接口。它们有各自的输入域和系数格式；调用方应同时交付系数生成规则、边界测试和误差结果。具体调用方式见 [Zeus 编程接口](#api-zeus-extensions)。

`early_exit_if` 可在 kernel 顶层根据单元素 FP32 标志跳过后续工作。采用默认策略时，推荐只传 `0.0` 或 `1.0`：前者继续，后者退出。其整数转换与作用域规则见[设备端提前退出](#api-early-exit)。

::: {.caution}
提前退出保留尚未写出的输出，必要时须先初始化输出。Triton 解释器不模拟该退出行为，不能用解释器结果验证设备控制流。
:::
