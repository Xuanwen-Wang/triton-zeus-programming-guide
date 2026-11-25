# 语言 API 参考 {#api-reference}

本章按开发顺序组织接口：先建立索引与 tile，再访问存储、变换形状、计算和归约，最后组织控制流与平台提示。查阅一个接口时，应同时确定输入 dtype、静态 tile shape、运行时有效范围以及输出归属。仅有函数名不足以描述一个可复用的调用契约。

接口条目使用统一体例：**用途、调用形式、参数与语义、Zeus 使用约束、示例、常见错误与关联操作**。调用形式省略编译器内部的 `_semantic` 等参数；除专门标出的高级参数外，保留真实的公开名称与默认值。片段运行在 `@triton.jit` 函数体内，`tl` 指 `triton.language`；大写 tile 尺寸为 `tl.constexpr`，指针、长度和步长来自该函数的参数。完整宿主调用见[算子开发实例](#examples)。

本章的“拒绝”表示该调用违反明确的接口条件，例如块指针同时传入 `mask`；“未列入基础使用范围”表示这里没有给出它的设备契约。两者不能混同。原子操作、GPU 专属异步指令、内联汇编和通信接口不属于本章的基础工作集；不能因为上游 Triton 中存在同名符号而直接采用。版本、能力范围和验证层级集中见[附录](#appendices)，后续条目只说明影响写法的条件。

## 索引与张量创建 {#api-creation}

### program_id：取得程序实例编号 {#api-program-id}

**用途。** 为当前实例计算自己的输入和输出区间。编号用于任务分工，不能用作线程内的元素编号。

```python
tl.program_id(axis)
```

**参数与语义。** `axis` 是编译期整数，前端接受 0、1、2。结果为该轴的标量实例编号。一个实例中的所有 tile 元素共享此编号，tile 内元素索引通常由 `tl.arange` 生成。

**Zeus 使用约束。** 基础执行路径使用一维核数 grid 和 `axis=0`。任务数大于核数时，由每个实例循环处理多个 tile。宿主传入的核数、`CORE_NUM` 和编译产物核数必须一致；增大 grid 不会自动产生 GPU 式的海量 block 调度。多维参数的前端形式不等于已经定义了相应的物理核映射。

```python
core = tl.program_id(0)
tiles = tl.cdiv(N, BLOCK)
for tile in range(core, tiles, CORE_NUM):
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets, offsets < N, other=0.0)
    tl.store(y_ptr + offsets, x, offsets < N)
```

**常见错误与关联操作。** 若所有实例都以零为 tile 起点，会重复写相同区间。若只写 `core * BLOCK` 而不循环，超过 `CORE_NUM * BLOCK` 的尾部不会处理。有关均衡分块与输出归属，见[编程模型](#programming-model)。

### arange：创建静态整数索引 {#api-arange}

**用途。** 建立 tile 内的连续坐标，用于地址、mask 和广播。

```python
tl.arange(start, end)
```

**参数与语义。** 返回半开区间 `[start,end)` 的一维 INT32 张量，形状为 `[end-start]`。两个端点必须在编译期确定，且 `end > start`；端点还须通过前端的 INT32 范围检查。`arange` 没有步长参数，步长通过结果乘法表达。

**Zeus 使用约束。** Zeus 前端默认允许非 2 次幂长度。需要启用 2 次幂兼容检查时，在编译进程环境中设置：

```bash
export TRITON_BLOCK_SHAPE_POWER_OF_TWO=1
```

允许构造某个静态形状，不代表所有后续算子均接受该布局或容量。本书基础示例采用规则尺寸，改变为 96、192 等长度时仍需检查完整计算链。

```python
lane = tl.arange(0, BLOCK)
even = 2 * lane
offsets = start + even
valid = offsets < N
```

`start` 是运行时起点时，应把它加在 `arange` 的结果上，而不是写进 `tl.arange(start, start + BLOCK)` 的端点。

**常见错误与关联操作。** `tl.arange(0, N)` 在 `N` 是运行时参数时不能创建动态 tile。正确做法是保持 `BLOCK` 静态，再用 `offsets < N` 限定有效范围。二维坐标由两条一维索引配合 `expand_dims` 组合。

### full 与 zeros：建立常量 tile {#api-full}

**用途。** 初始化 accumulator、填充值或固定系数。

```python
tl.full(shape, value, dtype)
tl.zeros(shape, dtype)
```

**参数与语义。** `shape` 是编译期整数元组，`dtype` 明确元素类型。`full` 把标量扩展到整个形状；`zeros` 等价于使用零初始化。返回的是 kernel 中的张量值，不是宿主设备内存分配接口。

**Zeus 使用约束。** 初始化大小要计入 L2 Buffer 工作集，尤其是循环外持续存活的矩阵 accumulator。基础用法采用常量 `value` 和 FP32 累加。若填充值来自运行时标量，需同时满足相应的标量类型与转换输入域，不能把常量初始化的行为直接外推到任意运行时值。

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
bias = tl.full((BLOCK_N,), 0.5, tl.float32)
result = acc + bias[None, :]
```

**常见错误与关联操作。** 在 K 循环内部重新置零会丢失前面所有块的乘积。`zeros` 也不会给 `out_ptr` 指向的设备空间自动写零；只有 `tl.store` 才产生相应输出。需要保留原输出的累加，应先定义读写和别名契约。

### cdiv：计算正整数块数 {#api-cdiv}

**用途。** 计算覆盖有效长度所需的 tile 数。

```python
tl.cdiv(x, div)
```

**参数与语义。** 实现为 `(x + div - 1) // div`。对于非负整数 `x` 和正整数 `div`，结果为向上取整的商；标量得到标量，张量表达式按广播规则处理。

**Zeus 使用约束。** 分块长度通常使用非负 i32 标量，tile 大小为正 constexpr。宿主应排除负长度、零除数以及中间加法溢出。更大输入应先确定地址与算术位宽，不能只扩大分配后继续沿用 i32 算术。

```python
tiles = tl.cdiv(N, BLOCK)
base = tiles // CORE_NUM
extra = tiles % CORE_NUM
owned = base + (core < extra)
first = core * base + tl.minimum(core, extra)
```

**常见错误与关联操作。** `N // BLOCK` 会漏掉非整块尾部；`cdiv` 只得到块数，不能保护最后一个 tile 的访存。仍须配合 `load/store` 的 mask 或块指针边界检查。

## 指针与访存 {#api-memory}

### load：从指针装载元素 {#api-load}

**用途。** 将普通设备数据或指定存储路径中的 tile 读入计算表达式。

```python
tl.load(
    pointer, mask=None, other=None, boundary_check=(),
    padding_option="", cache_modifier="",
    eviction_policy="", volatile=False,
    memory_type="default", weight_dtype=None,
    ugroup=None, page_size=None, uscale=None,
)
```

**参数与语义。** `pointer` 可以是标量指针、指针张量或块指针。标量指针返回标量；指针张量返回相同广播形状的值张量；块指针返回 `block_shape` 对应的 tile。结果元素类型由指针决定。

普通指针使用 `mask` 指定可访问元素，`other` 转换为指针元素类型后填入未装载位置。指定 `other` 时必须同时指定 mask。块指针使用 `boundary_check` 与 `padding_option`；此时不能再传 mask 或 other。空填充选项不会给越界位置承诺一个可用于计算的值。

**Zeus 使用约束。** 普通 GDG 数据采用 `memory_type="default"`。`"weight"` 是保留的接口字符串，用于 LDG 相关的权重访问路径；名称来自历史接口，不是将 LDG 限定为只存权重。`weight_dtype`、`ugroup`、`page_size`、`uscale` 参与专用格式、分组或量化路径，不应在普通 BF16 load 中试填。必须先有匹配的布局和资源封装。GPU 缓存修饰符及 `volatile` 也不能代替 Zeus 的依赖和同步规则。

专用 K/V 模式的页长、索引表与 payload 布局见[LDG 数据布局与访问](#ldg-memory)。普通 `weight` 布局承载 KV 与专用 `kcache/vcache` 是不同契约，不能交叉使用其描述符。

独立 UScale 使用指向 16 KiB 对齐 LDG 页首的标量 `*bf16` 指针，且必须配合 `memory_type="weight"` 与显式 `ugroup`。整数地址、块指针或普通 GDG 系数张量不能直接替代它。相邻与独立 scale 的组织见[量化矩阵乘与 UScale](#quantized-matmul)。

```python
offsets = start + tl.arange(0, BLOCK)
valid = offsets < N
x = tl.load(x_ptr + offsets, mask=valid, other=0.0)
x32 = x.to(tl.float32)
```

**常见错误与关联操作。** `tl.where(valid, tl.load(...), 0)` 会先形成装载，不能保护越界。mask 为假且未给出 other 的值，不应参与归约。若一个指针张量重复指向同一地址，读取的逻辑语义仍需与后端访问模式相容；不能把任意 gather 当成规则块访问。关联接口为 `make_block_ptr`、`advance` 和 `store`。

### store：向有效输出区域写回 {#api-store}

**用途。** 把计算值保存到普通输出或具有明确契约的存储路径。

```python
tl.store(
    pointer, value, mask=None, boundary_check=(),
    cache_modifier="", eviction_policy="",
    memory_type="default", store_mode="default",
    blocksize=None,
)
```

**参数与语义。** 普通指针张量把 `value` 和 mask 广播到指针的形状，并把值转换到指针元素类型；返回值不代表保存后的张量。块指针使用 `boundary_check`，不接受 mask。块写回的 tile shape 与元素类型必须匹配，因此基础写法在 store 前显式完成输出转换。

**Zeus 使用约束。** 每个程序实例应拥有独立写入区间。对齐和物理写粒度还可能使逻辑相邻元素共享一个外部存储字；低精度的尾部和并发原地更新必须服从分配与独占写入契约。`store_mode` 和 `blocksize` 属于专用路径，普通输出保持默认值。

写入 LDG 时，LW 目的格式必须与后续读取方式一致；具体 K/V payload 的追加写入见[LDG 数据布局与访问](#ldg-memory)。改变 store 的模式字符串不会使旧分配自动变成新布局。

```python
y = y32.to(tl.bfloat16)
tl.store(out_ptr + offsets, y, mask=offsets < N)
```

**常见错误与关联操作。** 两个实例即使都满足 `offsets < N`，仍可能覆盖同一输出。输出 mask 也不能补救此前的输入越界。对块指针写 BF16 输出前，先把 FP32 accumulator 转成 BF16，避免元素类型不匹配。

### make_block_ptr：描述规则的矩形访问 {#api-block-ptr}

**用途。** 将张量边界、步长、tile 起点和静态尺寸组合为块指针，使循环中的装载和推进更清楚。

```python
tl.make_block_ptr(
    base, shape, strides, offsets, block_shape, order,
)
```

**参数与语义。** `base` 是普通元素指针。`shape` 为完整张量的各维长度，`strides` 为元素步长，`offsets` 为 tile 起点，`block_shape` 为 tile 的静态形状，`order` 为维度编号的一个完整排列。五组维度信息必须具有相同长度。前端以 i64 处理 shape/stride，以 i32 处理偏移；静态 block shape 也须落在合法整数范围。

**Zeus 使用约束。** 优先使用具有明确矩形边界的连续或规则 stride 数据。`order` 是访问描述，不会把输入重新排布成所写顺序，也不会消除布局转换代价。基础矩阵例采用二维块，LDG 权重使用逻辑 `[N,K]` 视图；其字节实际按 LDG 格式 pack，不能按普通行主序解释。

```python
block = tl.make_block_ptr(
    base=x_ptr, shape=(M, N), strides=(N, 1),
    offsets=(row0, col0),
    block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
)
x = tl.load(block, boundary_check=(0, 1),
            padding_option="zero")
```

**常见错误与关联操作。** 将 `shape` 写成 tile 大小，会使第二个及后续 tile 被误判越界。将字节数作为 stride，则会按元素大小重复缩放偏移。`order` 有重复轴或缺轴时不合法。尾部检查由使用该块的 load/store 完成，构造块指针本身不搬运数据。

### advance：沿张量维度推进块指针 {#api-advance}

**用途。** 在保持 shape、stride 和 tile shape 不变时移动访问窗口。

```python
tl.advance(base, offsets)
```

**参数与语义。** `base` 是块指针；`offsets` 是与块 rank 一致的偏移元组。每项表示沿对应逻辑维度推进多少个元素。返回新块指针，原变量不会自动改变。

**Zeus 使用约束。** 推进后的块仍受完整 shape 约束。Matmul 的 A `[M,K]` 与 LDG 操作数的逻辑 W `[N,K]` 视图都沿第二维推进 K；若另一算法直接使用 B `[K,N]` 的存储描述，其推进方向会不同。先写清访存描述的轴，再决定推进向量。

```python
for k in range(tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_block, boundary_check=(0, 1),
                padding_option="zero")
    w = tl.load(w_block, boundary_check=(0, 1),
                padding_option="zero", memory_type="weight")
    acc += tl.dot(a, tl.trans(w))
    a_block = tl.advance(a_block, (0, BLOCK_K))
    w_block = tl.advance(w_block, (0, BLOCK_K))
```

**常见错误与关联操作。** 只调用 `tl.advance(block, ...)` 而不接收返回值，会反复读取相同窗口。以上轮已经推进的指针为基准，再增加 `k * BLOCK_K` 会累加错误；若始终以固定初始模板为基准，该绝对偏移写法合法。顺序遍历可选择每轮推进一个 tile，或从固定模板定位当前 tile，不能混用两种基准。

## 形状与数据视图 {#api-shape}

### broadcast 与 broadcast_to：扩展单例维度 {#api-broadcast}

**用途。** 将行系数、通道系数或标量与目标 tile 对齐。

```python
tl.broadcast(input, other)
tl.broadcast_to(input, *shape)
```

**参数与语义。** `broadcast` 返回一对已对齐的张量。`broadcast_to` 返回目标静态形状的一个张量，形状可写为元组或多个整数。两个维度必须相等或其中之一为 1；rank 不同时在较小 rank 的左侧补单例维度。

**Zeus 使用约束。** 广播描述元素关系，不保证物理零开销。布局变化和后继执行单元可能需要物化或搬运。行/通道关系应在源码中显式保留，避免相同长度掩盖轴写错的问题。

```python
row_scale = tl.reshape(row_scale, (BLOCK_M, 1))
scaled = x * tl.broadcast_to(
    row_scale, (BLOCK_M, BLOCK_N)
)
```

**常见错误与关联操作。** `[M]` 与 `[M,N]` 直接广播时，长度 M 会尝试匹配最后一维 N，并不是按行缩放。正确的行缩放使用 `[M,1]`。当 M 恰好等于 N 时，这类错误尤其容易隐藏，测试应包含不相等的两维。

### expand_dims 与 None 索引：插入单例轴 {#api-expand-dims}

**用途。** 为广播明确行轴与列轴，不改变元素总数。

```python
tl.expand_dims(input, axis)
```

**参数与语义。** 在指定位置插入长度为 1 的维度；前端支持单个轴或轴序列。对一维 `r`，`r[:, None]` 形成 `[R,1]`；对一维 `c`，`c[None,:]` 形成 `[1,C]`。

**Zeus 使用约束。** 这是静态形状操作，不能把运行时整数用作新轴的尺寸。它不负责分配宿主内存；后端可能根据后续广播与布局消除或物化相关视图。

```python
rows = tl.arange(0, BLOCK_M)
cols = tl.arange(0, BLOCK_N)
offsets = rows[:, None] * stride_m
offsets = offsets + cols[None, :] * stride_n
```

**常见错误与关联操作。** 插入了错误轴仍可能得到可广播的张量，但生成错误地址。建议用一个 `[2,3]` 的逻辑矩阵逐个验证坐标，再扩大 tile。关联接口为 `broadcast_to`、`reshape` 与 `sum(keep_dims=True)`。

### reshape 与 view：改变静态形状 {#api-reshape}

**用途。** 合并或拆分逻辑维度，为后续运算组织 tile。

```python
tl.reshape(input, *shape, can_reorder=False)
tl.view(input, *shape)
```

**参数与语义。** 目标形状必须在编译期确定且元素总数相同。`reshape` 默认保持元素顺序；`can_reorder=True` 允许更宽松的重排，不能用于依赖固定位置语义的索引。已弃用的 `view` 等价于允许重排的 reshape，因此不应把它当作默认 reshape 的同义写法；新代码明确使用 reshape。

**Zeus 使用约束。** 逻辑等元素数还不足以保证物理布局可直接复用。对转置、切片或非连续中间结果进行 reshape，可能需要实际搬运；不满足可证明的布局条件时应调整访问或显式重排，不能仅切换接口名绕过诊断。

```python
flat = tl.arange(0, BLOCK_M * BLOCK_N)
matrix = tl.reshape(flat, (BLOCK_M, BLOCK_N))
row_sum = tl.sum(matrix.to(tl.float32), axis=1)
```

**常见错误与关联操作。** 把 `[M,N]` reshape 成 `[N,M]` 不等于转置。对于 `[0,1,2,3,4,5]`，reshape 为 `[2,3]` 的第一行为 `[0,1,2]`，再 reshape 为 `[3,2]` 的第一行是 `[0,1]`；转置后的第一行则是 `[0,3]`。需要轴交换时使用 `trans` 或 `permute`。

### trans 与 permute：按轴置换元素视图 {#api-trans}

**用途。** 交换矩阵的逻辑行列，或对更高 rank 的 tile 明确排列各轴。

```python
tl.trans(input, *dims)
tl.permute(input, *dims)
```

**参数与语义。** 维度排列可作为元组或逐项参数。`trans` 不提供 dims 时交换最后两个轴；`permute` 显式给出完整排列。输出 shape 按同一排列重排，元素之间的对应关系由轴变换确定。

**Zeus 使用约束。** Matmul 中 LDG 权重的逻辑 W `[N,K]` tile 使用 `tl.trans` 得到 B `[K,N]`。这不意味着宿主权重 buffer 变成了普通 `[K,N]` 张量，也不能据此取消 LDG pack 与资源要求。一般转置的物理实现由布局与消费者决定，不能假定总是免费。

```python
w_nk = tl.load(w_block, memory_type="weight",
               boundary_check=(0, 1),
               padding_option="zero")
b_kn = tl.trans(w_nk)
product = tl.dot(a_mk, b_kn)
```

**常见错误与关联操作。** 同时在地址构造中交换 stride、又对装载结果转置，可能把数据转置两次。先区分“源存储中的坐标”和“消费者看到的坐标”，再决定变换放在哪一层。

## 类型、表达式与选择 {#api-elementwise}

### cast 与 tensor.to：转换数值或位模式 {#api-cast}

**用途。** 显式确定计算精度、输出舍入或编码解释。

```python
tl.cast(input, dtype, fp_downcast_rounding=None,
        bitcast=False)
input.to(dtype, fp_downcast_rounding=None, bitcast=False)
```

**参数与语义。** 普通转换保持 shape 并改变元素 dtype。浮点缩窄默认采用 `rtne`；`rtz` 表达朝零舍入。`fp_downcast_rounding` 只能用于浮点缩窄，不能附加到 FP32→INT32 或 BF16→FP32。`bitcast=True` 保留位模式，非指针类型必须具有相同位宽。

**Zeus 使用约束。** 基础数值路径为 BF16 装载、FP32 计算、BF16 写回，最后一次缩窄采用默认 RTNE。显式 RTZ、特殊值和其他 dtype 的组合需满足对应转换路径，不能仅因前端接受参数就改变标准样例的数值契约。位重解释也不能制造有效的 LDG 分配或权重地址表。

```python
x32 = x.to(tl.float32)
y32 = x32 * scale + bias
y = y32.to(tl.bfloat16)
```

**常见错误与关联操作。** FP32 高 16 位截取是 BF16 截断参考，不是 RTNE。将负浮点转换到整数也不等价于向下取整；编写量化时先定义舍入与饱和，再选接口。转换位置应和参考实现逐项对应。

### 算术运算符：逐元素计算与整数商余数 {#api-arithmetic}

**用途。** 组合标准表达式，计算数值结果与索引。

```python
a + b
a - b
a * b
a / b
a // b
a % b
```

**参数与语义。** 二元运算先做形状广播与类型提升。`/` 表达除法；两个整数的 `//` 在 Triton 张量表达式中使用向零截断的整数商，`%` 使用相应余数。负数时这与 Python 宿主的向下取整规则不同。比如设备整数 `-7 // 3` 为 `-2`，余数为 `-1`；宿主 Python 则为 `-3` 和 `2`。

**Zeus 使用约束。** 数据路径优先显式提升为 FP32，索引保持非负整数，避免把宿主的负数算术约定带进地址计算。分母应排除零，索引加乘不得溢出。混合有符号与无符号商余数被前端拒绝，应先转换到明确的共同类型。

```python
x32 = x.to(tl.float32)
y32 = z.to(tl.float32)
value = x32 + alpha * y32
tile = index // BLOCK
lane = index % BLOCK
```

**常见错误与关联操作。** BF16 tile 加一个 Python `0.5` 不一定提升到 FP32；常量的类型提升规则与运行时 FP32 标量不同。要求 FP32 中间精度时显式 `.to(tl.float32)`。参见[数据类型与累加精度](#types-and-precision)。

### 比较与逻辑组合：构造布尔 tile {#api-compare}

**用途。** 创建边界 mask 和逐元素条件。

```python
a < b
a <= b
a == b
a != b
condition_a & condition_b
condition_a | condition_b
~condition_a
```

**参数与语义。** 比较按广播后的 shape 产生布尔结果。布尔张量使用 `&`、`|` 和 `~` 逐元素组合。组合多个比较时，为每个比较加括号，避免 Python 运算符优先级改变表达式。

**Zeus 使用约束。** 访存边界通常由整数索引与长度比较形成。不要把一个多元素布尔 tile 当成宿主 `bool` 或单一运行时分支条件；它属于逐元素 mask 或 `where` 的条件。

```python
valid = (rows[:, None] < M) & (cols[None, :] < N)
x = tl.load(x_ptr + offsets, mask=valid, other=0.0)
```

**常见错误与关联操作。** `rows < M & cols < N` 并不是所需的两个比较。Python 的 `and/or/not` 也不应代替逐元素逻辑。涉及 NaN 时，浮点比较还必须按算法约定处理；不能把有限输入的比较行为外推到特殊值。

### where：逐元素选择已经合法的值 {#api-where}

**用途。** 在两个形状相容的计算结果间逐元素选择。

```python
tl.where(condition, x, y)
```

**参数与语义。** 条件应为布尔值或布尔 tile，x/y 经类型提升和广播后形成同一结果形状。条件为真取 x，为假取 y。该操作不提供 Python 条件表达式那样的分支短路保证。

**Zeus 使用约束。** 输入值必须已经可以安全计算；无效访存仍需由 load/store 的 mask 保护。若一个分支包含除法，应先构造安全分母，再选择结果，而不是期待 where 消除非法分支。

```python
safe_den = tl.where(den != 0.0, den, 1.0)
quotient = num / safe_den
result = tl.where(den != 0.0, quotient, 0.0)
```

**常见错误与关联操作。** `tl.where(mask, 1.0 / den, 0.0)` 不能证明除零从未发生。同理，where 不能替代输出初始化、核间同步或资源所有权检查。

### maximum 与 minimum：逐元素截限 {#api-minimum-maximum}

**用途。** 计算逐元素上下界，或对有限数据实现 clamp。

```python
tl.maximum(x, y, propagate_nan=tl.PropagateNan.NONE)
tl.minimum(x, y, propagate_nan=tl.PropagateNan.NONE)
```

**参数与语义。** 两个输入广播到共同形状并做类型提升，每个位置返回较大或较小值。它们保持 tile 的元素数，与沿一个轴缩减元素数的 `tl.max/tl.min` 不同。NaN 传播选项属于数值契约。

**Zeus 使用约束。** 本书的截限用法针对有限 FP32 值；对于 NaN、带符号零及精确传播模式，须使用相应转换/比较路径的定义。整数形式可用于非负块数和分工边界，但不能借此替代索引溢出检查。

```python
x32 = x.to(tl.float32)
clipped = tl.minimum(tl.maximum(x32, -1.0), 1.0)
```

**常见错误与关联操作。** `tl.maximum(x, 0)` 不是一行的最大值；它只是逐元素截限。若要求一个 tile 的最大值，使用归约 `tl.max(x, axis=...)`。

### abs：逐元素绝对值 {#api-abs}

**用途。** 构造绝对误差、幅度或最大绝对值归约。

```python
tl.abs(x)
```

**参数与语义。** 返回与输入相同形状的绝对值。浮点路径用于数值幅度；整数最小负数在相同位宽中没有对应的正数，需要单独定义溢出处理。

**Zeus 使用约束。** 常见用法是有限 FP32 tile。BF16 输入在误差统计或缩放计算前先提升为 FP32。计算 INT8/INT32 幅度时，不应直接假设最小负数可在原类型中表示为正。

```python
x32 = x.to(tl.float32)
amax = tl.max(tl.abs(x32), axis=0)
safe_amax = tl.maximum(amax, 1.0e-6)
```

**常见错误与关联操作。** 对误差做 abs 后还需要说明如何聚合、除以什么参考量。一个较小平均误差可能掩盖极少数大误差，精度诊断同时关注最大值和分布。

## 归约、矩阵乘与数学函数 {#api-reduction-math}

### sum：沿静态轴求和 {#api-sum}

**用途。** 计算行和、平方和及分块的局部统计量。

```python
tl.sum(input, axis=None, keep_dims=False, dtype=None)
```

**参数与语义。** `axis` 指向当前 tile 的逻辑轴；`None` 表示归约所有维度。`keep_dims=True` 保留长度为 1 的被归约轴。指定 dtype 时先转换输入再求和。默认将不足 32 位的整数提升到相同有符号性的 32 位整数；浮点输入需要显式选择计算精度，不能依赖自动提升 BF16 或 FP16。

**Zeus 使用约束。** 基础归约显式使用 FP32，归约轴和 tile shape 静态可知。它只归约当前 tile，既不跨多个 loop tile 自动累计，也不跨程序实例通信。较长行要在算法中定义块间累计；跨核归约另需完整同步与汇总路径。

```python
x32 = x.to(tl.float32)
sum_sq = tl.sum(x32 * x32, axis=1, keep_dims=True)
mean_sq = sum_sq / width
y32 = x32 * tl.rsqrt(mean_sq + eps)
```

**常见错误与关联操作。** 输入尾部补零可以保持求和，但均值仍除以有效 `width`，不能除以填充后的 tile 宽度。加法树改变时浮点结合顺序也可能改变，因此参考计算需匹配累加类型与容差。

### max：沿轴求最大值 {#api-max}

**用途。** 计算每行最大值或最大绝对值，为稳定指数或量化缩放提供统计量。

```python
tl.max(
    input, axis=None, return_indices=False,
    return_indices_tie_break_left=True, keep_dims=False,
)
```

**参数与语义。** 默认只返回值。结果 shape 移除归约轴，启用 keep_dims 时保留单例轴。若同时需要值和索引，可设 `return_indices=True`；平局结果由 tie-break 参数决定。

**Zeus 使用约束。** 基础条目采用 FP32 的值归约与显式 axis。索引返回涉及多结果归约、平局及 NaN 规则，不在本条目的值归约实例范围。mask 无效位置应填入不可能胜出的下界，且至少存在一个有效元素。下例限定 FP32 输入位于 `[-10000,10000]`，使用有限哨兵。

```python
x = tl.load(x_ptr + offsets, offsets < N,
            other=-1.0e9)
row_max = tl.max(x, axis=0)
```

数学上的最大值归约也可用负无穷作为单位元，但设备路径中的特殊值行为应单独验证，不能仅替换这个常量便扩大输入域。

**常见错误与关联操作。** 全负输入若用零填充，最大值会被尾部零污染。对空行则必须在宿主或控制流中单独处理，不能让“空集合最大值”意外进入归一化。

### min：沿轴求最小值 {#api-min}

**用途。** 得到最小值及下界统计，或对一个 tile 做区间分析。

```python
tl.min(
    input, axis=None, return_indices=False,
    return_indices_tie_break_left=True, keep_dims=False,
)
```

**参数与语义。** axis、keep_dims 和返回索引的规则与 max 对称。低精度浮点值归约会提升计算类型；为了让数值边界可见，推荐显式转换为 FP32。

**Zeus 使用约束。** 对基础值归约使用静态 axis，尾部填入不会小于任一有效输入的上界，保证至少存在一个有效元素。下例同样限定 FP32 输入位于 `[-10000,10000]`。值加索引的返回形式不能仅根据这个例子外推。

```python
x = tl.load(x_ptr + offsets, offsets < N,
            other=1.0e9)
minimum = tl.min(x, axis=0)
```

**常见错误与关联操作。** 全正数据用零填充会得到错误的零最小值。若后续要计算 `1 / minimum`，归约正确并不等于输入满足除法定义域；须独立检查零、符号和量级。

### dot：块矩阵乘与外部累加 {#api-dot}

**用途。** 表达矩阵乘的数学结构，由编译器映射到 PE 数据路径。

```python
tl.dot(
    input, other, acc=None, input_precision=None,
    allow_tf32=None, max_num_imprecise_acc=None,
    out_dtype=tl.float32,
)
```

**参数与语义。** 二维输入形状分别为 `[M,K]` 与 `[K,N]`，结果为 `[M,N]`。两输入的收缩维必须相等。`acc` 若提供，必须匹配输出 shape 和累加类型。前端还能描述批量形状，但本书基本 Matmul 使用二维 BF16 输入与 FP32 输出。

**Zeus 使用约束。** 采用普通 `tl.dot(a,b)`，在 K 循环外建立 FP32 accumulator，并用外部加法累加每个块的结果。当前前端配置不承诺任意非零 dot accumulator 直接融合；不要未经核对把外部加法改成三参数 dot。`out_dtype=tl.bfloat16` 被前端拒绝，先得到 FP32 结果，再在最终输出边界缩窄。

LDG 权重以逻辑 `[N,K]` tile 描述，实际字节由 LDG 格式组织；使用 `memory_type="weight"` 装载后 `tl.trans` 为 `[K,N]`。PE 模式和 minitile 由编译器选择。GPU TF32、Tensor Core 或手写 PE 模式参数不属于此基础调用。

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
for k_tile in range(tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_block, boundary_check=(0, 1),
                padding_option="zero")
    w = tl.load(w_block, boundary_check=(0, 1),
                padding_option="zero", memory_type="weight")
    acc += tl.dot(a, tl.trans(w))
    a_block = tl.advance(a_block, (0, BLOCK_K))
    w_block = tl.advance(w_block, (0, BLOCK_K))
out = acc.to(tl.bfloat16)
```

**常见错误与关联操作。** K 尾块要同时保护 A 和 W，不能只保护输出 M/N。普通 Tensor 指针不是 LDG 权重资源的通用替代。INT8 输入的 FP32 与 INT32 累加、FP8 编码及量化尺度均是独立契约，不能由 BF16 例子推导。

### rsqrt：计算倒平方根 {#api-rsqrt}

**用途。** 用于 RMSNorm 等算法中的归一化因子。

```python
tl.rsqrt(x)
```

**参数与语义。** 对每个元素计算倒平方根，结果保持输入形状。基础路径使用 FP32，算法应明确输入必须为正。

**Zeus 使用约束。** 倒平方根涉及近似与迭代配置；`--rsqrt-scsq` 是相关编译参数，基础值为 1。改变配置会同时影响数值误差和执行行为，应作为一项独立实验记录。不要把扩展的 `positive_normal_rsqrt` 与普通接口视为同一契约：前者还有 LUT 与严格的正正常数输入域。

```python
x32 = x.to(tl.float32)
mean_sq = tl.sum(x32 * x32, axis=0) / width
inv_rms = tl.rsqrt(mean_sq + eps)
y = x32 * inv_rms
```

**常见错误与关联操作。** `eps` 应加在平方均值之后；在 x 上先加 eps 会改变数学定义。小正数、接近零、很大输入和非有限值应分组验证，不能用随机正常数的平均误差替代边界分析。

### exp：有限 FP32 输入的指数计算 {#api-exp}

**用途。** 构造稳定指数、激活函数或概率计算的中间项。

```python
tl.exp(x)
```

**参数与语义。** 逐元素计算自然指数，shape 不变。基础用法将输入显式转换为 FP32。指数的大输入会放大相对误差或溢出，极小输入可能下溢。

**Zeus 使用约束。** 普通 `tl.exp` 由平台数学路径完成。它与调用方提供表系数的 `ztl.exp_with_lut` 是两个不同入口；仅替换成自定义表不能继承普通接口的误差边界。下例用每行最大值平移来控制指数输入量级，要求每行非空并已正确处理尾部。

```python
x32 = x.to(tl.float32)
row_max = tl.max(x32, axis=1, keep_dims=True)
weights = tl.exp(x32 - row_max)
den = tl.sum(weights, axis=1, keep_dims=True)
probability = weights / den
```

**常见错误与关联操作。** 上例说明计算关系，不替代完整 Softmax 的 mask、特殊值和输出契约。若一行全部为无效填充，最大值平移可能形成无效算术；应在进入该路径前排除空行。

## 控制流与编译期检查 {#api-control}

### range 与 tl.range：组织运行时循环 {#api-range}

**用途。** 在固定 tile shape 下遍历可变数量的块。

```python
range(start, end, step)
tl.range(
    arg1, arg2=None, step=None, num_stages=None,
    loop_unroll_factor=None, disallow_acc_multi_buffer=False,
    flatten=False, warp_specialize=False, disable_licm=False,
)
```

**参数与语义。** 单参数形式从零到上界，双参数形式指定起点与终点，步长默认 1。在 JIT kernel 中，循环边界可以来自运行时整数；循环携带的张量形状与类型仍要在各次迭代保持一致。

**Zeus 使用约束。** 基础分块优先使用普通 range，先确保正步长、有界迭代和输出归属。`tl.range` 的高级属性来自 Triton 前端，不能把 `num_stages`、`warp_specialize` 等直接解释为 Zeus 的双缓冲或执行单元数量。平台流水和并行通过对应的 Zeus 选项分析实现。

```python
acc = tl.full((), 0.0, tl.float32)
for tile in range(tile_count):
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets, offsets < N, other=0.0)
    acc += tl.sum(x.to(tl.float32), axis=0)
```

**常见错误与关联操作。** 在循环内部改变 tile 维度不是动态形状循环。循环次数可能为零时，循环后的每个值都必须有合法的初值。大范围展开会放大代码与资源占用，应和 `static_range` 区分。

### static_range：展开小型静态循环 {#api-static-range}

**用途。** 为少量、固定次数的操作生成编译期展开结构。

```python
tl.static_range(arg1, arg2=None, step=None)
```

**参数与语义。** 起点、终点与步长必须为 constexpr。它与 range 具有相同的半开区间计数方式，同时引导编译器积极展开循环。

**Zeus 使用约束。** 适用于少量固定分支或小窗口计算。展开不自动提供并行，也不改变操作之间的依赖。展开次数较大时，会增加编译时间、指令体积和中间值的存活范围。

```python
acc = tl.zeros((BLOCK,), tl.float32)
for part in tl.static_range(4):
    offsets = part * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    acc += x.to(tl.float32)
```

示例要求输入至少有 `4 * BLOCK` 个元素；若实际长度可变，应加入相应 mask，或改用运行时循环。

**常见错误与关联操作。** 把运行时 `N` 传给 static_range 会违反编译期条件。用它替换所有循环也不是通用性能优化，应该先确认循环确实小且展开后的资源仍合适。

### static_assert 与 static_print：检查特化配置 {#api-static-check}

**用途。** 在编译阶段拒绝不合法的形状配置，并输出需要观察的静态信息。

```python
tl.static_assert(cond, msg="")
tl.static_print(*values, sep=" ", end="\n",
                file=None, flush=False)
```

**参数与语义。** static_assert 的条件必须在编译期能够求值，失败时终止该特化。static_print 观察编译期值，打印的是特化形成时的信息，不是每次设备运行的数据。

**Zeus 使用约束。** 检查适合放在 kernel 开头，例如正 tile 大小、核数选择和两个静态维度的关系。它们不代替宿主对每次运行时指针、长度、dtype 与 stride 的检查。

```python
tl.static_assert(BLOCK > 0, "BLOCK must be positive")
tl.static_assert(CORE_NUM == 1 or CORE_NUM == 2)
tl.static_print("tile", BLOCK, "cores", CORE_NUM)
```

**常见错误与关联操作。** 命中编译缓存时未重新打印，不代表 kernel 未运行。`tl.static_assert(N > 0)` 不能检查普通运行时 N；此类条件应在宿主封装中使用普通检查。

## Zeus 扩展 {#api-zeus-extensions}

### softmax：沿一行完成稳定归一化 {#api-zeus-softmax}

**用途。** 用一个专用操作表达稳定 Softmax，由编译器保留该操作所需的有效域与归一化关系。

```python
tl.zeus.softmax(x, axis=-1)
```

**参数与语义。** `x` 为输入 tile，`axis` 是编译期归约轴，默认最后一维。数学上先减去该轴最大值，再计算指数并除以指数和。返回值保持输入 tile 的类型和形状；本书采用有限 FP32 输入与 FP32 输出。

**Zeus 使用约束。** 完整示例使用 `[1,1024]` 静态行块、`axis=1`，每个有效行由一次操作完成。实际行宽通过块指针的 shape 传入，load 和 store 均给出边界维度。它不是将多次 tile softmax 结果简单拼接的跨块算法；一行超过该模板宽度时，应改用具有跨块状态的实现。

```python
x = tl.load(
    input_block, boundary_check=(0, 1),
    padding_option="zero",
)
y = tl.zeus.softmax(x, axis=1)
tl.store(output_block, y, boundary_check=(0, 1))
```

片段中的两个块指针都使用真实的 `(rows, width)` 作为完整 shape，静态 block_shape 为 `(1,1024)`。专用 Softmax 路径与这套边界信息配合，处理当前行的有效范围；不要删掉真实 shape 后仅靠输出 store mask 修复分母。

**常见错误与关联操作。** 此处的补零不等价于普通 `max → exp → sum` 表达式的尾部策略。手动展开公式时，补零会参与指数和，必须另外排除无效列。验证专用接口时同时检查逐元素误差、行和与全负输入；仅检查输出没有越界不能证明归一化范围正确。完整算例见[算子开发实例](#examples)。

以下条目的 `ztl` 来自 `triton.backends.triton_shared.language`。扩展的价值在于表达平台专属意图；引入之前应保留对应标准写法的正确性基线。

### vp_hint 与 dt_hint：请求单个操作的拆分策略 {#api-executor-hint}

**用途。** 对产生某个值的逻辑操作指定局部执行单元策略。

```python
ztl.vp_hint(value, num_vp="auto")
ztl.dt_hint(value, num_dt="auto")
```

**参数与语义。** 返回与 value 相同的逻辑值，并附加供编译器消费的策略。计数接受 `"auto"` 或 1–4 的整数；整数为精确要求，无法满足时不能静默当成其他计数。

**Zeus 使用约束。** 适用性取决于操作、shape、布局、别名及物理资源。VP 与 DT 的可拆分条件不同，不能将一种操作的配置照搬到另一种。全局 `--num-vp` 请求与局部精确 hint 也不是同一层含义。

```python
value = x32 + y32
value = ztl.vp_hint(value, num_vp=1)
tl.store(out_ptr + offsets, value.to(tl.bfloat16), valid)
```

**常见错误与关联操作。** 忽略返回值会让提示脱离后继计算，甚至被优化移除。指定更多单元也可能增加拆分、同步和写回成本。优化选择应以[性能分析与调优](#performance)的同条件对照为依据。

### parallel_executor_hint 与 double_buffer_hint {#api-kernel-hint}

**用途。** 为整个 kernel 开启独立操作并行分析或实验性双缓冲分析。

```python
ztl.parallel_executor_hint()
ztl.double_buffer_hint()
```

**参数与语义。** 两个接口都不接受用户参数，也不返回计算结果。它们改变编译分析入口，不是设备运行时屏障或异步任务句柄。

**Zeus 使用约束。** 须在 kernel 顶层调用，通常作为第一条语句；for/if 不会将作用域缩到单次迭代。编译器仍检查独立性、别名、容量和依赖；实验性 DoubleBuffer 只改写满足条件的循环。

```python
ztl.parallel_executor_hint()
a = x32 + bias_x
b = y32 * scale_y
result = a + b
```

**常见错误与关联操作。** 这个例子展示 a/b 的逻辑独立性，但实际并行还要经过编译器分析。图中有两条计算分支并不等于执行时有两个活跃 lane。双缓冲同样不能取消读后写或写后读依赖。

### early_exit_if：按设备标量跳过后续工作 {#api-early-exit}

**用途。** 在协议明确允许时，按设备中已有的条件提前结束后续计算。

```python
ztl.early_exit_if(flag, nonzero=True)
```

**参数与语义。** flag 必须是单元素 FP32 张量。`nonzero=True` 时非零触发退出，False 时零触发退出。退出意味着文本上后续原本要执行的写入被跳过；这些输出保持先前内容。

**Zeus 使用约束。** 输入条件与输出初始化须遵守同一协议。各核条件可能不同时，后续不能有全核同步。解释器将此标记视为恒等操作，不能据此证明设备提前退出正确。

```python
flag = tl.load(flag_ptr + tl.arange(0, 1))
ztl.early_exit_if(flag, nonzero=True)
x = tl.load(x_ptr + offsets, valid, other=0.0)
tl.store(out_ptr + offsets, x, valid)
```

**常见错误与关联操作。** 触发退出时，未初始化的 empty 输出仍保留旧内容。调用方须预先初始化、返回有效状态，或按协议决定能否读取；跳过写入不会自动输出零。
