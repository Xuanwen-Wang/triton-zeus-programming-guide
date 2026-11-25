# 快速上手 {#quick-start}

本章用 BF16 向量加法完成一次最小开发闭环：用 PyTorch 准备输入和参考结果，通过 Python JIT 启动 Triton kernel，再同步、读回并比较。完整源码位于随书示例 `examples/vector_add.py`。无论 SDK 配置的是功能模拟还是物理设备，都使用同一个 Python 调用入口；执行后端由配套运行时接管。

两组上手输入共用 $\alpha=1$、`BLOCK=1024` 和 `CORE_NUM=2`。$N=2048$ 检查完整 tile，$N=2500$ 检查尾块。先使用可精确表示的小数做严格比较，之后再加入随机输入与误差评估，可以较快区分索引、边界和数值问题。

## 安装和检查环境

### 准备配套工具

安装所用发行版本配套的 Zeus Triton、PyTorch、`torch_zeus`、`zecc_jit` 原生扩展和 SDK 运行时。使用功能模拟 SDK 时，还应按发行说明配置配套模拟器；使用物理设备 SDK 时，完成驱动和设备检查。后续 Python 程序负责传入张量，运行时负责提交编译产物与执行参数，读者无需另写模拟器启动脚本。

不同发行环境对 Python 版本、native ABI 和库搜索路径的要求可能不同，安装时使用同一份发布清单。不要在已验证的环境里单独升级一个同名包，再沿用旧版本的编译器或运行时。源码开发环境还应使用该 checkout 的环境激活入口，确认环境变量没有指向其他构建目录。

进入随书工程根目录，先生成环境记录

```bash
mkdir -p build
python examples/check_environment.py \
  > build/environment.json
```

`zeus_language_extension` 应为 `true`，表示已加载 Zeus 语言扩展。再检查 `executable`、模块 `origin` 和工具路径：`triton` 应为 Zeus 配套构建，`zecc` 与 `zeus-compiler` 应来自本次工具链。版本号相同但路径不同，仍可能是不同的源码或构建产物。

脚本默认只检查导入位置和工具路径，不启动 kernel。需要检查 PyTorch 设备注册时，显式运行

```bash
python examples/check_environment.py --probe-runtime
```

`runtime_probe` 中的 `registered` 表示 `torch.zeus` 接口已注册，`available` 表示运行时判断设备可用。安装环境如使用模拟器适配器，这两项也可能成立。测试记录应同时注明实际后端；本章脚本不会自动替换后端或修改模拟器环境变量。

### 常见检查结果

| 现象 | 首先核对 | 处理方向 |
|---|---|---|
| 无法导入 `triton` | 当前解释器、环境激活 | 使用配套 Python 环境 |
| `tl.zeus` 不存在 | `triton.__file__`、wheel 来源 | 核对是否为 Zeus 构建 |
| 找不到 `zecc` | `PATH` 与 wheel 工具目录 | 恢复配套工具入口 |
| `torch_zeus` 导入失败 | native 库与 SDK 配套 | 查看首次导入异常 |
| 设备不可用 | 运行时配置与设备状态 | 先完成 SDK 的设备检查 |
| 编译能通过，启动失败 | 运行时、产物与参数 ABI | 查阅编译执行与调试章节 |

Table: 上手环境检查的常见问题

排查时保留最早出现的错误。后续的“模块不可用”或“启动失败”可能只是前一项失败的结果。检查程序会报告实际加载位置，适合随最小复现一起保存。

### 用同一个 Python 入口选择执行环境

本章以 `--run-device` 启动的 Python 运行时路径为主线。该选项表示执行 Zeus kernel；它本身不选择硅片或模拟器。功能模拟 SDK 会自动启动或连接已配置的模拟服务，加载 kernel、准备参数并执行，结果仍通过同一个输出 Tensor 读回。程序中的 PyTorch 参考负责比较结果，不需要注册到另一套测试工具中。

| 当前环境 | 建议的起点 | 成功后可以继续 |
|---|---|---|
| 已安装 PyTorch 与 Triton Python 包，不要求设备运行时 | CPU 参考 | 检查输入生成与预期值 |
| 已配置功能模拟 SDK 与运行时 | 检查运行时，再执行示例 | 编译、模拟执行、结果比较与边界回归 |
| 已配置物理设备 SDK 与运行时 | 检查运行时，再执行示例 | 设备执行与结果比较 |

Table: 相同 Python 示例在不同环境中的用途

首次运行建议选择 JIT 模式，固定 `BLOCK=1024`、`CORE_NUM=2` 和 `alpha=1.0`。先不同时更改长度、dtype、核数和运行模式，便于将失败定位到一个阶段。功能模拟的后端配置见[功能模拟与物理设备的运行时选择](#runtime-backends)。离线编译用于查看或交付产物，首次 Python 调用会按需完成编译，无须预先手工生成 `.zbin`。

## 第一个向量加法 {#vector-add}

### 输入和输出契约

kernel 处理两个长度相同的连续一维 BF16 张量，输出为同形状的独立 BF16 张量。元素总数 `numel` 使用非负 `i32`，`alpha` 使用 FP32 标量。上手例的输入是有限小数，$\alpha=1$；一般输入的极值、特殊值和融合运算误差需要另按数值契约验证。

| 参数 | 含义 | 本例约束 |
|---|---|---|
| `out_ptr` | 输出首地址 | BF16，容量不少于 `numel` |
| `in1_ptr`、`in2_ptr` | 两个输入首地址 | BF16，同长度、连续 |
| `numel` | 有效元素总数 | $N\ge0$，且 $N+\mathrm{BLOCK}-1\le2^{31}-1$ |
| `alpha` | 第二个输入的系数 | 上手固定为 1.0 |
| `BLOCK` | 每个 tile 的元素数 | 特化常量，取 1024 |
| `CORE_NUM` | 本例参与工作的核数 | 特化常量，取 2 |

Table: 向量加法的参数契约

宿主封装负责验证形状、dtype、设备和连续性，并为输出分配新存储。上述长度限制同时保护 i32 参数和 `tl.cdiv` 的中间加法；仅检查 N 能表示为 i32 还不够。`numel=0` 时直接返回空输出，不发起设备任务。数据指针的有效期至少覆盖 kernel 完成之前的整个执行过程。

输入允许共享只读数据，但输出由封装单独分配，不与输入别名。若将示例扩展为原地更新，还必须检查每个元素是否在覆盖前已经完成所有读取。`x` 和 `z` 指向同一块输入，与输出覆盖输入，是两种不同的别名关系。

完整 `launch` 函数选择只接受一维连续输入、一个或两个核，以及本例使用的正 2 次幂 block 大小。这些检查定义该示例的使用范围；不能把示例对 block 的检查扩大解释为 Zeus 所有接口的统一形状限制。

### 为每个核分配连续 tile

下面的代码先计算 tile 总数，再用商和余数划分任务。它与完整示例中的实现相同

```python
core_id = tl.program_id(0)
total_blocks = tl.cdiv(numel, BLOCK)
quotient = total_blocks // CORE_NUM
remainder = total_blocks % CORE_NUM
count = quotient + tl.where(core_id < remainder, 1, 0)
start = core_id * quotient + tl.minimum(core_id, remainder)
```

在 $N=2500$ 时，`total_blocks=3`，`quotient=1`，`remainder=1`。编号较小的核额外获得一个 tile，因此核 0 从 tile 0 开始处理两个 tile，核 1 从 tile 2 开始处理一个 tile。两核的有效输出区间分别是 `[0,2048)` 和 `[2048,2500)`，没有重叠，也没有遗漏。

::: {.caution}
本例使用 `grid=(CORE_NUM,)`，`CORE_NUM=2` 表示两个参与核。三个 tile 由核内循环处理，不能沿用 GPU 的 `grid=(total_blocks,)`。完整推导见[编程模型](#kernel-launch)。
:::

### 用块指针描述一维数据

块指针同时描述整个逻辑张量和一次处理的 tile。输入 `x` 的模板为

```python
x_block = tl.make_block_ptr(
    in1_ptr, (numel,), (1,), (0,), (BLOCK,), (0,)
)
```

字段依次为首地址、逻辑 shape、元素 stride、初始 offset、tile shape 和维度次序。`shape=(numel,)` 是整个输入的有效范围，`block_shape=(BLOCK,)` 是一次 load 的逻辑元素数，两者不必相等。连续一维张量的 stride 为 1，指针偏移以元素为单位。

第二个输入和输出采用相同的逻辑 shape。循环每次用 `tl.advance` 把模板移动到当前 tile 的起点。`tl.advance` 生成新的块指针视图，不会搬运数据，也不会改变原输入的内容。

### 装载、计算和保存

每次循环处理一个 tile，主数据流如下

```python
offset = (start + local_tile) * BLOCK
x = tl.load(
    tl.advance(x_block, (offset,)),
    boundary_check=(0,), padding_option="zero",
).to(tl.float32)
z = tl.load(
    tl.advance(z_block, (offset,)),
    boundary_check=(0,), padding_option="zero",
).to(tl.float32)
y = (x + alpha * z).to(tl.bfloat16)
tl.store(
    tl.advance(y_block, (offset,)), y,
    boundary_check=(0,),
)
```

`boundary_check=(0,)` 指定在第 0 维检查边界。尾块中越界的 load 位置使用零填充，store 只写逻辑范围内的元素。对 $N=2500$ 的最后一块，452 个位置对应有效输入和输出，另外 572 个位置只参与 tile 内的占位计算。

读取 BF16 后转换为 FP32，使加法与乘法在指定的计算类型中进行；保存前再转换为 BF16。默认 BF16 转换采用最近偶数舍入。参考程序同样先读取 BF16 输入的真实值，在 FP32 中计算，然后转换为 BF16。对更一般的 $\alpha$，是否使用融合乘加及特殊值规则会影响逐位比较，应按[精度与边界测试](#debugging)规定输入域与误差标准。

## 编译、运行与检查输出

### 先检查 CPU 参考

在随书工程根目录运行

```bash
python examples/vector_add.py --numel 2048
python examples/vector_add.py --numel 2500
```

输出的 `execution` 值为 `cpu_reference_only`，表示只生成输入与 CPU 参考，尚未启动 Zeus kernel。默认输入按八分之一步长构造，前八个参考输出为

```text
[-3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25]
```

固定输入的用途是让索引与尾块问题容易复现。改变随机种子、dtype、长度和 `alpha` 的数值回归放在最小闭环通过之后进行。CPU 参考成功也不能替代设备执行检查。

输入生成公式为 $x_i=((i\bmod33)-16)/8$、$z_i=((i\bmod17)-8)/8$，之后转换为 BF16。两路使用不同周期，因此块边界之后不会一直重复同样的局部内容。八分之一步长在这里能够精确表示，有助于把错误首先定位到索引或数据路径，而不是浮点舍入。

例如 $i=2048$ 时，$x_i=-1.75$、$z_i=0$，期望输出为 $-1.75$；$i=2499$ 时，$x_i=1.0$、$z_i=-1.0$，期望输出为 0。这两个位置分别是尾块首元素和整个张量末元素，比只检查前八个值更能覆盖尾块路径。

### 通过 Python 启动 Zeus 运行时

在已经完成配套安装和运行时检查的环境中，选择 JIT 模式并添加 `--run-device`。若 SDK 已选择 JIT，直接运行下面两条 Python 命令即可。

```bash
export ZECC_KERNEL_MODE=jit
python examples/vector_add.py --numel 2048 --run-device
python examples/vector_add.py --numel 2500 --run-device
```

脚本导入 `torch_zeus`，将输入复制到 `zeus:0`，分配输出并调用 kernel。Zeus Triton 识别输入的设备类型，将调用交给 `zecc_jit`；首次特化按需编译，随后由运行时在当前 stream 上执行。若该 SDK 配置为功能模拟，这条调用链自动进入模拟器；宿主输入、输出对象和比较代码保持相同。设备编号可以通过 `--device zeus:1` 修改，但必须是当前运行时中实际存在的设备。启动部分的核心写法为

```python
add_bf16[(cores,)](
    out, x, z, x.numel(), float(alpha),
    BLOCK=block, CORE_NUM=cores,
    zeus_enable_jit=True,
)
```

`zeus_enable_jit=True` 允许在所选模式需要时进行编译；实际采用 JIT、AOT 或 hybrid，由运行时模式配置决定。该选项不会把 AOT 模式扩大为允许即时编译。它属于 Zeus 运行时适配层，不是 kernel 的数值参数；`BLOCK` 和 `CORE_NUM` 则是 kernel 的特化常量。

::: {.tip}
首次运行直接使用本节的 Python 命令和 PyTorch 参考比较。SDK 已配置功能模拟时，同一个 JIT 入口会提交到模拟后端，无须另写模拟测试脚本；`--run-device` 本身不选择运行后端。
:::

宿主在读取结果前调用 `torch.zeus.synchronize()`，随后把输出复制回 CPU，与相同 BF16 输入产生的参考结果比较。默认的有限输入和 $\alpha=1$ 使用 `rtol=0`、`atol=0` 的严格比较。通过时结果记录含有 `execution="zeus_runtime"`、`exact_match=true` 和 `max_abs=0.0`。`zeus_runtime` 表示调用经过设备运行时，并不表示一定在物理硬件上执行；保存运行输出时同时记录 SDK 后端。

### 宿主侧的数据与资源生命周期

完整示例的宿主执行顺序如下：先在 CPU 上创建 BF16 输入和参考结果，再导入设备后端，将输入传输到目标设备；随后分配独立输出，提交 kernel，等待设备工作完成，最后读回并比较。

::: {.keep}
下面片段可在 `examples/` 目录中使用，调用随书已经定义的封装，而不重复编写 kernel：

```python
import torch
import torch_zeus
from vector_add import make_inputs, reference, launch

x_cpu, z_cpu = make_inputs(2500)
expected = reference(x_cpu, z_cpu)
x = x_cpu.to("zeus:0")
z = z_cpu.to("zeus:0")
out = launch(x, z, block=1024, cores=2)
torch.zeus.synchronize()
actual = out.cpu()
torch.testing.assert_close(
    actual, expected, rtol=0, atol=0
)
```
:::

输入和输出对象在同步前都保持存活。读取 `actual` 之后，宿主即可使用比较结果。后续为提高吞吐而缩小同步范围时，仍要保留相同的生产消费关系，不能因为删除显式等待而允许提前覆盖输入或释放工作区。

示例出现异常时，应先保留完整异常和环境记录。通过移除同步、删除边界保护或放宽误差使命令不再报错，都会改变原来的验收问题，不能作为上手成功的依据。

### 用 ZECC 离线编译

离线编译可以独立检查源码与工具链的衔接，也适合查看编译产物。下面使用完整签名和常量生成 `.zbin`

```bash
zecc --compile-only-zbin \
  -c examples/vector_add.py \
  -o build/vector-add.zbin \
  --kernel-name add_bf16 \
  --signature 'out_ptr=*bf16' \
  --signature 'in1_ptr=*bf16' \
  --signature 'in2_ptr=*bf16' \
  --signature numel=i32 \
  --signature alpha=fp32 \
  --constant BLOCK=1024 \
  --constant CORE_NUM=2 \
  --loc=full
```

`--compile-only-zbin` 模式下，命令行常量集合替换模块中的默认集合，因此要同时提供 `BLOCK` 和 `CORE_NUM`。`--loc=full` 请求源位置 sidecar 和对应的 TTIR 文件，便于将后续诊断关联回源程序。编译产物的文件名、签名和常量应随本次测试记录保存。

此时得到的是目标程序，还没有运行任何输入。离线编译通过证明当前源码与特化可生成二进制；正确性仍由 Python 运行时执行后的结果比较确认。普通 JIT 调用会解析自己的特化，单独生成这份 `.zbin` 不会让下一次 Python 调用自动改用它。需要部署预编译产物时，使用[编译、执行与框架接入](#compile-and-run)中的 AOT bundle 流程。

::: {.keep}

### 理解一次运行的输出

完整示例把输入生成、参考计算、kernel 调用与结果检查放在同一个程序中。以 N=2500 的运行结果为例，关键字段如下：

```text
"execution": "zeus_runtime"
"numel": 2500
"alpha": 1.0
"exact_match": true
"max_abs": 0.0
```

:::

这些字段只在运行和比较成功后输出。若比较失败，`torch.testing.assert_close` 抛出异常；若编译、加载或执行失败，程序会在对应阶段报告错误。看到编译日志或 `.zbin` 文件存在，还不能判断结果比较已经通过。

将新算法接入这一模式时，保留四个部分：输入生成函数、PyTorch 参考函数、带参数检查的 `launch` 封装，以及同步读回后的比较。参考函数应使用实际送入设备的 dtype 和舍入后输入；例如 BF16 输入不能用其尚未舍入的 FP32 原值替代。这样功能模拟与物理设备可以复用同一套输入和结果判断。

两组输入通过后，可保持其他参数不变，补查第二个 tile 仅有一个有效元素的情形：

```bash
python examples/vector_add.py --numel 1025 --run-device
```

检查整个有效输出，不用平均误差掩盖少数尾部错误。空输入、多轮循环、数值边界和错误定位见[调试与正确性验证](#debugging)；改变 block、核数或计算参数时，按该章方法重新确定测试范围。
