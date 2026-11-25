# 快速上手 {#quick-start}

本章用 BF16 向量加法完成一次最小开发闭环：用 PyTorch 准备输入和参考结果，通过 Python JIT 启动 Triton kernel，再同步、读回并比较。完整源码位于随书示例 `examples/vector_add.py`。无论 SDK 配置的是功能模拟还是物理设备，都使用同一个 Python 调用入口；执行后端由配套运行时接管。

两组上手输入共用 $\alpha=1$、`BLOCK=1024` 和 `CORE_NUM=2`。$N=2048$ 检查完整 tile，$N=2500$ 检查尾块。先使用可精确表示的小数做严格比较，之后再加入随机输入与误差评估，可以较快区分索引、边界和数值问题。

## 安装和检查环境 {#setup-environment}

### 取得同版示例与配套 SDK {#example-download}

在支持附件的 PDF 阅读器中打开本指南，从附件面板保存 `Triton-Zeus-Examples-<版本>.zip`。解压后，命令从含 `examples/` 的目录运行。附件与封面版本一同构建。也可从[指南仓库](https://github.com/Xuanwen-Wang/triton-zeus-programming-guide)取得源码，按该版 `pdf-build.json` 核对示例文件的 SHA-256。使用和分发范围见[版权声明](#copyright)。

通过北极雄芯的产品交付或技术支持渠道取得适用 Zeus V3 配置的 SDK、安装说明和配套清单。按同一份清单完成安装与后端准备，并记录下表项目。具体版本号随实际 SDK 交付。

| 配套项 | 需要核对的内容 |
|---|---|
| 设备与系统 | Zeus V3 设备配置、Linux/架构、驱动版本或模拟器版本 |
| Python 与框架 | Python、PyTorch、Zeus Triton、`torch_zeus` 的包与 ABI |
| 编译和运行时 | `zecc_jit`、ZECC 与 Zeus 编译器构建，SDK 发行标识 |
| 执行配置 | JIT 模式、设备索引、功能模拟或物理设备 |

Table: 客户安装与运行记录的配套项

### 检查环境

进入随书工程根目录，先生成环境记录

```bash
mkdir -p build
python examples/check_environment.py \
  > build/environment.json
```

`zeus_language_extension` 应为 `true`，表示已加载 Zeus 语言扩展。再检查 `executable`、模块 `origin` 和工具路径：`triton` 应为 Zeus 配套构建，`zecc` 与 `zeus-compiler` 应来自本次工具链。版本号相同但路径不同，仍可能是不同的源码或构建产物。

脚本默认收集导入位置和工具路径。准备使用 JIT 时，运行严格前置检查。

```bash
python examples/check_environment.py --require-jit \
  > build/jit-environment.json
```

`jit_prerequisites_ready=true` 表示 Zeus 语言扩展、`zecc_jit` 导入和设备运行时可用性检查均通过。任一前置检查失败时，严格模式返回非零。可用设备由 SDK 运行时报告，也可能是模拟器设备。实际后端随运行结果从 SDK 配置记录，kernel 执行通过本章向量加法结果确认。

### 常见检查结果

| 现象 | 首先核对 | 处理方向 |
|---|---|---|
| 无法导入 `triton` | 当前解释器、环境激活 | 使用配套 Python 环境 |
| `tl.zeus` 不存在 | `triton.__file__`、wheel 来源 | 核对是否为 Zeus 构建 |
| 找不到 `zecc` | `PATH` 与 wheel 工具目录 | 恢复配套工具入口 |
| `torch_zeus` 导入失败 | native 库与 SDK 配套 | 查看首次导入异常 |
| `zecc_jit` 导入失败 | 原生扩展与 Python ABI | 核对 SDK 清单与运行库搜索路径 |
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

::: {.keep}

### 用块指针描述一维数据

块指针同时描述整个逻辑张量和一次处理的 tile。下面以输入 `x` 的模板为例。

```python
x_block = tl.make_block_ptr(
    base=in1_ptr,
    shape=(numel,),
    strides=(1,),
    offsets=(0,),
    block_shape=(BLOCK,),
    order=(0,),
)
```

`base` 指向输入首元素。`shape=(numel,)` 给出整个输入的有效长度，`block_shape=(BLOCK,)` 给出每次 load 的 tile 长度。`strides=(1,)` 表示相邻元素间隔一个元素，`offsets=(0,)` 将模板放在输入起点，`order=(0,)` 指定唯一的维度。

:::

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

宿主调用 `torch.zeus.synchronize(out.device)` 等待输出所属设备，然后在 CPU 逐项比较。设备 smoke test 固定为有限输入和 $\alpha=1$，容差为零。`--alpha` 可用于 CPU 参考的其他有限值。扩展设备用例时，应先确定 FP32 融合或分步乘加的数值契约，再定义容差。

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
torch.zeus.synchronize(out.device)
actual = out.cpu()
torch.testing.assert_close(
    actual, expected, rtol=0, atol=0
)
```
:::

输入和输出对象在同步前都保持存活。读取 `actual` 之后，宿主即可使用比较结果。后续为提高吞吐而缩小同步范围时，仍要保留相同的生产消费关系，不能因为删除显式等待而允许提前覆盖输入或释放工作区。

示例出现异常时，应先保留完整异常和环境记录。通过移除同步、删除边界保护或放宽误差使命令不再报错，都会改变原来的验收问题，不能作为上手成功的依据。

::: {.keep}

### 用 ZECC 离线编译

离线编译检查所用源码和特化能否生成相应设备产物。下面使用完整签名和常量生成 `.zbin`。

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

:::

`--compile-only-zbin` 模式下，命令行常量集合替换模块中的默认集合，因此要同时提供 `BLOCK` 和 `CORE_NUM`。`--loc=full` 请求源位置 sidecar 和对应的 TTIR 文件，便于将后续诊断关联回源程序。编译产物的文件名、签名和常量应随本次测试记录保存。

此时得到的是目标程序，还没有运行任何输入。离线编译通过证明当前源码与特化可生成二进制；正确性仍由 Python 运行时执行后的结果比较确认。普通 JIT 调用会解析自己的特化，单独生成这份 `.zbin` 不会让下一次 Python 调用自动改用它。需要部署预编译产物时，使用[编译、执行与框架接入](#compile-and-run)中的 AOT bundle 流程。

::: {.keep}

### 理解一次运行的输出

下面是 N=2500 成功执行且逐项相等时的关键字段。

```json
{
  "execution": "zeus_runtime",
  "kernel_launched": true,
  "device": "zeus:0",
  "numel": 2500,
  "exact_match": true,
  "max_abs": 0.0
}
```

:::

将 JSON、`jit-environment.json`、SDK 发行标识及实际后端一同保存。非空设备用例成功后应有 `kernel_launched=true`。CPU 参考和设备空输入分别输出 `cpu_reference_only` 和 `empty_output`，两者均为 `kernel_launched=false`。编译、执行或结果比较失败会以异常结束。

扩展用例时保留输入、CPU 参考、带参数检查的 launch 和同步回读，按设备实际收到的 dtype 与数值计算参考。两组输入通过后，补查第二个 tile 仅有一个有效元素。

```bash
python examples/vector_add.py --numel 1025 --run-device
```

检查全部有效元素。更多边界和误差用例见[调试与正确性验证](#debugging)。
