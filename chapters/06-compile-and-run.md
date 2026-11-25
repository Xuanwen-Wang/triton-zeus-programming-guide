# 编译、执行与框架接入 {#compile-and-run}

Kernel 的 Python 源码描述设备计算，宿主程序负责准备输入、选择特化、提交执行并消费结果。Triton Zeus 支持运行时 JIT 和离线编译两类工作方式；同一份源码可以生成独立 `.zbin`，也可以生成嵌入设备二进制的宿主对象文件。

选择工作方式前，先确定交付物。交互式开发通常使用 JIT，固定部署可以使用 AOT bundle，C/C++ 集成可以使用 ZECC 生成的对象和启动头文件。无论采用哪条路径，输入输出的数值定义、运行时参数顺序和编译常量都必须一致。

![同一特化在 JIT、AOT 与 hybrid 模式中的解析过程](assets/jit-aot.svg){#fig-jit-aot width=100%}

## Python 接入与运行时 JIT {#python-jit}

### 从 Python 调用到运行时

Python 设备调用需要配套的 Zeus Triton、`torch_zeus`、`zecc_jit` 原生扩展和 SDK。`torch_zeus` 注册 PyTorch 的 `zeus` 设备与 `torch.zeus` 接口；Zeus Triton 在 kernel 参数中识别 Zeus 张量，转入 `zecc_jit` 的启动适配器。

这些组件承担不同职责。`@triton.jit` 保存 kernel 的 Python 定义；启动适配器根据实参区分指针、标量和 constexpr；`libzecc_jit` 选择已编译特化或触发编译；运行时按设备 ABI 打包参数，并向当前 stream 提交 kernel。

上手时应确认正在使用配套 Zeus 构建。仅安装上游 Triton，或者仅能导入后端中用于生成 TTIR 的 CPU 适配类，都不意味着设备启动路径可用。环境检查见[快速上手](#quick-start)。

下面复用完整示例中的宿主封装。脚本默认运行 CPU 参考；显式选择 `--run-device` 才调用已配置的 Zeus 运行时。

```bash
python examples/vector_add.py --numel 2048 --run-device
python examples/vector_add.py --numel 2500 --run-device
```

这两条命令同时适用于配置完毕的功能模拟 SDK。运行时会管理模拟服务、内存传输、二进制和参数提交；Python 程序仍使用 Zeus Tensor、普通 kernel 调用、同步和 `.cpu()` 读回。参考结果直接用 PyTorch 编写并通过 `torch.testing.assert_close` 比较，不需要为运行时另行登记参考函数。

### 功能模拟与物理设备的运行时选择 {#runtime-backends}

执行后端和编译模式是两个独立选择。`ZECC_KERNEL_MODE` 决定特化从 JIT 缓存还是 AOT bundle 获取；功能模拟或物理设备决定生成的 kernel 在哪里执行。选择 AOT 不会自动改成物理设备，选择 JIT 也不意味着在 CPU 上解释 kernel。

配套 SDK 应在激活时配置 Python、native 库、编译器及执行后端。模拟运行时提供以下显式选择；路径使用 SDK 实际交付的位置，在首次 Zeus 内存分配和启动前设置。

| 模拟运行时配置 | 值的含义 | 配套位置 |
|---|---|---|
| `ZEUS_V3_BACKEND=local-cpp` | 使用本地 C++ 模拟器后端 | `ZEUS3_CPP_SIMULATOR` 指向配套模拟器可执行文件 |
| `ZEUS_V3_BACKEND=local-python` | 使用本地 Python 模拟器后端 | `ZEUSV3_SIMULATOR_DIR` 指向配套模拟器目录 |

Table: 模拟运行时的显式后端选择

功能模拟环境应选用发行清单中的功能模拟器及其数据文件。运行时按上述配置自动启动或连接服务；单进程向量示例无需手工管理模拟服务。SDK 已提供配置时，保持该配置即可。需要切换后端时退出旧 Python 进程，在新环境中重新启动，避免已初始化的运行时继续使用旧连接。

物理设备环境使用对应的 SDK 运行时、驱动和设备配置；不能把模拟后端变量改成一个未定义的值来切换到物理设备。用户程序的 kernel、输入生成和结果检查可以保持不变，环境配套与执行结果仍需重新确认。

`torch.zeus.is_available()` 表示当前运行时报告有可用设备，`--run-device` 表示示例会提交 Zeus kernel；二者都不单独证明使用物理硬件。功能模拟结果用于检查算法和数据路径，不能作为物理设备持续带宽或执行时间的测量。记录结果时注明 SDK 后端、实际模拟器或设备，以及比较用的输入和容差。

### 参数如何进入 kernel

向量加法的调用包括三个指针、两个运行时标量和两个编译常量。

```python
add_bf16[(cores,)](
    out, x, z, x.numel(), float(alpha),
    BLOCK=block, CORE_NUM=cores,
    zeus_enable_jit=True,
)
```

其中 `out`、`x`、`z` 必须属于同一设备，dtype 和布局满足宿主封装的要求；`numel` 决定本次有效范围；`alpha` 作为 FP32 标量进入参数包；`BLOCK` 和 `CORE_NUM` 进入特化身份。

当前适配器把一维 grid 的第一个值传给 `num_cores`，并据签名和 constexpr 解析对应特化。因此本书使用 `(cores,)` 启动，再由 kernel 源码中的循环遍历分配到各核的 tile。逻辑 tile 总数不由启动器自动扩展为多轮 program；它属于 kernel 的任务分配。`CORE_NUM`、请求的启动核数和二进制的核数描述应保持一致，实际执行还须符合产物中的核入口。GPU 常见的每 tile 一个 program、persistent 组织方式和 grid-stride 循环之间的区别，见[Kernel 启动与程序实例](#kernel-launch)。

`zeus_enable_jit=True` 允许运行时按所选模式进行 JIT，但不会覆盖显式选择的 AOT 模式。`zeus_strict` 为兼容已有调用保留；当前解析失败均明确报错，是否允许从 AOT 回到 JIT 由部署模式决定。因此，同一份示例可以在不修改 kernel 调用的情况下切换到纯 AOT。

适配器根据参数注解和实际对象建立签名。普通 Python 整数按范围推断为 i32 或 i64，普通 Python 浮点数推断为 FP32；指针由张量 dtype 和设备信息确定。对于需要固定 ABI 的接口，应显式约束标量位宽，避免某次长度越过整数范围后变成另一种签名。

`num_warps`、`num_stages` 等名称可能作为兼容启动元数据被接受，但不能据此推断 Zeus 有 GPU warp，或已经按相应数值建立硬件流水。真正影响 Zeus 编译的参数应通过有文档说明的编译选项或平台接口传递。

### 启动参数的作用域

一个调用同时包含运行时实参、编译常量和启动配置。三者不能交换位置：`alpha` 是每次执行读取的值；`BLOCK` 决定生成的静态 tile；`zeus_mode` 决定到哪里查找这个特化。参数名以 kernel 的形式参数为准，适配器按照函数定义顺序整理运行时参数。

| 启动关键字 | 值与作用 |
|---|---|
| `zeus_mode` | `jit`、`aot` 或 `hybrid`；选择本次解析模式 |
| `zeus_enable_jit` | 布尔值；`False` 将解析限制到 AOT |
| `zeus_log` | 布尔值；覆盖本次启动的日志设置 |
| `zeus_target` | 目标名称，本书使用 `zeus-v3` |
| `zeus_device_index` | 显式设备编号，必须与指针设备一致 |
| `zeus_compiler_extra_flags` | 本次特化的编译器选项字符串 |

Table: Python kernel 启动配置

模式优先级从单次 `zeus_mode` 开始，未指定时读取 `ZECC_KERNEL_MODE`，仍未指定时使用 `jit`；之后再应用 `zeus_enable_jit=False` 的限制。未知模式在启动前报错。`zeus_enable_jit=True` 不会把显式 `aot` 放宽成 `hybrid`，这一点尤其适用于同时服务开发和部署环境的公共封装。

如果输入来自不同设备，或显式设备编号与指针所指设备不同，启动器拒绝调用。对只有标量、没有可推断设备的低层调用，应显式传入设备编号。设备编号选择的是运行位置，不改变指针所属设备，也不执行跨设备复制。

### 显式参数包接口

常规 Python 算子优先使用 `kernel[grid](...)`。当宿主框架已经管理签名、AOT 身份和参数顺序时，可以直接调用 `zecc_jit.launch`，将这些契约集中在一处。它的关键参数为：

```python
zecc_jit.launch(
    kernel, argpack=argpack, signature=signature,
    constants=None, op_name=None, kernel_name=None,
    source_path=None, source_sha256=None,
    target="zeus-v3", num_cores=1, attrs=None,
    device_index=None, enable_jit=True,
    strict=True, log=None, mode=None,
)
```

除 `kernel` 外均为关键字参数；`argpack` 和 `signature` 必须提供。`op_name` 用于诊断归属，`kernel_name` 选择编译入口，`source_path` 或 `kernel` 提供可读取的源码身份。`source_sha256` 用于持有准确源码摘要的 AOT 部署；它不能为 JIT 凭空提供待编译的源码。若同时给出文件和摘要，两者必须一致。`strict` 仅保留调用兼容性，不控制失败后的回退策略。

下面的参数包对应向量加法。将两个代码块顺序放入同一个 Python 脚本，并从指南根目录执行；它们使用与完整示例相同的输入和数学参考。

```python
import torch
import torch_zeus
import zecc_jit
from examples.vector_add import add_bf16, make_inputs
from examples.vector_add import reference, signature

host_x, host_z = make_inputs(2500)
x = host_x.to("zeus:0")
z = host_z.to("zeus:0")
out = torch.empty_like(x)
args = [
    zecc_jit.tensor_ptr(out, "out_ptr", direction="out"),
    zecc_jit.tensor_ptr(x, "in1_ptr", direction="in"),
    zecc_jit.tensor_ptr(z, "in2_ptr", direction="in"),
    zecc_jit.i32(x.numel(), "numel"),
    zecc_jit.f32(1.0, "alpha"),
]
```

```python
zecc_jit.launch(
    add_bf16, argpack=args, signature=signature,
    constants={"BLOCK": 1024, "CORE_NUM": 2},
    num_cores=2, device_index=0,
)
torch.zeus.synchronize(0)
expected = reference(host_x, host_z)
torch.testing.assert_close(
    out.cpu(), expected, rtol=0, atol=0,
)
```

参数包中的指针对象保留 dtype、设备和地址信息，不能用任意 Python 整数替换一个尚未分配的设备张量。`i32`、`i64`、`u32`、`u64`、`f32`、`f64`、`boolean` 建立相应的宿主参数记录；记录类型必须与编译签名一致。存在某种标量打包方式，并不表示 kernel 可用该类型执行任意设备运算。

特别注意 `f32(1.0)` 的参数包类型与 Triton 签名中的 `fp32` 是两套接口中的名称。不要为了让字符串看起来一致而将它们互换。输入/输出 `direction` 用于描述参数方向，不负责分配、复制或隐式同步。

### Stream、同步与结果生命周期

Kernel 提交成功表示运行时已接受操作，不能直接等同于输出已经可被宿主读取。原生启动器使用目标设备的当前 stream；输入、输出与临时分配必须在相关操作完成前保持有效。

入门脚本在取回结果前显式同步。

```python
actual = launch(x_device, z_device, alpha=1.0)
torch.zeus.synchronize()
actual_cpu = actual.cpu()
torch.testing.assert_close(actual_cpu, expected,
                           rtol=0, atol=0)
```

这里使用零容差，是因为首个 smoke test 的输入和 `alpha=1` 被设计为可精确表示。扩展到一般随机数据和其他算法后，应使用[精度与边界测试](#debugging)中的数值契约，不能把零容差当作所有算子的统一标准。

异步流水中应通过所用运行时的 stream 和事件机制表达跨 stream 依赖。只在 Python 中先后调用两个函数，并不能替代跨 stream 同步。对 Graph 或长期复用 buffer 的集成，还需检查输入内容更新、地址稳定性和 replay 期间的资源生命周期。

![跨 stream 消费与输入输出资源的有效期](assets/runtime-lifecycle.svg){#fig-runtime-lifecycle width=100%}

### 用 stream 表达依赖

同一 stream 中提交的操作按其顺序执行。不同 stream 之间使用 `wait_stream` 建立依赖边：消费 stream 后续提交的操作，等待生产 stream 在这个等待点之前已提交的工作。它不约束生产 stream 将来才提交的新工作，也不是让整个宿主线程等待。

下面连续调用两次向量算子，第二次依赖第一次的输出。输入复制先发生在当前 stream，producer 在读取输入前等待该 stream；consumer 再等待 producer。代码没有假定复制与计算一定重叠，也没有把 Python 语句顺序当作设备完成顺序。

```python
import torch
import torch_zeus
from examples.vector_add import launch, make_inputs, reference

hx, hz = make_inputs(2500)
x, z = hx.to("zeus:0"), hz.to("zeus:0")
initial = torch.zeus.current_stream(0)
producer = torch.zeus.Stream(device=0)
consumer = torch.zeus.Stream(device=0)
producer.wait_stream(initial)
with torch.zeus.stream(producer):
    middle = launch(x, z)
consumer.wait_stream(producer)
with torch.zeus.stream(consumer):
    result = launch(middle, z)
consumer.synchronize()
expected = reference(reference(hx, hz), hz)
torch.testing.assert_close(
    result.cpu(), expected, rtol=0, atol=0,
)
```

`x`、`z`、`middle` 和 `result` 在同步完成前均保持可达。消费者提交后立即把 `middle` 交给另一个写入者复用，仍可能破坏结果：wait 只处理等待方的执行顺序，不使内存自动成为不可修改的快照。

需要更细的完成点时，使用事件记录生产进度，再让消费 stream 等待该事件。

```python
ready = producer.record_event()
consumer.wait_event(ready)
with torch.zeus.stream(consumer):
    result = launch(middle, z)
consumer.synchronize()
```

上述片段替换前例中的 `consumer.wait_stream(...)` 及其后续消费部分。事件应记录在生产操作之后。若放在生产之前，它只证明旧工作完成，不能保护新输出。`stream.synchronize()` 等待该 stream，`torch.zeus.synchronize(device)` 等待目标设备；选择能够覆盖实际依赖的最小同步范围。

### JIT 的冷启动与缓存

首次遇到一个特化时，JIT 可能需要生成 TTIR、编译 `.zbin` 并加载。后续相同特化可命中进程内缓存；进程重启后还可命中持久缓存。这三种情况的耗时含义不同。

持久缓存默认位于 `$XDG_CACHE_HOME/zecc/jit`，未设置 `XDG_CACHE_HOME` 时使用 `$HOME/.cache/zecc/jit`。可以为单次任务指定独立目录。

```bash
mkdir -p build/jit-cache
ZECC_JIT_CACHE_DIR="$PWD/build/jit-cache" \
ZECC_JIT_LOG=1 \
python examples/vector_add.py --run-device
```

缓存身份包含源代码、工具链、签名、常量、属性、有效编译选项和运行时 ABI 等信息。修改 `BLOCK`、编译器或影响代码生成的选项，应形成新的特化。多进程共享持久缓存时，按 key 的文件锁用于避免相同缺失项的重复编译。

`ZECC_JIT_CACHE_DISABLE=1` 禁用持久缓存，但不会清空当前进程已经持有的内存缓存。比较编译工具链或测量首次编译成本时，应使用新进程并记录实际加载路径。清除磁盘缓存不能替代重启已经载入旧原生库的 Python 进程。

### 区分缓存身份与动态数据

缓存保存编译结果，不缓存某次运算的张量内容。相同特化处理第二批输入时，运行时仍重新打包指针和标量，并提交设备执行。结果没有随输入变化，首先检查 buffer 更新、别名和 Graph 输入地址，不能据此直接判断编译缓存失效。

| 调用变化 | 对特化身份的影响 |
|---|---|
| `numel` 从 2048 变成 2500，仍为 i32 | 仅改变运行时有效范围 |
| `alpha` 从 1.0 变成 0.5，仍为 FP32 | 仅改变本次标量值 |
| 输入地址改变，dtype 和布局契约不变 | 更新参数包中的地址 |
| `BLOCK` 从 1024 变成 512 | 生成另一组静态 tile |
| `CORE_NUM` 或 launch 核数改变 | 改变核数相关特化 |
| 源码、目标或有效编译选项改变 | 重新匹配或编译产物 |
| Python 整数超出 i32，推断为 i64 | 签名变化，不再是原 ABI |

Table: 向量示例中动态值与特化的关系

持久缓存通过按 key 的锁和原子写入发布完整产物；缓存不可用时，JIT 可继续通过编译获得内存中的产物。因此缓存目录故障可能表现为每次新进程都重新编译，而不是 kernel 数值错误。排查时先用日志辨认内存命中、磁盘命中和冷编译，再检查目录权限、空间和工具身份。

## ZECC 离线编译与 AOT {#zecc-and-aot}

### 编译一份独立 `.zbin`

ZECC 随 Zeus Triton wheel 提供。它接受 Triton `.py`、SimC `.c` 和 Zeus IR `.zelir`；本节从与上手章相同的 `examples/vector_add.py` 生成 `.zbin`。

```bash
mkdir -p build
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

这个命令将运行时参数的类型与编译期常量明确分开。`numel` 没有被写成常量，因此同一份产物可用于满足契约的不同有效长度；`BLOCK` 和 `CORE_NUM` 已经固化到特化中。

`--compile-only-zbin` 路径把命令行 constants 视为完整集合，不能只给出某个常量的局部修改而省略其他 constexpr。对象文件路径则可以使用模块中的 `signature` 与 `constants` 默认值，并在提供常量覆盖时合并。两条路径的差别应在构建脚本中显式保留。

| 产物 | 内容 | 用途 |
|---|---|---|
| `vector-add.zbin` | Zeus 设备二进制与参数描述 | 运行时加载、仿真或部署 |
| `vector-add.zbin.debug` | 源位置映射 sidecar | 指令与源码定位 |
| `vector-add.ttir.mlir` | `full` 模式保留的 TTIR | 对照前端语义与位置 |

Table: 开启完整源位置后生成的文件

编译成功证明源码能够通过当前编译路径生成产物。它尚未检查运行时指针、输入数据和结果；接下来仍需按所选运行环境执行并对拍。

### 对象文件与类型化启动器

需要 C/C++ 链接时，可以生成嵌入 `.zbin` 的对象文件，并同时导出类型化启动头文件。

```bash
mkdir -p build/include
zecc -c examples/vector_add.py \
  -o build/vector-add.o \
  --kernel-name add_bf16 \
  --gen-include-dir build/include
```

对于示例的签名，生成的启动接口具有下面的参数结构。

```c
zertRet_t zecc_add_bf16_launch(
    zertStream_t stream,
    int num_cores,
    void* out_ptr,
    const void* in1_ptr,
    const void* in2_ptr,
    int32_t numel,
    float alpha);
```

对象中嵌入的是设备程序，不是把整个 kernel 改写成 CPU 计算。宿主仍需通过配套运行时分配设备内存、准备输入、传入 stream 并处理返回码。编译与链接时使用 SDK 交付的头文件和库，不能把 CPU 指针直接当作设备地址传给启动器。

类型化头文件减少了参数顺序和宽度出错的机会，但 `const void*` 本身不会验证张量 dtype、shape 或容量。宿主封装仍需完成这些检查。对象输出没有携带 `.debug` sidecar 的约定位置；需要源位置诊断时，使用独立 `.zbin` 输出路径。

### 组织可复现的离线构建

一个源码文件可以包含 kernel、宿主参考函数和命令行入口。ZECC 导入模块以获得指定的 `@triton.jit` 函数，因此模块顶层应保持可导入：不要在顶层分配设备内存、启动 kernel 或根据当前输入修改全局常量。把演示流程放在 `if __name__ == "__main__":` 保护下，完整示例采用了这一结构。

模块中的 `signature` 按形式参数名给出非 constexpr 的类型，`constants` 给出全部 constexpr。构建时应检查四件事：入口名称存在，签名没有遗漏运行时参数，编译常量齐全，核数与计划部署一致。模块还需要能够在所选 Python 环境中导入 `torch`、`triton` 等依赖；Python 导入失败属于构建环境问题，不能通过修改设备编译选项修复。

为独立二进制添加元数据输出时，在前面的完整命令中加入：

```bash
--meta-out build/vector-add.meta.json \
--ttir-out build/vector-add.entry.mlir
```

这两个选项是命令片段，应追加到同一次 `zecc` 调用中。`--meta-out` 保存编译元信息；`--ttir-out` 明确指定入口 TTIR 的保存位置；它们不会执行输入对拍。需要追踪产物时可直接计算摘要：

```bash
sha256sum examples/vector_add.py \
  build/vector-add.zbin \
  build/vector-add.meta.json
```

复现目录应保存完整命令、源码摘要、签名、常量和最终产物摘要。若需保留诊断 sidecar，将其与对应二进制一起管理。不能只复制一个新 `.zbin` 覆盖旧文件，却继续使用旧 `.debug`：这会把正确指令映射到错误的源码位置。

### 从代表性调用生成 AOT bundle

AOT bundle 保存的是精确的运行时特化集合。先在 JIT 模式运行代表性输入，同时记录特化，再由工具离线构建 bundle。

```bash
mkdir -p build
ZECC_KERNEL_MODE=jit \
ZECC_AOT_RECORD_FILE="$PWD/build/kernel-specs.jsonl" \
python examples/vector_add.py --numel 2048 --run-device

ZECC_KERNEL_MODE=jit \
ZECC_AOT_RECORD_FILE="$PWD/build/kernel-specs.jsonl" \
python examples/vector_add.py --numel 2500 --run-device
```

这两次调用检查整块与尾块行为，但它们使用同一组 `BLOCK` 和 `CORE_NUM`。`numel` 是运行时参数，因此不应仅因有效长度不同就人为制作两份编译常量不同的 kernel。

```bash
python -m triton.tools.extra.zecc.build_aot_bundle \
  --specs build/kernel-specs.jsonl \
  --output build/vector-add-aot
```

部署时显式选择 bundle 和运行模式。

```bash
ZECC_KERNEL_MODE=aot \
ZECC_AOT_BUNDLE="$PWD/build/vector-add-aot" \
python examples/vector_add.py --numel 2500 --run-device
```

构建后保持 kernel 源文件及其特化身份稳定。若修改了源码、constexpr、签名或相关属性，应重新生成对应条目。AOT 并非只按函数名查找一份二进制；源摘要、目标、参数签名、常量、属性、核数和 ABI 都参与匹配。

加载器检查 manifest schema、运行时 ABI、产物 SHA-256、`.zbin` 结构、参数大小和核数。缺少某个特化时，纯 AOT 模式报错；hybrid 模式可以回到 JIT。上线前应选择明确的缺失行为，避免把现场编译时间混进稳定延迟。

| 模式 | 命中已有特化 | 缺少特化 |
|---|---|---|
| `jit` | 使用进程内或持久缓存 | 编译并缓存 |
| `aot` | 使用已加载 bundle | 报错 |
| `hybrid` | 优先使用 bundle | 进入 JIT 缓存路径 |

Table: 三种部署模式的行为

### 加载与检查 bundle

可以在服务启动阶段显式加载 bundle，让身份或文件损坏错误尽早暴露。

```python
import torch_zeus
import zecc_jit

keys = zecc_jit.load_aot_bundle("build/vector-add-aot")
if not keys:
    raise RuntimeError("the bundle contains no kernels")
print("registered kernels:", len(keys))
```

`load_aot_bundle(path)` 接受 bundle 目录或 manifest 路径，返回已注册身份的元组。目录方式读取其中的 `manifest.json`。加载器先校验文件组织、schema、ABI 和各条目的内容，再由原生接口检查二进制描述。只检查 JSON 能被解析不足以证明 bundle 可用。

每个条目描述以下四类信息：源码摘要与入口名称；类型、常量与属性；目标与核数；产物相对路径、SHA-256 与参数大小。产物路径必须位于 bundle 根目录中，指向根目录外部的文件不会被当作合法依赖。发布时整体复制 bundle，保持相对目录结构；不要只移动 manifest 或手工替换条目的摘要来掩盖产物变化。

`ZECC_AOT_BUNDLE` 还可使用平台路径分隔符连接多个 bundle；Linux 下为冒号。环境配置的 bundle 按规范化路径加载，当前进程不会因同一路径的文件被覆盖而自动重新装载。因此升级部署使用新进程，先验证新 bundle，再切换服务实例。

| 失败现象 | 应检查的契约 |
|---|---|
| 产物摘要不匹配 | 是否完整复制，是否替换了 `.zbin` |
| 参数大小或核数不匹配 | manifest 与二进制是否来自同一构建 |
| 找不到 AOT 特化 | 源码、常量、签名、属性是否变化 |
| 运行时 ABI 不兼容 | bundle 与运行时是否属于配套发布组合 |
| 文件加载成功但数值错误 | 输入布局、资源、动态参数和数值参考 |

Table: AOT 加载与执行故障的区分

源码修改包括不改变算法的文件编辑。部署记录使用精确源码身份时，仅重排源码文件也可能使旧条目无法匹配。服务应使用构建时归档的源码或已记录的摘要，不应在启动脚本中重新拼接一份“等价”kernel 文本。

### 收集足够的特化覆盖

采集代表性输入时，应按真正改变编译身份的维度设计集合。例如向量长度虽然动态，但宿主封装若针对短向量选择 `BLOCK=256`、长向量选择 `BLOCK=1024`，就存在两个特化。核数、dtype、layout 分支与融合开关也可能改变编译身份。反过来，单纯增加同一特化下的随机输入次数只扩大数值测试覆盖，不增加 AOT 条目覆盖。

推荐把部署验证分为两次运行：第一次用 JIT 收集并检查全部分支；第二次关闭运行时编译，只使用 AOT bundle 重放相同覆盖集合。第二次若缺少条目，应回到采集阶段补齐导致缺失的调用条件，不把部署模式临时改成 hybrid 来绕过问题。

当确实允许动态形状引入新特化时，可以选择 hybrid。此时仍需把首次编译延迟与稳定执行延迟分开记录，并确保现场具有完整编译依赖、可写工作目录与足够资源。hybrid 的回退策略解决特化缺失，不会修复不合法的 dtype、错误的指针或损坏的 bundle。

### 编译选项也是特化的一部分

Python 启动支持 `zeus_compiler_extra_flags`，用于为单个特化传入编译器选项。该属性在编译子进程中替换 `ZEUS_COMPILER_EXTRA_FLAGS`；显式空字符串表示清除进程默认选项，未提供则继承。

```python
add_bf16[(2,)](
    out, x, z, x.numel(), 1.0,
    BLOCK=1024, CORE_NUM=2,
    zeus_compiler_extra_flags="",
)
```

这对可复现基线很有用：一次评估不会意外继承终端中遗留的调优参数。使用非空值时，应将有效选项与特化一起归档，并确保原生运行时已链接支持该属性的 `libzecc_jit`。

当前 AOT schema 2 会记录每个条目的有效编译选项。若未通过属性记录，却在环境中遗留非空 `ZEUS_COMPILER_EXTRA_FLAGS`，bundle 构建会拒绝这种无法完整表达身份的输入。修复方式是把所需选项显式纳入特化记录，而不是只在构建机器上手工设置环境。

### Graph 捕获前准备产物

Graph 重放减少的是重复提交的工作，不重新执行 Python 控制流来选择新的 constexpr。捕获时必须已经有可用的 kernel 产物：先预加载 AOT，或者在捕获开始前完成相同特化的 JIT 预热。启动器在捕获期间只查找已经准备好的产物；缺少产物会报告需要预加载或预热，不会在捕获中临时编译。

下面给出固定地址输入的最小流程。它要求配套运行时提供 Zeus Graph；与普通 launch 一样，输入为一维连续 BF16、核数为 2。前一代码块完成分配和预热，后一代码块在同一进程中捕获、修改输入内容并重放。

```python
import torch
import torch_zeus
from torch_zeus.zeus import ZEUSGraph, graph, Stream
from examples.vector_add import add_bf16, make_inputs
from examples.vector_add import reference

hx, hz = make_inputs(2500)
x, z = hx.to("zeus:0"), hz.to("zeus:0")
out = torch.empty_like(x)
capture_stream = Stream(device=0)

def submit():
    add_bf16[(2,)](
        out, x, z, 2500, 1.0,
        BLOCK=1024, CORE_NUM=2,
    )

submit()
torch.zeus.synchronize(0)
```

```python
recorded = ZEUSGraph()
with graph(recorded, stream=capture_stream):
    submit()

changed = (hx.float() * 2).to(torch.bfloat16)
with torch.zeus.stream(capture_stream):
    x.copy_(changed)
    recorded.replay()
capture_stream.synchronize()
torch.testing.assert_close(
    out.cpu(), reference(changed, hz), rtol=0, atol=0,
)
recorded.reset()
```

此处更新的是 `x` 所在内存的内容，`x` 的设备地址、形状和类型不变；`2500` 与 `1.0` 是被捕获调用中的标量，不会因为宿主后来给同名 Python 变量赋值而自动更新。若要改变标量、shape 或 buffer 地址，应采用配套 Graph 的明确更新机制或重新捕获，不依赖 Python 闭包的重新求值。

捕获失败后，不应把部分记录当作完整 Graph 使用。保留原始异常和捕获阶段，释放失败对象，再从稳定的分配与预热阶段重新开始。重放输出要用至少两组不同输入检查，单次对拍无法识别输入未更新或误复用录制结果的问题。

## 查看 IR、汇编与二进制 {#compiler-artifacts}

### 沿编译阶段定位问题

编译阶段逐步把高层张量语义转成设备命令。阅读中间表示的目的，是找出某个性质在哪个阶段发生变化，例如尾块保护是否仍在、一个输入被复制了几次、归约轴是否正确，以及同步是否连接了正确的生产者和消费者。

![Triton 源码到 Zeus 设备产物的主要编译阶段](assets/compiler-pipeline.svg){#fig-compiler-pipeline width=100%}

| 阶段 | 主要表示 | 优先检查的问题 |
|---|---|---|
| TTIR | Triton 张量、指针和控制流 | dtype、静态 shape、边界与 constexpr |
| Linalg / MemRef | 循环、结构化运算与存储视图 | 索引、stride、copy 和归约关系 |
| ZeusHigh | VP、PE、DT 与存储资源 | 操作分工、布局、复用与并行策略 |
| ZeusLow | 设备配置、命令和同步 | 依赖、指令顺序与资源生命周期 |
| 汇编 / `.zbin` | 可读指令或设备程序 | 最终参数、命令和可执行产物身份 |

Table: 各阶段的阅读重点

对于同一个错误，先检查最早失去正确语义的阶段。若 TTIR 的 stride 已经错误，继续调整后端流水不会修复输入地址；若高层语义正确而执行后出现数据损坏，则应继续检查低层搬运、别名与同步。

### 在源码开发环境导出中间产物

配套源码环境提供 `zeus_compile.py`。下面假设 `ZEUS_SRC` 已指向与当前工具链配套的 `triton_shared` 源码目录，并从本指南根目录执行。

```bash
python "$ZEUS_SRC/zeus_compile.py" \
  examples/vector_add.py \
  -S -o build/vector-add.S \
  --dump-dir build/vector-add-ir \
  --num-vp 1 --num-dt 1
```

`-S` 导出汇编，`--dump-dir` 保留中间文件。显式写出 VP 和 DT 数量，有助于避免环境中的默认值影响基线。需要停在特定阶段时，可以使用该工具的 `--stop-after`；当前入口接受 `ttir`、`zeushigh` 和 `asm`。

IR 文件是诊断材料，不应脱离原 kernel、签名和常量单独解读。保存一次复现时，把输入源码、完整命令、工具版本和生成的 `.zbin` 摘要放在同一记录中；后续比较只调整一个明确因素。

### 源位置 sidecar

ZECC 的 `--loc=source` 保留源位置映射，`--loc=full` 还保留入口 TTIR。运行时 JIT 使用对应环境变量。

```bash
mkdir -p build/jit-debug
ZECC_JIT_LOC=full \
ZECC_JIT_DUMP_DIR="$PWD/build/jit-debug" \
python examples/vector_add.py --run-device
```

`kernel.zbin.debug` 保存源位置映射；完整模式另存 `kernel.ttir.mlir`。这些文件按 key 写入 dump 目录。缓存命中时也会物化所需 sidecar；如果已请求调试信息而编译器无法生成，应修正工具链配套。

源位置通常记录源码路径。向其他团队或客户提供诊断包前，应使用编译工具支持的路径前缀映射，并检查最终 sidecar。正文与算例使用可移植的相对路径，诊断产物也应遵循相同的交付约定。

### 建立一次可比较的诊断

先固定输入源码与完整编译命令，再在一个新目录中生成产物。诊断目录与可执行缓存有不同用途：缓存可以被运行时自动命中和重建，诊断目录应保存这次观察对应的 TTIR、汇编、二进制与错误日志，便于之后对照。

对于 `--loc=full`，首先确认 `.zbin.debug` 和 TTIR 文件存在并与二进制同时生成；其次确认源码路径能定位到已归档版本；最后再把一条异常指令沿位置映射追溯到原 kernel。缺失源位置不是“优化后没有这条语句”的证据，也可能是编译时没有启用位置保留。

路径前缀映射由 `zeus-compiler` 的 `--loc-prefix-map=old=new` 处理。例如通过单次启动的编译选项把构建根目录映射为稳定的源码根；`old` 应取实际构建路径，`new` 应取诊断包约定的路径。映射只改变诊断位置的表达，不移动源文件，也不改变运行时加载路径。

当两份产物的性能不同，先比较源码、签名、常量、目标和有效选项是否一致，再比较编译阶段。只对比汇编文件大小或最终指令数，不能证明哪条路径更快；搬运量、依赖、并发和资源占用必须结合实际执行记录分析。

### 按失败阶段缩小范围

错误应按首次出现的位置分类。导入报错先检查 Python 与模块依赖；TTIR 生成失败先检查 constexpr、形状和类型；设备转换失败检查相应操作与布局；产物加载失败检查文件身份和 ABI；提交后同步报错再检查动态地址、资源有效期与设备日志。

不要把所有错误都归结为编译器不支持。例如 `other` 没有搭配 mask、块指针错误地传入 `mask`、两个设备的 Tensor 混用、AOT 缺失当前 `BLOCK`，都有明确的接口修复方法。保留第一个完整错误信息，避免被上层重试逻辑覆盖成笼统的“执行失败”。

一个有效复现包只需包含能触发问题的最小 kernel、准确的签名与常量、输入生成方式、完整命令、预期行为及实际产物身份。若问题依赖某个值，应保留这个值而不只保留 shape；若依赖第二次运行或跨 stream 顺序，则复现流程必须保留相同的调用顺序。

## 算子封装与集成 {#operator-integration}

### 宿主封装的职责

一个可复用算子不能只暴露裸 kernel。宿主封装应在启动前检查输入设备、dtype、shape、stride 和合法输入域，选择已支持的特化，分配输出，并约定异步完成方式。

向量加法的封装采用独立输出，限定一维连续 BF16 输入，检查两个输入长度一致。空张量直接返回空输出，不提交无意义的设备搬运。这样的检查让错误发生在接口边界，避免直到低层访存阶段才暴露。

若支持原地更新，应明确哪些输入可以与输出别名。`out=x`、`out=z` 与任意重叠切片是不同条件：前两者可能存在可证明的逐元素顺序，任意重叠切片则可能覆盖后续仍要读取的数据。不要因为一个原地样例通过，就对所有别名关系放行。

### 参数 ABI 与资源信息 {#argument-resources}

原生启动器使用逐参数指针数组。`args[i]` 指向第 i 个参数值的存储位置；如果参数本身是设备指针，该位置中存放的是设备地址。把“参数值的地址”和“张量数据的地址”混为一谈，会使每个参数都按错误层级解引用。

![参数数组指向参数值，指针参数的值再指向设备数据](assets/argument-abi.svg){width=100%}

运行时标量的宽度必须与签名一致。`i32` 长度使用 32 位整数，`fp32` 缩放使用 32 位浮点数；不能仅因为宿主 C++ 的 `long` 在某台机器上是 64 位，就把它无条件用于所有长度参数。

一个参数的 ABI 描述回答“这些字节如何解释”，资源描述回答“地址属于哪块分配，以及这次执行会读还是写”。二者互相配合，但不能互相替代。把 dtype 写对不能保证分配仍然有效；声明 `access="write"` 也不能修复错误的 stride 或物理布局。

| 实参类别 | 参数包内容 | 宿主需要保持的条件 |
|---|---|---|
| 普通 Tensor 指针 | 单个设备地址 | dtype、容量、设备与生命周期 |
| i32 / FP32 标量 | 4 字节参数值 | 类型、范围与数值语义 |
| i64 / FP64 标量 | 8 字节参数值 | 对应签名及具体计算路径 |
| 未包装 LocalMem | 各核地址的描述 | 核数、分片与 LDG 布局 |
| LocalMem 原始指针 | 单核地址及资源信息 | 显式读写权限与布局契约 |

Table: 参数 ABI 与资源契约

未包装的 LocalMem 对象沿兼容的 `T_WEIGHT` 参数描述进入启动器。该描述包含每个核的 LDG 地址，kernel 的 `*i64` 形式参数对应地址表；kernel 先按 `program_id(0)` 取得本核地址，再构造元素指针。因此 `*i64` 在这里描述的是地址表元素，不代表真实权重或 KV cache 必须使用 INT64 数据类型。

`T_WEIGHT` 与 `memory_type="weight"` 保留了接口的历史命名。LDG 是承载权重和 KV cache 的 3D DRAM；实际张量还可能使用 `kcache`、`vcache` 等专用物理组织。内存分配类型、参数描述和访存模式必须与算子的布局契约一起选择，不能因为它们最终都涉及设备地址就混用。

### LocalMem 原始指针与读写声明

原始指针接口用于 kernel 直接接收某个单核 LocalMem 分配的元素指针，而不是地址表的情形。

```python
import zecc_jit

zecc_jit.require_localmem_resources()
payload = zecc_jit.localmem_ptr(cache, access="readwrite")
```

这里的 `cache` 必须是已经按目标算子布局分配的单核 LocalMem 对象。片段只构造指针包装，不分配空间，也不把线性 Tensor 重新排布到 LDG。`require_localmem_resources()` 检查原生启动器是否提供对应资源能力；若报告版本不匹配，应使用配套原生库，不要移除检查后继续提交。

`localmem_ptr(value, *, access="read")` 返回保留 owner 的指针包装。它可作为 `kernel[grid](...)` 的一个普通指针实参；与未包装对象相比，参数 ABI 已经改变。需要地址表的示例，如本书的 KV cache 写入 kernel，不能直接用该包装替代其 `cache_page_ptr_list`。

`access` 必须准确描述本次调用：`read` 只读；`write` 写入而不依赖旧值；`readwrite` 读取旧数据后更新。资源系统将同一分配的多个使用合并，并在写入后使相关线性视图缓存失效。如果实际写入却声明为只读，后续线性视图可能继续使用旧内容；访问声明不是可随意省略的性能提示。

需要显式参数包时，`tensor_ptr` 接受资源元信息：

```python
arg = zecc_jit.tensor_ptr(
    payload, "cache_ptr", direction="inout",
    resource={"kind": "localmem", "access": "readwrite"},
)
```

该记录不改变原始指针的 ABI。参数名 `cache_ptr` 必须对应实际 kernel 定义；例中沿用上一片段的 `payload`，不会产生另一次分配。多核分片对象应分别处理各个 `core_tensors` 分配；不能把整个多核对象包装成一个单核指针，掩盖地址与核的对应关系。

资源启动器检查分配是否仍有效、是否属于启动设备，并保留完成所需的分配引用。普通异步提交会记录 stream 使用；Graph 捕获则将引用附着到 Graph，并在相应重放之前处理写入视图缓存。上述管理保护分配有效性，不自动建立跨 stream 的读写先后关系。生产者和消费者仍须使用本章给出的 wait 或事件依赖。

### 外部访问保证与参数属性

某些窄元素访问需要对完整外部存储字进行读取或读改写。逻辑上只访问一个 BF16 元素，不一定意味着底层只接触这两个字节。若调用方确实提供了额外的可读范围或独占范围，可在源码模块级声明 `arg_attrs`。

```python
arg_attrs = {
    "in1_ptr": {"zeus.external_word_readable": 1},
    "out_ptr": {"zeus.external_word_exclusive_rmw": 1},
    "alpha": {"zeus.runtime_fp_narrowing_safe": 1},
}
```

这是契约形式示例，不应直接添加到向量加法源码。第一项声明每个访问元素周围的完整外部字可读；第二项声明当前 kernel 独占读改写触及的完整外部字；第三项声明所有运行时 `alpha` 值都满足相关窄化转换的安全条件。没有证明这些前提时，应保持属性缺省。

声明独占范围时要考虑 tile 或核之间的边界。两个核分别更新同一个外部字内的不同 BF16 元素，虽然逻辑元素互不重叠，仍不满足“各自独占完整外部字”。修复应从任务划分、对齐或单一写入者入手，不能通过填写属性取消真实的数据竞争。

属性值必须为整数 `1`；参数名称、属性名称和参数类型均受检查。指针属性不能用于标量，窄化安全属性限定在 FP32 运行时标量。模块中的多个 `jit_specs` 共享这组属性；若入口需要互不相同的保证，应拆分模块或重新组织公共契约。属性属于编译身份的一部分，修改后需要重新生成部署产物。

### 接入框架后的回归

先测试宿主封装，再把它接入框架算子注册或模型层。回归至少覆盖直接调用、框架调用和重复调用三条路径，确保注册后的 dtype 转换、连续化、输出分配和 stream 选择没有改变独立算例的行为。

对需要动态形状的模型，应记录实际出现的 shape、stride 和 constexpr 组合，再检查 AOT 覆盖或 JIT 缓存行为。对使用 Graph 的模型，应使用变化的输入数据执行 replay，确认输出依赖的是本次输入，而不是录制时遗留的结果。

最后分别测量冷编译、稳定 kernel 执行和框架端到端耗时。封装引入的数据复制、额外同步或多次启动都可能成为实际开销；设备 kernel 本身加速并不自动意味着框架调用同样加速。
