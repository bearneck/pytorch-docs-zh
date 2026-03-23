---
orphan: true
---

(torch.compiler_troubleshooting_old)=

# PyTorch 2.0 故障排除（旧版）

**作者**: [Michael Lazos](https://github.com/mlazos)

:::{note}
本文档已过时，现在主要作为如何运行 `torch.compile` 最小化工具的主要参考资料。
请参阅[更新的故障排除文档](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html)。
此外，还有一份更[全面的 torch.compile 手册](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab)可用。
:::

我们正在积极开发调试工具、性能分析器，并改进我们的错误和警告信息。下表列出了可用的工具及其典型用途。如需更多帮助，请参阅 {ref}`diagnosing-runtime-errors`。

```{eval-rst}
.. list-table:: 标题
   :widths: 25 25 50
   :header-rows: 1

   * - 工具
     - 用途
     - 使用方法
   * - 信息日志
     - 查看编译的摘要步骤
     - ``torch._logging.set_logs(dynamo = logging.INFO)`` 或 ``TORCH_LOGS="dynamo"``
   * - 调试日志
     - 查看编译的详细步骤（打印每个跟踪的指令）
     - ``torch._logging.set_logs(dynamo = logging.DEBUG)`` 和
       ``torch._dynamo.config.verbose = True``，或 ``TORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1``
   * - 适用于任何后端的最小化工具
     - 为任何后端找到能复现错误的最小子图
     - 设置环境变量 ``TORCHDYNAMO_REPRO_AFTER="dynamo"``
   * - 适用于 ``TorchInductor`` 的最小化工具
     - 如果已知错误发生在 ``AOTAutograd`` 之后，找到在 ``TorchInductor`` 降级过程中能复现错误的最小子图
     - 设置环境变量 ``TORCHDYNAMO_REPRO_AFTER="aot"``
   * - Dynamo 精度最小化工具
     - 当您怀疑问题出在 ``AOTAutograd`` 时，找到能复现 eager 模式模型与优化模型之间精度问题的最小子图
     - ``TORCHDYNAMO_REPRO_AFTER="dynamo" TORCHDYNAMO_REPRO_LEVEL=4``
   * - Inductor 精度最小化工具
     - 当您怀疑问题出在后端（例如 inductor）时，找到能复现 eager 模式模型与优化模型之间精度问题的最小子图。
       如果此工具无效，请尝试使用 Dynamo 精度最小化工具。
     - ``TORCHDYNAMO_REPRO_AFTER="aot" TORCHDYNAMO_REPRO_LEVEL=4``
   * - ``torch._dynamo.explain``
     - 查找图中断并显示其原因
     - ``torch._dynamo.explain(fn)(*inputs)``
   * - 记录/回放
     - 记录并回放帧以复现图捕获期间的错误
     - ``torch._dynamo.config.replay_record_enabled = True``
   * - TorchDynamo 函数名过滤
     - 仅编译具有给定名称的函数，以减少调试问题时的干扰
     - 设置环境变量 ``TORCHDYNAMO_DEBUG_FUNCTION=<name>``
   * - TorchInductor 调试日志
     - 打印 TorchInductor 的通用调试信息以及生成的 Triton/C++ 代码
     - ``torch._inductor.config.debug = True``
   * - TorchInductor 追踪
     - 显示每个 TorchInductor 阶段所花费的时间 + 输出代码和图可视化
     - 设置环境变量 TORCH_COMPILE_DEBUG=1 或
       ``torch._inductor.config.trace.enabled = True``
```

除了信息和调试日志外，您还可以使用 [torch.\_logging](https://pytorch.org/docs/main/logging.html) 进行更细粒度的日志记录。

(diagnosing-runtime-errors)=
## 诊断运行时错误

从高层次来看，TorchDynamo 堆栈包括从 Python 代码进行的图捕获（TorchDynamo）和一个后端编译器。例如，一个后端编译器可能包括反向图追踪（AOTAutograd）和图降级（TorchInductor）\*。错误可能发生在堆栈的任何组件中，并将提供完整的堆栈跟踪。

要确定错误发生在哪个组件中，您可以使用信息级别日志 `torch._logging.set_logs(dynamo = logging.INFO)` 或 `TORCH_LOGS="dynamo"`，并查找 `Step #: ...` 输出。日志在每个步骤的开始和结束时记录，因此错误对应的步骤应该是最近记录的、其结束尚未被记录的步骤。这些步骤对应堆栈的以下部分：

| 步骤 | 组件           |
| ---- | -------------- |
| 1    | TorchDynamo    |
| 2    | 编译器后端     |
| 3    | TorchInductor  |

如果信息日志不够详细，您可以使用可用的后端选项。这些选项包括：

- `"eager"`：仅运行 TorchDynamo 前向图捕获，然后使用 PyTorch 运行捕获的图。这可以指示错误是否由 TorchDynamo 引发。
- `"aot_eager"`：运行 TorchDynamo 捕获前向图，然后运行 AOTAutograd 追踪反向图，不进行任何额外的后端编译器步骤。然后使用 PyTorch eager 运行前向和反向图。这对于将问题缩小到 AOTAutograd 很有用。

缩小问题范围的一般步骤如下：

1. 使用 `"eager"` 后端运行您的程序。如果错误不再发生，则问题出在正在使用的后端编译器中（如果使用 TorchInductor，请继续步骤 2。如果不是，请参阅 {ref}`minifying-backend-compiler-errors`）。如果使用 `"eager"` 后端时错误仍然发生，则错误是由于 {ref}`torchdynamo-errors`。
2. 仅当使用 `TorchInductor` 作为后端编译器时才需要此步骤。使用 `"aot_eager"` 后端运行模型。如果此后端引发错误，则错误发生在 AOTAutograd 追踪期间。如果使用此后端时错误不再发生，则 {ref}`minifying-torchinductor-errors`。

以下各节将分析这些情况。

:::{note}
TorchInductor 后端包含 AOTAutograd 追踪和 TorchInductor 编译器本身。我们将通过以下方式区分：将 `TorchInductor` 称为后端，而将 TorchInductor 降级（lowering）阶段称为对 AOTAutograd 追踪的计算图进行降级处理的阶段。
:::

(torchdynamo-errors)=

### Torchdynamo 错误

如果使用 `"eager"` 后端时出现错误，那么错误很可能源自 TorchDynamo。以下是一个会生成错误的示例代码。

```py
import torch

import torch._dynamo as dynamo


def test_assertion_error():
    y = torch.ones(200, 200)
    z = {y: 5}
    return z

compiled_test_assertion_error = torch.compile(test_assertion_error, backend="eager")

compiled_test_assertion_error()
```

上述代码会生成以下错误：

```
torch._dynamo.convert_frame: [ERROR] WON'T CONVERT test_assertion_error /scratch/mlazos/torchdynamo/../test/errors.py line 26
due to:
Traceback (most recent call last):
  File "/scratch/mlazos/torchdynamo/torchdynamo/symbolic_convert.py", line 837, in BUILD_MAP
    assert isinstance(k, ConstantVariable) or (
AssertionError

from user code:
   File "/scratch/mlazos/torchdynamo/../test/errors.py", line 34, in test_assertion_error
    z = {y: 5}

Set torch._dynamo.config.verbose=True for more information
==========
```

如消息所示，您可以设置 `torch._dynamo.config.verbose=True` 来获取 TorchDynamo 和用户代码中错误的完整堆栈跟踪。除了这个标志，您还可以通过 `torch._logging.set_logs(dynamo = logging.INFO)` 或 `TORCH_LOGS="dynamo"` 来设置 TorchDynamo 的 `log_level`。这些级别包括：

- `logging.DEBUG` 或 `TORCH_LOGS="+dynamo"`：打印遇到的每条指令，以及下面列出的所有日志级别。
- `logging.INFO`：打印每个被编译的函数（原始和修改后的字节码）以及捕获的计算图，以及下面列出的所有日志级别。
- `logging.WARNING`（默认）：打印图中断（graph breaks），以及下面列出的所有日志级别。
- `logging.ERROR`：仅打印错误。

如果模型非常大，日志可能会变得非常庞大。如果错误发生在模型 Python 代码的深处，仅执行发生错误的帧（frame）可以更容易地进行调试。有两个工具可以实现这一点：

- 将环境变量 `TORCHDYNAMO_DEBUG_FUNCTION` 设置为所需的函数名，将仅对具有该名称的函数运行 torchdynamo。
- 启用记录/回放工具（设置 `torch._dynamo.config.replay_record_enabled = True`），该工具在遇到错误时会转储执行记录。然后可以回放此记录，仅运行发生错误的帧。

### 诊断 TorchInductor 错误

如果错误在使用 `"eager"` 后端时不出现，那么错误源是后端编译器（[示例错误](https://gist.github.com/mlazos/2f13681e3cc6c43b3911f336327032de)）。TorchDynamo 有[不同的选择](./user_guide/torch_compiler/torch.compiler.md)作为后端编译器，其中 TorchInductor 满足大多数用户的需求。本节以 TorchInductor 作为主要示例，但有些工具也可用于其他后端编译器。

以下是我们关注的部分堆栈：

选择 TorchInductor 作为后端时，AOTAutograd 用于从 torchdynamo 捕获的前向图生成反向图。需要注意的是，错误可能发生在此追踪过程中，也可能发生在 TorchInductor 将前向图和反向图降级为 GPU 代码或 C++ 的过程中。一个模型通常由成百上千个 FX 节点组成，因此缩小到发生问题的确切节点可能非常困难。幸运的是，有一些工具可以自动将这些输入图最小化到导致问题的节点。第一步是确定错误是发生在使用 AOTAutograd 追踪反向图期间，还是发生在 TorchInductor 降级期间。如上文步骤 2 所述，可以使用 `"aot_eager"` 后端来单独运行 AOTAutograd 而不进行降级。如果使用此后端时错误仍然出现，这表明错误发生在 AOTAutograd 追踪期间。

以下是一个示例：

```py
import torch

import torch._dynamo as dynamo

model = torch.nn.Sequential(*[torch.nn.Linear(200, 200) for _ in range(5)])

def test_backend_error():

    y = torch.ones(200, 200)
    x = torch.ones(200, 200)
    z = x + y
    a = torch.ops.aten._foobar(z)  # 会出错的虚拟函数
    return model(a)


compiled_test_backend_error = torch.compile(test_backend_error, backend="inductor")
compiled_test_backend_error()
```

运行此代码应该会得到以下错误，并附带更长的堆栈跟踪：

```
Traceback (most recent call last):
  File "/scratch/mlazos/torchdynamo/torchinductor/graph.py", line 246, in call_function
    return lowerings[target](*args, **kwargs)
  File "/scratch/mlazos/torchdynamo/torchinductor/lowering.py", line 185, in wrapped
    return decomp_fn(*args, **kwargs)
  File "/scratch/mlazos/torchdynamo/torchinductor/lowering.py", line 810, in _foobar
    assert False
AssertionError
...
```

[完整堆栈跟踪的错误](https://gist.github.com/mlazos/d6947854aa56d686800259a164c62100)

如果您随后将 `torch.compile(backend="inductor")` 更改为 `torch.compile(backend="aot_eager")`，它将无错误运行，因为[问题](https://github.com/pytorch/torchdynamo/blob/d09e50fbee388d466b5252a63045643166006f77/torchinductor/lowering.py#:~:text=%23%20This%20shouldn%27t%20be,assert%20False)出在 TorchInductor 降级过程中，而不是在 AOTAutograd 中。

(minifying-torchinductor-errors)=

### 最小化 TorchInductor 错误

从这里开始，让我们运行最小化工具来获取最小化复现。设置环境变量 `TORCHDYNAMO_REPRO_AFTER="aot"`（或直接设置 `torch._dynamo.config.repro_after="aot"`）将生成一个 Python 程序，该程序将 AOTAutograd 生成的图缩减为能复现错误的最小子图。（下面有一个示例，我们最小化 TorchDynamo 生成的图）使用此环境变量运行程序应显示[几乎相同的输出](https://gist.github.com/mlazos/0458ab828aa403c779fe73c012aa5982)，并额外有一行指示 `minifier_launcher.py` 的写入位置。输出目录可通过将 `torch._dynamo.config.base_dir` 设置为有效的目录名来配置。最后一步是运行最小化工具并检查其是否成功运行。成功运行的示例如[此](https://gist.github.com/mlazos/e6ea41ccce68a7b1b8a7a09acb1b206a)。如果最小化工具成功运行，它会生成可运行的 Python 代码来复现确切的错误。对于我们的示例，代码如下：

```python
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

# torch version: 1.13.0a0+gitfddfc44
# torch cuda version: 11.6
# torch git version: fddfc4488afb207971c54ad4bf58130fdc8a4dc5


# CUDA Info:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2022 NVIDIA Corporation
# Built on Thu_Feb_10_18:23:41_PST_2022
# Cuda compilation tools, release 11.6, V11.6.112
# Build cuda_11.6.r11.6/compiler.30978841_0

# GPU Hardware Info:
# NVIDIA A100-SXM4-40GB : 8

from torch.nn import *

class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, add):
        _foobar = torch.ops.aten._foobar.default(add);  add = None
        return (_foobar,)

args = [((200, 200), (200, 1), torch.float32, 'cpu')]
args = [rand_strided(shape, stride, dtype, device) for shape, stride, dtype, device in args]
mod = make_fx(Repro())(*args)
from torch._inductor.compile_fx import compile_fx_inner

compiled = compile_fx_inner(mod, args)
compiled(*args)
```

`Repro` 模块的 `forward` 方法包含了导致问题的确切操作符。提交问题时，请包含任何最小化的复现代码以帮助调试。

(minifying-backend-compiler-errors)=

### 最小化后端编译器错误

对于 TorchInductor 以外的后端编译器，查找导致错误的子图的过程与 {ref}`minifying-torchinductor-errors` 中的步骤几乎相同，但有一个重要的注意事项。即，最小化工具现在将运行在 TorchDynamo 追踪的图上，而不是 AOTAutograd 的输出图上。让我们通过一个示例来了解。

```py
import torch

import torch._dynamo as dynamo

model = torch.nn.Sequential(*[torch.nn.Linear(200, 200) for _ in range(5)])
# 一个玩具编译器，如果图中包含 relu 就会失败
def toy_compiler(gm: torch.fx.GraphModule, _):
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            assert False

    return gm


def test_backend_error():
    y = torch.ones(200, 200)
    x = torch.ones(200, 200)
    z = x + y
    a = torch.relu(z)
    return model(a)


compiled_test_backend_error = torch.compile(test_backend_error, backend=toy_compiler)
compiled_test_backend_error()
```

为了在 TorchDynamo 追踪前向图后运行代码，你可以使用 `TORCHDYNAMO_REPRO_AFTER` 环境变量。使用 `TORCHDYNAMO_REPRO_AFTER="dynamo"`（或 `torch._dynamo.config.repro_after="dynamo"`）运行此程序应产生[此输出](https://gist.github.com/mlazos/244e3d5b53667e44078e194762c0c92b)，并在 `{torch._dynamo.config.base_dir}/repro.py` 中生成以下代码。

:::{note}
TORCHDYNAMO_REPRO_AFTER 的另一个选项是 `"aot"`，它将在生成反向图后运行最小化工具。
:::

```python
import torch
import torch._dynamo as dynamo
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch._dynamo.debug_utils import run_fwd_maybe_bwd

from torch.nn import *

class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, add):
        relu = torch.relu(add);  add = None
        return (relu,)


mod = Repro().cuda()
opt_mod = torch.compile(mod, backend="None")


args = [((200, 200), (200, 1), torch.float32, 'cpu', False)]
args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]


with torch.cuda.amp.autocast(enabled=False):
    ref = run_fwd_maybe_bwd(mod, args)
    res = run_fwd_maybe_bwd(opt_mod, args)
```

最小化工具成功地将图缩减到在 `toy_compiler` 中引发错误的操作符。与 {ref}`minifying-torchinductor-errors` 中过程的另一个区别是，最小化工具在遇到后端编译器错误后会自动运行。成功运行后，最小化工具将 `repro.py` 写入 `torch._dynamo.config.base_dir`。

## 性能分析

### 访问 TorchDynamo 性能分析器

TorchDynamo 有一个内置的统计函数，用于收集和显示每个编译阶段所花费的时间。这些统计信息可以在执行 Torch._Dynamo 后通过调用 `torch._dynamo.utils.compile_times()` 来访问。默认情况下，这会返回一个字符串，表示按名称划分的每个 TorchDynamo 函数所花费的编译时间。

### 使用 TORCH_COMPILE_DEBUG 调试 TorchInductor

TorchInductor 有一个内置的统计和追踪功能，用于显示每个编译阶段所花费的时间、输出代码、输出图可视化和 IR 转储。这是一个调试工具，旨在使理解和排查 TorchInductor 内部问题更容易。

让我们用以下测试程序 (`repro.py`) 运行一个示例：

```
import torch
```

@torch.compile()
def test_model(x):
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.LayerNorm(10),
        torch.nn.ReLU(),
    )
    return model(x)


y = test_model(torch.ones(10, 10))
```

设置环境变量 `TORCH_COMPILE_DEBUG=1` 将创建一个调试跟踪目录，默认情况下该目录位于当前目录下，名为 `torch_compile_debug`（可以通过 `torchdynamo` 配置字段 `debug_dir_root` 以及环境变量 `TORCH_COMPILE_DEBUG_DIR` 覆盖此设置）。在此目录内，每次运行都会有一个单独的文件夹，以运行的时间戳和进程 ID 命名：

```
$ env TORCH_COMPILE_DEBUG=1 python repro.py
$ cd torch_compile_debug
$ ls
run_2023_03_01_08_20_52_143510-pid_180167
```

在运行文件夹中，会有一个 `torchdynamo` 目录，其中包含调试日志；以及一个 `torchinductor` 文件夹，其中为每个已编译的内核包含一个子文件夹，存放着 inductor 的调试产物。

```
$ cd
run_2023_03_01_08_20_52_143510-pid_180167
$ ls
torchinductor  torchdynamo
```

进一步进入 `torchinductor` 目录，`\*.log` 文件是编译的 AOT Autograd 阶段的日志，`model__0_forward_1.0` 包含 inductor 的调试产物。

```
$ cd torchinductor
$ ls
aot_model___0_debug.log  model__0_forward_1.0
$ cd model__0_forward_1.0
$ ls
debug.log  fx_graph_readable.py  fx_graph_runnable.py  fx_graph_transformed.py  ir_post_fusion.txt  ir_pre_fusion.txt  output_code.py
```

以下是内容摘要：

- `fx_graph_readable.py` 和 `fx_graph_runnable.py` 是 inductor 接收到的 `fx_graph` 的可读和可运行版本。
- `fx_graph_transformed.py` 是 inductor 运行所有 fx 传递后的 fx 图。
- `ir\*.txt` 是融合前后的 inductor 中间表示。
- `output_code.py` 是子图的已编译 triton 内核。

以下是测试程序的[示例调试目录内容](https://gist.github.com/jansel/f4af078791ad681a0d4094adeb844396)：

```
import torch

@torch.compile()
def test_model(x):
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.LayerNorm(10),
        torch.nn.ReLU(),
    )
    return model(x)


y = test_model(torch.ones(10, 10))
```

该调试跟踪中的每个文件都可以通过 `torch._inductor.config.trace.*` 启用或禁用。配置文件和图表默认都是禁用的，因为生成它们的开销较大。

这种新调试格式中的单个节点如下所示：

```
buf1: SchedulerNode(ComputedBuffer)
buf1.writes =
    {   MemoryDep(name='buf1', index=0, size=()),
        MemoryDep(name='buf1', index=0, size=(s0,))}
buf1.unmet_dependencies = {MemoryDep(name='buf0', index=c0, size=(s0,))}
buf1.met_dependencies = {MemoryDep(name='primals_2', index=c0, size=(s0,))}
buf1.group.device = cuda:0
buf1.group.iteration = (1, s0)
buf1.sizes = ([], [s0])
class buf1_loop_body:
    var_ranges = {z0: s0}
    index0 = z0
    index1 = 0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf0', get_index, False)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('primals_2', get_index_1, False)
        add = ops.add(load, load_1)
        get_index_2 = self.get_index('index1')
        reduction = ops.reduction('buf1', torch.float32, torch.float32, 'sum', get_index_2, add)
        return reduction
```

更多示例请参阅[示例调试目录输出](https://gist.github.com/jansel/f4af078791ad681a0d4094adeb844396)。

% _内存性能分析
% ----------------
%
% 待补充

### 图中断

给定如下程序：

```python
def some_fun(x):
    ...

compiled_fun = torch.compile(some_fun, ...)
...
```

TorchDynamo 将尝试将 `some_fun` 中的所有 torch/张量操作编译到单个 FX 图中，但它可能无法将所有内容捕获到一个图中。

某些图中断原因是 TorchDynamo 无法克服的，并且不容易修复。例如，调用 torch 之外的 C 扩展对 torchdynamo 是不可见的，并且可能执行任意操作，而 TorchDynamo 无法引入必要的守卫（参见 {ref}`making-dynamo-sound-guards`）来确保编译后的程序可以安全地重用。如果生成的片段很小，图中断可能会影响性能。为了最大化性能，尽可能减少图中断的数量非常重要。

## 识别图中断的原因

要识别程序中的所有图中断及其相关原因，可以使用 `torch._dynamo.explain`。此工具在提供的函数上运行 TorchDynamo 并汇总遇到的图中断。以下是一个使用示例：

```python
import torch
import torch._dynamo as dynamo
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    print("woo")
    if b.sum() < 0:
        b = b * -1
    return x * b
explanation = dynamo.explain(toy_example)(torch.randn(10), torch.randn(10))
print(explanation_verbose)
"""
Graph Count: 3
Graph Break Count: 2
Op Count: 5
Break Reasons:
  Break Reason 1:
    Reason: builtin: print [<class 'torch._dynamo.variables.constant.ConstantVariable'>] False
    User Stack:
      <FrameSummary file foo.py, line 5 in toy_example>
  Break Reason 2:
    Reason: generic_jump TensorVariable()
    User Stack:
      <FrameSummary file foo.py, line 6 in torch_dynamo_resume_in_toy_example_at_5>
Ops per Graph:
  ...
Out Guards:
  ...
"""
```

输出包括：

- `out_guards` - 一个列表的列表，其中每个子列表包含必须通过的守卫，以确保跟踪的图是有效的。
- `graphs` - 成功跟踪的图模块列表。
- `ops_per_graph` - 一个列表的列表，其中每个子列表包含在图中运行的操作。

要在遇到第一个图中断时抛出错误，请使用 `fullgraph` 模式。此模式禁用 TorchDynamo 的 Python 回退，仅当整个程序可以转换为单个图时才会成功。使用示例：

```python
def toy_example(a, b):
   ...

compiled_toy = torch.compile(toy_example, fullgraph=True, backend=<compiler>)(a, b)
```

### 过度重编译

当 TorchDynamo 编译一个函数（或其一部分）时，它会基于局部变量和全局变量做出某些假设，以允许编译器进行优化，并将这些假设表示为在运行时检查特定值的守卫。如果其中任何一个守卫失败，Dynamo 将重新编译该函数（或部分），重编译次数最多可达 `torch._dynamo.config.recompile_limit` 次。如果你的程序达到了缓存限制，你首先需要确定是哪个守卫失败，以及程序的哪一部分触发了它。

如果你的程序表现出有限程度的动态性，你可能能够调整 TorchDynamo 缓存限制，以便编译并缓存每个变体。但如果缓存限制设置得过高，你可能会发现重编译的成本超过了任何优化带来的好处。

```
torch._dynamo.config.recompile_limit = <your desired cache limit>
```

TorchDynamo 计划支持许多常见的动态张量形状情况，例如变化的批次大小或序列长度。它不计划支持秩动态性。在此期间，可以设置特定的缓存限制，并与分桶技术配合使用，从而为某些动态模型实现可接受的重编译次数。

## 精度调试

如果你设置环境变量 `TORCHDYNAMO_REPRO_LEVEL=4`，精度问题也可以被最小化重现。它采用类似 git bisect 的模式运行，一个完整的重现命令可能类似于 `TORCHDYNAMO_REPRO_AFTER="aot" TORCHDYNAMO_REPRO_LEVEL=4`。我们需要这样做的原因是，下游编译器（无论是 Triton 代码还是 C++ 后端）都会生成代码，这些下游编译器的数值结果可能在细微之处有所不同，却对你的训练稳定性产生巨大影响。因此，精度调试器对于我们检测代码生成或后端编译器中的错误非常有用。

如果你想确保在 torch 和 triton 中随机数生成是相同的，可以启用 `torch._inductor.config.fallback_random = True`

## 扩展调试

可以通过使用以下实验性标志来启用扩展调试。

`TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED` - 如果守卫的字符串表示与此标志值匹配，则提供扩展调试信息。例如，将其设置为 "Ne(s0, 10)"，以便在发出守卫时生成完整的 Python 和 C++ 回溯。
`TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL` - 在分配特定符号时提供扩展调试信息。例如，将其设置为 "u2"，以便在创建此符号时生成完整的 Python 和 C++ 回溯。
`TORCHDYNAMO_EXTENDED_DEBUG_CPP` - 为所有扩展调试设置以及错误提供扩展调试信息（C++ 回溯）。例如，将其设置为 "1"。C++ 回溯速度慢且信息冗长，因此在默认的扩展调试中不包含它。

## 冷启动计时和缓存损坏调试

为了测量冷启动编译时间或调试缓存损坏，可以传递 `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` 或设置 `torch.compiler.config.force_disable_caches = True`，这将覆盖任何其他缓存配置选项并禁用所有编译时缓存。