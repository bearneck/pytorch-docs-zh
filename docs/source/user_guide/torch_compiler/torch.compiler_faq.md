# 常见问题解答

**作者**: [Mark Saroufim](https://github.com/msaroufim)

## `torch.compile` 是否支持训练？

`torch.compile` 支持训练，它使用 AOTAutograd 来捕获反向传播过程：

1. `.forward()` 图和 `optimizer.step()` 由 TorchDynamo 的 Python `evalframe` 前端捕获。
2. 对于 TorchDynamo 捕获的每个 `.forward()` 片段，它使用 AOTAutograd 生成一个反向图片段。
3. 每对前向图和反向图（可选地）进行最小切割分区，以保存前向和反向之间的最小状态。
4. 前向和反向图对被包装在 `autograd.function` 模块中。
5. 用户代码调用 `.backward()` 仍然会触发 eager 模式的自动求导引擎，该引擎将每个*已编译的反向*图当作一个操作来运行，同时运行任何未编译的 eager 操作的 `.backward()` 函数。

## 是否支持分布式代码？

`torch.compile` 支持 `DistributedDataParallel` (DDP)。正在考虑支持其他分布式训练库。

分布式代码在 dynamo 中具有挑战性的主要原因是，AOTAutograd 会展开前向传播和反向传播，并为后端提供两个图进行优化。这对于分布式代码来说是个问题，因为我们理想情况下希望通信操作与计算操作重叠。Eager 模式的 PyTorch 通过不同的方式实现这一点，对于 DDP/FSDP，它使用自动求导钩子、模块钩子以及模块状态的修改/突变。在 dynamo 的简单应用中，由于 AOTAutograd 编译的函数与调度器钩子的交互方式，本应在反向传播期间操作之后直接运行的钩子可能会延迟到整个反向传播操作的编译区域之后。

使用 Dynamo 优化 DDP 的基本策略在 [distributed.py](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/backends/distributed.py) 中概述，其主要思想是在 [DDP 桶边界](https://pytorch.org/docs/stable/notes/ddp.html#internal-design) 处进行图中断。

当 DDP 中的每个节点需要与其他节点同步其权重时，它会将其梯度和参数组织到桶中，这减少了通信时间，并允许节点将其部分梯度广播给其他等待的节点。

分布式代码中的图中断意味着你可以期望 dynamo 及其后端优化分布式程序的计算开销，但不会优化其通信开销。如果减小的图大小剥夺了编译器的融合机会，图中断可能会干扰编译加速。然而，随着图大小的增加，收益会递减，因为当前大多数计算优化都是局部融合。因此，在实践中，这种方法可能已经足够。

## 我仍然需要导出整个图吗？

对于绝大多数模型，你可能不需要，你可以直接使用 `torch.compile()`，但在某些情况下，完整图是必要的，你可以通过简单地运行 `torch.compile(..., fullgraph=True)` 来确保完整图。这些情况包括：

- 大规模训练运行，例如需要流水线并行和其他高级分片策略的 $250K+ 参数模型。
- 推理优化器，如 [TensorRT](https://github.com/pytorch/TensorRT) 或 [AITemplate](https://github.com/facebookincubator/AITemplate)，它们比训练优化器更激进地进行融合。
- 移动端训练或推理。

未来的工作将包括将通信操作追踪到图中，将这些操作与计算优化协调，并优化通信操作。

## 为什么我的代码崩溃了？

如果你的代码在没有 `torch.compile` 时运行正常，但在启用后开始崩溃，那么最重要的第一步是找出故障发生在堆栈的哪个部分。要进行故障排除，请按照以下步骤操作，只有在当前步骤成功时才尝试下一步。

1. `torch.compile(..., backend="eager")` 仅运行 TorchDynamo 前向图捕获，然后使用 PyTorch 运行捕获的图。如果这失败了，那么 TorchDynamo 有问题。
2. `torch.compile(..., backend="aot_eager")` 运行 TorchDynamo 捕获前向图，然后运行 AOTAutograd 追踪反向图，无需任何额外的后端编译器步骤。然后 PyTorch eager 将用于运行前向和反向图。如果这失败了，那么 AOTAutograd 有问题。
3. `torch.compile(..., backend="inductor")` 运行 TorchDynamo 捕获前向图，然后运行 AOTAutograd 使用 TorchInductor 编译器追踪反向图。如果这失败了，那么 TorchInductor 有问题。

## 为什么编译很慢？

- **Dynamo 编译** – TorchDynamo 内置了一个统计函数，用于收集和显示每个编译阶段所花费的时间。在执行 `torch._dynamo` 后，可以通过调用 `torch._dynamo.utils.compile_times()` 来访问这些统计数据。默认情况下，它会返回一个字符串，表示按名称统计的每个 TorchDynamo 函数所花费的编译时间。
- **Inductor 编译** – TorchInductor 内置了统计和跟踪功能，用于显示每个编译阶段所花费的时间、输出代码、输出图可视化以及 IR 转储。使用 `env TORCH_COMPILE_DEBUG=1 python repro.py`。这是一个调试工具，旨在通过输出类似[此示例](https://gist.github.com/jansel/f4af078791ad681a0d4094adeb844396)的内容，使调试/理解 TorchInductor 内部机制更加容易。该调试跟踪中的每个文件都可以通过 `torch._inductor.config.trace.*` 来启用/禁用。配置文件和图表默认都是禁用的，因为生成它们的开销较大。更多示例请参阅[示例调试目录输出](https://gist.github.com/jansel/f4af078791ad681a0d4094adeb844396)。
- **过度重新编译**
  当 TorchDynamo 编译一个函数（或其一部分）时，它会基于局部变量和全局变量做出某些假设，以允许编译器进行优化，并将这些假设表示为在运行时检查特定值的守卫。如果其中任何一个守卫失败，Dynamo 将重新编译该函数（或部分），最多重新编译 `torch._dynamo.config.recompile_limit` 次。如果你的程序达到了缓存限制，首先需要确定是哪个守卫失败，以及程序的哪部分触发了它。使用 `TORCH_TRACE/tlparse` 或 `TORCH_LOGS=recompiles` 来追踪问题的根源，更多细节请查看 *torch.compiler_troubleshooting*。

## 为什么在生产环境中重新编译？

在某些情况下，你可能不希望程序预热后出现意外的编译。例如，如果你在一个对延迟敏感的应用中处理生产流量。为此，TorchDynamo 提供了一种替代模式，该模式会使用先前编译好的图，但不会生成新的图：

```python
frozen_toy_example = dynamo.run(toy_example)
frozen_toy_example(torch.randn(10), torch.randn(10))
```

## 如何加速我的代码？

加速 PyTorch 代码主要有三种方式：

1.  **内核融合**：通过垂直融合将连续的操作融合起来，以避免过多的读/写。例如，融合两个连续的余弦操作意味着你可以进行 1 次读取和 1 次写入，而不是 2 次读取和 2 次写入。水平融合：最简单的例子是批处理，即单个矩阵与一批样本相乘，但更一般的情况是分组 GEMM，其中一组矩阵乘法被一起调度。
2.  **乱序执行**：编译器的一种通用优化，通过查看图中精确的数据依赖关系，我们可以决定执行节点的最佳时机以及哪些缓冲区可以被重用。
3.  **自动工作放置**：类似于乱序执行，但通过将图的节点与物理硬件或内存等资源进行匹配，我们可以设计一个合适的调度方案。

以上是加速 PyTorch 代码的通用原则，但不同的后端会在优化内容上做出不同的权衡。例如，Inductor 首先会尽可能地融合所有内容，然后才生成 [Triton](https://openai.com/blog/triton/) 内核。

此外，Triton 还提供了加速，因为它具有自动内存合并、内存管理以及每个流式多处理器内的调度功能，并且被设计用于处理分块计算。

然而，无论你使用哪个后端，最好使用基准测试和观察的方法，尝试使用 PyTorch 分析器，直观地检查生成的内核，并尝试自己了解发生了什么。


## 为什么我没有看到加速效果？

### 图中断

使用 dynamo 时，你未能看到预期加速效果的主要原因是过多的图中断。那么，什么是图中断？

给定一个像这样的程序：

```python
def some_fun(x):
    ...

torch.compile(some_fun)(x)
...
```

TorchDynamo 会尝试将 `some_fun()` 中的所有 torch/张量操作编译成一个单一的 FX 图，但它可能无法将所有内容捕获到一个图中。

有些图中断的原因是 TorchDynamo 无法克服的，例如调用 PyTorch 之外的 C 扩展对 TorchDynamo 是不可见的，并且可能执行任意操作，而 TorchDynamo 无法引入必要的守卫来确保编译后的程序可以安全地重用。

> 为了最大化性能，尽可能减少图中断的数量非常重要。

### 识别图中断的原因

要识别程序中所有的图中断及其相关原因，可以使用 `torch._dynamo.explain`。这个工具在提供的函数上运行 TorchDynamo，并汇总遇到的图中断。以下是一个使用示例：

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
print(explanation)
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

要在遇到第一个图中断时抛出错误，可以通过使用 `fullgraph=True` 来禁用 Python 回退，如果你使用过基于导出的编译器，应该对此很熟悉。

```python
def toy_example(a, b):
   ...

torch.compile(toy_example, fullgraph=True, backend=<compiler>)(a, b)
```

### 为什么我的代码更改后没有重新编译？

如果你通过设置 `env TORCHDYNAMO_DYNAMIC_SHAPES=1 python model.py` 启用了动态形状，那么你的代码在形状变化时将不会重新编译。我们已添加了对动态形状的支持，当形状变化小于2倍时，可以避免重新编译。这在计算机视觉中图像尺寸变化或自然语言处理中序列长度变化等场景中特别有用。在推理场景中，通常无法预先知道批量大小，因为你需要处理来自不同客户端应用程序的输入。

一般来说，TorchDynamo 会尽量避免不必要的重新编译。例如，如果 TorchDynamo 发现了3个计算图，而你的更改只修改了其中一个图，那么只有那个图会重新编译。另一个避免潜在缓慢编译时间的技巧是预热模型：先编译一次，之后的编译会快得多。冷启动编译时间仍然是我们重点关注的指标。

## 为什么我得到的结果不正确？

如果你设置环境变量 `TORCHDYNAMO_REPRO_LEVEL=4`，精度问题也可以被最小化。它采用类似 git bisect 的模式运行，完整的复现命令可能类似于 `TORCHDYNAMO_REPRO_AFTER="aot" TORCHDYNAMO_REPRO_LEVEL=4`。我们需要这个功能的原因是下游编译器（无论是 Triton 代码还是 C++ 后端）会生成代码，这些下游编译器的数值计算可能在细微之处有所不同，但对你的训练稳定性产生巨大影响。因此，精度调试器对我们检测代码生成或后端编译器中的错误非常有用。

如果你想确保 torch 和 triton 之间的随机数生成相同，可以启用 `torch._inductor.config.fallback_random = True`

## 为什么我遇到内存不足（OOM）问题？

Dynamo 仍是一个 alpha 版本产品，因此存在一些导致 OOM 的原因。如果你遇到 OOM，请按以下顺序尝试禁用配置，然后在 GitHub 上提交问题，以便我们解决根本问题：
1. 如果你在使用动态形状，请尝试禁用它（默认已禁用）：`env TORCHDYNAMO_DYNAMIC_SHAPES=0 python model.py`
2. 在 inductor 中，默认启用了带有 Triton 的 CUDA 图，但移除它们可能缓解一些 OOM 问题：`torch._inductor.config.triton.cudagraphs = False`

## `torch.func` 是否与 `torch.compile` 兼容（用于 `grad` 和 `vmap` 变换）？

对使用 `torch.compile` 的函数应用 `torch.func` 变换是可行的：

```python
import torch

@torch.compile
def f(x):
    return torch.sin(x)

def g(x):
    return torch.grad(f)(x)

x = torch.randn(2, 3)
g(x)
```

### 在 `torch.compile` 处理的函数内部调用 `torch.func` 变换

### 使用 `torch.compile` 编译 `torch.func.grad`

```python
import torch

def wrapper_fn(x):
    return torch.func.grad(lambda x: x.sin().sum())(x)

x = torch.randn(3, 3, 3)
grad_x = torch.compile(wrapper_fn)(x)
```

### 使用 `torch.compile` 编译 `torch.vmap`

```python
import torch

def my_fn(x):
    return torch.vmap(lambda x: x.sum(1))(x)

x = torch.randn(3, 3, 3)
output = torch.compile(my_fn)(x)
```

### 编译除支持函数之外的其他函数（应急方案）

对于其他变换，作为一种变通方法，可以使用 `torch._dynamo.allow_in_graph`

`allow_in_graph` 是一个应急方案。如果你的代码无法与 `torch.compile`（它内省 Python 字节码）一起工作，但你相信它可以通过符号追踪方法（如 `jax.jit`）工作，那么请使用 `allow_in_graph`。

通过使用 `allow_in_graph` 来注解一个函数，你必须确保你的代码满足以下要求：

- 函数中的所有输出仅依赖于输入，而不依赖于任何捕获的张量。
- 你的函数是纯函数。也就是说，它不改变任何状态。这个要求可能会放宽；我们实际上支持从外部看是纯函数的函数：它们可能包含原地 PyTorch 操作，但不能改变全局状态或函数的输入。
- 你的函数不会引发数据相关的错误。

```python
import torch

@torch.compile
def f(x):
    return torch._dynamo.allow_in_graph(torch.vmap(torch.sum))(x)

x = torch.randn(2, 3)
f(x)
```

一个常见的陷阱是使用 `allow_in_graph` 来注解一个调用 `nn.Module` 的函数。这是因为输出现在依赖于 `nn.Module` 的参数。要使其工作，请使用 `torch.func.functional_call` 来提取模块状态。

## NumPy 是否与 `torch.compile` 兼容？

从 2.1 版本开始，`torch.compile` 能够理解处理 NumPy 数组的原生 NumPy 程序，以及通过 `x.numpy()`、`torch.from_numpy` 和相关函数在 PyTorch 和 NumPy 之间转换的混合 PyTorch-NumPy 程序。


### `torch.compile` 支持哪些 NumPy 功能？

`torch.compile` 中的 NumPy 遵循 NumPy 2.0 预发布版。

一般来说，`torch.compile` 能够追踪大多数 NumPy 结构，当它无法追踪时，它会回退到即时执行模式，让 NumPy 执行那段代码。即便如此，仍有一些功能在 `torch.compile` 中的语义与 NumPy 略有不同：

- NumPy 标量：我们将它们建模为 0 维数组。也就是说，`np.float32(3)` 在 `torch.compile` 下返回一个 0 维数组。为了避免图中断，最好使用这个 0 维数组。如果这破坏了你的代码，你可以通过将 NumPy 标量转换为相关的 Python 标量类型 `bool/int/float` 来绕过这个问题。
- 负步长：`np.flip` 和带负步长的切片会返回一个副本。
- 类型提升：NumPy 的类型提升规则将在 NumPy 2.0 中改变。新规则在 [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html) 中描述。`torch.compile` 实现了 NEP 50 而不是当前即将弃用的规则。
- `{tril,triu}_indices_from/{tril,triu}_indices` 返回数组而不是数组元组。

对于某些功能，我们暂不支持追踪，并会优雅地回退到 NumPy 执行：

- 非数值数据类型，如日期时间、字符串、字符、void、结构化数据类型和记录数组。
- 长数据类型 `np.float128/np.complex256` 以及某些无符号数据类型 `np.uint16/np.uint32/np.uint64`。
- `ndarray` 子类。
- 掩码数组。
- 深奥的 ufunc 机制，如 `axes=[(n,k),(k,m)->(n,m)]` 和 ufunc 方法（例如 `np.add.reduce`）。
- 对 `complex64/complex128` 数组进行排序/排序。
- NumPy 的 `np.poly1d` 和 `np.polynomial`。
- 具有 2 个或更多返回值的函数中的位置参数 `out1, out2`（`out=tuple` 可以正常工作）。
- `__array_function__`、`__array_interface__` 和 `__array_wrap__`。
- `ndarray.ctypes` 属性。

### 我可以使用 `torch.compile` 编译 NumPy 代码吗？

当然可以！`torch.compile` 原生理解 NumPy 代码，并将其视为 PyTorch 代码处理。为此，只需用 `torch.compile` 装饰器包装 NumPy 代码即可。

```python
import torch
import numpy as np

@torch.compile
def numpy_fn(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.sum(X[:, :, None] * Y[:, None, :], axis=(-2, -1))

X = np.random.randn(1024, 64)
Y = np.random.randn(1024, 64)
Z = numpy_fn(X, Y)
assert isinstance(Z, np.ndarray)
```

使用环境变量 `TORCH_LOGS=output_code` 执行此示例，我们可以看到 `torch.compile` 能够将乘法和求和融合到一个 C++ 内核中。它还能够使用 OpenMP 并行执行它们（原生 NumPy 是单线程的）。这可以轻松地将您的 NumPy 代码速度提升 `n` 倍，其中 `n` 是处理器中的核心数！

以这种方式追踪 NumPy 代码也支持在编译代码内部进行图中断。

### 我可以在 CUDA 上执行 NumPy 代码并通过 `torch.compile` 计算梯度吗？

是的，您可以！为此，您只需在 `torch.device("cuda")` 上下文中执行代码即可。考虑以下示例：

```python
import torch
import numpy as np

@torch.compile
def numpy_fn(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.sum(X[:, :, None] * Y[:, None, :], axis=(-2, -1))

X = np.random.randn(1024, 64)
Y = np.random.randn(1024, 64)
with torch.device("cuda"):
    Z = numpy_fn(X, Y)
assert isinstance(Z, np.ndarray)
```

在此示例中，`numpy_fn` 将在 CUDA 上执行。为了实现这一点，`torch.compile` 会自动将 `X` 和 `Y` 从 CPU 移动到 CUDA，然后将结果 `Z` 从 CUDA 移回 CPU。如果我们在同一程序运行中多次执行此函数，我们可能希望避免所有这些相当昂贵的内存复制。为此，我们只需调整 `numpy_fn`，使其接受 CUDA 张量并返回张量。我们可以使用 `torch.compiler.wrap_numpy` 来实现：

```python
@torch.compile(fullgraph=True)
@torch.compiler.wrap_numpy
def numpy_fn(X, Y):
    return np.sum(X[:, :, None] * Y[:, None, :], axis=(-2, -1))

X = torch.randn(1024, 64, device="cuda")
Y = torch.randn(1024, 64, device="cuda")
Z = numpy_fn(X, Y)
assert isinstance(Z, torch.Tensor)
assert Z.device.type == "cuda"
```

在这里，我们显式地在 CUDA 内存中创建张量，并将它们传递给函数，该函数在 CUDA 设备上执行所有计算。`wrap_numpy` 负责在 `torch.compile` 级别将任何 `torch.Tensor` 输入标记为具有 `np.ndarray` 语义的输入。在编译器内部标记张量是一个非常廉价的操作，因此在运行时不会发生数据复制或数据移动。

使用此装饰器，我们还可以对 NumPy 代码进行微分！

```python
@torch.compile(fullgraph=True)
@torch.compiler.wrap_numpy
def numpy_fn(X, Y):
    return np.mean(np.sum(X[:, :, None] * Y[:, None, :], axis=(-2, -1)))

X = torch.randn(1024, 64, device="cuda", requires_grad=True)
Y = torch.randn(1024, 64, device="cuda")
Z = numpy_fn(X, Y)
assert isinstance(Z, torch.Tensor)
Z.backward()
# X.grad 现在保存了计算的梯度
print(X.grad)
```

我们一直使用 `fullgraph=True`，因为在此上下文中图中断会带来问题。当发生图中断时，我们需要具体化 NumPy 数组。由于 NumPy 数组没有 `device` 或 `requires_grad` 的概念，这些信息在图中断期间会丢失。

我们无法通过图中断传播梯度，因为图中断代码可能执行任意不知道如何微分的代码。另一方面，在 CUDA 执行的情况下，我们可以像第一个示例中那样，通过使用 `torch.device("cuda")` 上下文管理器来解决这个问题：

```python
@torch.compile
@torch.compiler.wrap_numpy
def numpy_fn(X, Y):
    prod = X[:, :, None] * Y[:, None, :]
    print("oops, a graph break!")
    return np.sum(prod, axis=(-2, -1))

X = torch.randn(1024, 64, device="cuda")
Y = torch.randn(1024, 64, device="cuda")

with torch.device("cuda"):
    Z = numpy_fn(X, Y)
assert isinstance(Z, torch.Tensor)
assert Z.device.type == "cuda"
```

在图中断期间，中间张量仍然需要移动到 CPU，但当图中断后恢复追踪时，图的其余部分仍在 CUDA 上追踪。考虑到这种 CUDA <> CPU 和 CPU <> CUDA 的移动，在 NumPy 上下文中图中断的代价相当高，应尽量避免，但至少它们允许追踪复杂的代码段。

### 如何在 `torch.compile` 下调试 NumPy 代码？

调试 JIT 编译的代码具有挑战性，因为现代编译器的复杂性及其引发的令人畏惧的错误。torch.compile 故障排除文档  包含了一些关于如何完成此任务的技巧和窍门。

如果上述方法不足以定位问题的根源，我们仍然可以使用一些其他 NumPy 特定的工具。我们可以通过禁用对 NumPy 函数的追踪来判断错误是否完全在 PyTorch 代码中：

```python
from torch._dynamo import config
config.trace_numpy = False
```

如果问题出在追踪的 NumPy 代码中，我们可以通过导入 `import torch._numpy as np`，使用 PyTorch 作为后端来急切执行 NumPy 代码（无需 `torch.compile`）。
这仅应用于**调试目的**，绝不能替代 PyTorch API，因为它**性能远低于** PyTorch API，并且作为私有 API，**可能随时更改而不另行通知**。无论如何，`torch._numpy` 是一个基于 PyTorch 实现的 NumPy Python 版本，`torch.compile` 在内部使用它将 NumPy 代码转换为 PyTorch 代码。它的代码相当易于阅读和修改，因此如果您在其中发现任何错误，请随时提交修复该错误的 PR 或直接提出问题。

如果导入 `torch._numpy as np` 后程序能够正常工作，那么问题很可能出在 TorchDynamo 中。如果是这种情况，请随时提交一个包含 最小复现示例  的问题。

### 我对一些 NumPy 代码使用了 `torch.compile`，但没有看到任何加速效果。

最好的起点是阅读
[关于如何调试这类 torch.compile 问题的通用建议教程](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_faq.html#why-am-i-not-seeing-speedups)。

由于使用了不支持的功能，可能会发生一些图中断。请参阅
不支持的 NumPy 功能 。更一般地说，需要记住的是，一些广泛使用的 NumPy 功能与编译器配合不佳。例如，原地修改使得编译器内部推理变得困难，并且通常比其非原地对应操作性能更差。因此，最好避免使用它们。使用 `out=` 参数也是如此。相反，应优先使用非原地操作，并让 `torch.compile` 来优化内存使用。对于数据依赖的操作也是如此，例如通过布尔掩码进行掩码索引，或者数据依赖的控制流，如 `if` 或 `while` 结构。

## 细粒度追踪应使用哪个 API？

在某些情况下，您可能需要将代码的一小部分排除在 torch.compile 编译之外。本节提供了一些答案，您可以在 *torchdynamo_fine_grain_tracing* 中找到更多信息。

### 如何在函数上实现图中断？

在函数上实现图中断不足以充分表达您希望 PyTorch 执行的操作。您需要更具体地描述您的用例。您可能需要考虑的一些最常见用例包括：

- 如果您希望在此函数帧以及递归调用的帧上禁用编译，请使用 `torch._dynamo.disable`。
- 如果您希望特定运算符（例如 `fbgemm`）使用急切模式，请使用 `torch._dynamo.disallow_in_graph`。

一些不常见的用例包括：

- 如果您希望在函数帧上禁用 TorchDynamo，但在递归调用的帧上重新启用它——请使用 `torch._dynamo.disable(recursive=False)`。
- 如果您希望防止函数帧的内联——在您希望防止内联的函数开头使用 `torch._dynamo.graph_break`。

### `torch._dynamo.disable` 和 `torch._dynamo.disallow_in_graph` 有什么区别？

`disallow_in_graph` 在运算符级别（更具体地说，是您在 TorchDynamo 提取的图中看到的运算符）上工作。

`disable` 在函数帧级别工作，决定 TorchDynamo 是否应查看该函数帧。

### `torch._dynamo.disable` 和 `torch._dynamo_skip` 有什么区别？


> 📝 **注意**
> `torch._dynamo_skip` 已弃用。


您很可能需要的是 `torch._dynamo.disable`。但在不太可能的情况下，您可能需要更精细的控制。假设您只想在 `a_fn` 函数上禁用追踪，但希望在 `aa_fn` 和 `ab_fn` 中继续追踪。下图演示了此用例：


> **FIGURE**：../../_static/img/fine_grained_apis/call_stack_diagram.png
> :alt: torch.compile + disable(a_fn, recursive=False) 的示意图


在这种情况下，您可以使用 `torch._dynamo.disable(recursive=False)`。
在早期版本中，此功能由 `torch._dynamo.skip` 提供。现在，这由 `torch._dynamo.disable` 内部的 `recursive` 标志支持。
