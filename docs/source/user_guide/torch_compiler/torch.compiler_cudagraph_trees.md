# CUDAGraph 树

## **背景**

### CUDAGraph

关于 CUDAGraph 的更详细背景，请阅读[使用 CUDAGraphs 加速 PyTorch](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)。

[CUDA Graphs](https://developer.nvidia.com/blog/cuda-10-features-revealed/) 首次亮相于 CUDA 10，它允许将一系列 CUDA 内核定义并封装为一个单元，即操作图，而不是一系列单独启动的操作。它提供了一种通过单个 CPU 操作启动多个 GPU 操作的机制，从而减少了启动开销。

CUDA Graphs 可以带来显著的加速效果，特别是对于 CPU 开销高或计算量小的模型。它有一些限制，要求相同的内核必须以相同的参数、依赖关系和内存地址运行。

- 无法实现任意的控制流（然而，通过 `torch.cond()` 表达的控制流可以在 CUDA Graph 中捕获。参见 {ref}`数据依赖的控制流 <graph-data-dependent-control-flow>`。）
- 触发主机到设备同步的内核（例如 .item()）会报错
- 内核的所有输入参数都固定为录制时的值
- CUDA 内存地址是固定的，但这些地址处的内存值可以改变
- 没有必要的 CPU 操作或 CPU 副作用

### PyTorch CUDAGraph 集成

PyTorch 提供了一个围绕 CUDAGraphs 的[便捷包装器](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html)，它处理了与 PyTorch 缓存分配器的一些棘手交互。

CachingAllocator 为所有新分配使用一个独立的内存池。在 CUDAGraph 录制期间，内存的统计、分配和释放与即时执行时完全相同。在重放时，只调用内核，分配器没有任何变化。在初始录制之后，分配器不知道用户程序中哪些内存正在被使用。

在即时分配和 cudagraph 分配之间使用独立的内存池可能会增加程序的内存使用量，如果有大量内存同时分配给两者的话。

### 创建图化可调用对象

[创建图化可调用对象](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html) 是 PyTorch 的一个抽象，用于在一系列可调用对象之间共享单个内存池。图化可调用对象利用了这样一个事实：在 CUDA Graph 录制期间，内存由缓存分配器精确统计，从而可以在不同的 CUDA Graph 录制之间安全地共享内存。在每次调用中，输出被保留为活动内存，防止一个可调用对象覆盖另一个的活动内存。图化可调用对象只能以单一顺序调用；第一次运行的内存地址被固化到第二次运行中，依此类推。

### TorchDynamo 之前的 CUDA Graphs 集成

使用 `cudagraph_trees=False` 运行时，不会在不同的图捕获之间重用内存，这可能导致内存使用量大幅增加。即使对于没有图中断的模型，这也有问题。前向传播和反向传播是独立的图捕获，因此前向和反向的内存池不共享。特别是，前向传播中保存的激活值内存无法在反向传播中回收。

## **CUDAGraph 树集成**

与图化可调用对象类似，CUDA Graph 树在所有图捕获中使用单个内存池。然而，CUDA Graph 树不是要求单一的调用序列，而是创建独立的 CUDA Graph 捕获树。让我们看一个示例：

```python
@torch.compile(mode="reduce-overhead")
def foo(x):
    # 图 1
    y = x * x * x
    # 此处触发图中断
    if y.sum() > 0:
        # 图 2
        z = y ** y
    else:
        # 图 3
        z = (y.abs() ** y.abs())
    torch._dynamo.graph_break()
    # 图 4
    return z * torch.rand_like(z)

# 第一次运行预热每个图，执行诸如 CuBlas 或 Triton 基准测试等操作
foo(torch.arange(0, 10, device="cuda"))
# 第二次运行进行 CUDA Graph 录制并重放
foo(torch.arange(0, 10, device="cuda"))
# 最终我们进入优化的 CUDA Graph 重放路径
foo(torch.arange(0, 10, device="cuda"))
```

在这个例子中，我们通过函数有两条独立的路径：1 -> 2 -> 4，或 1 -> 3 -> 4。

我们通过构建一个 CUDA Graph 录制磁带（在这个例子中是 1 -> 2 -> 4），在不同的录制之间共享单个内存池中的所有内存。我们添加了一些不变量来确保内存始终位于录制时的相同位置，并且用户程序中不存在可能被覆盖的活动张量。

- 适用 CUDA Graphs 的相同约束：相同的内核必须以相同的参数（静态大小、地址等）调用
- 在录制和重放之间必须观察到相同的内存模式：如果一个图的张量输出在录制期间在另一个图之后消亡，那么在重放期间也必须如此。
- CUDA 池中的活动内存强制两个录制之间存在依赖关系
- 这些录制只能以单一顺序调用 1 -> 2 -> 4

所有内存都在单个内存池中共享，因此与即时执行相比没有额外的内存开销。现在，如果我们遇到一条新路径并运行图 3 会发生什么？

图 1 被重放，然后我们遇到尚未录制的图 3。在图重放期间，私有内存池不会更新，因此 y 不会反映在分配器中。如果不加注意，我们会覆盖它。为了支持在重放其他图后重用相同的内存池，我们将内存池检查点恢复到图 1 结束时的状态。现在我们的活动张量反映在缓存分配器中，我们就可以安全地运行新图了。

首先，我们会命中已在图1中记录好的优化路径 `CUDAGraph.replay()`。接着我们会命中图3。和之前一样，我们需要在记录前对图进行一次预热运行。在预热运行时，内存地址尚未固定，因此图4也会回退到 inductor 的非 cudagraph 调用。

第二次命中图3时，我们已经预热完毕并准备好记录。我们记录图3，然后再次记录图4，因为输入内存地址已经改变。这就创建了一个 CUDA 图记录树。一棵 CUDA 图树！

```
  1
 / \\
2   3
 \\   \\
  4   4
```

### 输入突变支持

输入突变函数指的是对输入张量进行原地写入操作的函数，如下所示：

```python
def foo(x, y):
    # 突变输入 x
    x.add_(1)
    return x + y
```

输入突变函数通常会给 CUDAGraph 树带来挑战。由于 CUDAGraph 对静态 CUDA 内存地址的要求，对于每个输入张量 x，CUDAGraph 树可能会分配一个静态内存地址 x'。在执行过程中，CUDAGraph 树首先将输入张量 x 复制到静态内存地址 x'，然后重放记录的 CUDAGraph。对于输入突变函数，x' 会被原地更新，但这不会反映在输入张量 x 上，因为 x 和 x' 位于不同的 CUDA 内存地址。

仔细研究输入突变函数，可以发现有三种类型的输入：

- **来自 eager 的输入**：我们假设这些张量在每次执行时其地址都会变化。因为 cudagraphs 会固定内存地址，我们需要在记录和执行图之前将这些输入复制到一个静态地址的张量上。
- **参数和缓冲区**：我们假设（并在运行时检查）这些张量在每次执行时都具有相同的张量地址。我们不需要复制它们的内容，因为记录的内存地址将与执行时的内存地址相同。
- **来自 CUDAGraph 树先前输出的张量**：因为 cudagraph 的输出张量地址是固定的，如果我们运行 CUDAGraph1，然后运行 CUDAGraph2，那么从 CUDAGraph1 输入到 CUDAGraph2 的输入将具有固定的内存地址。这些输入，与参数和缓冲区一样，不需要复制到静态地址张量。我们会检查以确保这些输入在运行时是稳定的，如果不稳定，我们将重新记录。

CUDAGraph 树支持对参数和缓冲区以及来自 CUDAGraph 树先前输出的张量进行输入突变。对于来自 eager 的输入的突变，CUDAGraph 树将在没有 CUDAGraph 的情况下运行该函数，并发出 *skipping due to mutated inputs* 日志。以下示例展示了 CUDAGraph 树对来自 CUDAGraph 树先前输出的张量的支持。

```python
import torch

@torch.compile(mode="reduce-overhead")
def foo(x):
    return x + 1

@torch.compile(mode="reduce-overhead")
def mut(x):
    return x.add_(2)

# 启用输入突变支持
torch._inductor.config.triton.cudagraph_support_input_mutation = True

for i in range(3):
    torch.compiler.cudagraph_mark_step_begin()
    inp = torch.rand([4], device="cuda")

    # 应用 CUDAGraph，因为 `foo` 不突变 `inp`
    tmp = foo(inp)
    # 虽然 `mut` 突变了 `tmp`，但 `tmp` 是一个由 CUDAGraph 管理的函数的输出。
    # 因此仍然应用 CUDAGraph。
    mut(tmp)


torch.compiler.cudagraph_mark_step_begin()
inp = torch.rand([4], device="cuda")

tmp = foo(inp)
# 虽然 `tmp` 是一个由 CUDAGraph 树管理的函数的输出，但 `tmp.clone()`
# 不是。因此 CUDAGraph 不会应用于 `mut`，并且会有一条日志
# `skipping cudagraphs due to mutated inputs`
mut(tmp.clone())
```

要为突变来自 eager 的输入的函数启用 CUDAGraph 树，请重写该函数以避免输入突变。

> **注意**\
> 对于 "reduce-overhead" 模式，通过设置 [torch.\_inductor.config.cudagraph_support_input_mutation = True](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L662) 来启用输入突变支持。

### 动态形状支持

[动态形状](https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html) 意味着输入张量在函数调用之间具有不同的形状。由于 CUDAGraph 要求固定的张量地址，CUDAGraph 树会为输入张量的每个唯一形状重新记录 CUDAGraph。这导致单个 inductor 图对应多个 CUDAGraph。当形状数量有限时（例如推理中的批大小），重新记录 CUDAGraph 是有益的。然而，如果输入张量形状频繁变化，甚至每次调用都变化，重新记录 CUDAGraph 可能就不划算了。在 CUDA 12.4 及 Driver Version 550+ 之前，Nvidia 在 CUDAGraph 中每个内核启动使用 64 KB 的设备内存。在大量重新记录 CUDAGraph 的情况下，这个内存开销可能非常显著。

对于输入张量形状频繁变化的函数，我们建议将输入张量填充到几个固定的张量形状，以便仍然享受 CUDAGraph 带来的好处。此外，设置 [torch.\_inductor.config.triton.cudagraph_skip_dynamic_graphs=True](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L653) 可以跳过对具有动态形状输入的函数进行 cudagraph 处理，只对具有静态输入张量形状的函数进行 cudagraph 处理。

### NCCL 支持

CUDAGraph 树支持包含 nccl 运算符的函数。虽然 CUDAGraph 树为 CUDAGraph 执行的是逐设备记录，但 NCCL 支持允许跨设备通信。

```python
@torch.compile(mode="reduce-overhead")
def func(x):
    y = x * x
    y = torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM)
    x = torch.nn.functional.silu(x)
    return x * y
```

### 跳过 CUDAGraph 的原因

由于 CUDAGraph 有一些要求，例如静态输入张量地址以及不支持 CPU 运算符，CUDAGraph 树会检查函数是否满足这些要求，并在必要时跳过 CUDAGraph。这里，我们列出了跳过 CUDAGraph 的常见原因。

- **输入原地修改**：CUDAGraph Trees 会跳过原地修改 eager 输入的函数。
  原地修改参数和缓冲区，或修改 CUDAGraph Tree 管理函数的输出张量仍然受支持。更多细节请参阅*输入原地修改支持*章节。
- **CPU 算子**：包含 CPU 算子的函数会被跳过。请将函数拆分为多个函数，并仅对包含 GPU 算子的函数应用 CUDAGraph Trees。
- **多设备算子**：如果函数包含在多个设备上的算子，则该函数会被跳过。目前，CUDAGraph 是按设备应用的。请使用 NCCL 等受支持的库进行跨设备通信。更多细节请参阅*NCCL 支持*章节。
- **自由无约束符号**：自由无约束符号通常出现在[动态形状](https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)期间。
  CUDAGraph Trees 目前会为每个唯一的输入张量形状记录一个 CUDAGraph。更多细节请参阅*动态形状支持*。
- **CUDAGraph 不安全的自定义算子**：某些自定义算子可能包含 cudagraph 不安全的算子，这会导致 cudagraph 被跳过。更多细节请参阅*CUDAGraph 不安全的自定义算子*。
- **不兼容的算子**：如果函数包含不兼容的算子，CUDAGraph Trees 会跳过该函数。请用受支持的算子替换函数中的这些算子。我们列出了不兼容算子的详尽列表：

```python
aten._fused_moving_avg_obs_fq_helper.default
aten._fused_moving_avg_obs_fq_helper_functional.default
aten.multinomial.default
fbgemm.dense_to_jagged.default
fbgemm.jagged_to_padded_dense.default
run_and_save_rng_state
run_with_rng_state
aten._local_scalar_dense
aten._assert_scalar
```

当 [torch.are_deterministic_algorithms_enabled()](https://pytorch.org/docs/stable/generated/torch.are_deterministic_algorithms_enabled.html) 启用时，以下算子不兼容。

```python
aten._fused_moving_avg_obs_fq_helper.default
aten._fused_moving_avg_obs_fq_helper_functional.default
aten.multinomial.default
fbgemm.dense_to_jagged.default
fbgemm.jagged_to_padded_dense.default
run_and_save_rng_state
run_with_rng_state
aten._local_scalar_dense
aten._assert_scalar
```

### CUDAGraph 不安全的自定义算子

默认情况下，自定义算子被假定为对 CUDAGraph 是安全的。然而，某些自定义算子可能包含不受支持的算子，例如 CPU 算子。由于编译器将自定义算子视为黑盒，用户必须通过设置 `torch._C.Tag.cudagraph_unsafe` 标签来显式地将这些算子标记为对 CUDAGraph 不安全，如下例所示。当函数包含 cudagraph 不安全的自定义算子时，除非启用了*CUDAGraph 分区*，否则该函数将被 CUDAGraph 跳过。

```python
@torch.library.custom_op(
    "mylib::modify",
    mutates_args=(),
    tags=(torch._C.Tag.cudagraph_unsafe,),
)
def modify(pic: torch.Tensor) -> torch.Tensor:
    pic1 = pic + 1
    pic1_cpu = (pic1.cpu() + 1) * 2
    return pic1_cpu.cuda() + pic

@modify.register_fake
def _(pic):
    return torch.empty_like(pic)
```

### CUDAGraph 分区

正如我们之前讨论的，CUDAGraph 不支持某些算子（例如 CPU 算子），这可能会限制其采用。CUDAGraph 分区是一种编译器解决方案，可以自动分离这些算子，重新排序算子以减少分区数量，并分别对每个分区应用 CUDAGraph。请设置 `torch._inductor.config.graph_partition=True` 来启用 CUDAGraph 分区。

考虑以下示例，其中 `x` 和 `y` 是 GPU 输入，但 `y_cpu` 是 CPU 张量。在没有图分区的情况下，由于 CPU 算子，此函数必须被跳过。启用图分区后，CPU 算子被分离出来，剩余的 GPU 算子被 cudagraph 化，从而生成两个独立的 CUDAGraph。

```python
def f(x, y):
    x1 = x + 1
    y1 = y + 1
    y_cpu = y1.cpu() + 1
    z = x @ y
    return x1 + y1 + z + y_cpu.cuda()
```

目前，CUDAGraph 分区支持分离以下类型的算子：

- **非 GPU 算子**：常见的例子包括 CPU 张量上的计算。
- **设备复制算子**：设备间的数据传输，例如上例中的 `y1.cpu()`。
- **控制流算子**：[控制流算子](https://docs.pytorch.org/docs/stable/cond.html) 被分离出来，因为它们尚未被 CUDAGraph 支持。
- **CUDAGraph 不安全的自定义算子**：标记有 `torch._C.Tag.cudagraph_unsafe` 的自定义算子被分离出来。详情请参阅*CUDAGraph 不安全的自定义算子*章节。
- **无约束符号整数**：更多信息请参阅*动态形状支持*章节。

### 限制

由于 CUDA Graph 会固定内存地址，CUDA Graphs 没有很好的方法来处理来自前一次调用的存活张量。

假设我们正在用以下代码对推理运行进行基准测试：

```python
import torch

@torch.compile(mode="reduce-overhead")
def my_model(x):
    y = torch.matmul(x, x)
    return y

x = torch.randn(10, 10, device="cuda")
y1 = my_model(x)
y2 = my_model(x)
print(y1)
# RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run.
```

在独立的 CUDA Graph 实现中，第一次调用的输出将被第二次调用覆盖。在 CUDAGraph Trees 中，我们不希望在迭代之间添加意外的依赖关系，这会导致我们无法达到热路径，也不希望过早释放前一次调用的内存。我们的启发式规则是：在推理中，对于 torch.compile，我们在每次调用时开始一个新的迭代；在训练中，只要没有尚未调用的待处理反向传播，我们也这样做。如果这些启发式规则不正确，您可以使用 [torch.compiler.cudagraph_mark_step_begin()](https://pytorch.org/docs/stable/generated/torch.compiler.cudagraph_mark_step_begin.html) 标记新迭代的开始，或者在开始下一次运行之前克隆前一次迭代的张量（在 torch.compile 外部）。

### 比较

| 注意事项      | 独立 CudaGraph                                         | CUDAGraph 树                                                        |
|---------------|------------------------------------------------------------|------------------------------------------------------------------------|
| 内存可能增加 | 每次图编译时（新尺寸等）              | 如果同时运行非 cudagraph 内存                           |
| 录制操作    | 图的任何新调用                           | 程序执行过程中遇到任何新的、唯一路径时都会重新录制   |
| 潜在问题      | 调用一个图会覆盖之前的调用    | 无法在模型的不同运行之间保持内存持久性 - 仅限单个训练循环或单次推理运行 |