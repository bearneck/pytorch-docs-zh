# CUDA 语义

`torch.cuda` 用于设置和运行 CUDA 操作。它会跟踪当前选定的 GPU，并且默认情况下，您分配的所有 CUDA 张量都将在该设备上创建。选定的设备可以通过 `torch.cuda.device` 上下文管理器来更改。

但是，一旦张量被分配，您就可以在其上执行操作，而不受选定设备的影响，并且结果将始终放置在与该张量相同的设备上。

默认情况下，不允许跨 GPU 操作，但 `torch.Tensor.copy_` 和其他具有复制功能的方法（如 `torch.Tensor.to` 和 `torch.Tensor.cuda`）除外。除非您启用了点对点内存访问，否则任何尝试在不同设备上的张量上启动操作的尝试都会引发错误。

下面是一个展示此功能的小示例：:

> cuda = torch.device(\'cuda\') \# 默认 CUDA 设备 cuda0 = torch.device(\'cuda:0\') cuda2 = torch.device(\'cuda:2\') \# GPU 2（这些是 0 索引的）
>
> x = torch.tensor(\[1., 2.\], device=cuda0) \# x.device 是 device(type=\'cuda\', index=0) y = torch.tensor(\[1., 2.\]).cuda() \# y.device 是 device(type=\'cuda\', index=0)
>
> with torch.cuda.device(1):
>
> :   \# 在 GPU 1 上分配一个张量 a = torch.tensor(\[1., 2.\], device=cuda)
>
>     \# 将张量从 CPU 传输到 GPU 1 b = torch.tensor(\[1., 2.\]).cuda() \# a.device 和 b.device 是 device(type=\'cuda\', index=1)
>
>     \# 您也可以使用 `Tensor.to` 来传输张量： b2 = torch.tensor(\[1., 2.\]).to(device=cuda) \# b.device 和 b2.device 是 device(type=\'cuda\', index=1)
>
>     c = a + b \# c.device 是 device(type=\'cuda\', index=1)
>
>     z = x + y \# z.device 是 device(type=\'cuda\', index=0)
>
>     \# 即使在上下文中，您也可以指定设备 \# （或者给 .cuda 调用一个 GPU 索引） d = torch.randn(2, device=cuda2) e = torch.randn(2).to(cuda2) f = torch.randn(2).cuda(cuda2) \# d.device, e.device, 和 f.device 都是 device(type=\'cuda\', index=2)

## Ampere（及更高版本）设备上的 TensorFloat-32 (TF32)

在 PyTorch 2.9 之后，我们提供了一套新的 API 来以更细粒度的方式控制 TF32 行为，并建议使用新的 API 以获得更好的控制。 我们可以为每个后端和每个操作符设置 float32 精度。我们还可以为特定操作符覆盖全局设置。

``` python
torch.backends.fp32_precision = "ieee"
torch.backends.cuda.matmul.fp32_precision = "ieee"
torch.backends.cudnn.fp32_precision = "ieee"
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cudnn.rnn.fp32_precision = "tf32"
```

对于 [cuda/cudnn]，fp32_precision 可以设置为 [ieee] 或 [tf32]。 [ieee] fp32_precision 表示我们将使用 [FP32] 作为内部计算精度。 [tf32] fp32_precision 表示我们将允许使用 [TF32] 作为内部计算精度。

如果 fp32_precision 设置为 [ieee]，我们可以为特定操作符覆盖通用设置。

``` python
torch.backends.cudnn.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "ieee"
torch.backends.cudnn.rnn.fp32_precision = "ieee"
```

如果 fp32_precision 设置为 [ieee]，我们也可以为特定后端覆盖通用设置。

``` python
torch.backends.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "ieee"
torch.backends.cudnn.conv.fp32_precision = "ieee"
torch.backends.cudnn.rnn.fp32_precision = "ieee"
```

对于上述两种情况，\`torch.backends.cudnn.conv.fp32_precision\` 和 [torch.backends.cudnn.rnn.fp32_precision] 都被覆盖为 [ieee]。

我们建议使用新设置以获得更好的控制。并且我们不支持混合使用新旧设置。


> ⚠️ **警告**
> 以下使用 [allow_tf32] 的旧设置将被弃用。我们建议使用上述新设置以获得更好的控制。并且我们不支持混合使用新旧设置。
>
> 从 PyTorch 1.7 开始，有一个名为 [allow_tf32] 的新标志。该标志在 PyTorch 1.7 到 PyTorch 1.11 中默认为 True，在 PyTorch 1.12 及更高版本中默认为 False。 此标志控制 PyTorch 是否允许在内部使用 TensorFloat32 (TF32) 张量核心（自 Ampere 以来在 NVIDIA GPU 上可用）来计算 matmul（矩阵乘法和批量矩阵乘法）和卷积。
>
> TF32 张量核心旨在通过将输入数据舍入为 10 位尾数，并以 FP32 精度累积结果，同时保持 FP32 动态范围，从而在 [torch.float32] 张量上实现更好的 matmul 和卷积性能。
>
> matmul 和卷积是分开控制的，它们对应的标志可以在以下位置访问：
>
> ``` python
> # 下面的标志控制是否在 matmul 上允许 TF32。此标志在 PyTorch 1.12 及更高版本中默认为 False。
> torch.backends.cuda.matmul.allow_tf32 = True
>
> # 下面的标志控制是否在 cuDNN 上允许 TF32。此标志默认为 True。
> torch.backends.cudnn.allow_tf32 = True
> ```
>
> matmul 的精度也可以通过 `torch.set_float32_matmul_precision` 更广泛地设置（不仅限于 CUDA）。 请注意，除了 matmul 和卷积本身之外，内部使用 matmul 或卷积的函数和 nn 模块也会受到影响。这些包括 [nn.Linear]、\`nn.Conv\*\`、cdist、tensordot、affine grid 和 grid sample、adaptive log softmax、GRU 和 LSTM。
>
> 要了解精度和速度，请参见下面的示例代码和基准测试数据（在 A100 上）：
>
> ``` python
> a_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
> b_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
> ab_full = a_full @ b_full
> mean = ab_full.abs().mean()  # 80.7277
>
> a = a_full.float()
> b = b_full.float()
>
> # 在 TF32 模式下进行矩阵乘法。
> torch.backends.cuda.matmul.allow_tf32 = True
> ab_tf32 = a @ b  # 在 GA100 上耗时 0.016 秒
> error = (ab_tf32 - ab_full).abs().max()  # 0.1747
> relative_error = error / mean  # 0.0022
>
> # 在禁用 TF32 的情况下进行矩阵乘法。
> torch.backends.cuda.matmul.allow_tf32 = False
> ab_fp32 = a @ b  # 在 GA100 上耗时 0.11 秒
> error = (ab_fp32 - ab_full).abs().max()  # 0.0031
> relative_error = error / mean  # 0.000039
> ```
>
> 从上面的例子可以看出，启用 TF32 后，在 A100 上的速度提高了约 7 倍，但与双精度相比的相对误差大约大了 2 个数量级。请注意，TF32 与单精度速度的确切比率取决于硬件代次，因为诸如内存带宽与计算能力的比率以及 TF32 与 FP32 矩阵乘法吞吐量的比率等特性可能因代次或型号而异。 如果需要完整的 FP32 精度，用户可以通过以下方式禁用 TF32：
>
> ``` python
> torch.backends.cuda.matmul.allow_tf32 = False
> torch.backends.cudnn.allow_tf32 = False
> ```
>
> 要在 C++ 中关闭 TF32 标志，可以执行：
>
> ``` C++
> at::globalContext().setAllowTF32CuBLAS(false);
> at::globalContext().setAllowTF32CuDNN(false);
> ```
>
> 有关 TF32 的更多信息，请参阅：
>
> - [TensorFloat-32](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/)
> - [CUDA 11](https://devblogs.nvidia.com/cuda-11-features-revealed/)
> - [Ampere architecture](https://devblogs.nvidia.com/nvidia-ampere-architecture-in-depth/)
>
> ## FP16 GEMM 中的降低精度归约
>
> （不同于为那些 FP16 累加比 FP32 累加具有更高吞吐量的硬件设计的完整 FP16 累加，参见 `完整 FP16 累加<fp16accumulation>`）
>
> fp16 GEMM 可能会使用一些中间降低精度归约（例如，使用 fp16 而非 fp32）来完成。这些选择性的精度降低可以在某些工作负载（特别是那些具有较大 [k] 维度的负载）和 GPU 架构上实现更高的性能，但代价是数值精度和潜在的溢出风险。
>
> V100 上的一些基准测试数据示例：
>
> ``` 
> [--------------------------- bench_gemm_transformer --------------------------]
>       [  m ,  k  ,  n  ]    |  allow_fp16_reduc=True  |  allow_fp16_reduc=False
> 1 threads: --------------------------------------------------------------------
>       [4096, 4048, 4096]    |           1634.6        |           1639.8
>       [4096, 4056, 4096]    |           1670.8        |           1661.9
>       [4096, 4080, 4096]    |           1664.2        |           1658.3
>       [4096, 4096, 4096]    |           1639.4        |           1651.0
>       [4096, 4104, 4096]    |           1677.4        |           1674.9
>       [4096, 4128, 4096]    |           1655.7        |           1646.0
>       [4096, 4144, 4096]    |           1796.8        |           2519.6
>       [4096, 5096, 4096]    |           2094.6        |           3190.0
>       [4096, 5104, 4096]    |           2144.0        |           2663.5
>       [4096, 5112, 4096]    |           2149.1        |           2766.9
>       [4096, 5120, 4096]    |           2142.8        |           2631.0
>       [4096, 9728, 4096]    |           3875.1        |           5779.8
>       [4096, 16384, 4096]   |           6182.9        |           9656.5
> （时间单位为微秒）。
> ```
>
> 如果需要完整的精度归约，用户可以通过以下方式禁用 fp16 GEMM 中的降低精度归约：
>
> ``` python
> torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
> ```
>
> 要在 C++ 中切换降低精度归约标志，可以执行：
>
> ``` C++
> at::globalContext().setAllowFP16ReductionCuBLAS(false);
> ```
>
> ## BF16 GEMM 中的降低精度归约
>
> 对于 BFloat16 GEMM，存在一个类似的标志（如上所述）。 请注意，对于 BF16，此开关默认设置为 [True]，如果您在工作负载中观察到数值不稳定性，您可能希望将其设置为 [False]。
>
> 如果不希望使用降低精度归约，用户可以通过以下方式禁用 bf16 GEMM 中的降低精度归约：
>
> ``` python
> torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
> ```
>
> 要在 C++ 中切换降低精度归约标志，可以执行：
>
> ``` C++
> at::globalContext().setAllowBF16ReductionCuBLAS(true);
> ```
>
> ## FP16 GEMM 中的完整 FP16 累加
>
> 某些 GPU 在 [完全]\_ 使用 FP16 进行 FP16 GEMM 累加时性能会提高，但代价是数值精度和更大的溢出可能性。请注意，此设置仅对计算能力为 7.0（Volta）或更高版本的 GPU 有效。
>
> 可以通过以下方式启用此行为：
>
> ``` python
> torch.backends.cuda.matmul.allow_fp16_accumulation = True
> ```
>
> 要在 C++ 中切换降低精度归约标志，可以执行：
>
> ``` C++
> at::globalContext().setAllowFP16AccumulationCuBLAS(true);
> ```
>
> ## 异步执行
>
> 默认情况下，GPU 操作是异步的。当您调用使用 GPU 的函数时，操作会被 *入队* 到特定设备，但不一定立即执行。这允许我们并行执行更多计算，包括在 CPU 或其他 GPU 上的操作。
>
> 通常，异步计算对调用者是不可见的，因为（1）每个设备按照操作入队的顺序执行它们，并且（2）PyTorch 在 CPU 和 GPU 之间或两个 GPU 之间复制数据时会自动执行必要的同步。因此，计算过程就像每个操作都是同步执行的一样。
>
> 您可以通过设置环境变量 `CUDA_LAUNCH_BLOCKING=1` 来强制进行同步计算。这在 GPU 上发生错误时非常有用。（在异步执行的情况下，此类错误直到操作实际执行后才会报告，因此堆栈跟踪不会显示错误发生的位置。）
>
> 异步计算的一个后果是，没有同步的时间测量是不准确的。为了获得精确的测量结果，应该在测量前调用 `torch.cuda.synchronize()`，或者使用 `torch.cuda.Event` 来记录时间，如下所示：:
>
> > start_event = torch.cuda.Event(enable_timing=True) end_event = torch.cuda.Event(enable_timing=True) start_event.record()
> >
> > \# 在此处运行一些操作
> >
> > end_event.record() torch.cuda.synchronize() \# 等待事件被记录！ elapsed_time_ms = start_event.elapsed_time(end_event)
>
> 作为例外，一些函数如 `torch.Tensor.to` 和 `torch.Tensor.copy_` 接受一个显式的 `non_blocking` 参数，允许调用者在不需要时绕过同步。另一个例外是 CUDA 流，如下所述。
>
> ### CUDA 流
>
> 一个 [CUDA 流](#cuda-流) 是属于特定设备的线性执行序列。通常您不需要显式创建它：默认情况下，每个设备使用自己的"默认"流。
>
> 每个流内部的操作按照它们创建的顺序串行执行，但来自不同流的操作可以以任何相对顺序并发执行，除非使用了显式的同步函数（如 `torch.cuda.synchronize` 或 `torch.cuda.Stream.wait_stream`）。例如，以下代码是不正确的：:
>
> > cuda = torch.device(\'cuda\') s = torch.cuda.Stream() \# 创建一个新流。 A = torch.empty((100, 100), device=cuda).normal\_(0.0, 1.0) with torch.cuda.stream(s): \# sum() 可能在 [normal]()() 完成之前就开始执行！ B = torch.sum(A)
>
> 当"当前流"是默认流时，PyTorch 会在数据移动时自动执行必要的同步，如上所述。然而，当使用非默认流时，用户有责任确保适当的同步。此示例的修正版本如下：:
>
> > cuda = torch.device(\'cuda\') s = torch.cuda.Stream() \# 创建一个新流。 A = torch.empty((100, 100), device=cuda).normal\_(0.0, 1.0) s.wait_stream(torch.cuda.default_stream(cuda)) \# 新增！ with torch.cuda.stream(s): B = torch.sum(A) A.record_stream(s) \# 新增！
>
> 这里有两个新增部分。`torch.cuda.Stream.wait_stream` 调用确保在我们开始在侧流上运行 `sum(A)` 之前，`normal_()` 的执行已经完成。`torch.Tensor.record_stream`（详见其文档）确保在 `sum(A)` 完成之前不会释放 A 的内存。您也可以在稍后的时间点手动等待该流，使用 `torch.cuda.default_stream(cuda).wait_stream(s)`（注意立即等待是没有意义的，因为这会阻止该流与默认流上的其他工作并行执行。）关于何时使用哪一个的更多细节，请参阅 `torch.Tensor.record_stream` 的文档。
>
> 请注意，即使没有读取依赖关系，这种同步也是必要的，例如，如本例所示：:
>
> > cuda = torch.device(\'cuda\') s = torch.cuda.Stream() \# 创建一个新流。 A = torch.empty((100, 100), device=cuda) s.wait_stream(torch.cuda.default_stream(cuda)) \# 仍然需要！ with torch.cuda.stream(s): [A.normal]()(0.0, 1.0) A.record_stream(s)
>
> 尽管在流 `s` 上的计算不读取 `A` 的内容，并且 `A` 没有其他用途，但仍然需要同步，因为 `A` 可能对应于 CUDA 缓存分配器重新分配的内存，而旧（已释放）内存上可能还有待处理的操作。
>
> ### 反向传播的流语义
>
> 每个反向 CUDA 操作在与对应前向操作相同的流上运行。如果前向传播在不同的流上并行运行独立操作，这有助于反向传播利用相同的并行性。
>
> 反向调用相对于周围操作的流语义与任何其他调用相同。反向传播会插入内部同步以确保这一点，即使反向操作在多个流上运行，如前一段所述。更具体地说，当调用 `autograd.backward<torch.autograd.backward>`、`autograd.grad<torch.autograd.grad>` 或 `tensor.backward<torch.Tensor.backward>`，并可选地提供 CUDA 张量作为初始梯度（例如，`autograd.backward(..., grad_tensors=initial_grads)<torch.autograd.backward>`、`autograd.grad(..., grad_outputs=initial_grads)<torch.autograd.grad>` 或 `tensor.backward(..., gradient=initial_grad)<torch.Tensor.backward>`）时，以下操作：
>
> 1.  可选地填充初始梯度，
> 2.  调用反向传播，以及
> 3.  使用梯度
>
> 具有与任何一组操作相同的流语义关系：:
>
> > s = torch.cuda.Stream()
> >
> > \# 安全，梯度在与 backward() 相同的流上下文中使用 with torch.cuda.stream(s): loss.backward() use grads
> >
> > \# 不安全 with torch.cuda.stream(s): loss.backward() use grads
> >
> > \# 安全，带有同步 with torch.cuda.stream(s): loss.backward() torch.cuda.current_stream().wait_stream(s) use grads
> >
> > \# 安全，填充初始梯度和调用反向传播在相同的流上下文中 with torch.cuda.stream(s): loss.backward(gradient=torch.ones_like(loss))
>
> \# 不安全，填充 initial_grad 和调用 backward 在不同的流上下文中， \# 没有同步 initial_grad = torch.ones_like(loss) with torch.cuda.stream(s): loss.backward(gradient=initial_grad)
>
> \# 安全，带有同步 initial_grad = torch.ones_like(loss) s.wait_stream(torch.cuda.current_stream()) with torch.cuda.stream(s): initial_grad.record_stream(s) loss.backward(gradient=initial_grad)
>
> #### 向后兼容性说明：在默认流上使用梯度
>
> 在 PyTorch 的早期版本（1.9 及更早）中，自动求导引擎总是将默认流与所有反向操作同步，因此以下模式：:
>
> > 
> >
> > with torch.cuda.stream(s):
> >
> > :   loss.backward()
> >
> > use grads
>
> 只要 `use grads` 发生在默认流上就是安全的。 在当前版本的 PyTorch 中，该模式不再安全。如果 `backward()` 和 `use grads` 在不同的流上下文中，你必须同步这些流：:
>
> > 
> >
> > with torch.cuda.stream(s):
> >
> > :   loss.backward()
> >
> > torch.cuda.current_stream().wait_stream(s) use grads
>
> 即使 `use grads` 是在默认流上。
>
> ## 内存管理
>
> PyTorch 使用缓存内存分配器来加速内存分配。这允许快速释放内存而无需设备同步。然而，分配器管理的未使用内存仍会在 `nvidia-smi` 中显示为已使用。你可以使用 `torch.cuda.memory_allocated` 和 `torch.cuda.max_memory_allocated` 来监控张量占用的内存，并使用 `torch.cuda.memory_reserved` 和 `torch.cuda.max_memory_reserved` 来监控缓存分配器管理的总内存量。调用 `torch.cuda.empty_cache` 会释放 PyTorch 中所有\*\*未使用的\*\*缓存内存，以便其他 GPU 应用程序可以使用这些内存。但是，张量占用的 GPU 内存不会被释放，因此这不会增加 PyTorch 可用的 GPU 内存量。
>
> 为了更好地理解 CUDA 内存随时间的使用情况，`torch_cuda_memory` 描述了捕获和可视化内存使用跟踪的工具。
>
> 对于更高级的用户，我们通过 `torch.cuda.memory_stats` 提供了更全面的内存基准测试。我们还提供了通过 `torch.cuda.memory_snapshot` 捕获内存分配器状态完整快照的能力，这可以帮助你理解代码产生的底层分配模式。
>
> ### 使用 `PYTORCH_ALLOC_CONF` 优化内存使用
>
> 使用缓存分配器可能会干扰内存检查工具，例如 `cuda-memcheck`。要使用 `cuda-memcheck` 调试内存错误，请在环境中设置 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 以禁用缓存。
>
> 缓存分配器的行为可以通过环境变量 `PYTORCH_ALLOC_CONF` 来控制。`PYTORCH_CUDA_ALLOC_CONF` 是其别名，仅用于向后兼容。 格式为 `PYTORCH_ALLOC_CONF=<option>:<value>,<option2>:<value2>...` 可用选项：
>
> - `backend` 允许选择底层分配器的实现。 目前有效的选项是 `native`（使用 PyTorch 的原生实现）和 `cudaMallocAsync`（使用 [CUDA 内置的异步分配器]()）。 `cudaMallocAsync` 需要 CUDA 11.4 或更高版本。默认值为 `native`。 `backend` 适用于进程使用的所有设备，不能按设备单独指定。
> - `large_segment_size_mb` 原生分配器使用小块和大块来管理已分配的内存。此设置用于配置大块的大小。默认值为 20 MB。
> - `max_split_size_mb` 阻止原生分配器拆分大于此大小（以 MB 为单位）的块。这可以减少碎片，并可能使一些临界工作负载在内存耗尽前完成。性能成本范围从"零"到"显著"，具体取决于分配模式。默认值是无限的，即所有块都可以拆分。 `torch.cuda.memory_stats` 和 `torch.cuda.memory_summary` 方法对于调优很有用。此选项应作为最后手段，用于处理因"内存不足"而中止并显示大量非活动拆分块的工作负载。 `max_split_size_mb` 仅在 `backend:native` 时有效。对于 `backend:cudaMallocAsync`，`max_split_size_mb` 将被忽略。
> - `roundup_power2_divisions` 有助于将请求的分配大小舍入到最近的 2 的幂次方除法，从而更好地利用块。在原生 CUDACachingAllocator 中，大小会向上舍入到 512 的块大小的倍数，这对于较小的大小效果很好。然而，对于附近的大分配，这可能效率低下，因为每个分配会进入不同大小的块，并且这些块的重用被最小化。这可能会产生大量未使用的块，并浪费 GPU 内存容量。此选项允许将分配大小舍入到最近的 2 的幂次方除法。例如，如果我们需要将大小 1200 向上舍入，且除法数为 4，则大小 1200 介于 1024 和 2048 之间，如果我们在它们之间进行 4 次除法，则值为 1024、1280、1536 和 1792。因此，分配大小 1200 将被舍入到 1280，作为最近的 2 的幂次方除法的上限。 指定单个值以应用于所有分配大小，或指定键值对数组以为每个 2 的幂次方区间单独设置 2 的幂次方除法。例如，要为所有小于 256MB 的分配设置 1 次除法，为 256MB 到 512MB 之间的分配设置 2 次除法，为 512MB 到 1GB 之间的分配设置 4 次除法，并为任何更大的分配设置 8 次除法，请将旋钮值设置为：\[256:1,512:2,1024:4,\>:8\]。 `roundup_power2_divisions` 仅在 `backend:native` 时有效。对于 `backend:cudaMallocAsync`，`roundup_power2_divisions` 将被忽略。
> - `max_non_split_rounding_mb` 将允许非拆分块以更好地重用，例如，一个 1024MB 的缓存块可以重用于 512MB 的分配请求。在默认情况下，我们只允许非拆分块最多舍入 20MB，因此一个 512MB 的块只能由 512-532 MB 大小的块提供。如果我们将此选项的值设置为 1024，它将允许 512-1536 MB 大小的块用于 512MB 的块，从而增加大块的重用。这也有助于减少延迟，避免昂贵的 cudaMalloc 调用。
> - `garbage_collection_threshold` 有助于主动回收未使用的 GPU 内存，以避免触发昂贵的同步并回收所有操作（release_cached_blocks），这对于延迟敏感的 GPU 应用程序（例如服务器）可能不利。设置此阈值（例如 0.8）后，如果 GPU 内存容量使用率超过阈值（即 GPU 应用程序分配的总内存的 80%），分配器将开始回收 GPU 内存块。该算法倾向于首先释放旧的和未使用的块，以避免释放正在被重用的块。阈值应大于 0.0 且小于 1.0。默认值设置为 1.0。
>
> `garbage_collection_threshold` 仅在 `backend:native` 时有效。
>
> :   使用 `backend:cudaMallocAsync` 时，`garbage_collection_threshold` 会被忽略。
>
> \* `expandable_segments` (实验性功能，默认: [False]) 如果设置为 [True]，此设置会指示
>
> :   分配器创建可以稍后扩展的 CUDA 分配，以更好地处理频繁改变分配大小的任务， 例如批量大小变化的情况。 通常对于大（\>2MB）分配，分配器会调用 cudaMalloc 来获取与用户请求大小相同的分配。 将来，如果这些分配是空闲的，其中的部分可以被其他请求重用。当程序进行许多 完全相同大小或大小为其整数倍的请求时，这种方法效果很好。许多深度学习模型遵循这种行为。 然而，一个常见的例外是批量大小在迭代之间略有变化，例如在批量推理中。 当程序最初以批量大小 [N] 运行时，它将进行适合该大小的分配。 如果将来它以大小 [N - 1] 运行，现有的分配仍然足够大。 但是，如果它以大小 [N + 1] 运行，那么它将不得不进行稍大的新分配。 并非所有张量的大小都相同。有些可能是 [(N + 1)\*A]，而其他可能是 [(N + 1)\*A\*B]， 其中 [A] 和 [B] 是模型中的一些非批量维度。 因为分配器在现有分配足够大时会重用它们，所以一些 [(N + 1)\*A] 分配实际上可以放入 已经存在的 [N\*B\*A] 段中，尽管不是完美匹配。随着模型运行，它会部分填满所有这些段， 在这些段的末尾留下无法使用的空闲内存片。分配器在某个时刻将需要 [cudaMalloc] 一个新的 [(N + 1)\*A\*B] 段。如果没有足够的内存，现在就无法回收 现有段末尾的空闲内存片。对于 50+ 层的模型，这种模式可能会重复 50+ 次， 从而产生许多碎片。
>
>     [expandable_segments] 允许分配器最初创建一个段，然后在需要更多内存时扩展其大小。 它不是为每个分配创建一个段，而是尝试（每个流）创建一个可以根据需要增长的段。 现在，当运行 [N + 1] 的情况时，分配将整齐地平铺到一个大段中，直到它被填满。 然后请求更多内存并附加到段的末尾。这个过程不会产生那么多无法使用的内存碎片， 因此更有可能成功找到所需内存。
>
> \* [pinned_use_cuda_host_register] 选项是一个布尔标志，用于确定是否
>
> :   使用 CUDA API 的 cudaHostRegister 函数来分配固定内存，而不是默认的 cudaHostAlloc。 当设置为 True 时，内存使用常规 malloc 分配，然后在调用 cudaHostRegister 之前 将页面映射到内存。这种页面的预映射有助于减少 cudaHostRegister 执行期间的锁定时间。
>
> \* [pinned_num_register_threads] 选项仅在 pinned_use_cuda_host_register
>
> :   设置为 True 时有效。默认情况下，使用一个线程来映射页面。此选项允许 使用更多线程来并行化页面映射操作，以减少固定内存的总体分配时间。 根据基准测试结果，此选项的一个良好值是 8。
>
> \* [pinned_use_background_threads] 选项是一个布尔标志，用于启用后台线程
>
> :   来处理事件。这避免了在快速分配路径中查询/处理事件相关的任何慢速路径。 此功能默认禁用。
>
> \* [pinned_reserve_segment_size_mb] 选项是以 MB 为单位的大小，用于为固定内存段保留。
>
> :   这预先分配一大段固定内存，然后用于分配小尺寸请求。这有助于减少昂贵的设备库调用次数。
>
> \* `graph_capture_record_stream_reuse` (实验性功能，默认: [False])
>
> :   如果设置为 [True]，CUDA 缓存分配器将尝试在 CUDA 图捕获期间通过使用图拓扑（而不是 CUDA 事件） 来确定何时可以安全重用已释放的块，从而回收设备内存。这可以减少在长时间捕获期间 跨多个流释放和重新分配缓冲区的峰值内存使用，特别是当捕获 DAG 频繁到达连接前沿时。
>
> \* `per_process_memory_fraction` 选项限制在所有 CUDA 设备上可以分配的内存量
>
> :   为可用内存的指定比例。这是一个介于 0 和 1 之间的值。尝试分配更多内存将引发内存不足错误。


> 📝 **注意**
> `CUDA 内存管理 API<cuda-memory-management-api>` 报告的一些统计信息 是特定于 `backend:native` 的，对于 `backend:cudaMallocAsync` 没有意义。 详情请参阅每个函数的文档字符串。
>
> ## 为 CUDA 使用自定义内存分配器
>
> 可以将分配器定义为 C/C++ 中的简单函数并将其编译为共享库， 下面的代码展示了一个基本分配器，它只是跟踪所有内存操作。
>
> ``` C++
> #include <sys/types.h>
> #include <cuda_runtime_api.h>
> #include <iostream>
> // 使用 g++ alloc.cc -o alloc.so -I/usr/local/cuda/include -shared -fPIC 编译
> extern "C" {
> void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
>    void *ptr;
>    cudaMalloc(&ptr, size);
>    std::cout<<"alloc "<<ptr<<size<<std::endl;
>    return ptr;
> }
>
> void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
>    std::cout<<"free "<<ptr<< " "<<stream<<std::endl;
>    cudaFree(ptr);
> }
> }
> ```
>
> 这可以通过 `torch.cuda.memory.CUDAPluggableAllocator` 在 Python 中使用。 用户需负责提供 [.so] 文件的路径以及符合上述签名的分配/释放函数名称。
>
> ``` python
> import torch
>
> # 加载分配器
> new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
>     'alloc.so', 'my_malloc', 'my_free')
> # 交换当前分配器
> torch.cuda.memory.change_current_allocator(new_alloc)
> # 这将使用新分配器在设备上分配内存
> b = torch.zeros(10, device='cuda')
>  python
> import torch
>
> # 执行初始内存分配
> b = torch.zeros(10, device='cuda')
> # 加载分配器
> new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
>     'alloc.so', 'my_malloc', 'my_free')
> # 由于当前分配器已实例化，此处将报错
> torch.cuda.memory.change_current_allocator(new_alloc)
> ```
>
> ## 在同一程序中混合使用不同的 CUDA 系统分配器
>
> 根据您的使用场景，`torch.cuda.change_current_allocator` 可能并非您所需，因为它会为整个程序交换 CUDA 分配器（类似于 `PYTORCH_ALLOC_CONF=backend:cudaMallocAsync`）。例如，如果交换的分配器没有缓存机制，您将失去 PyTorch 的 CUDACachingAllocator 的所有优势。相反，您可以使用 `torch.cuda.MemPool` 有选择地标记 PyTorch 代码的某个区域以使用自定义分配器。这将允许您在同一 PyTorch 程序中使用多个 CUDA 系统分配器，同时保留 CUDACachingAllocator 的大部分优势（例如缓存）。通过使用 `torch.cuda.MemPool`，您可以利用自定义分配器实现多种功能，例如：
>
> - 使用 `ncclMemAlloc` 分配器为 all-reduce 操作分配输出缓冲区可以启用 NVLink 交换归约（NVLS）。这可以减少 GPU 资源（SM 和复制引擎）上重叠的计算与通信内核之间的争用，特别是在张量并行工作负载中。
> - 对于基于 Grace CPU 的系统，使用 `cuMemCreate` 为 all-gather 操作分配主机输出缓冲区并指定 `CU_MEM_LOCATION_TYPE_HOST_NUMA` 可以启用基于扩展 GPU 内存（EGM）的从源 GPU 到目标 CPU 的内存传输。这加速了 all-gather 操作，因为传输通过 NVLink 进行，否则将通过带宽受限的网络接口卡（NIC）链路进行。这种加速的 all-gather 反过来可以加速模型检查点保存。
> - 如果您正在构建模型，并且最初不想考虑内存密集型模块（例如嵌入表）的最佳内存放置，或者您有一个对性能不敏感且不适合 GPU 的模块，那么您可以使用 `cudaMallocManaged` 为该模块分配内存，并首选 CPU 位置，从而先让模型运行起来。


> 📝 **注意**
> 虽然 `cudaMallocManaged` 通过 CUDA 统一虚拟内存（UVM）提供了便捷的自动内存管理，但不建议用于深度学习工作负载。对于适合 GPU 内存的深度学习工作负载，显式放置始终优于 UVM，因为没有页面错误且访问模式保持可预测。当 GPU 内存饱和时，UVM 必须执行昂贵的双重传输，在引入新页面之前将页面驱逐到 CPU。
>
> 以下代码展示了包装在 `torch.cuda.memory.CUDAPluggableAllocator` 中的 `ncclMemAlloc`。
>
> ``` python
> import os
>
> import torch
> import torch.distributed as dist
> from torch.cuda.memory import CUDAPluggableAllocator
> from torch.distributed.distributed_c10d import _get_default_group
> from torch.utils import cpp_extension
>
>
> # 创建分配器
> nccl_allocator_source = """
> #include <nccl.h>
> #include <iostream>
> extern "C" {
>
> void* nccl_alloc_plug(size_t size, int device, void* stream) {
>   std::cout << "Using ncclMemAlloc" << std::endl;
>   void* ptr;
>   ncclResult_t err = ncclMemAlloc(&ptr, size);
>   return ptr;
>
> }
>
> void nccl_free_plug(void* ptr, size_t size, int device, void* stream) {
>   std::cout << "Using ncclMemFree" << std::endl;
>   ncclResult_t err = ncclMemFree(ptr);
> }
>
> }
> """
> nccl_allocator_libname = "nccl_allocator"
> nccl_allocator = torch.utils.cpp_extension.load_inline(
>     name=nccl_allocator_libname,
>     cpp_sources=nccl_allocator_source,
>     with_cuda=True,
>     extra_ldflags=["-lnccl"],
>     verbose=True,
>     is_python_module=False,
>     build_directory="./",
> )
>
> allocator = CUDAPluggableAllocator(
>     f"./{nccl_allocator_libname}.so", "nccl_alloc_plug", "nccl_free_plug"
> ).allocator()
>
> # 设置分布式环境
> rank = int(os.getenv("RANK"))
> local_rank = int(os.getenv("LOCAL_RANK"))
> world_size = int(os.getenv("WORLD_SIZE"))
> torch.cuda.set_device(local_rank)
> dist.init_process_group(backend="nccl")
> device = torch.device(f"cuda:{local_rank}")
> default_pg = _get_default_group()
> backend = default_pg._get_backend(device)
>
> # 注意：为方便起见，ProcessGroupNCCL 后端提供了
> # ncclMemAlloc 分配器作为 backend.mem_allocator
> allocator = backend.mem_allocator
> ```
>
> 现在，您可以通过将此分配器传递给 `torch.cuda.MemPool` 来定义一个新的内存池：
>
> ``` python
> pool = torch.cuda.MemPool(allocator)
> ```
>
> 然后，该池可以与 `torch.cuda.use_mem_pool` 上下文管理器一起使用，以将张量分配到该池中：
>
> ``` python
> with torch.cuda.use_mem_pool(pool):
>     # 张量使用传入池中的 ncclMemAlloc 进行分配
>     tensor = torch.arange(1024 * 1024 * 2, device=device)
>     print(f"tensor ptr on rank {rank} is {hex(tensor.data_ptr())}")
>
> # 使用 ncclCommRegister 注册用户缓冲区（在底层调用）
> backend.register_mem_pool(pool)
> ```
>
> \# 集体操作使用零拷贝 NVLS dist.all_reduce(tensor\[0:4\]) torch.cuda.synchronize() print(tensor\[0:4\])
>
> 注意上述示例中 `register_mem_pool` 的用法。这是 NVLS 归约操作的一个额外步骤，用户缓冲区需要向 NCCL 注册。用户可以通过类似的 `deregister_mem_pool` 调用来注销缓冲区。
>
> 要回收内存，用户首先需要确保没有操作正在使用该内存池。当没有任何张量持有对该内存池的引用时，在内存池被删除时，`torch.cuda.empty_cache` 会在内部被调用，从而将所有内存返还给系统。
>
> ``` python
> del tensor, del pool
> ```
>
> 用户在创建 MemPool 时可以选择性地指定一个 `use_on_oom` 布尔值（默认为 False）。如果为 True，那么 CUDACachingAllocator 将能够在内存不足时，作为最后的手段使用此内存池中的内存，而不是直接报内存不足错误。
>
> ``` python
> pool = torch.cuda.MemPool(allocator, use_on_oom=True)
> with torch.cuda.use_mem_pool(pool):
>     a = torch.randn(40 * 1024 * 1024, dtype=torch.uint8, device="cuda")
> del a
>
> # 在内存限制下，为了避免内存不足，这将通过使用内存池的内存而成功执行
> b = torch.randn(40 * 1024 * 1024, dtype=torch.uint8, device="cuda")
> ```
>
> 以下 `torch.cuda.MemPool.use_count` 和 `torch.cuda.MemPool.snapshot` API 可用于调试目的：
>
> ``` python
> pool = torch.cuda.MemPool(allocator)
>
> # 此时内存池的使用计数应为 1，因为 MemPool 对象持有一个引用
> assert pool.use_count() == 1
>
> nelem_1mb = 1024 * 1024 // 4
>
> with torch.cuda.use_mem_pool(pool):
>     out_0 = torch.randn(nelem_1mb, device="cuda")
>
>     # 此时内存池的使用计数应为 2，因为 use_mem_pool 持有一个引用
>     assert pool.use_count() == 2
>
> # 此时内存池的使用计数应回到 1，因为 use_mem_pool 释放了其引用
> assert pool.use_count() == 1
>
> with torch.cuda.use_mem_pool(pool):
>     # 内存池应有 1 个段，因为我们上面进行了一个小分配（1 MB），所以 CUDACachingAllocator 将其打包进了一个 2 MB 的缓冲区
>     assert len(pool.snapshot()) == 1
>
>     out_1 = torch.randn(nelem_1mb, device="cuda")
>
>     # 内存池应仍然只有 1 个段，因为我们进行了另一个小分配（1 MB），它被打包进了现有的 2 MB 缓冲区
>     assert len(pool.snapshot()) == 1
>
>     out_2 = torch.randn(nelem_1mb, device="cuda")
>
>     # 现在内存池应有 2 个段，因为 CUDACachingAllocator 必须创建一个新的 2 MB 缓冲区来容纳 out_2
>     assert len(pool.snapshot()) == 2
> ```


> 📝 **注意**
> - `torch.cuda.MemPool` 持有对内存池的一个引用。当你使用 `torch.cuda.use_mem_pool` 上下文管理器时，它也会获取对内存池的另一个引用。在退出上下文管理器时，它会释放其引用。此后，理想情况下应该只有张量持有对内存池的引用。一旦张量释放了它们的引用，内存池的使用计数将为 1，这反映出只有 `torch.cuda.MemPool` 对象持有一个引用。只有在那时，当使用 `del` 调用内存池的析构函数时，内存池持有的内存才能被返还给系统。
> - `torch.cuda.MemPool` 目前不支持 CUDACachingAllocator 的 `expandable_segments` 模式。
> - [NCCL 有特定要求]() 来使缓冲区与 NVLS 归约操作兼容。这些要求可能在动态工作负载中被破坏，例如，CUDACachingAllocator 发送给 NCCL 的缓冲区可能被拆分，从而导致未正确对齐。在这些情况下，NCCL 可以使用回退算法而不是 NVLS。
> - 像 `ncclMemAlloc` 这样的分配器，由于对齐要求（`CU_MULTICAST_GRANULARITY_RECOMMENDED`, `CU_MULTICAST_GRANULARITY_MINIMUM`），可能会使用比请求更多的内存，并可能导致你的工作负载内存不足。
>
> ## 在 H100/H200 GPU 上使用自定义内存分配器调整 NVLink 性能
>
> 在少数情况下，H100/H200 GPU 上 NVLink 的性能可能受到数据物理内存布局的影响，这为开发人员提供了一个机会来调整他们的应用程序以获得最佳吞吐量。
>
> 数据物理内存布局影响性能的一个例子是，当通信内核发出不平衡的 NVLink 读/写操作时。在下图中，我们可以看到每个线程束在每个单次访问波中都以一致的跨步模式访问内存地址。我们可以通过调整工作负载中的跨步大小来获得更平衡的负载，或者可以实现一个自定义的 CUDA 分配器。
>
> ``` 
> _______________________________  _______________________________      _______________________________
> | Warp 0 Reading | No-reading |  | Warp 1 Reading | No-reading |  ...  Warp N Reading | No-reading |
> _______________________________  _______________________________      _______________________________
> <----------------------------->
>         跨步大小
> ```
>
> 这样的分配器可以为内核维护连续的虚拟内存地址，同时策略性地安排到物理内存地址的映射（例如，通过混洗）。这种技术允许开发人员探索不同的物理访问模式，以找到最高效的一种，从而在不修改内核逻辑的情况下解锁更高的性能。如前所述，使用 PyTorch 的自定义分配器支持可以实现这种分配器的实际实现，其中 malloc 和 free 函数是：
>
> ``` C++
> // 假设系统有 8 个 GPU
> struct CustomAllocInfo {
>   void** devPtr;  // 这将是可用的虚拟内存地址
>   CUdeviceptr dptr;
>   size_t totalSize;  // 已分配内存的总大小
>   size_t padded_size;
>   int device_id;
>   std::vector<CUmemGenericAllocationHandle> handles;  // 物理内存分配句柄
> };
>
> // 循环遍历页面
> cudaError_t customCudaMalloc(CustomAllocInfo* info) {
>     if (!info) return cudaErrorInvalidValue;
>
>     CUdeviceptr dptr;
>
>     // 冗余物理内存分配的句柄，有助于截断物理内存中的跨步模式
>     std::vector<CUmemGenericAllocationHandle> handles_redundant;
>
>     size_t granularity = 0;
>     CUmemAllocationProp prop = {};
>
>     int currentDev = info->device_id;
>     size_t totalSize = info->totalSize;
>
>     prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
>     prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
>     prop.location.id = currentDev;
>     cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
>     size_t padded_size = ROUND_UP(totalSize, granularity);
>
>     info->padded_size = padded_size;
>
>     // 循环遍历页面
>     size_t iter_granularity = granularity * 64; // 64 * granularity 配合 shift_size = 2 有效
>     uint32_t iteration_count = (totalSize + iter_granularity - 1) / iter_granularity;
>
>     cuMemAddressReserve(&dptr, padded_size, 0ULL, 0ULL, 0ULL);
>
>     const int shift_size = 2;
>     for (size_t i = 0; i < iteration_count; i+=shift_size) {
>
>         CUmemGenericAllocationHandle allocHandle[shift_size];
>         for (int shift = 0; (shift < shift_size)&&(i+shift < iteration_count); shift++){
>             CHECK_CUDA(cuMemCreate(&allocHandle[shift], iter_granularity, &prop, 0));
>             info->handles.push_back(allocHandle[shift]);
>         }
>
>         for (int shift = 0; (shift < shift_size)&&(i+shift < iteration_count); shift++){
>
>             // 映射实现移位 (shift -> (shift+1)%shift_size  )
>             CHECK_CUDA(cuMemMap(dptr + (i+shift) * iter_granularity, iter_granularity, 0, allocHandle[(shift+1)%shift_size], 0));
>
>             setupMultiGPUAccess(dptr + (i+shift) * iter_granularity, iter_granularity, {0, 1, 2, 3, 4, 5, 6, 7}); // 为所有 8 个 GPU 启用访问
>         }
>
>         // std::cout << "这里我们分配一个冗余页面 (2MB)..." << std::endl;
>         // 这是在交换之上的额外优化。它有助于进一步"打破"物理访问模式。
>         // 如果仅通过交换工作负载已经达到 SOL 性能，则可以省略此步骤。
>         CUmemGenericAllocationHandle allocHandle_redundant;
>         CHECK_CUDA(cuMemCreate(&allocHandle_redundant, granularity, &prop, 0));
>         handles_redundant.push_back(allocHandle_redundant);
>     }
>
>     *info->devPtr = (void*)dptr;
>     info->dptr = dptr;
>
>     // 释放每个冗余分配
>     for (auto handle : handles_redundant) {
>         // std::cout << "这里我们释放一个冗余页面 (2MB)..." << std::endl;
>         CHECK_CUDA(cuMemRelease(handle));
>     }
>
>     return cudaSuccess;
> }
>
> void customCudaFree(CustomAllocInfo* info) {
>     if (!info) return;
>
>     // CHECK_CUDA(cudaSetDevice(info->device_id));
>
>     CHECK_CUDA(cuMemUnmap(info->dptr, info->padded_size));
>
>     // 取消映射并释放每个分配
>     for (auto handle : info->handles) {
>         CHECK_CUDA(cuMemRelease(handle));
>     }
>
>     // 释放虚拟地址空间
>     // CHECK_CUDA(cuMemAddressFree((CUdeviceptr)*info->devPtr, info->padded_size));
>     CHECK_CUDA(cuMemAddressFree(info->dptr, info->padded_size));
> }
> ```
>
> ## cuBLAS 工作空间
>
> 对于每个 cuBLAS 句柄和 CUDA 流的组合，如果该句柄和流组合执行需要工作空间的 cuBLAS 内核，则会分配一个 cuBLAS 工作空间。 为了避免重复分配工作空间，这些工作空间不会被释放，除非调用 `torch._C._cuda_clearCublasWorkspaces()`。每次分配的工作空间大小可以通过环境变量 `CUBLAS_WORKSPACE_CONFIG` 指定，格式为 `:[SIZE]:[COUNT]`。 例如，默认的每次分配工作空间大小为 `CUBLAS_WORKSPACE_CONFIG=:4096:2:16:8`，这指定了总大小为 `2 * 4096 + 8 * 16 KiB`。要强制 cuBLAS 避免使用工作空间，请设置 `CUBLAS_WORKSPACE_CONFIG=:0:0`。
>
> ## cuFFT 计划缓存
>
> 对于每个 CUDA 设备，使用一个 cuFFT 计划的 LRU 缓存来加速在具有相同配置的相同几何形状的 CUDA 张量上重复运行 FFT 方法（例如 `torch.fft.fft`）。因为一些 cuFFT 计划可能会分配 GPU 内存，所以这些缓存有最大容量。
>
> 您可以使用以下 API 控制和查询当前设备缓存的属性：
>
> - `torch.backends.cuda.cufft_plan_cache.max_size` 给出缓存的容量（在 CUDA 10 及更新版本上默认为 4096，在较旧的 CUDA 版本上默认为 1023）。直接设置此值会修改容量。
> - `torch.backends.cuda.cufft_plan_cache.size` 给出当前驻留在缓存中的计划数量。
> - `torch.backends.cuda.cufft_plan_cache.clear()` 清除缓存。
>
> 要控制和查询非默认设备的计划缓存，您可以使用 `torch.device` 对象或设备索引来索引 `torch.backends.cuda.cufft_plan_cache` 对象，并访问上述属性之一。例如，要设置设备 `1` 的缓存容量，可以写 `torch.backends.cuda.cufft_plan_cache[1].max_size = 10`。
>
> ## 即时编译
>
> PyTorch 对在 CUDA 张量上执行的某些操作（如 torch.special.zeta）进行即时编译。这种编译可能耗时较长（根据硬件和软件配置，最长可达数秒），并且单个运算符可能发生多次编译，因为许多 PyTorch 运算符实际上会从多种内核中选择，每种内核都必须编译一次，具体取决于输入。每个进程会发生一次编译，如果使用内核缓存则仅编译一次。
>
> 默认情况下，如果定义了 XDG_CACHE_HOME，PyTorch 会在 \$XDG_CACHE_HOME/torch/kernels 中创建内核缓存；如果未定义，则在 \$HOME/.cache/torch/kernels 中创建（Windows 除外，目前尚不支持内核缓存）。可以通过两个环境变量直接控制缓存行为。如果 USE_PYTORCH_KERNEL_CACHE 设置为 0，则不会使用缓存；如果设置了 PYTORCH_KERNEL_CACHE_PATH，则该路径将用作内核缓存，而不是默认位置。
>
> ## 最佳实践
>
> ### 设备无关代码
>
> 由于 PyTorch 的结构，您可能需要显式编写设备无关（CPU 或 GPU）代码；例如，创建一个新张量作为循环神经网络的初始隐藏状态。
>
> 第一步是确定是否应使用 GPU。一种常见模式是使用 Python 的 `argparse` 模块读取用户参数，并结合 `torch.cuda.is_available` 设置一个可用于禁用 CUDA 的标志。在下文中，`args.device` 会生成一个 `torch.device` 对象，可用于将张量移动到 CPU 或 CUDA。
>
>     import argparse
>     import torch
>
>     parser = argparse.ArgumentParser(description='PyTorch Example')
>     parser.add_argument('--disable-cuda', action='store_true',
>                         help='Disable CUDA')
>     args = parser.parse_args()
>     args.device = None
>     if not args.disable_cuda and torch.cuda.is_available():
>         args.device = torch.device('cuda')
>     else:
>         args.device = torch.device('cpu')


> 📝 **注意**
> 在评估给定环境中 CUDA 的可用性 (`torch.cuda.is_available`) 时，PyTorch 的默认行为是调用 CUDA Runtime API 方法 [cudaGetDeviceCount](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f)。因为如果尚未初始化，此调用会反过来初始化 CUDA Driver API（通过 [cuInit](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE_1g0a2f1517e1bd8502c7194c3a8c134bc3)），所以运行过 `torch.cuda.is_available` 的进程的后续分支将因 CUDA 初始化错误而失败。
>
> 您可以在导入执行 `torch.cuda.is_available` 的 PyTorch 模块之前（或直接执行它之前），在环境中设置 `PYTORCH_NVML_BASED_CUDA_CHECK=1`，以引导 `torch.cuda.is_available` 尝试基于 NVML 的评估 ([nvmlDeviceGetCount_v2](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1ga93623b195bff04bbe3490c33c8a42d))。如果基于 NVML 的评估成功（即 NVML 发现/初始化未失败），则 `torch.cuda.is_available` 调用不会污染后续分支。
>
> 如果 NVML 发现/初始化失败，`torch.cuda.is_available` 将回退到标准的 CUDA Runtime API 评估，并且上述分支限制将适用。
>
> 请注意，上述基于 NVML 的 CUDA 可用性评估提供的保证比默认的 CUDA Runtime API 方法（要求 CUDA 初始化成功）更弱。在某些情况下，基于 NVML 的检查可能成功，但后续的 CUDA 初始化会失败。
>
> 现在我们有了 `args.device`，我们可以使用它在所需设备上创建张量。
>
>     x = torch.empty((8, 42), device=args.device)
>     net = Network().to(device=args.device)
>
> 这可以在许多情况下用于生成设备无关代码。以下是使用数据加载器时的示例：
>
>     cuda0 = torch.device('cuda:0')  # CUDA GPU 0
>     for i, x in enumerate(train_loader):
>         x = x.to(cuda0)
>
> 在系统上使用多个 GPU 时，可以使用 `CUDA_VISIBLE_DEVICES` 环境标志来管理 PyTorch 可用的 GPU。如上所述，要手动控制张量创建在哪个 GPU 上，最佳实践是使用 `torch.cuda.device` 上下文管理器。
>
>     print("Outside device is 0")  # 在设备 0 上（大多数场景中的默认设备）
>     with torch.cuda.device(1):
>         print("Inside device is 1")  # 在设备 1 上
>     print("Outside device is still 0")  # 在设备 0 上
>
> 如果您有一个张量，并希望在同一设备上创建相同类型的新张量，则可以使用 `torch.Tensor.new_*` 方法（参见 `torch.Tensor`）。 虽然前面提到的 `torch.*` 工厂函数（`tensor-creation-ops`）依赖于当前 GPU 上下文和传入的属性参数，但 `torch.Tensor.new_*` 方法会保留张量的设备和其他属性。
>
> 这是在创建模块时推荐的做法，其中在前向传播期间需要在内部创建新张量。
>
>     cuda = torch.device('cuda')
>     x_cpu = torch.empty(2)
>     x_gpu = torch.empty(2, device=cuda)
>     x_cpu_long = torch.empty(2, dtype=torch.int64)
>
>     y_cpu = x_cpu.new_full([3, 2], fill_value=0.3)
>     print(y_cpu)
>
>         tensor([[ 0.3000,  0.3000],
>                 [ 0.3000,  0.3000],
>                 [ 0.3000,  0.3000]])
>
>     y_gpu = x_gpu.new_full([3, 2], fill_value=-5)
>     print(y_gpu)
>
>         tensor([[-5.0000, -5.0000],
>                 [-5.0000, -5.0000],
>                 [-5.0000, -5.0000]], device='cuda:0')
>
>     y_cpu_long = x_cpu_long.new_tensor([[1, 2, 3]])
>     print(y_cpu_long)
>
>         tensor([[ 1,  2,  3]])
>
> 如果您想创建与另一个张量相同类型和大小的张量，并用 1 或 0 填充，`torch.ones_like` 或 `torch.zeros_like` 提供了便捷的辅助函数（它们也保留张量的 `torch.device` 和 `torch.dtype`）。
>
>     x_cpu = torch.empty(2, 3)
>     x_gpu = torch.empty(2, 3)
>
>     y_cpu = torch.ones_like(x_cpu)
>     y_gpu = torch.zeros_like(x_gpu)
>
> ### 使用固定内存缓冲区


> ⚠️ **警告**
> 这是一个高级技巧。如果过度使用固定内存，当 RAM 不足时可能导致严重问题，并且您应该注意固定操作通常是一项昂贵的操作。
>
> 当数据来自固定（页锁定）内存时，主机到 GPU 的复制会快得多。CPU 张量和存储暴露了一个 `torch.Tensor.pin_memory` 方法，该方法返回对象的副本，并将数据放入固定区域。
>
> 此外，一旦固定了张量或存储，您就可以使用异步 GPU 复制。只需向 `torch.Tensor.to` 或 `torch.Tensor.cuda` 调用传递一个额外的 `non_blocking=True` 参数。这可以用于将数据传输与计算重叠。
>
> 您可以通过向 `torch.utils.data.DataLoader` 的构造函数传递 `pin_memory=True`，使其返回放置在固定内存中的批次。
>
> ### 使用 nn.parallel.DistributedDataParallel 替代 multiprocessing 或 nn.DataParallel
>
> 大多数涉及批量输入和多个 GPU 的用例应默认使用 `torch.nn.parallel.DistributedDataParallel` 来利用多个 GPU。
>
> 将 CUDA 模型与 `torch.multiprocessing` 结合使用有重要的注意事项；除非精确满足数据处理要求，否则您的程序很可能出现不正确或未定义的行为。
>
> 建议使用 `torch.nn.parallel.DistributedDataParallel` 而不是 `torch.nn.DataParallel` 进行多 GPU 训练，即使只有一个节点。
>
> `torch.nn.parallel.DistributedDataParallel` 和 `torch.nn.DataParallel` 的区别在于：`torch.nn.parallel.DistributedDataParallel` 使用多进程，为每个 GPU 创建一个进程，而 `torch.nn.DataParallel` 使用多线程。通过使用多进程，每个 GPU 都有其专用进程，这避免了 Python 解释器 GIL 带来的性能开销。
>
> 如果您使用 `torch.nn.parallel.DistributedDataParallel`，可以使用 [torch.distributed.launch] 工具来启动您的程序，参见 `distributed-launch`。
>
> ## CUDA 图
>
> CUDA 图是 CUDA 流及其依赖流执行的工作（主要是内核及其参数）的记录。 关于底层 CUDA API 的一般原理和细节，请参见 [Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) 和 CUDA C 编程指南的 [Graphs section](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)。
>
> PyTorch 支持使用 [流捕获]() 来构建 CUDA 图，该模式将 CUDA 流置于\*捕获模式\*。发送到捕获流的 CUDA 工作实际上不会在 GPU 上运行。相反，工作被记录在图中。
>
> 捕获之后，可以\*启动\*图来运行 GPU 工作任意多次。每次重放都使用相同的参数运行相同的内核。对于指针参数，这意味着使用相同的内存地址。通过在每次重放之前用新数据（例如，来自新批次）填充输入内存，您可以在新数据上重新运行相同的工作。
>
> ### 为什么使用 CUDA 图？
>
> 重放图牺牲了典型即时执行的动态灵活性，以换取\*\*显著降低的 CPU 开销\*\*。图的参数和内核是固定的，因此图重放跳过了所有参数设置和内核分发的层级，包括 Python、C++ 和 CUDA 驱动程序的开销。在底层，重放通过一次调用 [cudaGraphLaunch](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597) 将整个图的工作提交给 GPU。重放中的内核在 GPU 上的执行也稍快一些，但消除 CPU 开销是主要的好处。
>
> 如果您的网络全部或部分是图安全的（通常这意味着静态形状和静态控制流，但请参阅其他 `限制<capture-constraints>`），并且您怀疑其运行时至少在一定程度上受 CPU 限制，那么您应该尝试 CUDA 图。
>
> ### PyTorch API


> ⚠️ **警告**
> 此 API 处于测试阶段，未来版本中可能会更改。
>
> PyTorch 通过原始的 `torch.cuda.CUDAGraph` 类和两个便捷包装器 `torch.cuda.graph` 和 `torch.cuda.make_graphed_callables` 来暴露图。
>
> `torch.cuda.graph` 是一个简单、通用的上下文管理器，用于捕获其上下文中的 CUDA 工作。 在捕获之前，通过运行几次即时迭代来预热要捕获的工作负载。预热必须在侧流上进行。 因为图在每次重放时都从相同的内存地址读取和写入，所以您必须在捕获期间保持对保存输入和输出数据的张量的长期引用。 要在新的输入数据上运行图，请将新数据复制到捕获的输入张量，重放图，然后从捕获的输出张量读取新的输出。 示例:
>
>     g = torch.cuda.CUDAGraph()
>
>     # 用于捕获的占位符输入
>     static_input = torch.empty((5,), device="cuda")
>
> \# 捕获前的预热 s = torch.cuda.Stream() s.wait_stream(torch.cuda.current_stream()) with torch.cuda.stream(s): for \_ in range(3): static_output = static_input \* 2 torch.cuda.current_stream().wait_stream(s)
>
> \# 捕获计算图 \# 为允许捕获，在上下文中自动将侧流设置为当前流 with torch.cuda.graph(g): static_output = static_input \* 2
>
> \# 用新数据填充计算图的输入内存以进行计算 [static_input.copy]()(torch.full((5,), 3, device=\"cuda\")) g.replay() \# static_output 保存结果 print(static_output) \# 全为 3 \* 2 = 6
>
> \# 用更多数据填充计算图的输入内存以进行计算 [static_input.copy]()(torch.full((5,), 4, device=\"cuda\")) g.replay() print(static_output) \# 全为 4 \* 2 = 8
>
> 有关实际和高级模式，请参阅 `全网络捕获<whole-network-capture>`、 `与 torch.cuda.amp 一起使用<graphs-with-amp>` 和 `与多流一起使用<multistream-capture>`。
>
> `torch.cuda.make_graphed_callables` 更为复杂。 `torch.cuda.make_graphed_callables` 接受 Python 函数和 `torch.nn.Module`s。对于每个传入的函数或模块， 它会分别创建前向传播和后向传播的计算图。请参阅 `部分网络捕获<partial-network-capture>`。
>
> #### 约束条件
>
> 如果一组操作不违反以下任何约束，则它是\*可捕获的\*。
>
> 这些约束适用于 `torch.cuda.graph` 上下文中的所有工作，以及您传递给 `torch.cuda.make_graphed_callables` 的任何可调用对象的前向和后向传播中的所有工作。
>
> 违反以下任何一条都可能导致运行时错误：
>
> - 捕获必须在非默认流上进行。（仅当您使用原始的 `CUDAGraph.capture_begin<torch.cuda.CUDAGraph.capture_begin>` 和 `CUDAGraph.capture_end<torch.cuda.CUDAGraph.capture_end>` 调用时才需关注此问题。 `torch.cuda.graph` 和 `torch.cuda.make_graphed_callables` 会为您设置一个侧流。）
> - 禁止同步 CPU 和 GPU 的操作（例如，`.item()` 调用）。
> - 允许 CUDA RNG 操作，并且在计算图中使用多个 `torch.Generator` 实例时， 必须在图捕获之前使用 `CUDAGraph.register_generator_state<torch.cuda.CUDAGraph.register_generator_state>` 进行注册。 避免在捕获期间使用 `Generator.get_state<torch.get_state>` 和 `Generator.set_state<torch.set_state>`； 相反，应使用 `Generator.graphsafe_set_state<torch.Generator.graphsafe_set_state>` 和 `Generator.graphsafe_get_state<torch.Generator.graphsafe_get_state>` 来安全地管理计算图上下文中的生成器状态。这确保了 CUDA 计算图中正确的 RNG 操作和生成器管理。
> - 禁止动态控制流（基于 CPU 或 GPU 数据），除非它基于 GPU 数据并通过高阶操作符 torch.cond() 实现。 请参阅 `数据依赖控制流<graph-data-dependent-control-flow>`。
>
> 违反以下任何一条可能导致静默数值错误或未定义行为：
>
> - 在一个进程中，一次只能进行一个捕获。
> - 捕获进行时，此进程中（在任何线程上）不得运行任何非捕获的 CUDA 工作。
> - CPU 工作不会被捕获。如果捕获的操作包含 CPU 工作，重放期间该工作将被省略。
> - 每次重放都读取和写入相同的（虚拟）内存地址。
> - 禁止动态形状。计算图假定捕获的操作序列中的每个张量在每次重放时都具有相同的大小和布局。
> - 允许在捕获中使用多个流，但存在 `限制<multistream-capture>`。
>
> #### 非约束条件
>
> - 一旦捕获，计算图可以在任何流上重放。
>
> ### 全网络捕获
>
> 如果您的整个网络是可捕获的，您可以捕获并重放整个迭代：:
>
> > N, D_in, H, D_out = 640, 4096, 2048, 1024 model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.Dropout(p=0.2), torch.nn.Linear(H, D_out), torch.nn.Dropout(p=0.1)).cuda() loss_fn = torch.nn.MSELoss() optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
> >
> > \# 用于捕获的占位符 static_input = torch.randn(N, D_in, device=\'cuda\') static_target = torch.randn(N, D_out, device=\'cuda\')
> >
> > \# 预热 \# 为方便起见，此处使用 static_input 和 static_target， \# 但在实际设置中，由于预热包括 optimizer.step() \# 您必须使用几批真实数据。 s = torch.cuda.Stream() s.wait_stream(torch.cuda.current_stream()) with torch.cuda.stream(s): for i in range(3): optimizer.zero_grad(set_to_none=True) y_pred = model(static_input) loss = loss_fn(y_pred, static_target) loss.backward() optimizer.step() torch.cuda.current_stream().wait_stream(s)
> >
> > \# 捕获 g = torch.cuda.CUDAGraph() \# 在捕获前将梯度设置为 None，以便 backward() 将从计算图的私有池中分配 .grad 属性 optimizer.zero_grad(set_to_none=True) with torch.cuda.graph(g): static_y_pred = model(static_input) static_loss = loss_fn(static_y_pred, static_target) static_loss.backward() optimizer.step()
> >
> > real_inputs = \[torch.rand_like(static_input) for \_ in range(10)\] real_targets = \[torch.rand_like(static_target) for \_ in range(10)\]
> >
> > for data, target in zip(real_inputs, real_targets):
> >
> > :   \# 用新数据填充图的输入内存以进行计算 [static_input.copy]()(data) [static_target.copy]()(target) \# replay() 包含前向传播、反向传播和优化器步骤。 \# 你甚至不需要在迭代之间调用 optimizer.zero_grad()， \# 因为捕获的反向传播会原地重新填充静态 .grad 张量。 g.replay() \# 参数已更新。static_y_pred、static_loss 和 .grad \# 属性保存了本次迭代数据计算的值。
>
> ### 部分网络捕获
>
> 如果你的网络中有部分不适合捕获（例如，由于动态控制流、动态形状、CPU 同步或必要的 CPU 端逻辑），你可以以即时执行的方式运行这些不安全的部分，并使用 `torch.cuda.make_graphed_callables` 仅对可安全捕获的部分进行图化。
>
> 默认情况下，`torch.cuda.make_graphed_callables` 返回的可调用对象是支持自动求导的，并且可以在训练循环中直接替换你传入的函数或 `nn.Module<torch.nn.Module>`。
>
> `torch.cuda.make_graphed_callables` 在内部创建 `torch.cuda.CUDAGraph` 对象，运行预热迭代，并根据需要维护静态输入和输出。因此（与 `torch.cuda.graph` 不同），你不需要手动处理这些。
>
> 在以下示例中，数据相关的动态控制流意味着网络无法端到端捕获，但 `torch.cuda.make_graphed_callables` 允许我们无论如何都将图安全的部分捕获并作为图运行：:
>
> > N, D_in, H, D_out = 640, 4096, 2048, 1024
> >
> > module1 = torch.nn.Linear(D_in, H).cuda() module2 = torch.nn.Linear(H, D_out).cuda() module3 = torch.nn.Linear(H, D_out).cuda()
> >
> > loss_fn = torch.nn.MSELoss() optimizer = torch.optim.SGD(chain(module1.parameters(), module2.parameters(), module3.parameters()), lr=0.1)
> >
> > \# 用于捕获的示例输入 \# 示例输入的 requires_grad 状态必须与 \# 每个可调用对象将看到的真实输入的 requires_grad 状态匹配。 x = torch.randn(N, D_in, device=\'cuda\') h = torch.randn(N, H, device=\'cuda\', requires_grad=True)
> >
> > module1 = torch.cuda.make_graphed_callables(module1, (x,)) module2 = torch.cuda.make_graphed_callables(module2, (h,)) module3 = torch.cuda.make_graphed_callables(module3, (h,))
> >
> > real_inputs = \[torch.rand_like(x) for \_ in range(10)\] real_targets = \[torch.randn(N, D_out, device=\"cuda\") for \_ in range(10)\]
> >
> > for data, target in zip(real_inputs, real_targets):
> >
> > :   optimizer.zero_grad(set_to_none=True)
> >
> >     tmp = module1(data) \# 前向操作以图的形式运行
> >
> >     if tmp.sum().item() \> 0:
> >
> >     :   tmp = module2(tmp) \# 前向操作以图的形式运行
> >
> >     else:
> >
> >     :   tmp = module3(tmp) \# 前向操作以图的形式运行
> >
> >     loss = loss_fn(tmp, target) \# module2 或 module3（取决于选择哪个）的反向操作， \# 以及 module1 的反向操作，都以图的形式运行 loss.backward() optimizer.step()
>
> ### 与 torch.cuda.amp 一起使用
>
> 对于典型的优化器，`GradScaler.step<torch.cuda.amp.GradScaler.step>` 会使 CPU 与 GPU 同步，这在捕获期间是被禁止的。为了避免错误，要么使用 `部分网络捕获<partial-network-capture>`，要么（如果前向传播、损失计算和反向传播是捕获安全的）捕获前向传播、损失计算和反向传播，但不捕获优化器步骤：:
>
> > \# 预热 \# 在实际设置中，使用几批真实数据。 s = torch.cuda.Stream() s.wait_stream(torch.cuda.current_stream()) with torch.cuda.stream(s): for i in range(3): optimizer.zero_grad(set_to_none=True) with torch.cuda.amp.autocast(): y_pred = model(static_input) loss = loss_fn(y_pred, static_target) scaler.scale(loss).backward() scaler.step(optimizer) scaler.update() torch.cuda.current_stream().wait_stream(s)
> >
> > \# 捕获 g = torch.cuda.CUDAGraph() optimizer.zero_grad(set_to_none=True) with torch.cuda.graph(g): with torch.cuda.amp.autocast(): static_y_pred = model(static_input) static_loss = loss_fn(static_y_pred, static_target) scaler.scale(static_loss).backward() \# 不要捕获 scaler.step(optimizer) 或 scaler.update()
> >
> > real_inputs = \[torch.rand_like(static_input) for \_ in range(10)\] real_targets = \[torch.rand_like(static_target) for \_ in range(10)\]
> >
> > for data, target in zip(real_inputs, real_targets):
> >
> > :   [static_input.copy]()(data) [static_target.copy]()(target) g.replay() \# 以即时执行方式运行 scaler.step 和 scaler.update scaler.step(optimizer) scaler.update()
>
> ### 与多流一起使用
>
> 捕获模式会自动传播到与捕获流同步的任何流。在捕获过程中，你可以通过向不同流发出调用来暴露并行性，但整个流依赖关系 DAG 必须在捕获开始后从初始捕获流分支出来，并在捕获结束前重新汇合到初始流：:
>
> > 
> >
> > with torch.cuda.graph(g):
> >
> > :   \# 在上下文管理器入口处，torch.cuda.current_stream() \# 是初始捕获流
> >
> >     \# 错误（没有从初始流分支出来或重新汇合） with torch.cuda.stream(s): cuda_work()
> >
> >     \# 正确： \# 从初始流分支出来 s.wait_stream(torch.cuda.current_stream()) with torch.cuda.stream(s): cuda_work() \# 在捕获结束前重新汇合到初始流 torch.cuda.current_stream().wait_stream(s)


> 📝 **注意**
> 为避免使用 nsight systems 或 nvprof 查看重放的高级用户产生混淆： 与即时执行不同，图在捕获过程中将非平凡的流 DAG 视为提示而非命令。在重放期间，图可能会将独立操作重新组织到不同的流上，或以不同的顺序将其加入队列（同时尊重原始 DAG 的整体依赖关系）。
>
> ### 与 DistributedDataParallel 一起使用
>
> #### NCCL \< 2.9.6
>
> 早于 2.9.6 版本的 NCCL 不允许捕获集合通信操作。 您必须使用 `部分网络捕获<partial-network-capture>`， 这将使 allreduce 操作延迟到反向传播的图外部分执行。
>
> 在使用 DDP 包装网络 *之前*，对可图化的网络部分调用 `torch.cuda.make_graphed_callables`。
>
> #### NCCL \>= 2.9.6
>
> 2.9.6 或更高版本的 NCCL 允许在图中进行集合通信。 捕获 `整个反向传播过程<whole-network-capture>` 的方法是可行的选项，但需要三个设置步骤。
>
> 1.  禁用 DDP 的内部异步错误处理：:
>
>     > os.environ\[\"NCCL_ASYNC_ERROR_HANDLING\"\] = \"0\" torch.distributed.init_process_group(\...)
>
> 2.  在完整反向传播捕获之前，必须在侧流上下文中构建 DDP：:
>
>     > 
>     >
>     > with torch.cuda.stream(s):
>     >
>     > :   model = DistributedDataParallel(model)
>
> 3.  您的预热必须在捕获前至少运行 11 次启用 DDP 的即时迭代。
>
> ### 图内存管理
>
> 捕获的图每次重放时都作用于相同的虚拟地址。 如果 PyTorch 释放了内存，后续重放可能会触发非法内存访问。 如果 PyTorch 将内存重新分配给新的张量，重放可能会破坏这些张量所看到的值。因此，图使用的虚拟地址必须在多次重放之间为图保留。PyTorch 缓存分配器通过检测捕获何时进行，并从图私有的内存池中满足捕获的分配来实现这一点。私有池会一直保持活动状态，直到其 `torch.cuda.CUDAGraph` 对象和捕获期间创建的所有张量都超出作用域。
>
> 私有池会自动维护。默认情况下，分配器为每次捕获创建一个独立的私有池。如果您捕获多个图，这种保守的方法确保图重放永远不会破坏彼此的值，但有时会不必要地浪费内存。
>
> #### 跨捕获共享内存
>
> 为了节省私有池中占用的内存，`torch.cuda.graph` 和 `torch.cuda.make_graphed_callables` 可选地允许不同的捕获共享同一个私有池。 如果您知道一组图将始终按照它们被捕获的相同顺序重放，并且永远不会并发重放，那么它们共享一个私有池是安全的。
>
> `torch.cuda.graph` 的 `pool` 参数是使用特定私有池的提示，可用于在多个图之间共享内存，如下所示：:
>
> > g1 = torch.cuda.CUDAGraph() g2 = torch.cuda.CUDAGraph()
> >
> > \# (为 g1 和 g2 创建静态输入，运行它们工作负载的预热\...)
> >
> > \# 捕获 g1 with torch.cuda.graph(g1): static_out_1 = g1_workload(static_in_1)
> >
> > \# 捕获 g2，提示 g2 可能与 g1 共享内存池 with torch.cuda.graph(g2, pool=g1.pool()): static_out_2 = g2_workload(static_in_2)
> >
> > [static_in_1.copy]()(real_data_1) [static_in_2.copy]()(real_data_2) g1.replay() g2.replay()
>
> 在不相互依赖输出的独立图之间共享内存池也是安全的，前提是它们永远不会并发运行。 请注意，当它们共享一个池时，重放一个图可能会破坏另一个图的输出，除非事先对输出调用 `torch.Tensor.clone`。 这种模式在运行时接受可变批次大小的推理服务器中经常使用。 vLLM 是一个显著的例子；参见 [此处](https://github.com/vllm-project/vllm/blob/938a81692ea318e59ead4750e7e7425bfd6a4896/vllm/platforms/interface.py#L508-L515) 和 [此处](https://github.com/vllm-project/vllm/blob/938a81692ea318e59ead4750e7e7425bfd6a4896/vllm/compilation/cuda_graph.py#L86-L89)。
>
> 对于 `torch.cuda.make_graphed_callables`，如果您想要对多个可调用对象进行图化，并且知道它们将始终以相同的顺序运行（且永远不会并发），请按照它们在实时工作负载中运行的顺序将它们作为元组传递，`torch.cuda.make_graphed_callables` 将使用共享的私有池捕获它们的图。
>
> 如果在实时工作负载中，您的可调用对象的运行顺序偶尔会改变，或者它们将并发运行，则不允许将它们作为元组传递给单次 `torch.cuda.make_graphed_callables` 调用。相反，您必须为每个可调用对象分别调用 `torch.cuda.make_graphed_callables`。
>
> ### 数据依赖控制流
>
> 如果控制流是使用 torch.cond() 实现的，那么数据依赖的控制流可以与 CUDA 图一起使用。如果您的函数使用了这个函数，使用 \"cudagraphs\" 后端编译它将通过 [条件节点]() 在生成的 CUDA 图中启用控制流。
>
> 使用 torch.cond() 对即时模式代码进行 CUDA 图捕获也将有效。
>
> 目前尚不支持对 torch.compile 的 inductor 后端，但不存在根本性的障碍。
>
> 下面演示了在使用 torch.cond 的代码上使用 cudagraphs 后端进行 torch.compile 的示例：:
>
> > import torch
> >
> > def true_fn(x):
> >
> > :   return x.sin()
> >
> > def false_fn(x):
> >
> > :   return x.cos()
> >
> > x = torch.randn(4, device=\"cuda\", requires_grad=False) pred = torch.tensor(False, device=\"cuda\", requires_grad=False) def foo(pred, x): with torch.inference_mode(): return torch.cond(pred, true_fn, false_fn, \[x\])
>
> \# 首次调用将运行 eager 模式进行预热，第二次调用将执行图捕获并重放，第三次及后续调用将仅执行图重放。 compiled_foo = torch.compile(foo, backend=\"cudagraphs\") for i in range(3): y = compiled_foo(pred, x)
>
> \# 将输出 x.sin() y = compiled_foo(\~pred, x)

