# AOTInductor 调试指南

在使用 [AOT Inductor](./torch.compiler_aot_inductor.md) 时，如果遇到 CUDA 非法内存访问（IMA）错误，本指南提供了一种系统性的方法来调试此类错误。AOT Inductor 是 PT2 技术栈的一部分，类似于 torch.compile，但它会生成一个可以在 C++ 环境中工作的编译产物。CUDA 非法内存错误可能非确定性地发生，有时甚至看起来是暂时性的。

从高层次来看，调试 CUDA IMA 错误主要有三个步骤：

- **完整性检查**：在进行深入调试之前，使用基本的调试标志来捕获常见问题。
- **定位 CUDA IMA**：使错误确定化并识别有问题的内核。
- **识别有问题的内核**：使用中间值调试来检查内核的输入和输出。

## 步骤 1：完整性检查

在深入进行可靠复现错误之前，先尝试一些现有的调试标志：

```bash
AOTI_RUNTIME_CHECK_INPUTS=1
TORCHINDUCTOR_NAN_ASSERTS=1
```

这些标志在编译时（更准确地说，是在代码生成时）生效：

- `AOTI_RUNTIME_CHECK_INPUTS=1` 检查输入是否满足编译期间使用的同一组守卫条件。更多详情请参阅 `torch.compiler_troubleshooting`。
- `TORCHINDUCTOR_NAN_ASSERTS=1` 在每个 Inductor 内核的前后添加代码生成，以检查 NaN 值。

## 步骤 2：定位 CUDA IMA

一个难点在于 CUDA IMA 错误可能是非确定性的。它们可能发生在不同的位置，有时甚至根本不发生（尽管这仅仅意味着数值计算在静默中出错）。使用以下两个标志，我们可以确定性地触发错误：

```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1
CUDA_LAUNCH_BLOCKING=1
```

这些标志在运行时生效：

- `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 禁用 PyTorch 的缓存分配器，该分配器会立即分配比所需更大的缓冲区以减少缓冲区分配次数。这通常是 CUDA 非法内存访问错误非确定性的原因。
![PyTorch 缓存分配器如何掩盖 CUDA 非法内存访问错误](../../_static/img/aoti_debugging_guide/cuda_ima_cca.png)
*图：PyTorch 缓存分配器如何掩盖 CUDA 非法内存访问错误*

- `CUDA_LAUNCH_BLOCKING=1` 强制内核逐个启动。如果没有这个标志，由于内核是异步启动的，我们会得到著名的“CUDA 内核错误可能在某个其他 API 调用时被异步报告”警告。

## 步骤 3：使用中间值调试器识别有问题的内核

AOTI 中间值调试器可以帮助精确定位有问题的内核，并获取该内核的输入和输出信息。

首先，使用：

```bash
AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=3
```

此标志在编译时生效，并在运行时逐个打印内核。结合之前的标志，这将让我们知道在错误发生之前启动了哪个内核。

但是，需要注意的是，仅仅因为错误发生在那个内核中，并不意味着该内核有问题。例如，可能是一个更早的内核有问题并产生了一些错误的输出。因此，自然的下一步是检查有问题的内核的输入：

```bash
AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT="triton_poi_fused_add_ge_logical_and_logical_or_lt_231,_add_position_embeddings_kernel_5" AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=2
```

要打印的过滤内核环境变量包含你想要检查的内核名称。如果内核的输入不符合预期，那么你就需要检查产生错误输入的内核。

## 其他调试工具

### 日志记录与追踪

- **tlparse / TORCH_TRACE**：提供完整的输出代码以供检查，并记录所使用的守卫条件集。更多详情请参阅 `tlparse / TORCH_TRACE <tlparse-torch-trace>`。
- **TORCH_LOGS**：使用 `TORCH_LOGS="+inductor,output_code"` 查看更多的 PT2 内部日志。更多详情请参阅 `TORCH_LOGS <torch-logs>`。
- **TORCH_SHOW_CPP_STACKTRACES**：设置 `TORCH_SHOW_CPP_STACKTRACES=1` 以潜在地查看更多堆栈跟踪。

### 常见问题来源

- [**动态形状**](./torch.compiler_dynamic_shapes.md)：历史上是许多 IMA 错误的来源。在调试动态形状场景时要特别注意。
- **自定义操作**：尤其是在 C++ 中实现并与动态形状一起使用时。需要将元函数 Symint 化。