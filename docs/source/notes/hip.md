# HIP (ROCm) 语义

ROCm™ 是 AMD 的开源软件平台，用于 GPU 加速的高性能计算和机器学习。HIP 是 ROCm 的 C++ 方言，旨在简化 CUDA 应用程序向可移植 C++ 代码的转换。在将 PyTorch 等现有 CUDA 应用程序转换为可移植 C++ 时，以及需要 AMD 和 NVIDIA 之间可移植性的新项目中，都会使用 HIP。

## HIP 接口复用 CUDA 接口

用于 HIP 的 PyTorch 有意复用了现有的 `torch.cuda` 接口。这有助于加速现有 PyTorch 代码和模型的移植，因为几乎不需要（如果有的话）修改代码。

`cuda-semantics` 中的示例在 HIP 上完全适用：:

> cuda = torch.device(\'cuda\') \# 默认 HIP 设备 cuda0 = torch.device(\'cuda:0\') \# \'rocm\' 或 \'hip\' 无效，请使用 \'cuda\' cuda2 = torch.device(\'cuda:2\') \# GPU 2（这些索引从 0 开始）
>
> x = torch.tensor(\[1., 2.\], device=cuda0) \# x.device 是 device(type=\'cuda\', index=0) y = torch.tensor(\[1., 2.\]).cuda() \# y.device 是 device(type=\'cuda\', index=0)
>
> with torch.cuda.device(1):
>
> :   \# 在 GPU 1 上分配张量 a = torch.tensor(\[1., 2.\], device=cuda)
>
>     \# 将张量从 CPU 传输到 GPU 1 b = torch.tensor(\[1., 2.\]).cuda() \# a.device 和 b.device 是 device(type=\'cuda\', index=1)
>
>     \# 你也可以使用 `Tensor.to` 来传输张量： b2 = torch.tensor(\[1., 2.\]).to(device=cuda) \# b.device 和 b2.device 是 device(type=\'cuda\', index=1)
>
>     c = a + b \# c.device 是 device(type=\'cuda\', index=1)
>
>     z = x + y \# z.device 是 device(type=\'cuda\', index=0)
>
>     \# 即使在上下文中，你也可以指定设备 \# （或者给 .cuda 调用一个 GPU 索引） d = torch.randn(2, device=cuda2) e = torch.randn(2).to(cuda2) f = torch.randn(2).cuda(cuda2) \# d.device, e.device, 和 f.device 都是 device(type=\'cuda\', index=2)

## 检查 HIP

无论你使用的是用于 CUDA 还是 HIP 的 PyTorch，调用 `torch.cuda.is_available` 的结果都是相同的。如果你使用的是支持 GPU 的 PyTorch 构建版本，它将返回 [True]。如果你必须检查正在使用的 PyTorch 版本，请参考以下示例：:

> 
>
> if torch.cuda.is_available() and torch.version.hip:
>
> :   \# 为 HIP 执行特定操作
>
> elif torch.cuda.is_available() and torch.version.cuda:
>
> :   \# 为 CUDA 执行特定操作

## ROCm 上的 TensorFloat-32(TF32)

ROCm 不支持 TF32。

## 内存管理

PyTorch 使用缓存内存分配器来加速内存分配。这允许在不进行设备同步的情况下快速释放内存。然而，分配器管理的未使用内存在 `rocm-smi` 中仍会显示为已使用。你可以使用 `torch.cuda.memory_allocated` 和 `torch.cuda.max_memory_allocated` 来监控张量占用的内存，并使用 `torch.cuda.memory_reserved` 和 `torch.cuda.max_memory_reserved` 来监控缓存分配器管理的总内存量。调用 `torch.cuda.empty_cache` 会释放 PyTorch 中所有\*\*未使用的\*\*缓存内存，以便其他 GPU 应用程序可以使用这些内存。但是，张量占用的 GPU 内存不会被释放，因此无法增加 PyTorch 可用的 GPU 内存量。

对于更高级的用户，我们通过 `torch.cuda.memory_stats` 提供更全面的内存基准测试。我们还提供通过 `torch.cuda.memory_snapshot` 捕获内存分配器状态完整快照的能力，这可以帮助你理解代码产生的底层分配模式。

要调试内存错误，请在环境中设置 `PYTORCH_NO_HIP_MEMORY_CACHING=1` 以禁用缓存。为了方便移植，也接受 `PYTORCH_NO_CUDA_MEMORY_CACHING=1`。

## hipBLAS 工作空间

对于每个 hipBLAS 句柄和 HIP 流的组合，如果该句柄和流组合执行需要工作空间的 hipBLAS 内核，则会分配一个 hipBLAS 工作空间。为了避免重复分配工作空间，除非调用 `torch._C._cuda_clearCublasWorkspaces()`，否则这些工作空间不会被释放；请注意，对于 CUDA 或 HIP，这是同一个函数。每次分配的工作空间大小可以通过环境变量 `HIPBLAS_WORKSPACE_CONFIG` 指定，格式为 `:[SIZE]:[COUNT]`。例如，环境变量 `HIPBLAS_WORKSPACE_CONFIG=:4096:2:16:8` 指定总大小为 `2 * 4096 + 8 * 16 KiB` 或 8 MiB。默认工作空间大小为 32 MiB；MI300 及更新版本默认为 128 MiB。要强制 hipBLAS 避免使用工作空间，请设置 `HIPBLAS_WORKSPACE_CONFIG=:0:0`。为了方便起见，也接受 `CUBLAS_WORKSPACE_CONFIG`。

## hipFFT/rocFFT 计划缓存

不支持设置 hipFFT/rocFFT 计划缓存的大小。

## torch.distributed 后端

目前，ROCm 上仅支持 torch.distributed 的 \"nccl\" 和 \"gloo\" 后端。

## C++ 中 CUDA API 到 HIP API 的映射

请参考：https://rocm.docs.amd.com/projects/HIP/en/latest/reference/api_syntax.html

注意：CUDA_VERSION 宏、cudaRuntimeGetVersion 和 cudaDriverGetVersion API 在语义上并不映射到与 HIP_VERSION 宏、hipRuntimeGetVersion 和 hipDriverGetVersion API 相同的值。在进行版本检查时，请不要互换使用它们。

例如：不要使用

`#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000` 来隐式排除 ROCm/HIP，

使用以下代码来不采用 ROCm/HIP 的代码路径：

`#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000 && !defined(USE_ROCM)`

或者，如果希望采用 ROCm/HIP 的代码路径：

`#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11000) || defined(USE_ROCM)`

或者，如果希望仅针对特定的 HIP 版本采用 ROCm/HIP 的代码路径：

`#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11000) || (defined(USE_ROCM) && ROCM_VERSION >= 40300)`

## 参考 CUDA 语义文档

对于此处未列出的任何部分，请参考 CUDA 语义文档： `cuda-semantics`

## 启用内核断言

ROCm 支持内核断言，但由于性能开销，默认是禁用的。可以通过从源代码重新编译 PyTorch 来启用它。

请在 cmake 命令参数中添加以下行：:

> -DROCM_FORCE_ENABLE_GPU_ASSERTS:BOOL=ON

## 启用/禁用 ROCm 可组合内核

为 SDPA 和 GEMM 启用可组合内核 (CK) 是一个分为两部分的过程。首先，用户必须在构建 PyTorch 时将相应的环境变量设置为 \'1\'

SDPA： `USE_ROCM_CK_SDPA=1`

GEMMs： `USE_ROCM_CK_GEMM=1`

其次，用户必须通过相应的 Python 调用显式请求将 CK 用作后端库

SDPA： `setROCmFAPreferredBackend('<choice>')`

GEMMs： `setBlasPreferredBackend('<choice>')`

要在任一场景中启用 CK，只需向这些函数传递 \'ck\' 即可。

为了将后端设置为 CK，用户必须使用正确的环境变量进行构建。否则，PyTorch 将打印警告并使用"默认"后端。对于 GEMMs，这将路由到 hipblas，对于 SDPA，它将路由到 aotriton。
