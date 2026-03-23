# torch.cuda


## 随机数生成器


## 通信集合操作


## 流和事件


## 图（测试版）


## 内存管理


## NVIDIA 工具扩展 (NVTX)


## Jiterator（测试版）


## 可调操作

某些操作可以使用多个库或多种技术来实现。例如，对于 CUDA 或 ROCm，GEMM 可以分别使用 cublas/cublasLt 库或 hipblas/hipblasLt 库来实现。如何知道哪种实现最快并应该被选择？这就是可调操作（TunableOp）提供的功能。某些运算符已使用多种策略作为可调运算符实现。在运行时，所有策略都会被分析，最快的策略将被选择用于所有后续操作。

有关如何使用它的信息，请参阅 `文档 <cuda.tunable>`。

```{toctree}
:hidden: true

cuda.tunable
```

## 流消毒器（原型）

CUDA 消毒器是一个用于检测 PyTorch 中流之间同步错误的原型工具。
有关如何使用它的信息，请参阅 `文档 <cuda._sanitizer>`。

```{toctree}
:hidden: true

cuda._sanitizer
```

## GPUDirect 存储（原型）

`torch.cuda.gds` 中的 API 提供了某些 cuFile API 的轻量级封装，允许在 GPU 内存和存储之间直接进行内存访问传输，避免了 CPU 中的反弹缓冲区。更多详细信息，请参阅 [cufile api 文档](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufile-io-api)。

这些 API 可以在 CUDA 版本大于或等于 12.6 时使用。为了使用这些 API，必须确保系统已根据 [GPUDirect 存储文档](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/contents.html) 进行了适当配置以使用 GPUDirect 存储。

有关如何使用这些 API 的示例，请参阅 `~torch.cuda.gds.GdsFile` 的文档。


## 绿色上下文（实验性）

`torch.cuda.green_contexts` 提供了 CUDA 绿色上下文 API 的轻量级封装，以便为 CUDA 内核启用更通用的 SM 资源预留。

这些 API 可以在 CUDA 版本大于或等于 12.8 的 PyTorch 中使用。

有关如何使用这些 API 的示例，请参阅 `~torch.cuda.green_contexts.GreenContext` 的文档。


% 此模块需要记录。暂时添加在此处

% 用于跟踪目的


.. autofunction:: torch.cuda.nccl.is_available
```


