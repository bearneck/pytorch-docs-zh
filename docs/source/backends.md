```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# torch.backends

```{eval-rst}
.. automodule:: torch.backends
```

`torch.backends` 控制 PyTorch 支持的各种后端的行为。

这些后端包括：

- `torch.backends.cpu`
- `torch.backends.cuda`
- `torch.backends.cudnn`
- `torch.backends.cusparselt`
- `torch.backends.mha`
- `torch.backends.mps`
- `torch.backends.mkl`
- `torch.backends.mkldnn`
- `torch.backends.nnpack`
- `torch.backends.openmp`
- `torch.backends.opt_einsum`
- `torch.backends.xeon`

## torch.backends.cpu

```{eval-rst}
.. automodule:: torch.backends.cpu
```

```{eval-rst}
.. autofunction::  torch.backends.cpu.get_cpu_capability
```

## torch.backends.cuda

```{eval-rst}
.. automodule:: torch.backends.cuda
```

```{eval-rst}
.. autofunction::  torch.backends.cuda.is_built
```

```{eval-rst}
.. currentmodule:: torch.backends.cuda.matmul
```

```{eval-rst}
.. attribute::  allow_tf32

    一个 :class:`bool`，控制是否允许在 Ampere 或更新架构的 GPU 上使用 TensorFloat-32 张量核心进行矩阵乘法运算。allow_tf32 即将被弃用。参见 :ref:`tf32_on_ampere`。
```

```{eval-rst}
.. attribute::  allow_fp16_reduced_precision_reduction

    一个 :class:`bool`，控制是否允许在 fp16 GEMM 运算中使用降低精度的归约操作（例如，使用 fp16 累加类型）。
    赋值一个元组 ``(allow_reduced_precision, allow_splitk)`` 还可以让你切换在调度到 cuBLASLt 时是否可以使用 split-K 启发式方法。``allow_splitk`` 默认为 ``True``。
```

```{eval-rst}
.. attribute::  allow_bf16_reduced_precision_reduction

    一个 :class:`bool`，控制是否允许在 bf16 GEMM 运算中使用降低精度的归约操作。
    赋值一个元组 ``(allow_reduced_precision, allow_splitk)`` 还可以让你切换在调度到 cuBLASLt 时是否可以使用 split-K 启发式方法。``allow_splitk`` 默认为 ``True``。
```

```{eval-rst}
.. currentmodule:: torch.backends.cuda
```

```{eval-rst}
.. attribute::  cufft_plan_cache

    ``cufft_plan_cache`` 包含每个 CUDA 设备的 cuFFT 计划缓存。
    通过 `torch.backends.cuda.cufft_plan_cache[i]` 查询特定设备 `i` 的缓存。

    .. currentmodule:: torch.backends.cuda.cufft_plan_cache
    .. attribute::  size

        一个只读的 :class:`int`，显示当前 cuFFT 计划缓存中的计划数量。

    .. attribute::  max_size

        一个 :class:`int`，控制 cuFFT 计划缓存的容量。

    .. method::  clear()

        清空 cuFFT 计划缓存。
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.preferred_blas_library
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.preferred_rocm_fa_library
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.preferred_linalg_library
```

```{eval-rst}
.. autoclass:: torch.backends.cuda.SDPAParams
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.flash_sdp_enabled
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.enable_mem_efficient_sdp
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.mem_efficient_sdp_enabled
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.enable_flash_sdp
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.math_sdp_enabled
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.enable_math_sdp
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.cudnn_sdp_enabled
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.enable_cudnn_sdp
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.is_flash_attention_available
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.can_use_flash_attention
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.can_use_efficient_attention
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.can_use_cudnn_attention
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.sdp_kernel
```

## torch.backends.cudnn

```{eval-rst}
.. automodule:: torch.backends.cudnn
```

```{eval-rst}
.. autofunction:: torch.backends.cudnn.version
```

```{eval-rst}
.. autofunction:: torch.backends.cudnn.is_available
```

```{eval-rst}
.. attribute::  enabled

    一个 :class:`bool`，控制是否启用 cuDNN。
```

```{eval-rst}
.. attribute::  allow_tf32

    一个 :class:`bool`，控制是否允许在 Ampere 或更新架构的 GPU 上使用 TensorFloat-32 张量核心进行 cuDNN 卷积运算。allow_tf32 即将被弃用。参见 :ref:`tf32_on_ampere`。
```

```{eval-rst}
.. attribute::  deterministic

    一个 :class:`bool`，如果为 True，会导致 cuDNN 仅使用确定性的卷积算法。
    另请参见 :func:`torch.are_deterministic_algorithms_enabled` 和
    :func:`torch.use_deterministic_algorithms`。
```

```{eval-rst}
.. attribute::  benchmark

    一个 :class:`bool`，如果为 True，会导致 cuDNN 对多种卷积算法进行基准测试并选择最快的。
```

```{eval-rst}
.. attribute::  benchmark_limit

    一个 :class:`int`，指定当 `torch.backends.cudnn.benchmark` 为 True 时，cuDNN 卷积算法尝试的最大数量。将 `benchmark_limit` 设置为零以尝试所有可用的算法。请注意，此设置仅影响通过 cuDNN v8 API 调度的卷积。
```

```{eval-rst}
.. py:module:: torch.backends.cudnn.rnn
```

## torch.backends.cusparselt

```{eval-rst}
.. automodule:: torch.backends.cusparselt
```

```{eval-rst}
.. autofunction:: torch.backends.cusparselt.version
```

```{eval-rst}
.. autofunction:: torch.backends.cusparselt.is_available
```

## torch.backends.mha

```{eval-rst}
.. automodule:: torch.backends.mha
```

```{eval-rst}
.. autofunction::  torch.backends.mha.get_fastpath_enabled
```

```{eval-rst}
.. autofunction::  torch.backends.mha.set_fastpath_enabled

```

## torch.backends.miopen

```{eval-rst}
.. automodule:: torch.backends.miopen
```

```{eval-rst}
.. attribute::  immediate

一个 :class:`bool` 值，如果为 True，将导致 MIOpen 使用立即模式
(https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html)。

## torch.backends.mps

```{eval-rst}
.. automodule:: torch.backends.mps
```

```{eval-rst}
.. autofunction::  torch.backends.mps.is_available
```

```{eval-rst}
.. autofunction::  torch.backends.mps.is_built

```

## torch.backends.mkl

```{eval-rst}
.. automodule:: torch.backends.mkl
```

```{eval-rst}
.. autofunction::  torch.backends.mkl.is_available
```

```{eval-rst}
.. autoclass::  torch.backends.mkl.verbose

```

## torch.backends.mkldnn

```{eval-rst}
.. automodule:: torch.backends.mkldnn
```

```{eval-rst}
.. autofunction::  torch.backends.mkldnn.is_available
```

```{eval-rst}
.. autoclass::  torch.backends.mkldnn.verbose
```

## torch.backends.nnpack

```{eval-rst}
.. automodule:: torch.backends.nnpack
```

```{eval-rst}
.. autofunction::  torch.backends.nnpack.is_available
```

```{eval-rst}
.. autofunction::  torch.backends.nnpack.flags
```

```{eval-rst}
.. autofunction::  torch.backends.nnpack.set_flags
```

## torch.backends.openmp

```{eval-rst}
.. automodule:: torch.backends.openmp
```

```{eval-rst}
.. autofunction::  torch.backends.openmp.is_available
```

% 其他后端的文档需要在此处添加。
% 这里的自动模块只是为了确保检查运行，但目前它们实际上
% 不会向渲染页面添加任何内容。

```{eval-rst}
.. py:module:: torch.backends.quantized
```

```{eval-rst}
.. py:module:: torch.backends.xnnpack
```

```{eval-rst}
.. py:module:: torch.backends.kleidiai

.. autofunction:: torch.backends.kleidiai.is_available
```

## torch.backends.opt_einsum

```{eval-rst}
.. automodule:: torch.backends.opt_einsum
```

```{eval-rst}
.. autofunction:: torch.backends.opt_einsum.is_available
```

```{eval-rst}
.. autofunction:: torch.backends.opt_einsum.get_opt_einsum
```

```{eval-rst}
.. attribute::  enabled

    一个 :class:`bool` 值，用于控制是否启用 opt_einsum（默认为 ``True``）。如果启用，
    并且 opt_einsum 可用，torch.einsum 将使用 opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/path_finding.html)
    来计算最优的收缩路径以获得更快的性能。

    如果 opt_einsum 不可用，torch.einsum 将回退到默认的从左到右的收缩路径。
```

```{eval-rst}
.. attribute::  strategy

    一个 :class:`str` 值，用于指定当 ``torch.backends.opt_einsum.enabled`` 为 ``True`` 时尝试的策略。
    默认情况下，torch.einsum 将尝试 "auto" 策略，但也支持 "greedy" 和 "optimal" 策略。
    请注意，"optimal" 策略在输入数量上是阶乘级的，因为它会尝试所有可能的路径。
    更多细节请参阅 opt_einsum 的文档 (https://optimized-einsum.readthedocs.io/en/stable/path_finding.html)。

```

## torch.backends.xeon

```{eval-rst}
.. automodule:: torch.backends.xeon
```

```{eval-rst}
.. py:module:: torch.backends.xeon.run_cpu
```