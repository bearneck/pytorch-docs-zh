.. _cuda_environment_variables:

CUDA 环境变量
==========================
有关 CUDA 运行时环境变量的更多信息，请参阅 `CUDA 环境变量 <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars>`_。

**PyTorch 环境变量**

.. list-table::
  :header-rows: 1

  * - 变量
    - 描述
  * - ``PYTORCH_NO_CUDA_MEMORY_CACHING``
    - 如果设置为 ``1``，则禁用 CUDA 中的内存分配缓存。这对于调试很有用。
  * - ``PYTORCH_ALLOC_CONF``
    - 有关此环境变量的更深入解释，请参阅 :ref:`cuda-memory-management`。``PYTORCH_CUDA_ALLOC_CONF`` 是其别名，仅为向后兼容性而提供。
  * - ``PYTORCH_NVML_BASED_CUDA_CHECK``
    - 如果设置为 ``1``，则在导入检查 CUDA 是否可用的 PyTorch 模块之前，PyTorch 将使用 NVML 来检查 CUDA 驱动程序是否正常工作，而不是使用 CUDA 运行时。如果分叉进程因 CUDA 初始化错误而失败，这可能会有所帮助。
  * - ``TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT``
    - cuDNN v8 API 的缓存限制。用于限制 cuDNN v8 API 使用的内存。默认值为 10000，假设每个 ExecutionPlan 为 200KiB，则大约对应 2GiB。设置为 ``0`` 表示无限制，设置为负值表示不缓存。
  * - ``TORCH_CUDNN_V8_API_DISABLED``
    - 如果设置为 ``1``，则禁用 cuDNN v8 API。并将回退到 cuDNN v7 API。
  * - ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE``
    - 如果设置为 ``1``，则强制启用 TF32，覆盖 ``set_float32_matmul_precision`` 设置。
  * - ``TORCH_NCCL_USE_COMM_NONBLOCKING``
    - 如果设置为 ``1``，则在 NCCL 中启用非阻塞错误处理。
  * - ``TORCH_NCCL_AVOID_RECORD_STREAMS``
    - 如果设置为 ``0``，则在 NCCL 中启用回退到基于记录流的同步行为。
  * - ``TORCH_CUDNN_V8_API_DEBUG``
    - 如果设置为 ``1``，则健全性检查是否正在使用 cuDNN V8。

**CUDA 运行时和库环境变量**

.. list-table::
  :header-rows: 1

  * - 变量
    - 描述
  * - ``CUDA_VISIBLE_DEVICES``
    - 应提供给 CUDA 运行时的 GPU 设备 ID 的逗号分隔列表。如果设置为 ``-1``，则不提供任何 GPU。
  * - ``CUDA_LAUNCH_BLOCKING``
    - 如果设置为 ``1``，则使 CUDA 调用同步。这对于调试很有用。
  * - ``CUBLAS_WORKSPACE_CONFIG``
    - 此环境变量用于为每次分配设置 cuBLAS 的工作空间配置。格式为 ``:[SIZE]:[COUNT]``。
      例如，每次分配的默认工作空间大小为 ``CUBLAS_WORKSPACE_CONFIG=:4096:2:16:8``，它指定总大小为 ``2 * 4096 + 8 * 16 KiB``。
      要强制 cuBLAS 避免使用工作空间，请设置 ``CUBLAS_WORKSPACE_CONFIG=:0:0``。
  * - ``CUDNN_CONV_WSCAP_DBG``
    - 类似于 ``CUBLAS_WORKSPACE_CONFIG``，此环境变量用于为每次分配设置 cuDNN 的工作空间配置。
  * - ``CUBLASLT_WORKSPACE_SIZE``
    - 类似于 ``CUBLAS_WORKSPACE_CONFIG``，此环境变量用于设置 cuBLASLT 的工作空间大小。
  * - ``CUDNN_ERRATA_JSON_FILE``
    - 可以设置为勘误过滤器的文件路径，该过滤器可以传递给 cuDNN 以避免特定的引擎配置，主要用于调试或硬编码自动调优。
  * - ``NVIDIA_TF32_OVERRIDE``
    - 如果设置为 ``0``，则全局禁用所有内核中的 TF32，覆盖所有 PyTorch 设置。