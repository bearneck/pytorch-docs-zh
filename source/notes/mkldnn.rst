.. meta::
   :description: torch.backends.mkldnn 使用指南，这是一个运行 MKLDNN 操作的 PyTorch 后端
   :keywords: 优化 PyTorch, MKLDNN

.. _mkldnn_backend:

MKLDNN 后端
---------------------------------------------------

MKLDNN 是一个开源的跨平台性能库，为深度学习应用提供基础构建模块。

.. code:: python

  # 下面的标志控制是否在 PyTorch 中启用 MKLDNN 后端。
  torch.backends.mkldnn.enabled = True

用户可以通过以下方式禁用 MKLDNN 后端：

.. code:: python

  torch.backends.mkldnn.enabled = False

.. _bf16_on_mkldnn:

MKLDNN 后端上的 Bfloat16 (BF16)
---------------------------------------------------

从 PyTorch 2.9 开始，提供了一组 API 来控制 `float32` 运算符的内部计算精度。

.. code:: python

  # 下面的标志控制 mkldnn 矩阵乘法的内部计算精度。默认 ieee 是 float32。
  torch.backends.mkldnn.matmul.fp32_precision = "ieee"

  # 下面的标志控制 mkldnn 卷积的内部计算精度。默认 ieee 是 float32。
  torch.backends.mkldnn.conv.fp32_precision = "ieee"

  # 下面的标志控制 mkldnn rnn 的内部计算精度。默认 ieee 是 float32。
  torch.backends.mkldnn.rnn.fp32_precision = "ieee"

请注意，除了矩阵乘法和卷积本身，内部使用矩阵乘法或卷积的函数和 nn 模块也会受到影响。这些包括 :class:`torch.nn.Linear`、:class:`torch.nn._ConvNd`、:func:`torch.cdist`、:func:`torch.tensordot`、:func:`torch.nn.functional.affine_grid` 和 :func:`torch.nn.functional.grid_sample`、:class:`torch.nn.AdaptiveLogSoftmaxWithLoss`、:class:`torch.nn.GRU` 以及 :class:`torch.nn.LSTM`。

要了解精度和速度的情况，请查看下面的示例代码和基准测试数据（在 SPR 上）：

.. code:: python

  torch.manual_seed(0)
  a_full = torch.randn(10240, 10240, dtype=torch.double)
  b_full = torch.randn(10240, 10240, dtype=torch.double)
  ab_full = a_full @ b_full
  mean = ab_full.abs().mean()  # 80.7451

  a = a_full.float()
  b = b_full.float()

  # 在 BF16 模式下进行矩阵乘法。
  torch.backends.mkldnn.matmul.fp32_precision = 'bf16'
  ab_bf16 = a @ b  # 预期通过 BF16 点积加速获得速度提升
  error = (ab_bf16 - ab_full).abs().max()  # 1.3704
  relative_error = error / mean  # 0.0170
  print(error, relative_error)

  # 在 TF32 模式下进行矩阵乘法。
  torch.backends.mkldnn.matmul.fp32_precision = 'tf32'
  ab_tf32 = a @ b  # 预期通过 TF32 点积加速获得速度提升
  error = (ab_tf32 - ab_full).abs().max()  # 0.0004
  relative_error = error / mean  # 0.00000552
  print(error, relative_error)

  # 在 FP32 模式下进行矩阵乘法。
  torch.backends.mkldnn.matmul.fp32_precision = 'ieee'
  ab_fp32 = a @ b
  error = (ab_fp32 - ab_full).abs().max()  # 0.0003
  relative_error = error / mean  # 0.00000317
  print(error, relative_error)

从上面的例子中，我们可以看到，使用 BF16 时，在 SPR 上的速度大约快 7 倍，并且与双精度相比的相对误差大约大了 2 个数量级。如果需要完整的 FP32 精度，用户可以通过以下方式禁用 BF16：

.. code:: python

  torch.backends.mkldnn.matmul.fp32_precision = 'ieee'
  torch.backends.mkldnn.conv.fp32_precision = 'ieee'
  torch.backends.mkldnn.rnn.fp32_precision = 'ieee'

要在 C++ 中关闭 BF16 标志，可以这样做：

.. code:: C++

  at::globalContext().setFloat32Precision("ieee", "mkldnn", "matmul");
  at::globalContext().setFloat32Precision("ieee", "mkldnn", "conv");
  at::globalContext().setFloat32Precision("ieee", "mkldnn", "rnn");

如果 fp32_precision 设置为 `ieee`，我们可以为特定的运算符或后端覆盖通用设置。

.. code:: python

  torch.backends.fp32_precision = "bf16"
  torch.backends.mkldnn.fp32_precision = "ieee"
  torch.backends.mkldnn.matmul.fp32_precision = "ieee"

对于这种情况，`torch.backends.mkldnn.fp32_precision` 和 `torch.backends.mkldnn.matmul.fp32_precision` 都被覆盖为 bf16。