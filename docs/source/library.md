(torch-library-docs)=

# torch.library

```{eval-rst}
.. py:module:: torch.library
.. currentmodule:: torch.library
```

torch.library 是一组用于扩展 PyTorch 核心算子库的 API。它包含用于测试自定义算子、创建新的自定义算子以及扩展使用 PyTorch C++ 算子注册 API（例如 aten 算子）定义的算子的实用工具。

关于如何有效使用这些 API 的详细指南，请参阅 [PyTorch 自定义算子介绍页面](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html) 以获取更多详细信息。

## 测试自定义算子

使用 {func}`torch.library.opcheck` 来测试自定义算子是否存在对 Python torch.library 和/或 C++ TORCH_LIBRARY API 的错误使用。此外，如果你的算子支持训练，请使用 {func}`torch.autograd.gradcheck` 来测试梯度在数学上是否正确。

```{eval-rst}
.. autofunction:: opcheck
```

## 在 Python 中创建新的自定义算子

使用 {func}`torch.library.custom_op` 来创建新的自定义算子。

```{eval-rst}
.. autofunction:: custom_op
.. autofunction:: triton_op
.. autofunction:: wrap_triton
```

## 扩展自定义算子（从 Python 或 C++ 创建）

使用 `register.*` 方法，例如 {func}`torch.library.register_kernel` 和 {func}`torch.library.register_fake`，来为任何算子添加实现（这些算子可能是使用 {func}`torch.library.custom_op` 或通过 PyTorch 的 C++ 算子注册 API 创建的）。

```{eval-rst}
.. autofunction:: register_kernel
.. autofunction:: register_autocast
.. autofunction:: register_autograd
.. autofunction:: register_fake
.. autofunction:: register_vmap
.. autofunction:: impl_abstract
.. autofunction:: get_ctx
.. autofunction:: register_torch_dispatch
.. autofunction:: infer_schema
.. autoclass:: torch._library.custom_ops.CustomOpDef
   :members: set_kernel_enabled
.. autofunction:: get_kernel
```

## 底层 API

以下 API 是 PyTorch C++ 底层算子注册 API 的直接绑定。

```{eval-rst}
.. warning:: 底层算子注册 API 和 PyTorch 调度器是一个复杂的 PyTorch 概念。我们建议在可能的情况下使用上述更高级别的 API（不需要 torch.library.Library 对象）。`这篇博客文章 <http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/>`_ 是了解 PyTorch 调度器的一个很好的起点。
```

一个引导你了解如何使用此 API 的示例教程可在 [Google Colab](https://colab.research.google.com/drive/1RRhSfk7So3Cn02itzLWE9K4Fam-8U011?usp=sharing) 上找到。

```{eval-rst}
.. autoclass:: torch.library.Library
  :members:

.. autofunction:: fallthrough_kernel

.. autofunction:: define

.. autofunction:: impl
```