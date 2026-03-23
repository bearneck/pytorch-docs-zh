Torch Library API
=================

PyTorch C++ API 提供了扩展 PyTorch 核心算子库的能力，允许用户定义自定义算子和数据类型。使用 Torch Library API 实现的扩展可以在 PyTorch 即时执行（eager）API 以及 TorchScript 中使用。

如需了解库 API 的教程式介绍，请参阅
`使用自定义 C++ 算子扩展 TorchScript
<https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_
教程。

宏
------

.. doxygendefine:: TORCH_LIBRARY

.. doxygendefine:: TORCH_LIBRARY_IMPL

类
-------

.. doxygenclass:: torch::Library
  :members:

.. doxygenclass:: torch::CppFunction
  :members:

函数
---------

.. doxygengroup:: torch-dispatch-overloads
  :content-only:

.. doxygengroup:: torch-schema-overloads
  :content-only: