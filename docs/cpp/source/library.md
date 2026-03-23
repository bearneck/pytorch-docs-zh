# Torch Library API

PyTorch C++ API 提供了扩展 PyTorch 核心算子库的能力，允许用户定义自定义算子和数据类型。使用 Torch Library API 实现的扩展可以在 PyTorch 即时执行（eager）API 以及 TorchScript 中使用。

如需了解库 API 的教程式介绍，请参阅 [使用自定义 C++ 算子扩展 TorchScript](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html) 教程。

## 宏


TORCH_LIBRARY


TORCH_LIBRARY_IMPL

## 类


torch::Library


torch::CppFunction

## 函数


torch-dispatch-overloads


torch-schema-overloads
