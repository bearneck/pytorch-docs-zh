orphan

:   

# C++


> 📝 **注意**
> 如果您正在寻找 PyTorch C++ API 文档，请直接访问 [此处](https://pytorch.org/cppdocs/)。
>
> PyTorch 提供了多种用于处理 C++ 的功能，最好根据您的需求从中选择。从高层次来看，以下支持是可用的：
>
> ## C++ 中的张量与自动求导
>
> PyTorch Python API 中的大多数张量和自动求导操作在 C++ API 中也可用。这些包括：
>
> - `torch::Tensor` 方法，例如 `add` / `reshape` / `clone`。有关可用方法的完整列表，请参阅：https://pytorch.org/cppdocs/api/classat_1_1_tensor.html
> - 外观和行为与 Python API 相同的 C++ 张量索引 API。有关其用法的详细信息，请参阅：https://pytorch.org/cppdocs/notes/tensor_indexing.html
> - 对于在 C++ 前端构建动态神经网络至关重要的张量自动求导 API 和 `torch::autograd` 包。更多详细信息，请参阅：https://pytorch.org/tutorials/advanced/cpp_autograd.html
>
> ## 使用 C++ 编写模型
>
> 我们提供了完全使用 C++ 编写和训练神经网络模型的完整能力，其中包含熟悉的组件，例如 `torch::nn` / `torch::nn::functional` / `torch::optim`，它们与 Python API 非常相似。
>
> - 有关 PyTorch C++ 模型编写和训练 API 的概述，请参阅：https://pytorch.org/cppdocs/frontend.html
> - 有关如何使用该 API 的详细教程，请参阅：https://pytorch.org/tutorials/advanced/cpp_frontend.html
> - 诸如 `torch::nn` / `torch::nn::functional` / `torch::optim` 等组件的文档可以在以下位置找到：https://pytorch.org/cppdocs/api/library_root.html
>
> ## C++ 的打包
>
> 有关如何安装 libtorch（包含上述所有 C++ API 的库）并与之链接的指南，请参阅：https://pytorch.org/cppdocs/installing.html。请注意，在 Linux 上提供了两种类型的 libtorch 二进制文件：一种使用 GCC pre-cxx11 ABI 编译，另一种使用 GCC cxx11 ABI 编译，您应根据系统使用的 GCC ABI 进行选择。

