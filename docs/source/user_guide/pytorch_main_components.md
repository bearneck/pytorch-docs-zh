# PyTorch 主要组件

PyTorch 是一个灵活且强大的深度学习库，提供了一套全面的工具用于构建、训练和部署机器学习模型。

## 用于基础深度学习的 PyTorch 组件

一些基础的 PyTorch 组件包括：

* **张量 (Tensors)** - 作为 PyTorch 基础数据结构的 N 维数组。它们支持自动微分、硬件加速，并为数学运算提供了全面的 API。

* **自动求导 (Autograd)** - PyTorch 的自动微分引擎，它跟踪在张量上执行的操作并动态构建计算图，以便能够计算梯度。

* **神经网络 API (Neural Network API)** - 一个用于构建神经网络的模块化框架，包含预定义的层、激活函数和损失函数。`nn.Module` 基类为创建具有参数管理的自定义网络架构提供了一个清晰的接口。

* **数据加载器 (DataLoaders)** - 用于高效数据处理的工具，提供批处理、打乱和并行数据加载等功能。它们抽象了数据预处理和迭代的复杂性，从而实现了优化的训练循环。

## PyTorch 编译器

PyTorch 编译器是一套优化模型执行并减少资源需求的工具。您可以在此处了解更多关于 PyTorch 编译器的信息 [here](https://docs.pytorch.org/docs/stable/torch.compiler_get_started.html)。
