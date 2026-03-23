# PyTorch C++ API

这些页面提供了 PyTorch C++ API 公共部分的文档。该 API 大致可分为五个部分：

- **ATen**：基础的张量和数学运算库，所有其他组件都构建于其上。
- **Autograd**：通过自动微分增强 ATen。
- **C++ Frontend**：用于训练和评估机器学习模型的高级构造。
- **TorchScript**：与 TorchScript JIT 编译器和解释器交互的接口。
- **C++ Extensions**：使用自定义 C++ 和 CUDA 例程扩展 Python API 的方法。

这些构建模块共同构成了一个面向研究和生产的 C++ 库，用于张量计算和动态神经网络，特别强调 GPU 加速以及快速的 CPU 性能。它目前已在 Facebook 的研究和生产中使用；我们期待迎来更多的 PyTorch C++ API 用户。

 warning
 title
Warning


目前，C++ API 应被视为处于"测试版"稳定性阶段；我们可能会对后端进行重大的破坏性更改，以改进 API，或者为了提供 PyTorch 的 Python 接口（这是我们最稳定且支持最好的接口）。


## ATen

ATen 本质上是一个张量库，PyTorch 中几乎所有其他 Python 和 C++ 接口都构建于其上。它提供了一个核心的 `Tensor` 类，并定义了数百个操作。这些操作大多都有 CPU 和 GPU 实现，`Tensor` 类会根据其类型动态分派到相应的实现。使用 ATen 的一个小示例如下：

``` cpp
#include <ATen/ATen.h>

at::Tensor a = at::ones({2, 2}, at::kInt);
at::Tensor b = at::randn({2, 2});
auto c = a + b.to(at::kInt);
```

这个 `Tensor` 类以及 ATen 中的所有其他符号都位于 `at::` 命名空间中，其文档见 [此处](https://pytorch.org/cppdocs/api/namespace_at.html#namespace-at)。

## Autograd

我们所说的 *autograd* 是指 PyTorch C++ API 中增强 ATen `Tensor` 类自动微分能力的部分。autograd 系统记录对张量的操作以形成 *autograd 图*。在此图中对叶子变量调用 `backwards()`，会通过跨越 autograd 图的函数和张量网络执行反向模式微分，最终产生梯度。以下示例展示了该接口的用法：

``` cpp
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

torch::Tensor a = torch::ones({2, 2}, torch::requires_grad());
torch::Tensor b = torch::randn({2, 2});
auto c = a + b;
c.backward(); // a.grad() 现在将保存 c 相对于 a 的梯度。
```

ATen 中的 `at::Tensor` 类默认不可微分。要添加 autograd API 提供的张量可微性，你必须使用 [torch::]{.title-ref} 命名空间中的张量工厂函数，而不是 [at::]{.title-ref} 命名空间中的函数。例如，使用 [at::ones]{.title-ref} 创建的张量将不可微分，而使用 [torch::ones]{.title-ref} 创建的张量则是可微分的。

## C++ Frontend

PyTorch C++ 前端为神经网络和通用 ML（机器学习）研究及生产用例提供了一个高级的、纯 C++ 的建模接口，其设计和功能在很大程度上遵循了 Python API。C++ 前端包括以下内容：

- 通过分层模块系统（类似于 `torch.nn.Module`）定义机器学习模型的接口；
- 一个用于最常见建模目的的"标准库"预置模块（例如卷积、RNN、批归一化等）；
- 一个优化 API，包括 SGD、Adam、RMSprop 等流行优化器的实现；
- 表示数据集和数据管道的方法，包括在多个 CPU 核心上并行加载数据的功能；
- 用于存储和加载训练会话检查点的序列化格式（类似于 `torch.utils.data.DataLoader`）；
- 将模型自动并行化到多个 GPU 上的功能（类似于 `torch.nn.parallel.DataParallel`）；
- 支持使用 pybind11 轻松将 C++ 模型绑定到 Python 的代码；
- 访问 TorchScript JIT 编译器的入口点；
- 有助于与 ATen 和 Autograd API 交互的实用工具。

有关 C++ 前端的更详细描述，请参阅 [此文档](https://pytorch.org/cppdocs/frontend.html)。与 C++ 前端相关的 [torch::]{.title-ref} 命名空间的相关部分包括 [torch::nn](https://pytorch.org/cppdocs/api/namespace_torch__nn.html#namespace-torch-nn)、 [torch::optim](https://pytorch.org/cppdocs/api/namespace_torch__optim.html#namespace-torch-optim)、 [torch::data](https://pytorch.org/cppdocs/api/namespace_torch__data.html#namespace-torch-data)、 [torch::serialize](https://pytorch.org/cppdocs/api/namespace_torch__serialize.html#namespace-torch-serialize)、 [torch::jit](https://pytorch.org/cppdocs/api/namespace_torch__jit.html#namespace-torch-jit) 和 [torch::python](https://pytorch.org/cppdocs/api/namespace_torch__python.html#namespace-torch-python)。 C++ 前端的示例可以在 [此代码库](https://github.com/pytorch/examples/tree/master/cpp) 中找到，该代码库正在持续积极地扩展。

 note
 title
Note


除非你有特殊原因必须仅限于使用 ATen 或 Autograd API，否则 C++ 前端是推荐进入 PyTorch C++ 生态系统的入口点。虽然在我们收集用户反馈（来自您！）期间它仍处于测试阶段，但它比 ATen 和 Autograd API 提供了更多的功能和更好的稳定性保证。


## TorchScript

TorchScript 是一种 PyTorch 模型的表示形式，能够被 TorchScript 编译器理解、编译和序列化。从根本上说，TorchScript 本身就是一种编程语言。它是使用 PyTorch API 的 Python 子集。TorchScript 的 C++ 接口主要包含三大功能：

- 一种加载和执行在 Python 中定义的序列化 TorchScript 模型的机制；
- 用于定义自定义运算符以扩展 TorchScript 标准操作库的 API；
- 从 C++ 对 TorchScript 程序进行即时编译。

如果您希望尽可能在 Python 中定义模型，随后将其导出到 C++ 用于生产环境和非 Python 推理，那么第一种机制可能对您非常有吸引力。您可以通过点击 [此链接](https://pytorch.org/tutorials/advanced/cpp_export.html) 了解更多信息。第二种 API 涉及您希望使用自定义运算符扩展 TorchScript 的场景，这些运算符同样可以在推理期间从 C++ 序列化和调用。最后，\`torch::jit::compile \<https://pytorch.org/cppdocs/api/function_namespacetorch_1_1jit_1a8660dc13a6b82336aadac667e6dccba1.html\>\`\_ 函数可用于直接从 C++ 访问 TorchScript 编译器。

## C++ 扩展

*C++ 扩展* 提供了一种简单而强大的方式来访问上述所有接口，以扩展 PyTorch 的常规 Python 用例。C++ 扩展最常用于在 C++ 或 CUDA 中实现自定义运算符，以加速标准 PyTorch 设置中的研究。C++ 扩展 API 并未向 PyTorch C++ API 添加任何新功能。相反，它提供了与 Python setuptools 的集成以及 JIT 编译机制，允许从 Python 访问 ATen、autograd 和其他 C++ API。要了解更多关于 C++ 扩展 API 的信息，请参阅 [此教程](https://pytorch.org/tutorials/advanced/cpp_extension.html)。

## 目录

 {.toctree maxdepth="2"}
installing frontend stable api/library_root


 {.toctree glob="" maxdepth="1" caption="说明"}
notes/\*


# 索引和表格

- `genindex`{.interpreted-text role="ref"}
- `modindex`{.interpreted-text role="ref"}
- `search`{.interpreted-text role="ref"}

## 致谢

PyTorch C++ 生态系统的此文档网站得益于 [Exhale](https://github.com/svenevs/exhale/) 项目以及其维护者 [svenevs](https://github.com/svenevs/) 慷慨投入的时间和精力。我们感谢 Stephen 的工作以及他为 PyTorch C++ 文档提供的帮助。
