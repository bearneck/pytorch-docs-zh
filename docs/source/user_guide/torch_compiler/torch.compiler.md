
# torch.compiler

`torch.compiler` 是一个命名空间，通过它向用户公开了一些内部编译器方法。该命名空间中的主要功能和特性是 `torch.compile`。

`torch.compile` 是 PyTorch 2.x 中引入的一个 PyTorch 函数，旨在解决 PyTorch 中精确捕获计算图的问题，并最终使软件工程师能够更快地运行他们的 PyTorch 程序。`torch.compile` 是用 Python 编写的，它标志着 PyTorch 从 C++ 向 Python 的过渡。

`torch.compile` 利用了以下底层技术：

- **TorchDynamo (torch._dynamo)** 是一个内部 API，它使用一个名为 Frame Evaluation API 的 CPython 功能来安全地捕获 PyTorch 计算图。通过 `torch.compiler` 命名空间向 PyTorch 用户公开了可供外部使用的方法。
- **TorchInductor** 是默认的 `torch.compile` 深度学习编译器，它为多种加速器和后端生成快速代码。您需要使用一个后端编译器才能通过 `torch.compile` 实现加速。对于 NVIDIA、AMD 和 Intel GPU，它利用 OpenAI Triton 作为关键的构建模块。
- **AOT Autograd** 不仅捕获用户级代码，还捕获反向传播，从而实现了“提前”捕获反向传播过程。这使得能够使用 TorchInductor 同时加速前向传播和反向传播。

为了更好地理解 `torch.compile` 在您代码上的追踪行为，或了解更多关于 `torch.compile` 内部原理的信息，请参阅 [`torch.compile` 编程模型](compile/programming_model.md)。


> 📝 **注意**
> 在某些情况下，本文档中可能会交替使用 `torch.compile`、TorchDynamo、`torch.compiler` 这些术语。


> ⚠️ **警告**
> `torch.compile` 可能不支持最近发布的 Python 主要版本。
>
> 如果您尝试在不支持的 Python 环境中使用 `@torch.compile`，可能会遇到类似以下的错误：
>
> ```
> RuntimeError: torch.compile is not supported on Python 3.xx.0+
>
> ```
>
> 请确保您当前的 Python 版本在 PyTorch 为 `torch.compile` 支持的范围内。
>
> 如果您在过新的 Python 版本上安装了 PyTorch，则需要切换到较早的 Python 版本才能使用 `torch.compile`。


如上所述，为了更快地运行您的工作流，通过 TorchDynamo 实现的 `torch.compile` 需要一个后端，将捕获的计算图转换为快速的机器码。不同的后端可以带来不同的优化效果。默认的后端称为 TorchInductor，也称为 *inductor*。TorchDynamo 有一个由我们的合作伙伴开发的受支持后端列表，可以通过运行 `torch.compiler.list_backends()` 来查看，每个后端都有其可选的依赖项。

一些最常用的后端包括：

**训练和推理后端**


**仅推理后端**


```{toctree}
:maxdepth: 1
:hidden:

torch.compiler_get_started.md
```

```{toctree}
:maxdepth: 1
:hidden:

core_concepts
```

```{toctree}
:maxdepth: 1
:hidden:

performance
```

```{toctree}
:maxdepth: 1
:hidden:

advanced
```

```{toctree}
:maxdepth: 1
:hidden:


troubleshooting_faqs
```

```{toctree}
:maxdepth: 1
:hidden:

api_reference
```