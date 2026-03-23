# 加速器集成

自 PyTorch 2.1 以来，社区在简化将新加速器集成到 PyTorch 生态系统的流程方面取得了显著进展。这些改进包括但不限于：对 `PrivateUse1` 调度键的优化、核心子系统扩展机制的引入与增强，以及关键模块（例如 `torch.accelerator`、`memory management`）的设备无关重构。总而言之，这些进展为加速器集成提供了一个**健壮**、**灵活**且**对开发者友好**的路径基础。

```{note}
本指南仍在完善中。更多详情，请参阅[路线图](https://github.com/pytorch/pytorch/issues/158917)。
```

## 为何重要？

此集成路径提供了几个主要优势：

* **速度**：可扩展性内置于所有核心 PyTorch 模块中。开发者可以独立地将新加速器集成到其下游代码库中——无需修改上游代码，也不受限于社区评审的带宽。
* **面向未来**：这是所有未来 PyTorch 功能的默认集成路径，意味着如果遵循此路径，随着新模块和功能的添加，它们将自动支持扩展到新的加速器。
* **自主性**：供应商完全掌控其加速器集成的时间线，从而实现快速迭代周期，并减少对上游协调的依赖。

## 目标受众

本文档面向：

* 正在将加速器集成到 PyTorch 的**加速器开发者**；
* 对关键模块内部工作原理感兴趣的**高级 PyTorch 用户**；

## 关于本文档

本指南旨在为 PyTorch 中新加速器的集成提供一个**现代集成路径的全面概述**。它涵盖了从底层设备原语到更高级别领域模块（如编译和量化）的完整集成面。结构遵循**模块化和场景驱动的方法**，每个主题都配有来自官方参考实现 [torch_openreg][OpenReg URL] 的相应代码示例，并且本系列围绕四个主要轴心构建：

* **运行时**：涵盖核心组件，如 Event、Stream、Memory、Generator、Guard、Hooks，以及支持的 C++ 脚手架。
* **运算符**：涉及 C++ 和 Python 实现中必需的最小运算符集、前向和后向运算符、回退运算符、直通、STUB 等。
* **Python 前端**：专注于模块的 Python 绑定和设备无关的 API。
* **高级模块**：探索与主要子系统（如 `AMP`、`Compiler`、`ONNX` 和 `Distributed` 等）的集成。

目标是帮助开发者：

* 理解加速器集成的完整范围；
* 遵循最佳实践以快速启动新加速器；
* 通过清晰、有针对性的示例避免常见陷阱。

接下来，我们将深入探讨本指南的每一章。每章聚焦于集成的某个关键方面，提供详细的解释和示例说明。由于某些章节建立在之前章节的基础上，建议读者按顺序阅读以获得更连贯的理解。

```{toctree}
:glob:
:maxdepth: 1

device
hooks
guard
autoload
operators
amp
profiler
```

[OpenReg URL]: https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg "OpenReg URL"