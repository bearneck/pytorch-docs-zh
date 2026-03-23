# 自定义运算符

**概述：**
- 使用自定义运算符可以让 `torch.compile` 将函数视为不透明。`torch.compile` 永远不会追踪该函数内部，且后端 Inductor 将原样运行该函数。

在以下任一情况下，您可能希望使用自定义运算符：
- 您的代码调用了某些 C/C++/CUDA 代码。Dynamo 是一个 Python 字节码解释器，通常无法处理绑定到 Python 的 C/C++/CUDA 函数调用。
- Dynamo 和非严格追踪模式在追踪某个函数时遇到困难，您希望 `torch.compile` 忽略该函数。

关于如何将 Python 函数包装成 `torch.compile` 能理解的自定义运算符，请参阅 [Python 自定义运算符教程](https://pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial) 获取更多详细信息。

对于更高级的用例，您可能需要使用我们的 C++ 自定义运算符 API；更多信息请参阅[此处](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)。
