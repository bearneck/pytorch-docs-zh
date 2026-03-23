常见问题解答
---

以下列出了用户在使用 C++ API 的各个部分时经常遇到的一些问题。

C++ 扩展
==============

来自 PyTorch/ATen 的未定义符号错误
*****************************************

**问题**：导入扩展时出现 ``ImportError``，提示 PyTorch 或 ATen 中的某些 C++ 符号未定义。例如：:

  >>> import extension
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  ImportError: /home/user/.pyenv/versions/3.7.1/lib/python3.7/site-packages/extension.cpython-37m-x86_64-linux-gnu.so: undefined symbol: _ZN2at19UndefinedTensorImpl10_singletonE

**解决方法**：解决方法是先 ``import torch``，然后再导入你的扩展。这将使你扩展所依赖的 PyTorch 动态（共享）库中的符号可用，从而在导入扩展时能够解析这些符号。

我使用 ``at::`` 中的函数创建了一个张量并出现错误
****************************************************************

**问题**：你使用例如 ``at::ones`` 或 ``at::randn`` 或 ``at::`` 命名空间中的任何其他张量工厂函数创建了一个张量，并遇到了错误。

**解决方法**：将工厂函数调用中的 ``at::`` 替换为 ``torch::``。你永远不应该使用 ``at::`` 命名空间中的工厂函数，因为它们会创建张量。相应的 ``torch::`` 函数会创建变量，而你的代码中应该只处理变量。