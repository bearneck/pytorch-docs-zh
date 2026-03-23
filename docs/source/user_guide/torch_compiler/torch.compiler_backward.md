
``torch.compile`` 具有不同的自动求导语义
==================================================

当你在模型的前向传播中对函数应用 ``torch.compile`` 时，
它会自动为编译后的函数生成反向传播。
在编译过程中，它会为反向传播追踪出一个图，
每当自动求导被调用时就会使用该图。我们将 ``torch.compile`` 内部
负责此功能的组件称为 ``AOTDispatcher``
（有时也称为 ``AOTAutograd``）。

因此，``torch.compile`` 在前向传播函数的编译过程中，
将计算的细节固化到追踪出的反向传播图中。
然而，在即时模式的 PyTorch 中，反向计算是动态的：
在前向传播之外，你可以将 ``tensor.backward()`` 或 ``torch.autograd.grad(...)`` 的调用
包装在可能改变其行为的上下文管理器中。

本文档记录了 ``torch.compile`` 的自动求导语义与
即时模式 PyTorch 有何不同，以及如何解决这些问题。

``Autocast`` 行为
---------------------

``torch.compile`` 固化了一个关于反向传播是否会在
环境自动混合精度上下文管理器下运行的假设。默认情况下，
使用 ``torch._functorch.config.backward_pass_autocast``
来控制该假设；错误的假设可能导致静默的错误。

选项包括：
- `"same_as_forward"`（默认）。
  我们假设 ``torch.compile`` 编译区域的反向传播
  将在与该区域运行时所处的相同自动混合精度上下文管理器下运行（如果有的话）。
  如果你的代码类似于以下形式，请使用此选项：
  ```py
  with torch.amp.autocast(...):
      y = torch.compile(region)(x)
      ...
      # 反向传播在与编译区域相同的自动混合精度上下文中运行
      z.backward()
  ```
- `"off"`。我们假设 torch.compile 编译区域的反向传播
  不会在任何自动混合精度上下文管理器下运行。
  如果你的代码类似于以下形式，请使用此选项：
  ```py
  with torch.amp.autocast(...):
      y = torch.compile(region)(x)
      ...
  # 反向传播在无自动混合精度的情况下运行。
  z.backward()
  ```
- 还有第三个选项。如果你将 ``torch._functorch.config.backward_pass_autocast``
  设置为一个 kwargs 列表，我们将假设反向传播在由这些 kwargs 构造的
  自动混合精度上下文下运行。

  例如，如果你的代码类似于以下形式：
  ```py
  y = torch.compile(region)(x)
  ...
  # 反向传播在特殊的上下文管理器下运行
  with torch.amp.autocast(**kwargs):
      z.backward()
  ```
  那么设置 ``torch._functorch.config.backward_pass_autocast = kwargs``。

使用 ``patch`` 将选项应用于特定的 ``torch.compile`` 调用：
```py
with torch.amp.autocast(...):
    with torch._functorch.config.patch(backward_pass_autocast="same_as_forward")
    y = torch.compile(region)(x)
    ...
    # 反向传播在与编译区域相同的自动混合精度上下文中运行
    z.backward()
```