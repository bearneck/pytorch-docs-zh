

# 自动微分包 - torch.autograd


## 前向模式自动微分


> ⚠️ **警告**
> 此 API 处于测试阶段。尽管函数签名不太可能更改，但在我们将其视为稳定版本之前，计划改进算子覆盖范围。


有关如何使用此 API 的详细步骤，请参阅[前向模式 AD 教程](https://pytorch.org/tutorials/intermediate/forward_ad_usage.html)。


## 函数式高阶 API


> ⚠️ **警告**
> 此 API 处于测试阶段。尽管函数签名不太可能更改，但在我们将其视为稳定版本之前，计划对性能进行重大改进。


本节包含基于上述基础 API 构建的 autograd 高阶 API，允许您计算雅可比矩阵、海森矩阵等。

此 API 适用于仅接受张量作为输入并仅返回张量的用户提供函数。
如果您的函数接受其他非张量参数或未设置 requires_grad 的张量，可以使用 lambda 表达式捕获它们。
例如，对于接受三个输入的函数 `f`：一个我们想要计算雅可比矩阵的张量、另一个应视为常量的张量以及一个布尔标志，即 `f(input, constant, flag=flag)`，
您可以这样使用：`functional.jacobian(lambda x: f(x, constant, flag=flag), input)`。


## 局部禁用梯度计算

有关无梯度模式与推理模式之间差异以及其他可能与这两者混淆的相关机制的更多信息，请参阅 `locally-disable-grad-doc`。
另请参阅 `torch-rst-local-disable-grad` 以获取可用于局部禁用梯度的函数列表。


## 默认梯度布局

当非稀疏的 `param` 在 `torch.autograd.backward` 或 `torch.Tensor.backward` 期间接收到非稀疏梯度时，
`param.grad` 按以下方式累积：

如果 `param.grad` 初始为 `None`：

1. 如果 `param` 的内存是非重叠且密集的，`.grad` 将创建与 `param` 匹配的步幅（从而匹配 `param` 的布局）。
2. 否则，`.grad` 将创建行主序连续步幅。

如果 `param` 已具有非稀疏的 `.grad` 属性：

3. 如果 `create_graph=False`，`backward()` 会就地累加到 `.grad` 中，这会保留其步幅。
4. 如果 `create_graph=True`，`backward()` 会将 `.grad` 替换为新张量 `.grad + new grad`，这会尝试（但不保证）匹配现有 `.grad` 的步幅。

建议使用默认行为（在第一次 `backward()` 之前让 `.grad` 保持为 `None`，以便根据 1 或 2 创建其布局，并根据 3 或 4 随时间保留）以获得最佳性能。
调用 `model.zero_grad()` 或 `optimizer.zero_grad()` 不会影响 `.grad` 布局。

实际上，在每次累积阶段之前将所有 `.grad` 重置为 `None`，例如：

```
for iterations...
    ...
    for param in model.parameters():
        param.grad = None
    loss.backward()
```

这样每次都会根据 1 或 2 重新创建它们，是 `model.zero_grad()` 或 `optimizer.zero_grad()` 的有效替代方案，可能会提高某些网络的性能。

### 手动梯度布局

如果您需要手动控制 `.grad` 的步幅，请在第一次 `backward()` 之前将 `param.grad =` 分配为具有所需步幅的零张量，并且永远不要将其重置为 `None`。
只要 `create_graph=False`，3 保证您的布局得以保留。
4 表明即使 `create_graph=True`，您的布局也*可能*得以保留。

## 张量的原地操作

在 autograd 中支持原地操作是一个难题，我们不建议在大多数情况下使用它们。Autograd 积极的缓冲区释放和重用使其非常高效，实际上原地操作显著降低内存使用的情况非常少。除非您在严重的内存压力下操作，否则可能永远不需要使用它们。

### 原地正确性检查

所有 `Tensor` 都会跟踪应用于它们的原地操作，如果实现检测到某个张量在其中一个函数中保存用于反向传播，但之后被原地修改，则在开始反向传播时会引发错误。这确保如果您使用原地函数且未看到任何错误，则可以确信计算的梯度是正确的。

## Variable（已弃用）


> ⚠️ **警告**
> Variable API 已被弃用：使用张量进行自动微分不再需要 Variables。Autograd 自动支持 `requires_grad` 设置为 `True` 的张量。以下是变更的快速指南：
>
> - `Variable(tensor)` 和 `Variable(tensor, requires_grad)` 仍按预期工作，但它们返回的是张量而不是 Variables。
> - `var.data` 与 `tensor.data` 相同。
> - 诸如 `var.backward(), var.detach(), var.register_hook()` 等方法现在可以在具有相同方法名的张量上使用。
>
> 此外，现在可以使用工厂方法创建具有 `requires_grad=True` 的张量，例如 `torch.randn`、`torch.zeros`、`torch.ones` 等，如下所示：
>
> `autograd_tensor = torch.randn((2, 3, 4), requires_grad=True)`


## 张量自动微分函数


## {hidden}`Function`


## 上下文方法混入类

当创建一个新的 `Function` 时，以下方法可用于 `ctx`。


## 自定义 Function 工具

用于 backward 方法的装饰器。


用于构建 PyTorch 工具的基础自定义 `Function`


## 数值梯度检查


% 仅为重置此文件其余部分的基础路径


## 性能分析器

Autograd 包含一个性能分析器，允许你检查模型中不同操作符的成本——无论是在 CPU 还是 GPU 上。目前实现了三种模式——仅使用 CPU 的 `~torch.autograd.profiler.profile`、基于 nvprof（同时记录 CPU 和 GPU 活动）的 `~torch.autograd.profiler.emit_nvtx`，以及基于 vtune 性能分析器的 `~torch.autograd.profiler.emit_itt`。


## 调试与异常检测


## Autograd 计算图

Autograd 提供了一些方法，允许用户检查计算图并在反向传播过程中介入行为。

如果张量是由 autograd 记录的操作的输出（即启用了 grad_mode 并且至少有一个输入需要梯度），那么 `torch.Tensor` 的 `grad_fn` 属性会持有一个 `torch.autograd.graph.Node`，否则为 `None`。


一些操作需要在正向传播期间保存中间结果，以便执行反向传播。
这些中间结果作为属性保存在 `grad_fn` 上，并且可以被访问。
例如：

```
>>> a = torch.tensor([0., 0., 0.], requires_grad=True)
>>> b = a.exp()
>>> print(isinstance(b.grad_fn, torch.autograd.graph.Node))
True
>>> print(dir(b.grad_fn))
['__call__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_raw_saved_result', '_register_hook_dict', '_saved_result', 'metadata', 'name', 'next_functions', 'register_hook', 'register_prehook', 'requires_grad']
>>> print(torch.allclose(b.grad_fn._saved_result, b))
True
```

你也可以使用钩子定义这些保存的张量应该如何打包/解包。
一个常见的应用是通过将这些中间结果保存到磁盘或 CPU 而不是留在 GPU 上，来用计算换取内存。如果你注意到你的模型在评估时适合 GPU，但在训练时不适合，这尤其有用。
另请参阅 `saved-tensors-hooks-doc`。


% 此模块需要文档记录。暂时添加在此处以供追踪


