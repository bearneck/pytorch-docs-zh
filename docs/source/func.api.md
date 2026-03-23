# torch.func API 参考


## 函数变换


## 用于处理 torch.nn.Module 的实用工具

通常，你可以对一个调用 `torch.nn.Module` 的函数进行变换。
例如，以下是一个计算函数雅可比矩阵的示例，该函数接收三个值并返回三个值：

```python
model = torch.nn.Linear(3, 3)

def f(x):
    return model(x)

x = torch.randn(3)
jacobian = jacrev(f)(x)
assert jacobian.shape == (3, 3)
```

但是，如果你想做类似计算模型参数雅可比矩阵的操作，那么需要一种方法来构造一个以参数为函数输入的函数。这就是 `functional_call` 的用途：它接受一个 nn.Module、变换后的 `parameters` 以及 Module 前向传播的输入。它返回使用替换后的参数运行 Module 前向传播的值。

以下是我们如何计算参数雅可比矩阵的方法：

```python
model = torch.nn.Linear(3, 3)

def f(params, x):
    return torch.func.functional_call(model, params, x)

x = torch.randn(3)
jacobian = jacrev(f)(dict(model.named_parameters()), x)
```


如果你正在寻找关于修复批归一化模块的信息，请遵循此处的指导：


## 调试实用工具

