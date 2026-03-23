---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---


# 控制流 - Cond

`torch.cond` 是一个结构化的控制流运算符。它可用于指定类似 if-else 的控制流，其逻辑实现可以看作如下所示。

```python
def cond(
    pred: Union[bool, torch.Tensor],
    true_fn: Callable,
    false_fn: Callable,
    operands: Tuple[torch.Tensor]
):
    if pred:
        return true_fn(*operands)
    else:
        return false_fn(*operands)
```

其独特之处在于它能够表达**数据相关的控制流**：它会被降级为一个条件运算符（`torch.ops.higher_order.cond`），该运算符会保留谓词、真值函数和假值函数。这为编写和部署那些根据张量操作的输入或中间输出的**值**或**形状**来改变模型架构的模型提供了极大的灵活性。

```{warning}
`torch.cond` 是 PyTorch 中的一个原型功能。它对输入和输出类型的支持有限。
请期待未来 PyTorch 版本中更稳定的实现。
更多关于功能分类的信息请参阅：https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## 示例

下面是一个使用 cond 根据输入形状进行分支的示例：

```{code-cell}
import torch

def true_fn(x: torch.Tensor):
    return x.cos()

def false_fn(x: torch.Tensor):
    return x.sin()

class DynamicShapeCondPredicate(torch.nn.Module):
    """
    基于动态形状谓词使用 cond 的基本示例。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cond(x.shape[0] > 4, true_fn, false_fn, (x,))

dyn_shape_mod = DynamicShapeCondPredicate()
```

我们可以即时运行模型，并期望结果根据输入形状而变化：

```{code-cell}
inp = torch.randn(3)
inp2 = torch.randn(5)
print(dyn_shape_mod(inp), false_fn(inp))
print(dyn_shape_mod(inp2), true_fn(inp2))
```

我们可以导出模型以进行进一步的转换和部署。这将得到一个如下所示的导出程序：

```{code-cell}
inp = torch.randn(4, 3)
ep = torch.export.export(
    DynamicShapeCondPredicate(),
    (inp,),
    dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}}
)
print(ep)
```

请注意，`torch.cond` 被降级为 `torch.ops.higher_order.cond`，其谓词变为关于输入形状的符号表达式，分支函数则成为顶层图模块的两个子图属性。

下面是另一个展示如何表达数据相关控制流的示例：

```{code-cell}
def true_fn(x: torch.Tensor):
    return x.cos() + x.sin()

def false_fn(x: torch.Tensor):
    return x.sin()

class DataDependentCondPredicate(torch.nn.Module):
    """
    基于数据相关谓词使用 cond 的基本示例。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cond(x.sum() > 4.0, true_fn, false_fn, (x,))

inp = torch.randn(4, 3)
ep = torch.export.export(DataDependentCondPredicate(), (inp,), dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}})
print(ep)
```

## torch.ops.higher_order.cond 的不变性

`torch.ops.higher_order.cond` 有几个有用的不变性：

- 对于谓词：
    - 谓词的动态性会被保留（例如，上例中显示的 `gt`）。
    - 如果用户程序中的谓词是常量（例如，一个 Python 布尔常量），则该运算符的 `pred` 将是一个常量。

- 对于分支：
    - 输入和输出签名将是一个扁平化的元组。
    - 它们是 `torch.fx.GraphModule`。
    - 原始函数中的闭包变为显式输入。没有闭包。
    - 不允许对输入或全局变量进行修改。

- 对于操作数：
    - 它也将是一个扁平元组。

- 用户程序中 `torch.cond` 的嵌套会变成嵌套的图模块。

## API 参考
