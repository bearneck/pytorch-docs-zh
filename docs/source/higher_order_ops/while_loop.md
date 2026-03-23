---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---


# 控制流 - While 循环

`torch.while_loop` 是一个结构化的控制流运算符，它会在条件为真时循环执行一个主体函数。
从逻辑上可以将其视为按以下方式实现：

```python
def while_loop(
    cond_fn: Callable[..., bool],
    body_fn: Callable[..., tuple],
    carried_inputs: tuple,
):
    val = carried_inputs
    while cond_fn(*val):
        val = body_fn(*val)
    return val
```

```{warning}
`torch.while_loop` 是 PyTorch 中的一个原型功能。它对输入和输出类型的支持有限。
请期待未来 PyTorch 版本中更稳定的实现。
更多关于功能分类的信息，请参阅：https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## 示例

下面是一个基本示例，使用 while_loop 进行迭代直到满足条件：

```{code-cell}
import torch
from torch._higher_order_ops import while_loop

class M(torch.nn.Module):

    def cond_fn(self, iter_count, x):
        return iter_count.sum() > 0

    def body_fn(self, iter_count, x):
        return iter_count - 1, x * 2

    def forward(self, init_iter, init_x):
        final_iter, final_x = while_loop(self.cond_fn, self.body_fn, (init_iter, init_x))
        return final_iter, final_x

m = M()
```

我们可以即时运行模型，并期望结果根据输入形状而变化：

```{code-cell}
_, final_x = m(torch.tensor([3]), torch.ones(3))
assert torch.equal(final_x, torch.ones(3) * 2**3)

_, final_x = m(torch.tensor([10]), torch.ones(3))
assert torch.equal(final_x, torch.ones(3) * 2**10)
```

我们可以导出模型以进行进一步的转换和部署。这为我们提供了一个保留 while_loop 结构的导出程序：

```{code-cell}
ep = torch.export.export(M(), (torch.tensor([10]), torch.ones(3)))
print(ep)
```

请注意，条件和主体函数都成为顶层图模块的子图属性。

## 限制

- `body_fn` 必须返回与输入具有相同元数据（形状、数据类型）的张量或整数。

- `body_fn` 和 `cond_fn` 不得就地修改 `carried_inputs`。在修改前需要进行克隆。

- `body_fn` 和 `cond_fn` 不得修改在函数外部创建的 Python 变量（例如，列表/字典）。

- `body_fn` 和 `cond_fn` 的输出不能与任何输入产生别名。需要进行克隆。

## API 参考

