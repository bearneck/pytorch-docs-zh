---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

(scan)=

# 控制流 - Scan

`torch.scan` 是一个结构化的控制流运算符，它使用一个组合函数执行包含性扫描。
它通常用于累积操作，如累加和（cumsum）、累积乘积（cumprod）或更一般的递推关系。
从逻辑上可以将其实现看作如下：

```python
def scan(
    combine_fn: Callable[[PyTree, PyTree], tuple[PyTree, PyTree]],
    init: PyTree,
    xs: PyTree,
    *,
    dim: int = 0,
    reverse: bool = False,
) -> tuple[PyTree, PyTree]:
    carry = init
    ys = []
    for i in range(xs.size(dim)):
        x_slice = xs.select(dim, i)
        carry, y = combine_fn(carry, x_slice)
        ys.append(y)
    return carry, torch.stack(ys)
```

```{warning}
`torch.scan` 是 PyTorch 中的一个原型功能。您可能会遇到编译错误。
更多关于功能分类的信息，请阅读：
https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## 示例

下面是一个使用 scan 计算累积和的示例：

```{code-cell}
import torch
from torch._higher_order_ops import scan

def add(carry: torch.Tensor, x: torch.Tensor):
    next_carry = carry + x
    y = next_carry.clone()  # clone 以避免输出-输出别名
    return next_carry, y

init = torch.zeros(1)
xs = torch.arange(5, dtype=torch.float32)

final_carry, cumsum = scan(add, init=init, xs=xs)
print(final_carry)
print(cumsum)
```

我们可以导出包含 scan 的模型，以进行进一步的转换和部署。
此示例使用动态形状以允许可变序列长度：

```{code-cell}
class ScanModule(torch.nn.Module):
    def forward(self, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        def combine_fn(carry, x):
            next_carry = carry + x
            return next_carry, next_carry.clone()

        init = torch.zeros_like(xs[0])
        return scan(combine_fn, init=init, xs=xs)

mod = ScanModule()
inp = torch.randn(5, 3)
ep = torch.export.export(mod, (inp,), dynamic_shapes={"xs": {0: torch.export.Dim.DYNAMIC}})
print(ep)
```

请注意，组合函数成为顶级图模块的子图属性。

## 限制

- `combine_fn` 必须为 `next_carry` 返回与 `init` 具有相同元数据（形状、数据类型）的张量。

- `combine_fn` 不得就地修改其输入。需要在修改前进行克隆。

- `combine_fn` 不得修改在函数外部创建的 Python 变量（例如，列表/字典）。

- `combine_fn` 的输出不能与任何输入产生别名。需要进行克隆。

## API 参考

```{eval-rst}
.. autofunction:: torch._higher_order_ops.scan.scan
```