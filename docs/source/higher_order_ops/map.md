---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---


# 控制流 - Map

`torch.map` 是一个结构化的控制流运算符，它在输入张量的首维度上应用一个函数。从逻辑上可以将其实现看作如下：

```python
def map(
    f: Callable[[PyTree, ...], PyTree],
    xs: Union[PyTree, torch.Tensor],
    *args,
):
    out = []
    for idx in range(xs.size(0)):
        xs_sliced = xs.select(0, idx)
        out.append(f(xs_sliced, *args))
    return torch.stack(out)
```

```{warning}
`torch._higher_order_ops.map` 是 PyTorch 中的一个原型功能。您可能会遇到编译错误。
更多关于功能分类的信息请阅读：https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## 示例

以下是一个使用 map 在批次上应用函数的示例：

```{code-cell}
import torch
from torch._higher_order_ops import map

def f(x):
    return x.sin() + x.cos()

xs = torch.randn(3, 4, 5)  # 包含 3 个张量的批次，每个张量大小为 4x5
# 将 f 应用于 3 个切片中的每一个
result = map(f, xs)  # 返回形状为 [3, 4, 5] 的张量
print(result)
```

我们可以导出包含 map 的模型以进行进一步的转换和部署。
此示例使用动态形状以允许可变的批次大小：

```{code-cell}
class MapModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        def body_fn(x):
            return x.sin() + x.cos()

        return map(body_fn, xs)

mod = MapModule()
inp = torch.randn(3, 4)
ep = torch.export.export(mod, (inp,), dynamic_shapes={"xs": {0: torch.export.Dim.DYNAMIC}})
print(ep)
```

请注意，`torch.map` 被降级为 `torch.ops.higher_order.map_impl`，并且主体函数成为顶级图模块的子图属性。

## 限制

- 被映射的 `xs` 只能由张量组成。

- `xs` 中所有张量的首维度必须一致且非零。

- 主体函数不得修改输入。

## API 参考

