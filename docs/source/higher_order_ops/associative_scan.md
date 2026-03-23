---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---


# 控制流 - 关联扫描

`torch.associative_scan` 是一个结构化的控制流运算符，它使用一个关联组合函数执行包含性扫描。从逻辑上可以看作按以下方式实现：

```python
def associative_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    xs: pytree.PyTree,
    dim: int,
    reverse: bool = False,
) -> pytree.PyTree:
    result = []
    carry = xs.select(dim, 0)
    result.append(carry)
    for i in range(1, xs.size(dim)):
        carry = combine_fn(carry, xs.select(dim, i))
        result.append(carry)
    return torch.stack(result, dim=dim)
```

由于要求 `combine_fn` 必须是关联的，因此可以使用树归约算法并行化计算，而不是顺序执行。这使得对于累积和、累积积或其他关联累积操作能够实现高效的 GPU 实现。

```{warning}
`torch.associative_scan` 是 PyTorch 中的一个原型功能。您可能会遇到编译错误。
更多关于功能分类的信息请参阅：https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## 示例

以下是一个使用 `associative_scan` 计算累积和的示例：

```{code-cell}
import torch
from torch._higher_order_ops.associative_scan import associative_scan

def add(x: torch.Tensor, y: torch.Tensor):
    return x + y

xs = torch.arange(1, 5, dtype=torch.float32)  # [1, 2, 3, 4]
cumsum = associative_scan(add, xs, dim=0, combine_mode="generic")
print(cumsum)
```

这是一个计算累积积的示例：

```{code-cell}
def mul(x: torch.Tensor, y: torch.Tensor):
    return x * y

xs = torch.arange(1, 5, dtype=torch.float32)  # [1, 2, 3, 4]
cumprod = associative_scan(mul, xs, dim=0, combine_mode="generic")
print(cumprod)
```

我们可以导出包含 associative_scan 的模型以进行进一步的转换和部署。
此示例使用动态形状以允许可变序列长度：

```python
class AssociativeScanModule(torch.nn.Module):
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        def combine_fn(x, y):
            return x + y

        return associative_scan(combine_fn, xs, dim=0, combine_mode="pointwise")

mod = AssociativeScanModule()
inp = torch.randn(5, 3, device="cuda")
dim_seq = torch.export.Dim("seq", min=2)
ep = torch.export.export(mod, (inp,), dynamic_shapes={"xs": {0: dim_seq}})
print(ep)
```

```
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, xs: "f32[s83, 3]"):
            # File: /data/users/angelayi/pytorch2/foo.py:25 in forward, code: return associative_scan(combine_fn, xs, dim=0, combine_mode="pointwise")
            movedim: "f32[s83, 3]" = torch.ops.aten.movedim.int(xs, 0, 0);  xs = None

            # File: <eval_with_key>.3:6 in forward, code: select_copy = torch.select_copy(l_leaves_xs_0_, 0, 0);  select_copy = None
            select_copy: "f32[3]" = torch.ops.aten.select_copy.int(movedim, 0, 0);  select_copy = None

            # File: <eval_with_key>.3:8 in forward, code: associative_scan = torch.ops.higher_order.associative_scan(associative_scan_combine_fn_0, [l_leaves_xs_0_], ());  associative_scan_combine_fn_0 = l_leaves_xs_0_ = None
            associative_scan_combine_graph_0 = self.associative_scan_combine_graph_0
            associative_scan = torch.ops.higher_order.associative_scan(associative_scan_combine_graph_0, [movedim], ());  associative_scan_combine_graph_0 = movedim = None
            getitem: "f32[s83, 3]" = associative_scan[0];  associative_scan = None

            # File: /data/users/angelayi/pytorch2/foo.py:25 in forward, code: return associative_scan(combine_fn, xs, dim=0, combine_mode="pointwise")
            movedim_1: "f32[s83, 3]" = torch.ops.aten.movedim.int(getitem, 0, 0);  getitem = None
            return (movedim_1,)

        class associative_scan_combine_graph_0(torch.nn.Module):
            def forward(self, arg0_1: "f32[3]", arg1_1: "f32[3]"):
                # File: <eval_with_key>.4:5 in forward, code: add = child + child_1;  child = child_1 = None
                add: "f32[3]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
                return [add]

Graph signature:
    # inputs
    xs: USER_INPUT

    # outputs
    movedim_1: USER_OUTPUT
```

请注意，`torch.associative_scan` 被降级为 `torch.ops.higher_order.associative_scan`，并且组合函数成为顶层图模块的子图属性。

## 限制

- `combine_fn` 必须是关联的：`combine_fn(combine_fn(a, b), c) == combine_fn(a, combine_fn(b, c))`。

- `combine_fn` 不得就地修改其输入。

- `combine_fn` 不得引用外部作用域中的变量（不支持闭包）。

- `combine_fn` 的输出不能是任何输入的别名。

## API 参考

