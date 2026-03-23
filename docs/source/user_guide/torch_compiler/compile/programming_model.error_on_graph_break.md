---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

```{code-cell}

import torch

import header_code
torch._logging.set_logs(graph_breaks=True)
```

# 切换 `error_on_graph_break`

**摘要：**

- 当 `fullgraph=False` 时，我们可以使用 `torch._dynamo.error_on_graph_break()` 来更灵活地处理图中断。

到目前为止，我们已经介绍了在 `torch.compile` 中处理图中断的两种方式：
1. `fullgraph=True` 在第一个图中断处报错，并额外保证从代码中只追踪一个图。
2. `fullgraph=False` 在遇到图中断时继续追踪。

如果我们希望禁止大部分代码出现图中断，但存在少数函数中的图中断难以消除，并且我们能够接受这些图中断，该怎么办？我们可以使用 `torch._dynamo.error_on_graph_break()` 来实现这一点。

`torch.compile` 有一个 `error_on_graph_break` 设置（初始值为 `False`）。
如果在 `error_on_graph_break` 设置为 `False` 时，代码中发生图中断或编译器错误，那么 `torch.compile` 将尝试在图中断/错误之后继续编译。
如果 `error_on_graph_break` 设置为 `True`，那么 `torch.compile` 将中止编译并将错误传播给用户代码。

`error_on_graph_break=True` 和 `fullgraph=True` 之间的一个显著区别是，前者**不保证只捕获一个图**。
`error_on_graph_break` **可以在编译时通过使用 `torch._dynamo.error_on_graph_break()` 上下文管理器/装饰器任意切换**。
相比之下，一旦 `fullgraph` 设置为 `True`，就不能再设置为 `False`。
最后，`error_on_graph_break` 的优先级低于 `fullgraph`——`error_on_graph_break` 仅在 `fullgraph=False` 时生效。

## `error_on_graph_break(False)` 示例

```{code-cell}
@torch._dynamo.error_on_graph_break(False)
def code_with_a_difficult_graph_break(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

def inner(x):
    return code_with_a_difficult_graph_break(x)

# 注意：fullgraph=False
@torch._dynamo.error_on_graph_break(True)
@torch.compile
def fn(x):
    return inner(x)

# 没有错误，但存在图中断
fn(torch.randn(3))
```

在 `error_on_graph_break(True)` 下使用 `error_on_graph_break(False)` 适用于我们希望最小化图中断（即遵循 `fullgraph=True` 编程模型）的情况，但存在一些代码段包含难以规避的非性能关键图中断。

`error_on_graph_break()` 也可以用作上下文管理器：

```{code-cell}
# 注意：fullgraph=False
@torch._dynamo.error_on_graph_break(True)
@torch.compile
def fn(x):
    x = x + 1
    with torch._dynamo.error_on_graph_break(False):
        torch._dynamo.graph_break()  # 没有错误
    return x + 2

# 没有错误，但存在图中断
fn(torch.randn(3))
```

您可以使用猴子补丁来切换无法编辑源代码的代码（例如框架代码）的 `error_on_graph_break`：

```{code-cell}
class ThirdPartyModule(torch.nn.Module):
    def forward(self, x):
        x = x + 1
        torch._dynamo.graph_break()
        return x + 2

tp_mod = ThirdPartyModule()
tp_mod.forward = torch._dynamo.error_on_graph_break(False)(tp_mod.forward)

@torch._dynamo.error_on_graph_break(True)
@torch.compile
def fn(x):
    return tp_mod.forward(x)

# 没有错误，但存在图中断
fn(torch.randn(3))
```

## `error_on_graph_break(True)` 示例

```{code-cell}
@torch._dynamo.error_on_graph_break(True)
def inner2(x):
    x = x + 1
    torch._dynamo.graph_break()  # 错误
    return x + 2

def inner(x):
    return inner2(x)

# fullgraph=False, error_on_graph_break=False
@torch.compile
def fn(x):
    x = x + 4
    torch._dynamo.graph_break()  # 没有错误
    return inner(x)

try:
    fn(torch.randn(3))
except Exception as e:
    print(e)
```

在 `error_on_graph_break(False)` 下使用 `error_on_graph_break(True)` 适用于我们希望灵活使用 `torch.compile`（即遵循 `fullgraph=False` 编程模型）的情况，但存在一些性能关键的代码段，我们希望确保这些代码段不包含图中断。

## `error_on_graph_break` 嵌套行为

`torch._dynamo.error_on_graph_break()` 也会影响嵌套调用的 `error_on_graph_break` 设置：

```{code-cell}
def inner(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

def inner2(x):
    with torch._dynamo.error_on_graph_break(False):
        return inner(x)

@torch._dynamo.error_on_graph_break(True)
@torch.compile
def fn(x):
    return inner2(x)

# 没有错误
fn(torch.randn(3))
```

`torch._dynamo.error_on_graph_break()` 可以在另一个 `torch._dynamo.error_on_graph_break()` 区域内使用：

```{code-cell}
def inner(x):
    x = x + 1
    with torch._dynamo.error_on_graph_break(False):
        torch._dynamo.graph_break()
    return x + 2

def inner2(x):
    with torch._dynamo.error_on_graph_break(True):
        return inner(x)

@torch.compile
def fn(x):
    return inner2(x)

# 没有错误
fn(torch.randn(3))
```

## 与 `fullgraph` 的交互

`fullgraph=True` 的优先级高于 `error_on_graph_break`：

```{code-cell}
@torch._dynamo.error_on_graph_break(False)
def inner(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

@torch.compile(fullgraph=True)
def fn(x):
    return inner(x)

try:
    fn(torch.randn(3))
except Exception as e:
    print(e)
```

`fullgraph=True` 不能切换回 `fullgraph=False`：

```{code-cell}
@torch.compile(fullgraph=False)
def inner(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

@torch.compile(fullgraph=True)
def fn(x):
    return inner(x)

try:
    fn(torch.randn(3))
except Exception as e:
    print(e)
{code-cell}
@torch.compile(fullgraph=True)
def inner(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2
python
@torch.compile(fullgraph=False)
def fn(x):
    return inner(x)

try:
    fn(torch.randn(3))
except Exception as e:
    print(e)
```

## `fullgraph=True/False` 与 `error_on_graph_break` 对比总结

下表总结了 `fullgraph=True/False` 和 `error_on_graph_break` 之间的区别：

|  | `error_on_graph_break=True` | `error_on_graph_break=False` (默认值) |
| --- | --- | --- |
| `fullgraph=True` | 图中断会导致错误。仅报告第一个图中断。**保证单一图。**<br><br>`fullgraph` 无法切换为 `False`。`error_on_graph_break` 无效。<br><br>用户代码必须完全兼容 `torch.compile`。保证不会因图中断导致性能下降（因为没有图中断）。<br><br>适用于对图中断敏感的代码：框架/库代码或需要获取最大性能的场景。防止下游用户代码无意中允许图中断。 | 与 `fullgraph=True` 且 `error_on_graph_break=True` 相同，因为当 `fullgraph=True` 时 `error_on_graph_break` 无效。 |
| `fullgraph=False` (默认值) | 图中断会导致错误。仅报告第一个图中断。**不保证单一图。**<br><br>`error_on_graph_break` 可以切换为 `False`。<br><br>用户代码必须完全兼容 `torch.compile`。保证不会因图中断导致性能下降（因为没有图中断）。<br><br>适用于对图中断敏感的用户代码。`error_on_graph_break` 可以切换为 `False` 来处理那些存在难以解决的图中断的代码段。 | 遇到图中断后会继续编译。报告所有图中断。<br><br>`error_on_graph_break` 可以切换为 `True`。<br><br>无需大量修改用户代码即可工作。性能可能因图中断而受到负面影响。<br><br>适用于开箱即用的用例、"非怪异"代码，或不需要榨取最大性能的场景。 |
