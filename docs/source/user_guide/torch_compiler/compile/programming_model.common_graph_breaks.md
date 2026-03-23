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
:tags: [remove-cell]
import torch

import header_code

torch._logging.set_logs(graph_breaks=True)
```

# 常见的图中断

以下是一些常见的图中断及其解决方法。

## 代码错误
你的代码可能包含错误（意味着即使没有 `torch.compile` 也无法执行）。在下面的示例中，由于多了一个参数，`torch.sin` 调用中存在拼写错误。**请始终禁用 `torch.compile` 以检查代码是否能正确运行。**

```{code-cell}
@torch.compile
def fn(x):
    y = torch.sin(x, x)
    return y

try:
    fn(torch.ones(3, 3))
except Exception as e:
    pass
```

Dynamo 会尽力提示图中断是否由你的代码引起。
但有时仍然很难从日志中判断图中断是由代码错误、更复杂的图中断还是 `torch.compile` 的 bug 引起的。为了区分，我们建议尝试在不使用 `torch.compile` 的情况下运行你的代码，看看是否仍然会出现图中断报告的错误。

你也可以使用 `torch.compiler.set_stance("force_eager")` 来快速禁用 `torch.compile`，而无需修改 `torch.compile` 调用：

```{code-cell}
@torch.compile
def fn(x):
    y = torch.sin(x, x)
    return y

try:
    with torch.compiler.set_stance("force_eager"):
        fn(torch.ones(3, 3))
except Exception as e:
    print(e)
```

有关使用 `set_stance` 进行调试的更多示例，请参阅 https://docs.pytorch.org/tutorials/recipes/torch_compiler_set_stance_tutorial.html#crashing-sooner。

## 数据依赖操作

`torch.compile` 会在数据依赖操作上发生图中断，例如数据依赖的控制流（if 语句、涉及张量的循环）和直接访问张量数据（`.item`、`.data_ptr`）。

```{code-cell}
@torch.compile
def fn(x):
    y = x.sum()
    if y > 0:
        return x + y.item()
    return x - y.item()

print(fn(torch.ones(3, 3)))
```

这些图中断的一般解决方法是避免执行数据依赖操作。一些具体的解决方法包括：

- 如果你的控制流实际上不依赖于数据值，请考虑修改代码，使其基于常量执行控制流。

```{code-cell}
# 旧代码
x = torch.randn(3, 3)
@torch.compile
def fn(y):
    if x.sum() > 0:
        return y + x
    else:
        return y - x

print(fn(torch.ones(3, 3)))
```

```{code-cell}
# 新代码
x = torch.randn(3, 3)
cond = (x.sum() > 0).item()
@torch.compile
def fn(y):
    if cond:
        return y + x
    else:
        return y - x

print(fn(torch.ones(3, 3)))
```

- 使用 {ref}`cond` 等高级操作符替代数据依赖的控制流。

```{code-cell}
# 旧代码
@torch.compile
def fn(x):
    if x.sum() > 0:
        return x + 1
    return x - 1

print(fn(torch.ones(3, 3)))
```

```{code-cell}
# 新代码
@torch.compile
def fn(x):
    return torch.cond(
        x.sum() > 0,
        lambda x: x + 1,
        lambda x: x - 1,
        (x,),
    )

print(fn(torch.ones(3, 3)))
```

- 如果你有 `.item()` 调用，可以尝试设置 `torch._dynamo.config.capture_scalar_outputs = True` 或环境变量 `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`。
- 将函数中有问题的部分包装在自定义操作符中。

## 打印和日志记录

打印/日志记录/发出警告会导致图中断。
你可以尝试使用 `torch._dynamo.config.reorderable_logging_functions` 来解决这个问题。
此配置用于重新排序日志记录函数，使其在跟踪函数的末尾被调用，从而避免图中断。
但是，如果发生突变等情况，记录的内容可能会有所不同。

注意：`reorderable_logging_functions` 有限制，这些函数必须返回 `None`，并且它们的参数必须仅限于张量、常量或格式字符串。

如果你不需要运行打印或日志记录函数，那么可以考虑使用 `torch.compiler.is_compiling()` 或 `torch._dyanmo.config.ignore_logging_functions` 来完全跳过该函数。更多详细信息请参阅[此页面](programming_model.fullgraph_true.skipping_functions)。