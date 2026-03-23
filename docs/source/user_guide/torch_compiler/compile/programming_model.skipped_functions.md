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
import logging
torch._logging.set_logs(dynamo=logging.DEBUG)
```

# 被跳过的函数

**摘要：**
- 有时，`torch.compile` 会完全放弃编译一个函数，转而以即时执行（eager）模式运行它，这可能导致潜在的优化机会丢失。
- 有一些方法可以绕过被跳过的函数，以便重新启用对有问题的代码进行追踪。

有时，当遇到图中断（graph break）或其他编译器错误时，使用 `fullgraph=False` 的 `torch.compile` 无法恢复追踪。在许多此类情况下，`torch.compile` 将完全跳过编译该函数，并以即时执行模式运行它。

请注意，跳过操作仅应用于当前函数，而**不**应用于任何嵌套的函数调用。`torch.compile` 仍将尝试编译嵌套调用。

<!-- TODO: 修复被跳过函数的日志记录。 -->

```{code-cell}
def inner1(x):
    return x + 1
def inner2(x):
    return x + 2
@torch.compile
def fn(x):
    x = inner1(x)
    torch._dynamo.skip_frame()
    x = inner2(x)
fn(torch.randn(3))
```

在上面的例子中，`torch.compile` 将追踪 `fn`（包括 `inner1`）直到遇到 `skip_frame`。然后 `fn` 被跳过并以即时执行模式运行——当 `inner1` 和 `inner2` 被调用时，它们会被编译。

跳过函数可能导致优化机会丢失，因此检查您希望编译的代码是否被跳过非常重要，如果是，则需要解决跳过问题。

## 循环中的图中断

如果图中断发生在循环中，`torch.compile` 无法恢复追踪：

```{code-cell}
@torch.compile
def fn(x):
    for i in range(5):
        x = x + 1
        if i == 3:
            torch._dynamo.graph_break()
    return x
fn(torch.randn(3))
```

在这个例子中，我们可以通过展开循环来避免跳过：

```{code-cell}
@torch.compile
def fn(x):
    def inner(i):
        nonlocal x
        x = x + 1
        if i == 3:
            torch._dynamo.graph_break()
    inner(0)
    inner(1)
    inner(2)
    inner(3)
    inner(4)
    return x
fn(torch.randn(3))
```

通常，解决导致跳过的图中断也将解决跳过问题。

## 上下文管理器中的图中断

另一个常见的无法恢复的图中断例子是发生在大多数上下文管理器中的图中断：

```{code-cell}
class CustomCtxManager:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        pass
@torch.compile
def fn(x):
    with CustomCtxManager():
        x = x + 1
        torch._dynamo.graph_break()
        return x + 1
fn(torch.randn(3))
```

我们可以通过将图中断移到上下文管理器外部来避免跳过：

```{code-cell}
@torch.compile
def fn(x):
    with CustomCtxManager():
        x = x + 1
    torch._dynamo.graph_break()
    with CustomCtxManager():
        return x + 1
fn(torch.randn(3))
```

有一些上下文管理器，Dynamo 可以在图中断后恢复追踪。其中一些可以在 `torch/_dynamo/variables/torch.py` 中的 `supported_ctx_manager_classes` 中找到。通常，任何由 `torch/_dynamo/variables/ctx_manager.py` 中的 `ContextWrappingVariable` 子类表示的上下文管理器都支持在图中断后恢复。例如：

```{code-cell}
import contextlib
@torch.compile
def fn(x):
    with contextlib.nullcontext():
        with torch.no_grad():
            x = x + 1
            torch._dynamo.graph_break()
            return x + 1
fn(torch.randn(3))
```

## Try 块中的图中断

Try 块中的图中断无法恢复：

```{code-cell}
@torch.compile
def fn(x):
    try:
        x = x + 1
        torch._dynamo.graph_break()
        return x + 1
    except Exception as e:
        pass
fn(torch.randn(3))
```

我们可以通过将图中断移到 try 块外部来避免跳过：

```{code-cell}
@torch.compile
def fn(x):
    try:
        x = x + 1
    except Exception as e:
        pass
    torch._dynamo.graph_break()
    try:
        return x + 1
    except Exception as e:
        pass
fn(torch.randn(3))
```

## 达到重新编译限制
请参阅 [更改缓存大小限制](programming_model.recompilation.changing_cache_size_limit)。

## 编译器错误
一些编译器错误会导致函数被跳过。其他编译器错误会导致硬错误（hard error）而不是函数被跳过。

## 处理被跳过的函数
通常，您可以通过修复导致函数被跳过的底层图中断或错误来解决跳过问题。

如果导致函数被跳过的图中断/错误难以修复，那么考虑将有问题的图中断/错误隔离在其自身的函数中，以便最小化被跳过的内容。

```{code-cell}
def inner1(x):
    return x + 1
def inner2(x):
    return x + 2
@torch.compile
def fn(x):
    x = inner1(x)
    def problematic_code():
        torch._dynamo.skip_frame()
    problematic_code()
    x = inner2(x)
fn(torch.randn(3))
```
