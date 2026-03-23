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
```

# 使用 `torch._dynamo.nonstrict_trace`

**摘要：**
- 使用 `nonstrict_trace` 在 `torch.compile` 编译区域内以非严格追踪模式追踪一个函数。
  你可能希望这样做，是因为 Dynamo 图在该函数内部某处中断，而你确信该函数可以进行非严格追踪。

考虑以下场景：

```{code-cell}
def get_magic_num():
    # 这个显式的图中断调用旨在模拟任何类型的 Dynamo 图中断，
    # 例如，函数是用 C 实现的，或者使用了 Dynamo 尚不支持的某些 Python 语言特性。
    torch._dynamo.graph_break()
    return torch.tensor([42])
@torch.compile(fullgraph=True)
def func(x):
    n = get_magic_num()
    return x + n
try:
    func(torch.rand(10))
except Exception as e:
    print(e)
```

如果我们运行上面的代码，将会收到来自 Dynamo 的错误，因为它在用户指定了 `fullgraph=True` 的情况下遇到了图中断。

在这些情况下，如果用户仍然希望保持 `fullgraph=True`，他们通常有几个选择：

1. 图中断是由于 Dynamo 尚不支持的语言特性导致的。
   在这种情况下，用户要么重写他们的代码，要么在 GitHub 上提交问题。
2. 图中断是由于调用了用 C 实现的函数。
   在这种情况下，用户可以尝试使用自定义操作符。
   用户也可以尝试提供一个 polyfill（一个用 Python 实现的参考实现），以便 Dynamo 能够追踪它。
3. 最坏的情况——内部编译器错误。在这种情况下，用户可能必须在 GitHub 上提交问题。

除了所有这些选项之外，如果导致图中断的函数调用满足某些要求，PyTorch 确实提供了一个替代方案 `torch._dynamo.nonstrict_trace`：

- 满足[通用非严格追踪](programming_model.non_strict_tracing_model)的要求。
- 输入和输出必须包含基本类型（例如，`int`、`float`、`list`、`dict`、`torch.Tensor`），
  或者已注册到 `torch.utils._pytree` 的用户定义类型。
- 函数必须在 `torch.compile` 编译区域之外定义。
- 函数读取的任何非输入值都将被视为常量（例如，全局张量），并且不会受到守卫保护。

在追踪对 `torch._dynamo.nonstrict_trace` 修饰的函数的调用时，`torch.compile` 会切换到[非严格追踪模式](programming_model.non_strict_tracing_model)，
并且 FX 图最终将包含该函数内部发生的所有相关张量操作。

对于上面的示例，我们可以使用 `torch._dynamo.nonstrict_trace` 来消除图中断：

```{code-cell}
@torch._dynamo.nonstrict_trace
def get_magic_num():
    # 这个显式的图中断调用旨在模拟任何类型的 Dynamo 图中断，
    # 例如，函数是用 C 实现的，或者使用了 Dynamo 尚不支持的某些 Python 语言特性。
    torch._dynamo.graph_break()
    return torch.tensor([42])
@torch.compile(fullgraph=True)
def func(x):
    n = get_magic_num()
    return x + n
print(func(torch.rand(10)))
# 没有图中断，也没有错误。
```

请注意，也可以在 `torch.compile` 编译区域内使用它：

```{code-cell}
def get_magic_num():
    # 这个显式的图中断调用旨在模拟任何类型的 Dynamo 图中断，
    # 例如，函数是用 C 实现的，或者使用了 Dynamo 尚不支持的某些 Python 语言特性。
    torch._dynamo.graph_break()
    return torch.tensor([42])
@torch.compile(fullgraph=True)
def func(x):
    n = torch._dynamo.nonstrict_trace(get_magic_num)()
    return x + n
print(func(torch.rand(10)))
# 没有图中断，也没有错误。
```
