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

torch._logging.set_logs(graph_breaks=True, graph_code=True)
```

# 禁用和抑制错误
对于某些模型架构，模型中的某些部分特别难以编译——要么存在许多图中断，要么会崩溃。
您可能希望显式禁用这些有问题的模型部分，以便可以将 `torch.compile` 应用于可正常工作的部分。您可以通过使用 `@torch.compiler.disable` 装饰器来实现这一点。
当 `torch.compile` 尝试调用一个被禁用的函数时，它会中断图并跳过跟踪该禁用函数，在调用后恢复跟踪。默认情况下，从禁用函数进行的所有递归调用也会被禁用。
使用 `recursive=False` 选项可以允许对递归调用进行编译。

```{code-cell}
def inner1(x):
    torch._dynamo.graph_break()  # 不被跟踪
    return x + 1  # 不被跟踪

@torch.compiler.disable
def outer1(x):
    x = x + 2  # 不被跟踪
    torch._dynamo.graph_break()  # 不被跟踪
    return inner1(x)

@torch.compile
def f(x):
    x = outer1(x)
    return x + 4  # 被跟踪

print(f(torch.ones(3)))
```

```{code-cell}
def inner2(x):
    torch._dynamo.graph_break()  # 被跟踪
    return x + 1  # 被跟踪

@torch.compiler.disable(recursive=False)
def outer2(x):
    x = x + 2  # 不被跟踪
    torch._dynamo.graph_break()  # 不被跟踪
    return inner2(x)

@torch.compile
def g(x):
    x = outer2(x)
    return x + 4  # 被跟踪

print(g(torch.ones(3)))
```

例如，可以使用 `torch.compiler.disable` 在推荐模型中禁用对稀疏架构的 `torch.compile`，因为稀疏架构难以编译。
预处理和日志记录函数是其他通常会导致大量图中断且无法从编译中获益的函数的例子。

如果您遇到编译器崩溃但希望无论如何继续执行，可以设置 `torch._dynamo.config.suppress_errors = True`。
当编译器崩溃时，我们将直接跳过跟踪该函数，稍后再重试。
**这不是最佳实践**——更好的做法是根据需要最终手动添加 `disable` 注解。