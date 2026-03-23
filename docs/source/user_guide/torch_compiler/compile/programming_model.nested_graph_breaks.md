# 嵌套图断点

概述：
- 嵌套函数中的图断点可能导致难以理解的编译器行为，我们将在下文进行说明
- 嵌套图断点会导致 {math}`\mathcal O(N)` 次重复的图断点行为

回顾一下，当 `torch.compile` 应用于一个函数时，任何嵌套的函数调用也会被追踪。
**嵌套图断点**指的是在嵌套函数调用中发生的任何图断点。

```python
def inner(x):
    ...
    torch._dynamo.graph_break()  # 嵌套图断点
    ...

@torch.compile
def outer(x):
    ...
    y = inner(x)
    ...
```

嵌套图断点周围的恢复语义可能会令人困惑，因此我们在此描述其行为。

回顾一下，在 `fullgraph=False` 模式下，[图断点的处理方式](programming_model.dynamo_core_concepts.graph_breaks)是：编译目前已确定的 FX 图，在常规 Python 中运行不受支持的代码，然后在不受支持的代码之后使用新的 FX 图恢复追踪。恢复函数实际上是一项相当复杂的技术壮举，因此仅支持在顶层函数上恢复追踪。

因此，我们可以在嵌套图断点之后，按照以下方式恢复追踪，但需遵守此限制：

首先，考虑以下示例，其中 `torch.compile` 从 `f` 开始追踪，并一直追踪到遇到 `inner1` 中的图断点。

```python
def inner1(x):
    x = x + 1
    torch._dynamo.graph_break()  # 由于图断点停止追踪
    return x + 2

def inner2(x):
    x = x + 4
    x = inner1(x)
    x = x + 8

@torch.compile
def f(x):
    # 从这里开始追踪
    x = x + 16
    x = inner2(x)
    x = x + 32

f(torch.randn(3))
```

由于我们只能从顶层函数恢复，因此我们在 `f` 中对 `inner2` 调用进行图断点。
```python
# torch.compile(f)(x) 的语义大致如下：
def compiled_f_semantics(x):
    y = x + 16
    z = inner2(y)
    return torch.compile(resume_f_semantics)(z)

def resume_f_semantics(x):
    return x + 32

compiled_f_semantics(torch.randn(3))
```

然后 `inner2` 会自动作为顶层函数被编译。
我们一直追踪直到再次遇到 `inner1` 中的图断点。

```python
def inner1(x):
    x = x + 1
    torch._dynamo.graph_break()  # 由于图断点停止追踪
    return x + 2

# 这个 torch.compile 会自动应用
@torch.compile
def inner2(x):
    # 从这里开始追踪
    x = x + 4
    x = inner1(x)
    x = x + 8

def compiled_f_semantics(x):
    y = x + 16
    z = inner2(y)
    return torch.compile(resume_f_semantics)(z)

def resume_f_semantics(x):
    return x + 32

compiled_f_semantics(torch.randn(3))
```

然后我们在 `inner2` 中对 `inner1` 调用进行图断点。
```python
def compiled_inner2_semantics(x):
    y = x + 4
    z = inner1(y)
    return torch.compile(resume_inner2_semantics)(z)

def resume_inner2_semantics(x):
    return x + 8
```

然后 `inner1` 会自动作为顶层函数被编译。
图断点来自 `inner1`，因此我们正常处理该图断点。
```python
# 这个 torch.compile 会自动应用
@torch.compile
def inner1(x):
    # 从这里开始追踪
    x = x + 1
    torch._dynamo.graph_break()  # 由于图断点停止追踪
    return x + 2

def compiled_f_semantics(x):
    y = x + 16
    z = compiled_inner2_semantics(y)
    return torch.compile(resume_f_semantics)(z)

def resume_f_semantics(x):
    return x + 32

def compiled_inner2_semantics(x):
    y = x + 4
    z = inner1(y)
    return torch.compile(resume_inner2_semantics)(z)

def resume_inner2_semantics(x):
    return x + 8

compiled_f_semantics(torch.randn(3))
```

`inner1` 被正常处理：

```python
def compiled_inner1_semantics(x):
    y = x + 1
    torch._dynamo.graph_break()
    return torch.compile(resume_inner1_semantics)(y)

def resume_inner1_semantics(x):
    return x + 2
```

因此，初始代码在语义上等价于：
```python
def compiled_f_semantics(x):
    y = x + 16
    z = compiled_inner2_semantics(y)
    return torch.compile(resume_f_semantics)(z)

def resume_f_semantics(x):
    return x + 32

def compiled_inner2_semantics(x):
    y = x + 4
    z = compiled_inner1_semantics(y)
    return torch.compile(resume_inner2_semantics)(z)

def resume_inner2_semantics(x):
    return x + 8

def compiled_inner1_semantics(x):
    y = x + 1
    torch._dynamo.graph_break()
    return torch.compile(resume_inner1_semantics)(y)

def resume_inner1_semantics(x):
    return x + 2

compiled_f_semantics(torch.randn(3))
```

特别要注意的是，我们追踪了 3 个顶层函数，并且对同一个图断点追踪了 3 次。
**这解释了为什么在使用 `torch.compile` 时可能会遇到重复的图断点。**

总之，嵌套图断点的处理方式是：
- 从顶层函数开始追踪，一直追踪到嵌套图断点
- 在顶层函数中对第二层函数的调用处进行图断点
- 编译目前已追踪的 PyTorch 操作并运行编译后的图
- 调用第二层函数，该函数会自动作为顶层函数被编译
- 在第二层函数调用后恢复追踪

请注意，处理此图断点的运行时间为 {math}`\mathcal O(NK)`，其中 {math}`N` 是嵌套深度，{math}`K` 是从顶层函数到图断点的指令数量。我们最终会追踪 {math}`\mathcal O(N^2)` 个帧，并且对同一个图断点追踪 {math}`\mathcal O(N)` 次。