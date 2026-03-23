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
```

# Dynamo 核心概念

**摘要：**

- Dynamo 作为 `torch.compile` 的前端，通过**追踪**来捕获 Python 函数（及其嵌套函数调用）的语义，将其转换为一系列线性操作（即“(FX) 计算图”）、剩余的字节码以及“守卫”（一组定义计算图和字节码有效性的条件列表）。
- 不支持的 Python 特性会导致**图中断**，此时 Dynamo 会编译从追踪中获取的部分计算图，然后运行不支持的代码，再恢复追踪。
- 图中断可能导致 torch.compile 运行缓慢，并阻碍后端优化机会。如果您未获得预期的性能，请检查是否存在图中断。

## Dynamo 追踪
`torch.compile` 的前端（Dynamo）是一个自定义的 Python 字节码解释器，旨在允许在 PyTorch 程序中进行图编译，同时保留 Python 的完全灵活性。给定一个待编译的函数，Dynamo 解释 Python 字节码，将 PyTorch 操作序列提取到一个或多个 FX 计算图中，这些图可以由后端进一步优化。

![Dynamo 摘要图](_static/dynamo_summary_diagram.png)

例如，对于上图中的函数 `f`，Dynamo 生成：
- 一个**FX 计算图**，它接收原始输入以及函数所需的一些额外输入。
- 可作为 `f` 直接替代的**Python 字节码**。在我们的示例中，字节码检索额外输入并将其传递给计算图，同时包含不可优化的 Python 副作用（列表追加操作）。
- **守卫**，用于指定计算图和字节码有效的条件。除非另有说明，Dynamo 生成的计算图会专门针对输入张量的形状。

(programming_model.dynamo_core_concepts.graph_breaks)=

## 图中断
Dynamo 追踪您的代码，并尝试将您的 PyTorch 代码捕获为单个 PyTorch 操作符（FX 计算图）的计算图。然而，这并非总是可行。当遇到无法追踪的代码时，就会发生“**图中断**”。在默认的 `torch.compile` 设置中，图中断涉及编译迄今为止已确定的部分 FX 计算图，在常规 Python 中运行不支持的代码，然后在不支持的代码之后恢复追踪并开始一个新的 FX 计算图。

图中断是 Dynamo 的一项特性，使其能够运行任意 Python 代码，并提取出可以各自独立优化的功能性子图。

但是，图中断可能导致 `torch.compile` 出现意外的缓慢。如果您未获得预期的加速效果，我们建议检查并移除图中断。

图中断可能发生在以下情况：

- 数据依赖的 if 语句
- 许多 Python 内置函数
- C 函数

```{code-cell}
:tags: [remove-cell]
torch._logging.set_logs(graph_breaks=True)
```

以下是一个由于调用不支持的操作 `torch.save` 而导致图中断的示例：

```{code-cell}
@torch.compile
def f(x):
   y = x ** 2  / 2
   torch.save(y, "foo.pt")  # torch.save 是一个不支持的操作
   z = y ** 3 / 6
   return z

x = torch.randn(3)
print(f(x))
```

```{code-cell}
:tags: [remove-cell]
import os
os.remove("foo.pt")
```

`torch.compile(f)(x)` 的语义大致如下：

```python
def compiled_f_semantics(x):
   y = torch.compile(g, fullgraph=True)(x)
   torch.save(y, "foo.pt")
   z = torch.compile(h, fullgraph=True)(x)
   return z

def g(x):
    return x ** 2  / 2

def h(x):
    return y ** 3 / 6
```

## 守卫

`torch.compile` 在追踪代码时会对运行时值做出一些假设。在追踪过程中，我们会生成“守卫”，这些是对这些假设的运行时检查。在后续调用已编译函数时，会运行这些守卫以确定是否可以重用先前编译的代码。运行时检查的示例包括常量值、类型和对象 ID。

以下是生成守卫的示例。`TENSOR_MATCH` 守卫检查输入的类型、设备、数据类型、形状等。

```{code-cell}
:tags: [remove-cell]
torch._logging.set_logs(guards=True)
```

```{code-cell}
@torch.compile
def fn(x):
    return x + 1

print(fn(torch.ones(3, 3)))
```

## 重新编译
如果所有先前编译代码实例的守卫检查都失败，那么 `torch.compile` 必须“重新编译”该函数，需要再次追踪原始代码。在下面的示例中，由于检查张量参数形状的守卫失败，需要重新编译。

```{code-cell}
:tags: [remove-cell]
torch._logging.set_logs(recompiles=True)
```

```{code-cell}
@torch.compile
def fn(x):
    return x + 1

print(fn(torch.ones(3, 3)))
print(fn(torch.ones(4, 4)))
```

## 动态形状

`torch.compile` 最初假设张量形状是静态/常量，并基于这些假设生成守卫。通过使用“动态形状”，我们可以让 `torch.compile` 生成能够接受不同形状张量输入的编译代码——避免每次形状不同时都重新编译。默认情况下，`torch.compile(dynamic=None)` 启用了自动动态形状——如果由于形状不匹配导致编译失败，会尝试使用动态形状重新编译。动态形状也可以完全启用（`dynamic=True`）或禁用（`dynamic=False`）。

下面，我们启用动态形状，并注意不再需要重新编译。

```{code-cell}
:tags: [remove-cell]
import logging
torch._logging.set_logs(dynamic=logging.DEBUG, recompiles=True)
```

```{code-cell}
@torch.compile(dynamic=True)
def fn(x):
    return x + 1

print(fn(torch.ones(3, 3)))
print(fn(torch.ones(4, 4)))
```

有关动态形状的更多信息，请参阅 [动态形状手册](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit?tab=t.0#heading=h.fh8zzonyw8ng)。