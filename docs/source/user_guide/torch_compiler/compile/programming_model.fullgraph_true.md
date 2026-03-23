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

# 使用 `fullgraph=True` 来识别并消除图中断

使用 `torch.compile(fullgraph=False)`（默认值）是开始使用 `torch.compile` 的好方法：它通过图中断的能力支持所有 Python 程序，并在常见情况下提供良好的性能。

但是，如果您想从模型中获取更多性能，您应该明确考虑哪些代码区域应该被编译：
- 我们建议使用 `torch.compile(fullgraph=True)` 来查找并消除代码中的图中断。
- 如果您是库开发者（或测试您的代码是否与 `torch.compile` “兼容”），我们建议使用 `torch.compile(fullgraph=True)` 进行测试。

`torch.compile(fullgraph=True)` 比 `fullgraph=False` 提供更强的保证：
我们将始终捕获一个单一的 FX 图进行编译（如果由于图中断而无法捕获，则会报错）。
**特别是，您必须解决遇到的每一个图中断。**

有多种策略可以解决图中断。

## 策略 1：重写不支持的代码以使用 Dynamo 支持的特性

许多图中断错误信息会给出一些关于如何重写代码以避免图中断的建议。
如果图中断仍然难以解决，请转到下一个策略
或向 [PyTorch GitHub 仓库](https://github.com/pytorch/pytorch/issues) 提交问题。

更多图中断示例及其解决方法可以在 [常见图中断](programming_model.common_graph_breaks) 中找到。

示例：Dynamo 不支持在被编译函数的输入 `list_iterator` 对象上调用 `next`。

```{code-cell}
@torch.compile(fullgraph=True)
def f(xs):
    a = next(xs)
    b = next(xs)
    return a + b

xs = [torch.tensor(1.), torch.tensor(2.)]
try:
    out = f(iter(xs))
except Exception as e:
    print(e)
```

相反，重写编译后的函数以接受一个列表。

```{code-cell}
@torch.compile(fullgraph=True)
def f_rewritten(xs):
    it = iter(xs)
    a = next(it)
    b = next(it)
    return a + b

f_rewritten(xs)
```

## 策略 2：纯函数总是可以通过逃生舱口进行编译。

**总结**：所有 Python 函数的空间非常庞大，因此 Dynamo 要能够无图中断地追踪每个 Python 函数是不切实际的。对于那些被认为是“纯”的、但 Dynamo 无法无图中断追踪的 Python 函数，我们提供了一些逃生舱口来尝试追踪这些函数：

1. 对纯 triton 内核使用 `custom_op` 或 `triton_op`。
2. 对仅使用 PyTorch 张量操作的纯函数使用 `nonstrict_trace`。
3. 对所有其他纯函数使用 `custom_op`。

“纯函数”是具有以下属性的函数：

- 确定性。给定相同的输入，纯函数总是返回相同的输出。
- 无外部副作用。纯函数没有任何外部可见的副作用，例如修改外部状态或执行 I/O 操作。函数内部保持的副作用是允许的（例如，修改中间张量）。一个值得注意的例外是，通常允许对函数输入张量进行 `torch.*` 操作的修改。
- 显式输入/输出。所有输入数据必须通过函数参数传递，所有输出都从函数返回。

有关示例，请参见 [纯函数](programming_model.non_strict_tracing_model.pure_functions)。

理论上，Dynamo 能够处理各种非纯函数，但可能对特定的 Python 语言特性支持不足。然而，纯函数总是可以通过逃生舱口进行编译。

如果您遇到图中断，可以尝试将周围的代码重构为一个纯函数，并使用绕过 Dynamo 追踪的逃生舱口：

1. 如果您希望函数中的张量操作出现在 Dynamo 输出图中（从而可优化），请使用 `torch._dynamo.nonstrict_trace`。`nonstrict_trace` 告诉 Dynamo 使用 **非严格追踪**。
2. 如果您希望函数对 `torch.compile`（前端 Dynamo 和后端）不透明，请使用自定义运算符。

请注意，这些逃生舱口也可以应用于非纯函数，但 **我们不提供任何正确性保证**。

示例：如果 Dynamo 不支持某些非严格可追踪的 Python 特性或 API（例如，它使用 PyTorch 操作），[请使用 `torch._dynamo.nonstrict_trace` 来捕获它](programming_model.dynamo_nonstrict_trace)。

```{code-cell}
# 这是一个 Dynamo 不支持的函数（由于 graph_break() 调用）。
def g(x):
    y = x.sin()
    torch._dynamo.graph_break()
    z = y.sin()
    return z

@torch.compile(fullgraph=True)
def f(x):
    w = x.sin()
    return g(w)

x = torch.randn(3)
try:
    f(x)  # 图中断：调用了 torch._dynamo.graph_break()
except Exception as e:
    print(e)

@torch.compile(fullgraph=True)
def f_rewritten(x):
    w = x.sin()
    return torch._dynamo.nonstrict_trace(g)(w)
f_rewritten(x)  # 正常工作
```

示例：使用 [自定义运算符](programming_model.custom_ops) 创建对 `torch.compile` 不透明的函数。

```{code-cell}
from torch.utils.cpp_extension import load_inline

# 平方操作的 C++ 源代码
cpp_source = """
torch::Tensor square_cpu(torch::Tensor input) {
    // 检查输入是否为 CPU 张量
    TORCH_CHECK(input.device().is_cpu(), "Input must be a CPU tensor");

    // 创建与输入形状和数据类型相同的输出张量
    torch::Tensor output = torch::empty_like(input);

    // 获取数据指针
    float* input_data = input.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    // 获取元素总数
    int64_t numel = input.numel();

```{code-cell}
# 使用 torch.library.custom_op 定义一个新的自定义运算符。
# 自定义运算符对于 torch.compile 是不透明的：
# 也就是说，torch.compile 不会窥探其内部。

@torch.library.custom_op("mylib::square", mutates_args=())
def square(x: torch.Tensor) -> torch.Tensor:
    return square_module.square_cpu(x)

# 使用 register_fake 为运算符添加一个 ``FakeTensor`` 内核
@square.register_fake
def _(x):
    return x.new_empty(x.size())

print(f(torch.randn(3, 3)))  # 无图中断
```

有关用于自定义 triton 内核的 `triton_op` 的更多信息，请参阅
[用户定义的 triton 内核教程](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)。


## 策略三：不编译代码

并非所有代码都适合编译。`torch.compile` 是一个用于张量计算的编译器；
它无法优化诸如磁盘 IO 之类的操作。尝试重构代码，使得不支持的操作不在编译区域内被调用。

```{code-cell}
@torch.compile(fullgraph=True)
def f(x):
    y = x ** 2  / 2
    torch.save(y, "foo.pt")
    z = y ** 3 / 6
    return z

x = torch.randn(3)
try:
    f(x)  # 图中断：torch.save 不被支持
except Exception as e:
    print(e)
```

```{code-cell}
def f_rewritten(x):
    y = g(x)
    torch.save(y, "foo.pt")
    z = h(y)
    return z

@torch.compile(fullgraph=True)
def g(x):
    y = x ** 2  / 2
    return y

@torch.compile(fullgraph=True)
def h(y):
    z = y ** 3 / 6
    return z

f_rewritten(x)
```

```{code-cell}
:tags: [remove-cell]
import os
os.remove("foo.pt")
```


如果你有一个有问题的函数，你不需要在编译下运行它，那么
考虑使用 `torch.compiler.is_compiling()` 来跳过这个有问题的函数。

```{code-cell}
@torch.compile(fullgraph=True)
def f(x):
    y = x ** 2  / 2
    if not torch.compiler.is_compiling():
        torch.save(y, "foo.pt")
    z = y ** 3 / 6
    return z

x = torch.randn(3)
f(x)  # torch.save 未被调用
```

如果你有一个在许多地方被调用的函数，并且你允许 `torch.compile`
无条件地跳过它，那么你可以将其添加到 `torch._dynamo.config.ignore_logging_functions`。

```{code-cell}
def bad_fn(y):
    torch.save(y, "foo.pt")

torch._dynamo.config.ignore_logging_functions.add(bad_fn)

@torch.compile(fullgraph=True)
def f(x):
    y = x ** 2  / 2
    bad_fn()
    z = y ** 3 / 6
    return z

x = torch.randn(3)
f(x)  # torch.save 未被调用
```

请注意，可以添加到 `ignore_logging_functions` 的函数类型有一些限制。
具体来说：
- 该函数可以接受任何参数，但**必须**返回 `None`。
- 函数应该是模块级函数、`logging.Logger.<method>`（忽略所有 `logging.Logger` 实例的该方法）或 `logger_obj.<method>`（仅忽略特定 `logger_obj` 实例的该方法）。

由于实现细节，其他函数可能会也可能不会被忽略。如果你想忽略一个 `ignore_logging_functions` 未能忽略的函数，请提交一个问题。