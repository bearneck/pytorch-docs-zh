
# Dynamo 深度解析

TorchDynamo（或简称 Dynamo）是 `torch.compile` 内部的追踪器，那些令人抓狂的回溯错误往往要归咎于它。然而，我们不能盲目地将这些错误归咎于 Dynamo。为了提供它所具备的灵活性，Dynamo 被赋予了理解任何 Python 程序的艰巨任务。具体来说，Dynamo 必须在内部实现 Python 编程语言的大部分功能！

在这篇文章中，我们将从头开始探讨 Dynamo 的内部设计。我们将讨论它提供的功能以及它是如何实现的。读完本文后，当你 `torch.compile` 一个 PyTorch 程序而编译出错，或者编译成功但加速效果未达预期时，你将能更好地理解问题所在。

## Dynamo 温和入门

在深入探讨所有实现细节之前，让我们先来讨论一下 Dynamo 是做什么的。

Dynamo 是一个追踪器。这意味着，给定一个函数及其输入，它会执行该函数并将一系列线性指令（不含控制流）记录到一个图中。例如，考虑以下程序：

```python
import torch

@torch.compile
def mse(x, y):
    z = (x - y) ** 2
    return z.sum()

x = torch.randn(200)
y = torch.randn(200)
mse(x, y)
```

如果我们将这个程序保存到文件 `example.py` 并运行

```bash
TORCH_LOGS=graph_code python example.py
```

我们会看到 Dynamo 追踪到的输出

```python
def forward(l_x_: torch.Tensor, l_y_: torch.Tensor):
    # File: example.py:5, code: z = (x - y) ** 2
    sub = l_x_ - l_y_
    z = sub ** 2
    # File: example.py:6, code: return z.sum()
    sum_1 = z.sum()
    return (sum_1,)
```

我们称之为 **给定输入下函数的图（或追踪）**。这是通过一个 [FX 图](https://pytorch.org/docs/main/fx.html) 来表示的。我们可以简单地将 FX 图视为一个存储函数调用列表的容器。

首先我们应该注意到，这个图是一个线性的 PyTorch 操作序列。[^1] Dynamo 记录所有 PyTorch 操作并按顺序存储它们。例如，它将 `z = (x - y) ** 2` 拆分为两个构成操作：`sub = l_x_ - l_y_` 和 `z = sub ** 2`。

当我们说追踪是线性的时，我们指的是没有分支或任何控制流。为了说明这一点，考虑

```python
import torch

@torch.compile
def fn(x, n):
    y = x ** 2
    if n >= 0:
        return (n + 1) * y
    else:
        return y / n

x = torch.randn(200)
fn(x, 2)
```

当使用 `TORCH_LOGS=graph_code` 执行时，它返回

```python
def forward(l_x_: torch.Tensor):
    # File: example.py:5, code: y = x ** 2
    y = l_x_ ** 2
    # File: example.py:7, code: return (n + 1) * y
    mul = 3 * y
    return (mul,)
```

我们看到 Dynamo 完全从追踪中移除了 `if` 语句，只记录了使用输入执行的操作。

因此，应该清楚的是，**函数的追踪取决于输入**。具体来说，这意味着追踪不是在我们写 `@torch.compile` 时生成的，而是在我们使用实际参数执行函数 `fn(x, 2)` 时生成的。

这里另一个需要注意的有趣之处是，Dynamo 移除了函数的第二个参数。相反，它将其视为常量，并在图中记录了操作 `n + 1` 的结果。这是 Dynamo 的另一个特性：Dynamo 会将任何非张量值视为常量……除了整数。现在让我们看看整数有何特殊之处。

Dynamo 的最后一个定义特性是它知道如何处理动态形状。符号形状指的是 Dynamo 追踪形状（更一般地说是整数）的能力，而不是将它们作为常量留下。这可以避免重新编译，并部署适用于生产环境中任意大小的通用模型。出现动态形状的主要例子包括批处理大小（我们可能使用固定的批处理大小训练模型，但随后对任意批处理大小进行推理），以及处理文本或音频时遇到的变长序列。

我们可以通过多次执行上面的例子来看到这一点

```python
import torch

@torch.compile
def fn(x, n):
    y = x ** 2
    if n >= 0:
        return (n + 1) * y
    else:
        return y / n

x = torch.randn(200)
fn(x, 2)
fn(x, 3)
fn(x, -2)
```

在这种情况下，`TORCH_LOGS=graph_code` 生成了另外两个图

```python
# 省略 n==2 的图

def forward(self, l_x_: torch.Tensor, l_n_: torch.SymInt):
    # File: a.py:5, code: y = x ** 2
    y = l_x_ ** 2

    # File: a.py:7, code: return (n + 1) * y
    add = l_n_ + 1
    mul = add * y
    return (mul,)
```

```python
def forward(self, l_x_: torch.Tensor, l_n_: torch.SymInt):
    # File: a.py:5, code: y = x ** 2
    y = l_x_ ** 2

    # File: a.py:9, code: return y / n
    truediv = y / l_n_
    return (truediv,)
```

Dynamo 检测到第一次调用后某个整数的值发生了变化，并开始追踪它。我们看到这些图是通用的，并通过一个 `SymInt` 类型的对象符号化地追踪变量 `n`。

如果在这些调用之后我们调用 `fn(x, 4)`，Dynamo 不会重新编译，而是重用已经追踪过的图。

总结一下：1. Dynamo 是一个 Python 追踪器 2. 给定一些输入，它返回一个包含已执行 PyTorch 函数的 FX 图 3. 如果检测到整数在调用之间发生变化，它也可以追踪整数 4. 它将任何不是张量或标量的值特化

当然，Dynamo 还做了更多事情，比如确定何时需要重新追踪、重写函数的字节码、实现图中断……为了保持介绍简短，我们将在后续部分逐步讨论所有这些内容。

## PEP 523：向 CPython 添加帧求值 API

现在假设我们被赋予实现 Dynamo 的任务。我们该从何处开始呢？对我们来说相当方便的是，[PEP 523](https://peps.python.org/pep-0523/) 随 Python 3.6 一起发布。这个 PEP [的设计初衷](https://peps.python.org/pep-0523/#a-jit-for-cpython) 是允许第三方为 Python 创建 JIT 编译器。让我们看看它是如何实现的。

**关于 CPython 的说明**：CPython 在内部实现为一个[栈机](https://en.wikipedia.org/wiki/Stack_machine)。Python 程序被编译成[字节码](https://en.wikipedia.org/wiki/Bytecode)，然后由这个解释器执行。要了解更多关于这些字节码的信息，请参阅标准库中的 [dis 模块](https://docs.python.org/3/library/dis.html)。另请参阅[开发者文档](https://devguide.python.org/internals/interpreter/)以了解 CPython 解释器的介绍。我们将假设读者熟悉栈机的概念。

PEP 523 公开了一个 API，允许用户添加一个自定义的逐函数解释器。然后，CPython 将使用这个解释器而不是它自己的解释器来执行函数。为了能够执行函数，在入口处，CPython 向自定义解释器提供以下内容：
- 函数的字节码
- 函数参数的值（即局部变量）及其名称
- 全局变量的值及其名称
- 内置函数，如 `abs` 或 `print`

你可以在[这里](https://github.com/pytorch/pytorch/blob/e891a3bba9f05697d72776f6e89347231a141f03/torch/csrc/dynamo/eval_frame.c#L50-L59)查看所有字段。[^2]

总之，CPython 向用户的解释器提供了执行函数所需的所有信息。[^3]

利用这个 API，我们可以通过实现一个解释器来实现一个追踪器，该解释器运行代码并在图中记录执行期间发生的所有 PyTorch 操作。这正是 Dynamo 所做的。

Dynamo 使用这个 CPython API 来解析所有这些对象，并将它们打包到[一个 Python 结构](https://github.com/pytorch/pytorch/blob/e891a3bba9f05697d72776f6e89347231a141f03/torch/csrc/dynamo/eval_frame.c#L93-L108)中。完成此操作后……它从 C 返回到 Python。除了这段与 CPython 通信的代码外，Dynamo 完全是用 Python 实现的。

应该清楚的是，装饰器 `@torch.compile` 的任务是安装必要的脚手架，以便在调用函数时将字节码、参数、全局变量等传递给 Dynamo。再次强调，`@torch.compile` 实际上并不编译任何东西。

## 在 Python 中实现 CPython

所以，我们回到了 Python 世界。我们有了一个函数的字节码，以及执行它所需的所有上下文。具体来说，我们到达了 [_convert_frame_assert](https://github.com/pytorch/pytorch/blob/b6df8414601e1e086e830ca9e919e7fdc8874e71/torch/_dynamo/convert_frame.py#L272-L274)。这就是装饰器 `torch.compile` 返回的函数！我们是从 [_dynamo.optimize](https://github.com/pytorch/pytorch/blob/b6df8414601e1e086e830ca9e919e7fdc8874e71/torch/_dynamo/eval_frame.py#L715-L727) 到达这个函数的。装饰器 `torch.compile` 只是 `_dynamo.optimize` 的一个友好 API。

在开始实现 Python 解释器之前，我们想要定义一个[中间表示](https://en.wikipedia.org/wiki/Intermediate_representation)。具体来说，我们想要将所有局部变量和全局变量包装在我们自己的内部类中。这使我们能够更好地跟踪这些对象，并将 Dynamo 眼中可以以相同方式处理的对象分组在一起。

内部类结构的父类是 `VariableTracker`，它代表了 Dynamo 理解的不同对象。例如，`ListVariable` 代表一个 `list` 对象，并在内部维护一个[VariableTracker 列表](https://github.com/pytorch/pytorch/blob/e38a3a6079a3861b4bc9f256120ec661f34e726d/torch/_dynamo/variables/lists.py#L48-L56)。另一个 `VariableTracker` 的例子是 [ConstantVariable](https://github.com/pytorch/pytorch/blob/83c0763dda1f93c6cf552ba88260a0dc7a3ecb70/torch/_dynamo/variables/constant.py#L30)。ConstantVariable 包装了所有[Dynamo 视为常量的对象](https://github.com/pytorch/pytorch/blob/83c0763dda1f93c6cf552ba88260a0dc7a3ecb70/torch/_dynamo/variables/constant.py#L98-L107)。我们还有针对需要特殊处理的对象的特殊子类，例如 [TensorVariable](https://github.com/pytorch/pytorch/blob/83c0763dda1f93c6cf552ba88260a0dc7a3ecb70/torch/_dynamo/variables/tensor.py#L68-L69)。所有这些内部类都定义在 [torch/_dynamo/variables](https://github.com/pytorch/pytorch/tree/83c0763dda1f93c6cf552ba88260a0dc7a3ecb70/torch/_dynamo/variables) 文件夹中。

Python 对象在 [VariableBuilder._wrap](https://github.com/pytorch/pytorch/blob/83c0763dda1f93c6cf552ba88260a0dc7a3ecb70/torch/_dynamo/variables/builder.py#L365) 中被包装到它们对应的 `VariableTracker` 类中。这个函数只是一个非常长的 `elif` 链，试图递归地将 Python 输入模式匹配到适当类型的 `VariableTracker`。

**调试提示**。当我们从 dynamo 得到意外结果时，有时是由构建器引起的。如果构建器的逻辑错误，有时 Dynamo 可能会将变量包装在错误的 `VariableTracker` 类型中，这可能会导致后续问题。查看错误中出现的 `VariableTracker` 类型以及抛出异常的 `VariableTracker` 方法，这在遇到 Dynamo 错误时非常有用。特别是，有时我们发现一个对象被跟踪为 `UserDefinedObjectVariable`（这是 Dynamo 的通用类），而它本应被跟踪为更具体的类型。在这些情况下，`VariableBuilder` 的逻辑往往是罪魁祸首。

**调试提示**。当使用 `TORCH_LOGS=dynamo` 运行程序时，打印出的工件之一是形式如下的行：

```
TRACE LOAD_GLOBAL y [TorchInGraphFunctionVariable(<built-in method any>), TensorVariable()]
```

这是原始程序的字节码以及该时刻的堆栈状态。这对于查找对象未被正确追踪到合适的 `VariableTracker` 的位置非常有用。

好的，现在我们有了追踪器的中间表示（IR），接下来我们*只需要*重新实现 CPython 的堆栈机。这由 [symbolic_convert.py](https://github.com/pytorch/pytorch/blob/69f112d5867f785a3a090a0c6d6644ae047033ac/torch/_dynamo/symbolic_convert.py) 文件中的 [InstructorTranslatorBase](https://github.com/pytorch/pytorch/blob/69f112d5867f785a3a090a0c6d6644ae047033ac/torch/_dynamo/symbolic_convert.py#L576-L594) 实现。

`InstructionTranslatorBase` 有大约 200 个方法，实现了几乎所有的 Python 字节码。例如，我们可以看到 `BUILD_LIST` 的实现：

```python
def BUILD_LIST(self, inst):
    items = self.popn(inst.argval)
    self.push(ListVariable(items, mutation_type=ValueMutationNew()))
```

这是由类似 `l = [2, 3, 4]` 这样的构造生成的字节码。在这种情况下，由于有三个元素，生成的字节码是 `BUILD_LIST 3`。这意味着我们从堆栈顶部弹出 `3` 个元素，并将一个由这三个元素组成的新列表对象推送到堆栈顶部。

## 生成输出图

有了符号化执行 Python 代码的方法，我们就可以提取在给定某些输入的情况下，程序符号化执行期间发生的 PyTorch 操作。这在 Dynamo 中通过 [OutputGraph](https://github.com/pytorch/pytorch/blob/69f112d5867f785a3a090a0c6d6644ae047033ac/torch/_dynamo/output_graph.py#L221-L230) 对象实现。`OutputGraph` 对象[绑定到一个 `InstructionTranslator` 对象](https://github.com/pytorch/pytorch/blob/69f112d5867f785a3a090a0c6d6644ae047033ac/torch/_dynamo/symbolic_convert.py#L2060-L2071)，它跟踪创建将由 Dynamo 返回的 FX 图所需的所有数据。

FX 图的所有输入和中间元素都是 `fx.Node`。在 Dynamo 中，`fx.Node` 被包装在 `fx.Proxy` 中。`fx.Proxy` 用于构建 FX 图。具体来说，它们将对它们执行的每个 PyTorch 操作记录到图中。你可以通过调用 [create_proxy](https://github.com/pytorch/pytorch/blob/fb80f05ee2e1cba17892980701bfd5dbce58349f/torch/_dynamo/output_graph.py#L430-L431) 来创建一个要添加到图中的新操作。然后，我们可以通过函数 [wrap_fx_proxy](https://github.com/pytorch/pytorch/blob/fb80f05ee2e1cba17892980701bfd5dbce58349f/torch/_dynamo/variables/builder.py#L1311) 将其添加到图中。

一个图存储对张量的操作……以及对符号整数的操作。我们稍后将讨论符号整数，但首先我们将讨论 Dynamo 如何解决一个相当重要的正确性问题。


## 确保 Dynamo 的正确性：守卫（Guards）

至此，我们有了一种完全忽略控制流来追踪程序的方法。为此，我们重新实现了整个 CPython……如果这听起来有点杀鸡用牛刀，那是因为确实如此。[torch.jit.trace](https://pytorch.org/docs/main/generated/torch.jit.trace.html) 已经实现了这个功能，而且没有所有这些复杂的机制，那么 Dynamo 的意义何在？

`torch.jit.trace` 的问题在于，正如其文档中警告的那样，它只在被追踪的程序不依赖于数据时才有效。换句话说，只有当程序本身是线性的时才有效。这意味着编写程序时不能使用 if-else、for-while 循环、异常。甚至我们使用的任何库都不能使用任何控制流！总而言之，在像 Python 这样动态的语言中不使用控制流，实际上是一个巨大的限制。

JAX 通过总是重新追踪并在重新追踪后缓存图来解决这个问题。而 Dynamo 则使用守卫来避免每次重新追踪整个程序。

**守卫** 是为了针对一组示例输入特化一个帧而做出的假设（关于输入的布尔表达式）。只有当这些假设对新输入成立时，重用图才是有效的。

例如，函数的任何常量输入，比如一个字符串，会安装一个守卫，声明该输入应为 `str` 类型且等于我们传递的字符串。运行

```python
import torch

@torch.compile
def fn(a, b):
    return a * len(b)

fn(torch.arange(10), "Hello")
```

并设置 `TORCH_LOGS=guards` 会打印出（以及其他守卫）：

```python
___check_type_id(L['b'], 94334122025024)
L['b'] == 'Hello'
```

这读作“局部变量 `b` 应具有特定类型（本例中为 `str`，由常量 `9433...` 表示）且其值应为 `'Hello'`”。如果我们随后传递不同的参数再次执行该函数：

```python
import torch

@torch.compile
def fn(a, b):
    return a * len(b)

fn(torch.arange(10), "Hello")
fn(torch.arange(10), "Hi")
```

我们可以通过运行 `TORCH_LOGS=recompiles` 看到失败的守卫：

```python
Recompiling function fn in script.py:3
triggered by the following guard failure(s):
     - L['b'] == 'Hello'
```

守卫在[函数的输入被包装到构建器时](https://github.com/pytorch/pytorch/blob/69f112d5867f785a3a090a0c6d6644ae047033ac/torch/_dynamo/variables/builder.py#L808-L810)以及[在程序执行期间](https://github.com/pytorch/pytorch/blob/69f112d5867f785a3a090a0c6d6644ae047033ac/torch/_dynamo/variables/dicts.py#L763-L769)累积。我们将在下一节展示更多守卫的示例，但首先让我们讨论一下源（sources）。

**源** 跟踪如何从进入当前帧时存在的原始局部或全局变量重建一个变量。具体来说，它跟踪原始的局部和全局对象以及它们包含的任何对象。在

```python
def foo(x: Tensor, y: List[Tensor]):
    a = x * y[0]
    return a * x
```

`x` 和 `y` 的源是 [LocalSource](https://github.com/pytorch/pytorch/blob/40dc0580a69565b06ec5263efe5d87cecc8200f7/torch/_dynamo/source.py#L80-L92)，而 `y[0]` 的源是 [GetItemSource](https://github.com/pytorch/pytorch/blob/40dc0580a69565b06ec5263efe5d87cecc8200f7/torch/_dynamo/source.py#L302)，其内部存储了一个 `LocalSource`。另一方面，`a` 将没有源，因为它是一个仅存在于 fx 图中的中间变量。

所有这些都在 [torch/_dynamo/source.py](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/source.py) 中定义。我们可以在以下示例中看到由 `GetItemSource` 生成的守卫：

```python
import torch

@torch.compile
def fn(x, l):
    return x * len(l[0])

fn(torch.randn(8), ["Hi", "Hello"])
```

生成以下守卫：

```python
___check_type_id(L['l'], 94439025877664)
len(L['l']) == 2
___check_type_id(L['l'][0], 94439025840192)
L['l'][0] == 'Hi'
___check_type_id(L['l'][1], 94439025840192)
L['l'][1] == 'Hello'
```

这里，我们看到由 `GetItemSource`（`[0]` 和 `[1]`）包装一个 `LocalSource`（`L['l']`）生成的代码。

至此，通过源和守卫，我们能够实现一个缓存系统来避免重新编译，而无需每次都重新追踪。

细心的读者会注意到，这还没有解释为什么我们需要对 Python 解释器进行如此精细的控制以至于必须重新实现它。我们展示的守卫示例依赖于输入对象，因此我们仍然可以在执行函数之前计算这些守卫。换句话说，我们可以在 `torch.jit.trace` 之上实现这个守卫系统，并以更少的努力获得相同的功能……接下来介绍符号形状。

## 符号形状

我们在引言中讨论的另一点是 Dynamo 知道如何追踪整数。为了实现这一点，我们使用一个符号类 [torch.SymInt](https://github.com/pytorch/pytorch/blob/fb80f05ee2e1cba17892980701bfd5dbce58349f/torch/__init__.py#L244-L249)，它表现得像 `int`，但会记录在其上执行的所有操作到输出的 FX 图中。[^4] 我们在引言中介绍符号整数追踪时已经见过这个类。

现在让我们讨论定义 Dynamo 中符号形状追踪的三个属性，以及如何实现它们。

### 默认静态

Dynamo 默认假设每个整数（无论是输入还是张量的形状）都是静态的。换句话说，在函数的第一次执行中，不会追踪任何整数。然后，只有当它检测到整数或形状在执行过程中改变了值时，才会追踪它并生成针对该变量的通用图。

我们已经在引言中使用整数看到了这种行为。现在让我们看一个使用张量形状的示例。

```python
import torch

@torch.compile
def fn(a, b):
    return a.shape[0] * a * b

fn(torch.randn(4, 3), torch.randn(4, 3))
fn(torch.randn(8, 3), torch.randn(8, 3))
```

使用 `TORCH_LOGS=graph_code` 运行此程序，我们看到这两个调用被追踪为：

```python
def forward(self, l_a_: torch.Tensor, l_b_: torch.Tensor):
    mul = 4 * l_a_
    mul_1 = mul * l_b_
    return (mul_1,)

def forward(self, s0: torch.SymInt, l_a_: torch.Tensor, l_b_: torch.Tensor):
    size = l_a_.size()
    getitem = size[0]
    mul = getitem * l_a_
    mul_1 = mul * l_b_
    return (mul_1,)
```

在第一个图中，形状被追踪为常量，但一旦它改变，就会使用 `SymInt` 符号化地追踪它。通常，查看中间值形状的更简单方法是使用 `TORCH_LOGS=graph_sizes` 运行程序：

```
TRACED GRAPH TENSOR SIZES
===== __compiled_fn_1 =====
l_a_: (s0, 3)
l_a_ (concrete): (8, 3)
l_b_: (s0, 3)
l_b_ (concrete): (8, 3)
mul: (s0, 3)
mul (concrete): (8, 3)
mul_1: (s0, 3)
mul_1 (concrete): (8, 3)
```

我们可以看到两个张量参数的第一个维度是动态的，因为它由 `s0` 变量表示。

我们可以通过运行 `TORCH_LOGS=guards` 来了解 Dynamo 如何实现这一点：

```python
# 第一次调用的守卫
check_tensor(L['a'], torch.float32, device=None, requires_grad=False, size=[4, 3], stride=[3, 1])
check_tensor(L['b'], torch.float32, device=None, requires_grad=False, size=[4, 3], stride=[3, 1])

# 第二次调用的守卫
check_tensor(L['a'], torch.float32, device=None, requires_grad=False, size=[None, 3], stride=[3, 1])
check_tensor(L['b'], torch.float32, device=None, requires_grad=False, size=[None, 3], stride=[3, 1])

L['b'].size()[0] == L['a'].size()[0]
2 <= L['a'].size()[0]
```

我们看到在第一次调用中，守卫检查张量具有某些固定的大小和步幅。这些守卫在第二次执行时失败，因此它重新追踪。由于是一个 `int` 守卫失败，在第二次迭代中，它符号化地追踪这个 `int`，并在这个更通用的内核上安装更通用的守卫。

**编译性能提示**。如果你知道某个维度的大小会变化，可以在调用 `torch.compile` 之前调用 [torch._dynamo.mark_dynamic](https://github.com/pytorch/pytorch/blob/66a76516bfc341b2b55bb2056d2faa9c2de46d69/torch/_dynamo/decorators.py#L176) 将其标记为动态。这将避免第一次使用静态形状进行编译。还有其他有用的实用函数，如 `maybe_mark_dynamic` 或 `mark_static`。你也可以通过调用 `torch.compile(dynamic=True)` 来追踪所有整数和形状。这主要用于调试目的。

### 0 和 1 总是被特化

无论我们是否将维度标记为动态，如果我们传递的输入中该维度为 0 或 1，Dynamo 将将其追踪为非动态，并为其生成特定的图。这就是为什么在上面的示例中，我们会找到形式为 `2 <= L['a'].size()[0]` 的守卫。

选择这一策略有几个原因，其中两个尤为重要：
- 当且仅当张量的任意维度为零时，该张量为空
- 仅当某个步幅为1时，张量才可能是连续的

这一策略决策**不**适用于普通的 Python 整数；如果我们认为某个 Python 整数应动态编译，默认情况下不会对其进行特化；其是否被特化取决于具体使用情况。

### 鸭子塑形

Dynamo 执行我们称之为“鸭子塑形”的操作。如果两个动态整数在追踪时具有相同的值，我们会假设它们相等并据此设置守卫。实际上，这意味着在上文的示例中，我们并非使用两个符号 `s0`、`s1`，而是将它们统一为 `s0`，并设置守卫 `L['b'].size()[0] == L['a'].size()[0]`。这使得编译器能够在执行融合操作的同时，生成足够通用的内核。

### 符号整数的守卫

现在我们已经从高层理解了符号形状的实现方式及其特性。那么，为什么符号形状迫使我们通过控制 CPython 解释器这条复杂路径来实现呢？考虑以下示例：

```python
import torch

@torch.compile(dynamic=True)
def fn(a):
    if a.shape[0] * 2 < 16:
        return a
    else:
        return a + 1

fn(torch.randn(8))
```

这段代码的守卫形式为 `2*L['a'].size()[0] >= 16`。这是基于函数输入的一个非平凡守卫，但它是在程序执行过程中注册的。更重要的是，直到我们看到基于 `SymNodeVariable` 参数的 `if` 语句条件时，我们才知道需要这个守卫。这类条件对 `torch.jit.trace` 是不可见的，需要对 Python 代码进行深度分析。

**调试提示** 使用 `TORCH_LOGS=dynamo` 运行此代码可以告诉我们这个守卫是在何处添加的：

```
eval 2*s0 >= 16 [guard added] at script.py:5 in fn (_dynamo/variables/tensor.py:812 in evaluate_expr)
```

在此处设置断点并查看回溯堆栈对于理解守卫的来源非常有用。

## 完善 Dynamo：图中断

通过我们讨论的所有工具，我们获得了一个能够追踪张量和整数上的 PyTorch 操作的追踪器，并配备了一个缓存系统，该系统知道何时可以重用先前追踪的图以及何时需要重新追踪。所有这些都在执行任意的 Python 代码！

但这存在一个小问题。“执行任意 Python 代码”这一说法可能过于宽泛。Dynamo 实现了 Python 的大部分功能，但它是否实现了更复杂的部分，如协程或异步？它是否实现了整个 Python 标准库？NumPy 也有 Python API。`torch.compile` 是否也能理解 NumPy？以及 Django？[^5]

Python 的生态系统非常庞大，其中很大一部分是用其他更高性能的语言（如 C++ 或 Rust）编写的，并且只暴露了 Python 绑定。Dynamo 无法追踪通过 C++ 实现的 Python 对象。当追踪器遇到它不理解的操作时，它能做什么？

机器学习追踪器处理此问题的常用方法是通知用户它们遇到了无法处理的操作，并完全放弃追踪。对于 PyTorch 来说，这将带来真正的可用性问题，因为其用户习惯于它提供的灵活性。一个现实世界的例子是 `doctr_det_predictor` 模型使用 NumPy 和 `cv2` 库来[对模型结果进行后处理](https://github.com/mindee/doctr/blob/f2114758d529ed8d3d0030581638f0520b6b98d8/doctr/models/detection/core.py#L86)。

这是另一个体现控制 CPython 价值的地方。Dynamo 不是报错退出，而是可以让 CPython 运行有问题的代码！为此，Dynamo 在追踪时生成一个包含有问题代码之前所有操作的图，以及另一个包含之后所有操作的图。[^6] 然后，在运行时，它将委托 CPython 执行第一个图，接着执行有问题的代码，最后执行第二个图。这种停止追踪并生成多个图的过程称为**图中断**。

一个小坦白：我在引言和前面部分一直在说谎。Dynamo 并不生成一个图，而是生成**多个图**！出于所有实际目的，在第二个图之后开始重新追踪可以被视为开始追踪一个新函数。图中断之后的新图将拥有自己的守卫、新的局部变量集等等。

要讨论如何实现图中断，我们首先需要重新审视 Dynamo 如何与 CPython 交互。通过 PEP 523，CPython 允许用户使用自己的帧评估机制。我们尚未讨论的是，CPython 也公开了自己的帧评估机制供他人使用。Dynamo 利用这一点让快速的 CPython 解释器运行编译后的代码。对于一个没有图中断的函数，使用相同参数调用该函数两次的程序，其完整的追踪/执行过程如下所示：

1. 在第一次调用函数时

   1. Dynamo 将函数追踪为 FX 图

      1. FX 图由编译器（Inductor）编译为高效的低级代码……但这又是另一个话题了

   2. 它重写函数的字节码，使其直接调用编译后的函数

   3. 它将这个新的字节码交给 CPython 并要求其运行它 [此处](https://github.com/pytorch/pytorch/blob/e891a3bba9f05697d72776f6e89347231a141f03/torch/csrc/dynamo/eval_frame.c#L1006)

2. 在第二次调用函数时

   1. 它将第一次调用的守卫与新参数进行比对 [此处](https://github.com/pytorch/pytorch/blob/e891a3bba9f05697d72776f6e89347231a141f03/torch/csrc/dynamo/eval_frame.c#L658)。由于参数与之前相同，守卫通过
   2. 它要求 CPython 运行与这些守卫关联的字节码 [此处](https://github.com/pytorch/pytorch/blob/e891a3bba9f05697d72776f6e89347231a141f03/torch/csrc/dynamo/eval_frame.c#L972-L975)

这个过程本身看起来过于复杂。为什么生成新的字节码并让 CPython 运行它，而不是简单地创建一个 C++ 绑定到编译后的函数并执行它？嗯，这种模式允许我们实现图中断！图中断生成的字节码具有以下结构：

1. 执行第一个图的字节码
2. 将栈保持为 CPython 执行第一个图后的状态。它还会重放此时可见的任何对局部或全局变量的修改
3. 导致 Dynamo 图中断的字节码
4. 执行第二个图的字节码

让我们通过一个简单的例子来看看

```python
import torch

@torch.compile
def fn(a):
    b = a + 2
    print("Hi")
    return b + a

fn(torch.randn(4))
```

使用 `TORCH_LOGS=bytecode` 运行此代码会显示初始字节码和修改后的字节码

```python
MODIFIED BYTECODE fn script.py line 3
 0 LOAD_GLOBAL              1 (__compiled_fn_0)
 2 LOAD_FAST                0 (a)
 4 CALL_FUNCTION            1
 6 STORE_FAST               3 (graph_out_0)
 8 LOAD_GLOBAL              0 (print)
10 LOAD_CONST               2 ('Hi')
12 LOAD_FAST                3 (graph_out_0)
14 LOAD_CONST               3 (0)
16 BINARY_SUBSCR
18 STORE_FAST               1 (b)

20 CALL_FUNCTION            1
22 LOAD_GLOBAL              2 (__resume_at_14_1)
24 ROT_TWO
26 LOAD_FAST                0 (a)
28 LOAD_FAST                1 (b)
30 CALL_FUNCTION            3
32 RETURN_VALUE

MODIFIED BYTECODE resume_in_fn script.py line 6
 0 LOAD_GLOBAL              1 (__compiled_fn_2)
 2 LOAD_FAST                2 (b)
 4 LOAD_FAST                1 (a)
 6 CALL_FUNCTION            2
 8 UNPACK_SEQUENCE          1
10 RETURN_VALUE
```

我们可以看到修改后的字节码被分成了两个函数，`fn`（原始函数）和一个名为 `resume_in_fn` 的函数。这第二个函数是 Dynamo 创建的一个函数，用于实现从图中断点开始的程序执行。这通常被称为[续延函数](https://en.wikipedia.org/wiki/Continuation)。这个续延函数只是用正确的参数调用第二个编译后的函数。初始函数的代码被重写，实现了我们之前描述的策略

- L0-4. 调用编译后的函数 (`a + 2`)。
- L6. 将其结果存储在一个名为 `graph_out_0` 的局部变量中。`graph_out_0` 是一个元组
- L8-18. 将栈保持为图中断点的状态
- L20. 执行导致图中断的代码
- L22-32. 调用编译后的续延函数 (`a + b`)

Dynamo 中栈的代码生成委托给了 `VariableTracker` 子类。Dynamo 中的每个 `VariableTracker` 对象都有一个 [reconstruct](https://github.com/pytorch/pytorch/blob/e891a3bba9f05697d72776f6e89347231a141f03/torch/_dynamo/variables/lists.py#L307-L309) 方法，该方法生成必要的字节码以在栈上创建它所代表的 Python 对象。

**调试提示**。图中断会损害性能，因此最好避免它们。使用 `TORCH_LOGS=graph_breaks` 运行程序是找出程序遇到多少图中断的好方法。它返回的信息是基于 `VariableTracker` 对象的，因此上述调试技巧有时也有助于找出导致图中断的原因。

## 结论

Dynamo 是一个复杂的软件。一旦你决定实现一个 CPython 解释器，你就知道这将是一段不平凡的旅程。话虽如此，我们希望这篇文章能帮助揭开它的一些神秘面纱。

Dynamo（大部分）是用 Python 实现的。我们留下了许多指向我们讨论过的代码片段的链接。我们希望阅读这些代码片段，并搜索调用它们的地方，或者在它们上面设置断点并查看调用栈，能帮助理解代码库的其余部分。

当然，学习一个软件如何工作的最好方法是扩展它。在这种情况下，最好的方法是查看 [github 上开放的 dynamo 问题](https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+label%3A%22module%3A+dynamo%22)。其中许多问题只需要对代码进行非常小的更改，一旦你找到需要更改的地方。

## 脚注

以下是本文档中提到的概念的额外细节和参考资料。

[^1]: 在文献中，这被称为有向无环图 (DAG)。

[^2]: 所有这些绑定代码都位于 `torch/csrc/dynamo/eval_frame.c`。

[^3]: 在 CPython 术语中，所有这些对象的集合被称为[一个帧](https://github.com/python/cpython/blob/f26bfe4b25f7e5a4f68fcac26207b7175abad208/Include/internal/pycore_frame.h#L57-L71)。

[^4]: 还有 `SymBool` 和 `SymFloat` 类。在撰写本文时，后者使用得并不多。

[^5]: 有趣的是，它确实理解 NumPy 代码！请查看[这篇博客文章](https://pytorch.org/blog/compiling-numpy-code/)和[文档](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_faq.html#does-numpy-work-with-torch-compile)。现在，这只是因为我们使用 PyTorch 重新实现了 NumPy。不过，祝你好运用 PyTorch 实现 Django……

[^6]: 假设只有一个有问题的代码片段。如果有更多，Dynamo 可以将代码分割成任意多个图。