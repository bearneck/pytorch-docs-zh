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

# 非严格追踪编程模型

**摘要：**
- **非严格追踪** 是一种追踪 Python 代码的方式，它比 Dynamo 的限制更少，但可能导致静默错误。
- 非严格追踪会运行一个 Python 函数，并利用 Python 和 PyTorch 的操作符重载能力，将执行过程中发生的张量操作记录到追踪图中。
- 如果一个函数符合某些约束条件，即该函数是**纯函数**且不直接操作 `Tensor.data_ptr()`，那么它就是**可非严格追踪的**。
- 非严格追踪可能会**特化**某些变量，并将它们视为**常量**，将这些变量的值固化到追踪图中。

`torch.compile` 的内部机制（`make_fx`、AOTDispatcher）使用**非严格追踪**。 [`torch._dynamo.nonstrict_trace`](programming_model.dynamo_nonstrict_trace) 也可以在 `torch.compile` 编译的代码中用于标记使用非严格追踪进行追踪的代码段。
非严格追踪会运行一个 Python 函数，并利用 Python 和 PyTorch 的操作符重载能力，将执行过程中发生的张量操作记录到追踪图中。

**`make_fx`** 是非严格追踪的主要入口点。对于以下函数，在执行输入时只走了上面的分支，因此它捕获的图只包含该分支。

```{code-cell}
from torch.fx.experimental.proxy_tensor import make_fx
def f(x):
    if x.shape[0] > 2:
        return x ** 2 / 6
    else:
        return x * 3
x = torch.randn(3)
gm = make_fx(f, tracing_mode="fake")(x)
gm.print_readable()
```

非严格追踪与 Dynamo（严格）追踪的不同之处在于**它是不安全的**，也就是说，给定一个函数，它捕获的张量操作图可能具有与原始函数不同的语义。
给定一个 Python 函数，Dynamo 追踪会捕获一个张量操作图和剩余的字节码，当它们组合起来时，其语义与原始 Python 函数相同。

(programming_model.non_strict_tracing_model.pure_functions)=

## 纯函数

非严格追踪仅在**纯函数**上是可靠的，因此只有纯函数才应该进行非严格追踪。

纯函数是具有以下属性的函数：

- **确定性。** 给定相同的输入，纯函数总是返回相同的输出。
- **无副作用。** 纯函数没有任何副作用，例如修改外部状态或执行 I/O 操作。
- **显式输入/输出。** 所有输入数据必须通过函数参数传递，所有输出都从函数返回。

以下是一些非纯函数的示例，对于这些函数，捕获的图的行为与原始函数不同。

### 示例 1：无显式输入（例如访问全局张量）
```{code-cell}
var = torch.tensor(1)
def function_with_global_access(y):
    return y + var
x = torch.tensor([0, 1, 2])
# 为了演示目的，需要 _allow_non_fake_inputs=True 来捕获全局变量。
gm = make_fx(
    function_with_global_access, tracing_mode="fake", _allow_non_fake_inputs=True
)(x)
# 非严格追踪捕获了全局变量的值 (1.)
print("1. 调用函数", function_with_global_access(x))
print("1. 调用图", gm(x))
# 然而，在更改全局变量后，捕获的图
# 产生的结果与原始函数不同
var = torch.tensor(2)
print("2. 调用函数", function_with_global_access(x))
print("2. 调用图", gm(x))
# 要捕获一个可以包含变化的 `var` 张量的图，
# 它必须是一个显式输入：
def function_fixed(y, var):
    return y + var
var = torch.tensor(3)
gm = make_fx(function_fixed, tracing_mode="fake")(x, var)
print("3. 调用函数", function_fixed(x, var))
print("3. 调用图", gm(x, var))
var = torch.tensor(4)
print("4. 调用函数", function_fixed(x, var))
print("4. 调用图", gm(x, var))
```

关于原因的解释，请参见 [特化与常量](specialization-and-constants)。

### 示例 2：副作用（打印）

```{code-cell}
def function_with_side_effect(y):
    print(y)
x = torch.tensor([0, 1, 2])
_ = function_with_side_effect(x)
```

在 Python 中运行 `f` 会打印一个张量作为副作用。

```{code-cell}
gm = make_fx(function_with_side_effect, tracing_mode="fake")(x)
```

在非严格追踪期间，这个打印操作发生在图捕获过程中。

```{code-cell}
_ = gm(x)
```

该图没有存储对 `print` 语句的调用，因此执行该图不会打印任何内容。

### 示例 3：副作用（输入列表修改）

```{code-cell}
lst = []
def function_with_input_list_mutation(lst):
    val = lst.pop()
    return val
x = torch.tensor([0, 1, 2])
y = torch.tensor([0, 1, 2])
# 每次执行函数时，列表的大小都会缩小
lst = [x, y]
function_with_input_list_mutation(lst)
print("一次调用后 len(lst)", len(lst))
function_with_input_list_mutation(lst)
print("两次调用后 len(lst)", len(lst))
# 使用非严格追踪，列表的长度在图捕获期间会缩小，
# 但在图的调用中不会。
lst = [x, y]
gm = make_fx(function_with_input_list_mutation, tracing_mode="fake")(lst)
print("图捕获后 len(lst)", len(lst))
gm(lst)
print("一次调用图后 len(lst)", len(lst))
gm(lst)
print("两次调用图后 len(lst)", len(lst))
```

### 禁止直接操作 data_ptr
直接操作 `Tensor.data_ptr` 是不可非严格追踪的。其背后的直觉是 PyTorch 无法判断你*如何*操作了 `data_ptr`。

```{code-cell}
import ctypes
# 创建一个包含单个元素的张量
tensor = torch.tensor([42], dtype=torch.int32)  # 为简化使用 int32
def function_with_data_ptr(tensor):
    # 获取数据指针
    ptr = tensor.data_ptr()
    # 将指针转换为 ctypes 指针
    ctypes_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int32))
    # 递增指针处的值
    ctypes_ptr.contents.value += 1
    return tensor
try:
    make_fx(function_with_data_ptr, tracing_mode="fake")(tensor)
except Exception as e:
    print(e)
```

(specialization-and-constants)=
## 特化与常量

非严格追踪会捕获一个可能在某些值上特化的计算图。这意味着捕获的计算图仅对这些值有效。我们说该计算图将这些值视为**常量**。

在非严格追踪过程中，所有非张量变量都被视为常量：

```{code-cell}
def f(x, y):
    return x + y
x = torch.tensor([0, 1, 2])
y = 3.14
gm = make_fx(f, tracing_mode="fake")(x, y)
gm.print_readable()
```

3.14 在计算图中是一个常量。

非严格追踪还会对输入张量的属性进行特化：

```{code-block}
def f(x):
    if x.shape[0] > 2:
        return x ** 2 / 6
    else:
        return x * 3
x = torch.randn(3)
gm = make_fx(f, tracing_mode="fake")(x)
gm.print_readable()
```

同时它也会对未直接传入函数的任何变量进行特化：

```{code-cell}
var = torch.tensor(1)
def f(x):
    return x + y
x = torch.randn(3)
gm = make_fx(f, tracing_mode="fake")(x)
gm.print_readable()
```