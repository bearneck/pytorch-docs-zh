# Dynamo 概述

在阅读本节之前，请先阅读 *torch.compiler_overview*。

TorchDynamo（简称 Dynamo）是一个 Python 级别的即时（JIT）编译器，旨在加速未经修改的 PyTorch 程序。Dynamo 通过挂钩 CPython 中的帧评估 API（[PEP 523](https://peps.python.org/pep-0523/)），在 Python 字节码即将执行前动态修改它。它将 Python 字节码重写，以提取 PyTorch 操作序列到一个 [FX Graph](https://pytorch.org/docs/stable/fx.html) 中，然后使用可定制的后端进行编译。它通过字节码分析创建这个 FX Graph，并设计为将 Python 执行与编译后端混合，以兼顾两者的优势——可用性和性能。

Dynamo 使得尝试不同的编译器后端变得容易，只需一行装饰器 `torch._dynamo.optimize()` 即可加速 PyTorch 代码，为方便起见，它被 `torch.compile()` 包装。

下图展示了 PyTorch 在使用 `torch.compile` 和不使用时的运作方式：

```{image} ../../_static/img/dynamo/TorchDynamo.png
```

`TorchInductor` 是 [Dynamo Graph](https://pytorch.org/docs/stable/fx.html) 支持的后端之一，它将图转换为 GPU 的 [Triton](https://github.com/openai/triton) 或 CPU 的 [C++/OpenMP](https://www.openmp.org/)。我们有一个[训练性能仪表板](https://github.com/pytorch/torchdynamo/issues/681#issuecomment-1233828468)，提供了不同训练后端的性能比较。你可以在 [PyTorch dev-discuss 上的 TorchInductor 帖子](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)中阅读更多内容。

如需深入了解，请阅读以下部分，观看深度解析视频，并查看 dev-discuss 主题。

- [Dynamo 深度解析视频](https://www.youtube.com/watch?v=egZB5Uxki0I)
- [dev-discuss 主题](https://dev-discuss.pytorch.org/search?q=TorchDynamo%20order%3Alatest)

## Dynamo 内部原理

**作者**: [Jason Ansel](https://github.com/jansel) 和 [Kaichao You](https://github.com/youkaichao)

本节将介绍一些 Dynamo 的内部原理，并展示 Dynamo 在底层是如何工作的。

### 什么是守卫？

Dynamo 即时运行，并根据动态属性特化图。以下是一个如何使用 Dynamo 的基本示例。可以使用 `torchdynamo.optimize` 装饰一个函数或方法以启用 Dynamo 优化：

```python
from typing import List
import torch
from torch import _dynamo as torchdynamo
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable

@torchdynamo.optimize(my_compiler)
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b
for _ in range(100):
    toy_example(torch.randn(10), torch.randn(10))
```

例如，上面的第一个图有以下守卫：

```
GUARDS:
hasattr(L['a'], '_dynamo_dynamic_indices') == False
hasattr(L['b'], '_dynamo_dynamic_indices') == False
utils_device.CURRENT_DEVICE == None
___skip_backend_check() or ___current_backend() == ___lookup_backend(140355900538256)
check_tensor(L['a'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[10], stride=[1])
check_tensor(L['b'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[10], stride=[1])
```

如果任何守卫失败，图将被重新捕获和重新编译。其中有趣的守卫是 `check_tensor`，它检查以下 `torch.Tensor` 属性：

- 张量的 Python 类（张量子类化等）
- dtype
- device
- requires_grad
- dispatch_key（应用了线程局部的包含/排除）
- ndim
- 尺寸\*
- 步幅\*

完全特化模式允许后端编译器假设一个完全静态的图。不幸的是，大多数后端需要这个。在非动态形状模式下，返回动态形状的操作符将触发图中断。

### Dynamo 在做什么？

如果你想更好地理解 Dynamo 在做什么，可以运行以下代码：

```
TORCH_LOGS="+dynamo,guards,bytecode"
```

如果你不熟悉 Python 字节码，可以添加一个反编译钩子，将字节码反编译为人类可读的源代码。一个可用的工具是 [depyf](https://github.com/youkaichao/depyf)。如果尚未安装 `depyf`，请运行 `pip install depyf`。然后，在运行任何代码之前添加以下代码以安装反编译钩子。

```python
import depyf
depyf.install()
```

这段代码会触发有用（但可能冗长）的打印输出。

例如，`toy_example` 中第一个图的打印输出是：

__compiled_fn_0 <eval_with_key>.1
操作码       名称      目标                                                  参数              关键字参数
-------------  -------  ------------------------------------------------------  ----------------  --------
placeholder    a        a                                                       ()                {}
placeholder    b        b                                                       ()                {}
call_function  abs_1    <built-in method abs of type object at 0x7f9ca082f8a0>  (a,)              {}
call_function  add      <built-in function add>                                 (abs_1, 1)        {}
call_function  truediv  <built-in function truediv>                             (a, add)          {}
call_method    sum_1    sum                                                     (b,)              {}
call_function  lt       <built-in function lt>                                  (sum_1, 0)        {}
output         output   output                                                  ((truediv, lt),)  {}
原始字节码 toy_example example.py 第 12 行
 14           0 LOAD_FAST                0 (a)
              2 LOAD_GLOBAL              0 (torch)
              4 LOAD_METHOD              1 (abs)
              6 LOAD_FAST                0 (a)
              8 CALL_METHOD              1
             10 LOAD_CONST               1 (1)
             12 BINARY_ADD
             14 BINARY_TRUE_DIVIDE
             16 STORE_FAST               2 (x)
 15          18 LOAD_FAST                1 (b)
             20 LOAD_METHOD              2 (sum)
             22 CALL_METHOD              0
             24 LOAD_CONST               2 (0)
             26 COMPARE_OP               0 (<)
             28 POP_JUMP_IF_FALSE       19 (to 38)
 16          30 LOAD_FAST                1 (b)
             32 LOAD_CONST               3 (-1)
             34 BINARY_MULTIPLY
             36 STORE_FAST               1 (b)
 17     >>   38 LOAD_FAST                2 (x)
             40 LOAD_FAST                1 (b)
             42 BINARY_MULTIPLY
             44 RETURN_VALUE
修改后的字节码 toy_example example.py 第 12 行
 12           0 LOAD_GLOBAL              3 (__compiled_fn_0)
              2 LOAD_FAST                0 (a)
              4 LOAD_FAST                1 (b)
              6 CALL_FUNCTION            2
              8 UNPACK_SEQUENCE          2
             10 STORE_FAST               2 (x)
             12 POP_JUMP_IF_FALSE       12 (to 24)
             14 LOAD_GLOBAL              4 (__resume_at_30_1)
             16 LOAD_FAST                1 (b)
             18 LOAD_FAST                2 (x)
             20 CALL_FUNCTION            2
             22 RETURN_VALUE
        >>   24 LOAD_GLOBAL              5 (__resume_at_38_2)
             26 LOAD_FAST                1 (b)
             28 LOAD_FAST                2 (x)
             30 CALL_FUNCTION            2
             32 RETURN_VALUE
可能的源代码：
def toy_example(a, b):
    __temp_1 = __compiled_fn_0(a, b)
    x = __temp_1[0]
    if __temp_1[1]:
        return __resume_at_30_1(b, x)
    return __resume_at_38_2(b, x)
如果您发现反编译的代码有误，请在 https://github.com/youkaichao/depyf/issues 提交问题。

顶部您可以看到 FX 图。
接下来，您可以看到函数的原始字节码，然后是 Dynamo 生成的修改后的字节码，以及供参考的反编译源代码。最后，您可以看到我们上面介绍过的守卫。

在修改后的字节码中，`__compiled_fn_0` 是 `my_compiler()`（已编译的图）的返回值。`__resume_at_30_1` 和 `__resume_at_38_2` 都是生成的延续函数，用于在图断点（在字节码偏移量 30 和 38 处）之后恢复执行。这些函数的形式如下：

```
__resume_at_<偏移量>:
    ... 如果需要则恢复栈状态 ...
    JUMP_ABSOLUTE <偏移量> 进入 toy_example
    ... toy_example 的原始字节码 ...
```

通过生成这个 `resume_at` 函数，我们强制函数的剩余部分在一个新的 Python 帧中执行，该帧会递归地触发 Dynamo，在首次执行到达该点时重新开始其捕获。

### 如何检查 Dynamo 生成的工件？

要检查 Dynamo 生成的工件，有一个 API `torch._dynamo.eval_frame._debug_get_cache_entry_list`，可以从函数的 `__code__` 对象中检索已编译的代码和守卫。一个已编译的函数可以有多个缓存条目，每个缓存条目包含一个用于检查守卫的生成函数，以及一个 `types.CodeType` 对象来保存满足守卫条件时要执行的代码。

```python
from torch._dynamo.eval_frame import _debug_get_cache_entry_list, innermost_fn
cache_entries = _debug_get_cache_entry_list(innermost_fn(toy_example))
cache_entry = cache_entries[0]
guard, code = cache_entry.check_fn, cache_entry.code
# 守卫函数接收输入帧的局部变量，并判断是否应触发重新编译。
import dis
dis.dis(guard)
dis.dis(code)
```

如果您了解 Python 字节码，就能理解上述输出。

对于守卫函数，无需检查其字节码。我们可以直接访问其守卫条件：

```python
for code_part in guard.code_parts:
    print(code_part)
```

输出是：

```
___guarded_code.valid
___check_global_state()
hasattr(L['a'], '_dynamo_dynamic_indices') == False
hasattr(L['b'], '_dynamo_dynamic_indices') == False
utils_device.CURRENT_DEVICE == None
___skip_backend_check() or ___current_backend() == ___lookup_backend(140215810860528)
___check_tensors(L['a'], L['b'], tensor_check_names=tensor_check_names)
```

只有当所有条件都满足时，守卫函数才返回 true，并执行已编译的代码。

对于已编译的代码，我们无法直接访问其源代码，必须进行反编译。

```python
from depyf import decompile
print(decompile(code))
```

输出是：

```python
def toy_example(a, b):
    __temp_1 = __compiled_fn_0(a, b)
    x = __temp_1[0]
    if __temp_1[1]:
        return __resume_at_30_1(b, x)
    return __resume_at_38_2(b, x)
```

代码中引用的一些名称包括：

- 已编译的函数，存储在包含原始函数 `toy_example` 的模块的全局命名空间中。这些名称包括 `__compiled_fn_0` / `__resume_at_30_1` / `__resume_at_38_2`。
- 用于检查防护条件的闭包变量。这些名称可以从 `guard.__code__.co_freevars` 访问，其值存储在 `guard.__closure__` 中。这些名称包括 `___guarded_code` / `___is_grad_enabled` / `___are_deterministic_algorithms_enabled` / `___is_torch_function_enabled` / `utils_device` / `___check_tensors` / `tensor_check_names`。
- `guard` 函数的参数 `L`。这是一个将 `toy_example` 的参数名称映射到其值的字典。这仅在函数被调用时可用，此时会用到帧求值 API。简而言之，`L` 是一个结构为 `{'a': value_a, 'b': value_b}` 的 `dict`。因此，你可以看到代码使用 `L['a']` 来引用输入变量 `a`。

图中断体现在已编译的 `toy_example` 代码中，我们必须使用 Python 解释器来选择要执行的后续图。

请注意，我们传递了一个简单的 `my_compiler` 函数作为后端编译器，因此子图代码 `__resume_at_38_2`、`__resume_at_30_1` 和 `__compiled_fn_0` 仍然是 Python 代码。这也可以被检查（请忽略函数名，只关注函数签名和函数体代码）：

```python
print("source code of __compiled_fn_0:")
print(innermost_fn(__compiled_fn_0).__self__.code)
print("=" * 60)
print("source code of __resume_at_30_1:")
print(decompile(__resume_at_30_1))
print("=" * 60)
print("source code of __resume_at_38_2:")
print(decompile(__resume_at_38_2))

source code of __compiled_fn_0:
def forward(self, L_a_ : torch.Tensor, L_b_ : torch.Tensor):
    l_a_ = L_a_
    l_b_ = L_b_
    abs_1 = torch.abs(l_a_)
    add = abs_1 + 1;  abs_1 = None
    truediv = l_a_ / add;  l_a_ = add = None
    sum_1 = l_b_.sum();  l_b_ = None
    lt = sum_1 < 0;  sum_1 = None
    return (truediv, lt)
# 要查看更多调试信息，请使用 ``graph_module.print_readable()``
============================================================
source code of __resume_at_30_1:
def <resume in toy_example>(b, x):
    b = b * -1
    return x * b
============================================================
source code of __resume_at_38_2:
def <resume in toy_example>(b, x):
    return x * b
```

然而，如果我们使用其他后端，例如内置的 `inductor`，子图代码将被编译为用于 GPU 的 CUDA 内核或用于 CPU 的 C++ 代码。

总而言之，编译后的代码在概念上等同于以下代码：

```python
def compiled_example(a, b):
    L = {'a': a, 'b': b}
    for guard, code in get_cache_entries():
        if guard(L):
            return code(a, b)
    recompile_and_add_another_cache_entry()
```

下图展示了 `torch.compile` 如何转换和优化用户编写的代码：它首先从用户编写的函数中提取计算图，然后将这些图编译成优化后的函数，最后将它们组装成一个新函数，该函数在功能上等同于用户编写的代码，但经过优化以获得良好的计算速度。

```{image} ../../_static/img/dynamo/flowchart.jpg
```

要了解更多关于所有这些如何在内部实现的信息，请参阅 *torch.compiler_dynamo_deepdive*。
