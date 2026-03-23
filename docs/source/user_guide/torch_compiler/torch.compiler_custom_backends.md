# 自定义后端

## 概述

`torch.compile` 提供了一种简单直接的方法，使用户能够定义自定义后端。

后端函数的约定格式为：
`(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable`。

后端函数可以在 `torch.compile` 的图追踪组件 TorchDynamo 追踪完一个 FX 图后被调用，并且需要返回一个与追踪到的 FX 图等效的已编译函数。返回的可调用对象应与传入后端的原始 `torch.fx.GraphModule` 的 `forward` 函数具有相同的约定：`(*args: torch.Tensor) -> List[torch.Tensor]`。

为了让 TorchDynamo 调用你的后端，请将你的后端函数作为 `torch.compile` 的 `backend` 关键字参数传入。例如：

```python
import torch

def my_custom_backend(gm, example_inputs):
    return gm.forward

def f(...):
    ...

f_opt = torch.compile(f, backend=my_custom_backend)

@torch.compile(backend=my_custom_backend)
def g(...):
    ...
```

更多示例见下文。

## 注册自定义后端

你可以使用 `register_backend` 装饰器来注册你的后端，例如：

```python
from torch._dynamo import register_backend

@register_backend
def my_compiler(gm, example_inputs):
    ...
```

除了 `register_backend` 装饰器，如果你的后端在另一个 Python 包中，你也可以通过 Python 包的入口点来注册你的后端，这为包为另一个包注册插件提供了一种方式。

:::{hint}
你可以在 [Python 打包文档](https://setuptools.pypa.io/en/latest/userguide/entry_point.html) 中了解更多关于 `entry_points` 的信息。
:::

要通过 `entry_points` 注册你的后端，你可以在包的 `setup.py` 文件中将你的后端函数添加到 `torch_dynamo_backends` 入口点组，如下所示：

```python
...
setup(
    ...
    'torch_dynamo_backends': [
        'my_compiler = your_module.submodule:my_compiler',
    ]
    ...
)
```

请将 `=` 前的 `my_compiler` 替换为你后端的名称，并将 `=` 后的部分替换为你后端函数的模块和函数名。安装包后，入口点将被添加到你的 Python 环境中。当你调用 `torch.compile(model, backend="my_compiler")` 时，PyTorch 会首先搜索已通过 `register_backend` 注册的名为 `my_compiler` 的后端。如果未找到，它将继续在所有通过 `entry_points` 注册的后端中搜索。

注册有两个目的：

- 你可以将一个包含后端函数名称的字符串传递给 `torch.compile`，而不是函数本身，例如 `torch.compile(model, backend="my_compiler")`。
- 这是与 [最小化工具](https://docs.pytorch.org/docs/main/torch.compiler_troubleshooting_old.html#minifier) 一起使用所必需的。最小化工具生成的任何代码都必须调用你的注册后端函数的代码，通常通过 `import` 语句。

## AOTAutograd 之后的自定义后端

可以定义由 AOTAutograd 而非 TorchDynamo 调用的自定义后端。这主要有两个原因：

- 用户可以定义支持模型训练的后端，因为 AOTAutograd 可以生成用于编译的反向图。
- AOTAutograd 生成由 [核心 Aten 算子](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_ir.html#core-aten-ir) 组成的 FX 图。因此，自定义后端只需要支持核心 Aten 算子集，这比整个 torch/Aten 算子集要小得多。

使用 `torch._dynamo.backends.common.aot_autograd` 包装你的后端，并像之前一样使用带有 `backend` 关键字参数的 `torch.compile`。由 `aot_autograd` 包装的后端函数应具有与之前相同的约定。

后端函数通过 `fw_compiler`（前向编译器）或 `bw_compiler`（反向编译器）关键字参数传递给 `aot_autograd`。如果未指定 `bw_compiler`，则反向编译函数默认为前向编译函数。

需要注意的一点是，AOTAutograd 要求后端返回的编译函数是“盒装”的。这可以通过使用 `functorch.compile.make_boxed_func` 包装编译函数来实现。

例如：

```python
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func

def my_compiler(gm, example_inputs):
    return make_boxed_func(gm.forward)

my_backend = aot_autograd(fw_compiler=my_compiler)  # bw_compiler=my_compiler

model_opt = torch.compile(model, backend=my_backend)
```

## 示例

### 调试后端

如果你想更好地理解编译过程中发生了什么，可以创建一个自定义编译器（本节中称为后端），它将漂亮地打印从 Dynamo 字节码分析中提取的 fx `GraphModule`，并返回一个 `forward()` 可调用对象。

例如：

```python
from typing import List
import torch
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # 返回一个 Python 可调用对象
@torch.compile(backend=my_compiler)
def fn(x, y):
    a = torch.cos(x)
    b = torch.sin(y)
    return a + b
fn(torch.randn(10), torch.randn(10))
```

运行上述示例会产生以下输出：

```
my_compiler() 调用时传入的 FX 计算图：
操作码         名称    目标                                                  参数        关键字参数
-------------  ------  ------------------------------------------------------  ----------  --------
placeholder    x       x                                                       ()          {}
placeholder    y       y                                                       ()          {}
call_function  cos     <built-in method cos of type object at 0x7f1a894649a8>  (x,)        {}
call_function  sin     <built-in method sin of type object at 0x7f1a894649a8>  (y,)        {}
call_function  add     <built-in function add>                                 (cos, sin)  {}
output         output  output                                                  ((add,),)   {}
```

这对于 `torch.nn.Module` 同样适用，如下所示：

```python
from typing import List
import torch
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # 返回一个可调用的 Python 函数
class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.relu(torch.cos(x))
mod = MockModule()
optimized_mod = torch.compile(mod, backend=my_compiler)
optimized_mod(torch.randn(10))
```

让我们再看一个包含控制流的例子：

```python
from typing import List
import torch
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # 返回一个可调用的 Python 函数
@torch.compile(backend=my_compiler)
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b
for _ in range(100):
    toy_example(torch.randn(10), torch.randn(10))
```

运行此示例会产生以下输出：

```
my_compiler() called with FX graph:
opcode         name     target                                                  args              kwargs
-------------  -------  ------------------------------------------------------  ----------------  --------
placeholder    a        a                                                       ()                {}
placeholder    b        b                                                       ()                {}
call_function  abs_1    <built-in method abs of type object at 0x7f8d259298a0>  (a,)              {}
call_function  add      <built-in function add>                                 (abs_1, 1)        {}
call_function  truediv  <built-in function truediv>                             (a, add)          {}
call_method    sum_1    sum                                                     (b,)              {}
call_function  lt       <built-in function lt>                                  (sum_1, 0)        {}
output         output   output                                                  ((truediv, lt),)  {}

my_compiler() called with FX graph:
opcode         name    target                   args         kwargs
-------------  ------  -----------------------  -----------  --------
placeholder    b       b                        ()           {}
placeholder    x       x                        ()           {}
call_function  mul     <built-in function mul>  (b, -1)      {}
call_function  mul_1   <built-in function mul>  (x, mul)     {}
output         output  output                   ((mul_1,),)  {}

my_compiler() called with FX graph:
opcode         name    target                   args       kwargs
-------------  ------  -----------------------  ---------  --------
placeholder    b       b                        ()         {}
placeholder    x       x                        ()         {}
call_function  mul     <built-in function mul>  (x, b)     {}
output         output  output                   ((mul,),)  {}

最后两个计算图的顺序是不确定的，取决于即时编译器先遇到哪一个。
```

### 高性能后端

集成一个提供卓越性能的自定义后端也很容易，我们将用一个真实的后端 [optimize_for_inference](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html) 来演示：

```python
def optimize_for_inference_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    scripted = torch.jit.script(gm)
    return torch.jit.optimize_for_inference(scripted)
```

然后，您应该能够使用以下方式优化任何现有代码：

```python
@torch.compile(backend=optimize_for_inference_compiler)
def code_to_accelerate():
    ...
```

### 可组合的后端

TorchDynamo 包含许多后端，可以通过 `torch._dynamo.list_backends()` 列出。您可以使用以下代码将这些后端组合在一起：

```python
from torch._dynamo import lookup_backend
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    try:
        trt_compiled = lookup_backend("tensorrt")(gm, example_inputs)
        if trt_compiled is not None:
            return trt_compiled
    except Exception:
        pass
    # 第一个后端失败，尝试其他方案...
    try:
        inductor_compiled = lookup_backend("inductor")(gm, example_inputs)
        if inductor_compiled is not None:
            return inductor_compiled
    except Exception:
        pass
    return gm.forward
```