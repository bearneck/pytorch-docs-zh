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
from compile import header_code

torch._logging.set_logs(graph_breaks=True, graph_code=True)
```

(dynamic_shapes)=
# 动态形状

本节解释了如何在 PyTorch 中处理动态形状，包括如何调试和修复常见错误、在算子中实现对动态形状的支持，以及理解其底层机制。

动态形状允许 PyTorch 模型处理具有不同维度的输入而无需重新编译。这使得模型更加灵活，能够在单个编译产物中处理不同的批次大小、序列长度或图像尺寸。动态形状通过符号化追踪张量维度而非使用具体值来实现，创建一个在运行时适应不同输入形状的计算图。默认情况下，PyTorch 假定所有输入形状都是静态的。

通常，深度学习编译器仅支持静态形状，输入形状变化时需要重新编译。虽然这种方法覆盖了许多用例，但在某些情况下是不够的：

- **可变维度** - 批次大小或序列长度变化，例如在自适应批处理中。
- **数据相关输出** - 模型根据输入数据产生输出，例如检测模型中的可变边界框。
- **稀疏表示** - 处理依赖于数据变化的稀疏结构，例如稀疏张量、不规则张量和图神经网络。

动态形状不支持动态秩的程序，即输入张量维度发生变化的程序，因为这不常见且会带来不必要的复杂性。

## 尺寸/整数为动态意味着什么？

动态形状通过使某些维度或整数变为动态来避免重新编译。例如，如果一个函数 `f(x)` 使用静态尺寸编译，它将需要为不同尺寸重新编译：

```{note}
为简单起见，此示例使用了 `@torch.compile(dynamic=True)`。请注意，由于此选项容易出错，不推荐使用。
关于启用动态形状的推荐方法，请参见 {ref}`enable-dynamic-behavior`。
```

```{code-cell}
import torch
@torch.compile(dynamic=False)
def f(x):
     return x* x.size()[0]

f(torch.rand(10))
f(torch.rand(20))
f(torch.rand(30))
f(torch.rand(40))
```

在生成的输出中，您可以看到创建了四个图。
请参阅相应的 <a href="../../_static/img/dynamic_shapes/tlparse1_dynamic_shapes_false.png" target="_blank">tlparse 输出</a>

通过使尺寸变为动态，该函数可以处理各种尺寸而无需重新编译：

```{code-cell}
import torch
@torch.compile(dynamic=True)
def f(x):
     return x* x.size()[0]

f(torch.rand(10))
f(torch.rand(20))
f(torch.rand(30))
f(torch.rand(40))
```

启用动态形状后，只创建了一个图。请参阅
相应的 <a href="../../_static/img/dynamic_shapes/tlparse2_dynamic_shapes_true.png" target="_blank">tlparse 输出</a>。

虽然对于这个小例子，编译时间差异很小，但更复杂的用例将显示出显著的性能改进。

(what_is_a_specialization)=
## 什么是特化？

**特化** 指的是通过检查控制流中的形状条件，为特定输入形状优化计算图。如果基于形状条件进入某个分支，则该图将针对该条件进行定制。如果新的输入不满足此条件，系统将重新编译该图。

特化允许您为特定的输入形状创建优化的计算图，这可以显著提高执行速度。

```{code-cell}
import torch
@torch.compile(dynamic=True)
def f(x):
    if x.size()[0] == 10:
        return x * 10

    if x.size()[0] <= 30:
        return x*200

    return x*x.size()[0]

f(torch.rand(10))
f(torch.rand(20))
f(torch.rand(30))
f(torch.rand(40))
f(torch.rand(50))
```

在上面的代码中，我们特化了图要求输入尺寸为 10 的情况，在这种情况下它将返回 `x * 10`。如果输入尺寸小于 30，它将返回 `x * 200`。在输出中，您可以看到这创建了三个图。

请参阅相应的 <a href="../../_static/img/dynamic_shapes/tlparse3_specialization.png" target="_blank">tlparse 输出</a>

这是为上述函数创建的图：

```{image} ../../_static/img/dynamic_shapes/dynamic_shapes_example_specialization.png
```

(enable-dynamic-behavior)=
## 启用动态行为

有以下几种方法可以使事物变为动态：

* {ref}`automatic_dynamic`
* {ref}`user_annotations` (推荐)
* {ref}`torch_compile_dynamic_true` (仅用于测试)
* {ref}`dynamic_shapes_advanced_control_options` (用于高级用例)

请阅读下面关于每个选项的说明。

(automatic_dynamic)=
### 自动动态

**自动动态** 是默认行为，其中 {func}`torch.compile` 在首次编译时假定使用静态形状，同时跟踪该次编译的输入尺寸。当触发重新编译时，它使用此信息来识别哪些维度已发生变化，并在第二次编译中将它们标记为动态。

(user_annotations)=
### 用户注解

有几个 API 允许用户通过名称或代码显式地将特定输入标记为动态。这对于避免初始编译（这些编译最终会因之前的工具而变为动态）很有用。它也用于标记那些不会自动被标记为动态的元素，例如神经网络模块参数等。用户注解是启用动态形状的首选方法。

#### `mark_dynamic(tensor, dim, min=min, max=max)`

> ⚠️ **警告**
>
> 切勿在 `torch.compile()` 正在编译的任何函数内部调用 `torch._dynamo.mark_dynamic()`（例如，模型的 `forward()` 方法或其调用的任何函数）。
>
> 此函数是一个*追踪时 API*。如果从已编译的代码内部调用它，Dynamo 将引发类似以下的错误：
>
> ```
> AssertionError: Attempt to trace forbidden callable
> ```
>
> **正确的用法**是在调用 `torch.compile` *之前*对输入张量调用 `mark_dynamic`，例如：
>
> ```python
> torch._dynamo.mark_dynamic(x, 0)
> compiled_model = torch.compile(model)
> ```

{func}`torch._dynamo.mark_dynamic` 函数将张量的一个维度标记为动态，如果该维度被特化，则会失败。它不适用于整数。仅当您知道使用此输入的所有计算图都收敛到单个动态计算图时，才使用此函数。否则，您可能会遇到具有误导性的约束违反错误。在这种情况下，请考虑使用 {func}`torch._dynamo.maybe_mark_dynamic`。目前，{func}`torch._dynamo.mark_dynamic` 的优先级不高于 `force_parameter_static_shapes = True` 或 `force_nn_module_property_static_shapes = True`。

如果您事先知道某个特定维度将是动态的，可以使用 {func}`torch._dynamo.mark_dynamic(tensor, dim)` 来避免初始重新编译。此外，如果您已经知道此维度的最小和最大可能值，可以使用 {func}`torch._dynamo.mark_dynamic(tensor, dim, min=min, max=max)` 来指定它们。

以下是一个简单的示例：

```{code-cell}
import torch

@torch.compile
def f(x):
    return x * x.size()[0]

x = torch.randn(10)
torch._dynamo.mark_dynamic(x, 0)

# 第一次调用时我们提供一个标记为动态的张量
f(x)
# 后续的这些调用将使用动态编译的代码
f(torch.randn(20))
f(torch.randn(30))
f(torch.randn(40))
```

#### `maybe_mark_dynamic(tensor, dim)`

{func}`torch._dynamo.maybe_mark_dynamic` 函数与 {func}`torch._dynamo.mark_dynamic` 共享所有属性，但如果尺寸被特化，它不会失败。当输入被多个计算图共享，或者特定框架下的计算图数量不收敛到一时，请使用此函数。例如，在上面的示例中，使用 {func}`torch._dynamo.maybe_mark_dynamic()`，因为尺寸为 0 和 1 的计算图将被特化。但是，您可以使用 {func}`torch._dynamo.mark_dynamic` 来确保永远不会特化。

#### `mark_unbacked(tensor, dim)`

{func}`torch._dynamo.decorators.mark_unbacked` 函数将张量的一个维度标记为无支撑维度。这不太可能是您需要的工具，但如果特化发生在条件 `guard_size_oblivious(x)` 内部，并且使用它可以消除特化，那么它可能有用。请确保它能修复特化问题，并且不会引入数据相关的错误，该错误会在您试图避免的特化位置或之前转换为计算图中断。使用下一个选项可能更好。

(dynamic_sources_allow_list)=
#### 动态允许列表 (`DYNAMIC_SOURCES`)

使用环境变量 `TORCH_COMPILE_DYNAMIC_SOURCES` 传递一个配置列表，其中包含要标记为动态的源名称。例如：`TORCH_COMPILE_DYNAMIC_SOURCES=L[‘x’],L[‘y’]`。使用 `tlparse` 中的 PGO 工件最容易找到这些动态源名称。您可以从 PGO 工件中复制并粘贴动态源名称。此方法适用于整数和张量尺寸，并且优先级高于所有其他强制静态形状的标志。如果标记为动态的内容被特化，或者提供的输入不存在，它不会抛出错误。

以下是一个示例：

```{code-cell}
import torch

@torch.compile()
def f(x):
     return x * x.size()[0]

with torch.compiler.config.patch(dynamic_sources="L['x']"):
    f(torch.rand(10))
f(torch.rand(20))
f(torch.rand(30))
f(torch.rand(40))
```

(torch.compiler.set_stance_eager_then_compile)=
#### `torch.compiler.set_stance ("eager_then_compile")`

有时，确定哪些输入应标记为动态可能具有挑战性。如果您愿意为第一批数据接受性能成本，另一个便捷的选择是使用 `eager_then_compile` 姿态，它会自动为您确定动态输入。更多信息，请参阅 {func}`torch.compiler.set_stance` 和 [使用 torch.compiler.set_stance 进行动态编译控制](https://docs.pytorch.org/tutorials/recipes/torch_compiler_set_stance_tutorial.html)。

(torch_compile_dynamic_true)=
### `torch.compile (dynamic=true)`（不推荐）

此设置强制所有尺寸和整数为动态，增加了遇到动态形状错误的可能性。由于容易出错，不建议设置此选项。它会使每个输入尺寸都变为动态，这可能导致性能下降并最终增加编译时间。

PyTorch 还为动态形状提供了高级控制选项，请参阅：{ref}`dynamic_shapes_advanced_control_options`。

## 接下来该做什么？

如果您遇到框架代码错误或特化问题，请提交问题以便审查和改进。如果问题出现在您的用户代码中，请考虑是否愿意重写代码以避免该问题。确定它是否影响正确性，或者是否是冗余检查。如果问题涉及带有 `constexpr` 参数的 Triton 自定义内核，请评估是否可以重写它以解决问题。

```{toctree}
:maxdepth: 1
compile/dynamic_shapes_core_concepts
compile/dynamic_shapes_troubleshooting
compile/dynamic_shapes_advanced_control_options
compile/dynamic_shapes_beyond_the_basics
```

```{seealso}
* [tlparse 文档](https://github.com/pytorch/tlparse)
* [动态形状手册](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit?tab=t.0#heading=h.fh8zzonyw8ng)
```