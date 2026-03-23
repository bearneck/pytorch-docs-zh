# 排查 GuardOnDataDependentSymNode 错误
当处理包含未绑定符号的 PyTorch 模型时，这些符号可能来自数据依赖操作（如 `item()`、`tolist()` 或 `nonzero()`），或来自使用 `torch._dynamo.decorators.mark_unbacked` 手动将某些输入尺寸标记为动态时，您可能会遇到 `GuardOnDataDependentSymNode` 错误。本节将解释这些错误是什么以及如何修复它们。

## 背景：

**已绑定动态形状** 是作为 PyTorch 2 中“无限重新编译”问题的解决方案而出现的。当像 `torch.ones(x)` 这样的函数在 `x=10` 时被编译，如果没有动态形状，Dynamo 会插入一个守卫来检查“输入 x 恰好为 10”，并生成一个针对尺寸 10 硬编码的图。使用 `x=20` 调用会触发另一次编译，依此类推。

为了解决这个问题，可以使用动态形状来停止硬编码尺寸，并以符号方式表示它们。然而，编译器仍然需要做出分支决策（例如 `if x < 1024`），因此我们为每个动态形状提供了一个“提示”；即编译期间使用的示例输入的具体值。该提示指导分支选择，Dynamo 会添加守卫以确保分支条件保持有效。这些被称为*已绑定*（或*可守卫*）形状，因为它们有提示支持，并且可以有约束它们的守卫。

**未绑定动态形状** 源于不同的需求：支持数据依赖操作，如 `x.item()`。对于此类操作，输出值取决于张量数据，并且在编译时未知。最初，这些操作会触发图中断，但这对于导出和性能来说是有问题的。为了将数据依赖操作保留在图中，我们以符号方式表示它们的输出——但与已绑定形状不同，我们没有提示来解决分支问题。这些被称为*未绑定*（或*无守卫*）形状。随着时间的推移，用户也故意为主图输入选择未绑定形状，以避免分支引起的重新编译，并编译适用于所有输入形状的图。

### 数据依赖错误

未绑定形状的一个关键挑战是处理分支：没有提示，编译器无法确定走哪条路径，默认行为是抛出 `GuardOnDataDependentSymNode` 错误。

## 框架代码与用户代码错误

数据依赖错误（DDEs）可能来自两个来源：**框架代码**（PyTorch 内部）和**用户代码**（您的模型）。历史上，DDEs 是一个主要的痛点——特别是对于导出用户——因为许多常见的框架操作，如重塑、切片、缩小、选择、连续性检查和广播检查，在遇到未绑定形状时会触发这些错误。

**框架代码不应再抛出 DDEs。** 我们已在整个 PyTorch 框架中实现了显式的未绑定语义，解决了主要的代码分支，并消除了绝大多数源自框架的 DDEs。以前失败的操作——如 `view`、`narrow`、`select` 和各种形状检查——现在通过自动选择适用于所有输入值的通用代码路径（有时可能偏离急切语义）来正确处理未绑定形状。这意味着您现在可以更可靠地捕获无特化的图，而不会遇到框架 DDEs。

如果您遇到源自 PyTorch 框架代码的 DDE（可通过错误消息中的“Potential framework code culprit”指向 `torch/` 下的文件来识别），这很可能是一个应该报告的 bug，并且可以使用本文档后面解释的相同方法进行修复。

请注意，有些操作本质上不友好于未绑定形状，因为它们需要知道动态形状的确切值。您可能遇到的 DDEs 通常源自**用户代码**——您的模型中依赖于数据依赖值的分支。

本文档的其余部分将解释如何处理代码中的未绑定形状。解决方案通常分为两类：

1. **通过重写代码使其具有弹性来避免 DDE** —— 重构您的代码，使其不需要在未绑定符号上进行分支，或者使用能够优雅处理未绑定形状的替代 API。
2. **使用 `torch._check` 提供提示** —— 当重写不可行时，用于向符号推理系统传授关于您的未绑定 `SymInts` 的事实。

## 常见错误模式
以下输出显示了常见的错误模式 `GuardOnDataDependentSymNode` 错误：

```sh
torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode: Could not guard on data-dependent expression Eq(u2, -1) (unhinted: Eq(u2, -1)).  (Size-like symbols: none)

Potential framework code culprit (scroll up for full backtrace):
  File "/data/users/ezyang/a/pytorch/torch/_prims_common/__init__.py", line 855, in infer_size
    if d == -1:

For more information, run with TORCH_LOGS="dynamic"
For extended logs when we create symbols, also add TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="u2"
If you suspect the guard was triggered from C++, add TORCHDYNAMO_EXTENDED_DEBUG_CPP=1
For more debugging help, see https://docs.google.com/document/d/1HSuTTVvYH1pTew89Rtpeu84Ht3nQEFTYhAX3Ypa_xJs/edit?usp=sharing
```

## 调试工具

以下是 PyTorch 中可用的一些调试工具列表，您可以使用它们来排查这些错误：

* `TORCH_LOGS="+dynamic"` - 显示有关符号操作的详细日志
* `TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="u2"` - 为特定符号提供扩展日志
* `TORCHDYNAMO_EXTENDED_DEBUG_CPP=1` - 当守卫从 C++ 触发时提供帮助

## 错误变体

以下是您可能遇到的错误变体列表：

| 错误变体 | 描述 |
|------------------|-------------|
| "Could not guard on data-dependent expression" | 当尝试从类似 `u0 == 0` 或 `u0 > 10` 的表达式中提取具体布尔值时发生 |
| "Could not extract specialized integer from data-dependent expression" | 当尝试提取具体整数值时发生。<br/> **常见原因：** <br/> - 依赖于该整数的控制流（例如，循环 `u0` 次）<br/> - 本可以符号化工作的代码中存在过度特化 |

## 如何诊断您的问题

### 步骤 1：检查潜在根源（Python 回溯）

异常会提供一个回溯，通常能指出问题所在。
鉴于 PT2 的回溯可能很长，错误消息也会提示一个潜在的框架代码根源。例如：

```sh
Potential framework code culprit (scroll up for full backtrace):
  File "/data/users/ezyang/a/pytorch/torch/_prims_common/__init__.py", line 855, in infer_size
    if d == -1:
```
### 步骤 2：检查 C++ 回溯

如果框架代码根源信息不足，守卫可能位于 C++ 代码中。您可以通过设置 `TORCHDYNAMO_EXTENDED_DEBUG_CPP=1` 来强制获取 C++ 回溯。这会提供一个详细的 C++ 回溯，其中穿插着 Python、CPython 和 C10/ATen/libtorch 的栈帧。请查找 `at::` 或 `c10::` 命名空间中类似于特定内核代码的符号，这些符号很可能与 Python 回溯中执行的内核相关。如果使用的是非调试版本的 PyTorch，内联优化可能导致栈帧缺失，需要调查源代码来定位问题。
例如，参见 https://github.com/pytorch/pytorch/pull/118579。

以下是一个调试会话中的 C++ 回溯示例：

```
[2024-02-08 08:20:45,259] torch.fx.experimental.symbolic_shapes: [INFO]   File "../
__gen_aten__/out/RegisterCompositeImplicitAutograd.cpp", line 2025, in at::
(anonymous namespace)::(anonymous namespace)
::wrapper_CompositeImplicitAutograd_Tensor_narrow(at::Tensor const&, long,
at::Tensor const&, c10::SymInt) [2024-02-08 08:20:45,259] torch.fx.experimental.
symbolic_shapes: [INFO]   File "../aten/src/ATen/native/TensorShape.cpp", line 1410,
in at::native::narrow_tensor_symint(at::Tensor const&, long, at::Tensor const&,
c10::SymInt) [2024-02-08 08:20:45,259] torch.fx.experimental.symbolic_shapes:
[INFO]   File "../__gen_aten__/out/core/TensorMethods.cpp", line 52, in long
at::Tensor::item<long>() const [2024-02-08 08:20:45,259] torch.fx.experimental.
symbolic_shapes: [INFO]   File "../ATen/core/TensorBody.h", line 4274, in
at::Tensor::item() const
```

在此示例中，`at::native::narrow_tensor_symint` 调用了 `item<long>`，这触发了一个对数据依赖的 `SymNode` 的守卫。

**请考虑以下几点：**

* 这个条件触发对数据依赖符号的守卫是否合理？
* 如果等式涉及两个不同的符号，我们是否应该知道它们实际上是相等的？
* 是否有可能教会那段代码如何处理输入，使其能以适用于所有形状的通用方式工作？

使用 `TORCH_LOGS=dynamic` 并检查用户栈跟踪对于理解如何修复问题至关重要，因为它们能指导您如何修改用户程序。

```sh
[INFO] create_unbacked_symint u0 [-9223372036854775808, 9223372036854775807] (w.py:40 in custom_op_meta)
```

此日志消息指示了无后端 `SymInt` 被分配的位置（`w.py:40`）。一个无后端的 `SymInt` 可能会被分配多次，因此需要跟踪它们的相等关系：

```sh
[INFO] set_replacement u1 = u0 (trivial_lhs) ValueRanges(lower=0, upper=9223372036854775807, is_bool=False)
```

## 修复错误

一旦确定了错误的根源，请按顺序问自己以下问题：

### 步骤 1：我能否重写代码以使用通用路径？

最佳解决方案是重构代码，使其完全不需要基于无后端符号进行分支。问自己：**是否存在一个适用于所有形状的通用代码路径？**

例如，与其这样写：
```python
i = x.item()
if i > 4:
    return x * 2
else:
    return x + 3
```

您能否重写逻辑，使其无需分支即可工作？如果分支仅用于优化或边缘情况处理，请考虑指定一个能处理所有形状的通用路径。

#### 用于谨慎分支的有用工具

PyTorch 提供了几个工具，以更友好的动态形状方式来表达分支：

**`statically_known_true(expr)`**：它：
- 从不添加新的守卫（无重新编译风险）
- 从不会因数据依赖而失败。

该 API 尝试在不添加守卫的情况下评估表达式。如果无法评估，则返回 `False`。将此用于不影响性能的短路逻辑或不值得重新编译的优化。

```python
from torch.fx.experimental.symbolic_shapes import statically_known_true

# 替代：if x.numel() > 10:
if statically_known_true(x.numel() > 10):
    # 优化路径
    ...
else:
    # 通用路径（当未知时采用）
    ...
```

**`guard_or_false(expr)` / `guard_or_true(expr)`**：这些函数可能会添加守卫（如果符号有后端），但永远不会因数据依赖错误而失败。如果由于数据依赖导致评估失败，它们会返回 `False` 或 `True`，而不是硬性失败。用于值得重新编译的性能优化：

```python
from torch.fx.experimental.symbolic_shapes import guard_or_false

# 替代：if x == 0:
if guard_or_false(x == 0):
    return 1
else:
    torch._check(x != 0)  # 通用路径的运行时检查
    return compute(x)
```
.

**`optimization_hint(expr, fallback=None)`**：将符号表达式评估为具体整数，**仅用于优化决策**（例如，选择更快的内核）。与 `guarding_hint_or_throw` 不同，它使用 `fallback` 值处理无后端符号。两个分支仍然必须对所有动态形状都正确——只有性能应该依赖于提示。

# 动态形状故障排除：数据依赖错误

当 PyTorch 编译代码时，它会尝试将 Python 操作转换为高效的计算图。然而，如果代码包含依赖于输入数据值的分支（例如 `if x.item() > 4:`），编译器就无法确定在运行时将执行哪个分支，从而导致错误。

## 错误示例

```python
def foo(x):
    i = x.item()
    if i > 4:          # 错误：i 的值取决于输入数据
        return x * 2
    else:
        return x + 3
```

## 如何修复

### 步骤 1：能否消除分支？

首先，检查分支是否真的必要。有时分支仅用于性能优化，而非正确性关键路径。在这种情况下，可以使用 `torch.compiler.allow_in_graph` 装饰器来避免数据依赖：

```python
from torch.compiler import allow_in_graph

@allow_in_graph
def optimization_hint(numel, fallback):
    return numel if numel is not None else fallback

# 仅用于优化，不用于正确性关键分支
if optimization_hint(x.numel(), fallback=0) > 1024:
    # 大张量的优化路径
    ...
else:
    # 通用路径
    ...
```

**重要提示：** 这些实用程序应仅用于不需要守卫的优化（例如，选择更快的代码路径）。不要将它们用于正确性关键的分支，因为选择的路径取决于跟踪期间示例输入的值。

### 步骤 2：我是否知道总是会走某条路径？

如果无法消除分支，请问自己：**对于我的特定模型，我是否知道总是会走某条路径？**

如果是，可以使用 `torch._check` 来告知编译器走哪条分支：

```python
i = x.item()
torch._check(i > 4)  # 断言对于您的用例，i > 4 总是为真
if i > 4:
    return x * 2
else:
    return x + 3
```

通过断言 `torch._check(i > 4)`，符号推理系统会了解到 `i > 4` 总是为 `True`，从而允许解析分支而不会出错。从编译器的角度来看，else 分支变成了死代码。

### torch._check(cond, msg_fn)

`torch._check` 是一个用于在运行时断言条件的函数，特别是在处理 PyTorch 中的符号整数（`SymInts`）时。

**使用示例：**

```python
torch._check(x.size(0) == y, lambda: f"size mismatch: {x.size(0)} != {y}")
```

上面的代码执行以下操作：

* 创建一个延迟的运行时断言，而不是编译时守卫
* 向符号推理系统传授关于您的未绑定 SymInts 的事实
* 可以通过用等效表达式替换来消除未绑定符号
* 细化符号的值范围
* 记住总是为真的布尔表达式

在语义上，该函数的行为类似于条件检查：
```python
if not cond:
    raise RuntimeError(msg_fn())
```
但存在一些关键差异：

* 在编译时，条件总是被假定为真，即使它涉及未绑定的 `SymInts`。实际检查被推迟到运行时，避免了编译时错误。我们不是设置守卫，而是实现一个延迟的运行时断言来在运行时验证条件。在编译时，我们假设条件不会触发错误，因此我们不需要确定它求值为 `True` 还是 `False`。

* 如果执行相等性测试 `u0 = RHS`，我们会尝试用 RHS 替换所有 `u0` 的实例。如果 RHS 没有未绑定符号，我们将总是这样做，因为移除未绑定符号是有益的——消除它们可以防止创建 `GuardOnDataDependentSymNode`。即使我们无法消除 u0，我们也可以细化它的值范围。值范围指定了变量可能取值的集合。默认情况下，类似大小的未绑定 SymInts 的值范围为 `[0, Inf]`；如果您断言它等于一个具有细化值范围的表达式，例如 `[2, 20]`，那么 `u0` 的值范围将更新为 `[2, 20]`。我们还对反向传播值范围提供了有限的支持。

* 如果执行布尔测试 `f(u0)`，我们将记住这个表达式总是求值为 True，并且如果您求值包含此表达式的表达式，我们将用 True 替换它。我们还支持对逻辑等价语句进行一些有限的推理。例如，如果您执行 `torch._check(u0 < 4)`，我们也会知道 `u0 >= 4` 求值为 `False`，因此在正常的非检查条件中执行这样的测试将会顺利进行。

您还可以使用 `torch._check` 来断言约束和细化值范围。例如，`torch._check(u0 >= 0)` 确定 `u0` 是非负的，将其值范围细化为 `[0, Inf]`。类似地，`torch._check(x > 7)` 将 `x` 约束为大于 7。

当未绑定符号传递给工厂函数如 `torch.empty` 时，它们会自动被识别为表示大小。

### 步骤 3：是否无法修复？

如果运行时确实需要两个分支（即有时 `i > 4`，有时 `i <= 4`），那么任何 `torch._check` 都无法帮助——无法按原样进行跟踪。在这种情况下，您可能需要考虑替代方法，例如使用 `torch.cond` 或填充。

另一个常见的无法修复的模式涉及使用数据依赖的值索引 Python 列表：

```python
return self.mlps[x.item()]
```

这里，`self.mlps` 是一个 Python 列表或 `ModuleList`，代码根据数据依赖的值进行分支。最简单的解决方案是在索引操作之前引发图中断。

## 一些常见的修复模式

### 在模型代码中使用 `torch._check` 进行健全性检查

如果模型代码中有验证条件的健全性检查，可以使用 `torch._check` 代替 `if` 语句。`torch._check` 通过将检查推迟到运行时来处理数据依赖，因此它们不会导致编译时错误。

**注意：** 对于 C++ 代码，使用 `TORCH_SYM_CHECK`，它是 `torch._check` 的 C++ 等效物。

组合条件时，使用 `sym_or`、`sym_and` 等来确保表达式不会被急切求值（这会触发数据依赖错误）：

```python
# 不要这样：
# if x != y or x > y:
#     raise RuntimeError("...")

# 要这样：
from torch.fx.experimental.symbolic_shapes import sym_or
torch._check(sym_or(x != y, x > y), lambda: "Validation failed: expected x != y or x > y")
```

### `u0` 实际上等于 `u1`，但我们不知道

多个未绑定的 `SymInts` 可能在编译时已知相等：

```python
i0 = x.sum().item()
i1 = x.sum().item()
return torch.randn(i0) + torch.randn(i1)
```

如果某处有 `torch._check(i0 == i1)`（在上面的示例中，此检查将发生在加法的形状检查规则内），我们将自动统一两个未绑定的 `SymInts` 并识别它们相等。但是，如果缺少这样的断言，您可能需要显式添加断言来实现这种统一。有关示例，请参阅 https://github.com/pytorch/pytorch/issues/111950。

```{note}
如果我们分配了一个未绑定的 `SymInt` 并立即将其设置为等于另一个，这些实例是无害的，并且不容易从框架中完全消除。
```

### `u0` 是一个张量

您可能过度分配未绑定 `SymInts` 的另一个原因是传递了一个 `Tensor` 并依赖其隐式转换为整数。许多接受整数的函数也会接受一个 `Tensor` 并自动在整数参数上调用 `item()`。检查 `TORCH_LOGS=dynamic` 以确定未绑定 `SymInts` 的数量是否符合预期或是否过多是有益的。当这种情况发生时，将在调用 PyTorch 函数的那一行分配一个新的 `SymInt`。

现在这个问题不太可能导致问题，因为 `t.item()` 的返回值被记忆化了，确保如果您多次调用它，您将始终获得相同的未绑定 `SymInt`。

### 过度特化问题

在非严格导出模式下，考虑以下代码：

```python
u0 = x.sum().item()
return y[:u0]
```

当尝试评估 `u0` 时，此代码将失败，因为当 `SymInt` 直接在 Python 切片内部使用（不使用 Dynamo）时，Python 会强制整数特化，如果它是未绑定的，则会失败。

要解决此问题，您可以重写程序以避免特化。对于上面的示例，您可以通过不使用切片来修复它：

```python
u0 = x.sum().item()
return y.narrow(0, 0, u0)
```

有关更多详细信息，请参阅相关问题 https://github.com/pytorch/pytorch/issues/111950。

### 使用长度而非偏移量

在处理可变序列长度时，通常会有表示序列长度或偏移量的张量。例如，给定 `values = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]`，您可能有 `lengths = [3, 2, 4]` 和 `offsets = [0, 3, 5, 9]`。虽然这些表示形式可以相互转换，但在将它们作为整数处理时（通过调用 `lengths.tolist()`），最好使用长度而不是偏移量。

原因是当您在 `values` 张量上执行 `torch.split()` 时，您需要为每个子序列创建张量，例如大小为 3、2 和 4 的张量。如果您有用于大小的未绑定 `SymInts`，它们会变成 `u0`、`u1` 和 `u2`。您可以轻松地指示它们是大小类型的，然后就完成了。但是，如果您有用于偏移量的未绑定 `SymInts`，它们会变成 `u1 - u0`、`u2 - u1`、`u3 - u2`，这会使事情复杂化。这些量不能方便地标记为大小类型，从而导致潜在问题。由于使用长度或偏移量编写代码相对简单，您应该优先使用长度。

```{seealso}
* *dynamic_shapes*
* *debugging-tlparse-torch-logs*
```
