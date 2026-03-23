(torchdynamo_fine_grain_tracing)=

# 用于细粒度追踪的 TorchDynamo API

:::{note}
本文档中 `torch.compiler.compile` 和 `torch.compile` 可互换使用。两种版本在您的代码中均可工作。
:::

`torch.compile` 对整个用户模型执行 TorchDynamo 追踪。然而，模型代码的一小部分可能无法被 `torch.compiler` 处理。在这种情况下，您可能希望在该特定部分禁用编译器，同时对模型的其余部分运行编译。本节描述了用于定义您希望跳过编译的代码部分的现有 API 及其相关用例。

可用于定义禁用编译的代码部分的 API 如下表所示：

```{eval-rst}
.. csv-table:: 用于控制细粒度追踪的 TorchDynamo API
   :header: "API", "描述", "何时使用？"
   :widths: auto

   "``torch.compiler.disable``", "在装饰的函数以及递归调用的函数上禁用 Dynamo。", "如果模型的一小部分无法用 ``torch.compile`` 处理，非常适合为用户解除阻塞。"
   "``torch._dynamo.disallow_in_graph``", "禁止标记的操作出现在 TorchDynamo 图中。TorchDynamo 会导致图中断，并在 eager（非编译）模式下运行该操作。\n\n此 API 适用于操作，而 ``torch.compiler.disable`` 适用于装饰函数。", "如果像 ``torch.ops.fbgemm.*`` 这样的自定义操作导致 ``torch.compile`` 函数出现问题，此 API 非常适合调试和解除阻塞。"
   "``torch.compile.allow_in_graph``", "被注解的可调用对象会原样进入 TorchDynamo 图。例如，作为 TorchDynamo 的黑盒。\n\n请注意 AOT Autograd 会追踪它，因此 ``allow_in_graph`` 仅是一个 Dynamo 级别的概念。", "此 API 适用于模型中已知具有 TorchDynamo 难以支持的特性（如钩子或 ``autograd.Function``）的部分。但是，每次使用 ``allow_in_graph`` **都必须仔细审查**（无图中断、无闭包）。"
   "``torch._dynamo.graph_break``", "添加图中断。图中断前后的代码会经过 TorchDynamo。", "**部署中很少使用** - 如果您认为需要此功能，很可能您需要的是 ``disable`` 或 ``disallow_in_graph``。"
   "``torch.compiler.is_compiling``", "指示图是否作为 torch.compile() 或 torch.export() 的一部分被执行/追踪。"
   "``torch.compiler.is_dynamo_compiling``", "指示图是否通过 TorchDynamo 进行追踪。它比 torch.compiler.is_compiling() 标志更严格，因为仅当使用 TorchDynamo 时才会设置为 True。"
   "``torch.compiler.is_exporting``", "指示图是否通过 export 进行追踪。它比 torch.compiler.is_compiling() 标志更严格，因为仅当使用 torch.export 时才会设置为 True。"
```

## `torch.compiler.disable`

`torch.compiler.disable` 在装饰的函数帧以及从该装饰函数帧递归调用的所有函数帧上禁用编译。

TorchDynamo 拦截每个 Python 函数帧的执行。因此，假设您有一个代码结构（下图），其中函数 `fn` 调用函数 `a_fn` 和 `b_fn`。并且 `a_fn` 调用 `aa_fn` 和 `ab_fn`。当您使用 PyTorch eager 模式而非 `torch.compile` 时，这些函数帧会按原样运行。使用 `torch.compile` 时，TorchDynamo 会拦截这些函数帧中的每一个（用绿色表示）：

:::{figure} ../../_static/img/fine_grained_apis/api_diagram.png
:alt: 不同 API 的调用栈图。
:::

假设函数 `a_fn` 导致 `torch.compile` 出现问题，并且这是模型中非关键的部分。您可以在函数 `a_fn` 上使用 `compiler.disable`。如上图所示，TorchDynamo 将停止查看源自 `a_fn` 调用的帧（白色表示原始的 Python 行为）。

要跳过编译，您可以使用 `@torch.compiler.disable` 装饰有问题的函数。

如果您不想更改源代码，也可以使用非装饰器语法。但是，我们建议尽可能避免这种风格。在这种情况下，您必须确保原始函数的所有用户现在都使用修补后的版本。

## `torch._dynamo.disallow_in_graph`

`torch._dynamo.disallow_in_graph` 禁止某个操作符（而非函数）出现在 TorchDynamo 提取的图中。请注意，这适用于操作符，而不像 `_dynamo.disable` 那样适用于一般函数。

假设您使用 PyTorch 编译模型。TorchDynamo 能够提取一个图，但随后您发现下游编译器失败。例如，元内核缺失，或者某个特定操作符的 Autograd 分发键设置不正确。然后您可以将该操作符标记为 `disallow_in_graph`，TorchDynamo 将导致图中断，并使用 PyTorch eager 模式运行该操作符。

需要注意的是，您必须找到对应的 Dynamo 级别操作符，而不是 ATen 级别操作符。更多信息请参见文档的“限制”部分。

:::{warning}
`torch._dynamo.disallow_in_graph` 是一个全局标志。如果您正在比较不同的后端编译器，在切换到其他编译器时，可能需要对被禁止的操作符调用 `allow_in_graph`。
:::

## `torch.compiler.allow_in_graph`

`torch.compiler.allow_in_graph` 在相关函数帧具有某些已知的 TorchDynamo 难以支持的特性（如钩子和 `autograd.Function`），并且您确信下游 PyTorch 组件（如 AOTAutograd）可以安全地追踪被装饰的函数时非常有用。当一个函数被 `allow_in_graph` 装饰时，TorchDynamo 将其视为黑盒，并将其原样放入生成的图中。

:::{warning}
`allow_in_graph` 会完全跳过 TorchDynamo 对装饰函数的处理，
省略所有 TorchDynamo 安全检查，包括图中断、闭包处理等。请谨慎使用 `allow_in_graph`。PyTorch 下游组件（例如 AOTAutograd）依赖 TorchDynamo 来处理复杂的 Python 特性，但 `allow_in_graph` 会绕过 TorchDynamo。使用 `allow_in_graph` 可能导致正确性问题以及难以调试的故障。
:::

## 限制

所有现有 API 均在 TorchDynamo 层级应用。因此，这些 API 仅能感知 TorchDynamo 可见的内容。这可能导致一些令人困惑的场景。

例如，`torch._dynamo.disallow_in_graph` 对 ATen 运算符无效，因为它们对 AOT Autograd 可见。例如，在上述示例中，`torch._dynamo.disallow_in_graph(torch.ops.aten.add)` 将不会生效。