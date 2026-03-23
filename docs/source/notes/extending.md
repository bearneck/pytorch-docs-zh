# 扩展 PyTorch

本文档将介绍扩展 `torch.nn`{.interpreted-text role="mod"}、`torch.autograd`{.interpreted-text role="mod"}、`torch`{.interpreted-text role="mod"} 以及编写自定义 C++ 扩展的方法。

## 添加新运算符

PyTorch 提供了大量作用于张量的运算符库（例如 `torch.add`{.interpreted-text role="func"}、`torch.sum`{.interpreted-text role="func"} 等）。然而，您可能希望为 PyTorch 引入新的自定义操作，并使其行为类似于 PyTorch 的内置运算符。为此，您必须通过 Python `torch-library-docs`{.interpreted-text role="ref"} 或 C++ TORCH_LIBRARY API 向 PyTorch 注册自定义操作。

更多详情请参阅 [PyTorch 自定义运算符主页](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)。

## 扩展 `torch.autograd`{.interpreted-text role="mod"} {#extending-autograd}

 currentmodule
torch.autograd


向 `~torch.autograd`{.interpreted-text role="mod"} 添加操作需要为每个操作实现一个新的 `Function`{.interpreted-text role="class"} 子类。请记住，Functions 是 `~torch.autograd`{.interpreted-text role="mod"} 用来编码操作历史并计算梯度的工具。

本文档的第一部分重点介绍反向模式自动微分，因为它是最广泛使用的功能。最后一部分讨论了前向模式自动微分的扩展。

### 何时使用

通常，如果您希望在模型中执行不可微分或依赖于非 PyTorch 库（例如 NumPy）的计算，但仍希望您的操作能够与其他操作链接并与自动微分引擎协同工作，那么请实现自定义函数。

在某些情况下，自定义函数也可用于提高性能和内存使用效率：如果您使用 [C++ 扩展](https://pytorch.org/tutorials/advanced/cpp_extension.html) 实现了前向和反向传播，可以将它们包装在 `~Function`{.interpreted-text role="class"} 中以与自动微分引擎交互。如果您希望减少为反向传播保存的缓冲区数量，可以使用自定义函数将多个操作组合在一起。

### 何时不使用

如果您已经可以使用 PyTorch 的内置操作编写函数，那么其反向计算图（很可能）已经能够被自动微分记录。在这种情况下，您不需要自己实现反向函数。考虑使用普通的 Python 函数。

如果您需要维护状态，即可训练参数，您应该（同时）使用自定义模块。有关扩展 `torch.nn`{.interpreted-text role="mod"} 的更多信息，请参阅下面的部分。

如果您希望在反向传播期间修改梯度或执行副作用，请考虑注册 [张量钩子](https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html#torch.Tensor.register_hook) 或 [模块钩子](https://pytorch.org/docs/stable/notes/modules.html#module-hooks)。

### 如何使用

请按以下步骤操作： 1. 子类化 `~Function`{.interpreted-text role="class"} 并实现 `~Function.forward`{.interpreted-text role="meth"}、（可选的）`~Function.setup_context`{.interpreted-text role="meth"} 和 `~Function.backward`{.interpreted-text role="meth"} 方法。 2. 在 [ctx]{.title-ref} 参数上调用适当的方法。 3. 声明您的函数是否支持 [双重反向传播](https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html)。 4. 使用 gradcheck 验证您的梯度是否正确。

**步骤 1：** 子类化 `Function`{.interpreted-text role="class"} 后，您需要定义 3 个方法：

- `~Function.forward`{.interpreted-text role="meth"} 是执行操作的代码。它可以接受任意数量的参数，其中一些可以是可选的（如果您指定了默认值）。这里接受所有类型的 Python 对象。跟踪历史记录（即 [requires_grad=True]{.title-ref}）的 `Tensor`{.interpreted-text role="class"} 参数将在调用前转换为不跟踪历史记录的张量，并且它们的使用将在计算图中注册。请注意，此逻辑不会遍历列表/字典/任何其他数据结构，只会考虑作为调用直接参数的张量。您可以返回单个 `Tensor`{.interpreted-text role="class"} 输出，或者在有多个输出时返回张量的 `tuple`{.interpreted-text role="class"}。此外，请参阅 `Function`{.interpreted-text role="class"} 的文档，查找只能在 `~Function.forward`{.interpreted-text role="meth"} 中调用的有用方法的描述。
- `~Function.setup_context`{.interpreted-text role="meth"}（可选）。您可以编写一个接受 [ctx]{.title-ref} 对象的"组合式" `~Function.forward`{.interpreted-text role="meth"}，或者（从 PyTorch 2.0 开始）编写一个不接受 [ctx]{.title-ref} 的独立 `~Function.forward`{.interpreted-text role="meth"} 和一个负责修改 [ctx]{.title-ref} 的 `~Function.setup_context`{.interpreted-text role="meth"} 方法。`~Function.forward`{.interpreted-text role="meth"} 应包含计算逻辑，而 `~Function.setup_context`{.interpreted-text role="meth"} 应仅负责 [ctx]{.title-ref} 的修改（不应包含任何计算逻辑）。通常，独立的 `~Function.forward`{.interpreted-text role="meth"} 和 `~Function.setup_context`{.interpreted-text role="meth"} 更接近 PyTorch 原生操作的工作方式，因此与各种 PyTorch 子系统的组合性更好。更多详情请参阅 `combining-forward-context`{.interpreted-text role="ref"}。
- `~Function.backward`{.interpreted-text role="meth"}（或 `~Function.vjp`{.interpreted-text role="meth"}）定义了梯度公式。它将接收与输出数量相同的 `Tensor`{.interpreted-text role="class"} 参数，每个参数代表相对于该输出的梯度。\*\*切勿\*\*就地修改这些参数，这一点非常重要。它应返回与输入数量相同的张量，每个张量包含相对于其对应输入的梯度。如果您的输入不需要梯度（`~ctx.needs_input_grad`{.interpreted-text role="attr"} 是一个布尔值元组，指示每个输入是否需要梯度计算），或者是非 `Tensor`{.interpreted-text role="class"} 对象，您可以返回 `python:None`{.interpreted-text role="class"}。此外，如果您对 `~Function.forward`{.interpreted-text role="meth"} 有可选参数，只要它们都是 `python:None`{.interpreted-text role="any"}，您可以返回比输入数量更多的梯度。

**步骤 2：** 您有责任正确使用 [ctx]{.title-ref} 中的函数，以确保新的 `Function`{.interpreted-text role="class"} 能够与自动微分引擎正常工作。

- `~torch.autograd.function.FunctionCtx.save_for_backward`{.interpreted-text role="meth"} 应被用于保存反向传播所需的任何张量（而不是直接保存在 `ctx` 上）。你不能对非张量使用 `save_for_backward`；这些应直接存储在 `ctx` 上。

  通过 `save_for_backward` 保存张量：

  1\. 允许自动微分引擎在 `autograd.Function` 的反向计算完成后立即清除它们。 （如果张量直接存储在 `ctx` 上，它将在自动微分图的生命周期内不必要地保持存活------通常直到迭代结束。）

  2.  有助于避免某些引用循环（例如，因为 `autograd.Function` 本身的张量输出会保持对 ctx 的引用）。
  3.  对于与依赖于 `torch.autograd.graph.saved_tensors_hooks`{.interpreted-text role="class"} 的功能（如激活检查点和卸载）的兼容性很重要。

  如果既非输入也非输出的张量被保存用于反向传播，你的 `~Function`{.interpreted-text role="class"} 可能不支持双反向传播（参见步骤 3）。

- `~torch.autograd.function.FunctionCtx.mark_dirty`{.interpreted-text role="meth"} 必须用于标记在前向函数中被原地修改的任何输入。

- `~torch.autograd.function.FunctionCtx.mark_non_differentiable`{.interpreted-text role="meth"} 必须用于告知引擎某个输出是否不可微分。默认情况下，所有可微分类型的输出张量都将被设置为需要梯度。不可微分类型（即整数类型）的张量永远不会被标记为需要梯度。

- `~torch.autograd.function.FunctionCtx.set_materialize_grads`{.interpreted-text role="meth"} 可用于告知自动微分引擎，在输出不依赖于输入的情况下，通过不具体化提供给反向函数的梯度张量来优化梯度计算。也就是说，如果设置为 False，Python 中的 None 对象或 C++ 中的"未定义张量"（即 [x.defined()]{.title-ref} 为 False 的张量 x）在调用反向函数之前将不会被转换为填充零的张量，因此你的代码需要将这些对象视为填充零的张量来处理。此设置的默认值为 True。

除了 `ctx` 方法外，`~Function`{.interpreted-text role="class"} 类还支持以下类属性：

- `~Function.clear_saved_tensors_on_access`{.interpreted-text role="attr"}：当在 `~Function`{.interpreted-text role="class"} 子类上设置为 `True` 时，在反向传播中访问 `ctx.saved_tensors` 将清除对这些张量的内部引用。这使得张量可以在 `saved_tensors` 返回的局部变量超出作用域后立即被释放，而不是等待缓冲区在 Node 执行结束时被清除。这可以减少在保存的张量仅需使用一次的反向传播中的峰值内存使用量。默认值为 `False`。请注意，启用此选项后，`saved_tensors` 只能访问一次；第二次访问将引发错误。

**步骤 3：** 如果你的 `~Function`{.interpreted-text role="class"} 不支持双反向传播，你应该通过用 `~function.once_differentiable`{.interpreted-text role="func"} 装饰 backward 来明确声明这一点。使用此装饰器后，尝试通过你的函数执行双反向传播将产生错误。有关双反向传播的更多信息，请参阅我们的双反向传播教程。

**步骤 4：** 建议使用 `torch.autograd.gradcheck`{.interpreted-text role="func"} 来检查你的反向函数是否正确计算了前向传播的梯度，方法是通过你的反向函数计算雅可比矩阵，并与使用有限差分法数值计算的雅可比矩阵进行逐元素比较。

### 示例

下面你可以找到一个 `Linear` 函数的代码，附有额外注释：:

> \# 继承自 Function class LinearFunction(Function):
>
> > \# 注意 forward、setup_context 和 backward 都是 \@staticmethods \@staticmethod def forward(input, weight, bias): output = input.mm(weight.t()) if bias is not None: output += bias.unsqueeze(0).expand_as(output) return output
> >
> > \@staticmethod \# inputs 是传递给 forward 的所有输入的元组。 \# output 是 forward() 的输出。 def setup_context(ctx, inputs, output): input, weight, bias = inputs ctx.save_for_backward(input, weight, bias)
> >
> > \# 此函数只有一个输出，因此只获得一个梯度 \@staticmethod def backward(ctx, grad_output): \# 这是一个非常方便的模式------在 backward 的开头 \# 解包 saved_tensors 并将所有关于输入的梯度初始化为 \# None。由于额外的尾部 None 会被忽略， \# 即使函数有可选输入，返回语句也很简单。 input, weight, bias = ctx.saved_tensors grad_input = grad_weight = grad_bias = None
> >
> > > \# 这些 needs_input_grad 检查是可选的，仅用于 \# 提高效率。如果你想简化代码，可以 \# 跳过它们。为不需要梯度的输入返回梯度 \# 不是错误。 if ctx.needs_input_grad\[0\]: grad_input = grad_output.mm(weight) if ctx.needs_input_grad\[1\]: grad_weight = grad_output.t().mm(input) if bias is not None and ctx.needs_input_grad\[2\]: grad_bias = grad_output.sum(0)
> > >
> > > return grad_input, grad_weight, grad_bias

现在，为了更轻松地使用这些自定义操作，我们建议为它们设置别名或将其包装在函数中。包装在函数中允许我们支持默认参数和关键字参数：:

> \# 选项 1：别名 linear = LinearFunction.apply
>
> \# 选项 2：包装在函数中，以支持默认参数和关键字参数。 def linear(input, weight, bias=None): return LinearFunction.apply(input, weight, bias)

这里，我们给出一个由非张量参数参数化的函数的额外示例：:

> 
>
> class MulConstant(Function):
>
> :   \@staticmethod def forward(tensor, constant): return tensor \* constant
>
>     \@staticmethod def setup_context(ctx, inputs, output): \# ctx 是一个上下文对象，可用于存储信息以供反向计算使用 tensor, constant = inputs ctx.constant = constant
>
>     \@staticmethod def backward(ctx, grad_output): \# 我们返回与参数数量相同的输入梯度。 \# forward 中非张量参数的梯度必须为 None。 return grad_output \* ctx.constant, None

这里，我们通过调用 set_materialize_grads(False) 来优化上述示例：:

> 
>
> class MulConstant(Function):
>
> :   \@staticmethod def forward(tensor, constant): return tensor \* constant
>
>     \@staticmethod def setup_context(ctx, inputs, output): tensor, constant = inputs ctx.set_materialize_grads(False) ctx.constant = constant
>
>     \@staticmethod def backward(ctx, grad_output): \# 这里我们必须处理 grad_output 张量为 None 的情况。此时我们可以跳过不必要的计算，直接返回 None。 if grad_output is None: return None, None
>
>     > \# 我们返回与参数数量相同的输入梯度。 \# forward 中非张量参数的梯度必须为 None。 return grad_output \* ctx.constant, None

以下是一个使用 `clear_saved_tensors_on_access` 来减少反向传播期间峰值内存的示例。该函数计算两次矩阵乘法，在反向传播中，我们在计算完中间张量的梯度后、计算剩余梯度之前释放该中间张量：:

> 
>
> class TwoMatmuls(Function):
>
> :   clear_saved_tensors_on_access = True
>
>     \@staticmethod def forward(ctx, x, weight1, weight2): inter = x.mm(weight1) ctx.save_for_backward(x, weight1, inter, weight2) return inter.mm(weight2)
>
>     \@staticmethod def backward(ctx, grad_output): x, weight1, inter, weight2 = ctx.saved_tensors
>
>     > \# 计算第二次矩阵乘法的梯度 grad_weight2 = inter.t().mm(grad_output) grad_inter = grad_output.mm(weight2.t())
>     >
>     > \# 释放不再需要的 inter 和 weight2 del inter, weight2
>     >
>     > \# 计算第一次矩阵乘法的梯度 grad_weight1 = x.t().mm(grad_inter) grad_x = grad_inter.mm(weight1.t())
>     >
>     > return grad_x, grad_weight1, grad_weight2

如果你需要保存在 `~Function.forward`{.interpreted-text role="meth"} 中计算的任何"中间"张量，它们必须作为输出返回，或者结合使用 `forward` 和 `~Function.setup_context`{.interpreted-text role="meth"}（参见 `combining-forward-context`{.interpreted-text role="ref"}）。 请注意，这意味着如果你希望梯度通过这些中间值流动，你需要为它们定义梯度公式（另请参阅 [双重反向传播教程](https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html) ）：:

> 
>
> class MyCube(torch.autograd.Function):
>
> :   \@staticmethod def forward(x): \# 我们希望保存 dx 用于反向传播。为此，它必须作为输出返回。 dx = 3 \* x \*\* 2 result = x \*\* 3 return result, dx
>
>     \@staticmethod def setup_context(ctx, inputs, output): x, = inputs result, dx = output ctx.save_for_backward(x, dx)
>
>     \@staticmethod def backward(ctx, grad_output, grad_dx): x, dx = ctx.saved_tensors \# 为了使 autograd.Function 能够处理高阶梯度，我们必须添加 [dx]{.title-ref} 的梯度贡献， \# 即 grad_dx \* 6 \* x。 result = grad_output \* dx + grad_dx \* 6 \* x return result
>
> \# 将 MyCube 包装在一个函数中，以便更清晰地显示输出是什么 def my_cube(x): result, dx = MyCube.apply(x) return result

 note
 title
Note


`backward` 的输入，即 `grad_output`{.interpreted-text role="attr"}，也可以是跟踪历史的张量。因此，如果 `backward` 是通过可微分操作实现的（例如，调用另一个自定义的 `~torch.autograd.Function`{.interpreted-text role="class"}），高阶导数将正常工作。 在这种情况下，使用 `save_for_backward` 保存的张量也可以在反向传播中使用，并且有梯度流回，但保存在 `ctx` 中的张量不会有梯度流回。 如果你需要保存在 `ctx` 中的张量有梯度流回，你应该将其设为自定义 `Function` 的输出，并使用 `save_for_backward` 保存。


你可能希望检查你实现的反向方法是否确实计算了函数的导数。可以通过使用小的有限差分与数值近似进行比较来实现：:

> from torch.autograd import gradcheck
>
> \# gradcheck 接受一个张量元组作为输入，检查用这些张量评估的梯度是否足够接近数值近似， \# 如果它们都满足此条件则返回 True。 input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True)) test = gradcheck(linear, input, eps=1e-6, atol=1e-4) print(test)

有关有限差分梯度比较的更多细节，请参见 `grad-check`{.interpreted-text role="ref"}。 如果你的函数用于高阶导数（对反向传播进行微分），你可以使用同一包中的 `gradgradcheck` 函数来检查高阶导数。

### 合并或分离的 `~Function.forward`{.interpreted-text role="meth"} 和 `~Function.setup_context`{.interpreted-text role="meth"} {#combining-forward-context}

定义 `~Function`{.interpreted-text role="class"} 主要有两种方式。要么：

- 定义一个将前向计算逻辑与 `~Function.setup_context`{.interpreted-text role="meth"} 结合的 `~Function.forward`{.interpreted-text role="meth"}
- （从 PyTorch 2.0 开始）定义独立的 `~Function.forward`{.interpreted-text role="meth"} 和 `~Function.setup_context`{.interpreted-text role="meth"}

我们推荐第二种方式（独立的 `~Function.forward`{.interpreted-text role="meth"} 和 `~Function.setup_context`{.interpreted-text role="meth"}）， 因为它更接近 PyTorch 原生操作的实现方式，并且能与 `torch.func`{.interpreted-text role="mod"} 变换组合使用。 然而，我们计划在将来同时支持两种方式； 将 `~Function.forward`{.interpreted-text role="meth"} 与 `~Function.setup_context`{.interpreted-text role="meth"} 结合使用会带来更大的灵活性， 因为您能够保存中间结果而不必将其作为输出返回。

关于如何定义具有独立 `~Function.forward`{.interpreted-text role="meth"} 和 `~Function.setup_context`{.interpreted-text role="meth"} 的 `~Function`{.interpreted-text role="class"}， 请参阅前一节。

以下是如何定义具有合并的 `~Function.forward`{.interpreted-text role="meth"} 和 `~Function.setup_context`{.interpreted-text role="meth"} 的 `Function`{.interpreted-text role="class"} 的示例：:

> 
>
> class LinearFunction(Function):
>
> :   \@staticmethod \# ctx 是 forward 的第一个参数 def forward(ctx, input, weight, bias=None): \# 前向传播可以使用 ctx。 ctx.save_for_backward(input, weight, bias) output = input.mm(weight.t()) if bias is not None: output += bias.unsqueeze(0).expand_as(output) return output
>
>     \@staticmethod def backward(ctx, grad_output): input, weight, bias = ctx.saved_tensors grad_input = grad_weight = grad_bias = None
>
>     > 
>     >
>     > if ctx.needs_input_grad\[0\]:
>     >
>     > :   grad_input = grad_output.mm(weight)
>     >
>     > if ctx.needs_input_grad\[1\]:
>     >
>     > :   grad_weight = grad_output.t().mm(input)
>     >
>     > if bias is not None and ctx.needs_input_grad\[2\]:
>     >
>     > :   grad_bias = grad_output.sum(0)
>     >
>     > return grad_input, grad_weight, grad_bias

### 前向模式自动微分 {#forward-ad-autograd-function}

重写前向模式自动微分公式的 API 非常相似，但有一些细微差别。 您可以实现 `~Function.jvp`{.interpreted-text role="meth"} 函数。

该函数将接收与输入数量相同的 `Tensor`{.interpreted-text role="class"} 参数，每个参数代表对应输入的梯度。 它应返回与输出数量相同的张量，每个张量包含对应输出的梯度。 `~Function.jvp`{.interpreted-text role="meth"} 将在 `~Function.forward`{.interpreted-text role="meth"} 方法之后、`~Function.apply`{.interpreted-text role="meth"} 返回之前被调用。

`~Function.jvp`{.interpreted-text role="meth"} 与 `~Function.backward`{.interpreted-text role="meth"} 函数有一些细微差别：

- 您可以使用 [ctx]{.title-ref} 将任何数据从 `~Function.forward`{.interpreted-text role="meth"} 传递到 `~Function.jvp`{.interpreted-text role="meth"} 函数。 如果该状态在 `~Function.backward`{.interpreted-text role="meth"} 中不需要，您可以在 `~Function.jvp`{.interpreted-text role="meth"} 函数末尾通过执行 `del ctx.foo` 来显式释放它。
- `~Function.jvp`{.interpreted-text role="meth"} 的实现必须是可反向微分的，或者显式检查给定的前向模式梯度中是否有设置了 `requires_grad` 的。
- `~Function.jvp`{.interpreted-text role="meth"} 函数必须与 `~Function.forward`{.interpreted-text role="meth"} 的视图/原地操作行为保持一致。 例如，如果第 [i]{.title-ref} 个输入被原地修改，那么第 [i]{.title-ref} 个梯度也必须原地更新。 类似地，如果第 [j]{.title-ref} 个输出是第 [k]{.title-ref} 个输入的视图，那么返回的第 [j]{.title-ref} 个输出梯度必须是给定第 [k]{.title-ref} 个输入梯度的视图。
- 由于用户无法指定需要计算哪些梯度，`~Function.jvp`{.interpreted-text role="meth"} 函数应始终计算所有输出的梯度。
- 前向模式梯度确实遵循 `~torch.autograd.function.FunctionCtx.set_materialize_grads`{.interpreted-text role="meth"} 设置的标志， 当此功能被禁用时，您可能会得到 [None]{.title-ref} 输入梯度。

### `torch.func`{.interpreted-text role="mod"} 变换和/或 `torch.vmap`{.interpreted-text role="func"}

详情请参阅 `func-autograd-function`{.interpreted-text role="ref"}。

## 扩展 `torch.nn`{.interpreted-text role="mod"}

 currentmodule
torch.nn


`~torch.nn`{.interpreted-text role="mod"} 导出两种接口------模块及其函数版本。 您可以通过两种方式扩展它，但我们建议对所有包含参数或缓冲区的层使用模块， 并建议对无参数操作（如激活函数、池化等）使用函数形式。

添加操作的功能版本已在上面的章节中完全涵盖。

### 添加 `Module`{.interpreted-text role="class"}

由于 `~torch.nn`{.interpreted-text role="mod"} 大量使用 `~torch.autograd`{.interpreted-text role="mod"}，添加新的 `Module`{.interpreted-text role="class"} 需要实现一个 `~torch.autograd.Function`{.interpreted-text role="class"}， 该函数执行操作并能计算梯度。从现在开始，假设我们想要实现一个 `Linear` 模块，并且我们已经如上文所示实现了函数。 添加此模块所需的代码非常少。现在，需要实现两个函数：

- `__init__`（\*可选\*）------接收诸如核大小、特征数量等参数，并初始化参数和缓冲区。
- `~Module.forward`{.interpreted-text role="meth"}------实例化一个 `~torch.autograd.Function`{.interpreted-text role="class"} 并使用它来执行操作。 它与上面显示的函数包装器非常相似。

以下是如何实现 `Linear` 模块的示例：:

> 
>
> class Linear(nn.Module):
>
> :   
>
>     def \_\_init\_\_(self, input_features, output_features, bias=True):
>
>     :   super().\_\_init\_\_() self.input_features = input_features self.output_features = output_features
>
>         \# nn.Parameter 是一种特殊的张量，一旦被赋值作为属性， \# 就会自动注册为 Module 的参数。参数和缓冲区需要被注册， \# 否则它们不会出现在 .parameters() 中（不适用于缓冲区）， \# 并且在调用例如 .cuda() 时不会被转换。你可以使用 \# .register_buffer() 来注册缓冲区。 \# nn.Parameters 默认需要梯度。 self.weight = nn.Parameter(torch.empty(output_features, input_features)) if bias: self.bias = nn.Parameter(torch.empty(output_features)) else: \# 你应该总是注册所有可能的参数，但可选的参数 \# 可以设置为 None。 self.register_parameter(\'bias\', None)
>
>         \# 不是一种非常智能的权重初始化方式 [nn.init.uniform]()(self.weight, -0.1, 0.1) if self.bias is not None: [nn.init.uniform]()(self.bias, -0.1, 0.1)
>
>     def forward(self, input):
>
>     :   \# 关于这里发生的情况，请参阅 autograd 部分的解释。 return LinearFunction.apply(input, self.weight, self.bias)
>
>     def extra_repr(self):
>
>     :   \# （可选）设置关于此模块的额外信息。你可以通过 \# 打印此类的对象来测试它。 return \'input_features={}, output_features={}, bias={}\'.format( self.input_features, self.output_features, self.bias is not None )

## 扩展 `torch`{.interpreted-text role="mod"} Python API {#extending-torch-python}

你可以通过定义一个自定义类，其方法与 `Tensor`{.interpreted-text role="class"} 匹配，来创建模拟 `Tensor`{.interpreted-text role="class"} 的自定义类型。但是，如果你希望能够将这些类型传递给顶层 `torch`{.interpreted-text role="mod"} 命名空间中接受 `Tensor`{.interpreted-text role="class"} 操作数的函数，例如 `torch.add`{.interpreted-text role="func"}，该怎么办？

如果你的自定义 Python 类型定义了一个名为 `__torch_function__` 的方法，当你的自定义类的实例被传递给 `torch`{.interpreted-text role="mod"} 命名空间中的函数时，PyTorch 将调用你的 `__torch_function__` 实现。这使得可以为 `torch`{.interpreted-text role="mod"} 命名空间中的任何函数定义自定义实现，你的 `__torch_function__` 实现可以调用这些函数，从而允许你的用户将你的自定义类型与他们已经为 `Tensor`{.interpreted-text role="class"} 编写的现有 PyTorch 工作流一起使用。这适用于与 `Tensor`{.interpreted-text role="class"} 无关的"鸭子"类型以及用户定义的 `Tensor`{.interpreted-text role="class"} 子类。

### 使用类似 `Tensor`{.interpreted-text role="class"} 的类型扩展 `torch`{.interpreted-text role="mod"}

 note
 title
Note


此功能受到 NumPy `__array_function__` 协议的启发。更多详细信息，请参阅 [NumPy 文档](https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch) 和 [NEP-0018](https://numpy.org/neps/nep-0018-array-function-protocol.html)。


为了具体说明，让我们从一个简单的示例开始，说明 API 分派机制。我们将创建一个自定义类型，表示一个 2D 标量张量，由阶数 `N` 和对角线元素的值 `value` 参数化：:

> 
>
> class ScalarTensor(object):
>
> :   
>
>     def \_\_init\_\_(self, N, value):
>
>     :   self.\_N = N self.\_value = value
>
>     def \_\_repr\_\_(self):
>
>     :   return \"ScalarTensor(N={}, value={})\".format(self.\_N, self.\_value)
>
>     def tensor(self):
>
>     :   return self.\_value \* torch.eye(self.\_N)

这个设计的第一次迭代并不是很有用。`ScalarTensor` 的主要功能是提供比基础张量类更紧凑的标量张量字符串表示：:

> \>\>\> d = ScalarTensor(5, 2) \>\>\> d ScalarTensor(N=5, value=2) \>\>\> d.tensor() tensor(\[\[2., 0., 0., 0., 0.\], \[0., 2., 0., 0., 0.\], \[0., 0., 2., 0., 0.\], \[0., 0., 0., 2., 0.\], \[0., 0., 0., 0., 2.\]\])

如果我们尝试将此对象与 `torch`{.interpreted-text role="mod"} API 一起使用，将会遇到问题：:

> \>\>\> import torch \>\>\> torch.mean(d) TypeError: mean(): argument \'input\' (position 1) must be Tensor, not ScalarTensor

向 `ScalarTensor` 添加 `__torch_function__` 实现可以使上述操作成功。让我们重新实现，这次添加一个 `__torch_function__` 实现：:

> HANDLED_FUNCTIONS = {} class ScalarTensor(object): def \_\_init\_\_(self, N, value): self.\_N = N self.\_value = value
>
> > 
> >
> > def \_\_repr\_\_(self):
> >
> > :   return \"ScalarTensor(N={}, value={})\".format(self.\_N, self.\_value)
> >
> > def tensor(self):
> >
> > :   return self.\_value \* torch.eye(self.\_N)
> >
> > \@classmethod def \_\_torch_function\_\_(cls, func, types, args=(), kwargs=None): if kwargs is None: kwargs = {} if func not in HANDLED_FUNCTIONS or not all( issubclass(t, (torch.Tensor, ScalarTensor)) for t in types ): return NotImplemented return HANDLED_FUNCTIONS\[func\](*args,*\*kwargs)

`__torch_function__` 方法接受四个参数：`func`，对被覆盖的 torch API 函数的引用；`types`，实现 `__torch_function__` 的类 Tensor 类型的列表；`args`，传递给函数的参数元组；以及 `kwargs`，传递给函数的关键字参数字典。它使用一个名为 `HANDLED_FUNCTIONS` 的全局分派表来存储自定义实现。此字典的键是 `torch` 命名空间中的函数，值是 `ScalarTensor` 的实现。

 note
 title
Note


使用全局分派表不是 `__torch_function__` API 的强制部分，它只是构建覆盖实现的有用设计模式。


仅凭这个类定义还不足以让 `torch.mean` 在处理 `ScalarTensor` 时正确工作------我们还需要为 `ScalarTensor` 操作数定义 `torch.mean` 的实现，并将该实现添加到 `HANDLED_FUNCTIONS` 调度表字典中。一种实现方式是定义一个装饰器：:

> import functools def implements(torch_function): \"\"\"Register a torch function override for ScalarTensor\"\"\" def decorator(func): functools.update_wrapper(func, torch_function) HANDLED_FUNCTIONS\[torch_function\] = func return func return decorator

这个装饰器可以应用到我们的覆盖实现上：:

> \@implements(torch.mean) def mean(input): return float(input.\_value) / input.\_N

经过这个修改，我们现在可以在 `ScalarTensor` 上使用 `torch.mean`：:

> \>\>\> d = ScalarTensor(5, 2) \>\>\> torch.mean(d) 0.4

当然，`torch.mean` 是最简单的覆盖函数示例，因为它只接受一个操作数。我们可以使用相同的机制来覆盖接受多个操作数的函数，其中任何一个操作数可能是定义了 `__torch_function__` 的张量或类张量对象，例如对于 `torch.add`{.interpreted-text role="func"}：:

> 
>
> def ensure_tensor(data):
>
> :   
>
>     if isinstance(data, ScalarTensor):
>
>     :   return data.tensor()
>
>     return torch.as_tensor(data)
>
> \@implements(torch.add) def add(input, other): try: if input.\_N == other.\_N: return ScalarTensor(input.\_N, input.\_value + other.\_value) else: raise ValueError(\"Shape mismatch!\") except AttributeError: return torch.add(ensure_tensor(input), ensure_tensor(other))

这个版本在双操作数都是 `ScalarTensor` 实例时采用快速路径，而当任一操作数不是 `ScalarTensor` 时则退化为将数据转换为张量的较慢路径。这使得覆盖函数在任一操作数是 `ScalarTensor` 或常规 `Tensor`{.interpreted-text role="class"} 时都能正确工作：:

> \>\>\> s = ScalarTensor(2, 2) \>\>\> torch.add(s, s) ScalarTensor(N=2, value=4) \>\>\> t = torch.tensor(\[\[1, 1,\], \[1, 1\]\]) \>\>\> torch.add(s, t) tensor(\[\[3., 1.\], \[1., 3.\]\])

请注意，我们的 `add` 实现不像 `torch.add`{.interpreted-text role="func"} 那样接受 `alpha` 或 `out` 作为关键字参数：:

> \>\>\> torch.add(s, s, alpha=2) TypeError: add() got an unexpected keyword argument \'alpha\'

为了速度和灵活性，`__torch_function__` 调度机制不会检查覆盖函数的签名是否与 `torch`{.interpreted-text role="mod"} API 中被覆盖函数的签名匹配。对于某些应用，忽略可选参数是可以接受的，但为了确保与 `Tensor`{.interpreted-text role="class"} 的完全兼容性，用户实现的 torch API 函数应当注意精确模拟被覆盖函数的 API。

`torch`{.interpreted-text role="mod"} API 中没有显式覆盖的函数会从 `__torch_function__` 返回 `NotImplemented`。如果所有定义了 `__torch_function__` 的操作数都返回 `NotImplemented`，PyTorch 将引发 `TypeError`。这意味着大多数情况下，当传递了该类型的实例时，没有为该类型提供显式覆盖的操作将引发 `TypeError`：:

> \>\>\> torch.mul(s, 3) TypeError: no implementation found for \'torch.mul\' on types that implement \_\_torch_function\_\_: \[ScalarTensor\]

实际上，这意味着如果你想使用这种方式的 `__torch_function__` 实现来覆盖函数，你需要显式实现完整的 `torch`{.interpreted-text role="mod"} API 或你用例关心的整个 API 子集。这可能是一个很高的要求，因为完整的 `torch`{.interpreted-text role="mod"} API 非常广泛。

另一种选择是，对于未处理的运算不返回 `NotImplemented`，而是在没有可用覆盖时将 `Tensor`{.interpreted-text role="class"} 传递给原始的 `torch`{.interpreted-text role="mod"} 函数。例如，如果我们将 `ScalarTensor` 的 `__torch_function__` 实现改为以下版本：:

> \@classmethod def \_\_torch_function\_\_(cls, func, types, args=(), kwargs=None): if kwargs is None: kwargs = {} if func not in HANDLED_FUNCTIONS or not all( issubclass(t, (torch.Tensor, ScalarTensor)) for t in types ): args = \[a.tensor() if hasattr(a, \'tensor\') else a for a in args\] return func(*args,kwargs) return HANDLED_FUNCTIONS\[func\](*args, \*\*kwargs)

那么 `torch.mul`{.interpreted-text role="func"} 将能正确工作，尽管返回类型将始终是 `Tensor`{.interpreted-text role="class"} 而不是 `ScalarTensor`{.interpreted-text role="class"}，即使两个操作数都是 `ScalarTensor`{.interpreted-text role="class"} 实例：:

> \>\>\> s = ScalarTensor(2, 2) \>\>\> torch.mul(s, s) tensor(\[\[4., 0.\], \[0., 4.\]\])

另请参阅下面的 `MetadataTensor` 示例，这是该模式的另一种变体，但始终返回 `MetadataTensor` 以在 `torch`{.interpreted-text role="mod"} API 的运算中传播元数据。

`__torch_function__` 协议设计用于完整覆盖 API，部分覆盖可能导致不理想的结果，特别是某些函数引发 `TypeError`。这对于子类尤其重要，其中 [torch.add]{.title-ref}、\`torch.Tensor.\_\_add\_\_\` 和 [torch.Tensor.add]{.title-ref} 三者都必须被覆盖，即使它们返回完全相同的结果。未能做到这一点也可能导致无限递归。如果需要从 `torch.Tensor` 子类实现函数，必须在实现中使用 `super().__torch_function__`。

### 子类化 `torch.Tensor`

从 1.7.0 版本开始，应用于 `torch.Tensor` 子类的 `torch.Tensor` 方法及公共 `torch.*` 命名空间中的函数将返回子类实例而不是 `torch.Tensor` 实例：：

> \>\>\> class SubTensor(torch.Tensor): \... pass \>\>\> type(torch.add(SubTensor(\[0\]), SubTensor(\[1\]))).\_\_name\_\_ \'SubTensor\' \>\>\> type(torch.add(SubTensor(\[0\]), torch.tensor(\[1\]))).\_\_name\_\_ \'SubTensor\'

如果存在多个子类，默认会选择继承层级最低的那个。如果无法唯一确定这种情况，则会引发 `TypeError`:

    >>> type(torch.add(SubTensor2([0]), SubTensor([1]))).__name__
    'SubTensor2'
    >>> type(torch.add(SubTensor2([0]), torch.tensor([1]))).__name__
    'SubTensor2'
    >>> torch.add(SubTensor([0]), OtherSubTensor([1]))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: no implementation found for 'torch.add' on types that implement __torch_function__: [SubTensor, OtherSubTensor]

如果希望对所有张量方法进行全局覆盖，可以使用 `__torch_function__`。以下是一个记录所有函数/方法调用的示例:

    class LoggingTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            # 注意：记录日志会调用 Tensor.__repr__，因此如果不造成无限递归，就无法记录 __repr__
            if func is not torch.Tensor.__repr__:
                logging.info(f"func: {func.__name__}, args: {args!r}, kwargs: {kwargs!r}")
            if kwargs is None:
                kwargs = {}
            return super().__torch_function__(func, types, args, kwargs)

但是，如果希望覆盖张量子类上的某个方法，可以通过直接覆盖该方法（为子类定义它）或使用 `__torch_function__` 并与 `func` 匹配来实现。

在子类的 `__torch_function__` 中，应始终调用 `super().__torch_function__(func, ...)` 而不是直接调用 `func`，这与 1.7.0 版本之前的情况不同。如果不这样做，可能会导致 `func` 递归地再次进入 `__torch_function__`，从而引发无限递归。

### 使用 `Tensor`{.interpreted-text role="class"} 包装器类型扩展 `torch`{.interpreted-text role="mod"}

另一种有用的场景是包装 `Tensor`{.interpreted-text role="class"} 的类型，无论是作为属性还是通过子类化。下面我们实现了这种类型的一个特例，即 `MetadataTensor`，它将一个元数据字典附加到 `Tensor`{.interpreted-text role="class"} 上，并通过 `torch`{.interpreted-text role="mod"} 操作进行传播。由于这是针对完整 `torch`{.interpreted-text role="mod"} API 的通用包装，我们不需要单独实现每个覆盖，因此可以使 `__torch_function__` 的实现对允许的操作更加宽松:

    class MetadataTensor(object):
        def __init__(self, data, metadata=None, **kwargs):
            self._t = torch.as_tensor(data, **kwargs)
            self._metadata = metadata

        def __repr__(self):
            return "Metadata:\n{}\n\ndata:\n{}".format(self._metadata, self._t)

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}
            metadatas = tuple(a._metadata for a in args if hasattr(a, '_metadata'))
            args = [getattr(a, '_t', a) for a in args]
            assert len(metadatas) > 0
            ret = func(*args, **kwargs)
            return MetadataTensor(ret, metadata=metadatas[0])

这个简单的实现不一定适用于 `torch`{.interpreted-text role="mod"} API 中的每个函数，但对于捕获大多数常见操作来说已经足够:

    >>> metadata = {'owner': 'Ministry of Silly Walks'}
    >>> m = MetadataTensor([[1, 2], [3, 4]], metadata=metadata)
    >>> t = torch.tensor([[1, 2], [1, 2]])
    >>> torch.add(t, m)
    Metadata:
    {'owner': 'Ministry of Silly Walks'}

    data:
    tensor([[2, 4],
            [4, 6]])
    >>> torch.mul(t, m)
    Metadata:
    {'owner': 'Ministry of Silly Walks'}

    data:
    tensor([[1, 4],
            [3, 8]])

### 对定义了 `__torch_function__` 的多种类型进行操作

可以在 torch API 中使用多个各自具有 `__torch_function__` 实现的不同类型，但需要特别注意。在这种情况下，规则如下：

- 调度操作会为每个操作数收集所有不同的 `__torch_function__` 实现，并按顺序调用它们：子类先于父类，否则按运算符表达式从左到右的顺序。
- 如果返回了除 `NotImplemented` 之外的任何值，则该值将作为结果返回。实现可以通过返回 `NotImplemented` 来表明它们不实现某个操作。
- 如果所有 `__torch_function__` 实现都返回 `NotImplemented`，PyTorch 将引发 `TypeError`。

### 测试 PyTorch API 覆盖的完整性

实现 `__torch_function__` 的一个棘手方面是，如果某些操作有覆盖而其他操作没有，用户最多会看到不一致的体验，最坏的情况是在使用没有覆盖的函数时在运行时看到错误。为了简化这个过程，PyTorch 提供了一个面向开发者的 API，用于确保对 `__torch_function__` 覆盖的完整支持。此 API 是私有的，未来可能会在没有警告的情况下更改。

首先，要获取所有可覆盖函数的列表，请使用 `torch.overrides._get_overridable_functions`。这将返回一个字典，其键是 `PyTorch` Python API 中的命名空间，值是该命名空间中可覆盖的函数列表。例如，让我们打印 `torch.nn.functional` 中前 5 个可覆盖函数的名称:

    >>> from torch.overrides import get_overridable_functions
    >>> func_dict = get_overridable_functions()
    >>> nn_funcs = func_dict[torch.nn.functional]
    >>> print([f.__name__ for f in nn_funcs[:5])

\[\'adaptive_avg_pool1d\', \'adaptive_avg_pool2d\', \'adaptive_avg_pool3d\',

:   \'adaptive_max_pool1d\', \'adaptive_max_pool1d_with_indices\'\]

此函数列表使得遍历所有可重写函数成为可能，但在实际中，若不为每个测试繁琐地手动复制每个函数的签名，这还不足以编写覆盖所有这些函数的测试。为了简化此过程，`torch.overrides._get_testing_overrides` 函数返回一个字典，将 `PyTorch` API 中的可重写函数映射到虚拟的 lambda 函数，这些 lambda 函数具有与原始函数相同的签名，但无条件地返回 -1。这些函数与 `inspect` 结合使用最为有用，可用于分析原始 `PyTorch` 函数的函数签名：:

> \>\>\> import inspect \>\>\> from torch.overrides import get_testing_overrides \>\>\> override_dict = get_testing_overrides() \>\>\> dummy_add = override_dict\[torch.add\] \>\>\> inspect.signature(dummy_add) \<Signature (input, other, out=None)\>

最后，`torch.overrides.get_ignored_functions` 返回一个函数元组，这些函数明确不能被 `__torch_function__` 重写。此列表可用于确认未出现在 `get_overridable_functions` 返回的字典中的函数确实无法被重写。

## 扩展 `torch`{.interpreted-text role="mod"} 原生 API {#extending-torch-c++}

虽然 `__torch_function__` 允许有效地扩展 PyTorch 纯 Python 组件的行为，但它不允许扩展 PyTorch 中 C++ 实现的部分。为此，`Tensor`{.interpreted-text role="class"} 子类也可以定义 `__torch_dispatch__`，它将能够在 C++ 级别重写行为。

要有效使用此功能，了解 PyTorch 原生部分的实现方式非常重要。其中最重要的组件是我们称之为"分发器"的部分（最佳描述可在此 [博客文章](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) 中找到，尽管它略有陈旧）。如其名称所示，它负责为特定函数调用调用正确的后端函数。例如，当调用 `torch.add(a, b)` 时，分发器将检查两个参数，确定此特定调用应使用哪个"特性"（自动微分、自动混合精度、函数化等）和哪个"后端"（CPU、CUDA、MPS 等），最后调用所有正确的内核。 内核执行的一个非常常见的操作是"重新分发"。例如，在 GPU 上使用自动混合精度运行神经网络时，第一次调用将是自动混合精度内核，它将处理任何潜在的自动混合精度逻辑并向下重新分发。下一个特性将是自动微分，它将正确创建自动微分图然后向下重新分发。最后，我们到达 CUDA 的后端内核，它将启动正确的 CUDA 内核并返回最终结果。在返回途中，自动微分会将图附加到输出上，最后，自动混合精度将有机会在退出时进行任何所需的更新。

分发器的一个配置是所有这些特性和后端键被调用的顺序。最新的列表及其顺序可以在 `DispatchKey.h` 文件内的 `DispatchKey` 枚举中找到。出于扩展 torch 的目的，本讨论中重要的顺序子集是：

vmap -\> Autocast -\> Autograd -\> ZeroTensor -\> Neg/Conj -\> Functionalize -\> Python -\> Backends

本讨论中最重要的键是 `Python`，因为每个定义了 `__torch_dispatch__` 方法的 Tensor 子类都将调用此特性。正是从这里调用用户定义的方法，并且可以任意重写行为。从那里，再次调用提供的 `func` 将执行"重新分发"。

此实现的一些重要含义是：

- 此代码"在所有特性之下"运行。因此，像常规后端一样，它仅负责生成每个 Tensor 的输出值（并且可以且应该忽略所有高级特性，如自动微分、自动混合精度等）。
- 如果任何高级特性实现了给定函数而未重新分发，它将永远不会到达 `Python` 键，因此 `__torch_dispatch__` 回调永远不会被触发。这尤其发生在 CompositeImplicitAutograd 函数上，这些函数在自动微分级别评估而无需重新分发。这是因为 CompositeImplicitAutograd 函数通过隐式调用其他原生操作来指定其自动微分公式，因此在自动微分级别，该函数被分解为其原生操作，并评估这些操作。
- 当回调到 Python 以及包装结果时，使用与常规 PyTorch Python/C++ 绑定相同的转换。特别是，某些对象无法在 Python 中表示，需要特殊处理（例如，未定义的张量变为 None）。
- 我们的原生函数被延迟填充为 `torch.ops.{namespace}.{func_name}.{overload_name}` 作为可调用的 Python 对象，以便从 Python 轻松与之交互。提供给 `__torch_dispatch__` 的 `func` 对象始终是此命名空间中的一个条目。此命名空间可用于直接调用原生操作，并绕过通常的 Python API 和绑定代码。

与 `__torch_function__` 能够拦截 torch 的所有 Python API 和 Tensor 方法类似，`__torch_dispatch__` 能够拦截所有对 aten 原生 API 的调用。请注意，进入调度器之前，Tensor 上的所有方法都会转换为函数调用，因此在这里都会显示为函数调用：`torch.add(a, 2)` 和 `a + 2` 将导致完全相同的 aten 调用。 这些函数大多定义在 `native_functions.yaml` 中，该文件指定了这些函数的属性及其后端实现。然后，它们的实现连同指定的特性会通过代码生成自动注册。 一些更特殊的函数或特性也会在 C++ 代码库的其他地方或用户定义的 C++ 扩展中注册。

也可以使用 `torch.library`{.interpreted-text role="mod"} 添加\`新的\`原生函数。这个 Python 特性允许为原生函数定义和/或添加新的实现。这可以用来添加缺失的内核、替换现有的内核或定义全新的原生函数。

你可以在 [subclass zoo](https://github.com/albanD/subclass_zoo) 仓库中找到许多基于 `__torch_dispatch__` 的子类示例。

### `__torch_dispatch__` 调用约定 {#torch-dispatch-calling-convention}

``` python
@classmethod
def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    pass
```

当用户使用具有 `__torch_dispatch__` 的输入调用运算符时，该调用可能会被转发到 `__torch_dispatch__`。在调用 `__torch_dispatch__` 之前，args 和 kwargs 会被规范化，即：

- `kwargs` 包含运算符模式中的仅关键字参数。 如果某个 kwarg 等于其默认值（在模式中），则不会传递它。
- `args` 包含所有其他参数，无论它们是如何传递给运算符的（位置参数 vs 关键字参数）。 如果某个 arg 等于其默认值，并且它是右侧的位置参数或其右侧的所有参数都未传递，则不会传递它。

## 使用模式扩展所有 `torch`{.interpreted-text role="mod"} API

遗憾的是，有些函数不接受 Tensor 输入。这意味着上述子类方法不能用于覆盖所有 PyTorch 函数的行为。此外，如果用例需要拦截每个函数调用，将每个 Tensor 更改为子类可能会过于侵入。

为了解决这个用例，我们引入了"模式"的概念。这些模式适用于 `__torch_function__` 和 `__torch_dispatch__` 覆盖，分别通过子类化 `torch.overrides.TorchFunctionMode`{.interpreted-text role="class"} 和 `torch.utils._python_dispatch.TorchDispatchMode`{.interpreted-text role="class"} 创建，并用作上下文管理器。

为了简化描述其与子类和其他模式的交互方式，每当进入模式的上下文管理器时，每个函数的行为都像是在参数列表的开头有一个额外的 Tensor 参数，该参数以模式作为子类。 这尤其意味着所有模式处理程序将在任何子类处理程序之前被调用，并且对应于内部上下文管理器的模式将始终首先运行。

同样重要的是要注意，在给定的模式处理程序中，该特定模式是禁用的，可以通过执行 `with self:` 手动重新启用。

以下是一个展示每种类型日志模式的示例：:

> import torch from torch.overrides import TorchFunctionMode, resolve_name from torch.utils.\_python_dispatch import TorchDispatchMode
>
> class FunctionLog(TorchFunctionMode):
>
> :   
>
>     def \_\_torch_function\_\_(self, func, types, args, kwargs=None):
>
>     :   print(f\"Function Log: {resolve_name(func)}(*{args},{kwargs})\") return func(*args, \*\*(kwargs or {}))
>
> class DispatchLog(TorchDispatchMode):
>
> :   
>
>     def \_\_torch_dispatch\_\_(self, func, types, args, kwargs=None):
>
>     :   print(f\"Dispatch Log: {func}(*{args},{kwargs})\") return func(*args, \*\*(kwargs or {}))
>
> def f():
>
> :   a = torch.rand(10, requires_grad=True) b = a \* 2 b.sum().backward()
>
> print(\"TorchFunctionMode logging:\") with FunctionLog(): f()
>
> print(\"TorchDispatchMode logging:\") with DispatchLog(): f()

打印结果如下，附有额外注释：:

> TorchFunctionMode logging: Function Log: torch.rand(*(10,),{\'requires_grad\': True}) Function Log: torch.Tensor.mul(*(tensor(\[0.7164, 0.9897, 0.1745, 0.9336, 0.4287, 0.7989, 0.2169, 0.7474, 0.5624, 0.5970\], requires_grad=True), 2), **None) Function Log: torch.Tensor.sum(\*(tensor(\[1.4328, 1.9794, 0.3490, 1.8671, 0.8573, 1.5977, 0.4338, 1.4948, 1.1249, 1.1939\], grad_fn=\<MulBackward0\>),),**None) \# 请注意，在 Python 层面，我们只看到对 backward 的调用，但看不到 autograd 引擎中发生的情况。 Function Log: torch.Tensor.backward(*(tensor(12.3307, grad_fn=\<SumBackward0\>),),*\*{\'gradient\': None, \'retain_graph\': None, \'create_graph\': False, \'inputs\': None})

TorchDispatchMode 日志记录：

:   \# 此处 autograd 的 requires_grad 标志被移除，同时填充了默认参数。 Dispatch Log: aten.rand.default(*(\[10\],),{\'device\': device(type=\'cpu\'), \'pin_memory\': False}) Dispatch Log: aten.mul.Tensor(*(tensor(\[0.2151, 0.6018, 0.8415, 0.9060, 0.2974, 0.7708, 0.6668, 0.0352, 0.7948, 0.6023\], requires_grad=True), 2), **{}) Dispatch Log: aten.sum.default(\*(tensor(\[0.4303, 1.2036, 1.6831, 1.8120, 0.5949, 1.5416, 1.3335, 0.0705, 1.5897, 1.2046\], grad_fn=\<MulBackward0\>),),**{}) \# 此处我们看不到对 backward 本身的调用，而是其组成部分。从这里开始，是创建初始梯度的工厂函数。 Dispatch Log: aten.ones_like.default(*(tensor(11.4637, grad_fn=\<SumBackward0\>),),{\'pin_memory\': False, \'memory_format\': torch.preserve_format}) \# 这是 sum 的反向传播 Dispatch Log: aten.expand.default(*(tensor(1.), \[10\]), **{}) Dispatch Log: aten.mul.Tensor(\*(tensor(\[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.\]), 2),**{}) Dispatch Log: aten.detach.default(*(tensor(\[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.\]),),{}) Dispatch Log: aten.detach.default(*(tensor(\[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.\]),), \*\*{})
