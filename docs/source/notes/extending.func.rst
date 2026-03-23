.. _func-autograd-function:

使用 autograd.Function 扩展 torch.func
===========================================

.. currentmodule:: torch.autograd

因此，您希望将 :class:`torch.autograd.Function` 与 :mod:`torch.func` 变换（如 :func:`torch.vmap`、:func:`torch.func.grad` 等）一起使用。

主要有两种使用场景：

- 您希望调用不包含 PyTorch 操作的代码，并使其与函数变换一起工作。也就是说，:class:`torch.autograd.Function` 的 forward/backward 等会调用来自其他系统（如 C++、CUDA、numpy）的函数。
- 您希望指定自定义梯度规则，类似于 JAX 的 `custom_vjp/custom_jvp <https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>`_。

PyTorch 将这两个概念结合到了 :class:`torch.autograd.Function` 中。

基本用法
-----------

本指南假设您熟悉 :ref:`extending-autograd`，其中解释了如何使用 :class:`torch.autograd.Function`。

:class:`torch.autograd.Function` 可以有一个接受 ctx 对象的 :meth:`~Function.forward` 方法，也可以有一个单独的 :meth:`~Function.forward` 方法（不接受 ``ctx``）和一个修改 ``ctx`` 对象的静态方法 :meth:`~Function.setup_context`。

只有后者支持函数变换：

- :meth:`~Function.forward` 是执行操作的代码，它不应接受 ``ctx`` 对象。
- ``setup_context(ctx, inputs, output)`` 是您可以在其中调用 ``ctx`` 方法的代码。在这里，您应该保存用于反向传播的张量（通过调用 ``ctx.save_for_backward(*tensors)``），或者保存非张量（通过将它们赋值给 ``ctx`` 对象）。

因为 :meth:`~Function.setup_context` 只接受 ``inputs`` 和 ``output``，所以唯一可以保存的量要么是输入或输出中的对象（如张量），要么是从它们派生的量（如 ``Tensor.shape``）。如果您希望保存 :meth:`Function.forward` 中的非输入中间激活用于反向传播，那么您需要将其作为 :meth:`~Function.forward` 的输出返回，以便它被传递给 :meth:`~Function.setup_context`。

根据变换的不同，

- 要支持反向模式自动微分（:func:`torch.func.grad`、:func:`torch.func.vjp`），:class:`torch.autograd.Function` 需要一个 :meth:`~Function.backward` 静态方法。
- 要支持 :func:`torch.vmap`，:class:`torch.autograd.Function` 需要一个 :meth:`~Function.vmap` 静态方法。
- 要支持 :func:`torch.func.jvp`，:class:`torch.autograd.Function` 需要一个 :meth:`~Function.jvp` 静态方法。
- 要支持变换的组合（如 :func:`torch.func.jacrev`、:func:`torch.func.jacfwd`、:func:`torch.func.hessian`）——您可能需要上述多个方法。

为了使 :class:`torch.autograd.Function` 能够与函数变换任意组合，我们建议除了 :meth:`~Function.forward` 和 :meth:`~Function.setup_context` 之外的所有其他静态方法都必须是可变换的：也就是说，它们必须仅由 PyTorch 操作符组成，或者调用其他 :class:`torch.autograd.Function`（这些函数可能调用 C++/CUDA 等）。

让我们来看一些常见用例的示例。

示例 1：autograd.Function 调用另一个系统
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

一个常见的情况是 :class:`torch.autograd.Function` 的 forward() 和 backward() 都调用另一个系统（如 C++、CUDA、numpy、triton）。

::

    import torch
    import numpy as np

    def to_numpy(tensor):
        return tensor.cpu().numpy()

    class NumpySort(torch.autograd.Function):
        # 注意 forward 不接受 ctx
        @staticmethod
        def forward(x, dim):
            device = x.device
            x = to_numpy(x)
            ind = np.argsort(x, axis=dim)
            ind_inv = np.argsort(ind, axis=dim)
            result = np.take_along_axis(x, ind, axis=dim)
            # 任何需要在反向传播中保存的中间变量必须作为输出返回。
            return (
                # 期望的输出
                torch.tensor(result, device=device),
                # 为反向传播保存的中间变量
                torch.tensor(ind, device=device),
                # 为反向传播保存的中间变量
                torch.tensor(ind_inv, device=device),
            )

        # setup_context 负责调用方法和/或赋值给 ctx 对象。
        # 请不要在 setup_context 中进行额外的计算（例如将张量相加）。
        @staticmethod
        def setup_context(ctx, inputs, output):
            x, dim = inputs
            # 注意 output 是您从 forward 返回的任何内容。
            # 如果您返回了多个值，那么 output 是一个包含多个值的元组。
            # 如果您返回了单个张量，那么 output 是一个张量。
            # 如果您返回了一个包含单个张量的元组，那么 output 是一个包含单个张量的元组。
            _, ind, ind_inv = output
            ctx.mark_non_differentiable(ind, ind_inv)
            # 张量必须通过 ctx.save_for_backward 保存。请不要直接将它们赋值给 ctx 对象。
            ctx.save_for_backward(ind, ind_inv)
            # 非张量可以通过将它们作为属性赋值给 ctx 对象来保存。
            ctx.dim = dim

        @staticmethod
        def backward(ctx, grad_output, _0, _1):
            # 为了使 autograd.Function 能够与函数变换任意组合，
            # 除了 forward 和 setup_context 之外的所有静态方法
            # 都必须以“可变换”的方式实现；也就是说，它们必须
            # 仅由 PyTorch 操作或 autograd.Function 组成。
            #
            # 例如，这允许我们进行双重反向传播和/或计算
            # 二阶梯度。
            #
            # 我们使用另一个 autograd.Function，NumpyTake，
            # 来编写 NumpySort 的反向传播。
            ind, ind_inv = ctx.saved_tensors
            return NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim), None

    class NumpyTake(torch.autograd.Function):
        @staticmethod
        def forward(x, ind, ind_inv, dim):
            device = x.device
            x = to_numpy(x)
            ind = to_numpy(ind)
            return torch.tensor(np.take_along_axis(x, ind, dim), device=device)

        @staticmethod
        def setup_context(ctx, inputs, output):
            x, ind, ind_inv, dim = inputs
            ctx.save_for_backward(ind, ind_inv)
            ctx.dim = dim

        @staticmethod
        def backward(ctx, grad_output):
            ind, ind_inv = ctx.saved_tensors
            result = NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim)
            return result, None, None, None


现在，为了更容易使用 ``NumpySort``（隐藏我们作为输出返回的中间结果，并允许默认参数和关键字参数），我们创建一个调用它的新函数：:

    def numpy_sort(x, dim=-1):
        result, _, _ = NumpySort.apply(x, dim)
        return result

这是一个完整性检查：:

    x = torch.randn(2, 3)
    grad_x = torch.func.grad(lambda x: numpy_sort(x).sum())(x)
    assert torch.allclose(grad_x, torch.ones_like(x))



示例 2：autograd.Function 指定自定义梯度规则
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

另一种常见情况是使用 PyTorch 操作实现的 :class:`torch.autograd.Function`。PyTorch 能够自动计算 PyTorch 操作的梯度，但也许我们希望自定义梯度的计算方式。我们可能希望使用与 PyTorch 提供的不同的自定义反向传播的一些原因包括：

- 提高数值稳定性
- 改变反向传播的性能特征
- 改变边缘情况的处理方式（例如 nan、inf）
- 修改梯度（例如梯度裁剪）

以下是一个 :class:`torch.autograd.Function` 的示例，用于函数 ``y = x ** 3``，我们改变了其性能特征（一些通常在反向传播期间进行的计算，即计算 dx，被移到了前向传播中进行）。

::

  class MyCube(torch.autograd.Function):
      @staticmethod
      def forward(x):
          result = x ** 3
          # 在常规 PyTorch 中，如果我们只运行 y = x ** 3，那么反向传播
          # 会计算 dx = 3 * x ** 2。在这个 autograd.Function 中，我们
          # 将计算移到了前向传播中进行。
          dx = 3 * x ** 2
          return result, dx

      @staticmethod
      def setup_context(ctx, inputs, output):
          x, = inputs
          result, dx = output
          ctx.save_for_backward(x, dx)

      @staticmethod
      def backward(ctx, grad_output, grad_dx):
          x, dx = ctx.saved_tensors
          # 为了使 autograd.Function 能够处理高阶
          # 梯度，我们必须添加 `dx` 的梯度贡献。
          result = grad_output * dx + grad_dx * 6 * x
          return result

现在，为了更容易使用 ``NumpySort``（并隐藏我们作为输出返回的中间结果），我们创建一个调用它的新函数：:

    def my_cube(x):
        result, _ = MyCube.apply(x)
        return result

这是一个计算二阶梯度的完整性检查：:

    x = torch.randn([])
    ggx = torch.func.grad(torch.func.grad(my_cube))(x)
    assert torch.allclose(ggx, 6 * x)

限制与注意事项
^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

    请仔细阅读 :class:`torch.autograd.Function` 与 torch.func 变换的这些限制。
    我们无法捕获许多此类情况并优雅地报错，因此它们将导致未定义行为。

请不要捕获正在被变换、requires_grad=True 或是双重张量的张量到 :class:`torch.autograd.Function` 的方法中。确保完全安全的唯一方法是，确保在 :class:`torch.autograd.Function` 的任何方法内部使用的张量必须直接作为输入传递（或通过 ctx 对象），而不是来自 :class:`torch.autograd.Function` 外部。

:class:`torch.autograd.Function` 不处理 pytree 中的张量（可能包含也可能不包含张量的任意嵌套 Python 数据结构）。要使这些张量被 autograd 跟踪，它们必须直接作为参数传递给 :class:`torch.autograd.Function`。这与 jax.{custom_vjp, custom_jvp} 不同，后者确实接受 pytree。

请仅使用 :meth:`~torch.autograd.function.FunctionCtx.save_for_backward` 或 :meth:`~torch.autograd.function.FunctionCtx.save_for_forward` 来保存张量。请不要直接将张量或张量集合赋值给 ctx 对象——这些张量将不会被跟踪。


:func:`torch.vmap` 支持
--------------------------

要将 :class:`torch.autograd.Function` 与 :func:`torch.vmap` 一起使用，您必须：

- 提供一个 :meth:`~Function.vmap` 静态方法，告诉我们 :class:`torch.autograd.Function` 在 :func:`torch.vmap` 下的行为
- 通过设置 ``generate_vmap_rule=True`` 要求我们自动生成它。

自动生成 vmap 规则
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果你的 :class:`torch.autograd.Function` 满足以下额外约束条件，那么我们
能够为其生成一个 vmap 规则。如果它不满足约束条件，或者你
希望在 vmap 下有自定义行为，请手动定义一个 vmap 静态方法（参见下一节）。

.. warning::

     我们无法轻松检查以下约束条件并优雅地报错。
     违反约束条件可能导致未定义行为。

- :class:`torch.autograd.Function` 的 :meth:`~Function.forward`、:meth:`~Function.backward`（如果存在）和 :meth:`~Function.jvp`
  （如果存在）静态方法必须能够通过 :func:`torch.vmap` 进行转换。即，
  它们必须仅由 PyTorch 操作组成（而不是例如 NumPy 或自定义
  CUDA 内核）。

示例::

    class MyCube(torch.autograd.Function):
        # 将 generate_vmap_rule 设置为 True，以要求 PyTorch 自动生成
        # 一个 vmap 规则。
        generate_vmap_rule = True

        @staticmethod
        def forward(x):
            result = x ** 3
            dx = 3 * x ** 2
            return result, dx

        @staticmethod
        def setup_context(ctx, inputs, output):
            x, = inputs
            result, dx = output
            ctx.save_for_backward(x, dx)

        @staticmethod
        def backward(ctx, grad_output, grad_dx):
            x, dx = ctx.saved_tensors
            result = grad_output * dx + grad_dx * 6 * x
            return result

    def my_cube(x):
        result, dx = MyCube.apply(x)
        return result

    x = torch.randn(3)
    result = torch.vmap(my_cube)(x)
    assert torch.allclose(result, x ** 3)


定义 vmap 静态方法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果你的 :class:`torch.autograd.Function` 调用了其他系统（如 NumPy、C++、CUDA、triton），
那么要使其与 :func:`torch.vmap` 或使用它的变换一起工作，你需要
手动定义一个 :meth:`~Function.vmap` 静态方法。

根据你想使用的变换和你的用例，你可能不需要
为所有 :class:`torch.autograd.Function` 添加 :meth:`~Function.vmap` 静态方法：

- 例如，:func:`torch.func.jacrev` 在反向传播过程中执行 :func:`~torch.vmap`。
  因此，如果你只对使用 :func:`torch.func.jacrev` 感兴趣，那么只有
  :meth:`~Function.backward` 静态方法需要是可 vmap 的。

不过，我们确实建议确保所有 :class:`torch.autograd.Function` 都支持
:func:`torch.vmap`，特别是如果你正在编写第三方库，并且希望你的
:class:`torch.autograd.Function` 能与 :func:`torch.func` 变换的所有组合一起工作。

从概念上讲，vmap 静态方法负责定义 :meth:`~Function.forward`
在 :func:`torch.vmap` 下的行为。也就是说，它定义了如何变换
:meth:`~Function.forward` 以在具有额外维度（被 vmap 的维度）的输入上运行。这类似于
:func:`torch.vmap` 在 PyTorch 操作上的实现方式：对于每个操作，我们定义一个 vmap 规则（有时也
称为“批处理规则”）。

以下是定义 :meth:`~Function.vmap` 静态方法的方式：

- 签名为 ``vmap(info, in_dims: Tuple[Optional[int]], *args)``，其中
  ``*args`` 与 :meth:`~Function.forward` 的参数相同。
- vmap 静态方法负责定义 :meth:`~Function.forward` 在 :func:`torch.vmap` 下的行为。
  也就是说，给定具有额外维度（由 ``in_dims`` 指定）的输入，我们如何计算 :meth:`~Function.forward` 的批处理版本？
- 对于 ``args`` 中的每个参数，``in_dims`` 都有一个对应的 ``Optional[int]``。
  如果参数不是张量或参数未被 vmap，则为 ``None``；
  否则，它是一个整数，指定张量的哪个维度被 vmap。
- ``info`` 是一个可能有所帮助的额外元数据集合：
  ``info.batch_size`` 指定被 vmap 的维度的大小，而
  ``info.randomness`` 是传递给 :func:`torch.vmap` 的 ``randomness`` 选项。
- vmap 静态方法的返回值是一个 ``(output, out_dims)`` 元组。类似于
  ``in_dims``，``out_dims`` 应与 ``output`` 具有相同的结构，并为每个输出包含一个
  ``out_dim``，指定输出是否具有 vmap 维度及其索引。

示例::

    def to_numpy(tensor):
        return tensor.cpu().numpy()

    class NumpySort(torch.autograd.Function):
        @staticmethod
        def forward(x, dim):
            device = x.device
            x = to_numpy(x)
            ind = np.argsort(x, axis=dim)
            ind_inv = np.argsort(ind, axis=dim)
            result = np.take_along_axis(x, ind, axis=dim)
            return (
                torch.tensor(result, device=device),
                torch.tensor(ind, device=device),
                torch.tensor(ind_inv, device=device),
            )

        @staticmethod
        def setup_context(ctx, inputs, output):
            x, dim = inputs
            _, ind, ind_inv = output
            ctx.mark_non_differentiable(ind, ind_inv)
            ctx.save_for_backward(ind, ind_inv)
            ctx.dim = dim

        @staticmethod
        def backward(ctx, grad_output, _0, _1):
            ind, ind_inv = ctx.saved_tensors
            return NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim), None

        # vmap 静态方法的签名为：
        # vmap(info, in_dims: Tuple[Optional[int]], *args)
        # 其中 *args 与 `forward` 的参数相同。
        @staticmethod
        def vmap(info, in_dims, x, dim):
            # 对于每个输入（x 和 dim），in_dims 存储一个 Optional[int]，
            # 该值表示：
            # - 如果输入未被 vmap 或输入不是张量，则为 None
            # - 如果输入被 vmap，则为表示被 vmap 维度索引的整数。
            x_bdim, _ = in_dims

# "vmap 规则"定义了在输入张量增加一个维度时如何执行操作的逻辑。在 NumpySort 中，x 有一个额外维度 (x_bdim)。vmap 规则就是简单地再次调用 NumpySort，但传递不同的 `dim` 参数。
            x = x.movedim(x_bdim, 0)
            # 正确处理负维度
            dim = dim if dim >= 0 else dim + x.dim() - 1
            result = NumpySort.apply(x, dim + 1)

            # vmap 规则必须返回一个包含两个元素的元组
            # 1. 输出。应与 forward() 返回的数量相同。
            # 2. 每个输出对应一个 Optional[int]，指定该输出是否正在被 vmap 处理，
            #    如果是，则指定正在被 vmap 处理的维度索引。
            #
            # NumpySort.forward 返回一个包含 3 个张量的元组。由于我们将
            # 被 vmap 处理的维度移动到了 `x` 的前面，该维度会出现在所有输出的第 0 维。
            # 返回值是 (output, out_dims) —— output 是一个包含 3 个张量的元组，
            # out_dims 是一个包含 3 个 Optional[int] 的元组。
            return NumpySort.apply(x, dim + 1), (0, 0, 0)

    class NumpyTake(torch.autograd.Function):
        @staticmethod
        def forward(x, ind, ind_inv, dim):
            device = x.device
            x = to_numpy(x)
            ind = to_numpy(ind)
            return torch.tensor(np.take_along_axis(x, ind, dim), device=device)

        @staticmethod
        def setup_context(ctx, inputs, output):
            x, ind, ind_inv, dim = inputs
            ctx.save_for_backward(ind, ind_inv)
            ctx.dim = dim

        @staticmethod
        def backward(ctx, grad_output):
            ind, ind_inv = ctx.saved_tensors
            result = NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim)
            return result, None, None, None

        @staticmethod
        def vmap(info, in_dims, x, ind, ind_inv, dim):
            x_bdim, ind_bdim, ind_inv_bdim, _ = in_dims

            # 策略是：将 {x, ind, ind_inv} 都扩展出被 vmap 处理的维度。
            # 然后，调用 NumpyTake(expanded_x, expanded_ind, expanded_ind_inv, new_dim)。

            # 通过将负维度转换为正数来处理负维度
            logical_dim = x.dim() if x_bdim is None else x_bdim - 1
            dim = dim if dim >= 0 else dim + logical_dim

            def maybe_expand_bdim_at_front(x, x_bdim):
                if x_bdim is None:
                    return x.expand(info.batch_size, *x.shape)
                return x.movedim(x_bdim, 0)

            # 如果张量没有被 vmap 处理的维度，就将其扩展出来。
            # 否则，将其移动到张量的最前面。
            x = maybe_expand_bdim_at_front(x, x_bdim)
            ind = maybe_expand_bdim_at_front(ind, ind_bdim)
            ind_inv = maybe_expand_bdim_at_front(ind_inv, ind_inv_bdim)

            # 返回值是一个元组 (output, out_dims)。由于 output 是一个张量，
            # 所以 out_dims 是一个 Optional[int]（而不是一个元组）。
            return NumpyTake.apply(x, ind, ind_inv, dim + 1), 0

    def numpy_sort(x, dim=-1):
        result, _, _ = NumpySort.apply(x, dim)
        return result

    x = torch.randn(2, 3)
    result = torch.vmap(numpy_sort)(x)
    assert torch.allclose(result, numpy_sort(result, 1))


.. note::

    vmap 静态方法应旨在保持整个 :class:`~torch.autograd.Function` 的语义。也就是说，（伪代码）``grad(vmap(MyFunc))`` 应该可以用 ``grad(map(MyFunc))`` 替换。

    如果你的 autograd.Function 在反向传播中有任何自定义行为，请记住这一点。

.. note::

    为一个 PyTorch 能够通过 ``generate_vmap_rule=True`` 生成 vmap 规则的 :class:`~torch.autograd.Function` 编写自定义的 vmap 静态方法是一个合法的用例。如果生成的 vmap 规则不符合你想要的语义，你可能希望这样做。

:func:`torch.func.jvp` 支持
------------------------------

为了支持前向模式自动微分，一个 :class:`torch.autograd.Function` 必须有一个 :meth:`~Function.jvp` 静态方法。
详情请参阅 :ref:`forward-ad-autograd-function`。