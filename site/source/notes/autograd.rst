.. _autograd-mechanics:

Autograd 机制
==================

本文档将概述 autograd 的工作原理以及它如何记录运算。虽然并非必须完全理解这些内容，但我们建议您熟悉它，因为这将有助于您编写更高效、更简洁的程序，并有助于调试。

.. _how-autograd-encodes-history:

Autograd 如何编码历史记录
--------------------------------

Autograd 是一个反向自动微分系统。从概念上讲，autograd 在执行运算时会记录一个图，记录所有创建数据的运算，从而得到一个有向无环图，其中叶子节点是输入张量，根节点是输出张量。通过从根节点到叶子节点追踪这个图，您可以使用链式法则自动计算梯度。

在内部，autograd 将此图表示为一个由 :class:`Function` 对象（实际上是表达式）组成的图，可以通过 :meth:`~torch.autograd.Function.apply` 方法来计算图求值的结果。在计算前向传播时，autograd 同时执行请求的计算，并构建一个表示梯度计算函数的图（每个 :class:`torch.Tensor` 的 ``.grad_fn`` 属性就是进入此图的入口点）。前向传播完成后，我们在反向传播中对此图进行求值以计算梯度。

需要注意的重要一点是，该图在每次迭代时都会从头开始重新创建，这正是允许使用任意 Python 控制流语句的原因，这些语句可以在每次迭代时改变图的整体形状和大小。您不必在启动训练之前编码所有可能的路径——您运行的就是您微分的。

.. _saved-tensors-doc:

保存的张量
^^^^^^^^^^^^^

某些运算需要在正向传播期间保存中间结果，以便执行反向传播。例如，函数 :math:`x\mapsto x^2` 会保存输入 :math:`x` 以计算梯度。

在定义自定义的 Python :class:`~torch.autograd.Function` 时，您可以使用 :func:`~torch.autograd.function._ContextMethodMixin.save_for_backward` 在正向传播期间保存张量，并使用 :attr:`~torch.autograd.function.Function.saved_tensors` 在反向传播期间检索它们。更多信息请参阅 :doc:`/notes/extending`。

对于 PyTorch 定义的运算（例如 :func:`torch.pow`），张量会根据需要自动保存。您可以通过查找以 ``_saved`` 为前缀的属性，来探索（出于教育或调试目的）某个 ``grad_fn`` 保存了哪些张量。

.. code::

    x = torch.randn(5, requires_grad=True)
    y = x.pow(2)
    print(x.equal(y.grad_fn._saved_self))  # True
    print(x is y.grad_fn._saved_self)  # True


在上面的代码中，``y.grad_fn._saved_self`` 指向与 `x` 相同的 Tensor 对象。但情况并非总是如此。例如：

.. code::

    x = torch.randn(5, requires_grad=True)
    y = x.exp()
    print(y.equal(y.grad_fn._saved_result))  # True
    print(y is y.grad_fn._saved_result)  # False


在底层，为了防止引用循环，PyTorch 在保存时对张量进行了 *打包*，并在读取时将其 *解包* 成不同的张量。这里，从访问 ``y.grad_fn._saved_result`` 得到的张量是与 ``y`` 不同的张量对象（但它们仍然共享相同的存储空间）。

一个张量是否会被打包成不同的张量对象，取决于它是否是自身 `grad_fn` 的输出，这是一个实现细节，可能会发生变化，用户不应依赖于此。

您可以使用 :ref:`saved-tensors-hooks-doc` 来控制 PyTorch 如何进行打包/解包。

.. _non-differentiable-func-grad:

不可微函数的梯度
------------------------------------------

使用自动微分进行梯度计算，仅当所使用的每个基本函数都是可微的时才有效。不幸的是，我们在实践中使用的许多函数不具备这个性质（例如，``relu`` 或 ``sqrt`` 在 ``0`` 处）。为了尽量减少不可微函数的影响，我们通过按顺序应用以下规则来定义基本运算的梯度：

#. 如果函数可微，因此在当前点存在梯度，则使用它。
#. 如果函数是凸函数（至少局部是），则使用最小范数的次梯度。
#. 如果函数是凹函数（至少局部是），则使用最小范数的超梯度（考虑 `-f(x)` 并应用上一点）。
#. 如果函数有定义，则通过连续性定义当前点的梯度（注意这里可能出现 ``inf``，例如 ``sqrt(0)``）。如果存在多个可能的值，则任意选择一个。
#. 如果函数未定义（例如 ``sqrt(-1)``、``log(-1)`` 或大多数输入为 ``NaN`` 的函数），则用作梯度的值是任意的（我们也可能引发错误，但这并不保证）。大多数函数会使用 ``NaN`` 作为梯度，但出于性能原因，某些函数会使用其他值（例如 ``log(-1)``）。
#. 如果函数不是确定性映射（即它不是 `数学函数`_），则它将被标记为不可微。如果在 ``no_grad`` 环境之外，在需要梯度的张量上使用它，这将在反向传播中导致错误。

.. _mathematical function: https://en.wikipedia.org/wiki/Function_%28mathematics%29

Autograd 中的除零操作
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 PyTorch 中执行除零操作（例如 ``x / 0``）时，前向传播将遵循 IEEE-754 浮点运算规则产生 ``inf`` 值。虽然这些 ``inf`` 值可以在计算最终损失之前被屏蔽掉（例如通过索引或掩码），但 autograd 系统仍然会跟踪并微分整个计算图，包括除零运算。

在反向传播过程中，这可能导致有问题的梯度表达式。例如：

.. code::

    x = torch.tensor([1., 1.], requires_grad=True)
    div = torch.tensor([0., 1.])

    y = x / div          # 结果为 [inf, 1]
    mask = div != 0      # [False, True]
    loss = y[mask].sum()
    loss.backward()
    print(x.grad)        # [nan, 1]，而不是 [0, 1]

在这个例子中，尽管我们只使用了掩码后的输出（排除了除以零的操作），但 autograd 仍然会通过完整的计算图计算梯度，包括除以零的操作。这导致被掩码元素的梯度为 ``nan``，可能引起训练不稳定。

为避免此问题，有几种推荐的方法：

1. 在除法前进行掩码：

.. code::

    x = torch.tensor([1., 1.], requires_grad=True)
    div = torch.tensor([0., 1.])

    mask = div != 0
    safe = torch.zeros_like(x)
    safe[mask] = x[mask] / div[mask]
    loss = safe.sum()
    loss.backward()      # 产生安全的梯度 [0, 1]

2. 使用 MaskedTensor（实验性 API）：

.. code::

    from torch.masked import as_masked_tensor

    x = torch.tensor([1., 1.], requires_grad=True)
    div = torch.tensor([0., 1.])

    y = x / div
    mask = div != 0
    loss = as_masked_tensor(y, mask).sum()
    loss.backward()      # 清晰地处理“未定义”与“零”梯度

关键原则是防止除以零的操作被记录在计算图中，而不是事后对其结果进行掩码。这确保了 autograd 只通过有效的操作计算梯度。

在处理可能产生 ``inf`` 或 ``nan`` 值的操作时，记住这种行为很重要，因为对输出进行掩码并不能防止有问题的梯度被计算。

.. _locally-disable-grad-doc:

局部禁用梯度计算
--------------------------------------

Python 提供了几种机制来局部禁用梯度计算：

要在整个代码块中禁用梯度，可以使用上下文管理器，如无梯度模式（no-grad mode）和推理模式（inference mode）。
为了更精细地从梯度计算中排除子图，可以设置张量的 ``requires_grad`` 字段。

除了讨论上述机制外，下面我们还描述了评估模式（:meth:`nn.Module.eval()`），这种方法并非用于禁用梯度计算，但由于其名称，常与上述三种机制混淆。

设置 ``requires_grad``
^^^^^^^^^^^^^^^^^^^^^^^^^

:attr:`requires_grad` 是一个标志，默认情况下为 false，*除非被包装在* ``nn.Parameter`` 中，它允许从梯度计算中精细地排除子图。它在正向传播和反向传播中都有效：

在正向传播期间，只有当至少一个输入张量需要梯度时，操作才会被记录在反向图中。
在反向传播期间（``.backward()``），只有具有 ``requires_grad=True`` 的叶子张量才会将梯度累积到它们的 ``.grad`` 字段中。

需要注意的是，尽管每个张量都有这个标志，但*设置*它只对叶子张量（没有 ``grad_fn`` 的张量，例如 ``nn.Module`` 的参数）有意义。
非叶子张量（具有 ``grad_fn`` 的张量）是那些与反向图相关联的张量。因此，它们的梯度将作为计算需要梯度的叶子张量梯度的中间结果而被需要。从这个定义可以清楚地看出，所有非叶子张量将自动具有 ``require_grad=True``。

设置 ``requires_grad`` 应该是控制模型的哪些部分参与梯度计算的主要方式，例如，如果在模型微调期间需要冻结预训练模型的部分参数。

要冻结模型的部分参数，只需对不希望更新的参数应用 ``.requires_grad_(False)``。如上所述，由于使用这些参数作为输入的计算在正向传播中不会被记录，因此在反向传播中它们的 ``.grad`` 字段不会被更新，因为它们首先不会成为反向图的一部分，正如所期望的那样。

由于这是一种常见模式，``requires_grad`` 也可以通过 :meth:`nn.Module.requires_grad_()` 在模块级别设置。
当应用于模块时，``.requires_grad_()`` 会对模块的所有参数生效（这些参数默认具有 ``requires_grad=True``）。

梯度模式
^^^^^^^^^^

除了设置 ``requires_grad`` 外，还有三种梯度模式可以从 Python 中选择，这些模式会影响 PyTorch 中的计算如何被 autograd 内部处理：默认模式（梯度模式）、无梯度模式和推理模式，所有这些都可以通过上下文管理器和装饰器切换。

.. list-table::
   :widths: 50 50 50 50 50
   :header-rows: 1

   * - 模式
     - 是否将操作排除在反向图记录之外
     - 是否跳过额外的 autograd 跟踪开销
     - 在该模式下创建的张量稍后是否可用于梯度模式
     - 示例
   * - 默认模式
     -
     -
     - ✓
     - 正向传播
   * - 无梯度模式
     - ✓
     -
     - ✓
     - 优化器更新
   * - 推理模式
     - ✓
     - ✓
     -
     - 数据处理、模型评估

默认模式（梯度模式）
^^^^^^^^^^^^^^^^^^^^^^^^

“默认模式”是当我们没有启用其他模式（如无梯度模式和推理模式）时隐式处于的模式。为了与“无梯度模式”区分，默认模式有时也称为“梯度模式”。

关于默认模式，最重要的是要知道它是唯一一个 ``requires_grad`` 生效的模式。在其他两种模式中，``requires_grad`` 总是被覆盖为 ``False``。

无梯度模式
^^^^^^^^^^^^

无梯度模式下的计算行为就好像所有输入都不需要梯度一样。
换句话说，无梯度模式下的计算永远不会被记录在反向图中，即使有输入具有 ``require_grad=True``。

当您需要执行不应被 autograd 记录的操作，但稍后仍希望在梯度模式下使用这些计算的输出时，请启用无梯度模式。此上下文管理器可以方便地禁用代码块或函数的梯度，而无需临时将张量设置为 `requires_grad=False`，然后再改回 `True`。

例如，在编写优化器时，无梯度模式可能很有用：在执行训练更新时，您希望原地更新参数，而不被 autograd 记录。您还打算在下一个前向传播中，在梯度模式下使用更新后的参数进行计算。

:ref:`nn-init-doc` 中的实现也依赖于无梯度模式来初始化参数，以避免在就地更新初始化参数时被 autograd 跟踪。

推理模式
^^^^^^^^^^^^^^

推理模式是无梯度模式的极端版本。与无梯度模式一样，推理模式下的计算不会被记录在后向图中，但启用推理模式将使 PyTorch 进一步加速您的模型。这种更好的运行时性能伴随着一个缺点：在推理模式下创建的张量在退出推理模式后，将无法用于被 autograd 记录的计算中。

当您执行与 autograd 没有交互的计算，并且不打算在推理模式下创建的张量用于任何稍后被 autograd 记录的计算时，请启用推理模式。

建议您在不需要 autograd 跟踪的代码部分（例如，数据处理和模型评估）尝试推理模式。如果它开箱即用适用于您的用例，那就是免费的性能提升。如果在启用推理模式后遇到错误，请检查您是否在退出推理模式后，在 autograd 记录的计算中使用了推理模式下创建的张量。如果在您的情况下无法避免这种使用，您可以随时切换回无梯度模式。

有关推理模式的详细信息，请参阅 `Inference Mode <https://pytorch.org/cppdocs/notes/inference_mode.html>`_。

有关推理模式的实现细节，请参阅 `RFC-0011-InferenceMode <https://github.com/pytorch/rfcs/pull/17>`_。

评估模式（`nn.Module.eval()`）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

评估模式不是一种局部禁用梯度计算的机制。尽管如此，它仍被包含在此处，因为它有时会被误认为是这样的机制。

从功能上讲，`module.eval()`（或等效的 `module.train(False)`）与无梯度模式和推理模式完全正交。`model.eval()` 如何影响您的模型完全取决于模型中使用的特定模块以及它们是否定义了任何训练模式特定的行为。

如果您的模型依赖于可能根据训练模式表现不同的模块，例如 :class:`torch.nn.Dropout` 和 :class:`torch.nn.BatchNorm2d`，以避免在验证数据上更新 BatchNorm 的运行统计信息，您有责任调用 `model.eval()` 和 `model.train()`。

建议您在训练时始终使用 `model.train()`，在评估模型（验证/测试）时始终使用 `model.eval()`，即使您不确定您的模型是否有训练模式特定的行为，因为您使用的模块可能会被更新以在训练和评估模式下表现不同。

使用 autograd 进行原地操作
---------------------------------

在 autograd 中支持原地操作是一个难题，我们在大多数情况下不鼓励使用它们。Autograd 积极的缓冲区释放和重用使其非常高效，并且很少有情况下原地操作能显著降低内存使用量。除非您在巨大的内存压力下操作，否则您可能永远不需要使用它们。

有两个主要原因限制了原地操作的适用性：

1. 原地操作可能会覆盖计算梯度所需的值。

2. 每个原地操作都需要重写计算图的实现。非原地版本只是分配新对象并保留对旧图的引用，而原地操作则需要将所有输入的创建者更改为代表此操作的 :class:`Function`。这可能很棘手，尤其是当有许多张量引用相同的存储时（例如，通过索引或转置创建），并且如果修改输入的存储被任何其他 :class:`Tensor` 引用，原地函数将引发错误。

原地正确性检查
^^^^^^^^^^^^^^^^^^^^^^^^^^^

每个张量都维护一个版本计数器，每次在任何操作中被标记为脏时都会递增。当 Function 保存任何张量用于反向传播时，其包含张量的版本计数器也会被保存。一旦您访问 `self.saved_tensors`，就会进行检查，如果它大于保存的值，则会引发错误。这确保了如果您使用原地函数且没有看到任何错误，您可以确信计算的梯度是正确的。

多线程 Autograd
----------------------

autograd 引擎负责运行计算反向传播所需的所有反向操作。本节将描述所有细节，帮助您在多线程环境中充分利用它。（这仅适用于 PyTorch 1.6+，因为之前版本的行为不同。）

用户可以使用多线程代码（例如 Hogwild 训练）训练模型，并且不会阻塞并发的反向计算，示例代码可能如下：

.. code::

    # 定义在不同线程中使用的训练函数
    def train_fn():
        x = torch.ones(5, 5, requires_grad=True)
        # 前向传播
        y = (x + 3) * (x + 4) * 0.5
        # 反向传播
        y.sum().backward()
        # 可能的优化器更新

# 用户编写自己的线程代码来驱动 train_fn
    threads = []
    for _ in range(10):
        p = threading.Thread(target=train_fn, args=())
        p.start()
        threads.append(p)

    for p in threads:
        p.join()


请注意用户应了解的一些行为：

CPU 上的并发性
^^^^^^^^^^^^^^^^^^

当你在 CPU 上的多个线程中通过 Python 或 C++ API 运行 ``backward()`` 或 ``grad()`` 时，你期望看到额外的并发性，而不是在执行期间以特定顺序序列化所有反向传播调用（这是 PyTorch 1.6 之前的行为）。

非确定性
^^^^^^^^^^^^^^^

如果你从多个线程并发调用 ``backward()`` 并且有共享的输入（例如 Hogwild CPU 训练），那么应该预期会出现非确定性。这可能发生，因为参数会自动在线程间共享，因此多个线程可能在梯度累积期间访问并尝试累加相同的 ``.grad`` 属性。这在技术上是不安全的，可能导致竞态条件，并且结果可能无效。

开发具有共享参数的多线程模型的用户应牢记线程模型，并理解上述描述的问题。

可以使用函数式 API :func:`torch.autograd.grad` 来计算梯度，而不是使用 ``backward()``，以避免非确定性。

图保留
^^^^^^^^^^^^^^^

如果 autograd 图的一部分在线程间共享，即先以单线程运行前向传播的第一部分，然后在多个线程中运行第二部分，那么图的第一部分是共享的。在这种情况下，不同线程在同一图上执行 ``grad()`` 或 ``backward()`` 可能会出现在一个线程执行过程中销毁图的问题，而另一个线程在这种情况下会崩溃。Autograd 将向用户报错，类似于在没有设置 ``retain_graph=True`` 的情况下调用两次 ``backward()``，并让用户知道他们应该使用 ``retain_graph=True``。

Autograd 节点上的线程安全性
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

由于 Autograd 允许调用者线程驱动其反向传播执行以实现潜在的并行性，因此确保在 CPU 上并行调用 ``backward()`` 且共享部分/整个 GraphTask 时的线程安全性非常重要。

自定义的 Python ``autograd.Function`` 由于 GIL 的存在是自动线程安全的。对于内置的 C++ Autograd 节点（例如 AccumulateGrad、CopySlices）和自定义的 ``autograd::Function``，Autograd 引擎使用线程互斥锁来确保可能具有状态写入/读取的 autograd 节点的线程安全性。

C++ 钩子上的非线程安全性
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Autograd 依赖用户编写线程安全的 C++ 钩子。如果你希望钩子在多线程环境中正确应用，你需要编写适当的线程锁定代码以确保钩子是线程安全的。

.. _complex_autograd-doc:

复数自动微分
----------------------------

简短版本：

- 当你使用 PyTorch 对任何具有复数定义域和/或值域的函数 :math:`f(z)` 进行微分时，梯度计算基于一个假设：该函数是一个更大的实值损失函数 :math:`g(input)=L` 的一部分。计算出的梯度是 :math:`\frac{\partial L}{\partial z^*}`（注意 z 的共轭），其负方向正是梯度下降算法中使用的最陡下降方向。因此，存在一条可行的路径，使现有的优化器能够直接用于复数参数。
- 此约定与 TensorFlow 的复数微分约定匹配，但与 JAX 不同（JAX 计算 :math:`\frac{\partial L}{\partial z}`）。
- 如果你有一个内部使用复数运算的实到实函数，这里的约定无关紧要：你总是会得到与仅使用实数运算实现时相同的结果。

如果你对数学细节感到好奇，或者想知道如何在 PyTorch 中定义复数导数，请继续阅读。

什么是复数导数？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

复数可微性的数学定义采用了导数的极限定义，并将其推广到复数运算。考虑一个函数 :math:`f: ℂ → ℂ`，

    .. math::
        f(z=x+yj) = u(x, y) + v(x, y)j

其中 :math:`u` 和 :math:`v` 是两个变量的实值函数，:math:`j` 是虚数单位。

使用导数定义，我们可以写出：

    .. math::
        f'(z) = \lim_{h \to 0, h \in C} \frac{f(z+h) - f(z)}{h}

为了使这个极限存在，不仅 :math:`u` 和 :math:`v` 必须是实可微的，而且 :math:`f` 还必须满足柯西-黎曼 `方程 <https://en.wikipedia.org/wiki/Cauchy%E2%80%93Riemann_equations>`_。换句话说：为实部和虚部步长 (:math:`h`) 计算的极限必须相等。这是一个更具限制性的条件。

复数可微函数通常称为全纯函数。它们行为良好，具有你从实可微函数中看到的所有优良性质，但在优化世界中几乎没有实际用途。对于优化问题，研究界只使用实值目标函数，因为复数不属于任何有序域，因此具有复数值的损失函数没有太大意义。

事实证明，没有有趣的实际值目标函数满足柯西-黎曼方程。因此，全纯函数的理论不能用于优化，大多数人因此使用 Wirtinger 微积分。

Wirtinger 微积分登场...
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

那么，我们拥有这套关于复可微性和全纯函数的伟大理论，却完全无法使用其中的任何内容，因为许多常用函数都不是全纯的。可怜的数学家该怎么办呢？好吧，Wirtinger 观察到，即使 :math:`f(z)` 不是全纯的，也可以将其重写为一个总是全纯的双变量函数 :math:`f(z, z*)`。这是因为 :math:`z` 分量的实部和虚部可以用 :math:`z` 和 :math:`z^*` 表示为：

    .. math::
        \begin{aligned}
            \mathrm{Re}(z) &= \frac {z + z^*}{2} \\
            \mathrm{Im}(z) &= \frac {z - z^*}{2j}
        \end{aligned}

Wirtinger 微积分建议转而研究 :math:`f(z, z^*)`，如果 :math:`f` 是实可微的，那么它保证是全纯的（另一种思考方式是将其视为坐标系的变换，从 :math:`f(x, y)` 变为 :math:`f(z, z^*)`。）这个函数具有偏导数 :math:`\frac{\partial }{\partial z}` 和 :math:`\frac{\partial}{\partial z^{*}}`。我们可以使用链式法则在这些偏导数与关于 :math:`z` 的实部和虚部分量的偏导数之间建立关系。

    .. math::
        \begin{aligned}
            \frac{\partial }{\partial x} &= \frac{\partial z}{\partial x} * \frac{\partial }{\partial z} + \frac{\partial z^*}{\partial x} * \frac{\partial }{\partial z^*} \\
                                         &= \frac{\partial }{\partial z} + \frac{\partial }{\partial z^*}   \\
            \\
            \frac{\partial }{\partial y} &= \frac{\partial z}{\partial y} * \frac{\partial }{\partial z} + \frac{\partial z^*}{\partial y} * \frac{\partial }{\partial z^*} \\
                                         &= 1j * \left(\frac{\partial }{\partial z} - \frac{\partial }{\partial z^*}\right)
        \end{aligned}

由上述方程，我们得到：

    .. math::
        \begin{aligned}
            \frac{\partial }{\partial z} &= 1/2 * \left(\frac{\partial }{\partial x} - 1j * \frac{\partial }{\partial y}\right)   \\
            \frac{\partial }{\partial z^*} &= 1/2 * \left(\frac{\partial }{\partial x} + 1j * \frac{\partial }{\partial y}\right)
        \end{aligned}

这就是你在 `维基百科 <https://en.wikipedia.org/wiki/Wirtinger_derivatives>`_ 上能找到的 Wirtinger 微积分的经典定义。

这一变换带来了许多美妙的结果。

- 其一，柯西-黎曼方程简化为仅仅表示 :math:`\frac{\partial f}{\partial z^*} = 0`（也就是说，函数 :math:`f` 可以完全用 :math:`z` 表示，而不涉及 :math:`z^*`）。
- 另一个重要（且有些反直觉）的结果是，正如我们稍后将看到的，当我们对实值损失进行优化时，更新变量应采取的步长由 :math:`\frac{\partial Loss}{\partial z^*}` 给出（而不是 :math:`\frac{\partial Loss}{\partial z}`）。

更多阅读，请查看：https://arxiv.org/pdf/0906.4835.pdf

Wirtinger 微积分在优化中有何用处？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

音频和其他领域的研究人员更常用梯度下降来优化具有复变量的实值损失函数。通常，这些人将实部和虚部视为可以单独更新的通道。对于步长 :math:`\alpha/2` 和损失 :math:`L`，我们可以在 :math:`ℝ^2` 中写出以下方程：

    .. math::
        \begin{aligned}
            x_{n+1} &= x_n - (\alpha/2) * \frac{\partial L}{\partial x}  \\
            y_{n+1} &= y_n - (\alpha/2) * \frac{\partial L}{\partial y}
        \end{aligned}

这些方程如何转换到复数空间 :math:`ℂ` 中呢？

    .. math::
        \begin{aligned}
            z_{n+1} &= x_n - (\alpha/2) * \frac{\partial L}{\partial x} + 1j * (y_n - (\alpha/2) * \frac{\partial L}{\partial y}) \\
                    &= z_n - \alpha * 1/2 * \left(\frac{\partial L}{\partial x} + j \frac{\partial L}{\partial y}\right) \\
                    &= z_n - \alpha * \frac{\partial L}{\partial z^*}
        \end{aligned}

发生了一些非常有趣的事情：Wirtinger 微积分告诉我们，我们可以将上述复变量更新公式简化为仅涉及共轭 Wirtinger 导数 :math:`\frac{\partial L}{\partial z^*}`，从而精确地给出了我们在优化中采取的步骤。

因为共轭 Wirtinger 导数为实值损失函数提供了精确的更新步骤，所以当您对一个具有实值损失的函数进行微分时，PyTorch 会给出这个导数。

PyTorch 如何计算共轭 Wirtinger 导数？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

通常，我们的导数公式将 `grad_output` 作为输入，代表我们已经计算过的传入向量-雅可比积，即 :math:`\frac{\partial L}{\partial s^*}`，其中 :math:`L` 是整个计算过程的损失（产生实值损失），而 :math:`s` 是我们函数的输出。这里的目标是计算 :math:`\partial L}{\partial z^*}`，其中 :math:`z` 是函数的输入。事实证明，在实值损失的情况下，我们*仅*计算 :math:`\frac{\partial L}{\partial s^*}` 就足够了，尽管链式法则意味着我们还需要能够访问 :math:`\frac{\partial L}{\partial s}`。如果您想跳过这个推导，请查看本节最后一个方程，然后跳到下一节。

让我们继续处理定义为 :math:`f(z) = f(x+yj) = u(x, y) + v(x, y)j` 的 :math:`f: ℂ → ℂ`。如上所述，autograd 的梯度约定围绕实值损失函数的优化展开，因此我们假设 :math:`f` 是更大的实值损失函数 :math:`g` 的一部分。使用链式法则，我们可以写出：

.. math::
        \frac{\partial L}{\partial z^*} = \frac{\partial L}{\partial u} * \frac{\partial u}{\partial z^*} + \frac{\partial L}{\partial v} * \frac{\partial v}{\partial z^*}
        :label: [1]

现在使用 Wirtinger 导数的定义，我们可以写出：

    .. math::
        \begin{aligned}
            \frac{\partial L}{\partial s} = 1/2 * \left(\frac{\partial L}{\partial u} - \frac{\partial L}{\partial v} j\right) \\
            \frac{\partial L}{\partial s^*} = 1/2 * \left(\frac{\partial L}{\partial u} + \frac{\partial L}{\partial v} j\right)
        \end{aligned}

这里需要注意的是，由于 :math:`u` 和 :math:`v` 是实函数，并且根据我们假设 :math:`f` 是实值函数的一部分，:math:`L` 也是实的，因此我们有：

    .. math::
        \left( \frac{\partial L}{\partial s} \right)^* = \frac{\partial L}{\partial s^*}
        :label: [2]

即，:math:`\frac{\partial L}{\partial s}` 等于 :math:`grad\_output^*`。

求解上述方程得到 :math:`\frac{\partial L}{\partial u}` 和 :math:`\frac{\partial L}{\partial v}`，我们得到：

    .. math::
        \begin{aligned}
            \frac{\partial L}{\partial u} = \frac{\partial L}{\partial s} + \frac{\partial L}{\partial s^*} \\
            \frac{\partial L}{\partial v} = 1j * \left(\frac{\partial L}{\partial s} - \frac{\partial L}{\partial s^*}\right)
        \end{aligned}
        :label: [3]

将 :eq:`[3]` 代入 :eq:`[1]`，我们得到：

    .. math::
        \begin{aligned}
            \frac{\partial L}{\partial z^*} &= \left(\frac{\partial L}{\partial s} + \frac{\partial L}{\partial s^*}\right) * \frac{\partial u}{\partial z^*} + 1j * \left(\frac{\partial L}{\partial s} - \frac{\partial L}{\partial s^*}\right) * \frac{\partial v}{\partial z^*}  \\
                                            &= \frac{\partial L}{\partial s} * \left(\frac{\partial u}{\partial z^*} + \frac{\partial v}{\partial z^*} j\right) + \frac{\partial L}{\partial s^*} * \left(\frac{\partial u}{\partial z^*} - \frac{\partial v}{\partial z^*} j\right)  \\
                                            &= \frac{\partial L}{\partial s} * \frac{\partial (u + vj)}{\partial z^*} + \frac{\partial L}{\partial s^*} * \frac{\partial (u + vj)^*}{\partial z^*}  \\
                                            &= \frac{\partial L}{\partial s} * \frac{\partial s}{\partial z^*} + \frac{\partial L}{\partial s^*} * \frac{\partial s^*}{\partial z^*}    \\
        \end{aligned}

使用 :eq:`[2]`，我们得到：

    .. math::
        \begin{aligned}
            \frac{\partial L}{\partial z^*} &= \left(\frac{\partial L}{\partial s^*}\right)^* * \frac{\partial s}{\partial z^*} + \frac{\partial L}{\partial s^*} * \left(\frac{\partial s}{\partial z}\right)^*  \\
                                            &= \boxed{ (grad\_output)^* * \frac{\partial s}{\partial z^*} + grad\_output * \left(\frac{\partial s}{\partial z}\right)^* }       \\
        \end{aligned}
        :label: [4]

最后一个方程对于编写自定义梯度非常重要，因为它将我们的导数公式分解为一个更简单、易于手动计算的公式。

如何为复变函数编写自定义导数公式？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

上面的框内方程给出了所有复变函数导数的通用公式。然而，我们仍然需要计算 :math:`\frac{\partial s}{\partial z}` 和 :math:`\frac{\partial s}{\partial z^*}`。有两种方法可以做到这一点：

    - 第一种方法是直接使用 Wirtinger 导数的定义，并通过 :math:`\frac{\partial s}{\partial x}` 和 :math:`\frac{\partial s}{\partial y}`（你可以用常规方式计算）来计算 :math:`\frac{\partial s}{\partial z}` 和 :math:`\frac{\partial s}{\partial z^*}`。
    - 第二种方法是使用变量替换技巧，将 :math:`f(z)` 重写为双变量函数 :math:`f(z, z^*)`，并通过将 :math:`z` 和 :math:`z^*` 视为独立变量来计算共轭 Wirtinger 导数。这通常更容易；例如，如果所讨论的函数是全纯的，则只会用到 :math:`z`（并且 :math:`\frac{\partial s}{\partial z^*}` 将为零）。

让我们以函数 :math:`f(z = x + yj) = c * z = c * (x+yj)` 为例，其中 :math:`c \in ℝ`。

使用第一种方法计算 Wirtinger 导数，我们有：

.. math::
    \begin{aligned}
        \frac{\partial s}{\partial z} &= 1/2 * \left(\frac{\partial s}{\partial x} - \frac{\partial s}{\partial y} j\right) \\
                                      &= 1/2 * (c - (c * 1j) * 1j)  \\
                                      &= c                          \\
        \\
        \\
        \frac{\partial s}{\partial z^*} &= 1/2 * \left(\frac{\partial s}{\partial x} + \frac{\partial s}{\partial y} j\right) \\
                                        &= 1/2 * (c + (c * 1j) * 1j)  \\
                                        &= 0                          \\
    \end{aligned}

使用 :eq:`[4]`，并设 `grad\_output = 1.0`（这是在 PyTorch 中对标量输出调用 :func:`backward` 时使用的默认梯度输出值），我们得到：

    .. math::
        \frac{\partial L}{\partial z^*} = 1 * 0 + 1 * c = c

使用第二种方法计算 Wirtinger 导数，我们直接得到：

    .. math::
        \begin{aligned}
           \frac{\partial s}{\partial z} &= \frac{\partial (c*z)}{\partial z}       \\
                                         &= c                                       \\
            \frac{\partial s}{\partial z^*} &= \frac{\partial (c*z)}{\partial z^*}       \\
                                         &= 0
        \end{aligned}

再次使用 :eq:`[4]`，我们得到 :math:`\frac{\partial L}{\partial z^*} = c`。如你所见，第二种方法涉及的计算更少，对于快速计算更为方便。

跨域函数的情况如何？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

有些函数将复数输入映射到实数输出，反之亦然。
这些函数构成了 :eq:`[4]` 的一个特例，我们可以使用链式法则推导：

    - 对于 :math:`f: ℂ → ℝ`，我们得到：

        .. math::
            \frac{\partial L}{\partial z^*} = 2 * grad\_output * \frac{\partial s}{\partial z^{*}}

    - 对于 :math:`f: ℝ → ℂ`，我们得到：

        .. math::
            \frac{\partial L}{\partial z^*} = 2 * \mathrm{Re}(grad\_output^* * \frac{\partial s}{\partial z^{*}})

.. _saved-tensors-hooks-doc:

已保存张量的钩子
----------------

你可以通过定义一对 ``pack_hook`` / ``unpack_hook`` 钩子来控制 :ref:`已保存张量如何打包/解包 <saved-tensors-doc>`。
``pack_hook`` 函数应以一个张量作为其唯一参数，但可以返回任何 Python 对象（例如另一个张量、一个元组，甚至包含文件名的字符串）。
``unpack_hook`` 函数以 ``pack_hook`` 的输出作为其唯一参数，并应返回一个用于反向传播的张量。
``unpack_hook`` 返回的张量只需要与传递给 ``pack_hook`` 作为输入的张量具有相同的内容。
特别是，任何与自动求导相关的元数据都可以忽略，因为它们将在解包过程中被覆盖。

这种钩子对的一个示例如下：

.. code::

    class SelfDeletingTempFile():
        def __init__(self):
            self.name = os.path.join(tmp_dir, str(uuid.uuid4()))

        def __del__(self):
            os.remove(self.name)

    def pack_hook(tensor):
        temp_file = SelfDeletingTempFile()
        torch.save(tensor, temp_file.name)
        return temp_file

    def unpack_hook(temp_file):
        return torch.load(temp_file.name)

注意，``unpack_hook`` 不应删除临时文件，因为它可能被多次调用：只要返回的 `SelfDeletingTempFile` 对象存在，临时文件就应该保持存在。
在上面的例子中，我们通过在不再需要时（在 `SelfDeletingTempFile` 对象被删除时）关闭临时文件来防止其泄漏。

.. note::

    我们保证 ``pack_hook`` 只会被调用一次，但 ``unpack_hook`` 可能会根据反向传播的需要被调用多次，并且我们期望它每次返回相同的数据。

.. warning::

    禁止对任何函数的输入执行原地操作，因为这可能导致意外的副作用。
    如果 pack 钩子的输入被原地修改，PyTorch 将抛出错误，但不会捕获 unpack 钩子的输入被原地修改的情况。

为已保存张量注册钩子
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

你可以通过调用 :class:`SavedTensor` 对象上的 :meth:`~torch.autograd.SavedTensor.register_hooks` 方法来为已保存张量注册一对钩子。
这些对象作为 ``grad_fn`` 的属性暴露，并以 ``_raw_saved_`` 前缀开头。

.. code::

    x = torch.randn(5, requires_grad=True)
    y = x.pow(2)
    y.grad_fn._raw_saved_self.register_hooks(pack_hook, unpack_hook)

``pack_hook`` 方法在钩子对注册后立即被调用。
``unpack_hook`` 方法在每次需要访问已保存张量时被调用，无论是通过 ``y.grad_fn._saved_self`` 还是在反向传播期间。

.. warning::

    如果你在已保存张量被释放后（即调用 backward 后）仍持有对 :class:`SavedTensor` 的引用，则禁止调用其 :meth:`~torch.autograd.SavedTensor.register_hooks` 方法。
    PyTorch 在大多数情况下会抛出错误，但在某些情况下可能无法做到，并可能导致未定义行为。

为已保存张量注册默认钩子
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

或者，你可以使用上下文管理器 :class:`~torch.autograd.graph.saved_tensors_hooks` 来注册一对钩子，该钩子将应用于在该上下文中创建的所有已保存张量。

示例：

.. code::

    # 仅将大小 >= 1000 的张量保存到磁盘
    SAVE_ON_DISK_THRESHOLD = 1000

    def pack_hook(x):
        if x.numel() < SAVE_ON_DISK_THRESHOLD:
            return x.detach()
        temp_file = SelfDeletingTempFile()
        torch.save(tensor, temp_file.name)
        return temp_file

    def unpack_hook(tensor_or_sctf):
        if isinstance(tensor_or_sctf, torch.Tensor):
            return tensor_or_sctf
        return torch.load(tensor_or_sctf.name)

    class Model(nn.Module):
        def forward(self, x):
            with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
              # ... 计算输出
              output = x
            return output

    model = Model()
    net = nn.DataParallel(model)

使用此上下文管理器定义的钩子是线程局部的。
因此，以下代码不会产生预期效果，因为钩子不会通过 `DataParallel`。

.. code::

      # 错误示例

      net = nn.DataParallel(model)
      with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
          output = net(input)

注意，使用这些钩子会禁用所有旨在减少张量对象创建的优化。
例如：

.. code::

    with torch.autograd.graph.saved_tensors_hooks(lambda x: x.detach(), lambda x: x):
        x = torch.randn(5, requires_grad=True)
        y = x * x

如果没有钩子，``x``、``y.grad_fn._saved_self`` 和 ``y.grad_fn._saved_other`` 都引用同一个张量对象。
使用钩子后，PyTorch 会将 `x` 打包并解包成两个新的张量对象，它们与原始的 `x` 共享相同的存储（不执行复制）。

.. _backward-hooks-execution:

反向钩子执行
------------------------

本节将讨论不同钩子在何时触发或不触发，然后讨论它们的触发顺序。
涉及的钩子包括：通过 :meth:`torch.Tensor.register_hook` 注册到 Tensor 的 backward 钩子、
通过 :meth:`torch.Tensor.register_post_accumulate_grad_hook` 注册到 Tensor 的梯度累加后钩子、
通过 :meth:`torch.autograd.graph.Node.register_hook` 注册到 Node 的后钩子，
以及通过 :meth:`torch.autograd.graph.Node.register_prehook` 注册到 Node 的前钩子。

特定钩子是否会触发
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

通过 :meth:`torch.Tensor.register_hook` 注册到 Tensor 的钩子会在计算该 Tensor 的梯度时执行。
（注意这并不要求执行该 Tensor 的 grad_fn。例如，如果 Tensor 作为 ``inputs`` 参数的一部分传递给 :func:`torch.autograd.grad`，
该 Tensor 的 grad_fn 可能不会执行，但注册到该 Tensor 的钩子始终会执行。）

通过 :meth:`torch.Tensor.register_post_accumulate_grad_hook` 注册到 Tensor 的钩子会在该 Tensor 的梯度累加后执行，
这意味着 Tensor 的 grad 字段已被设置。而通过 :meth:`torch.Tensor.register_hook` 注册的钩子在梯度计算过程中运行，
通过 :meth:`torch.Tensor.register_post_accumulate_grad_hook` 注册的钩子仅在反向传播结束时由 autograd 更新 Tensor 的 grad 字段后触发。
因此，梯度累加后钩子只能注册到叶子 Tensor。在非叶子 Tensor 上通过 :meth:`torch.Tensor.register_post_accumulate_grad_hook` 注册钩子会报错，
即使调用 `backward(retain_graph=True)` 也是如此。

使用 :meth:`torch.autograd.graph.Node.register_hook` 或 :meth:`torch.autograd.graph.Node.register_prehook`
注册到 :class:`torch.autograd.graph.Node` 的钩子仅在其注册的 Node 被执行时触发。

特定 Node 是否执行可能取决于反向传播是通过 :func:`torch.autograd.grad` 还是 :func:`torch.autograd.backward` 调用的。
具体来说，当你在一个 Node 上注册钩子，而该 Node 对应作为 ``inputs`` 参数传递给 :func:`torch.autograd.grad` 或 :func:`torch.autograd.backward` 的 Tensor 时，
你应当注意这些差异。

如果使用 :func:`torch.autograd.backward`，无论是否指定了 ``inputs`` 参数，上述所有钩子都会执行。
这是因为 `.backward()` 会执行所有 Node，即使它们对应被指定为输入的 Tensor。
（注意，这种对应作为 ``inputs`` 传递的 Tensor 的额外 Node 的执行通常是不必要的，但仍会执行。此行为可能更改；你不应依赖它。）

另一方面，如果使用 :func:`torch.autograd.grad`，注册到对应传递给 ``input`` 的 Tensor 的 Node 的反向钩子可能不会执行，
因为这些 Node 不会被执行，除非有另一个输入依赖于该 Node 的梯度结果。

不同钩子的触发顺序
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

事件发生的顺序如下：

#. 执行注册到 Tensor 的钩子
#. 执行注册到 Node 的前钩子（如果 Node 被执行）
#. 为保留梯度的 Tensor 更新 ``.grad`` 字段
#. 执行 Node（遵循上述规则）
#. 对于累积了 ``.grad`` 的叶子 Tensor，执行梯度累加后钩子
#. 执行注册到 Node 的后钩子（如果 Node 被执行）

如果在同一 Tensor 或 Node 上注册了多个同类型钩子，它们会按照注册顺序执行。
后执行的钩子可以观察到先前钩子对梯度所做的修改。

特殊钩子
^^^^^^^^^^^^^

:func:`torch.autograd.graph.register_multi_grad_hook` 是通过注册到 Tensor 的钩子实现的。
每个独立的 Tensor 钩子按照上述 Tensor 钩子顺序触发，当最后一个 Tensor 梯度计算完成后调用注册的多梯度钩子。

:meth:`torch.nn.modules.module.register_module_full_backward_hook` 是通过注册到 Node 的钩子实现的。
在前向计算过程中，钩子被注册到对应模块输入和输出的 grad_fn。由于模块可能接收多个输入并返回多个输出，
首先会在前向计算前对模块输入应用一个虚拟的自定义 autograd Function，并在前向输出返回前对模块输出应用，
以确保这些 Tensor 共享单个 grad_fn，然后我们可以将钩子附加到该 grad_fn。

Tensor 被原地修改时 Tensor 钩子的行为
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

通常注册到 Tensor 的钩子接收输出相对于该 Tensor 的梯度，其中 Tensor 的值取为计算反向传播时的值。

但是，如果你注册钩子到 Tensor，然后原地修改该 Tensor，在就地修改前注册的钩子同样接收输出相对于该 Tensor 的梯度，
但 Tensor 的值取为原地修改前的值。

如果你希望前一种情况的行为，应在完成所有原地修改后再将钩子注册到 Tensor。例如：

.. code::

    t = torch.tensor(1., requires_grad=True).sin()
    t.cos_()
    t.register_hook(fn)
    t.backward()

此外，了解以下内部机制会很有帮助：
当钩子注册到张量时，它们实际上会永久绑定到该张量的 grad_fn。
因此，如果该张量随后被原地修改，即使张量现在有了新的 grad_fn，
在修改前注册的钩子仍会与旧的 grad_fn 保持关联。
例如，当 autograd 引擎在计算图中访问该张量的旧 grad_fn 时，这些钩子仍会被触发。