.. _amp-examples:

自动混合精度示例
=======================================

.. currentmodule:: torch.amp

通常，“自动混合精度训练”是指同时使用 :class:`torch.autocast` 和 :class:`torch.amp.GradScaler` 进行训练。

:class:`torch.autocast` 的实例为选定区域启用自动转换。自动转换会自动选择操作的精度，以提高性能，同时保持准确性。

:class:`torch.amp.GradScaler` 的实例有助于方便地执行梯度缩放步骤。梯度缩放通过最小化梯度下溢，提高具有 ``float16`` 梯度（在 CUDA 和 XPU 上默认）的网络的收敛性，如 :ref:`此处<gradient-scaling>` 所述。

:class:`torch.autocast` 和 :class:`torch.amp.GradScaler` 是模块化的。在下面的示例中，每个都按照其各自的文档建议使用。

（此处的示例仅为说明。有关可运行的详细步骤，请参阅
`自动混合精度教程 <https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html>`_。）

.. contents:: :local:

典型的混合精度训练
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    # 以默认精度创建模型和优化器
    model = Net().cuda()
    optimizer = optim.SGD(model.parameters(), ...)

    # 在训练开始时创建一次 GradScaler。
    scaler = GradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()

            # 使用 autocast 运行前向传播。
            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(input)
                loss = loss_fn(output, target)

            # 缩放损失。对缩放后的损失调用 backward() 以创建缩放后的梯度。
            # 不建议在 autocast 下进行反向传播。
            # 反向操作以 autocast 为相应前向操作选择的相同数据类型运行。
            scaler.scale(loss).backward()

            # scaler.step() 首先反缩放优化器分配参数的梯度。
            # 如果这些梯度不包含 inf 或 NaN，则调用 optimizer.step()，
            # 否则跳过 optimizer.step()。
            scaler.step(optimizer)

            # 更新下一次迭代的缩放因子。
            scaler.update()

.. _working-with-unscaled-gradients:

处理未缩放的梯度
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

所有由 ``scaler.scale(loss).backward()`` 产生的梯度都是缩放后的。如果你希望在 ``backward()`` 和 ``scaler.step(optimizer)`` 之间修改或检查参数的 ``.grad`` 属性，你应该先对它们进行反缩放。例如，梯度裁剪操作一组梯度，使其全局范数（参见 :func:`torch.nn.utils.clip_grad_norm_`）或最大幅度（参见 :func:`torch.nn.utils.clip_grad_value_`）:math:`<=` 某个用户设定的阈值。如果你尝试在*未*反缩放的情况下进行裁剪，梯度的范数/最大幅度也将被缩放，因此你设定的阈值（本意是*未缩放*梯度的阈值）将是无效的。

``scaler.unscale_(optimizer)`` 对 ``optimizer`` 分配参数持有的梯度进行反缩放。如果你的模型包含分配给其他优化器（例如 ``optimizer2``）的参数，你可以单独调用 ``scaler.unscale_(optimizer2)`` 来反缩放这些参数的梯度。

梯度裁剪
-----------------

在裁剪之前调用 ``scaler.unscale_(optimizer)`` 可以让你像往常一样裁剪未缩放的梯度：:

    scaler = GradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(input)
                loss = loss_fn(output, target)
            scaler.scale(loss).backward()

            # 原地反缩放优化器分配参数的梯度
            scaler.unscale_(optimizer)

            # 由于优化器分配参数的梯度已反缩放，可以像往常一样裁剪：
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # 优化器的梯度已经反缩放，因此 scaler.step 不会再次反缩放它们，
            # 但如果梯度包含 inf 或 NaN，它仍会跳过 optimizer.step()。
            scaler.step(optimizer)

            # 更新下一次迭代的缩放因子。
            scaler.update()

``scaler`` 会记录本次迭代中已为此优化器调用了 ``scaler.unscale_(optimizer)``，因此 ``scaler.step(optimizer)`` 知道在（内部）调用 ``optimizer.step()`` 之前无需冗余地反缩放梯度。

.. currentmodule:: torch.amp.GradScaler

.. warning::
    :meth:`unscale_<unscale_>` 每个优化器每次 :meth:`step<step>` 调用只应调用一次，
    并且仅在该优化器分配参数的所有梯度都已累积之后。
    在每个 :meth:`step<step>` 之间对给定优化器调用两次 :meth:`unscale_<unscale_>` 会触发 RuntimeError。

处理缩放后的梯度
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

梯度累积
---------------------

梯度累积在大小为 ``batch_per_iter * iters_to_accumulate``（如果分布式则为 ``* num_procs``）的有效批次上累加梯度。缩放因子应针对有效批次进行校准，这意味着 inf/NaN 检查、如果发现 inf/NaN 梯度则跳过步骤以及缩放因子更新应在有效批次粒度上进行。此外，在给定有效批次的梯度累积期间，梯度应保持缩放状态，且缩放因子应保持恒定。如果在累积完成之前对梯度进行反缩放（或缩放因子发生变化），下一次反向传播会将缩放后的梯度加到未缩放的梯度（或以不同因子缩放的梯度）上，之后将无法恢复 :meth:`step<step>` 必须应用的累积未缩放梯度。

因此，如果您希望 :meth:`unscale_<unscale_>` 梯度（例如，以便对未缩放的梯度进行裁剪），请在所有用于即将执行的 :meth:`step<step>` 的（缩放后）梯度都已累积之后，在调用 :meth:`step<step>` 之前立即调用 :meth:`unscale_<unscale_>`。同时，仅在迭代结束时，当您已为一个完整有效批次调用过 :meth:`step<step>` 后，才调用 :meth:`update<update>`：:

    scaler = GradScaler()

    for epoch in epochs:
        for i, (input, target) in enumerate(data):
            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(input)
                loss = loss_fn(output, target)
                loss = loss / iters_to_accumulate

            # 累积缩放后的梯度。
            scaler.scale(loss).backward()

            if (i + 1) % iters_to_accumulate == 0:
                # 可以在此处进行 unscale_（例如，以便对未缩放的梯度进行裁剪）

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

.. currentmodule:: torch.amp

梯度惩罚
----------------

梯度惩罚的实现通常使用 :func:`torch.autograd.grad` 创建梯度，将它们组合以生成惩罚值，并将该惩罚值添加到损失中。

以下是一个不使用梯度缩放或自动混合精度的 L2 惩罚的普通示例：:

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)

            # 创建梯度
            grad_params = torch.autograd.grad(outputs=loss,
                                              inputs=model.parameters(),
                                              create_graph=True)

            # 计算惩罚项并将其添加到损失中
            grad_norm = 0
            for grad in grad_params:
                grad_norm += grad.pow(2).sum()
            grad_norm = grad_norm.sqrt()
            loss = loss + grad_norm

            loss.backward()

            # 如果需要，可以在此处裁剪梯度

            optimizer.step()

若要实现*带有*梯度缩放的梯度惩罚，传递给 :func:`torch.autograd.grad` 的 ``outputs`` 张量应被缩放。因此，生成的梯度也将是缩放后的，在将它们组合以生成惩罚值之前，应进行反缩放。

此外，惩罚项计算属于前向传播的一部分，因此应放在 :class:`autocast` 上下文中。

以下是对同一个 L2 惩罚的实现方式：:

    scaler = GradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(input)
                loss = loss_fn(output, target)

            # 为 autograd.grad 的反向传播缩放损失，生成 scaled_grad_params
            scaled_grad_params = torch.autograd.grad(outputs=scaler.scale(loss),
                                                     inputs=model.parameters(),
                                                     create_graph=True)

            # 在计算惩罚项之前创建未缩放的 grad_params。scaled_grad_params 不属于任何优化器，
            # 因此使用普通除法而不是 scaler.unscale_：
            inv_scale = 1./scaler.get_scale()
            grad_params = [p * inv_scale for p in scaled_grad_params]

            # 计算惩罚项并将其添加到损失中
            with autocast(device_type='cuda', dtype=torch.float16):
                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                loss = loss + grad_norm

            # 像往常一样对反向调用应用缩放。
            # 累积正确缩放的叶节点梯度。
            scaler.scale(loss).backward()

            # 可以在此处进行 unscale_（例如，以便对未缩放的梯度进行裁剪）

            # step() 和 update() 照常进行。
            scaler.step(optimizer)
            scaler.update()


处理多个模型、损失和优化器
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: torch.amp.GradScaler

如果您的网络有多个损失，必须对每个损失单独调用 :meth:`scaler.scale<scale>`。
如果您的网络有多个优化器，可以对其中任何一个单独调用 :meth:`scaler.unscale_<unscale_>`，并且必须对每个优化器单独调用 :meth:`scaler.step<step>`。

但是，:meth:`scaler.update<update>` 只应调用一次，且在当前迭代中使用的所有优化器都已执行 step 之后：:

    scaler = torch.amp.GradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer0.zero_grad()
            optimizer1.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                output0 = model0(input)
                output1 = model1(input)
                loss0 = loss_fn(2 * output0 + 3 * output1, target)
                loss1 = loss_fn(3 * output0 - 5 * output1, target)

            #（此处的 retain_graph 与 amp 无关，它存在是因为在此示例中，
            # 两个 backward() 调用共享图的一些部分。）
            scaler.scale(loss0).backward(retain_graph=True)
            scaler.scale(loss1).backward()

            # 您可以选择哪些优化器接收显式的反缩放操作，
            # 如果您希望检查或修改它们所拥有的参数的梯度。
            scaler.unscale_(optimizer0)

            scaler.step(optimizer0)
            scaler.step(optimizer1)

            scaler.update()

每个优化器都会检查其梯度中是否存在 infs/NaNs，并独立决定是否跳过该步骤。这可能导致一个优化器跳过步骤而另一个不跳过。由于步骤跳过很少发生（每几百次迭代一次），这不应阻碍收敛。如果您在向多优化器模型添加梯度缩放后观察到收敛效果不佳，请报告错误。

.. currentmodule:: torch.amp

.. _amp-multigpu:

多 GPU 工作
^^^^^^^^^^^^^^^^^^^^^^^^^^

此处描述的问题仅影响 :class:`autocast`。:class:`GradScaler` 的使用方式保持不变。

.. _amp-dataparallel:

单进程中的 DataParallel
--------------------------------

即使 :class:`torch.nn.DataParallel` 会生成线程在每个设备上运行前向传播，autocast 状态也会在每个线程中传播，因此以下代码可以正常工作::

    model = MyModel()
    dp_model = nn.DataParallel(model)

    # 在主线程中设置 autocast
    with autocast(device_type='cuda', dtype=torch.float16):
        # dp_model 的内部线程将启用 autocast。
        output = dp_model(input)
        # loss_fn 也启用 autocast
        loss = loss_fn(output)

DistributedDataParallel，每个进程一个 GPU
--------------------------------------------

:class:`torch.nn.parallel.DistributedDataParallel` 的文档建议每个进程使用一个 GPU 以获得最佳性能。在这种情况下，``DistributedDataParallel`` 不会在内部生成线程，因此 :class:`autocast` 和 :class:`GradScaler` 的使用不受影响。

DistributedDataParallel，每个进程多个 GPU
--------------------------------------------------

在这种情况下，:class:`torch.nn.parallel.DistributedDataParallel` 可能会生成一个辅助线程在每个设备上运行前向传播，类似于 :class:`torch.nn.DataParallel`。:ref:`解决方法相同<amp-dataparallel>`：将 autocast 作为模型 ``forward`` 方法的一部分应用，以确保在辅助线程中启用它。

.. _amp-custom-examples:

Autocast 和自定义 Autograd 函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果您的网络使用 :ref:`自定义 autograd 函数<extending-autograd>`（:class:`torch.autograd.Function` 的子类），并且任何函数满足以下条件，则需要修改以确保与 autocast 兼容：

* 接受多个浮点张量输入，
* 包装了任何可 autocast 的操作（参见 :ref:`Autocast 操作参考<autocast-op-reference>`），或
* 需要特定的 ``dtype``（例如，如果它包装了仅针对 ``dtype`` 编译的 `CUDA 扩展 <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_）。

在所有情况下，如果您导入的函数无法修改其定义，一个安全的备用方案是在出现错误的使用点禁用 autocast 并强制在 ``float32``（或 ``dtype``）中执行::

    with autocast(device_type='cuda', dtype=torch.float16):
        ...
        with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            output = imported_function(input1.float(), input2.float())

如果您是函数的作者（或可以修改其定义），更好的解决方案是使用 :func:`torch.amp.custom_fwd` 和 :func:`torch.amp.custom_bwd` 装饰器，如下文相关案例所示。

具有多个输入或可 autocast 操作的函数
--------------------------------------------------

分别对 ``forward`` 和 ``backward`` 应用 :func:`custom_fwd<custom_fwd>` 和 :func:`custom_bwd<custom_bwd>`（不带参数）。这确保 ``forward`` 在当前 autocast 状态下执行，而 ``backward`` 在与 ``forward`` 相同的 autocast 状态下执行（这可以防止类型不匹配错误）::

    class MyMM(torch.autograd.Function):
        @staticmethod
        @custom_fwd
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            return a.mm(b)
        @staticmethod
        @custom_bwd
        def backward(ctx, grad):
            a, b = ctx.saved_tensors
            return grad.mm(b.t()), a.t().mm(grad)

现在 ``MyMM`` 可以在任何地方调用，无需禁用 autocast 或手动转换输入::

    mymm = MyMM.apply

    with autocast(device_type='cuda', dtype=torch.float16):
        output = mymm(input1, input2)

需要特定 ``dtype`` 的函数
------------------------------------------

考虑一个需要 ``torch.float32`` 输入的自定义函数。对 ``forward`` 应用 :func:`custom_fwd(device_type='cuda', cast_inputs=torch.float32)<custom_fwd>`，对 ``backward`` 应用 :func:`custom_bwd(device_type='cuda')<custom_bwd>`。如果 ``forward`` 在启用 autocast 的区域中运行，装饰器会将浮点张量输入转换为由参数 `device_type <../amp.html>`_（本例中为 `CUDA`）指定的设备上的 ``float32``，并在 ``forward`` 和 ``backward`` 期间局部禁用 autocast::

    class MyFloat32Func(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type='cuda', cast_inputs=torch.float32)
        def forward(ctx, input):
            ctx.save_for_backward(input)
            ...
            return fwd_output
        @staticmethod
        @custom_bwd(device_type='cuda')
        def backward(ctx, grad):
            ...

现在 ``MyFloat32Func`` 可以在任何地方调用，无需手动禁用 autocast 或转换输入::

    func = MyFloat32Func.apply

    with autocast(device_type='cuda', dtype=torch.float16):
        # func 将在 float32 中运行，无论周围的 autocast 状态如何
        output = func(input)