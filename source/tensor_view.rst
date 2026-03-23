.. currentmodule:: torch

.. _tensor-view-doc:

张量视图
=============

PyTorch 允许一个张量成为现有张量的一个 ``View``（视图）。视图张量与其基础张量共享相同的基础数据。支持 ``View`` 可以避免显式的数据复制，从而使我们能够进行快速且内存高效的形状重塑、切片和逐元素操作。

例如，要获取现有张量 ``t`` 的一个视图，可以调用 ``t.view(...)``。

::

    >>> t = torch.rand(4, 4)
    >>> b = t.view(2, 8)
    >>> t.storage().data_ptr() == b.storage().data_ptr()  # `t` 和 `b` 共享相同的基础数据。
    True
    # 修改视图张量也会改变基础张量。
    >>> b[0][0] = 3.14
    >>> t[0][0]
    tensor(3.14)

由于视图与其基础张量共享基础数据，如果你在视图中编辑数据，它也会反映在基础张量中。

通常，PyTorch 操作会返回一个新的张量作为输出，例如 :meth:`~torch.Tensor.add`。但对于视图操作，输出是输入张量的视图，以避免不必要的数据复制。创建视图时不会发生数据移动，视图张量只是改变了其解释相同数据的方式。对连续张量取视图可能会产生一个非连续张量。用户应额外注意，因为连续性可能对性能有隐含影响。:meth:`~torch.Tensor.transpose` 就是一个常见的例子。

::

    >>> base = torch.tensor([[0, 1],[2, 3]])
    >>> base.is_contiguous()
    True
    >>> t = base.transpose(0, 1)  # `t` 是 `base` 的一个视图。此处没有发生数据移动。
    # 视图张量可能是非连续的。
    >>> t.is_contiguous()
    False
    # 要获得一个连续张量，当 `t` 不连续时，调用 `.contiguous()` 来强制复制数据。
    >>> c = t.contiguous()

作为参考，以下是 PyTorch 中视图操作的完整列表：

- 基本切片和索引操作，例如 ``tensor[0, 2:, 1:7:2]`` 返回基础 ``tensor`` 的视图，参见下面的注释。
- :meth:`~torch.Tensor.adjoint`
- :meth:`~torch.Tensor.as_strided`
- :meth:`~torch.Tensor.detach`
- :meth:`~torch.Tensor.diagonal`
- :meth:`~torch.Tensor.expand`
- :meth:`~torch.Tensor.expand_as`
- :meth:`~torch.Tensor.movedim`
- :meth:`~torch.Tensor.narrow`
- :meth:`~torch.Tensor.permute`
- :meth:`~torch.Tensor.select`
- :meth:`~torch.Tensor.squeeze`
- :meth:`~torch.Tensor.transpose`
- :meth:`~torch.Tensor.t`
- :attr:`~torch.Tensor.T`
- :attr:`~torch.Tensor.H`
- :attr:`~torch.Tensor.mT`
- :attr:`~torch.Tensor.mH`
- :attr:`~torch.Tensor.real`
- :attr:`~torch.Tensor.imag`
- :meth:`~torch.Tensor.view_as_real`
- :meth:`~torch.Tensor.unflatten`
- :meth:`~torch.Tensor.unfold`
- :meth:`~torch.Tensor.unsqueeze`
- :meth:`~torch.Tensor.view`
- :meth:`~torch.Tensor.view_as`
- :meth:`~torch.Tensor.unbind`
- :meth:`~torch.Tensor.split`
- :meth:`~torch.Tensor.hsplit`
- :meth:`~torch.Tensor.vsplit`
- :meth:`~torch.Tensor.tensor_split`
- :meth:`~torch.Tensor.split_with_sizes`
- :meth:`~torch.Tensor.swapaxes`
- :meth:`~torch.Tensor.swapdims`
- :meth:`~torch.Tensor.chunk`
- :meth:`~torch.Tensor.indices` (仅稀疏张量)
- :meth:`~torch.Tensor.values`  (仅稀疏张量)

.. note::
   通过索引访问张量内容时，PyTorch 遵循 Numpy 的行为：基本索引返回视图，而高级索引返回一个副本。无论是通过基本索引还是高级索引进行的赋值都是原地操作。更多示例请参阅 `Numpy 索引文档 <https://numpy.org/doc/stable/user/basics.indexing.html>`_。

还有一些具有特殊行为的操作值得一提：

- :meth:`~torch.Tensor.reshape`、:meth:`~torch.Tensor.reshape_as` 和 :meth:`~torch.Tensor.flatten` 可能返回视图或新张量，用户代码不应依赖它是否是视图。
- :meth:`~torch.Tensor.contiguous` 如果输入张量已经是连续的，则返回**自身**，否则通过复制数据返回一个新的连续张量。

关于 PyTorch 内部实现的更详细说明，请参考 `ezyang 关于 PyTorch 内部机制的博客文章 <http://blog.ezyang.com/2019/05/pytorch-internals/>`_。