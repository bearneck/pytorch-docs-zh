# torch.utils.deterministic

```{eval-rst}
.. py:module:: torch.utils.deterministic
.. currentmodule:: torch.utils.deterministic

.. attribute:: fill_uninitialized_memory

    一个 :class:`bool` 类型的属性，如果设置为 True，当 :meth:`torch.use_deterministic_algorithms()` 设置为 ``True`` 时，会导致未初始化的内存被填充为一个已知值。浮点数和复数值被设置为 NaN，整数值被设置为最大值。

    默认值：``True``

    填充未初始化的内存会对性能产生不利影响。因此，如果你的程序是有效的，并且没有使用未初始化的内存作为操作的输入，那么可以关闭此设置以获得更好的性能，同时仍然保持确定性。

    当此设置开启时，以下操作将填充未初始化的内存：

        * :func:`torch.Tensor.resize_` 当被调用的张量不是量化张量时
        * :func:`torch.empty`
        * :func:`torch.empty_strided`
        * :func:`torch.empty_permuted`
        * :func:`torch.empty_like`
```