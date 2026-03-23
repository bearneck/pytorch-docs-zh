.. _broadcasting-semantics:

广播语义
======================

许多 PyTorch 操作支持 NumPy 的广播语义。
详情请参阅 https://numpy.org/doc/stable/user/basics.broadcasting.html。

简而言之，如果一个 PyTorch 操作支持广播，那么其张量参数可以自动扩展为相同大小（无需复制数据）。

通用语义
-----------------
两个张量在满足以下规则时是“可广播的”：

- 从尾部维度开始迭代维度大小时，维度大小必须相等，其中一个为 1，或者其中一个维度不存在。

例如：:

    >>> x=torch.empty(5,7,3)
    >>> y=torch.empty(5,7,3)
    # 相同形状总是可广播的（即上述规则始终成立）

    >>> x=torch.empty((0,))
    >>> y=torch.empty(2,2)
    # x 和 y 不可广播，因为 x 的 0 大小维度与 y 的 2 大小维度不匹配。

    # 可以对齐尾部维度
    >>> x=torch.empty(5,3,4,1)
    >>> y=torch.empty(  3,1,1)
    # x 和 y 是可广播的。
    # 第 1 个尾部维度：两者大小均为 1
    # 第 2 个尾部维度：y 的大小为 1
    # 第 3 个尾部维度：x 大小 == y 大小
    # 第 4 个尾部维度：y 维度不存在

    # 但是：
    >>> x=torch.empty(5,2,4,1)
    >>> y=torch.empty(  3,1,1)
    # x 和 y 不可广播，因为在第 3 个尾部维度中 2 != 3

如果两个张量 :attr:`x`、:attr:`y` 是“可广播的”，则结果张量大小按如下方式计算：

- 如果 :attr:`x` 和 :attr:`y` 的维度数量不相等，则在维度较少的张量的维度前添加 1，使其长度相等。
- 然后，对于每个维度大小，结果维度大小是 :attr:`x` 和 :attr:`y` 在该维度上大小的最大值。

例如：:

    # 可以对齐尾部维度以便于阅读
    >>> x=torch.empty(5,1,4,1)
    >>> y=torch.empty(  3,1,1)
    >>> (x+y).size()
    torch.Size([5, 3, 4, 1])

    # 但并非必需：
    >>> x=torch.empty(1)
    >>> y=torch.empty(3,1,7)
    >>> (x+y).size()
    torch.Size([3, 1, 7])

    >>> x=torch.empty(5,2,4,1)
    >>> y=torch.empty(3,1,1)
    >>> (x+y).size()
    RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1

原地操作语义
------------------
一个复杂之处在于，原地操作不允许原地张量因广播而改变形状。

例如：:

    >>> x=torch.empty(5,3,4,1)
    >>> y=torch.empty(3,1,1)
    >>> (x.add_(y)).size()
    torch.Size([5, 3, 4, 1])

    # 但是：
    >>> x=torch.empty(1,3,1)
    >>> y=torch.empty(3,1,7)
    >>> (x.add_(y)).size()
    RuntimeError: The expanded size of the tensor (1) must match the existing size (7) at non-singleton dimension 2.

向后兼容性
-----------------------
PyTorch 的早期版本允许某些逐点函数在不同形状的张量上执行，只要每个张量中的元素数量相等即可。然后，通过将每个张量视为 1 维来执行逐点操作。PyTorch 现在支持广播，并且“1 维”逐点行为被视为已弃用，在张量不可广播但具有相同元素数量的情况下将生成 Python 警告。

请注意，广播的引入可能导致向后不兼容的更改，这种情况发生在两个张量形状不同但可广播且具有相同元素数量时。
例如：:

    >>> torch.add(torch.ones(4,1), torch.randn(4))

以前会产生大小为 torch.Size([4,1]) 的张量，但现在会产生大小为 torch.Size([4,4]) 的张量。
为了帮助识别代码中可能存在的由广播引起的向后不兼容情况，您可以将 `torch.utils.backcompat.broadcast_warning.enabled` 设置为 `True`，这将在这种情况下生成 Python 警告。

例如：:

    >>> torch.utils.backcompat.broadcast_warning.enabled=True
    >>> torch.add(torch.ones(4,1), torch.ones(4))
    __main__:1: UserWarning: self and other do not have the same shape, but are broadcastable, and have the same number of elements.
    Changing behavior in a backwards incompatible manner to broadcasting rather than viewing as 1-dimensional.