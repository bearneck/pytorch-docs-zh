# torch.Size

{class}`torch.Size` 是调用 {func}`torch.Tensor.size` 的返回类型。它描述了原始张量所有维度的大小。
作为 {class}`tuple` 的子类，它支持常见的序列操作，如索引和获取长度。

示例：

```{code-block} python
    >>> x = torch.ones(10, 20, 30)
    >>> s = x.size()
    >>> s
    torch.Size([10, 20, 30])
    >>> s[1]
    20
    >>> len(s)
    3
```

```{eval-rst}
.. autoclass:: torch.Size
   :members:
   :undoc-members:
   :inherited-members:
```