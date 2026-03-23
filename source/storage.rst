torch.Storage
=============

在 PyTorch 中，常规张量是一个多维数组，由以下组件定义：

- Storage：张量的实际数据，以连续的字节一维数组形式存储。
- ``dtype``：张量中元素的数据类型，例如 torch.float32 或 torch.int64。
- ``shape``：一个元组，表示张量在每个维度上的大小。
- Stride：在每个维度上从一个元素移动到下一个元素所需的步长。
- Offset：存储中张量数据开始的起点。对于新创建的张量，这通常为 0。

这些组件共同定义了张量的结构和数据，其中存储保存实际数据，其余部分作为元数据。

无类型存储 API
-------------------

:class:`torch.UntypedStorage` 是一个连续的、一维的元素数组。其长度等于张量的字节数。存储作为张量的底层数据容器。
通常，使用常规构造函数（如 :func:`~torch.zeros`、:func:`~torch.zeros_like` 或 :func:`~torch.Tensor.new_zeros`）在 PyTorch 中创建的张量，其张量存储与张量本身之间存在一一对应关系。

然而，一个存储允许多个张量共享。
例如，张量的任何视图（通过 :meth:`~torch.Tensor.view` 或某些类型的索引（如整数和切片，但并非所有类型）获得）将指向与原始张量相同的底层存储。
当序列化和反序列化共享同一存储的张量时，这种关系会被保留，张量继续指向同一存储。有趣的是，反序列化指向单个存储的多个张量可能比反序列化多个独立张量更快。

张量存储可以通过 :meth:`~torch.Tensor.untyped_storage` 方法访问。这将返回一个 :class:`torch.UntypedStorage` 类型的对象。
幸运的是，存储具有通过 :meth:`torch.UntypedStorage.data_ptr` 方法访问的唯一标识符。
在常规设置中，具有相同数据存储的两个张量将具有相同的存储 ``data_ptr``。
然而，张量本身可以指向两个独立的存储，一个用于其数据属性，另一个用于其梯度属性。每个都需要自己的 ``data_ptr()``。通常，不能保证 :meth:`torch.Tensor.data_ptr` 和 :meth:`torch.UntypedStorage.data_ptr` 匹配，也不应假设这一点成立。

无类型存储在一定程度上独立于构建在其上的张量。实际上，这意味着具有不同 dtype 或形状的张量可以指向同一存储。
这也意味着张量存储可以被更改，如下例所示：

    >>> t = torch.ones(3)
    >>> s0 = t.untyped_storage()
    >>> s0
     0
     0
     128
     63
     0
     0
     128
     63
     0
     0
     128
     63
    [torch.storage.UntypedStorage(device=cpu) of size 12]
    >>> s1 = s0.clone()
    >>> s1.fill_(0)
     0
     0
     0
     0
     0
     0
     0
     0
     0
     0
     0
     0
    [torch.storage.UntypedStorage(device=cpu) of size 12]
    >>> # 用归零的存储填充张量
    >>> t.set_(s1, storage_offset=t.storage_offset(), stride=t.stride(), size=t.size())
    tensor([0., 0., 0.])

.. warning::
  请注意，如本例所示直接修改张量的存储不是推荐的做法。
  这种低级操作仅出于教育目的进行说明，以展示张量与其底层存储之间的关系。通常，使用标准的 ``torch.Tensor`` 方法（如 :meth:`~torch.Tensor.clone` 和 :meth:`~torch.Tensor.fill_`）来实现相同的结果更高效且更安全。

除了 ``data_ptr``，无类型存储还有其他属性，例如 :attr:`~torch.UntypedStorage.filename`（如果存储指向磁盘上的文件）、:attr:`~torch.UntypedStorage.device` 或用于设备检查的 :attr:`~torch.UntypedStorage.is_cuda`。存储也可以通过方法如 :attr:`~torch.UntypedStorage.copy_`、:attr:`~torch.UntypedStorage.fill_` 或 :attr:`~torch.UntypedStorage.pin_memory` 进行原地或非原地操作。更多信息，请查看下面的 API 参考。请记住，修改存储是低级 API，存在风险！
这些 API 中的大多数也存在于张量级别：如果存在，应优先使用张量级别的 API 而非其存储对应项。

特殊情况
-------------

我们提到，具有非 None ``grad`` 属性的张量实际上包含两部分数据。
在这种情况下，:meth:`~torch.Tensor.untyped_storage` 将返回 :attr:`~torch.Tensor.data` 属性的存储，而梯度的存储可以通过 ``tensor.grad.untyped_storage()`` 获得。

    >>> t = torch.zeros(3, requires_grad=True)
    >>> t.sum().backward()
    >>> assert list(t.untyped_storage()) == [0] * 12  # 张量的存储仅为 0
    >>> assert list(t.grad.untyped_storage()) != [0] * 12  # 梯度的存储不是

还有一些特殊情况，张量没有典型的存储，或者根本没有存储：
  - ``"meta"`` 设备上的张量：``"meta"`` 设备上的张量用于形状推断，不保存实际数据。
  - 伪张量：PyTorch 编译器使用的另一个内部工具是 `FakeTensor <https://pytorch.org/docs/stable/torch.compiler_fake_tensor.html>`_，它基于类似的概念。

张量子类或类张量对象也可能表现出不寻常的行为。通常，我们不期望许多用例需要在存储级别进行操作！

.. autoclass:: torch.UntypedStorage
   :members:
   :undoc-members:
   :inherited-members:

遗留类型化存储
--------------------

.. warning::
  出于历史原因，PyTorch 之前使用类型化的存储类，这些类现已弃用，应避免使用。以下内容详细介绍了该 API，以防您遇到它，但强烈不建议使用。
  除 :class:`torch.UntypedStorage` 之外的所有存储类都将在未来被移除，并且所有情况下都将使用 :class:`torch.UntypedStorage`。

:class:`torch.Storage` 是与默认数据类型 (:func:`torch.get_default_dtype()`) 对应的存储类的别名。例如，如果默认数据类型是 :attr:`torch.float`，则 :class:`torch.Storage` 解析为 :class:`torch.FloatStorage`。

:class:`torch.<type>Storage` 和 :class:`torch.cuda.<type>Storage` 类，如 :class:`torch.FloatStorage`、:class:`torch.IntStorage` 等，实际上从未被实例化。调用它们的构造函数会创建一个具有适当 :class:`torch.dtype` 和 :class:`torch.device` 的 :class:`torch.TypedStorage`。:class:`torch.<type>Storage` 类拥有与 :class:`torch.TypedStorage` 相同的所有类方法。

:class:`torch.TypedStorage` 是一个特定 :class:`torch.dtype` 元素的连续一维数组。它可以被赋予任何 :class:`torch.dtype`，内部数据将被相应地解释。:class:`torch.TypedStorage` 包含一个 :class:`torch.UntypedStorage`，后者将数据保存为无类型的字节数组。

每个跨步的 :class:`torch.Tensor` 都包含一个 :class:`torch.TypedStorage`，它存储了 :class:`torch.Tensor` 视图中的所有数据。

.. autoclass:: torch.TypedStorage
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: torch.DoubleStorage
   :members:
   :undoc-members:

.. autoclass:: torch.FloatStorage
   :members:
   :undoc-members:

.. autoclass:: torch.HalfStorage
   :members:
   :undoc-members:

.. autoclass:: torch.LongStorage
   :members:
   :undoc-members:

.. autoclass:: torch.IntStorage
   :members:
   :undoc-members:

.. autoclass:: torch.ShortStorage
   :members:
   :undoc-members:

.. autoclass:: torch.CharStorage
   :members:
   :undoc-members:

.. autoclass:: torch.ByteStorage
   :members:
   :undoc-members:

.. autoclass:: torch.BoolStorage
   :members:
   :undoc-members:

.. autoclass:: torch.BFloat16Storage
   :members:
   :undoc-members:

.. autoclass:: torch.ComplexDoubleStorage
   :members:
   :undoc-members:

.. autoclass:: torch.ComplexFloatStorage
   :members:
   :undoc-members:

.. autoclass:: torch.QUInt8Storage
   :members:
   :undoc-members:

.. autoclass:: torch.QInt8Storage
   :members:
   :undoc-members:

.. autoclass:: torch.QInt32Storage
   :members:
   :undoc-members:

.. autoclass:: torch.QUInt4x2Storage
   :members:
   :undoc-members:

.. autoclass:: torch.QUInt2x4Storage
   :members:
   :undoc-members: