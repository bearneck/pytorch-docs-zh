.. currentmodule:: torch

.. _tensor-doc:

torch.Tensor
===================================

:class:`torch.Tensor` 是一个包含单一数据类型元素的多维矩阵。有关 dtype 支持的更多详细信息，请参阅 :ref:`dtype-doc`。

初始化和基本操作
---------------------------------

可以使用 :func:`torch.tensor` 构造函数从 Python :class:`list` 或序列构造张量：

::

    >>> torch.tensor([[1., -1.], [1., -1.]])
    tensor([[ 1.0000, -1.0000],
            [ 1.0000, -1.0000]])
    >>> torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    tensor([[ 1,  2,  3],
            [ 4,  5,  6]])

.. warning::

    :func:`torch.tensor` 总是复制 :attr:`data`。如果你有一个 Tensor :attr:`data` 并且只想更改其 ``requires_grad`` 标志，请使用 :meth:`~torch.Tensor.requires_grad_` 或 :meth:`~torch.Tensor.detach` 来避免复制。
    如果你有一个 numpy 数组并且想避免复制，请使用 :func:`torch.as_tensor`。

可以通过向构造函数或张量创建操作传递 :class:`torch.dtype` 和/或 :class:`torch.device` 来构造特定数据类型的张量：

::

    >>> torch.zeros([2, 4], dtype=torch.int32)
    tensor([[ 0,  0,  0,  0],
            [ 0,  0,  0,  0]], dtype=torch.int32)
    >>> cuda0 = torch.device('cuda:0')
    >>> torch.ones([2, 4], dtype=torch.float64, device=cuda0)
    tensor([[ 1.0000,  1.0000,  1.0000,  1.0000],
            [ 1.0000,  1.0000,  1.0000,  1.0000]], dtype=torch.float64, device='cuda:0')

有关构建张量的更多信息，请参阅 :ref:`tensor-creation-ops`

可以使用 Python 的索引和切片表示法访问和修改张量的内容：

::

    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    >>> print(x[1][2])
    tensor(6)
    >>> x[0][1] = 8
    >>> print(x)
    tensor([[ 1,  8,  3],
            [ 4,  5,  6]])

使用 :meth:`torch.Tensor.item` 从包含单个值的张量中获取一个 Python 数字：

::

    >>> x = torch.tensor([[1]])
    >>> x
    tensor([[ 1]])
    >>> x.item()
    1
    >>> x = torch.tensor(2.5)
    >>> x
    tensor(2.5000)
    >>> x.item()
    2.5

有关索引的更多信息，请参阅 :ref:`indexing-slicing-joining`

可以创建具有 :attr:`requires_grad=True` 的张量，以便 :mod:`torch.autograd` 记录对其的操作以进行自动微分。

::

    >>> x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
    >>> out = x.pow(2).sum()
    >>> out.backward()
    >>> x.grad
    tensor([[ 2.0000, -2.0000],
            [ 2.0000,  2.0000]])

每个张量都有一个关联的 :class:`torch.Storage`，用于保存其数据。张量类还提供了存储的多维、`跨步 <https://en.wikipedia.org/wiki/Stride_of_an_array>`_ 视图，并定义了其上的数值操作。

.. note::
   有关张量视图的更多信息，请参阅 :ref:`tensor-view-doc`。

.. note::
   有关 :class:`torch.Tensor` 的 :class:`torch.dtype`、:class:`torch.device` 和 :class:`torch.layout` 属性的更多信息，请参阅 :ref:`tensor-attributes-doc`。

.. note::
   修改张量的方法会带有下划线后缀标记。例如，:func:`torch.FloatTensor.abs_` 就地计算绝对值并返回修改后的张量，而 :func:`torch.FloatTensor.abs` 在新的张量中计算结果。

.. note::
   要更改现有张量的 :class:`torch.device` 和/或 :class:`torch.dtype`，请考虑在张量上使用 :meth:`~torch.Tensor.to` 方法。

.. warning::
   :class:`torch.Tensor` 的当前实现引入了内存开销，因此在具有许多微小张量的应用程序中可能导致意外的高内存使用。如果这是你的情况，请考虑使用一个大型结构。

张量类参考
----------------------

.. class:: Tensor()

   根据你的使用场景，有几种主要的创建张量的方法。

   - 要使用现有数据创建张量，请使用 :func:`torch.tensor`。
   - 要创建具有特定大小的张量，请使用 ``torch.*`` 张量创建操作（参见 :ref:`tensor-creation-ops`）。
   - 要创建与另一个张量具有相同大小（和相似类型）的张量，请使用 ``torch.*_like`` 张量创建操作（参见 :ref:`tensor-creation-ops`）。
   - 要创建与另一个张量类型相似但大小不同的张量，请使用 ``tensor.new_*`` 创建操作。
   - 有一个遗留的构造函数 ``torch.Tensor``，不鼓励使用。请改用 :func:`torch.tensor`。

.. method:: Tensor.__init__(self, data)

   此构造函数已弃用，我们建议改用 :func:`torch.tensor`。此构造函数的行为取决于 ``data`` 的类型。

   * 如果 ``data`` 是一个 Tensor，则返回原始 Tensor 的别名。与 :func:`torch.tensor` 不同，这会跟踪自动微分，并将梯度传播到原始 Tensor。对于这种 ``data`` 类型，不支持 ``device`` 关键字参数。

   * 如果 ``data`` 是一个序列或嵌套序列，则创建一个默认 dtype（通常是 ``torch.float32``）的张量，其数据是序列中的值，必要时执行强制转换。值得注意的是，这与 :func:`torch.tensor` 不同，因为此构造函数总是构造一个浮点张量，即使输入全是整数。

   * 如果 ``data`` 是一个 :class:`torch.Size`，则返回一个该大小的空张量。

   此构造函数不支持显式指定返回张量的 ``dtype`` 或 ``device``。我们建议使用提供此功能的 :func:`torch.tensor`。

   Args:
       data (array_like): 用于构造张量的数据。

   Keyword args:
       device (:class:`torch.device`, optional): 返回张量的期望设备。
           默认值：如果为 None，则与此张量相同的 :class:`torch.device`。

.. autoattribute:: Tensor.T
.. autoattribute:: Tensor.H
.. autoattribute:: Tensor.mT
.. autoattribute:: Tensor.mH

.. autosummary::
    :toctree: generated
    :nosignatures:

Tensor.new_tensor
Tensor.new_full
Tensor.new_empty
Tensor.new_ones
Tensor.new_zeros

Tensor.is_cuda
Tensor.is_quantized
Tensor.is_meta
Tensor.device
Tensor.grad
Tensor.ndim
Tensor.real
Tensor.imag
Tensor.nbytes
Tensor.itemsize

Tensor.abs
Tensor.abs_
Tensor.absolute
Tensor.absolute_
Tensor.acos
Tensor.acos_
Tensor.arccos
Tensor.arccos_
Tensor.add
Tensor.add_
Tensor.addbmm
Tensor.addbmm_
Tensor.addcdiv
Tensor.addcdiv_
Tensor.addcmul
Tensor.addcmul_
Tensor.addmm
Tensor.addmm_
Tensor.sspaddmm
Tensor.addmv
Tensor.addmv_
Tensor.addr
Tensor.addr_
Tensor.adjoint
Tensor.allclose
Tensor.amax
Tensor.amin
Tensor.aminmax
Tensor.angle
Tensor.apply_
Tensor.argmax
Tensor.argmin
Tensor.argsort
Tensor.argwhere
Tensor.asin
Tensor.asin_
Tensor.arcsin
Tensor.arcsin_
Tensor.as_strided
Tensor.atan
Tensor.atan_
Tensor.arctan
Tensor.arctan_
Tensor.atan2
Tensor.atan2_
Tensor.arctan2
Tensor.arctan2_
Tensor.all
Tensor.any
Tensor.backward
Tensor.baddbmm
Tensor.baddbmm_
Tensor.bernoulli
Tensor.bernoulli_
Tensor.bfloat16
Tensor.bincount
Tensor.bitwise_not
Tensor.bitwise_not_
Tensor.bitwise_and
Tensor.bitwise_and_
Tensor.bitwise_or
Tensor.bitwise_or_
Tensor.bitwise_xor
Tensor.bitwise_xor_
Tensor.bitwise_left_shift
Tensor.bitwise_left_shift_
Tensor.bitwise_right_shift
Tensor.bitwise_right_shift_
Tensor.bmm
Tensor.bool
Tensor.byte
Tensor.broadcast_to
Tensor.cauchy_
Tensor.ceil
Tensor.ceil_
Tensor.char
Tensor.cholesky
Tensor.cholesky_inverse
Tensor.cholesky_solve
Tensor.chunk
Tensor.clamp
Tensor.clamp_
Tensor.clip
Tensor.clip_
Tensor.clone
Tensor.contiguous
Tensor.copy_
Tensor.conj
Tensor.conj_physical
Tensor.conj_physical_
Tensor.resolve_conj
Tensor.resolve_neg
Tensor.copysign
Tensor.copysign_
Tensor.cos
Tensor.cos_
Tensor.cosh
Tensor.cosh_
Tensor.corrcoef
Tensor.count_nonzero
Tensor.cov
Tensor.acosh
Tensor.acosh_
Tensor.arccosh
Tensor.arccosh_
Tensor.cpu
Tensor.cross
Tensor.cuda
Tensor.logcumsumexp
Tensor.cummax
Tensor.cummin
Tensor.cumprod
Tensor.cumprod_
Tensor.cumsum
Tensor.cumsum_
Tensor.chalf
Tensor.cfloat
Tensor.cdouble
Tensor.data_ptr
Tensor.deg2rad
Tensor.dequantize
Tensor.det
Tensor.dense_dim
Tensor.detach
Tensor.detach_
Tensor.diag
Tensor.diag_embed
Tensor.diagflat
Tensor.diagonal
Tensor.diagonal_scatter
Tensor.fill_diagonal_
Tensor.fmax
Tensor.fmin
Tensor.diff
Tensor.digamma
Tensor.digamma_
Tensor.dim
Tensor.dim_order
Tensor.dist
Tensor.div
Tensor.div_
Tensor.divide
Tensor.divide_
Tensor.dot
Tensor.double
Tensor.dsplit
Tensor.element_size
Tensor.eq
Tensor.eq_
Tensor.equal
Tensor.erf
Tensor.erf_
Tensor.erfc
Tensor.erfc_
Tensor.erfinv
Tensor.erfinv_
Tensor.exp
Tensor.exp_
Tensor.expm1
Tensor.expm1_
Tensor.expand
Tensor.expand_as
Tensor.exponential_
Tensor.fix
Tensor.fix_
Tensor.fill_
Tensor.flatten
Tensor.flip
Tensor.fliplr
Tensor.flipud
Tensor.float
Tensor.float_power
Tensor.float_power_
Tensor.floor
Tensor.floor_
Tensor.floor_divide
Tensor.floor_divide_
Tensor.fmod
Tensor.fmod_
Tensor.frac
Tensor.frac_
Tensor.frexp
Tensor.gather
Tensor.gcd
Tensor.gcd_
Tensor.ge
Tensor.ge_
Tensor.greater_equal
Tensor.greater_equal_
Tensor.geometric_
Tensor.geqrf
Tensor.ger
Tensor.get_device
Tensor.gt
Tensor.gt_
Tensor.greater
Tensor.greater_
Tensor.half
Tensor.hardshrink
Tensor.heaviside
Tensor.histc
Tensor.histogram
Tensor.hsplit
Tensor.hypot
Tensor.hypot_
Tensor.i0
Tensor.i0_
Tensor.igamma
Tensor.igamma_
Tensor.igammac
Tensor.igammac_
Tensor.index_add_
Tensor.index_add
Tensor.index_copy_
Tensor.index_copy
Tensor.index_fill_
Tensor.index_fill
Tensor.index_put_
Tensor.index_put
Tensor.index_reduce_
Tensor.index_reduce
Tensor.index_select
Tensor.indices
Tensor.inner
Tensor.int
Tensor.int_repr
Tensor.inverse
Tensor.isclose
Tensor.isfinite
Tensor.isinf
Tensor.isposinf
Tensor.isneginf
Tensor.isnan
Tensor.is_contiguous
Tensor.is_complex
Tensor.is_conj
Tensor.is_floating_point
Tensor.is_inference
Tensor.is_leaf
Tensor.is_pinned
Tensor.is_set_to
Tensor.is_shared
Tensor.is_signed
Tensor.is_sparse
Tensor.istft
Tensor.isreal
Tensor.item
Tensor.kthvalue
Tensor.lcm
Tensor.lcm_
Tensor.ldexp
Tensor.ldexp_
Tensor.le
Tensor.le_
Tensor.less_equal
Tensor.less_equal_
Tensor.lerp
Tensor.lerp_
Tensor.lgamma
Tensor.lgamma_
Tensor.log
Tensor.log_
Tensor.logdet
Tensor.log10
Tensor.log10_
Tensor.log1p
Tensor.log1p_
Tensor.log2
Tensor.log2_
Tensor.log_normal_
Tensor.logaddexp
Tensor.logaddexp2
Tensor.logsumexp
Tensor.logical_and
Tensor.logical_and_
Tensor.logical_not
Tensor.logical_not_
Tensor.logical_or
Tensor.logical_or_
Tensor.logical_xor
Tensor.logical_xor_
Tensor.logit
Tensor.logit_
Tensor.long
Tensor.lt
Tensor.lt_
Tensor.less
Tensor.less_
Tensor.lu
Tensor.lu_solve
Tensor.as_subclass
Tensor.map_
Tensor.masked_scatter_
Tensor.masked_scatter
Tensor.masked_fill_
Tensor.masked_fill
Tensor.masked_select
Tensor.matmul
Tensor.matrix_power
Tensor.matrix_exp
Tensor.max
Tensor.maximum
Tensor.mean
Tensor.module_load
Tensor.nanmean
Tensor.median
Tensor.nanmedian
Tensor.min
Tensor.minimum
Tensor.mm
Tensor.smm
Tensor.mode
Tensor.movedim
Tensor.moveaxis
Tensor.msort
Tensor.mul
Tensor.mul_
Tensor.multiply
Tensor.multiply_
Tensor.multinomial
Tensor.mv
Tensor.mvlgamma
Tensor.mvlgamma_
Tensor.nansum
Tensor.narrow
Tensor.narrow_copy
Tensor.ndimension
Tensor.nan_to_num
Tensor.nan_to_num_
Tensor.ne
Tensor.ne_
Tensor.not_equal
Tensor.not_equal_
Tensor.neg
Tensor.neg_
Tensor.negative
Tensor.negative_
Tensor.nelement
Tensor.nextafter
Tensor.nextafter_
Tensor.nonzero
Tensor.norm
Tensor.normal_
Tensor.numel
Tensor.numpy
Tensor.orgqr
Tensor.ormqr
Tensor.outer
Tensor.permute
Tensor.pin_memory
Tensor.pinverse
Tensor.polygamma
Tensor.polygamma_
Tensor.positive
Tensor.pow
Tensor.pow_
Tensor.prod
Tensor.put_
Tensor.qr
Tensor.qscheme
Tensor.quantile
Tensor.nanquantile
Tensor.q_scale
Tensor.q_zero_point
Tensor.q_per_channel_scales
Tensor.q_per_channel_zero_points
Tensor.q_per_channel_axis
Tensor.rad2deg
Tensor.random_
Tensor.ravel
Tensor.reciprocal
Tensor.reciprocal_
Tensor.record_stream
Tensor.register_hook
Tensor.register_post_accumulate_grad_hook
Tensor.remainder
Tensor.remainder_
Tensor.renorm
Tensor.renorm_
Tensor.repeat
Tensor.repeat_interleave
Tensor.requires_grad
Tensor.requires_grad_
Tensor.reshape
Tensor.reshape_as
Tensor.resize_
Tensor.resize_as_
Tensor.retain_grad
Tensor.retains_grad
Tensor.roll
Tensor.rot90
Tensor.round
Tensor.round_
Tensor.rsqrt
Tensor.rsqrt_
Tensor.scatter
Tensor.scatter_
Tensor.scatter_add_
Tensor.scatter_add
Tensor.scatter_reduce_
Tensor.scatter_reduce
Tensor.select
Tensor.select_scatter
Tensor.set_
Tensor.share_memory_
Tensor.short
Tensor.sigmoid
Tensor.sigmoid_
Tensor.sign
Tensor.sign_
Tensor.signbit
Tensor.sgn
Tensor.sgn_
Tensor.sin
Tensor.sin_
Tensor.sinc
Tensor.sinc_
Tensor.sinh
Tensor.sinh_
Tensor.asinh
Tensor.asinh_
Tensor.arcsinh
Tensor.arcsinh_
Tensor.shape
Tensor.size
Tensor.slogdet
Tensor.slice_scatter
Tensor.softmax
Tensor.sort
Tensor.split
Tensor.sparse_mask
Tensor.sparse_dim
Tensor.sqrt
Tensor.sqrt_
Tensor.square
Tensor.square_
Tensor.squeeze
Tensor.squeeze_
Tensor.std
Tensor.stft
Tensor.storage
Tensor.untyped_storage
Tensor.storage_offset
Tensor.storage_type
Tensor.stride
Tensor.sub
Tensor.sub_
Tensor.subtract
Tensor.subtract_
Tensor.sum
Tensor.sum_to_size
Tensor.svd
Tensor.swapaxes
Tensor.swapdims
Tensor.t
Tensor.t_
Tensor.tensor_split
Tensor.tile
Tensor.to
Tensor.to_mkldnn
Tensor.take
Tensor.take_along_dim
Tensor.tan
Tensor.tan_
Tensor.tanh
Tensor.tanh_
Tensor.atanh
Tensor.atanh_
Tensor.arctanh
Tensor.arctanh_
Tensor.tolist
Tensor.topk
Tensor.to_dense
Tensor.to_sparse
Tensor.to_sparse_csr
Tensor.to_sparse_csc
Tensor.to_sparse_bsr
Tensor.to_sparse_bsc
Tensor.trace
Tensor.transpose
Tensor.transpose_
Tensor.triangular_solve
Tensor.tril
Tensor.tril_
Tensor.triu
Tensor.triu_
Tensor.true_divide
Tensor.true_divide_
Tensor.trunc
Tensor.trunc_
Tensor.type
Tensor.type_as
Tensor.unbind
Tensor.unflatten
Tensor.unfold
Tensor.uniform_
Tensor.unique
Tensor.unique_consecutive
Tensor.unsqueeze
Tensor.unsqueeze_
Tensor.values
Tensor.var
Tensor.vdot
Tensor.view
Tensor.view_as
Tensor.vsplit
Tensor.where
Tensor.xlogy
Tensor.xlogy_
Tensor.xpu
Tensor.zero_