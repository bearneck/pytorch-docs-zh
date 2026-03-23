
# torch.Tensor

`torch.Tensor` 是一个包含单一数据类型元素的多维矩阵。有关 dtype 支持的更多详细信息，请参阅 `dtype-doc`。

## 初始化和基本操作

可以使用 `torch.tensor` 构造函数从 Python `list` 或序列构造张量：

    >>> torch.tensor([[1., -1.], [1., -1.]])
    tensor([[ 1.0000, -1.0000],
            [ 1.0000, -1.0000]])
    >>> torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    tensor([[ 1,  2,  3],
            [ 4,  5,  6]])


> ⚠️ **警告**
> `torch.tensor` 总是复制 `data`。如果你有一个 Tensor `data` 并且只想更改其 `requires_grad` 标志，请使用 `torch.Tensor.requires_grad_` 或 `torch.Tensor.detach` 来避免复制。 如果你有一个 numpy 数组并且想避免复制，请使用 `torch.as_tensor`。
>
> 可以通过向构造函数或张量创建操作传递 `torch.dtype` 和/或 `torch.device` 来构造特定数据类型的张量：
>
>     >>> torch.zeros([2, 4], dtype=torch.int32)
>     tensor([[ 0,  0,  0,  0],
>             [ 0,  0,  0,  0]], dtype=torch.int32)
>     >>> cuda0 = torch.device('cuda:0')
>     >>> torch.ones([2, 4], dtype=torch.float64, device=cuda0)
>     tensor([[ 1.0000,  1.0000,  1.0000,  1.0000],
>             [ 1.0000,  1.0000,  1.0000,  1.0000]], dtype=torch.float64, device='cuda:0')
>
> 有关构建张量的更多信息，请参阅 `tensor-creation-ops`
>
> 可以使用 Python 的索引和切片表示法访问和修改张量的内容：
>
>     >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
>     >>> print(x[1][2])
>     tensor(6)
>     >>> x[0][1] = 8
>     >>> print(x)
>     tensor([[ 1,  8,  3],
>             [ 4,  5,  6]])
>
> 使用 `torch.Tensor.item` 从包含单个值的张量中获取一个 Python 数字：
>
>     >>> x = torch.tensor([[1]])
>     >>> x
>     tensor([[ 1]])
>     >>> x.item()
>     1
>     >>> x = torch.tensor(2.5)
>     >>> x
>     tensor(2.5000)
>     >>> x.item()
>     2.5
>
> 有关索引的更多信息，请参阅 `indexing-slicing-joining`
>
> 可以创建具有 `requires_grad=True` 的张量，以便 `torch.autograd` 记录对其的操作以进行自动微分。
>
>     >>> x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
>     >>> out = x.pow(2).sum()
>     >>> out.backward()
>     >>> x.grad
>     tensor([[ 2.0000, -2.0000],
>             [ 2.0000,  2.0000]])
>
> 每个张量都有一个关联的 `torch.Storage`，用于保存其数据。张量类还提供了存储的多维、\`跨步 \<https://en.wikipedia.org/wiki/Stride_of_an_array\>\`\_ 视图，并定义了其上的数值操作。


> 📝 **注意**
> 有关张量视图的更多信息，请参阅 `tensor-view-doc`。


> 📝 **注意**
> 有关 `torch.Tensor` 的 `torch.dtype`、`torch.device` 和 `torch.layout` 属性的更多信息，请参阅 `tensor-attributes-doc`。


> 📝 **注意**
> 修改张量的方法会带有下划线后缀标记。例如，`torch.FloatTensor.abs_` 就地计算绝对值并返回修改后的张量，而 `torch.FloatTensor.abs` 在新的张量中计算结果。


> 📝 **注意**
> 要更改现有张量的 `torch.device` 和/或 `torch.dtype`，请考虑在张量上使用 `torch.Tensor.to` 方法。


> ⚠️ **警告**
> `torch.Tensor` 的当前实现引入了内存开销，因此在具有许多微小张量的应用程序中可能导致意外的高内存使用。如果这是你的情况，请考虑使用一个大型结构。
>
> ## 张量类参考

::: Tensor()
根据你的使用场景，有几种主要的创建张量的方法。

- 要使用现有数据创建张量，请使用 `torch.tensor`。
- 要创建具有特定大小的张量，请使用 `torch.*` 张量创建操作（参见 `tensor-creation-ops`）。
- 要创建与另一个张量具有相同大小（和相似类型）的张量，请使用 `torch.*_like` 张量创建操作（参见 `tensor-creation-ops`）。
- 要创建与另一个张量类型相似但大小不同的张量，请使用 `tensor.new_*` 创建操作。
- 有一个遗留的构造函数 `torch.Tensor`，不鼓励使用。请改用 `torch.tensor`。


Tensor.\_\_init\_\_(self, data)

此构造函数已弃用，我们建议改用 `torch.tensor`。此构造函数的行为取决于 `data` 的类型。

- 如果 `data` 是一个 Tensor，则返回原始 Tensor 的别名。与 `torch.tensor` 不同，这会跟踪自动微分，并将梯度传播到原始 Tensor。对于这种 `data` 类型，不支持 `device` 关键字参数。
- 如果 `data` 是一个序列或嵌套序列，则创建一个默认 dtype（通常是 `torch.float32`）的张量，其数据是序列中的值，必要时执行强制转换。值得注意的是，这与 `torch.tensor` 不同，因为此构造函数总是构造一个浮点张量，即使输入全是整数。
- 如果 `data` 是一个 `torch.Size`，则返回一个该大小的空张量。

此构造函数不支持显式指定返回张量的 `dtype` 或 `device`。我们建议使用提供此功能的 `torch.tensor`。

Args:

:   data (array_like): 用于构造张量的数据。

Keyword args:

:   

    device (`torch.device`, optional): 返回张量的期望设备。

    :   默认值：如果为 None，则与此张量相同的 `torch.device`。


Tensor.T


Tensor.H


Tensor.mT


Tensor.mH


Tensor.new_tensor Tensor.new_full Tensor.new_empty Tensor.new_ones Tensor.new_zeros

Tensor.is_cuda Tensor.is_quantized Tensor.is_meta Tensor.device Tensor.grad Tensor.ndim Tensor.real Tensor.imag Tensor.nbytes Tensor.itemsize

Tensor.abs [Tensor.abs]() Tensor.absolute [Tensor.absolute]() Tensor.acos [Tensor.acos]() Tensor.arccos [Tensor.arccos]() Tensor.add [Tensor.add]() Tensor.addbmm [Tensor.addbmm]() Tensor.addcdiv [Tensor.addcdiv]() Tensor.addcmul [Tensor.addcmul]() Tensor.addmm [Tensor.addmm]() Tensor.sspaddmm Tensor.addmv [Tensor.addmv]() Tensor.addr [Tensor.addr]() Tensor.adjoint Tensor.allclose Tensor.amax Tensor.amin Tensor.aminmax Tensor.angle [Tensor.apply]() Tensor.argmax Tensor.argmin Tensor.argsort Tensor.argwhere Tensor.asin [Tensor.asin]() Tensor.arcsin [Tensor.arcsin]() Tensor.as_strided Tensor.atan [Tensor.atan]() Tensor.arctan [Tensor.arctan]() Tensor.atan2 [Tensor.atan2]() Tensor.arctan2 [Tensor.arctan2]() Tensor.all Tensor.any Tensor.backward Tensor.baddbmm [Tensor.baddbmm]() Tensor.bernoulli [Tensor.bernoulli]() Tensor.bfloat16 Tensor.bincount Tensor.bitwise_not [Tensor.bitwise_not]() Tensor.bitwise_and [Tensor.bitwise_and]() Tensor.bitwise_or [Tensor.bitwise_or]() Tensor.bitwise_xor [Tensor.bitwise_xor]() Tensor.bitwise_left_shift [Tensor.bitwise_left_shift]() Tensor.bitwise_right_shift [Tensor.bitwise_right_shift]() Tensor.bmm Tensor.bool Tensor.byte Tensor.broadcast_to [Tensor.cauchy]() Tensor.ceil [Tensor.ceil]() Tensor.char Tensor.cholesky Tensor.cholesky_inverse Tensor.cholesky_solve Tensor.chunk Tensor.clamp [Tensor.clamp]() Tensor.clip [Tensor.clip]() Tensor.clone Tensor.contiguous [Tensor.copy]() Tensor.conj Tensor.conj_physical [Tensor.conj_physical]() Tensor.resolve_conj Tensor.resolve_neg Tensor.copysign [Tensor.copysign]() Tensor.cos [Tensor.cos]() Tensor.cosh [Tensor.cosh]() Tensor.corrcoef Tensor.count_nonzero Tensor.cov Tensor.acosh [Tensor.acosh]() Tensor.arccosh [Tensor.arccosh]() Tensor.cpu Tensor.cross Tensor.cuda Tensor.logcumsumexp Tensor.cummax Tensor.cummin Tensor.cumprod [Tensor.cumprod]() Tensor.cumsum [Tensor.cumsum]() Tensor.chalf Tensor.cfloat Tensor.cdouble Tensor.data_ptr Tensor.deg2rad Tensor.dequantize Tensor.det Tensor.dense_dim Tensor.detach [Tensor.detach]() Tensor.diag Tensor.diag_embed Tensor.diagflat Tensor.diagonal Tensor.diagonal_scatter [Tensor.fill_diagonal]() Tensor.fmax Tensor.fmin Tensor.diff Tensor.digamma [Tensor.digamma]() Tensor.dim Tensor.dim_order Tensor.dist Tensor.div [Tensor.div]() Tensor.divide [Tensor.divide]() Tensor.dot Tensor.double Tensor.dsplit Tensor.element_size Tensor.eq [Tensor.eq]() Tensor.equal Tensor.erf [Tensor.erf]() Tensor.erfc [Tensor.erfc]() Tensor.erfinv [Tensor.erfinv]() Tensor.exp [Tensor.exp]() Tensor.expm1 [Tensor.expm1]() Tensor.expand Tensor.expand_as [Tensor.exponential]() Tensor.fix [Tensor.fix]() [Tensor.fill]() Tensor.flatten Tensor.flip Tensor.fliplr Tensor.flipud Tensor.float Tensor.float_power [Tensor.float_power]() Tensor.floor [Tensor.floor]() Tensor.floor_divide [Tensor.floor_divide]() Tensor.fmod [Tensor.fmod]() Tensor.frac [Tensor.frac]() Tensor.frexp Tensor.gather Tensor.gcd [Tensor.gcd]() Tensor.ge [Tensor.ge]() Tensor.greater_equal [Tensor.greater_equal]() [Tensor.geometric]() Tensor.geqrf Tensor.ger Tensor.get_device Tensor.gt [Tensor.gt]() Tensor.greater [Tensor.greater]() Tensor.half Tensor.hardshrink Tensor.heaviside Tensor.histc Tensor.histogram Tensor.hsplit Tensor.hypot [Tensor.hypot]() Tensor.i0 [Tensor.i0]() Tensor.igamma [Tensor.igamma]() Tensor.igammac [Tensor.igammac]() [Tensor.index_add]() Tensor.index_add [Tensor.index_copy]() Tensor.index_copy [Tensor.index_fill]() Tensor.index_fill [Tensor.index_put]() Tensor.index_put [Tensor.index_reduce]() Tensor.index_reduce Tensor.index_select Tensor.indices Tensor.inner Tensor.int Tensor.int_repr Tensor.inverse Tensor.isclose Tensor.isfinite Tensor.isinf Tensor.isposinf Tensor.isneginf Tensor.isnan Tensor.is_contiguous Tensor.is_complex Tensor.is_conj Tensor.is_floating_point Tensor.is_inference Tensor.is_leaf Tensor.is_pinned Tensor.is_set_to Tensor.is_shared Tensor.is_signed Tensor.is_sparse Tensor.istft Tensor.isreal Tensor.item Tensor.kthvalue Tensor.lcm [Tensor.lcm]() Tensor.ldexp [Tensor.ldexp]() Tensor.le [Tensor.le]() Tensor.less_equal [Tensor.less_equal]() Tensor.lerp [Tensor.lerp]() Tensor.lgamma [Tensor.lgamma]() Tensor.log [Tensor.log]() Tensor.logdet Tensor.log10 [Tensor.log10]() Tensor.log1p [Tensor.log1p]() Tensor.log2 [Tensor.log2]() [Tensor.log_normal]() Tensor.logaddexp Tensor.logaddexp2 Tensor.logsumexp Tensor.logical_and [Tensor.logical_and]() Tensor.logical_not [Tensor.logical_not]() Tensor.logical_or [Tensor.logical_or]() Tensor.logical_xor [Tensor.logical_xor]() Tensor.logit [Tensor.logit]() Tensor.long Tensor.lt [Tensor.lt]() Tensor.less [Tensor.less]() Tensor.lu Tensor.lu_solve Tensor.as_subclass [Tensor.map]() [Tensor.masked_scatter]() Tensor.masked_scatter [Tensor.masked_fill]() Tensor.masked_fill Tensor.masked_select Tensor.matmul Tensor.matrix_power Tensor.matrix_exp Tensor.max Tensor.maximum Tensor.mean Tensor.module_load Tensor.nanmean Tensor.median Tensor.nanmedian Tensor.min Tensor.minimum Tensor.mm Tensor.smm Tensor.mode Tensor.movedim Tensor.moveaxis Tensor.msort Tensor.mul [Tensor.mul]() Tensor.multiply [Tensor.multiply]() Tensor.multinomial Tensor.mv Tensor.mvlgamma [Tensor.mvlgamma]() Tensor.nansum Tensor.narrow Tensor.narrow_copy Tensor.ndimension Tensor.nan_to_num [Tensor.nan_to_num]() Tensor.ne [Tensor.ne]() Tensor.not_equal [Tensor.not_equal]() Tensor.neg [Tensor.neg]() Tensor.negative [Tensor.negative]() Tensor.nelement Tensor.nextafter [Tensor.nextafter]() Tensor.nonzero Tensor.norm [Tensor.normal]() Tensor.numel Tensor.numpy Tensor.orgqr Tensor.ormqr Tensor.outer Tensor.permute Tensor.pin_memory Tensor.pinverse Tensor.polygamma [Tensor.polygamma]() Tensor.positive Tensor.pow [Tensor.pow]() Tensor.prod [Tensor.put]() Tensor.qr Tensor.qscheme Tensor.quantile Tensor.nanquantile Tensor.q_scale Tensor.q_zero_point Tensor.q_per_channel_scales Tensor.q_per_channel_zero_points Tensor.q_per_channel_axis Tensor.rad2deg [Tensor.random]() Tensor.ravel Tensor.reciprocal [Tensor.reciprocal]() Tensor.record_stream Tensor.register_hook Tensor.register_post_accumulate_grad_hook Tensor.remainder [Tensor.remainder]() Tensor.renorm [Tensor.renorm]() Tensor.repeat Tensor.repeat_interleave Tensor.requires_grad [Tensor.requires_grad]() Tensor.reshape Tensor.reshape_as [Tensor.resize]() [Tensor.resize_as]() Tensor.retain_grad Tensor.retains_grad Tensor.roll Tensor.rot90 Tensor.round [Tensor.round]() Tensor.rsqrt [Tensor.rsqrt]() Tensor.scatter [Tensor.scatter]() [Tensor.scatter_add]() Tensor.scatter_add [Tensor.scatter_reduce]() Tensor.scatter_reduce Tensor.select Tensor.select_scatter [Tensor.set]() [Tensor.share_memory]() Tensor.short Tensor.sigmoid [Tensor.sigmoid]() Tensor.sign [Tensor.sign]() Tensor.signbit Tensor.sgn [Tensor.sgn]() Tensor.sin [Tensor.sin]() Tensor.sinc [Tensor.sinc]() Tensor.sinh [Tensor.sinh]() Tensor.asinh [Tensor.asinh]() Tensor.arcsinh [Tensor.arcsinh]() Tensor.shape Tensor.size Tensor.slogdet Tensor.slice_scatter Tensor.softmax Tensor.sort Tensor.split Tensor.sparse_mask Tensor.sparse_dim Tensor.sqrt [Tensor.sqrt]() Tensor.square [Tensor.square]() Tensor.squeeze [Tensor.squeeze]() Tensor.std Tensor.stft Tensor.storage Tensor.untyped_storage Tensor.storage_offset Tensor.storage_type Tensor.stride Tensor.sub [Tensor.sub]() Tensor.subtract [Tensor.subtract]() Tensor.sum Tensor.sum_to_size Tensor.svd Tensor.swapaxes Tensor.swapdims Tensor.t [Tensor.t]() Tensor.tensor_split Tensor.tile Tensor.to Tensor.to_mkldnn Tensor.take Tensor.take_along_dim Tensor.tan [Tensor.tan]() Tensor.tanh [Tensor.tanh]() Tensor.atanh [Tensor.atanh]() Tensor.arctanh [Tensor.arctanh]() Tensor.tolist Tensor.topk Tensor.to_dense Tensor.to_sparse Tensor.to_sparse_csr Tensor.to_sparse_csc Tensor.to_sparse_bsr Tensor.to_sparse_bsc Tensor.trace Tensor.transpose [Tensor.transpose]() Tensor.triangular_solve Tensor.tril [Tensor.tril]() Tensor.triu [Tensor.triu]() Tensor.true_divide [Tensor.true_divide]() Tensor.trunc [Tensor.trunc]() Tensor.type Tensor.type_as Tensor.unbind Tensor.unflatten Tensor.unfold [Tensor.uniform]() Tensor.unique Tensor.unique_consecutive Tensor.unsqueeze [Tensor.unsqueeze]() Tensor.values Tensor.var Tensor.vdot Tensor.view Tensor.view_as Tensor.vsplit Tensor.where Tensor.xlogy [Tensor.xlogy]() Tensor.xpu [Tensor.zero]()
