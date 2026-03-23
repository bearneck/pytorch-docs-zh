


# 命名张量操作覆盖范围

请先阅读 `named_tensors-doc` 以了解命名张量的介绍。

本文档是关于*名称推断*的参考，该过程定义了命名张量如何：

1. 使用名称来提供额外的自动运行时正确性检查
2. 将名称从输入张量传播到输出张量

以下是所有支持命名张量的操作及其相关名称推断规则的列表。

如果您在此处未找到列出的操作，但它对您的用例有帮助，请[搜索是否已有相关 issue 被提交](https://github.com/pytorch/pytorch/issues?q=is%3Aopen+is%3Aissue+label%3A%22module%3A+named+tensor%22)，如果没有，请[提交一个](https://github.com/pytorch/pytorch/issues/new/choose)。


> ⚠️ **警告**
> 命名张量 API 是实验性的，可能会发生变化。


":meth:`Tensor.abs`, :func:`torch.abs`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.abs_`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.acos`, :func:`torch.acos`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.acos_`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.add`, :func:`torch.add`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.add_`,:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.addmm`, :func:`torch.addmm`",:ref:`contracts_away_dims-doc`
   :meth:`Tensor.addmm_`,:ref:`contracts_away_dims-doc`
   ":meth:`Tensor.addmv`, :func:`torch.addmv`",:ref:`contracts_away_dims-doc`
   :meth:`Tensor.addmv_`,:ref:`contracts_away_dims-doc`
   :meth:`Tensor.align_as`,参见文档
   :meth:`Tensor.align_to`,参见文档
   ":meth:`Tensor.all`, :func:`torch.all`",None
   ":meth:`Tensor.any`, :func:`torch.any`",None
   ":meth:`Tensor.asin`, :func:`torch.asin`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.asin_`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.atan`, :func:`torch.atan`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.atan2`, :func:`torch.atan2`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.atan2_`,:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.atan_`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.bernoulli`, :func:`torch.bernoulli`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.bernoulli_`,None
   :meth:`Tensor.bfloat16`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.bitwise_not`, :func:`torch.bitwise_not`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.bitwise_not_`,None
   ":meth:`Tensor.bmm`, :func:`torch.bmm`",:ref:`contracts_away_dims-doc`
   :meth:`Tensor.bool`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.byte`,:ref:`keeps_input_names-doc`
   :func:`torch.cat`,:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.cauchy_`,None
   ":meth:`Tensor.ceil`, :func:`torch.ceil`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.ceil_`,None
   :meth:`Tensor.char`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.chunk`, :func:`torch.chunk`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.clamp`, :func:`torch.clamp`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.clamp_`,None
   :meth:`Tensor.copy_`,:ref:`out_function_semantics-doc`
   ":meth:`Tensor.cos`, :func:`torch.cos`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.cos_`,None
   ":meth:`Tensor.cosh`, :func:`torch.cosh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.cosh_`,None
   ":meth:`Tensor.acosh`, :func:`torch.acosh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.acosh_`,None
   :meth:`Tensor.cpu`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.cuda`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.cumprod`, :func:`torch.cumprod`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.cumsum`, :func:`torch.cumsum`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.data_ptr`,None
   ":meth:`Tensor.deg2rad`, :func:`torch.deg2rad`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.deg2rad_`,None
   ":meth:`Tensor.detach`, :func:`torch.detach`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.detach_`,None
   ":attr:`Tensor.device`, :func:`torch.device`",None
   ":meth:`Tensor.digamma`, :func:`torch.digamma`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.digamma_`,None
   :meth:`Tensor.dim`,None
   ":meth:`Tensor.div`, :func:`torch.div`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.div_`,:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.dot`, :func:`torch.dot`",None
   :meth:`Tensor.double`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.element_size`,None
   :func:`torch.empty`,:ref:`factory-doc`
   :func:`torch.empty_like`,:ref:`factory-doc`
   ":meth:`Tensor.eq`, :func:`torch.eq`",:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.erf`, :func:`torch.erf`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.erf_`,None
   ":meth:`Tensor.erfc`, :func:`torch.erfc`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.erfc_`,None
   ":meth:`Tensor.erfinv`, :func:`torch.erfinv`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.erfinv_`,None
   ":meth:`Tensor.exp`, :func:`torch.exp`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.exp_`,None
   :meth:`Tensor.expand`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.expm1`, :func:`torch.expm1`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.expm1_`,None
   :meth:`Tensor.exponential_`,None
   :meth:`Tensor.fill_`,None
   ":meth:`Tensor.flatten`, :func:`torch.flatten`",参见文档
   :meth:`Tensor.float`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.floor`, :func:`torch.floor`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.floor_`,None
   ":meth:`Tensor.frac`, :func:`torch.frac`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.frac_`,None
   ":meth:`Tensor.ge`, :func:`torch.ge`",:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.get_device`, :func:`torch.get_device`",None
   :attr:`Tensor.grad`,None
   ":meth:`Tensor.gt`, :func:`torch.gt`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.half`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.has_names`,参见文档
   ":meth:`Tensor.index_fill`, :func:`torch.index_fill`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.index_fill_`,None
   :meth:`Tensor.int`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.is_contiguous`,None
   :attr:`Tensor.is_cuda`,None
   ":meth:`Tensor.is_floating_point`, :func:`torch.is_floating_point`",None
   :attr:`Tensor.is_leaf`,None
   :meth:`Tensor.is_pinned`,None
   :meth:`Tensor.is_shared`,None
   ":meth:`Tensor.is_signed`, :func:`torch.is_signed`",None
   :attr:`Tensor.is_sparse`,None
   :attr:`Tensor.is_sparse_csr`,None
   :func:`torch.is_tensor`,None
   :meth:`Tensor.item`,None
   :attr:`Tensor.itemsize`,None
   ":meth:`Tensor.kthvalue`, :func:`torch.kthvalue`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.le`, :func:`torch.le`",:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.log`, :func:`torch.log`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.log10`, :func:`torch.log10`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.log10_`,None
   ":meth:`Tensor.log1p`, :func:`torch.log1p`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.log1p_`,None
   ":meth:`Tensor.log2`, :func:`torch.log2`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.log2_`,None
   :meth:`Tensor.log_`,None
   :meth:`Tensor.log_normal_`,None
   ":meth:`Tensor.logical_not`, :func:`torch.logical_not`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.logical_not_`,None
   ":meth:`Tensor.logsumexp`, :func:`torch.logsumexp`",:ref:`removes_dimensions-doc`
   :meth:`Tensor.long`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.lt`, :func:`torch.lt`",:ref:`unifies_names_from_inputs-doc`
   :func:`torch.manual_seed`,None
   ":meth:`Tensor.masked_fill`, :func:`torch.masked_fill`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.masked_fill_`,None
   ":meth:`Tensor.masked_select`, :func:`torch.masked_select`",将掩码与输入对齐，然后统一来自输入张量的名称
   ":meth:`Tensor.matmul`, :func:`torch.matmul`",:ref:`contracts_away_dims-doc`
   ":meth:`Tensor.mean`, :func:`torch.mean`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.median`, :func:`torch.median`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.nanmedian`, :func:`torch.nanmedian`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.mm`, :func:`torch.mm`",:ref:`contracts_away_dims-doc`
   ":meth:`Tensor.mode`, :func:`torch.mode`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.mul`, :func:`torch.mul`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.mul_`,:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.mv`, :func:`torch.mv`",:ref:`contracts_away_dims-doc`
   :attr:`Tensor.names`,参见文档
   ":meth:`Tensor.narrow`, :func:`torch.narrow`",:ref:`keeps_input_names-doc`
   :attr:`Tensor.nbytes`,None
   :attr:`Tensor.ndim`,None
   :meth:`Tensor.ndimension`,None
   ":meth:`Tensor.ne`, :func:`torch.ne`",:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.neg`, :func:`torch.neg`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.neg_`,None
   :func:`torch.normal`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.normal_`,None
   ":meth:`Tensor.numel`, :func:`torch.numel`",None
   :func:`torch.ones`,:ref:`factory-doc`
   ":meth:`Tensor.pow`, :func:`torch.pow`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.pow_`,None
   ":meth:`Tensor.prod`, :func:`torch.prod`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.rad2deg`, :func:`torch.rad2deg`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.rad2deg_`,None
   :func:`torch.rand`,:ref:`factory-doc`
   :func:`torch.rand`,:ref:`factory-doc`
   :func:`torch.randn`,:ref:`factory-doc`
   :func:`torch.randn`,:ref:`factory-doc`
   :meth:`Tensor.random_`,None
   ":meth:`Tensor.reciprocal`, :func:`torch.reciprocal`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.reciprocal_`,None
   :meth:`Tensor.refine_names`,参见文档
   :meth:`Tensor.register_hook`,None
   :meth:`Tensor.register_post_accumulate_grad_hook`,None
   :meth:`Tensor.rename`,参见文档
   :meth:`Tensor.rename_`,参见文档
   :attr:`Tensor.requires_grad`,None
   :meth:`Tensor.requires_grad_`,None
   :meth:`Tensor.resize_`,仅允许不改变形状的调整大小操作
   :meth:`Tensor.resize_as_`,仅允许不改变形状的调整大小操作
   ":meth:`Tensor.round`, :func:`torch.round`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.round_`,None
   ":meth:`Tensor.rsqrt`, :func:`torch.rsqrt`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.rsqrt_`,None
   ":meth:`Tensor.select`, :func:`torch.select`",:ref:`removes_dimensions-doc`
   :meth:`Tensor.short`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.sigmoid`, :func:`torch.sigmoid`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sigmoid_`,None
   ":meth:`Tensor.sign`, :func:`torch.sign`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sign_`,None
   ":meth:`Tensor.sgn`, :func:`torch.sgn`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sgn_`,None
   ":meth:`Tensor.sin`, :func:`torch.sin`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sin_`,None
   ":meth:`Tensor.sinh`, :func:`torch.sinh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sinh_`,None
   ":meth:`Tensor.asinh`, :func:`torch.asinh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.asinh_`,None
   :meth:`Tensor.size`,None
   ":meth:`Tensor.softmax`, :func:`torch.softmax`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.split`, :func:`torch.split`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.sqrt`, :func:`torch.sqrt`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sqrt_`,None
   ":meth:`Tensor.squeeze`, :func:`torch.squeeze`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.std`, :func:`torch.std`",:ref:`removes_dimensions-doc`
   :func:`torch.std_mean`,:ref:`removes_dimensions-doc`
   :meth:`Tensor.stride`,None
   ":meth:`Tensor.sub`, :func:`torch.sub`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.sub_`,:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.sum`, :func:`torch.sum`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.tan`, :func:`torch.tan`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.tan_`,None
   ":meth:`Tensor.tanh`, :func:`torch.tanh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.tanh_`,None
   ":meth:`Tensor.atanh`, :func:`torch.atanh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.atanh_`,None
   :func:`torch.tensor`,:ref:`factory-doc`
   :meth:`Tensor.to`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.topk`, :func:`torch.topk`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.transpose`, :func:`torch.transpose`",:ref:`permutes_dimensions-doc`
   ":meth:`Tensor.trunc`, :func:`torch.trunc`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.trunc_`,None
   :meth:`Tensor.type`,None
   :meth:`Tensor.type_as`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.unbind`, :func:`torch.unbind`",:ref:`removes_dimensions-doc`
   :meth:`Tensor.unflatten`,参见文档
   :meth:`Tensor.uniform_`,None
   ":meth:`Tensor.var`, :func:`torch.var`",:ref:`removes_dimensions-doc`
   :func:`torch.var_mean`,:ref:`removes_dimensions-doc`
   :meth:`Tensor.zero_`,None
   :func:`torch.zeros`,:ref:`factory-doc`

```


## 保持输入名称

所有逐点一元函数以及一些其他一元函数遵循此规则。

- 名称检查：无
- 名称传播：输入张量的名称传播到输出。

```
>>> x = torch.randn(3, 3, names=('N', 'C'))
>>> x.abs().names
('N', 'C')
```


## 移除维度

所有像 `~Tensor.sum` 这样的归约操作通过归约指定维度来移除维度。其他操作如 `~Tensor.select` 和 `~Tensor.squeeze` 也会移除维度。

凡是能传递整数维度索引给操作符的地方，也可以传递维度名称。接受维度索引列表的函数也可以接受维度名称列表。

- 名称检查：如果 `dim` 或 `dims` 作为名称列表传入，检查这些名称是否存在于 `self` 中。
- 名称传播：如果输入张量中由 `dim` 或 `dims` 指定的维度在输出张量中不存在，那么这些维度对应的名称不会出现在 `output.names` 中。

```
>>> x = torch.randn(1, 3, 3, 3, names=('N', 'C', 'H', 'W'))
>>> x.squeeze('N').names
('C', 'H', 'W')

>>> x = torch.randn(3, 3, 3, 3, names=('N', 'C', 'H', 'W'))
>>> x.sum(['N', 'C']).names
('H', 'W')

# 带有 keepdim=True 的归约操作实际上不会移除维度。
>>> x = torch.randn(3, 3, 3, 3, names=('N', 'C', 'H', 'W'))
>>> x.sum(['N', 'C'], keepdim=True).names
('N', 'C', 'H', 'W')
```


## 统一输入的名称

所有二元算术运算遵循此规则。进行广播的操作仍然从右到左按位置广播，以保持与未命名张量的兼容性。要按名称执行显式广播，请使用 `Tensor.align_as`。

- 名称检查：所有名称必须从右到左按位置匹配。即，在 `tensor + other` 中，对于所有 `i` 在 `(-min(tensor.dim(), other.dim()) + 1, -1]` 范围内，`match(tensor.names[i], other.names[i])` 必须为真。
- 名称检查：此外，所有命名维度必须从右到左对齐。在匹配过程中，如果我们将一个命名维度 `A` 与一个未命名维度 `None` 匹配，那么 `A` 不得出现在包含未命名维度的张量中。
- 名称传播：统一两个张量从右到左的成对名称以产生输出名称。

例如，

```
# tensor: Tensor[   N, None]
# other:  Tensor[None,    C]
>>> tensor = torch.randn(3, 3, names=('N', None))
>>> other = torch.randn(3, 3, names=(None, 'C'))
>>> (tensor + other).names
('N', 'C')
```

名称检查：

- `match(tensor.names[-1], other.names[-1])` 为 `True`
- `match(tensor.names[-2], tensor.names[-2])` 为 `True`
- 因为我们将 `tensor` 中的 `None` 与 `'C'` 匹配，检查确保 `'C'` 不存在于 `tensor` 中（确实不存在）。
- 检查确保 `'N'` 不存在于 `other` 中（确实不存在）。

最后，输出名称通过 `[unify('N', None), unify(None, 'C')] = ['N', 'C']` 计算得出。

更多示例：

```
# 维度从右到左不匹配：
# tensor: Tensor[N, C]
# other:  Tensor[   N]
>>> tensor = torch.randn(3, 3, names=('N', 'C'))
>>> other = torch.randn(3, names=('N',))
>>> (tensor + other).names
RuntimeError: 尝试广播 dims ['N', 'C'] 和 dims ['N'] 时出错：维度 'C' 和维度 'N' 从右到左处于相同位置但不匹配。

# 匹配 tensor.names[-1] 和 other.names[-1] 时维度未对齐：
# tensor: Tensor[N, None]
# other:  Tensor[      N]
>>> tensor = torch.randn(3, 3, names=('N', None))
>>> other = torch.randn(3, names=('N',))
>>> (tensor + other).names
RuntimeError: 尝试广播 dims ['N'] 和 dims ['N', None] 时维度未对齐：维度 'N' 在两个列表中从右到左出现的位置不同。
```


> 📝 **注意**
> 在最后两个示例中，都可以通过名称对齐张量然后执行加法。使用 `Tensor.align_as` 按名称对齐张量，或使用 `Tensor.align_to` 将张量对齐到自定义的维度顺序。


## 置换维度

一些操作，如 `Tensor.t()`，会置换维度的顺序。维度名称附加在各个维度上，因此它们也会被置换。

如果操作符接受位置索引 `dim`，它也可以接受维度名称作为 `dim`。

- 名称检查：如果 `dim` 作为名称传入，检查它是否存在于张量中。
- 名称传播：以与正在置换的维度相同的方式置换维度名称。

```
>>> x = torch.randn(3, 3, names=('N', 'C'))
>>> x.transpose('N', 'C').names
('C', 'N')
```


## 收缩维度

矩阵乘法函数遵循此规则的某种变体。让我们先看 `torch.mm`，然后推广到批量矩阵乘法的规则。

对于 `torch.mm(tensor, other)`：

- 名称检查：无
- 名称传播：结果名称为 `(tensor.names[-2], other.names[-1])`。

```
>>> x = torch.randn(3, 3, names=('N', 'D'))
>>> y = torch.randn(3, 3, names=('in', 'out'))
>>> x.mm(y).names
('N', 'out')
```

本质上，矩阵乘法在两个维度上执行点积，将它们收缩。当两个张量进行矩阵乘法时，收缩的维度会消失，不会出现在输出张量中。

`torch.mv`、`torch.dot` 的工作方式类似：名称推断不检查输入名称，并移除参与点积的维度：

```
>>> x = torch.randn(3, 3, names=('N', 'D'))
>>> y = torch.randn(3, names=('something',))
>>> x.mv(y).names
('N',)
```

现在，让我们看看 `torch.matmul(tensor, other)`。假设 `tensor.dim() >= 2` 且 `other.dim() >= 2`。

- 检查名称：检查输入的批次维度是否对齐且可广播。
  关于输入对齐的含义，请参阅 `unifies_names_from_inputs-doc`。
- 传播名称：结果名称通过统一批次维度并移除收缩维度获得：
  `unify(tensor.names[:-2], other.names[:-2]) + (tensor.names[-2], other.names[-1])`。

示例：

```
# 矩阵 Tensor['C', 'D'] 和 Tensor['E', 'F'] 的批次矩阵乘法。
# 'A', 'B' 是批次维度。
>>> x = torch.randn(3, 3, 3, 3, names=('A', 'B', 'C', 'D'))
>>> y = torch.randn(3, 3, 3, names=('B', 'E', 'F'))
>>> torch.matmul(x, y).names
('A', 'B', 'C', 'F')
```

最后，许多矩阵乘法函数都有融合了 `add` 的版本，例如 `addmm` 和 `addmv`。这些函数被视为组合了 `mm` 的名称推断和 `add` 的名称推断。


## 工厂函数

工厂函数现在接受一个新的 `names` 参数，该参数为每个维度关联一个名称。

```
>>> torch.zeros(2, 3, names=('N', 'C'))
tensor([[0., 0., 0.],
        [0., 0., 0.]], names=('N', 'C'))
```


## out 函数和原地操作变体

指定为 `out=` 的张量具有以下行为：

- 如果它没有命名的维度，则操作计算出的名称会传播给它。
- 如果它有任意命名的维度，则操作计算出的名称必须与现有名称完全相等。否则，操作会报错。

所有原地操作方法都会修改输入，使其名称等于名称推断计算出的名称。例如：

```
>>> x = torch.randn(3, 3)
>>> y = torch.randn(3, 3, names=('N', 'C'))
>>> x.names
(None, None)

>>> x += y
>>> x.names
('N', 'C')
```