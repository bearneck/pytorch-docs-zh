```{eval-rst}
.. automodule:: torch.masked
.. automodule:: torch.masked.maskedtensor
```

```{eval-rst}
.. currentmodule:: torch
```

(masked-docs)=

# torch.masked

## 简介

### 动机

:::{warning}
PyTorch 的掩码张量 API 目前处于原型阶段，未来可能会发生变化。
:::

MaskedTensor 作为 {class}`torch.Tensor` 的扩展，为用户提供了以下能力：

* 使用任何掩码语义（例如可变长度张量、nan* 运算符等）
* 区分 0 梯度和 NaN 梯度
* 各种稀疏应用（参见下方教程）

"已指定"和"未指定"在 PyTorch 中有着悠久的历史，但缺乏正式的语义，且肯定缺乏一致性；事实上，MaskedTensor 正是源于普通 {class}`torch.Tensor` 类无法妥善处理的诸多问题。因此，MaskedTensor 的一个主要目标是成为 PyTorch 中"已指定"和"未指定"值的权威来源，使其成为一等公民而非事后考虑。反过来，这应进一步释放[稀疏性](https://pytorch.org/docs/stable/sparse.html)的潜力，实现更安全、更一致的运算符，并为用户和开发者提供更顺畅、更直观的体验。

### 什么是 MaskedTensor？

MaskedTensor 是一个张量子类，由 1) 输入（数据）和 2) 掩码组成。掩码告诉我们输入中的哪些条目应被包含或忽略。

举例来说，假设我们想要掩码掉所有等于 0 的值（用灰色表示）并取最大值：

```{eval-rst}
.. image:: _static/img/masked/tensor_comparison.jpg
      :scale: 50%
```

顶部是普通张量示例，底部是 MaskedTensor，其中所有 0 值都被掩码掉。根据是否使用掩码，这显然会产生不同的结果，但这种灵活的结构允许用户在计算过程中系统地忽略他们希望忽略的任何元素。

我们已经编写了一些现有教程来帮助用户入门，例如：

- [概述 – 新用户的起点，讨论如何使用 MaskedTensor 及其用途](https://docs.pytorch.org/tutorials/unstable/maskedtensor_overview)
- [稀疏性 – MaskedTensor 支持稀疏 COO 和 CSR 数据及掩码张量](https://docs.pytorch.org/tutorials/unstable/maskedtensor_sparsity)
- [Adagrad 稀疏语义 – 一个实际示例，展示 MaskedTensor 如何简化稀疏语义和实现](https://docs.pytorch.org/tutorials/unstable/maskedtensor_adagrad)
- [高级语义 – 讨论为何做出某些决策（例如要求二元/归约操作的掩码匹配）、与 NumPy 的 MaskedArray 的差异以及归约语义](https://docs.pytorch.org/tutorials/unstable/maskedtensor_advanced_semantics)

## 支持的运算符

### 一元运算符

一元运算符是仅包含单个输入的运算符。将它们应用于 MaskedTensor 相对简单：如果数据在给定索引处被掩码，我们应用该运算符，否则我们将继续掩码该数据。

可用的一元运算符包括：

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    absolute
    acos
    arccos
    acosh
    arccosh
    angle
    asin
    arcsin
    asinh
    arcsinh
    atan
    arctan
    atanh
    arctanh
    bitwise_not
    ceil
    clamp
    clip
    conj_physical
    cos
    cosh
    deg2rad
    digamma
    erf
    erfc
    erfinv
    exp
    exp2
    expm1
    fix
    floor
    frac
    lgamma
    log
    log10
    log1p
    log2
    logit
    i0
    isnan
    nan_to_num
    neg
    negative
    positive
    pow
    rad2deg
    reciprocal
    round
    rsqrt
    sigmoid
    sign
    sgn
    signbit
    sin
    sinc
    sinh
    sqrt
    square
    tan
    tanh
    trunc
```

可用的原地一元运算符包括以上所有运算符**除了**：

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    angle
    positive
    signbit
    isnan
```

### 二元运算符

正如您在教程中可能看到的，{class}`MaskedTensor` 也实现了二元操作，但有一个注意事项：两个 MaskedTensor 中的掩码必须匹配，否则将引发错误。如错误提示所述，如果您需要支持特定运算符或对它们的行为方式有建议的语义，请在 GitHub 上提交问题。目前，我们决定采用最保守的实现，以确保用户确切了解正在发生的情况，并对掩码语义的决策保持谨慎。

可用的二元运算符包括：

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    add
    atan2
    arctan2
    bitwise_and
    bitwise_or
    bitwise_xor
    bitwise_left_shift
    bitwise_right_shift
    div
    divide
    floor_divide
    fmod
    logaddexp
    logaddexp2
    mul
    multiply
    nextafter
    remainder
    sub
    subtract
    true_divide
    eq
    ne
    le
    ge
    greater
    greater_equal
    gt
    less_equal
    lt
    less
    maximum
    minimum
    fmax
    fmin
    not_equal
```

可用的原地二元运算符包括以上所有运算符**除了**：

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    logaddexp
    logaddexp2
    equal
    fmin
    minimum
    fmax
```

### 归约操作

以下归约操作可用（支持自动梯度）。更多信息，[概述](https://pytorch.org/tutorials/unstable/maskedtensor_overview.html)教程详细介绍了归约的一些示例，而[高级语义](https://pytorch.org/tutorials/unstable/maskedtensor_advanced_semantics.html)教程则深入讨论了我们是如何决定某些归约语义的。

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

sum
mean
amin
amax
argmin
argmax
prod
all
norm
var
std
```

### 视图和选择函数

我们还包含了一系列视图和选择函数；直观上，这些运算符会同时应用于数据和掩码，然后将结果包装在 {class}`MaskedTensor` 中。举个简单的例子，考虑 {func}`select` 函数：

```python
    >>> data = torch.arange(12, dtype=torch.float).reshape(3, 4)
    >>> data
    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]])
    >>> mask = torch.tensor([[True, False, False, True], [False, True, False, False], [True, True, True, True]])
    >>> mt = masked_tensor(data, mask)
    >>> data.select(0, 1)
    tensor([4., 5., 6., 7.])
    >>> mask.select(0, 1)
    tensor([False,  True, False, False])
    >>> mt.select(0, 1)
    MaskedTensor(
      [      --,   5.0000,       --,       --]
    )
```

目前支持以下操作：

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    atleast_1d
    broadcast_tensors
    broadcast_to
    cat
    chunk
    column_stack
    dsplit
    flatten
    hsplit
    hstack
    kron
    meshgrid
    narrow
    nn.functional.unfold
    ravel
    select
    split
    stack
    t
    transpose
    vsplit
    vstack
    Tensor.expand
    Tensor.expand_as
    Tensor.reshape
    Tensor.reshape_as
    Tensor.unfold
    Tensor.view
```

```{eval-rst}
.. 此模块需要文档记录。暂时添加在此处
.. 用于跟踪目的
.. py:module:: torch.masked.maskedtensor.binary
.. py:module:: torch.masked.maskedtensor.core
.. py:module:: torch.masked.maskedtensor.creation
.. py:module:: torch.masked.maskedtensor.passthrough
.. py:module:: torch.masked.maskedtensor.reductions
.. py:module:: torch.masked.maskedtensor.unary
```