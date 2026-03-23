torch
=====
.. automodule:: torch
.. currentmodule:: torch

张量
-------
.. autosummary::
    :toctree: generated
    :nosignatures:

    is_tensor
    is_storage
    is_complex
    is_conj
    is_floating_point
    is_nonzero
    set_default_dtype
    get_default_dtype
    set_default_device
    get_default_device
    set_default_tensor_type
    numel
    set_printoptions
    set_flush_denormal

.. _tensor-creation-ops:

创建操作
~~~~~~~~~~~~

.. note::
    随机采样创建操作列在 :ref:`random-sampling` 下，包括：
    :func:`torch.rand`
    :func:`torch.rand_like`
    :func:`torch.randn`
    :func:`torch.randn_like`
    :func:`torch.randint`
    :func:`torch.randint_like`
    :func:`torch.randperm`
    你也可以使用 :func:`torch.empty` 配合 :ref:`inplace-random-sampling`
    中的方法来创建值从更广泛分布中采样的 :class:`torch.Tensor`。

.. autosummary::
    :toctree: generated
    :nosignatures:

    tensor
    sparse_coo_tensor
    sparse_csr_tensor
    sparse_csc_tensor
    sparse_bsr_tensor
    sparse_bsc_tensor
    asarray
    as_tensor
    as_strided
    from_file
    from_numpy
    from_dlpack
    frombuffer
    zeros
    zeros_like
    ones
    ones_like
    arange
    range
    linspace
    logspace
    eye
    empty
    empty_like
    empty_strided
    full
    full_like
    quantize_per_tensor
    quantize_per_channel
    dequantize
    complex
    polar
    heaviside

.. _indexing-slicing-joining:

索引、切片、连接、变换操作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    adjoint
    argwhere
    cat
    concat
    concatenate
    conj
    chunk
    dsplit
    column_stack
    dstack
    gather
    hsplit
    hstack
    index_add
    index_copy
    index_reduce
    index_select
    masked_select
    movedim
    moveaxis
    narrow
    narrow_copy
    nonzero
    permute
    reshape
    row_stack
    select
    scatter
    diagonal_scatter
    select_scatter
    slice_scatter
    scatter_add
    scatter_reduce
    segment_reduce
    split
    squeeze
    stack
    swapaxes
    swapdims
    t
    take
    take_along_dim
    tensor_split
    tile
    transpose
    unbind
    unravel_index
    unsqueeze
    vsplit
    vstack
    where

.. _accelerators:

加速器
----------------------------------
在 PyTorch 代码库中，我们将“加速器”定义为与 CPU 一起使用以加速计算的 :class:`torch.device`。这些设备使用异步执行方案，以 :class:`torch.Stream` 和 :class:`torch.Event` 作为执行同步的主要方式。我们还假设在给定主机上一次只能有一个这样的加速器可用。这允许我们将当前加速器用作相关概念（如固定内存、Stream 设备类型、FSDP 等）的默认设备。

截至目前，加速器设备包括（无特定顺序）：:doc:`"CUDA" <cuda>`、:doc:`"MTIA" <mtia>`、:doc:`"XPU" <xpu>`、:doc:`"MPS" <mps>`、"HPU" 以及 PrivateUse1（许多不在 PyTorch 代码库本身的设备）。

PyTorch 生态系统中的许多工具使用 fork 来创建子进程（例如数据加载或算子内并行），因此尽可能延迟任何会阻止后续 fork 的操作非常重要。这一点在这里尤其重要，因为大多数加速器的初始化都有这种效果。实际上，你应该记住，默认情况下检查 :func:`torch.accelerator.current_accelerator` 是一个编译时检查，因此它始终是 fork 安全的。相反，向此函数传递 ``check_available=True`` 标志或调用 :func:`torch.accelerator.is_available()` 通常会阻止后续的 fork。

一些后端提供了一个实验性的可选选项，以使运行时可用性检查成为 fork 安全的。例如，使用 CUDA 设备时，可以使用 ``PYTORCH_NVML_BASED_CUDA_CHECK=1``。

.. autosummary::
    :toctree: generated
    :nosignatures:

    Stream
    Event

.. _generators:

生成器
----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    Generator

.. _random-sampling:

随机采样
----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    seed
    manual_seed
    initial_seed
    get_rng_state
    set_rng_state

.. autoattribute:: torch.default_generator
   :annotation:  返回默认的 CPU torch.Generator

.. 以下内容实际上似乎不存在。
   https://github.com/pytorch/pytorch/issues/27780
   .. autoattribute:: torch.cuda.default_generators
      :annotation:  如果 cuda 可用，返回一个默认 CUDA torch.Generator 的元组。
                    返回的 CUDA torch.Generator 数量等于系统中可用的 GPU 数量。
.. autosummary::
    :toctree: generated
    :nosignatures:

    bernoulli
    multinomial
    normal
    poisson
    rand
    rand_like
    randint
    randint_like
    randn
    randn_like
    randperm

.. _inplace-random-sampling:

原地随机采样
~~~~~~~~~~~~~~~~~~~~~~~~

在张量上还定义了一些原地随机采样函数。点击链接查看其文档：

- :func:`torch.Tensor.bernoulli_` - :func:`torch.bernoulli` 的原地版本
- :func:`torch.Tensor.cauchy_` - 从柯西分布中抽取的数字
- :func:`torch.Tensor.exponential_` - 从指数分布中抽取的数字
- :func:`torch.Tensor.geometric_` - 从几何分布中抽取的元素
- :func:`torch.Tensor.log_normal_` - 从对数正态分布中抽取的样本
- :func:`torch.Tensor.normal_` - :func:`torch.normal` 的原地版本
- :func:`torch.Tensor.random_` - 从离散均匀分布中采样的数字
- :func:`torch.Tensor.uniform_` - 从连续均匀分布中采样的数字

准随机采样
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: sobolengine.rst

    quasirandom.SobolEngine

序列化
----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    save
    load

并行性
----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    get_num_threads
    set_num_threads
    get_num_interop_threads
    set_num_interop_threads

.. _torch-rst-local-disable-grad:

局部禁用梯度计算
--------------------------------------
上下文管理器 :func:`torch.no_grad`、:func:`torch.enable_grad` 和 :func:`torch.set_grad_enabled` 有助于在局部禁用和启用梯度计算。关于它们用法的更多细节，请参阅 :ref:`locally-disable-grad`。这些上下文管理器是线程局部的，因此如果你使用 ``threading`` 模块等将工作发送到另一个线程，它们将不起作用。

示例::

  >>> x = torch.zeros(1, requires_grad=True)
  >>> with torch.no_grad():
  ...     y = x * 2
  >>> y.requires_grad
  False

  >>> is_train = False
  >>> with torch.set_grad_enabled(is_train):
  ...     y = x * 2
  >>> y.requires_grad
  False

  >>> torch.set_grad_enabled(True)  # 这也可以作为函数使用
  >>> y = x * 2
  >>> y.requires_grad
  True

  >>> torch.set_grad_enabled(False)
  >>> y = x * 2
  >>> y.requires_grad
  False

.. autosummary::
    :toctree: generated
    :nosignatures:

    no_grad
    enable_grad
    autograd.grad_mode.set_grad_enabled
    is_grad_enabled
    autograd.grad_mode.inference_mode
    is_inference_mode_enabled

数学运算
---------------

常量
~~~~~~~~~~~~~~~~~~~~~~

======================================= ===========================================
``e``                                       自然对数的底数，欧拉数 (~2.7183)。:attr:`math.e` 的别名。
``inf``                                     浮点正无穷大。:attr:`math.inf` 的别名。
``nan``                                     浮点“非数字”值。此值不是合法数字。:attr:`math.nan` 的别名。
``pi``                                      圆的周长与直径之比 (~3.1416)。:attr:`math.pi` 的别名。
======================================= ===========================================

逐点运算
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    absolute
    acos
    arccos
    acosh
    arccosh
    add
    addcdiv
    addcmul
    angle
    asin
    arcsin
    asinh
    arcsinh
    atan
    arctan
    atanh
    arctanh
    atan2
    arctan2
    bitwise_not
    bitwise_and
    bitwise_or
    bitwise_xor
    bitwise_left_shift
    bitwise_right_shift
    ceil
    clamp
    clip
    conj_physical
    copysign
    cos
    cosh
    deg2rad
    div
    divide
    digamma
    erf
    erfc
    erfinv
    exp
    exp2
    expm1
    fake_quantize_per_channel_affine
    fake_quantize_per_tensor_affine
    fix
    float_power
    floor
    floor_divide
    fmod
    frac
    frexp
    gradient
    imag
    ldexp
    lerp
    lgamma
    log
    log10
    log1p
    log2
    logaddexp
    logaddexp2
    logical_and
    logical_not
    logical_or
    logical_xor
    logit
    hypot
    i0
    igamma
    igammac
    mul
    multiply
    mvlgamma
    nan_to_num
    neg
    negative
    nextafter
    polygamma
    positive
    pow
    quantized_batch_norm
    quantized_max_pool1d
    quantized_max_pool2d
    rad2deg
    real
    reciprocal
    remainder
    round
    rsqrt
    sigmoid
    sign
    sgn
    signbit
    sin
    sinc
    sinh
    softmax
    sqrt
    square
    sub
    subtract
    tan
    tanh
    true_divide
    trunc
    xlogy

归约运算
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    argmax
    argmin
    amax
    amin
    aminmax
    all
    any
    max
    min
    dist
    logsumexp
    mean
    nanmean
    median
    nanmedian
    mode
    norm
    nansum
    prod
    quantile
    nanquantile
    std
    std_mean
    sum
    unique
    unique_consecutive
    var
    var_mean
    count_nonzero
    hash_tensor

比较运算
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    allclose
    argsort
    eq
    equal
    ge
    greater_equal
    gt
    greater
    isclose
    isfinite
    isin
    isinf
    isposinf
    isneginf
    isnan
    isreal
    kthvalue
    le
    less_equal
    lt
    less
    maximum
    minimum
    fmax
    fmin
    ne
    not_equal
    sort
    topk
    msort

谱运算
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    stft
    istft
    bartlett_window
    blackman_window
    hamming_window
    hann_window
    kaiser_window

其他运算
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    atleast_1d
    atleast_2d
    atleast_3d
    bincount
    block_diag
    broadcast_tensors
    broadcast_to
    broadcast_shapes
    bucketize
    cartesian_prod
    cdist
    clone
    combinations
    corrcoef
    cov
    cross
    cummax
    cummin
    cumprod
    cumsum
    diag
    diag_embed
    diagflat
    diagonal
    diff
    einsum
    flatten
    flip
    fliplr
    flipud
    kron
    rot90
    gcd
    histc
    histogram
    histogramdd
    meshgrid
    lcm
    logcumsumexp
    ravel
    renorm
    repeat_interleave
    roll
    searchsorted
    tensordot
    trace
    tril
    tril_indices
    triu
    triu_indices
    unflatten
    vander
    view_as_real
    view_as_complex
    resolve_conj
    resolve_neg

BLAS 和 LAPACK 运算
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

addbmm
addmm
addmv
addr
baddbmm
bmm
chain_matmul
cholesky
cholesky_inverse
cholesky_solve
dot
geqrf
ger
inner
inverse
det
logdet
slogdet
lu
lu_solve
lu_unpack
matmul
matrix_power
matrix_exp
mm
mv
orgqr
ormqr
outer
pinverse
qr
svd
svd_lowrank
pca_lowrank
lobpcg
trapz
trapezoid
cumulative_trapezoid
triangular_solve
vdot

Foreach 操作
~~~~~~~~~~~~~~~~~~

.. warning::
    此 API 处于测试阶段，未来可能会更改。
    不支持前向模式自动微分。

.. autosummary::
    :toctree: generated
    :nosignatures:

    _foreach_abs
    _foreach_abs_
    _foreach_acos
    _foreach_acos_
    _foreach_asin
    _foreach_asin_
    _foreach_atan
    _foreach_atan_
    _foreach_ceil
    _foreach_ceil_
    _foreach_cos
    _foreach_cos_
    _foreach_cosh
    _foreach_cosh_
    _foreach_erf
    _foreach_erf_
    _foreach_erfc
    _foreach_erfc_
    _foreach_exp
    _foreach_exp_
    _foreach_expm1
    _foreach_expm1_
    _foreach_floor
    _foreach_floor_
    _foreach_log
    _foreach_log_
    _foreach_log10
    _foreach_log10_
    _foreach_log1p
    _foreach_log1p_
    _foreach_log2
    _foreach_log2_
    _foreach_neg
    _foreach_neg_
    _foreach_tan
    _foreach_tan_
    _foreach_sin
    _foreach_sin_
    _foreach_sinh
    _foreach_sinh_
    _foreach_round
    _foreach_round_
    _foreach_sqrt
    _foreach_sqrt_
    _foreach_lgamma
    _foreach_lgamma_
    _foreach_frac
    _foreach_frac_
    _foreach_reciprocal
    _foreach_reciprocal_
    _foreach_sigmoid
    _foreach_sigmoid_
    _foreach_trunc
    _foreach_trunc_
    _foreach_zero_

工具函数
----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    compiled_with_cxx11_abi
    result_type
    can_cast
    promote_types
    use_deterministic_algorithms
    are_deterministic_algorithms_enabled
    is_deterministic_algorithms_warn_only_enabled
    set_deterministic_debug_mode
    get_deterministic_debug_mode
    set_float32_matmul_precision
    get_float32_matmul_precision
    set_warn_always
    get_device_module
    is_warn_always_enabled
    vmap
    _assert
    typename

符号数字
----------------
.. autoclass:: SymInt
    :members:

.. autoclass:: SymFloat
    :members:

.. autoclass:: SymBool
    :members:

.. autosummary::
    :toctree: generated
    :nosignatures:

    sym_float
    sym_fresh_size
    sym_int
    sym_max
    sym_min
    sym_not
    sym_ite
    sym_sqrt
    sym_sum

导出路径
-------------
.. autosummary::
    :toctree: generated
    :nosignatures:

.. warning::
    此功能为原型，未来可能会有破坏兼容性的更改。

    export
    generated/exportdb/index

控制流
------------

.. warning::
    此功能为原型，未来可能会有破坏兼容性的更改。

.. autosummary::
    :toctree: generated
    :nosignatures:

    cond

优化
-------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    compile

`torch.compile 文档 <https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html>`__

操作符标签
------------------------------------
.. autoclass:: Tag
    :members:

.. 仅用于跟踪而添加的空子模块。
.. py:module:: torch.contrib
.. py:module:: torch.utils.backcompat

.. 此模块仅在内部用于 ROCm 构建。
.. py:module:: torch.utils.hipify

.. 此模块需要文档。暂时添加于此以便跟踪
.. py:module:: torch.utils.model_dump
.. py:module:: torch.utils.viz
.. py:module:: torch.quasirandom
.. py:module:: torch.return_types
.. py:module:: torch.serialization
.. py:module:: torch.signal.windows.windows
.. py:module:: torch.sparse.semi_structured
.. py:module:: torch.storage
.. py:module:: torch.torch_version
.. py:module:: torch.types
.. py:module:: torch.version

.. 编译器配置模块 - 文档位于 torch.compiler.config.md
.. py:module:: torch.compiler.config
   :noindex:

.. 隐藏的别名（例如 torch.functional.broadcast_tensors()）。我们希望仅 `torch.broadcast_tensors()` 可见。
.. toctree::
    :hidden:

    torch.aliases.md