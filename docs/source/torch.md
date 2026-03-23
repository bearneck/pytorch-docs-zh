# torch

 automodule
torch


 currentmodule
torch


## 张量

 {.autosummary toctree="generated" nosignatures=""}
is_tensor is_storage is_complex is_conj is_floating_point is_nonzero set_default_dtype get_default_dtype set_default_device get_default_device set_default_tensor_type numel set_printoptions set_flush_denormal


### 创建操作 {#tensor-creation-ops}

 note
 title
Note


随机采样创建操作列在 `random-sampling`{.interpreted-text role="ref"} 下，包括： `torch.rand`{.interpreted-text role="func"} `torch.rand_like`{.interpreted-text role="func"} `torch.randn`{.interpreted-text role="func"} `torch.randn_like`{.interpreted-text role="func"} `torch.randint`{.interpreted-text role="func"} `torch.randint_like`{.interpreted-text role="func"} `torch.randperm`{.interpreted-text role="func"} 你也可以使用 `torch.empty`{.interpreted-text role="func"} 配合 `inplace-random-sampling`{.interpreted-text role="ref"} 中的方法来创建值从更广泛分布中采样的 `torch.Tensor`{.interpreted-text role="class"}。


 {.autosummary toctree="generated" nosignatures=""}
tensor sparse_coo_tensor sparse_csr_tensor sparse_csc_tensor sparse_bsr_tensor sparse_bsc_tensor asarray as_tensor as_strided from_file from_numpy from_dlpack frombuffer zeros zeros_like ones ones_like arange range linspace logspace eye empty empty_like empty_strided full full_like quantize_per_tensor quantize_per_channel dequantize complex polar heaviside


### 索引、切片、连接、变换操作 {#indexing-slicing-joining}

 {.autosummary toctree="generated" nosignatures=""}
adjoint argwhere cat concat concatenate conj chunk dsplit column_stack dstack gather hsplit hstack index_add index_copy index_reduce index_select masked_select movedim moveaxis narrow narrow_copy nonzero permute reshape row_stack select scatter diagonal_scatter select_scatter slice_scatter scatter_add scatter_reduce segment_reduce split squeeze stack swapaxes swapdims t take take_along_dim tensor_split tile transpose unbind unravel_index unsqueeze vsplit vstack where


## 加速器 {#accelerators}

在 PyTorch 代码库中，我们将"加速器"定义为与 CPU 一起使用以加速计算的 `torch.device`{.interpreted-text role="class"}。这些设备使用异步执行方案，以 `torch.Stream`{.interpreted-text role="class"} 和 `torch.Event`{.interpreted-text role="class"} 作为执行同步的主要方式。我们还假设在给定主机上一次只能有一个这样的加速器可用。这允许我们将当前加速器用作相关概念（如固定内存、Stream 设备类型、FSDP 等）的默认设备。

截至目前，加速器设备包括（无特定顺序）：`"CUDA" <cuda>`{.interpreted-text role="doc"}、`"MTIA" <mtia>`{.interpreted-text role="doc"}、`"XPU" <xpu>`{.interpreted-text role="doc"}、`"MPS" <mps>`{.interpreted-text role="doc"}、\"HPU\" 以及 PrivateUse1（许多不在 PyTorch 代码库本身的设备）。

PyTorch 生态系统中的许多工具使用 fork 来创建子进程（例如数据加载或算子内并行），因此尽可能延迟任何会阻止后续 fork 的操作非常重要。这一点在这里尤其重要，因为大多数加速器的初始化都有这种效果。实际上，你应该记住，默认情况下检查 `torch.accelerator.current_accelerator`{.interpreted-text role="func"} 是一个编译时检查，因此它始终是 fork 安全的。相反，向此函数传递 `check_available=True` 标志或调用 `torch.accelerator.is_available()`{.interpreted-text role="func"} 通常会阻止后续的 fork。

一些后端提供了一个实验性的可选选项，以使运行时可用性检查成为 fork 安全的。例如，使用 CUDA 设备时，可以使用 `PYTORCH_NVML_BASED_CUDA_CHECK=1`。

 {.autosummary toctree="generated" nosignatures=""}
Stream Event


## 生成器 {#generators}

 {.autosummary toctree="generated" nosignatures=""}
Generator


## 随机采样 {#random-sampling}

 {.autosummary toctree="generated" nosignatures=""}
seed manual_seed initial_seed get_rng_state set_rng_state


 {.autoattribute annotation="返回默认的 CPU torch.Generator"}
torch.default_generator


 {.autosummary toctree="generated" nosignatures=""}
bernoulli multinomial normal poisson rand rand_like randint randint_like randn randn_like randperm


### 原地随机采样 {#inplace-random-sampling}

在张量上还定义了一些原地随机采样函数。点击链接查看其文档：

- `torch.Tensor.bernoulli_`{.interpreted-text role="func"} - `torch.bernoulli`{.interpreted-text role="func"} 的原地版本
- `torch.Tensor.cauchy_`{.interpreted-text role="func"} - 从柯西分布中抽取的数字
- `torch.Tensor.exponential_`{.interpreted-text role="func"} - 从指数分布中抽取的数字
- `torch.Tensor.geometric_`{.interpreted-text role="func"} - 从几何分布中抽取的元素
- `torch.Tensor.log_normal_`{.interpreted-text role="func"} - 从对数正态分布中抽取的样本
- `torch.Tensor.normal_`{.interpreted-text role="func"} - `torch.normal`{.interpreted-text role="func"} 的原地版本
- `torch.Tensor.random_`{.interpreted-text role="func"} - 从离散均匀分布中采样的数字
- `torch.Tensor.uniform_`{.interpreted-text role="func"} - 从连续均匀分布中采样的数字

### 准随机采样

 {.autosummary toctree="generated" nosignatures="" template="sobolengine.rst"}
quasirandom.SobolEngine


## 序列化

 {.autosummary toctree="generated" nosignatures=""}
save load


## 并行性

 {.autosummary toctree="generated" nosignatures=""}
get_num_threads set_num_threads get_num_interop_threads set_num_interop_threads


## 局部禁用梯度计算 {#torch-rst-local-disable-grad}

上下文管理器 `torch.no_grad`{.interpreted-text role="func"}、`torch.enable_grad`{.interpreted-text role="func"} 和 `torch.set_grad_enabled`{.interpreted-text role="func"} 有助于在局部禁用和启用梯度计算。关于它们用法的更多细节，请参阅 `locally-disable-grad`{.interpreted-text role="ref"}。这些上下文管理器是线程局部的，因此如果你使用 `threading` 模块等将工作发送到另一个线程，它们将不起作用。

示例:

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

 {.autosummary toctree="generated" nosignatures=""}
no_grad enable_grad autograd.grad_mode.set_grad_enabled is_grad_enabled autograd.grad_mode.inference_mode is_inference_mode_enabled


## 数学运算

### 常量

  ------- --------------------------------------------------------------------------------------
  `e`     自然对数的底数，欧拉数 (\~2.7183)。`math.e`{.interpreted-text role="attr"} 的别名。
  `inf`   浮点正无穷大。`math.inf`{.interpreted-text role="attr"} 的别名。
  `nan`   浮点"非数字"值。此值不是合法数字。`math.nan`{.interpreted-text role="attr"} 的别名。
  `pi`    圆的周长与直径之比 (\~3.1416)。`math.pi`{.interpreted-text role="attr"} 的别名。
  ------- --------------------------------------------------------------------------------------

### 逐点运算

 {.autosummary toctree="generated" nosignatures=""}
abs absolute acos arccos acosh arccosh add addcdiv addcmul angle asin arcsin asinh arcsinh atan arctan atanh arctanh atan2 arctan2 bitwise_not bitwise_and bitwise_or bitwise_xor bitwise_left_shift bitwise_right_shift ceil clamp clip conj_physical copysign cos cosh deg2rad div divide digamma erf erfc erfinv exp exp2 expm1 fake_quantize_per_channel_affine fake_quantize_per_tensor_affine fix float_power floor floor_divide fmod frac frexp gradient imag ldexp lerp lgamma log log10 log1p log2 logaddexp logaddexp2 logical_and logical_not logical_or logical_xor logit hypot i0 igamma igammac mul multiply mvlgamma nan_to_num neg negative nextafter polygamma positive pow quantized_batch_norm quantized_max_pool1d quantized_max_pool2d rad2deg real reciprocal remainder round rsqrt sigmoid sign sgn signbit sin sinc sinh softmax sqrt square sub subtract tan tanh true_divide trunc xlogy


### 归约运算

 {.autosummary toctree="generated" nosignatures=""}
argmax argmin amax amin aminmax all any max min dist logsumexp mean nanmean median nanmedian mode norm nansum prod quantile nanquantile std std_mean sum unique unique_consecutive var var_mean count_nonzero hash_tensor


### 比较运算

 {.autosummary toctree="generated" nosignatures=""}
allclose argsort eq equal ge greater_equal gt greater isclose isfinite isin isinf isposinf isneginf isnan isreal kthvalue le less_equal lt less maximum minimum fmax fmin ne not_equal sort topk msort


### 谱运算

 {.autosummary toctree="generated" nosignatures=""}
stft istft bartlett_window blackman_window hamming_window hann_window kaiser_window


### 其他运算

 {.autosummary toctree="generated" nosignatures=""}
atleast_1d atleast_2d atleast_3d bincount block_diag broadcast_tensors broadcast_to broadcast_shapes bucketize cartesian_prod cdist clone combinations corrcoef cov cross cummax cummin cumprod cumsum diag diag_embed diagflat diagonal diff einsum flatten flip fliplr flipud kron rot90 gcd histc histogram histogramdd meshgrid lcm logcumsumexp ravel renorm repeat_interleave roll searchsorted tensordot trace tril tril_indices triu triu_indices unflatten vander view_as_real view_as_complex resolve_conj resolve_neg


### BLAS 和 LAPACK 运算

 {.autosummary toctree="generated" nosignatures=""}


addbmm addmm addmv addr baddbmm bmm chain_matmul cholesky cholesky_inverse cholesky_solve dot geqrf ger inner inverse det logdet slogdet lu lu_solve lu_unpack matmul matrix_power matrix_exp mm mv orgqr ormqr outer pinverse qr svd svd_lowrank pca_lowrank lobpcg trapz trapezoid cumulative_trapezoid triangular_solve vdot

### Foreach 操作

 warning
 title
Warning


此 API 处于测试阶段，未来可能会更改。 不支持前向模式自动微分。


 {.autosummary toctree="generated" nosignatures=""}
[foreach_abs]{#foreach_abs} [foreach_abs]{#foreach_abs}\_ [foreach_acos]{#foreach_acos} [foreach_acos]{#foreach_acos}\_ [foreach_asin]{#foreach_asin} [foreach_asin]{#foreach_asin}\_ [foreach_atan]{#foreach_atan} [foreach_atan]{#foreach_atan}\_ [foreach_ceil]{#foreach_ceil} [foreach_ceil]{#foreach_ceil}\_ [foreach_cos]{#foreach_cos} [foreach_cos]{#foreach_cos}\_ [foreach_cosh]{#foreach_cosh} [foreach_cosh]{#foreach_cosh}\_ [foreach_erf]{#foreach_erf} [foreach_erf]{#foreach_erf}\_ [foreach_erfc]{#foreach_erfc} [foreach_erfc]{#foreach_erfc}\_ [foreach_exp]{#foreach_exp} [foreach_exp]{#foreach_exp}\_ [foreach_expm1]{#foreach_expm1} [foreach_expm1]{#foreach_expm1}\_ [foreach_floor]{#foreach_floor} [foreach_floor]{#foreach_floor}\_ [foreach_log]{#foreach_log} [foreach_log]{#foreach_log}\_ [foreach_log10]{#foreach_log10} [foreach_log10]{#foreach_log10}\_ [foreach_log1p]{#foreach_log1p} [foreach_log1p]{#foreach_log1p}\_ [foreach_log2]{#foreach_log2} [foreach_log2]{#foreach_log2}\_ [foreach_neg]{#foreach_neg} [foreach_neg]{#foreach_neg}\_ [foreach_tan]{#foreach_tan} [foreach_tan]{#foreach_tan}\_ [foreach_sin]{#foreach_sin} [foreach_sin]{#foreach_sin}\_ [foreach_sinh]{#foreach_sinh} [foreach_sinh]{#foreach_sinh}\_ [foreach_round]{#foreach_round} [foreach_round]{#foreach_round}\_ [foreach_sqrt]{#foreach_sqrt} [foreach_sqrt]{#foreach_sqrt}\_ [foreach_lgamma]{#foreach_lgamma} [foreach_lgamma]{#foreach_lgamma}\_ [foreach_frac]{#foreach_frac} [foreach_frac]{#foreach_frac}\_ [foreach_reciprocal]{#foreach_reciprocal} [foreach_reciprocal]{#foreach_reciprocal}\_ [foreach_sigmoid]{#foreach_sigmoid} [foreach_sigmoid]{#foreach_sigmoid}\_ [foreach_trunc]{#foreach_trunc} [foreach_trunc]{#foreach_trunc}\_ [foreach_zero]{#foreach_zero}\_


## 工具函数

 {.autosummary toctree="generated" nosignatures=""}
compiled_with_cxx11_abi result_type can_cast promote_types use_deterministic_algorithms are_deterministic_algorithms_enabled is_deterministic_algorithms_warn_only_enabled set_deterministic_debug_mode get_deterministic_debug_mode set_float32_matmul_precision get_float32_matmul_precision set_warn_always get_device_module is_warn_always_enabled vmap [assert]{#assert} typename


## 符号数字

 {.autoclass members=""}
SymInt


 {.autoclass members=""}
SymFloat


 {.autoclass members=""}
SymBool


 {.autosummary toctree="generated" nosignatures=""}
sym_float sym_fresh_size sym_int sym_max sym_min sym_not sym_ite sym_sqrt sym_sum


## 导出路径

 {.autosummary toctree="generated" nosignatures=""}


 warning
 title
Warning


此功能为原型，未来可能会有破坏兼容性的更改。

export generated/exportdb/index


## 控制流

 warning
 title
Warning


此功能为原型，未来可能会有破坏兼容性的更改。


 {.autosummary toctree="generated" nosignatures=""}
cond


## 优化

 {.autosummary toctree="generated" nosignatures=""}
compile


[torch.compile 文档](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html)

## 操作符标签

 {.autoclass members=""}
Tag


 {.toctree hidden=""}
torch.aliases.md

