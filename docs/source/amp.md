

# 自动混合精度包 - torch.amp

% 以下两个模块目前缺少文档条目。暂时在此处添加它们。

% 这不会为渲染页面添加任何内容


`torch.amp` 为混合精度提供了便捷方法，其中一些操作使用 `torch.float32` (`float`) 数据类型，而其他操作使用较低精度的浮点数据类型 (`lower_precision_fp`)：`torch.float16` (`half`) 或 `torch.bfloat16`。某些操作，如线性层和卷积，在 `lower_precision_fp` 中要快得多。其他操作，如归约，通常需要 `float32` 的动态范围。混合精度尝试将每个操作与其适当的数据类型匹配。

通常，使用 `torch.float16` 数据类型的"自动混合精度训练"会同时使用 `torch.autocast` 和 `torch.amp.GradScaler`，如 `自动混合精度示例<amp-examples>` 和 [自动混合精度教程](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) 所示。然而，`torch.autocast` 和 `torch.GradScaler` 是模块化的，如果需要可以单独使用。如 `torch.autocast` 的 CPU 示例部分所示，在 CPU 上使用 `torch.bfloat16` 数据类型的"自动混合精度训练/推理"仅使用 `torch.autocast`。


> ⚠️ **警告**
> `torch.cuda.amp.autocast(args...)` 和 `torch.cpu.amp.autocast(args...)` 已弃用。请改用 `torch.amp.autocast("cuda", args...)` 或 `torch.amp.autocast("cpu", args...)`。
> `torch.cuda.amp.GradScaler(args...)` 和 `torch.cpu.amp.GradScaler(args...)` 已弃用。请改用 `torch.amp.GradScaler("cuda", args...)` 或 `torch.amp.GradScaler("cpu", args...)`。


`torch.autocast` 和 `torch.cpu.amp.autocast` 是 `1.10` 版本中的新功能。

```{contents}
:local: true
```


## 自动转换


## 梯度缩放

如果特定操作的前向传递具有 `float16` 输入，则该操作的反向传递将产生 `float16` 梯度。幅度较小的梯度值可能无法用 `float16` 表示。这些值将下溢为零（"下溢"），因此相应参数的更新将丢失。

为了防止下溢，"梯度缩放"将网络的损失乘以一个缩放因子，并在缩放后的损失上调用反向传递。通过网络反向传播的梯度随后被相同的因子缩放。换句话说，梯度值具有更大的幅度，因此它们不会下溢为零。

在优化器更新参数之前，每个参数的梯度（`.grad` 属性）应取消缩放，这样缩放因子就不会干扰学习率。


> 📝 **注意**
> AMP/fp16 可能不适用于每个模型！例如，大多数 bf16 预训练模型无法在最大值为 65504 的 fp16 数值范围内运行，并且会导致梯度溢出而不是下溢。在这种情况下，缩放因子可能会降低到 1 以下，以尝试将梯度带到 fp16 动态范围内可表示的数字。虽然人们可能期望缩放因子始终高于 1，但我们的 GradScaler 并不保证这一点以保持性能。如果在使用 AMP/fp16 运行时遇到损失或梯度中的 NaN，请验证您的模型是否兼容。


## 自动转换操作参考


### 操作资格

在 `float64` 或非浮点数据类型中运行的操作不符合条件，无论是否启用自动转换，它们都将以这些类型运行。

只有原地操作和 Tensor 方法符合条件。在启用自动转换的区域中允许使用原地变体和显式提供 `out=...` Tensor 的调用，但它们不会经过自动转换。例如，在启用自动转换的区域中，`a.addmm(b, c)` 可以自动转换，但 `a.addmm_(b, c)` 和 `a.addmm(b, c, out=d)` 不能。为了获得最佳性能和稳定性，在启用自动转换的区域中优先使用原地操作。

使用显式 `dtype=...` 参数调用的操作不符合条件，并且将产生尊重 `dtype` 参数的输出。


### CUDA 操作特定行为

以下列表描述了在启用自动转换的区域中符合条件操作的行为。无论这些操作是作为 `torch.nn.Module` 的一部分、作为函数还是作为 `torch.Tensor` 方法调用，它们总是经过自动转换。如果函数在多个命名空间中公开，则无论命名空间如何，它们都会经过自动转换。

下面未列出的操作不会经过自动转换。它们按照其输入定义的类型运行。但是，如果未列出的操作位于自动转换操作的下游，自动转换仍可能更改其运行的类型。

如果某个操作未列出，我们假设它在 `float16` 中是数值稳定的。如果您认为某个未列出的操作在 `float16` 中数值不稳定，请提交问题。

#### 可以自动转换为 `float16` 的 CUDA 操作

`__matmul__`,
`addbmm`,
`addmm`,
`addmv`,
`addr`,
`baddbmm`,
`bmm`,
`chain_matmul`,
`multi_dot`,
`conv1d`,
`conv2d`,
`conv3d`,
`conv_transpose1d`,
`conv_transpose2d`,
`conv_transpose3d`,
`GRUCell`,
`linear`,
`LSTMCell`,
`matmul`,
`mm`,
`mv`,
`prelu`,
`RNNCell`

#### 可以自动转换为 `float32` 的 CUDA 操作

`__pow__`,
`__rdiv__`,
`__rpow__`,
`__rtruediv__`,
`acos`,
`asin`,
`binary_cross_entropy_with_logits`,
`cosh`,
`cosine_embedding_loss`,
`cdist`,
`cosine_similarity`,
`cross_entropy`,
`cumprod`,
`cumsum`,
`dist`,
`erfinv`,
`exp`,
`expm1`,
`group_norm`,
`hinge_embedding_loss`,
`kl_div`,
`l1_loss`,
`layer_norm`,
`log`,
`log_softmax`,
`log10`,
`log1p`,
`log2`,
`margin_ranking_loss`,
`mse_loss`,
`multilabel_margin_loss`,
`multi_margin_loss`,
`nll_loss`,
`norm`,
`normalize`,
`pdist`,
`poisson_nll_loss`,
`pow`,
`prod`,
`reciprocal`,
`rsqrt`,
`sinh`,
`smooth_l1_loss`,
`soft_margin_loss`,
`softmax`,
`softmin`,
`softplus`,
`sum`,
`renorm`,
`tan`,
`triplet_margin_loss`

#### 提升到最宽输入类型的 CUDA 操作

这些操作不需要特定的数据类型来保证稳定性，但接受多个输入并要求输入的数据类型匹配。如果所有输入都是 `float16`，操作将以 `float16` 运行。如果任何输入是 `float32`，自动转换会将所有输入转换为 `float32` 并以 `float32` 运行该操作。

`addcdiv`,
`addcmul`,
`atan2`,
`bilinear`,
`cross`,
`dot`,
`grid_sample`,
`index_put`,
`scatter_add`,
`tensordot`

此处未列出的一些操作（例如，像 `add` 这样的二元操作）本身会提升输入类型，无需自动转换干预。如果输入是 `float16` 和 `float32` 的混合，无论是否启用自动转换，这些操作都会以 `float32` 运行并产生 `float32` 输出。

#### 优先使用 `binary_cross_entropy_with_logits` 而非 `binary_cross_entropy`

`torch.nn.functional.binary_cross_entropy`（以及包装它的 `torch.nn.BCELoss`）的反向传播可能会产生无法用 `float16` 表示的梯度。在启用自动转换的区域中，前向输入可能是 `float16`，这意味着反向梯度必须能用 `float16` 表示（将前向输入的 `float16` 自动转换为 `float32` 没有帮助，因为该转换必须在反向传播中反转）。因此，`binary_cross_entropy` 和 `BCELoss` 在启用自动转换的区域中会引发错误。

许多模型在二元交叉熵层之前使用 sigmoid 层。在这种情况下，使用 `torch.nn.functional.binary_cross_entropy_with_logits` 或 `torch.nn.BCEWithLogitsLoss` 将这两个层组合起来。`binary_cross_entropy_with_logits` 和 `BCEWithLogits` 可以安全地进行自动转换。


### XPU 操作特定行为（实验性）

以下列表描述了在启用自动转换的区域中符合条件的操作的行为。无论这些操作是作为 `torch.nn.Module` 的一部分、作为函数还是作为 `torch.Tensor` 方法调用，它们总是会经过自动转换。如果函数在多个命名空间中公开，则无论哪个命名空间，它们都会经过自动转换。

下面未列出的操作不会经过自动转换。它们按照其输入定义的类型运行。但是，如果未列出的操作位于自动转换操作的下游，自动转换仍可能改变其运行的类型。

如果一个操作未列出，我们假定它在 `float16` 中是数值稳定的。如果您认为某个未列出的操作在 `float16` 中数值不稳定，请提交问题。

#### 可以自动转换为 `float16` 的 XPU 操作

`addbmm`,
`addmm`,
`addmv`,
`addr`,
`baddbmm`,
`bmm`,
`chain_matmul`,
`multi_dot`,
`conv1d`,
`conv2d`,
`conv3d`,
`conv_transpose1d`,
`conv_transpose2d`,
`conv_transpose3d`,
`GRUCell`,
`linear`,
`LSTMCell`,
`matmul`,
`mm`,
`mv`,
`RNNCell`

#### 可以自动转换为 `float32` 的 XPU 操作

`__pow__`,
`__rdiv__`,
`__rpow__`,
`__rtruediv__`,
`binary_cross_entropy_with_logits`,
`cosine_embedding_loss`,
`cosine_similarity`,
`cumsum`,
`dist`,
`exp`,
`group_norm`,
`hinge_embedding_loss`,
`kl_div`,
`l1_loss`,
`layer_norm`,
`log`,
`log_softmax`,
`margin_ranking_loss`,
`nll_loss`,
`normalize`,
`poisson_nll_loss`,
`pow`,
`reciprocal`,
`rsqrt`,
`soft_margin_loss`,
`softmax`,
`softmin`,
`sum`,
`triplet_margin_loss`

#### 提升到最宽输入类型的 XPU 操作

这些操作不需要特定的数据类型来保证稳定性，但接受多个输入并要求输入的数据类型匹配。如果所有输入都是 `float16`，操作将以 `float16` 运行。如果任何输入是 `float32`，自动转换会将所有输入转换为 `float32` 并以 `float32` 运行该操作。

`bilinear`,
`cross`,
`grid_sample`,
`index_put`,
`scatter_add`,
`tensordot`

此处未列出的一些操作（例如，像 `add` 这样的二元操作）本身会提升输入类型，无需自动转换干预。如果输入是 `float16` 和 `float32` 的混合，无论是否启用自动转换，这些操作都会以 `float32` 运行并产生 `float32` 输出。


### CPU 操作特定行为

以下列表描述了在启用自动转换的区域中符合条件的操作的行为。无论这些操作是作为 `torch.nn.Module` 的一部分、作为函数还是作为 `torch.Tensor` 方法调用，它们总是会经过自动转换。如果函数在多个命名空间中公开，则无论哪个命名空间，它们都会经过自动转换。

下面未列出的操作不会经过自动转换。它们按照其输入定义的类型运行。但是，如果未列出的操作位于自动转换操作的下游，自动转换仍可能改变其运行的类型。

如果一个操作未列出，我们假定它在 `bfloat16` 中是数值稳定的。如果您认为某个未列出的操作在 `bfloat16` 中数值不稳定，请提交问题。`float16` 共享 `bfloat16` 的列表。

#### 可以自动转换为 `bfloat16` 的 CPU 操作

`conv1d`,
`conv2d`,
`conv3d`,
`bmm`,
`mm`,
`linalg_vecdot`,
`baddbmm`,
`addmm`,
`addbmm`,
`linear`,
`matmul`,
`_convolution`,
`conv_tbc`,
`mkldnn_rnn_layer`,
`conv_transpose1d`,
`conv_transpose2d`,
`conv_transpose3d`,
`prelu`,
`scaled_dot_product_attention`,
`_native_multi_head_attention`

#### 可以自动转换为 `float32` 的 CPU 操作

`avg_pool3d`,
`binary_cross_entropy`,
`grid_sampler`,
`grid_sampler_2d`,
`_grid_sampler_2d_cpu_fallback`,
`grid_sampler_3d`,
`polar`,
`prod`,
`quantile`,
`nanquantile`,
`stft`,
`cdist`,
`trace`,
`view_as_complex`,
`cholesky`,
`cholesky_inverse`,
`cholesky_solve`,
`inverse`,
`lu_solve`,
`orgqr`,
`inverse`,
`ormqr`,
`pinverse`,
`max_pool3d`,
`max_unpool2d`,
`max_unpool3d`,
`adaptive_avg_pool3d`,
`reflection_pad1d`,
`reflection_pad2d`,
`replication_pad1d`,
`replication_pad2d`,
`replication_pad3d`,
`mse_loss`,
`cosine_embedding_loss`,
`nll_loss`,
`nll_loss2d`,
`hinge_embedding_loss`,
`poisson_nll_loss`,
`cross_entropy_loss`,
`l1_loss`,
`huber_loss`,
`margin_ranking_loss`,
`soft_margin_loss`,
`triplet_margin_loss`,
`multi_margin_loss`,
`ctc_loss`,
`kl_div`,
`multilabel_margin_loss`,
`binary_cross_entropy_with_logits`,
`fft_fft`,
`fft_ifft`,
`fft_fft2`,
`fft_ifft2`,
`fft_fftn`,
`fft_ifftn`,
`fft_rfft`,
`fft_irfft`,
`fft_rfft2`,
`fft_irfft2`,
`fft_rfftn`,
`fft_irfftn`,
`fft_hfft`,
`fft_ihfft`,
`linalg_cond`,
`linalg_matrix_rank`,
`linalg_solve`,
`linalg_cholesky`,
`linalg_svdvals`,
`linalg_eigvals`,
`linalg_eigvalsh`,
`linalg_inv`,
`linalg_householder_product`,
`linalg_tensorinv`,
`linalg_tensorsolve`,
`fake_quantize_per_tensor_affine`,
`geqrf`,
`_lu_with_info`,
`qr`,
`svd`,
`triangular_solve`,
`fractional_max_pool2d`,
`fractional_max_pool3d`,
`adaptive_max_pool3d`,
`multilabel_margin_loss_forward`,
`linalg_qr`,
`linalg_cholesky_ex`,
`linalg_svd`,
`linalg_eig`,
`linalg_eigh`,
`linalg_lstsq`,
`linalg_inv_ex`

#### 提升至最宽输入类型的 CPU 操作

这些操作不需要特定的数据类型来保证稳定性，但它们接收多个输入并要求输入的数据类型匹配。如果所有输入都是 `bfloat16`，操作将以 `bfloat16` 运行。如果任何输入是 `float32`，自动转换会将所有输入转换为 `float32` 并以 `float32` 运行该操作。

`cat`,
`stack`,
`index_copy`

此处未列出的一些操作（例如，像 `add` 这样的二元操作）本身就会提升输入类型，无需自动转换的干预。如果输入是 `bfloat16` 和 `float32` 的混合，无论是否启用自动转换，这些操作都会以 `float32` 运行并产生 `float32` 输出。

% 此模块需要文档记录。暂时添加于此
% 用于跟踪目的


