# 数值精度

在现代计算机中，浮点数使用 IEEE 754 标准表示。 有关浮点运算和 IEEE 754 标准的更多详细信息，请参阅 [浮点运算](https://en.wikipedia.org/wiki/Floating-point_arithmetic)。 特别需要注意的是，浮点数提供的精度有限（单精度浮点数约为 7 位十进制数字， 双精度浮点数约为 16 位十进制数字），并且浮点加法和乘法不满足结合律， 因此运算顺序会影响结果。 正因为如此，PyTorch 不能保证对于数学上相同的浮点计算产生逐位相同的结果。 同样，也不能保证跨 PyTorch 版本、单个提交或不同平台产生逐位相同的结果。 特别是，即使对于逐位相同的输入，并且在控制了随机性来源之后，CPU 和 GPU 的结果也可能不同。

## 批处理计算或切片计算

PyTorch 中的许多操作支持批处理计算，即对输入批次中的元素执行相同的操作。 例如 `torch.mm` 和 `torch.bmm`。 可以将批处理计算实现为对批次元素的循环，并将必要的数学运算应用于各个批次元素， 但出于效率原因，我们通常不这样做，而是对整个批次执行计算。 在这种情况下，我们调用的数学库以及 PyTorch 操作的内部实现可能会产生与非批处理计算略有不同的结果。 具体来说，假设 `A` 和 `B` 是维度适合批处理矩阵乘法的 3D 张量。 那么 `(A@B)[0]`（批处理结果的第一个元素）不能保证与 `A[0]@B[0]`（输入批次第一个元素的矩阵乘积）逐位相同， 尽管在数学上它们是相同的计算。

类似地，应用于张量切片的操作不能保证产生与应用于完整张量的相同操作结果的切片相同的结果。 例如，假设 `A` 是一个二维张量。`A.sum(-1)[0]` 不能保证与 `A[:,0].sum()` 逐位相等。

## 极值

当输入包含极大值，使得中间结果可能超出所用数据类型的范围时，最终结果也可能溢出， 即使它在原始数据类型中是可表示的。例如：

``` python
import torch
a=torch.tensor([1e20, 1e20]) # 默认 fp32 类型
a.norm() # 产生 tensor(inf)
a.double().norm() # 产生 tensor(1.4142e+20, dtype=torch.float64)，在 fp32 中可表示
```

## 线性代数 (`torch.linalg`)

### 非有限值

`torch.linalg` 使用的外部库（后端）不保证在输入包含非有限值（如 `inf` 或 `NaN`）时的行为。 因此，PyTorch 也不保证。这些操作可能返回包含非有限值的张量，或引发异常，甚至导致段错误。

建议在调用这些函数之前使用 `torch.isfinite` 来检测这种情况。

### 线性代数中的极值

`torch.linalg` 中的函数比其他 PyTorch 函数有更多的 [极值](#极值) 问题。

`linalg solvers` 和 `linalg inverses` 假设输入矩阵 `A` 是可逆的。如果它接近不可逆（例如，如果它有非常小的奇异值）， 那么这些算法可能会静默地返回不正确的结果。这些矩阵被称为 [病态矩阵](https://nhigham.com/2020/03/19/what-is-a-condition-number/)。 如果提供病态输入，这些函数的结果在使用相同输入但在不同设备上时，或通过关键字 `driver` 使用不同后端时，可能会有所不同。

谱运算如 `svd`、`eig` 和 `eigh` 在其输入的奇异值彼此接近时，也可能返回不正确的结果（并且它们的梯度可能是无限的）。 这是因为用于计算这些分解的算法对于这些输入难以收敛。

在 `float64` 中运行计算（如 NumPy 默认所做的那样）通常有帮助，但并不能在所有情况下解决这些问题。 通过 `torch.linalg.svdvals` 分析输入的谱，或通过 `torch.linalg.cond` 分析其条件数，可能有助于检测这些问题。

## Nvidia Ampere（及更高版本）设备上的 TensorFloat-32(TF32)

在 Ampere（及后续）Nvidia GPU 上，PyTorch 可以使用 TensorFloat32 (TF32) 来加速数学密集型运算，特别是矩阵乘法和卷积运算。 当使用 TF32 张量核心执行运算时，仅读取输入尾数的前 10 位。 这可能会降低精度并产生意外结果（例如，将矩阵与单位矩阵相乘可能产生与输入不同的结果）。 默认情况下，TF32 张量核心在矩阵乘法中禁用，在卷积中启用，尽管大多数神经网络工作负载在使用 TF32 时与使用 fp32 时具有相同的收敛行为。 如果您的网络不需要完整的 float32 精度，我们建议通过 `torch.backends.cuda.matmul.fp32_precision = "tf32"`（`` `torch.backends.cuda.matmul.allow_tf32 = True `` 将被弃用）为矩阵乘法启用 TF32 张量核心。 如果您的网络在矩阵乘法和卷积运算中都需要完整的 float32 精度，则也可以通过 `torch.backends.cudnn.conv.fp32_precision = "ieee"`（`torch.backends.cudnn.allow_tf32 = False` 将被弃用）为卷积运算禁用 TF32 张量核心。

更多信息请参阅 `TensorFloat32<tf32_on_ampere>`。

## FP16 和 BF16 GEMM 的降低精度归约

半精度 GEMM 运算通常以单精度进行中间累加（归约），以提高数值精度和增强对溢出的鲁棒性。出于性能考虑，某些 GPU 架构（尤其是较新的架构）允许将中间累加结果截断为较低的精度（例如半精度）。从模型收敛的角度来看，这种变化通常是良性的，但可能导致意外结果（例如，当最终结果本应可用半精度表示时出现 `inf` 值）。 如果降低精度归约存在问题，可以通过以下方式关闭： `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False`

对于 BF16 GEMM 运算也存在类似的标志，且默认开启。如果 BF16 降低精度归约存在问题，可以通过以下方式关闭： `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`

更多信息请参阅 `allow_fp16_reduced_precision_reduction<fp16reducedprecision>` 和 `allow_bf16_reduced_precision_reduction<bf16reducedprecision>`

## 缩放点积注意力 (SDPA) 中 FP16 和 BF16 的降低精度归约

当使用 FP16/BF16 输入时，简单的 SDPA 数学后端可能会因使用低精度中间缓冲区而累积显著的数值误差。为了缓解此问题，默认行为现在涉及将 FP16/BF16 输入上转为 FP32。计算在 FP32/TF32 中执行，然后将最终的 FP32 结果下转回 FP16/BF16。这将提高使用 FP16/BF16 输入的数学后端最终输出的数值精度，但会增加内存使用量，并可能因计算从 FP16/BF16 BMM 转移到 FP32/TF32 BMM/矩阵乘法而导致数学后端性能下降。

对于更倾向于使用降低精度归约以提升速度的场景，可以通过以下设置启用： `torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)`

## AMD Instinct MI200 设备上降低精度的 FP16 和 BF16 GEMM 与卷积运算

在 AMD Instinct MI200 GPU 上，FP16 和 BF16 的 V_DOT2 和 MFMA 矩阵指令会将输入和输出的非规格化值刷新为零。FP32 和 FP64 的 MFMA 矩阵指令不会将输入和输出的非规格化值刷新为零。受影响的指令仅由 rocBLAS (GEMM) 和 MIOpen (卷积) 内核使用；所有其他 PyTorch 操作不会遇到此行为。所有其他受支持的 AMD GPU 也不会遇到此行为。

rocBLAS 和 MIOpen 为受影响的 FP16 运算提供了替代实现。未提供 BF16 运算的替代实现；BF16 数字比 FP16 数字具有更大的动态范围，因此不太可能遇到非规格化值。对于 FP16 替代实现，FP16 输入值被转换为中间 BF16 值，然后在累加 FP32 运算后转换回 FP16 输出。这样，输入和输出类型保持不变。

当使用 FP16 精度进行训练时，一些模型可能因 FP16 非规格化值被刷新为零而无法收敛。非规格化值更频繁地出现在训练的反向传播过程中进行梯度计算时。PyTorch 默认会在反向传播过程中使用 rocBLAS 和 MIOpen 的替代实现。可以使用环境变量 ROCBLAS_INTERNAL_FP16_ALT_IMPL 和 MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL 覆盖默认行为。这些环境变量的行为如下：

+---------------+-------------+-------------+-------------+
|               | 前向传播    | 反向传播    |             |
+===============+=============+=============+=============+
| 环境变量未设置 \| 原始实现 \| 替代实现                  |
+---------------------------------------------------------+
| 环境变量设为 1 \| 替代实现 \| 替代实现                  |
+---------------------------------------------------------+
| 环境变量设为 0 \| 原始实现 \| 原始实现                  |
+---------------------------------------------------------+

以下是可能使用 rocBLAS 的操作列表：

- torch.addbmm
- torch.addmm
- torch.baddbmm
- torch.bmm
- torch.mm
- torch.nn.GRUCell
- torch.nn.LSTMCell
- torch.nn.Linear
- torch.sparse.addmm
- 以下 torch.\_C.\_ConvBackend 实现：
  - slowNd
  - slowNd_transposed
  - slowNd_dilated
  - slowNd_dilated_transposed

以下是可能使用 MIOpen 的操作列表：

- torch.nn.Conv\[Transpose\]Nd
- 以下 torch.\_C.\_ConvBackend 实现：
  - ConvBackend::Miopen
  - ConvBackend::MiopenDepthwise
  - ConvBackend::MiopenTranspose
