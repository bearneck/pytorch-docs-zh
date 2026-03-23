 automodule
torch.sparse


 currentmodule
torch


# torch.sparse {#sparse-docs}

 warning
 title
Warning


PyTorch 的稀疏张量 API 目前处于测试阶段，可能在近期发生变更。 我们非常欢迎通过 GitHub issue 提交功能请求、错误报告和一般性建议。


## 为何以及何时使用稀疏性

默认情况下，PyTorch 将 `torch.Tensor`{.interpreted-text role="class"} 的元素连续存储在物理内存中。这为需要快速访问元素的各种数组处理算法提供了高效的实现。

然而，一些用户可能决定使用 *元素大部分为零值* 的张量来表示数据，例如图邻接矩阵、剪枝后的权重或点云。我们认识到这些是重要的应用场景，并旨在通过稀疏存储格式为这些用例提供性能优化。

多年来，人们开发了各种稀疏存储格式，如 COO、CSR/CSC、半结构化、LIL 等。虽然它们在具体布局上有所不同，但都通过高效表示零值元素来压缩数据。我们将未压缩的值称为 *指定* 元素，以区别于 *未指定* 的压缩元素。

通过压缩重复的零，稀疏存储格式旨在节省各种 CPU 和 GPU 上的内存和计算资源。特别是对于高稀疏度或高度结构化的稀疏性，这可能会带来显著的性能影响。因此，稀疏存储格式可以被视为一种性能优化手段。

与许多其他性能优化手段一样，稀疏存储格式并不总是有利的。当为您的用例尝试稀疏格式时，您可能会发现执行时间反而增加了。

如果您在分析时期望看到性能的显著提升，但实际测量结果却是性能下降，我们非常鼓励您提交一个 GitHub issue。这有助于我们优先实现高效的内核和更广泛的性能优化。

我们使得尝试不同的稀疏布局以及它们之间的转换变得容易，而不会对哪种最适合您的特定应用做出主观判断。

## 功能概述

我们希望通过为每种布局提供转换例程，使得从给定的稠密张量构造稀疏张量变得简单直接。

在下一个示例中，我们将一个具有默认稠密（跨步）布局的 2D 张量转换为由 COO 内存布局支持的 2D 张量。在这种情况下，只存储非零元素的值和索引。

> \>\>\> a = torch.tensor(\[\[0, 2.\], \[3, 0\]\]) \>\>\> a.to_sparse() tensor(indices=tensor(\[\[0, 1\], \[1, 0\]\]), values=tensor(\[2., 3.\]), size=(2, 2), nnz=2, layout=torch.sparse_coo)

PyTorch 目前支持 `COO<sparse-coo-docs>`{.interpreted-text role="ref"}、`CSR<sparse-csr-docs>`{.interpreted-text role="ref"}、`CSC<sparse-csc-docs>`{.interpreted-text role="ref"}、`BSR<sparse-bsr-docs>`{.interpreted-text role="ref"} 和 `BSC<sparse-bsc-docs>`{.interpreted-text role="ref"}。

我们还提供了一个支持 `半结构化稀疏性<sparse-semi-structured-docs>`{.interpreted-text role="ref"} 的原型实现。更多详情请参阅参考文献。

请注意，我们提供了这些格式的轻微泛化。

批处理：GPU 等设备需要批处理以获得最佳性能，因此我们支持批处理维度。

我们目前提供了一个非常简单的批处理版本，其中稀疏格式的每个组件本身都是批处理的。这也要求每个批次条目具有相同数量的指定元素。在此示例中，我们从 3D 稠密张量构造一个 3D（批处理）CSR 张量。

> \>\>\> t = torch.tensor(\[\[\[1., 0\], \[2., 3.\]\], \[\[4., 0\], \[5., 6.\]\]\]) \>\>\> t.dim() 3 \>\>\> t.to_sparse_csr() tensor(crow_indices=tensor(\[\[0, 1, 3\], \[0, 1, 3\]\]), col_indices=tensor(\[\[0, 0, 1\], \[0, 0, 1\]\]), values=tensor(\[\[1., 2., 3.\], \[4., 5., 6.\]\]), size=(2, 2, 2), nnz=3, layout=torch.sparse_csr)

稠密维度：另一方面，某些数据（如图嵌入）可能更适合被视为向量的稀疏集合，而不是标量。

在此示例中，我们从 3D 跨步张量创建一个具有 2 个稀疏维度和 1 个稠密维度的 3D 混合 COO 张量。如果 3D 跨步张量中的整行都为零，则不会存储该行。但是，如果该行中的任何值非零，则会完整存储该行。这减少了索引的数量，因为我们需要每行一个索引，而不是每个元素一个索引。但它也增加了值的存储量。因为只有 *完全* 为零的行可以被省略，而任何非零值元素的存在都会导致整行被存储。

> \>\>\> t = torch.tensor(\[\[\[0., 0\], \[1., 2.\]\], \[\[0., 0\], \[3., 4.\]\]\]) \>\>\> t.to_sparse(sparse_dim=2) tensor(indices=tensor(\[\[0, 1\], \[1, 1\]\]), values=tensor(\[\[1., 2.\], \[3., 4.\]\]), size=(2, 2, 2), nnz=2, layout=torch.sparse_coo)

## 运算符概述

从根本上说，对具有稀疏存储格式的张量进行的操作与对具有跨步（或其他）存储格式的张量进行的操作行为相同。存储的特殊性，即数据的物理布局，会影响操作的性能，但不应该影响其语义。

我们正在积极增加对稀疏张量的运算符覆盖。用户目前不应期望获得与稠密张量相同级别的支持。有关列表，请参阅我们的 `运算符<sparse-ops-docs>`{.interpreted-text role="ref"} 文档。

> \>\>\> b = torch.tensor(\[\[0, 0, 1, 2, 3, 0\], \[4, 5, 0, 6, 0, 0\]\]) \>\>\> b_s = b.to_sparse_csr() \>\>\> b_s.cos() Traceback (most recent call last): File \"\<stdin\>\", line 1, in \<module\> RuntimeError: unsupported tensor layout: SparseCsr \>\>\> b_s.sin() tensor(crow_indices=tensor(\[0, 3, 6\]), col_indices=tensor(\[2, 3, 4, 0, 1, 3\]), values=tensor(\[ 0.8415, 0.9093, 0.1411, -0.7568, -0.9589, -0.2794\]), size=(2, 6), nnz=6, layout=torch.sparse_csr)

如上例所示，我们不支持诸如 [cos]{.title-ref} 这类非零保留的一元运算符。非零保留一元运算的输出将无法像输入那样充分利用稀疏存储格式的优势，并可能导致内存的灾难性增长。我们建议用户先显式转换为稠密张量，然后再执行运算。

> \>\>\> b_s.to_dense().cos() tensor(\[\[ 1.0000, -0.4161\], \[-0.9900, 1.0000\]\])

我们了解到，一些用户希望在 [cos]{.title-ref} 这类运算中忽略压缩的零值，而不是保留运算的精确语义。为此，我们可以参考 [torch.masked]{.title-ref} 及其 [MaskedTensor]{.title-ref}，它同样由稀疏存储格式和内核支持并提供动力。

还需注意，目前用户无法选择输出布局。例如，将稀疏张量与常规的跨步张量相加会得到一个跨步张量。一些用户可能更希望结果保持稀疏布局，因为他们知道结果仍然足够稀疏。

> \>\>\> a + b.to_sparse() tensor(\[\[0., 3.\], \[3., 0.\]\])

我们认识到，能够高效生成不同输出布局的内核访问可能非常有用。后续操作可能会因接收到特定布局而显著受益。我们正在开发一个用于控制结果布局的 API，并认识到这是一个重要的功能，可以为任何给定模型规划更优的执行路径。

## 稀疏半结构化张量 {#sparse-semi-structured-docs}

 warning
 title
Warning


稀疏半结构化张量目前是一个原型功能，可能会发生变化。如果您发现错误或有反馈意见，请随时提交问题。


半结构化稀疏是一种稀疏数据布局，最初由 NVIDIA 的 Ampere 架构引入。它也被称为\*\*细粒度结构化稀疏\*\*或\*\*2:4 结构化稀疏\*\*。

这种稀疏布局存储每 [2n]{.title-ref} 个元素中的 [n]{.title-ref} 个元素，其中 [n]{.title-ref} 由张量数据类型（dtype）的宽度决定。最常用的 dtype 是 float16，其中 [n=2]{.title-ref}，因此称为"2:4 结构化稀疏"。

半结构化稀疏在 [这篇 NVIDIA 博客文章](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt) 中有更详细的解释。

在 PyTorch 中，半结构化稀疏通过张量子类实现。通过子类化，我们可以重写 `__torch_dispatch__`，从而在执行矩阵乘法时使用更快的稀疏内核。我们还可以在子类中以压缩形式存储张量，以减少内存开销。

在这种压缩形式中，稀疏张量仅存储\*指定的\*元素和一些编码掩码的元数据。

 note
 title
Note


半结构化稀疏张量的指定元素和元数据掩码一起存储在一个单一的扁平压缩张量中。它们彼此附加以形成连续的内存块。

压缩张量 = \[ 原始张量的指定元素 \| 元数据掩码 \]

对于大小为 [(r, c)]{.title-ref} 的原始张量，我们期望前 [m \* k // 2]{.title-ref} 个元素是保留的元素，张量的其余部分是元数据。

为了使用户更容易查看指定元素和掩码，可以使用 `.indices()` 和 `.values()` 分别访问掩码和指定元素。

- `.values()` 返回指定元素，存储在一个大小为 [(r, c//2)]{.title-ref} 的张量中，且与稠密矩阵具有相同的 dtype。
- `.indices()` 返回元数据掩码，存储在一个大小为 [(r, c//2)]{.title-ref} 的张量中。如果 dtype 是 torch.float16 或 torch.bfloat16，则元素类型为 `torch.int16`；如果 dtype 是 torch.int8，则元素类型为 `torch.int32`。


对于 2:4 稀疏张量，元数据开销很小------每个指定元素仅需 2 比特。

 note
 title
Note


需要注意的是，`torch.float32` 仅支持 1:2 稀疏度。因此，它不遵循上述相同的公式。


这里，我们分解如何计算 2:4 稀疏张量的压缩比（稠密大小 / 稀疏大小）。

设 [(r, c) = tensor.shape]{.title-ref} 且 [e = bitwidth(tensor.dtype)]{.title-ref}，因此对于 `torch.float16` 和 `torch.bfloat16`，\`e = 16\`；对于 `torch.int8`，\`e = 8\`。

$$\begin{aligned}
M_{dense} = r \times c \times e \\
M_{sparse} = M_{specified} + M_{metadata} = r \times \frac{c}{2} \times e + r \times \frac{c}{2} \times 2 = \frac{rce}{2} + rc =rce(\frac{1}{2} +\frac{1}{e})
\end{aligned}$$

使用这些计算，我们可以确定原始稠密表示和新稀疏表示的总内存占用。

这为我们提供了一个简单的压缩比公式，该公式仅依赖于张量数据类型的比特宽度。

$$C = \frac{M_{sparse}}{M_{dense}} =  \frac{1}{2} + \frac{1}{e}$$

使用此公式，我们发现对于 `torch.float16` 或 `torch.bfloat16`，压缩比为 56.25%；对于 `torch.int8`，压缩比为 62.5%。

### 构建稀疏半结构化张量

您可以通过简单地使用 `torch.to_sparse_semi_structured` 函数将稠密张量转换为稀疏半结构化张量。

还需注意，我们仅支持 CUDA 张量，因为半结构化稀疏的硬件兼容性仅限于 NVIDIA GPU。

以下是半结构化稀疏支持的数据类型。请注意，每种数据类型都有其自身的形状约束和压缩因子。

  ----------------------------------------------------------------------------------------
  PyTorch dtype      形状约束                                        压缩因子   稀疏模式
  ------------------ ----------------------------------------------- ---------- ----------
  `torch.float16`    张量必须是 2D 且 (r, c) 都必须是 64 的正倍数    9/16       2:4

  `torch.bfloat16`   张量必须是 2D 且 (r, c) 都必须是 64 的正倍数    9/16       2:4

  `torch.int8`       张量必须是 2D 且 (r, c) 都必须是 128 的正倍数   10/16      2:4
  ----------------------------------------------------------------------------------------

要构建一个半结构化稀疏张量，首先需要创建一个符合 2:4（或半结构化）稀疏格式的常规稠密张量。 为此，我们平铺一个小的 1x4 条带以创建一个 16x16 的 float16 稠密张量。 之后，我们可以调用 `to_sparse_semi_structured` 函数来压缩它以加速推理。

> \>\>\> from torch.sparse import to_sparse_semi_structured \>\>\> A = torch.Tensor(\[0, 0, 1, 1\]).tile((128, 32)).half().cuda() tensor(\[\[0., 0., 1., \..., 0., 1., 1.\], \[0., 0., 1., \..., 0., 1., 1.\], \[0., 0., 1., \..., 0., 1., 1.\], \..., \[0., 0., 1., \..., 0., 1., 1.\], \[0., 0., 1., \..., 0., 1., 1.\], \[0., 0., 1., \..., 0., 1., 1.\]\], device=\'cuda:0\', dtype=torch.float16) \>\>\> A_sparse = to_sparse_semi_structured(A) SparseSemiStructuredTensor(shape=torch.Size(\[128, 128\]), transposed=False, values=tensor(\[\[1., 1., 1., \..., 1., 1., 1.\], \[1., 1., 1., \..., 1., 1., 1.\], \[1., 1., 1., \..., 1., 1., 1.\], \..., \[1., 1., 1., \..., 1., 1., 1.\], \[1., 1., 1., \..., 1., 1., 1.\], \[1., 1., 1., \..., 1., 1., 1.\]\], device=\'cuda:0\', dtype=torch.float16), metadata=tensor(\[\[-4370, -4370, -4370, \..., -4370, -4370, -4370\], \[-4370, -4370, -4370, \..., -4370, -4370, -4370\], \[-4370, -4370, -4370, \..., -4370, -4370, -4370\], \..., \[-4370, -4370, -4370, \..., -4370, -4370, -4370\], \[-4370, -4370, -4370, \..., -4370, -4370, -4370\], \[-4370, -4370, -4370, \..., -4370, -4370, -4370\]\], device=\'cuda:0\', dtype=torch.int16))

### 稀疏半结构化张量操作

目前，半结构化稀疏张量支持以下操作：

- torch.addmm(bias, dense, sparse.t())
- torch.mm(dense, sparse)
- torch.mm(sparse, dense)
- aten.linear.default(dense, sparse, bias)
- aten.t.default(sparse)
- aten.t.detach(sparse)

要使用这些操作，只需在张量具有半结构化稀疏格式的零元素后，传递 `to_sparse_semi_structured(tensor)` 的输出，而不是直接使用 `tensor`，如下所示：

> \>\>\> a = torch.Tensor(\[0, 0, 1, 1\]).tile((64, 16)).half().cuda() \>\>\> b = torch.rand(64, 64).half().cuda() \>\>\> c = torch.mm(a, b) \>\>\> a_sparse = to_sparse_semi_structured(a) \>\>\> torch.allclose(c, torch.mm(a_sparse, b)) True

### 使用半结构化稀疏加速 nn.Linear

如果权重已经是半结构化稀疏的，只需几行代码就可以加速模型中的线性层：

> \>\>\> input = torch.rand(64, 64).half().cuda() \>\>\> mask = torch.Tensor(\[0, 0, 1, 1\]).tile((64, 16)).cuda().bool() \>\>\> linear = nn.Linear(64, 64).half().cuda() \>\>\> linear.weight = nn.Parameter(to_sparse_semi_structured(linear.weight.masked_fill(\~mask, 0)))

## 稀疏 COO 张量 {#sparse-coo-docs}

PyTorch 实现了所谓的坐标格式（Coordinate format），或 COO 格式，作为实现稀疏张量的存储格式之一。在 COO 格式中，指定的元素以元素索引元组和对应值的形式存储。具体来说，

> - 指定元素的索引收集在大小为 `(ndim, nse)`、元素类型为 `torch.int64` 的 `indices` 张量中，
> - 对应的值收集在大小为 `(nse,)`、元素类型为任意整数或浮点数的 `values` 张量中，

其中 `ndim` 是张量的维度，`nse` 是指定元素的数量。

 note
 title
Note


稀疏 COO 张量的内存消耗至少为 `(ndim * 8 + <元素类型字节大小>) * nse` 字节（加上存储其他张量数据的常量开销）。

跨步张量的内存消耗至少为 `product(<张量形状>) * <元素类型字节大小>`。

例如，一个包含 100,000 个非零 32 位浮点数的 10,000 x 10,000 张量，在使用 COO 张量布局时，内存消耗至少为 `(2 * 8 + 4) * 100 000 = 2 000 000` 字节；而在使用默认的跨步张量布局时，内存消耗至少为 `10 000 * 10 000 * 4 = 400 000 000` 字节。请注意，使用 COO 存储格式可以节省 200 倍的内存。


### 构造

可以通过向函数 `torch.sparse_coo_tensor`{.interpreted-text role="func"} 提供索引和值两个张量，以及稀疏张量的大小（当无法从索引和值张量推断时）来构造稀疏 COO 张量。

假设我们想定义一个稀疏张量，在位置 (0, 2) 处有元素 3，在位置 (1, 0) 处有元素 4，在位置 (1, 2) 处有元素 5。未指定的元素假定具有相同的值，即填充值，默认为零。那么我们可以这样写：

> \>\>\> i = \[\[0, 1, 1\], \[2, 0, 2\]\] \>\>\> v = \[3, 4, 5\] \>\>\> s = torch.sparse_coo_tensor(i, v, (2, 3)) \>\>\> s tensor(indices=tensor(\[\[0, 1, 1\], \[2, 0, 2\]\]), values=tensor(\[3, 4, 5\]), size=(2, 3), nnz=3, layout=torch.sparse_coo) \>\>\> s.to_dense() tensor(\[\[0, 0, 3\], \[4, 0, 5\]\])

请注意，输入 `i` 不是索引元组的列表。如果你想以这种方式编写索引，应该在将它们传递给稀疏构造函数之前进行转置：

> \>\>\> i = \[\[0, 2\], \[1, 0\], \[1, 2\]\] \>\>\> v = \[3, 4, 5 \] \>\>\> s = torch.sparse_coo_tensor(list(zip(\*i)), v, (2, 3)) \>\>\> \# 或者另一种等效的公式来得到 s \>\>\> s = torch.sparse_coo_tensor(torch.tensor(i).t(), v, (2, 3)) \>\>\> torch.sparse_coo_tensor(i.t(), v, torch.Size(\[2,3\])).to_dense() tensor(\[\[0, 0, 3\], \[4, 0, 5\]\])

可以通过仅指定其大小来构造一个空的稀疏 COO 张量：

\>\>\> torch.sparse_coo_tensor(size=(2, 3))

:   

    tensor(indices=tensor(\[\], size=(2, 0)),

    :   values=tensor(\[\], size=(0,)), size=(2, 3), nnz=0, layout=torch.sparse_coo)

### 稀疏混合 COO 张量 {#sparse-hybrid-coo-docs}

PyTorch 实现了对具有标量值的稀疏张量的扩展，使其支持具有（连续）张量值的稀疏张量。这类张量被称为混合张量。

PyTorch 混合 COO 张量扩展了稀疏 COO 张量，允许 `values` 张量是一个多维张量，因此我们有：

> - 指定元素的索引收集在大小为 `(sparse_dims, nse)`、元素类型为 `torch.int64` 的 `indices` 张量中，
> - 对应的（张量）值收集在大小为 `(nse, dense_dims)`、元素类型为任意整数或浮点数的 `values` 张量中。

 note
 title
Note


我们使用 (M + K) 维张量来表示一个 N 维稀疏混合张量，其中 M 和 K 分别是稀疏维度和稠密维度的数量，满足 M + K == N。


假设我们想创建一个 (2 + 1) 维张量，其中位置 (0, 2) 处的条目为 \[3, 4\]，位置 (1, 0) 处的条目为 \[5, 6\]，位置 (1, 2) 处的条目为 \[7, 8\]。我们可以这样写：

> 
>
> \>\>\> i = \[\[0, 1, 1\],
>
> :   \[2, 0, 2\]\]
>
> \>\>\> v = \[\[3, 4\], \[5, 6\], \[7, 8\]\] \>\>\> s = torch.sparse_coo_tensor(i, v, (2, 3, 2)) \>\>\> s tensor(indices=tensor(\[\[0, 1, 1\], \[2, 0, 2\]\]), values=tensor(\[\[3, 4\], \[5, 6\], \[7, 8\]\]), size=(2, 3, 2), nnz=3, layout=torch.sparse_coo)
>
> \>\>\> s.to_dense() tensor(\[\[\[0, 0\], \[0, 0\], \[3, 4\]\], \[\[5, 6\], \[0, 0\], \[7, 8\]\]\])

一般来说，如果 `s` 是一个稀疏 COO 张量，且 `M = s.sparse_dim()`，`K = s.dense_dim()`，那么我们有如下不变式：

> - `M + K == len(s.shape) == s.ndim` - 张量的维度是稀疏维度和稠密维度的总和，
> - `s.indices().shape == (M, nse)` - 稀疏索引被显式存储，
> - `s.values().shape == (nse,) + s.shape[M : M + K]` - 混合张量的值是 K 维张量，
> - `s.values().layout == torch.strided` - 值以步幅张量的形式存储。

 note
 title
Note


稠密维度总是跟在稀疏维度之后，也就是说，不支持混合稀疏维度和稠密维度。


 note
 title
Note


为了确保构造的稀疏张量具有一致的索引、值和大小，可以通过 `check_invariants=True` 关键字参数在每个张量创建时启用不变式检查，或者全局使用 `torch.sparse.check_sparse_tensor_invariants`{.interpreted-text role="class"} 上下文管理器实例。默认情况下，稀疏张量不变式检查是禁用的。


### 未合并的稀疏 COO 张量 {#sparse-uncoalesced-coo-docs}

PyTorch 稀疏 COO 张量格式允许稀疏\*未合并\*张量，其中索引中可能存在重复的坐标；在这种情况下，解释是该索引处的值是所有重复值条目的总和。例如，可以为同一个索引 `1` 指定多个值 `3` 和 `4`，从而得到一个一维未合并张量：

> \>\>\> i = \[\[1, 1\]\] \>\>\> v = \[3, 4\] \>\>\> s=torch.sparse_coo_tensor(i, v, (3,)) \>\>\> s tensor(indices=tensor(\[\[1, 1\]\]), values=tensor( \[3, 4\]), size=(3,), nnz=2, layout=torch.sparse_coo)

而合并过程将使用求和将多值元素累积为单个值：

> \>\>\> s.coalesce() tensor(indices=tensor(\[\[1\]\]), values=tensor(\[7\]), size=(3,), nnz=1, layout=torch.sparse_coo)

通常，`torch.Tensor.coalesce`{.interpreted-text role="meth"} 方法的输出是一个具有以下属性的稀疏张量：

- 指定张量元素的索引是唯一的，
- 索引按字典序排序，
- `torch.Tensor.is_coalesced()`{.interpreted-text role="meth"} 返回 `True`。

 note
 title
Note


在大多数情况下，您不需要关心稀疏张量是否已合并，因为大多数操作在给定稀疏合并或未合并张量时的工作方式相同。

然而，有些操作在未合并张量上可以实现得更高效，而有些则在合并张量上更高效。

例如，稀疏 COO 张量的加法是通过简单连接索引张量和值张量来实现的：

> \>\>\> a = torch.sparse_coo_tensor(\[\[1, 1\]\], \[5, 6\], (2,)) \>\>\> b = torch.sparse_coo_tensor(\[\[0, 0\]\], \[7, 8\], (2,)) \>\>\> a + b tensor(indices=tensor(\[\[0, 0, 1, 1\]\]), values=tensor(\[7, 8, 5, 6\]), size=(2,), nnz=4, layout=torch.sparse_coo)

如果您重复执行可能产生重复条目的操作（例如 `torch.Tensor.add`{.interpreted-text role="func"}），您应该偶尔合并您的稀疏张量，以防止它们变得过大。

另一方面，索引的字典序排序对于实现涉及许多元素选择操作的算法可能是有利的，例如切片或矩阵乘积。


### 使用稀疏 COO 张量

让我们考虑以下示例：

> 
>
> \>\>\> i = \[\[0, 1, 1\],
>
> :   \[2, 0, 2\]\]
>
> \>\>\> v = \[\[3, 4\], \[5, 6\], \[7, 8\]\] \>\>\> s = torch.sparse_coo_tensor(i, v, (2, 3, 2))

如上所述，稀疏 COO 张量是一个 `torch.Tensor`{.interpreted-text role="class"} 实例，为了将其与使用其他布局的 [Tensor]{.title-ref} 实例区分开来，可以使用 `torch.Tensor.is_sparse`{.interpreted-text role="attr"} 或 `torch.Tensor.layout`{.interpreted-text role="attr"} 属性：

> \>\>\> isinstance(s, torch.Tensor) True \>\>\> s.is_sparse True \>\>\> s.layout == torch.sparse_coo True

稀疏维度和稠密维度的数量可以分别使用方法 `torch.Tensor.sparse_dim`{.interpreted-text role="meth"} 和 `torch.Tensor.dense_dim`{.interpreted-text role="meth"} 获取。例如：

> \>\>\> s.sparse_dim(), s.dense_dim() (2, 1)

如果 `s` 是一个稀疏 COO 张量，那么可以使用 `torch.Tensor.indices()`{.interpreted-text role="meth"} 和 `torch.Tensor.values()`{.interpreted-text role="meth"} 方法获取其 COO 格式数据。

 note
 title
Note


目前，只有当张量实例是合并状态时才能获取 COO 格式数据：

> \>\>\> s.indices() RuntimeError: Cannot get indices on an uncoalesced tensor, please call .coalesce() first

要获取未合并张量的 COO 格式数据，请使用 `torch.Tensor._values()`{.interpreted-text role="func"} 和 `torch.Tensor._indices()`{.interpreted-text role="func"}：

> \>\>\> s.\_indices() tensor(\[\[0, 1, 1\], \[2, 0, 2\]\])

 warning
 title
Warning


调用 `torch.Tensor._values()`{.interpreted-text role="meth"} 将返回一个\*分离的\*张量。 要跟踪梯度，必须改用 `torch.Tensor.coalesce().values()`{.interpreted-text role="meth"}。


构造一个新的稀疏 COO 张量会产生一个未合并的张量：

> \>\>\> s.is_coalesced() False

但可以使用 `torch.Tensor.coalesce`{.interpreted-text role="meth"} 方法构造稀疏 COO 张量的合并副本：

> \>\>\> s2 = s.coalesce() \>\>\> s2.indices() tensor(\[\[0, 1, 1\], \[2, 0, 2\]\])

在处理未合并的稀疏 COO 张量时，必须考虑未合并数据的可加性：相同索引的值是求和的项，其计算结果给出相应张量元素的值。例如，稀疏未合并张量的标量乘法可以通过将所有未合并的值乘以标量来实现，因为 `c * (a + b) == c * a + c * b` 成立。然而，任何非线性操作，例如平方根，不能通过对未合并数据应用该操作来实现，因为 `sqrt(a + b) == sqrt(a) + sqrt(b)` 通常不成立。

稀疏 COO 张量的切片（步长为正）仅支持稠密维度。索引同时支持稀疏维度和稠密维度：

> \>\>\> s\[1\] tensor(indices=tensor(\[\[0, 2\]\]), values=tensor(\[\[5, 6\], \[7, 8\]\]), size=(3, 2), nnz=2, layout=torch.sparse_coo) \>\>\> s\[1, 0, 1\] tensor(6) \>\>\> s\[1, 0, 1:\] tensor(\[6\])

在 PyTorch 中，稀疏张量的填充值不能显式指定，通常假定为零。但是，存在一些操作可能以不同方式解释填充值。例如，`torch.sparse.softmax`{.interpreted-text role="func"} 在假设填充值为负无穷大的情况下计算 softmax。

## 稀疏压缩张量 {#sparse-compressed-docs}

稀疏压缩张量代表一类稀疏张量，其共同特点是使用一种编码压缩特定维度的索引，这种编码使得稀疏压缩张量的线性代数内核能够进行某些优化。这种编码基于 [压缩稀疏行 (CSR)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) 格式，PyTorch 稀疏压缩张量在此基础上扩展，支持稀疏张量批次、允许多维张量值，并以稠密块的形式存储稀疏张量值。

 note
 title
Note


我们使用 (B + M + K) 维张量来表示一个 N 维稀疏压缩混合张量，其中 B、M 和 K 分别是批次维度、稀疏维度和稠密维度的数量，满足 `B + M + K == N`。稀疏压缩张量的稀疏维度数始终为二，即 `M == 2`。


 note
 title
Note


如果满足以下不变量，我们说一个索引张量 `compressed_indices` 使用了 CSR 编码：

- `compressed_indices` 是一个连续的跨步 32 位或 64 位整数张量
- `compressed_indices` 的形状是 `(*batchsize, compressed_dim_size + 1)`，其中 `compressed_dim_size` 是压缩维度的数量（例如行或列）
- `compressed_indices[..., 0] == 0`，其中 `...` 表示批次索引
- `compressed_indices[..., compressed_dim_size] == nse`，其中 `nse` 是指定元素的数量
- 对于 `i=1, ..., compressed_dim_size`，满足 `0 <= compressed_indices[..., i] - compressed_indices[..., i - 1] <= plain_dim_size`，其中 `plain_dim_size` 是普通维度的数量（与压缩维度正交，例如列或行）。

为确保构造的稀疏张量具有一致的索引、值和大小，可以通过 `check_invariants=True` 关键字参数在每个张量创建时启用不变量检查，或全局使用 `torch.sparse.check_sparse_tensor_invariants`{.interpreted-text role="class"} 上下文管理器实例。默认情况下，稀疏张量不变量检查是禁用的。


 note
 title
Note


将稀疏压缩布局推广到 N 维张量时，关于指定元素数量的计算可能会引起一些混淆。当稀疏压缩张量包含批次维度时，指定元素的数量将对应于每个批次中此类元素的数量。当稀疏压缩张量具有稠密维度时，所考虑的元素现在变为 K 维数组。同样，对于块稀疏压缩布局，二维块被视为被指定的元素。以一个三维块稀疏张量为例，它有一个长度为 `b` 的批次维度和一个形状为 `p, q` 的块。如果该张量有 `n` 个指定元素，那么实际上我们每个批次指定了 `n` 个块。该张量的 `values` 形状将为 `(b, n, p, q)`。这种对指定元素数量的解释源于所有稀疏压缩布局都是从二维矩阵压缩派生而来的。批次维度被视为稀疏矩阵的堆叠，稠密维度将元素的意义从简单的标量值更改为具有自身维度的数组。

### 稀疏 CSR 张量 {#sparse-csr-docs}

CSR 格式相对于 COO 格式的主要优势在于能更好地利用存储，并且计算操作（如使用 MKL 和 MAGMA 后端的稀疏矩阵-向量乘法）速度更快。

在最简单的情况下，一个 (0 + 2 + 0) 维的稀疏 CSR 张量由三个一维张量组成：`crow_indices`、`col_indices` 和 `values`：

> - `crow_indices` 张量包含压缩的行索引。这是一个大小为 `nrows + 1`（行数加 1）的一维张量。`crow_indices` 的最后一个元素是指定元素的数量 `nse`。该张量根据给定行的起始位置编码 `values` 和 `col_indices` 中的索引。张量中每个连续数字减去前一个数字表示给定行中的元素数量。
> - `col_indices` 张量包含每个元素的列索引。这是一个大小为 `nse` 的一维张量。
> - `values` 张量包含 CSR 张量元素的值。这是一个大小为 `nse` 的一维张量。

 note
 title
Note


索引张量 `crow_indices` 和 `col_indices` 的元素类型应为 `torch.int64`（默认）或 `torch.int32`。如果要使用启用 MKL 的矩阵运算，请使用 `torch.int32`。这是因为 PyTorch 默认链接的是 MKL LP64，它使用 32 位整数索引。


在一般情况下，(B + 2 + K) 维的稀疏 CSR 张量由两个 (B + 1) 维的索引张量 `crow_indices` 和 `col_indices`，以及一个 (1 + K) 维的 `values` 张量组成，满足：

> - `crow_indices.shape == (*batchsize, nrows + 1)`
> - `col_indices.shape == (*batchsize, nse)`
> - `values.shape == (nse, *densesize)`

而稀疏 CSR 张量的形状为 `(*batchsize, nrows, ncols, *densesize)`，其中 `len(batchsize) == B` 且 `len(densesize) == K`。

 note
 title
Note


稀疏 CSR 张量的批次是相互依赖的：所有批次中指定元素的数量必须相同。这个有些人为的约束允许高效存储不同 CSR 批次的索引。


 note
 title
Note


稀疏维度和稠密维度的数量可以使用 `torch.Tensor.sparse_dim`{.interpreted-text role="meth"} 和 `torch.Tensor.dense_dim`{.interpreted-text role="meth"} 方法获取。批次维度可以从张量形状计算得出：`batchsize = tensor.shape[:-tensor.sparse_dim() - tensor.dense_dim()]`。


 note
 title
Note


稀疏 CSR 张量的内存消耗至少为 `(nrows * 8 + (8 + <元素类型大小（字节）> * prod(densesize)) * nse) * prod(batchsize)` 字节（加上存储其他张量数据的常量开销）。

使用 `稀疏 COO 格式介绍中的注释<sparse-coo-docs>`{.interpreted-text role="ref"} 的相同示例数据，一个包含 100,000 个非零 32 位浮点数的 10,000 x 10,000 张量在使用 CSR 张量布局时，内存消耗至少为 `(10000 * 8 + (8 + 4 * 1) * 100 000) * 1 = 1 280 000` 字节。请注意，与使用 COO 和 strided 格式相比，使用 CSR 存储格式分别节省了 1.6 倍和 310 倍的内存。


#### CSR 张量的构建

稀疏 CSR 张量可以直接使用 `torch.sparse_csr_tensor`{.interpreted-text role="func"} 函数构建。用户必须分别提供行索引、列索引和值张量，其中行索引必须使用 CSR 压缩编码指定。`size` 参数是可选的，如果未提供，将从 `crow_indices` 和 `col_indices` 推导得出。

> \>\>\> crow_indices = torch.tensor(\[0, 2, 4\]) \>\>\> col_indices = torch.tensor(\[0, 1, 0, 1\]) \>\>\> values = torch.tensor(\[1, 2, 3, 4\]) \>\>\> csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.float64) \>\>\> csr tensor(crow_indices=tensor(\[0, 2, 4\]), col_indices=tensor(\[0, 1, 0, 1\]), values=tensor(\[1., 2., 3., 4.\]), size=(2, 2), nnz=4, dtype=torch.float64) \>\>\> csr.to_dense() tensor(\[\[1., 2.\], \[3., 4.\]\], dtype=torch.float64)

 note
 title
Note


推导出的 `size` 中稀疏维度的值是根据 `crow_indices` 的大小和 `col_indices` 中的最大索引值计算的。如果所需的列数大于推导出的 `size` 中的列数，则必须显式指定 `size` 参数。


从 strided 或稀疏 COO 张量构建二维稀疏 CSR 张量的最简单方法是使用 `torch.Tensor.to_sparse_csr`{.interpreted-text role="meth"} 方法。(strided) 张量中的任何零值都将被解释为稀疏张量中的缺失值：

\>\>\> a = torch.tensor(\[\[0, 0, 1, 0\], \[1, 2, 0, 0\], \[0, 0, 0, 0\]\], dtype=torch.float64)

:   \>\>\> sp = a.to_sparse_csr() \>\>\> sp tensor(crow_indices=tensor(\[0, 1, 3, 3\]), col_indices=tensor(\[2, 0, 1\]), values=tensor(\[1., 1., 2.\]), size=(3, 4), nnz=3, dtype=torch.float64)

#### CSR 张量操作

稀疏矩阵-向量乘法可以通过 `tensor.matmul`{.interpreted-text role="meth"} 方法执行。这是目前 CSR 张量唯一支持的数学运算。

> \>\>\> vec = torch.randn(4, 1, dtype=torch.float64) \>\>\> sp.matmul(vec) tensor(\[\[0.9078\], \[1.3180\], \[0.0000\]\], dtype=torch.float64)

### 稀疏 CSC 张量 {#sparse-csc-docs}

稀疏 CSC（压缩稀疏列）张量格式实现了 CSC 格式，用于存储二维张量，并扩展支持批量的稀疏 CSC 张量以及值为多维张量的情况。

 note
 title
Note


当转置涉及交换稀疏维度时，稀疏 CSC 张量本质上是稀疏 CSR 张量的转置。


与 `稀疏 CSR 张量 <sparse-csr-docs>`{.interpreted-text role="ref"} 类似，稀疏 CSC 张量由三个张量组成：`ccol_indices`、`row_indices` 和 `values`：

> - `ccol_indices` 张量包含压缩的列索引。这是一个形状为 `(*batchsize, ncols + 1)` 的 (B + 1) 维张量。 最后一个元素是指定元素的数量 `nse`。该张量根据给定列开始的位置编码 `values` 和 `row_indices` 中的索引。 张量中每个连续数字减去其前一个数字表示给定列中的元素数量。
> - `row_indices` 张量包含每个元素的行索引。这是一个形状为 `(*batchsize, nse)` 的 (B + 1) 维张量。
> - `values` 张量包含 CSC 张量元素的值。这是一个形状为 `(nse, *densesize)` 的 (1 + K) 维张量。

#### CSC 张量的构造

稀疏 CSC 张量可以直接使用 `torch.sparse_csc_tensor`{.interpreted-text role="func"} 函数构造。用户必须分别提供行索引、列索引和值张量，其中列索引必须使用 CSR 压缩编码指定。`size` 参数是可选的，如果未提供，将从 `row_indices` 和 `ccol_indices` 张量推断。

> \>\>\> ccol_indices = torch.tensor(\[0, 2, 4\]) \>\>\> row_indices = torch.tensor(\[0, 1, 0, 1\]) \>\>\> values = torch.tensor(\[1, 2, 3, 4\]) \>\>\> csc = torch.sparse_csc_tensor(ccol_indices, row_indices, values, dtype=torch.float64) \>\>\> csc tensor(ccol_indices=tensor(\[0, 2, 4\]), row_indices=tensor(\[0, 1, 0, 1\]), values=tensor(\[1., 2., 3., 4.\]), size=(2, 2), nnz=4, dtype=torch.float64, layout=torch.sparse_csc) \>\>\> csc.to_dense() tensor(\[\[1., 3.\], \[2., 4.\]\], dtype=torch.float64)

 note
 title
Note


稀疏 CSC 张量构造函数的压缩列索引参数位于行索引参数之前。


(0 + 2 + 0) 维的稀疏 CSC 张量可以使用 `torch.Tensor.to_sparse_csc`{.interpreted-text role="meth"} 方法从任何二维张量构造。(跨步) 张量中的任何零值将被解释为稀疏张量中的缺失值：

> \>\>\> a = torch.tensor(\[\[0, 0, 1, 0\], \[1, 2, 0, 0\], \[0, 0, 0, 0\]\], dtype=torch.float64) \>\>\> sp = a.to_sparse_csc() \>\>\> sp tensor(ccol_indices=tensor(\[0, 1, 2, 3, 3\]), row_indices=tensor(\[1, 1, 0\]), values=tensor(\[1., 2., 1.\]), size=(3, 4), nnz=3, dtype=torch.float64, layout=torch.sparse_csc)

### 稀疏 BSR 张量 {#sparse-bsr-docs}

稀疏 BSR（块压缩稀疏行）张量格式实现了 BSR 格式，用于存储二维张量，并扩展支持批量的稀疏 BSR 张量以及值为多维张量块的情况。

稀疏 BSR 张量由三个张量组成：`crow_indices`、`col_indices` 和 `values`：

> - `crow_indices` 张量包含压缩的行索引。这是一个形状为 `(*batchsize, nrowblocks + 1)` 的 (B + 1) 维张量。 最后一个元素是指定块的数量 `nse`。该张量根据给定列块开始的位置编码 `values` 和 `col_indices` 中的索引。 张量中每个连续数字减去其前一个数字表示给定行中的块数量。
> - `col_indices` 张量包含每个元素的列块索引。这是一个形状为 `(*batchsize, nse)` 的 (B + 1) 维张量。
> - `values` 张量包含收集到二维块中的稀疏 BSR 张量元素的值。这是一个形状为 `(nse, nrowblocks, ncolblocks, *densesize)` 的 (1 + 2 + K) 维张量。

#### BSR 张量的构造

稀疏 BSR 张量可以直接使用 `torch.sparse_bsr_tensor`{.interpreted-text role="func"} 函数构造。用户必须分别提供行块索引、列块索引和值张量，其中行块索引必须使用 CSR 压缩编码指定。 `size` 参数是可选的，如果未提供，将从 `crow_indices` 和 `col_indices` 张量推断。

\>\>\> crow_indices = torch.tensor(\[0, 2, 4\])

:   \>\>\> col_indices = torch.tensor(\[0, 1, 0, 1\]) \>\>\> values = torch.tensor(\[\[\[0, 1, 2\], \[6, 7, 8\]\], \... \[\[3, 4, 5\], \[9, 10, 11\]\], \... \[\[12, 13, 14\], \[18, 19, 20\]\], \... \[\[15, 16, 17\], \[21, 22, 23\]\]\]) \>\>\> bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, dtype=torch.float64) \>\>\> bsr tensor(crow_indices=tensor(\[0, 2, 4\]), col_indices=tensor(\[0, 1, 0, 1\]), values=tensor(\[\[\[ 0., 1., 2.\], \[ 6., 7., 8.\]\], \[\[ 3., 4., 5.\], \[ 9., 10., 11.\]\], \[\[12., 13., 14.\], \[18., 19., 20.\]\], \[\[15., 16., 17.\], \[21., 22., 23.\]\]\]), size=(4, 6), nnz=4, dtype=torch.float64, layout=torch.sparse_bsr) \>\>\> bsr.to_dense() tensor(\[\[ 0., 1., 2., 3., 4., 5.\], \[ 6., 7., 8., 9., 10., 11.\], \[12., 13., 14., 15., 16., 17.\], \[18., 19., 20., 21., 22., 23.\]\], dtype=torch.float64)

(0 + 2 + 0) 维的稀疏 BSR 张量可以通过使用 `torch.Tensor.to_sparse_bsr`{.interpreted-text role="meth"} 方法从任意二维张量构造，该方法还需要指定值块的大小：

> \>\>\> dense = torch.tensor(\[\[0, 1, 2, 3, 4, 5\], \... \[6, 7, 8, 9, 10, 11\], \... \[12, 13, 14, 15, 16, 17\], \... \[18, 19, 20, 21, 22, 23\]\]) \>\>\> bsr = dense.to_sparse_bsr(blocksize=(2, 3)) \>\>\> bsr tensor(crow_indices=tensor(\[0, 2, 4\]), col_indices=tensor(\[0, 1, 0, 1\]), values=tensor(\[\[\[ 0, 1, 2\], \[ 6, 7, 8\]\], \[\[ 3, 4, 5\], \[ 9, 10, 11\]\], \[\[12, 13, 14\], \[18, 19, 20\]\], \[\[15, 16, 17\], \[21, 22, 23\]\]\]), size=(4, 6), nnz=4, layout=torch.sparse_bsr)

### 稀疏 BSC 张量 {#sparse-bsc-docs}

稀疏 BSC（块压缩稀疏列）张量格式实现了 BSC 格式，用于存储二维张量，并扩展支持批处理的稀疏 BSC 张量以及值为多维张量块的情况。

一个稀疏 BSC 张量由三个张量组成：`ccol_indices`、`row_indices` 和 `values`：

> - `ccol_indices` 张量包含压缩的列索引。这是一个 (B + 1) 维张量，形状为 `(*batchsize, ncolblocks + 1)`。最后一个元素是指定块的数量 `nse`。该张量根据给定行块的起始位置编码 `values` 和 `row_indices` 中的索引。张量中每个连续数字减去前一个数字表示给定列中的块数。
> - `row_indices` 张量包含每个元素的行块索引。这是一个 (B + 1) 维张量，形状为 `(*batchsize, nse)`。
> - `values` 张量包含收集到二维块中的稀疏 BSC 张量元素的值。这是一个 (1 + 2 + K) 维张量，形状为 `(nse, nrowblocks, ncolblocks, *densesize)`。

#### BSC 张量的构造

稀疏 BSC 张量可以直接使用 `torch.sparse_bsc_tensor`{.interpreted-text role="func"} 函数构造。用户必须分别提供行和列块索引以及值张量，其中列块索引必须使用 CSR 压缩编码指定。 `size` 参数是可选的，如果未提供，将从 `ccol_indices` 和 `row_indices` 张量推导得出。

> \>\>\> ccol_indices = torch.tensor(\[0, 2, 4\]) \>\>\> row_indices = torch.tensor(\[0, 1, 0, 1\]) \>\>\> values = torch.tensor(\[\[\[0, 1, 2\], \[6, 7, 8\]\], \... \[\[3, 4, 5\], \[9, 10, 11\]\], \... \[\[12, 13, 14\], \[18, 19, 20\]\], \... \[\[15, 16, 17\], \[21, 22, 23\]\]\]) \>\>\> bsc = torch.sparse_bsc_tensor(ccol_indices, row_indices, values, dtype=torch.float64) \>\>\> bsc tensor(ccol_indices=tensor(\[0, 2, 4\]), row_indices=tensor(\[0, 1, 0, 1\]), values=tensor(\[\[\[ 0., 1., 2.\], \[ 6., 7., 8.\]\], \[\[ 3., 4., 5.\], \[ 9., 10., 11.\]\], \[\[12., 13., 14.\], \[18., 19., 20.\]\], \[\[15., 16., 17.\], \[21., 22., 23.\]\]\]), size=(4, 6), nnz=4, dtype=torch.float64, layout=torch.sparse_bsc)

### 处理稀疏压缩张量的工具

所有稀疏压缩张量------CSR、CSC、BSR 和 BSC 张量------在概念上非常相似，因为它们的索引数据分为两部分：使用 CSR 编码的所谓压缩索引，以及正交于压缩索引的所谓普通索引。这使得这些张量上的各种工具可以共享相同的实现，这些实现通过张量布局进行参数化。

#### 稀疏压缩张量的构造

稀疏 CSR、CSC、BSR 和 CSC 张量可以通过使用 `torch.sparse_compressed_tensor`{.interpreted-text role="func"} 函数构造，该函数具有与上述构造函数 `torch.sparse_csr_tensor`{.interpreted-text role="func"}、`torch.sparse_csc_tensor`{.interpreted-text role="func"}、`torch.sparse_bsr_tensor`{.interpreted-text role="func"} 和 `torch.sparse_bsc_tensor`{.interpreted-text role="func"} 相同的接口，但需要一个额外的必需参数 `layout`。以下示例通过向 `torch.sparse_compressed_tensor`{.interpreted-text role="func"} 函数指定相应的布局参数，说明了使用相同输入数据构造 CSR 和 CSC 张量的方法：

> \>\>\> compressed_indices = torch.tensor(\[0, 2, 4\]) \>\>\> plain_indices = torch.tensor(\[0, 1, 0, 1\]) \>\>\> values = torch.tensor(\[1, 2, 3, 4\]) \>\>\> csr = torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, layout=torch.sparse_csr) \>\>\> csr tensor(crow_indices=tensor(\[0, 2, 4\]), col_indices=tensor(\[0, 1, 0, 1\]), values=tensor(\[1, 2, 3, 4\]), size=(2, 2), nnz=4, layout=torch.sparse_csr) \>\>\> csc = torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, layout=torch.sparse_csc) \>\>\> csc tensor(ccol_indices=tensor(\[0, 2, 4\]), row_indices=tensor(\[0, 1, 0, 1\]), values=tensor(\[1, 2, 3, 4\]), size=(2, 2), nnz=4, layout=torch.sparse_csc) \>\>\> (csr.transpose(0, 1).to_dense() == csc.to_dense()).all() tensor(True)

## 支持的操作 {#sparse-ops-docs}

### 线性代数操作

下表总结了稀疏矩阵上支持的线性代数操作，其中操作数的布局可能不同。这里 `T[layout]` 表示具有给定布局的张量。类似地， `M[layout]` 表示矩阵（2-D PyTorch 张量），而 `V[layout]` 表示向量（1-D PyTorch 张量）。此外，`f` 表示一个 标量（浮点数或 0-D PyTorch 张量），`*` 是逐元素 乘法，`@` 是矩阵乘法。

  -----------------------------------------------------------------------------------------------------------------------------------------------------
  PyTorch 操作                                            支持稀疏梯度?   布局签名
  ------------------------------------------------------- --------------- -----------------------------------------------------------------------------
  `torch.mv`{.interpreted-text role="func"}               否              `M[sparse_coo] @ V[strided] -> V[strided]`

  `torch.mv`{.interpreted-text role="func"}               否              `M[sparse_csr] @ V[strided] -> V[strided]`

  `torch.matmul`{.interpreted-text role="func"}           否              `M[sparse_coo] @ M[strided] -> M[strided]`

  `torch.matmul`{.interpreted-text role="func"}           否              `M[sparse_csr] @ M[strided] -> M[strided]`

  `torch.matmul`{.interpreted-text role="func"}           否              `M[SparseSemiStructured] @ M[strided] -> M[strided]`

  `torch.matmul`{.interpreted-text role="func"}           否              `M[strided] @ M[SparseSemiStructured] -> M[strided]`

  `torch.mm`{.interpreted-text role="func"}               否              `M[sparse_coo] @ M[strided] -> M[strided]`

  `torch.mm`{.interpreted-text role="func"}               否              `M[SparseSemiStructured] @ M[strided] -> M[strided]`

  `torch.mm`{.interpreted-text role="func"}               否              `M[strided] @ M[SparseSemiStructured] -> M[strided]`

  `torch.sparse.mm`{.interpreted-text role="func"}        是              `M[sparse_coo] @ M[strided] -> M[strided]`

  `torch.smm`{.interpreted-text role="func"}              否              `M[sparse_coo] @ M[strided] -> M[sparse_coo]`

  `torch.hspmm`{.interpreted-text role="func"}            否              `M[sparse_coo] @ M[strided] -> M[hybrid sparse_coo]`

  `torch.bmm`{.interpreted-text role="func"}              否              `T[sparse_coo] @ T[strided] -> T[strided]`

  `torch.addmm`{.interpreted-text role="func"}            否              `f * M[strided] + f * (M[sparse_coo] @ M[strided]) -> M[strided]`

  `torch.addmm`{.interpreted-text role="func"}            否              `f * M[strided] + f * (M[SparseSemiStructured] @ M[strided]) -> M[strided]`

  `torch.addmm`{.interpreted-text role="func"}            否              `f * M[strided] + f * (M[strided] @ M[SparseSemiStructured]) -> M[strided]`

  `torch.sparse.addmm`{.interpreted-text role="func"}     是              `f * M[strided] + f * (M[sparse_coo] @ M[strided]) -> M[strided]`

  `torch.sparse.spsolve`{.interpreted-text role="func"}   否              `SOLVE(M[sparse_csr], V[strided]) -> V[strided]`

  `torch.sspaddmm`{.interpreted-text role="func"}         否              `f * M[sparse_coo] + f * (M[sparse_coo] @ M[strided]) -> M[sparse_coo]`

  `torch.lobpcg`{.interpreted-text role="func"}           否              `GENEIG(M[sparse_coo]) -> M[strided], M[strided]`

  `torch.pca_lowrank`{.interpreted-text role="func"}      是              `PCA(M[sparse_coo]) -> M[strided], M[strided], M[strided]`

  `torch.svd_lowrank`{.interpreted-text role="func"}      是              `SVD(M[sparse_coo]) -> M[strided], M[strided], M[strided]`
  -----------------------------------------------------------------------------------------------------------------------------------------------------

其中\"支持稀疏梯度?\"列表示 PyTorch 操作是否支持 对稀疏矩阵参数的向后传播。除 `torch.smm`{.interpreted-text role="func"} 外，所有 PyTorch 操作 都支持对 strided 矩阵参数的向后传播。

 note
 title
Note


目前，PyTorch 不支持布局签名为 `M[strided] @ M[sparse_coo]` 的 矩阵乘法。然而， 应用程序仍然可以使用矩阵关系 `D @ S == (S.t() @ D.t()).t()` 来计算此操作。


### 张量方法与稀疏

以下 Tensor 方法与稀疏张量相关：

 {.autosummary toctree="generated" nosignatures=""}
Tensor.is_sparse Tensor.is_sparse_csr Tensor.dense_dim Tensor.sparse_dim Tensor.sparse_mask Tensor.to_sparse Tensor.to_sparse_coo Tensor.to_sparse_csr Tensor.to_sparse_csc Tensor.to_sparse_bsr Tensor.to_sparse_bsc Tensor.to_dense Tensor.values


以下 Tensor 方法特定于稀疏 COO 张量：

 {.autosummary toctree="generated" nosignatures=""}
Tensor.coalesce [Tensor.sparse_resize]() [Tensor.sparse_resize_and_clear]() Tensor.is_coalesced Tensor.indices


以下方法特定于 `稀疏 CSR 张量 <sparse-csr-docs>`{.interpreted-text role="ref"} 和 `稀疏 BSR 张量 <sparse-bsr-docs>`{.interpreted-text role="ref"}：

 {.autosummary toctree="generated" nosignatures=""}
Tensor.crow_indices Tensor.col_indices


以下方法特定于 `稀疏 CSC 张量 <sparse-csc-docs>`{.interpreted-text role="ref"} 和 `稀疏 BSC 张量 <sparse-bsc-docs>`{.interpreted-text role="ref"}：

 {.autosummary toctree="generated" nosignatures=""}
Tensor.row_indices Tensor.ccol_indices


以下 Tensor 方法支持稀疏 COO 张量：

`~torch.Tensor.add`{.interpreted-text role="meth"} `~torch.Tensor.add_`{.interpreted-text role="meth"} `~torch.Tensor.addmm`{.interpreted-text role="meth"} `~torch.Tensor.addmm_`{.interpreted-text role="meth"} `~torch.Tensor.any`{.interpreted-text role="meth"} `~torch.Tensor.asin`{.interpreted-text role="meth"} `~torch.Tensor.asin_`{.interpreted-text role="meth"} `~torch.Tensor.arcsin`{.interpreted-text role="meth"} `~torch.Tensor.arcsin_`{.interpreted-text role="meth"} `~torch.Tensor.bmm`{.interpreted-text role="meth"} `~torch.Tensor.clone`{.interpreted-text role="meth"} `~torch.Tensor.deg2rad`{.interpreted-text role="meth"} `~torch.Tensor.deg2rad_`{.interpreted-text role="meth"} `~torch.Tensor.detach`{.interpreted-text role="meth"} `~torch.Tensor.detach_`{.interpreted-text role="meth"} `~torch.Tensor.dim`{.interpreted-text role="meth"} `~torch.Tensor.div`{.interpreted-text role="meth"} `~torch.Tensor.div_`{.interpreted-text role="meth"} `~torch.Tensor.floor_divide`{.interpreted-text role="meth"} `~torch.Tensor.floor_divide_`{.interpreted-text role="meth"} `~torch.Tensor.get_device`{.interpreted-text role="meth"} `~torch.Tensor.index_select`{.interpreted-text role="meth"} `~torch.Tensor.isnan`{.interpreted-text role="meth"} `~torch.Tensor.log1p`{.interpreted-text role="meth"} `~torch.Tensor.log1p_`{.interpreted-text role="meth"} `~torch.Tensor.mm`{.interpreted-text role="meth"} `~torch.Tensor.mul`{.interpreted-text role="meth"} `~torch.Tensor.mul_`{.interpreted-text role="meth"} `~torch.Tensor.mv`{.interpreted-text role="meth"} `~torch.Tensor.narrow_copy`{.interpreted-text role="meth"} `~torch.Tensor.neg`{.interpreted-text role="meth"} `~torch.Tensor.neg_`{.interpreted-text role="meth"} `~torch.Tensor.negative`{.interpreted-text role="meth"} `~torch.Tensor.negative_`{.interpreted-text role="meth"} `~torch.Tensor.numel`{.interpreted-text role="meth"} `~torch.Tensor.rad2deg`{.interpreted-text role="meth"} `~torch.Tensor.rad2deg_`{.interpreted-text role="meth"} `~torch.Tensor.resize_as_`{.interpreted-text role="meth"} `~torch.Tensor.size`{.interpreted-text role="meth"} `~torch.Tensor.pow`{.interpreted-text role="meth"} `~torch.Tensor.sqrt`{.interpreted-text role="meth"} `~torch.Tensor.square`{.interpreted-text role="meth"} `~torch.Tensor.smm`{.interpreted-text role="meth"} `~torch.Tensor.sspaddmm`{.interpreted-text role="meth"} `~torch.Tensor.sub`{.interpreted-text role="meth"} `~torch.Tensor.sub_`{.interpreted-text role="meth"} `~torch.Tensor.t`{.interpreted-text role="meth"} `~torch.Tensor.t_`{.interpreted-text role="meth"} `~torch.Tensor.transpose`{.interpreted-text role="meth"} `~torch.Tensor.transpose_`{.interpreted-text role="meth"} `~torch.Tensor.zero_`{.interpreted-text role="meth"}

### 专用于稀疏张量的 Torch 函数

 {.autosummary toctree="generated" nosignatures=""}
sparse_coo_tensor sparse_csr_tensor sparse_csc_tensor sparse_bsr_tensor sparse_bsc_tensor sparse_compressed_tensor sparse.sum sparse.addmm sparse.sampled_addmm sparse.mm sspaddmm hspmm smm sparse.softmax sparse.spsolve sparse.log_softmax sparse.spdiags


### 其他函数

以下 `torch`{.interpreted-text role="mod"} 函数支持稀疏张量：

`~torch.cat`{.interpreted-text role="func"} `~torch.dstack`{.interpreted-text role="func"} `~torch.empty`{.interpreted-text role="func"} `~torch.empty_like`{.interpreted-text role="func"} `~torch.hstack`{.interpreted-text role="func"} `~torch.index_select`{.interpreted-text role="func"} `~torch.is_complex`{.interpreted-text role="func"} `~torch.is_floating_point`{.interpreted-text role="func"} `~torch.is_nonzero`{.interpreted-text role="func"} `~torch.is_same_size`{.interpreted-text role="func"} `~torch.is_signed`{.interpreted-text role="func"} `~torch.is_tensor`{.interpreted-text role="func"} `~torch.lobpcg`{.interpreted-text role="func"} `~torch.mm`{.interpreted-text role="func"} `~torch.native_norm`{.interpreted-text role="func"} `~torch.pca_lowrank`{.interpreted-text role="func"} `~torch.select`{.interpreted-text role="func"} `~torch.stack`{.interpreted-text role="func"} `~torch.svd_lowrank`{.interpreted-text role="func"} `~torch.unsqueeze`{.interpreted-text role="func"} `~torch.vstack`{.interpreted-text role="func"} `~torch.zeros`{.interpreted-text role="func"} `~torch.zeros_like`{.interpreted-text role="func"}

要管理检查稀疏张量不变量，请参阅：

 {.autosummary toctree="generated" nosignatures=""}
sparse.check_sparse_tensor_invariants


要将稀疏张量与 `~torch.autograd.gradcheck`{.interpreted-text role="func"} 函数一起使用， 请参阅：

 {.autosummary toctree="generated" nosignatures=""}
sparse.as_sparse_gradcheck


### 保零一元函数

我们的目标是支持所有"保零一元函数"：将零映射为零的单参数函数。

如果您发现我们缺少您需要的保零一元函数， 我们鼓励您为此功能请求创建一个问题。 与往常一样，请在创建问题之前先尝试使用搜索功能。

以下运算符目前支持稀疏 COO/CSR/CSC/BSR/CSR 张量输入。

`~torch.abs`{.interpreted-text role="func"} `~torch.asin`{.interpreted-text role="func"} `~torch.asinh`{.interpreted-text role="func"} `~torch.atan`{.interpreted-text role="func"} `~torch.atanh`{.interpreted-text role="func"} `~torch.ceil`{.interpreted-text role="func"} `~torch.conj_physical`{.interpreted-text role="func"} `~torch.floor`{.interpreted-text role="func"} `~torch.log1p`{.interpreted-text role="func"} `~torch.neg`{.interpreted-text role="func"} `~torch.round`{.interpreted-text role="func"} `~torch.sin`{.interpreted-text role="func"} `~torch.sinh`{.interpreted-text role="func"} `~torch.sign`{.interpreted-text role="func"} `~torch.sgn`{.interpreted-text role="func"} `~torch.signbit`{.interpreted-text role="func"} `~torch.tan`{.interpreted-text role="func"} `~torch.tanh`{.interpreted-text role="func"} `~torch.trunc`{.interpreted-text role="func"} `~torch.expm1`{.interpreted-text role="func"} `~torch.sqrt`{.interpreted-text role="func"} `~torch.angle`{.interpreted-text role="func"} `~torch.isinf`{.interpreted-text role="func"} `~torch.isposinf`{.interpreted-text role="func"} `~torch.isneginf`{.interpreted-text role="func"} `~torch.isnan`{.interpreted-text role="func"} `~torch.erf`{.interpreted-text role="func"} `~torch.erfinv`{.interpreted-text role="func"}
