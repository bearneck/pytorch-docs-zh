# torch.nested


## 简介

```{warning}
  嵌套张量目前尚未处于积极开发阶段。请自行承担使用风险。
```

嵌套张量允许将不规则形状的数据作为单个张量进行包含和操作。此类数据在底层以高效的打包表示形式存储，同时暴露标准的 PyTorch 张量接口以应用操作。

嵌套张量的一个常见应用是表示不同领域中存在的可变长度序列数据的批次，例如不同的句子长度、图像大小以及音频/视频剪辑长度。传统上，此类数据通过将序列填充到批次内的最大长度、对填充形式执行计算，然后进行掩码处理以移除填充来处理。这种方法效率低下且容易出错，而嵌套张量的存在正是为了解决这些问题。

在嵌套张量上调用操作的 API 与常规 ``torch.Tensor`` 并无不同，允许与现有模型无缝集成，主要区别在于 `输入的构造 <construction>`。

由于这是一个原型功能，`支持的操作 <supported operations>` 集合有限，但正在不断增长。我们欢迎问题报告、功能请求和贡献。有关贡献的更多信息，请参阅 [此 Readme](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/nested/README.md)。


## 构造

```{note}

  PyTorch 中存在两种形式的嵌套张量，通过构造时指定的布局进行区分。布局可以是 ``torch.strided`` 或 ``torch.jagged`` 之一。
  我们建议尽可能使用 ``torch.jagged`` 布局。虽然它目前仅支持单个不规则维度，但它具有更好的操作覆盖范围，正在积极开发中，并且与 ``torch.compile`` 集成良好。本文档遵循此建议，为简洁起见，将具有 ``torch.jagged`` 布局的嵌套张量称为 "NJTs"。
```

构造很简单，涉及将张量列表传递给 ``torch.nested.nested_tensor`` 构造函数。具有 ``torch.jagged`` 布局的嵌套张量（也称为 "NJT"）支持单个不规则维度。此构造函数将根据下面 `data_layout`_ 部分描述的布局，将输入张量复制到一个打包的、连续的内存块中。

```
>>> a, b = torch.arange(3), torch.arange(5) + 3
>>> a
tensor([0, 1, 2])
>>> b
tensor([3, 4, 5, 6, 7])
>>> nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)
>>> print([component for component in nt])
[tensor([0, 1, 2]), tensor([3, 4, 5, 6, 7])]
```

列表中的每个张量必须具有相同的维度数，但形状可以在单个维度上有所不同。如果输入组件的维度不匹配，构造函数将抛出错误。
```
>>> a = torch.randn(50, 128) # 2D tensor
>>> b = torch.randn(2, 50, 128) # 3D tensor
>>> nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)
...
RuntimeError: When constructing a nested tensor, all tensors in list must have the same dim
```

在构造过程中，可以通过常用的关键字参数选择 dtype、device 以及是否需要梯度。

```
>>> nt = torch.nested.nested_tensor([a, b], layout=torch.jagged, dtype=torch.float32, device="cuda", requires_grad=True)
>>> print([component for component in nt])
[tensor([0., 1., 2.], device='cuda:0',
       grad_fn=<UnbindBackwardAutogradNestedTensor0>), tensor([3., 4., 5., 6., 7.], device='cuda:0',
       grad_fn=<UnbindBackwardAutogradNestedTensor0>)]
```

``torch.nested.as_nested_tensor`` 可用于保留传递给构造函数的张量的自动求导历史。当使用此构造函数时，梯度将通过嵌套张量流回原始组件。请注意，此构造函数仍会将输入组件复制到一个打包的、连续的内存块中。

```
>>> a = torch.randn(12, 512, requires_grad=True)
>>> b = torch.randn(23, 512, requires_grad=True)
>>> nt = torch.nested.as_nested_tensor([a, b], layout=torch.jagged, dtype=torch.float32)
>>> nt.sum().backward()
>>> a.grad
tensor([[1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        ...,
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.]])
>>> b.grad
tensor([[1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        ...,
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.]])
```

上述函数都创建了连续的 NJT，其中分配了一块内存来存储底层组件的打包形式（更多细节请参阅下面的 `data_layout`_ 部分）。

也可以在具有填充的现有密集张量上创建非连续的 NJT 视图，从而避免内存分配和复制。``torch.nested.narrow()`` 是实现此目的的工具。

```
>>> padded = torch.randn(3, 5, 4)
>>> seq_lens = torch.tensor([3, 2, 5], dtype=torch.int64)
>>> nt = torch.nested.narrow(padded, dim=1, start=0, length=seq_lens, layout=torch.jagged)
>>> nt.shape
torch.Size([3, j1, 4])
>>> nt.is_contiguous()
False
```

请注意，嵌套张量充当原始填充密集张量的视图，引用相同的内存而不进行复制/分配。对非连续 NJT 的操作支持有些有限，因此如果遇到支持缺口，始终可以使用 ``contiguous()`` 转换为连续的 NJT。


## 数据布局和形状

出于效率考虑，嵌套张量通常将其张量分量打包到连续的内存块中，并维护额外的元数据来指定批次项的边界。对于 ``torch.jagged`` 布局，连续内存块存储在 ``values`` 分量中，而 ``offsets`` 分量则用于界定不规则维度的批次项边界。

![image](_static/img/nested/njt_visual.png)

在必要时，可以直接访问底层的 NJT 组件。

```
>>> a = torch.randn(50, 128) # 文本 1
>>> b = torch.randn(32, 128) # 文本 2
>>> nt = torch.nested.nested_tensor([a, b], layout=torch.jagged, dtype=torch.float32)
>>> nt.values().shape  # 注意不规则维度的"打包"；无需填充
torch.Size([82, 128])
>>> nt.offsets()
tensor([ 0, 50, 82])
```

直接从锯齿状的 ``values`` 和 ``offsets`` 组件构造 NJT 也可能很有用；``torch.nested.nested_tensor_from_jagged()`` 构造函数就是为此目的服务的。

```
>>> values = torch.randn(82, 128)
>>> offsets = torch.tensor([0, 50, 82], dtype=torch.int64)
>>> nt = torch.nested.nested_tensor_from_jagged(values=values, offsets=offsets)
```

NJT 具有一个明确定义的形状，其维度比其组件多 1。不规则维度的底层结构由一个符号值表示（如下例中的 ``j1``）。

```
>>> a = torch.randn(50, 128)
>>> b = torch.randn(32, 128)
>>> nt = torch.nested.nested_tensor([a, b], layout=torch.jagged, dtype=torch.float32)
>>> nt.dim()
3
>>> nt.shape
torch.Size([2, j1, 128])
```

NJT 必须具有相同的不规则结构才能彼此兼容。例如，要运行涉及两个 NJT 的二元操作，不规则结构必须匹配（即它们的形状中必须具有相同的不规则形状符号）。具体来说，每个符号对应一个确切的 ``offsets`` 张量，因此两个 NJT 必须具有相同的 ``offsets`` 张量才能彼此兼容。

```
>>> a = torch.randn(50, 128)
>>> b = torch.randn(32, 128)
>>> nt1 = torch.nested.nested_tensor([a, b], layout=torch.jagged, dtype=torch.float32)
>>> nt2 = torch.nested.nested_tensor([a, b], layout=torch.jagged, dtype=torch.float32)
>>> nt1.offsets() is nt2.offsets()
False
>>> nt3 = nt1 + nt2
RuntimeError: cannot call binary pointwise function add.Tensor with inputs of shapes (2, j2, 128) and (2, j3, 128)
```

在上面的例子中，尽管两个 NJT 的概念形状相同，但它们没有引用同一个 ``offsets`` 张量，因此它们的形状不同，并且不兼容。我们认识到这种行为不够直观，正在努力为嵌套张量的 Beta 版本放宽此限制。有关解决方法，请参阅本文档的 `故障排除 <ragged_structure_incompatibility>` 部分。

除了 ``offsets`` 元数据外，NJT 还可以计算并缓存其组件的最小和最大序列长度，这对于调用特定的内核（例如 SDPA）可能很有用。目前还没有公开的 API 来访问这些信息，但这将在 Beta 版本中改变。


## 支持的操作

本节列出了您可能觉得有用的嵌套张量常见操作列表。这并非详尽无遗，因为 PyTorch 中有数千个操作。虽然目前其中相当一部分支持嵌套张量，但完全支持是一项庞大的任务。嵌套张量的理想状态是全面支持所有可用于非嵌套张量的 PyTorch 操作。为了帮助我们实现这一目标，请考虑：

* 在此处[请求](https://github.com/pytorch/pytorch/issues/118107)您的用例所需的特定操作，以帮助我们确定优先级。
* 贡献代码！为给定的 PyTorch 操作添加嵌套张量支持并不太难；有关详细信息，请参阅下面的[贡献](contributions)部分。

### 查看嵌套张量组件

``unbind()`` 允许您检索嵌套张量组件的视图。

```
>>> import torch
>>> a = torch.randn(2, 3)
>>> b = torch.randn(3, 3)
>>> nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)
>>> nt.unbind()
(tensor([[-0.9916, -0.3363, -0.2799],
        [-2.3520, -0.5896, -0.4374]]), tensor([[-2.0969, -1.0104,  1.4841],
        [ 2.0952,  0.2973,  0.2516],
        [ 0.9035,  1.3623,  0.2026]]))
>>> nt.unbind()[0] is not a
True
>>> nt.unbind()[0].mul_(3)
tensor([[ 3.6858, -3.7030, -4.4525],
        [-2.3481,  2.0236,  0.1975]])
>>> nt.unbind()
(tensor([[-2.9747, -1.0089, -0.8396],
        [-7.0561, -1.7688, -1.3122]]), tensor([[-2.0969, -1.0104,  1.4841],
        [ 2.0952,  0.2973,  0.2516],
        [ 0.9035,  1.3623,  0.2026]]))
```

请注意，``nt.unbind()[0]`` 不是副本，而是底层内存的一个切片，它表示嵌套张量的第一个条目或组件。

#### 与填充张量之间的转换

``torch.nested.to_padded_tensor()`` 将 NJT 转换为具有指定填充值的填充密集张量。不规则维度将被填充到最大序列长度的大小。

```
>>> import torch
>>> a = torch.randn(2, 3)
>>> b = torch.randn(6, 3)
>>> nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)
>>> padded = torch.nested.to_padded_tensor(nt, padding=4.2)
>>> padded
tensor([[[ 1.6107,  0.5723,  0.3913],
         [ 0.0700, -0.4954,  1.8663],
         [ 4.2000,  4.2000,  4.2000],
         [ 4.2000,  4.2000,  4.2000],
         [ 4.2000,  4.2000,  4.2000],
         [ 4.2000,  4.2000,  4.2000]],
        [[-0.0479, -0.7610, -0.3484],
         [ 1.1345,  1.0556,  0.3634],
         [-1.7122, -0.5921,  0.0540],
         [-0.5506,  0.7608,  2.0606],
         [ 1.5658, -1.1934,  0.3041],
         [ 0.1483, -1.1284,  0.6957]]])
```

这可以作为绕过 NJT 支持不足的应急方案，但理想情况下应尽可能避免此类转换，以实现最佳内存使用和性能，因为更高效的嵌套张量布局不会物化填充。

反向转换可以通过 ``torch.nested.narrow()`` 实现，该函数将不规则结构应用于给定的稠密张量以生成 NJT。请注意，默认情况下此操作不会复制底层数据，因此输出的 NJT 通常是非连续的。如果需要连续的 NJT，在此处显式调用 ``contiguous()`` 可能很有用。

```
>>> padded = torch.randn(3, 5, 4)
>>> seq_lens = torch.tensor([3, 2, 5], dtype=torch.int64)
>>> nt = torch.nested.narrow(padded, dim=1, length=seq_lens, layout=torch.jagged)
>>> nt.shape
torch.Size([3, j1, 4])
>>> nt = nt.contiguous()
>>> nt.shape
torch.Size([3, j2, 4])
```

### 形状操作

嵌套张量支持广泛的形状操作，包括视图操作。

```
>>> a = torch.randn(2, 6)
>>> b = torch.randn(4, 6)
>>> nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)
>>> nt.shape
torch.Size([2, j1, 6])
>>> nt.unsqueeze(-1).shape
torch.Size([2, j1, 6, 1])
>>> nt.unflatten(-1, [2, 3]).shape
torch.Size([2, j1, 2, 3])
>>> torch.cat([nt, nt], dim=2).shape
torch.Size([2, j1, 12])
>>> torch.stack([nt, nt], dim=2).shape
torch.Size([2, j1, 2, 6])
>>> nt.transpose(-1, -2).shape
torch.Size([2, 6, j1])
```

### 注意力机制

由于可变长度序列是注意力机制的常见输入，嵌套张量支持重要的注意力算子
[Scaled Dot Product Attention (SDPA)](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 和
[FlexAttention](https://pytorch.org/docs/stable/nn.attention.flex_attention.html#module-torch.nn.attention.flex_attention)。
有关 NJT 与 SDPA 的使用示例，请参见
[此处](https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html#multiheadattention)；
有关 NJT 与 FlexAttention 的使用示例，请参见
[此处](https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html#flexattention-njt)。


## 与 torch.compile 配合使用

NJT 设计用于与 ``torch.compile()`` 配合以实现最佳性能，我们始终建议尽可能将 ``torch.compile()`` 与 NJT 结合使用。无论 NJT 是作为编译函数或模块的输入传递，还是在函数内部内联实例化，NJT 都能开箱即用且无需图中断。

```{note}
    如果您的用例无法使用 ``torch.compile()``，使用 NJT 可能仍能带来性能和内存使用上的好处，但这种情况并不绝对。重要的是，所操作的张量要足够大，以确保性能提升不会被 Python 张量子类的开销所抵消。
```

```
>>> import torch
>>> a = torch.randn(2, 3)
>>> b = torch.randn(4, 3)
>>> nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)
>>> def f(x): return x.sin() + 1
...
>>> compiled_f = torch.compile(f, fullgraph=True)
>>> output = compiled_f(nt)
>>> output.shape
torch.Size([2, j1, 3])
>>> def g(values, offsets): return torch.nested.nested_tensor_from_jagged(values, offsets) * 2.
...
>>> compiled_g = torch.compile(g, fullgraph=True)
>>> output2 = compiled_g(nt.values(), nt.offsets())
>>> output2.shape
torch.Size([2, j1, 3])
```

请注意，NJT 支持
[动态形状](https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)，
以避免因不规则结构变化而导致不必要的重新编译。

```
>>> a = torch.randn(2, 3)
>>> b = torch.randn(4, 3)
>>> c = torch.randn(5, 3)
>>> d = torch.randn(6, 3)
>>> nt1 = torch.nested.nested_tensor([a, b], layout=torch.jagged)
>>> nt2 = torch.nested.nested_tensor([c, d], layout=torch.jagged)
>>> def f(x): return x.sin() + 1
...
>>> compiled_f = torch.compile(f, fullgraph=True)
>>> output1 = compiled_f(nt1)
>>> output2 = compiled_f(nt2)  # 注意：即使不规则结构不同，也无需重新编译
```

如果在使用 NJT + ``torch.compile`` 时遇到问题或晦涩的错误，请提交 PyTorch issue。在 ``torch.compile`` 中完全支持子类是一项长期工作，目前可能仍存在一些不完善之处。


## 故障排除

本节包含使用嵌套张量时可能遇到的常见错误，以及这些错误的原因和解决建议。


### 未实现的操作

随着嵌套张量操作支持的增加，此错误已越来越少见，但考虑到 PyTorch 中有数千个操作，目前仍有可能遇到。

```
    NotImplementedError: aten.view_as_real.default
```

错误信息很直接；我们尚未为这个特定操作添加支持。如果您愿意，可以自行[贡献](contributions)实现，或者直接[请求](https://github.com/pytorch/pytorch/issues/118107)我们在未来的 PyTorch 版本中添加对此操作的支持。


### 不规则结构不兼容

```
    RuntimeError: cannot call binary pointwise function add.Tensor with inputs of shapes (2, j2, 128) and (2, j3, 128)
```

当调用一个操作多个 NJT 且这些 NJT 具有不兼容的不规则结构的操作时，会发生此错误。目前，要求输入的 NJT 具有完全相同的 ``offsets`` 组成部分，才能拥有相同的符号化不规则结构符号（例如 ``j1``）。

针对这种情况，一种变通方法是直接从 ``values`` 和 ``offsets`` 组件构造 NJT。当两个 NJT 引用相同的 ``offsets`` 组件时，它们被视为具有相同的不规则结构，因此是兼容的。

>>> a = torch.randn(50, 128)
>>> b = torch.randn(32, 128)
>>> nt1 = torch.nested.nested_tensor([a, b], layout=torch.jagged, dtype=torch.float32)
>>> nt2 = torch.nested.nested_tensor_from_jagged(values=torch.randn(82, 128), offsets=nt1.offsets())
>>> nt3 = nt1 + nt2
>>> nt3.shape
torch.Size([2, j1, 128])

### torch.compile 中的数据依赖操作

```
    torch._dynamo.exc.Unsupported: data dependent operator: aten._local_scalar_dense.default; to enable, set torch._dynamo.config.capture_scalar_outputs = True
```

当在 `torch.compile` 内部调用执行数据依赖操作的操作符时，会发生此错误；这通常发生在需要检查 NJT 的 `offsets` 值以确定输出形状的操作中。例如：

```
>>> a = torch.randn(50, 128)
>>> b = torch.randn(32, 128)
>>> nt = torch.nested.nested_tensor([a, b], layout=torch.jagged, dtype=torch.float32)
>>> def f(nt): return nt.chunk(2, dim=0)[0]
...
>>> compiled_f = torch.compile(f, fullgraph=True)
>>> output = compiled_f(nt)
```

在此示例中，在 NJT 的批次维度上调用 `chunk()` 需要检查 NJT 的 `offsets` 数据，以在打包的参差维度内划定批次项的边界。作为一种变通方法，可以设置几个 `torch.compile` 标志：

```
>>> torch._dynamo.config.capture_dynamic_output_shape_ops = True
>>> torch._dynamo.config.capture_scalar_outputs = True
```

如果在设置这些标志后，您仍然看到数据依赖操作符错误，请向 PyTorch 提交问题。`torch.compile()` 的这一领域仍在大力开发中，NJT 支持的某些方面可能还不完整。


## 贡献

如果您想为嵌套张量开发做出贡献，最有影响力的方式之一是为当前不支持的 PyTorch 操作符添加嵌套张量支持。这个过程通常包括几个简单的步骤：

1.  确定要添加的操作符名称；这应该类似于 `aten.view_as_real.default`。可以在 `aten/src/ATen/native/native_functions.yaml` 中找到此操作符的签名。
2.  按照 `torch/nested/_internal/ops.py` 中为其他操作符建立的模式，在其中注册一个操作符实现。使用来自 `native_functions.yaml` 的签名进行模式验证。

实现操作符最常见的方式是将 NJT 解包为其组成部分，在底层的 `values` 缓冲区上重新分派该操作符，并将相关的 NJT 元数据（包括 `offsets`）传播到一个新的输出 NJT。如果操作符的预期输出形状与输入不同，则必须计算新的 `offsets` 等元数据。

当操作符应用于批次或参差维度时，这些技巧可以帮助快速获得一个可行的实现：

*   对于*非批次维度*的操作，基于 `unbind()` 的回退应该有效。
*   对于参差维度上的操作，可以考虑使用一个不会对输出产生负面偏差的、经过适当选择的填充值，将其转换为填充密集张量，运行操作符，然后再转换回 NJT。在 `torch.compile` 内部，这些转换可以被融合以避免具体化填充的中间张量。


## 构造和转换函数的详细文档


