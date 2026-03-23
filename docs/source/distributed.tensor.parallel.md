:::{role} hidden
    :class: hidden-section
:::

# Tensor Parallelism - torch.distributed.tensor.parallel

Tensor Parallelism (TP) 构建在 PyTorch DistributedTensor ([DTensor](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/README.md)) 之上，并提供不同的并行风格：列并行、行并行和序列并行。

:::{warning}
Tensor Parallelism API 是实验性的，可能会发生变化。
:::

使用 Tensor Parallelism 并行化你的 `nn.Module` 的入口点是：

```{eval-rst}
.. automodule:: torch.distributed.tensor.parallel
```

```{eval-rst}
.. currentmodule:: torch.distributed.tensor.parallel
```

```{eval-rst}
.. autofunction::  parallelize_module
```

Tensor Parallelism 支持以下并行风格：

```{eval-rst}
.. autoclass:: torch.distributed.tensor.parallel.ColwiseParallel
  :members:
  :undoc-members:
```

```{eval-rst}
.. autoclass:: torch.distributed.tensor.parallel.RowwiseParallel
  :members:
  :undoc-members:
```

```{eval-rst}
.. autoclass:: torch.distributed.tensor.parallel.SequenceParallel
  :members:
  :undoc-members:
```

为了简单地使用 DTensor 布局配置 nn.Module 的输入和输出，并执行必要的布局重分布，而不将模块参数分布到 DTensor，可以在调用 `parallelize_module` 时，在 `parallelize_plan` 中使用以下 `ParallelStyle`：

```{eval-rst}
.. autoclass:: torch.distributed.tensor.parallel.PrepareModuleInput
  :members:
  :undoc-members:
```

```{eval-rst}
.. autoclass:: torch.distributed.tensor.parallel.PrepareModuleOutput
  :members:
  :undoc-members:
```

```{eval-rst}
.. autoclass:: torch.distributed.tensor.parallel.PrepareModuleInputOutput
  :members:
  :undoc-members:
```

:::{note}
当使用 `Shard(dim)` 作为上述 `ParallelStyle` 的输入/输出布局时，我们假设输入/输出激活张量在 TP 操作的 `DeviceMesh` 上的张量维度 `dim` 上是均匀分片的。例如，由于 `RowwiseParallel` 接受在最后一个维度上分片的输入，它假设输入张量已经在最后一个维度上均匀分片。对于激活张量非均匀分片的情况，可以直接将 DTensor 传递给分区后的模块，并使用 `use_local_output=False` 在每个 `ParallelStyle` 后返回 DTensor，其中 DTensor 可以跟踪非均匀分片信息。
:::

对于像 Transformer 这样的模型，我们建议用户在 `parallelize_plan` 中同时使用 `ColwiseParallel` 和 `RowwiseParallel`，以便为整个模型（即 Attention 和 MLP）实现所需的分片。

通过以下上下文管理器支持并行化的交叉熵损失计算（损失并行）：

```{eval-rst}
.. autofunction:: torch.distributed.tensor.parallel.loss_parallel
```
:::{warning}
    loss_parallel API 是实验性的，可能会发生变化。
:::