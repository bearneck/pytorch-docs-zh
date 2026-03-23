> **CURRENTMODULE**：torch.distributed.tensor


# torch.distributed.tensor


> 📝 **注意**
> `torch.distributed.tensor` 目前处于 alpha 状态且正在开发中，我们承诺文档中列出的大多数 API 保持向后兼容，但必要时可能会有 API 变更。


## PyTorch DTensor（分布式张量）

PyTorch DTensor 提供了简单灵活的张量分片原语，能够透明地处理分布式逻辑，包括跨设备/主机的分片存储、算子计算和集合通信。`DTensor` 可用于构建不同的并行解决方案，并在处理多维分片时支持分片状态字典表示。

请参阅基于 `DTensor` 构建的 PyTorch 原生并行解决方案示例：

- [Tensor Parallel](https://pytorch.org/docs/main/distributed.tensor.parallel.html)
- [FSDP2](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)


`DTensor` 遵循 SPMD（单程序多数据）编程模型，使用户能够像编写**具有相同收敛特性的单设备程序**一样编写分布式程序。它通过指定 `DeviceMesh` 和 `Placement` 来提供统一的张量分片布局（DTensor 布局）：

- `DeviceMesh` 使用 n 维数组表示集群的设备拓扑和通信器。
- `Placement` 描述逻辑张量在 `DeviceMesh` 上的分片布局。DTensor 支持三种放置类型：`Shard`、`Replicate` 和 `Partial`。

### DTensor 类 API


`DTensor` 是 `torch.Tensor` 的子类。这意味着一旦创建了 `DTensor`，就可以以与 `torch.Tensor` 非常相似的方式使用它，包括运行不同类型的 PyTorch 算子，就像在单设备上运行一样，从而为 PyTorch 算子提供适当的分布式计算。

除了现有的 `torch.Tensor` 方法外，它还提供了一组额外的方法来与 `torch.Tensor` 交互、将 DTensor 布局 `redistribute` 到新的 DTensor、在所有设备上获取完整的张量内容等。


### DeviceMesh 作为分布式通信器


`DeviceMesh` 由 DTensor 构建，作为描述集群设备拓扑并表示多维通信器（基于 `ProcessGroup`）的抽象。要了解如何创建/使用 DeviceMesh 的详细信息，请参阅 [DeviceMesh 教程](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)。

### DTensor 放置类型


DTensor 在每个 `DeviceMesh` 维度上支持以下类型的 `Placement`：


## 创建 DTensor 的不同方式


有三种方式可以构造 `DTensor`：
: - `distribute_tensor` 在每个 rank 上从逻辑或“全局” `torch.Tensor` 创建 `DTensor`。这可用于分片叶子 `torch.Tensor`（即模型参数/缓冲区和输入）。
  - `DTensor.from_local` 在每个 rank 上从本地 `torch.Tensor` 创建 `DTensor`，可用于从非叶子 `torch.Tensor`（即前向/反向传播期间的中间激活张量）创建 `DTensor`。
  - DTensor 提供了专门的张量工厂函数（例如 `empty`、`ones`、`randn` 等），允许通过直接指定 `DeviceMesh` 和 `Placement` 来创建不同的 `DTensor`。与 `distribute_tensor` 相比，这可以直接在设备上实例化分片内存，而不是在初始化逻辑张量内存后再执行分片。

### 从逻辑 torch.Tensor 创建 DTensor

`torch.distributed` 中的 SPMD（单程序多数据）编程模型启动多个进程（例如通过 `torchrun`）来执行相同的程序，这意味着程序内部的模型会首先在不同的进程上初始化（即模型可能在 CPU、元设备上初始化，或者如果有足够内存则直接在 GPU 上初始化）。

`DTensor` 提供了 `distribute_tensor` API，可以将模型权重或张量分片为 `DTensor`，它会在每个进程上从“逻辑”张量创建 DTensor。这将使创建的 `DTensor` 遵循单设备语义，这对于**数值正确性**至关重要。


除了 `distribute_tensor`，DTensor 还提供了 `distribute_module` API，以便在 `nn.Module` 级别更轻松地进行分片。


### DTensor 工厂函数

DTensor 还提供了专门的张量工厂函数，允许使用类似 torch.Tensor 的工厂函数 API（即 torch.ones、torch.empty 等）直接创建 `DTensor`，只需额外为创建的 `DTensor` 指定 `DeviceMesh` 和 `Placement`：


### 随机操作

DTensor 提供了分布式随机数生成器功能，以确保在分片张量上的随机操作获得唯一值，而在副本张量上的随机操作获得相同的值。该系统要求所有参与的进程（例如 SPMD 进程）在执行每个 dtensor 随机操作之前，都从相同的生成器状态开始。如果满足此条件，它能确保在每个 dtensor 随机操作完成后，所有进程都达到相同的状态。在随机操作期间，不会执行通信来同步 RNG 状态。

接受 `generator` 关键字的操作符将使用用户传入的生成器（如果传入），否则使用设备的默认生成器。无论使用哪个生成器，在 DTensor 操作后它都会被推进。将同一个生成器同时用于 DTensor 和非 DTensor 操作是有效的，但必须注意确保非 DTensor 操作在所有进程上同等推进生成器状态。

当将 DTensor 与流水线并行一起使用时，每个流水线阶段的进程应使用不同的种子，而同一流水线阶段内的进程应使用相同的种子。

DTensor 的 RNG 基础设施基于 philox 的 RNG 算法，并支持任何基于 philox 的后端（cuda 和其他类似 cuda 的设备），但不幸的是目前还不支持 CPU 后端。

## 调试


### 日志记录

启动程序时，您可以使用来自 [torch._logging](https://pytorch.org/docs/main/logging.html#module-torch._logging) 的 `TORCH_LOGS` 环境变量来开启额外的日志记录：

- `TORCH_LOGS=+dtensor` 将显示 `logging.DEBUG` 及以上级别的消息。
- `TORCH_LOGS=dtensor` 将显示 `logging.INFO` 及以上级别的消息。
- `TORCH_LOGS=-dtensor` 将显示 `logging.WARNING` 及以上级别的消息。

### 调试工具

为了调试应用了 DTensor 的程序，并更详细地了解底层发生了哪些集合通信操作，DTensor 提供了 `CommDebugMode`：


为了可视化维度少于 3 的 DTensor 的分片情况，DTensor 提供了 `visualize_sharding`：


## 实验性功能

`DTensor` 还提供了一组实验性功能。这些功能要么处于原型阶段，要么基本功能已完成但正在寻求用户反馈。如果您对这些功能有任何反馈，请向 PyTorch 提交一个问题。


% 缺少文档的模块，必要时稍后添加文档


## 混合 Tensor 和 DTensor 操作

所以你遇到了以下错误信息。
```
got mixed torch.Tensor and DTensor, need to convert all
torch.Tensor to DTensor before calling distributed operators!
```

有两种情况。

### 情况 1：这是用户错误

遇到此错误最常见的方式是创建一个常规张量（使用工厂函数），然后执行 Tensor-DTensor 操作，如下所示：

```
tensor = torch.arange(10)
return tensor + dtensor
```

我们不允许混合的 Tensor-DTensor 操作：如果任何操作（例如 torch.add）的输入是 DTensor，那么所有 Tensor 输入也必须是 DTensor。这是因为语义不明确。我们不知道 `tensor` 在各个进程上是相同的还是不同的，因此我们要求用户弄清楚如何从 `tensor` 构造一个具有准确放置位置的 DTensor。

如果每个进程确实拥有相同的 `tensor`，那么请构造一个副本 DTensor：

```
tensor = torch.arange(10)
tensor = DTensor.from_local(tensor, placements=(Replicate(),))
return tensor + dtensor
```

如果你想创建一个带有分片的 DTensor，以下是操作方法。从语义上讲，这意味着你的张量数据在分片之间分割，并且操作作用于“完整堆叠的数据”。

```
tensor = torch.full([], RANK)
tensor = DTensor.from_local(tensor, placements=(Shard(0),))
return tensor + dtensor
```

除了这些情况，你可能还想对你的张量做其他事情（这些并不是仅有的两个选项！）。

## 情况 2：错误来自 PyTorch 框架代码

有时问题是 PyTorch 框架代码试图执行混合的 Tensor-DTensor 操作。这些是 PyTorch 中的错误，请提交一个问题以便我们修复它们。

在用户方面，你唯一能做的就是避免使用导致问题的操作，并提交错误报告。

对于 PyTorch 开发者：修复此问题的一种方法是重写 PyTorch 框架代码以避免混合的 Tensor-DTensor 代码（如上一节所述）。

对于 PyTorch 开发者：第二种方法是在 PyTorch 框架代码的适当位置开启 DTensor 隐式复制。开启后，任何混合的 Tensor-DTensor 操作都将假定非 DTensor 可以被复制。使用时请小心，因为这可能导致静默错误。

- [在 Python 中开启隐式复制](https://github.com/pytorch/pytorch/blob/d8e6b2fddc54c748d976e8f0ebe4b63ebe36d85b/torch/distributed/tensor/experimental/__init__.py#L15)
- [在 C++ 中开启隐式复制](https://github.com/pytorch/pytorch/blob/7a0f93344e2c851b9bcf2b9c3225a323d48fde26/aten/src/ATen/DTensorState.h#L10)
