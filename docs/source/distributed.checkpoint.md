# 分布式检查点 - torch.distributed.checkpoint

分布式检查点（DCP）支持从多个 rank 并行加载和保存模型。它处理加载时的重分片，使得可以在一种集群拓扑中保存，并在另一种拓扑中加载。

DCP 在几个重要方面与 `torch.save` 和 `torch.load` 不同：

- 每个检查点会生成多个文件，每个 rank 至少一个。
- 它就地操作，意味着模型应首先分配其数据，然后 DCP 使用该存储空间。

加载和保存检查点的入口点如下：

## 其他资源：

- [分布式检查点（DCP）入门指南](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)
- [使用分布式检查点（DCP）进行异步保存](https://pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)
- [TorchTitan 检查点文档](https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md)
- [TorchTitan DCP 实现](https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/checkpoint.py)


以下模块对于异步检查点（`torch.distributed.checkpoint.async_save`）使用的暂存机制进行额外自定义也很有用：


除了上述入口点，如下所述的 `Stateful` 对象在保存/加载期间提供了额外的自定义功能。


这个[示例](https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/examples/fsdp_checkpoint_example.py)展示了如何使用 PyTorch 分布式检查点来保存一个 FSDP 模型。

以下类型定义了检查点期间使用的 IO 接口：


以下类型定义了检查点期间使用的规划器接口：


我们提供了一个基于文件系统的存储层：


我们还提供了其他存储层，包括与 HuggingFace safetensors 交互的存储层：


我们提供了 `LoadPlanner` 和 `SavePlanner` 的默认实现，可以处理所有 torch.distributed 结构，如 FSDP、DDP、ShardedTensor 和 DistributedTensor。


由于历史设计决策，即使原始未并行化的模型相同，`FSDP` 和 `DDP` 的状态字典也可能具有不同的键或完全限定名称（例如，layer1.weight）。此外，`FSDP` 提供了各种类型的模型状态字典，例如完整状态字典和分片状态字典。另外，优化器状态字典使用参数 ID 而不是完全限定名称来标识参数，这在使用并行化（例如流水线并行）时可能会引起问题。

为了解决这些挑战，我们提供了一组 API，使用户能够轻松管理状态字典。`get_model_state_dict()` 返回一个模型状态字典，其键与未并行化模型状态字典返回的键保持一致。类似地，`get_optimizer_state_dict()` 提供了优化器状态字典，其键在所有应用的并行化中保持一致。为了实现这种一致性，`get_optimizer_state_dict()` 将参数 ID 转换为与未并行化模型状态字典中相同的完全限定名称。

请注意，这些 API 返回的结果可以直接与 `torch.distributed.checkpoint.save()` 和 `torch.distributed.checkpoint.load()` 方法配合使用，无需任何额外转换。

提供了 `set_model_state_dict()` 和 `set_optimizer_state_dict()` 来加载由它们各自的 getter API 生成的模型和优化器状态字典。

请注意，`set_optimizer_state_dict()` 只能在优化器调用 `backward()` 之前或调用 `step()` 之后调用。

请注意，此功能是实验性的，API 签名将来可能会更改。


对于习惯于使用和共享 `torch.save` 格式模型的用户，提供了以下方法，用于在格式之间进行转换的离线实用程序。


以下类也可用于从 torch.save 格式在线加载模型并进行重分片。


以下实验性接口旨在提高生产环境中的可观测性：
