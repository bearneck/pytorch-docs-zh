# 分布式数据并行 {#ddp}

 warning
 title
Warning


`torch.nn.parallel.DistributedDataParallel`{.interpreted-text role="class"} 的实现会随时间演进。本设计说明基于 v1.4 版本的状态编写。


`torch.nn.parallel.DistributedDataParallel`{.interpreted-text role="class"} (DDP) 透明地执行分布式数据并行训练。本页描述了其工作原理并揭示了实现细节。

## 示例

让我们从一个简单的 `torch.nn.parallel.DistributedDataParallel`{.interpreted-text role="class"} 示例开始。该示例使用 `torch.nn.Linear`{.interpreted-text role="class"} 作为本地模型，用 DDP 包装它，然后在 DDP 模型上运行一次前向传播、一次反向传播和一个优化器步骤。之后，本地模型上的参数将被更新，不同进程上的所有模型应该完全相同。

``` 
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP


def example(rank, world_size):
    # 创建默认进程组
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # 创建本地模型
    model = nn.Linear(10, 10).to(rank)
    # 构建 DDP 模型
    ddp_model = DDP(model, device_ids=[rank])
    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # 前向传播
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # 反向传播
    loss_fn(outputs, labels).backward()
    # 更新参数
    optimizer.step()

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # 使用 c10d 默认 "env" 初始化模式时需要设置的环境变量。
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
```

DDP 可与 TorchDynamo 协同工作。当与 TorchDynamo 一起使用时，应在编译模型之前应用 DDP 模型包装器，以便 torchdynamo 可以基于 DDP 桶大小应用 `DDPOptimizer`（图中断优化）。（更多信息请参阅 [TorchDynamo DDPOptimizer](./ddp.html#torchdynamo-ddpoptimizer)。）

``` 
ddp_model = DDP(model, device_ids=[rank])
ddp_model = torch.compile(ddp_model)
```

## 内部设计

本节通过深入探讨一次迭代中每个步骤的细节，揭示 `torch.nn.parallel.DistributedDataParallel`{.interpreted-text role="class"} 在底层的工作原理。

- **前提条件**: DDP 依赖于 c10d `ProcessGroup` 进行通信。因此，应用程序必须在构建 DDP 之前创建 `ProcessGroup` 实例。
- **构建**: DDP 构造函数接收对本地模块的引用，并从 rank 0 进程向组内所有其他进程广播 `state_dict()`，以确保所有模型副本从完全相同的状态开始。然后，每个 DDP 进程创建一个本地 `Reducer`，该 `Reducer` 稍后将在反向传播过程中负责梯度同步。为了提高通信效率，`Reducer` 将参数梯度组织到桶中，并一次减少一个桶。桶大小可以通过在 DDP 构造函数中设置 [bucket_cap_mb]{.title-ref} 参数来配置。参数梯度到桶的映射是在构建时根据桶大小限制和参数大小确定的。模型参数按照给定模型的 `Model.parameters()` 的（大致）逆序分配到桶中。使用逆序的原因是 DDP 期望梯度在反向传播过程中大致按照该顺序准备就绪。下图展示了一个示例。注意，`grad0` 和 `grad1` 在 `bucket1` 中，而另外两个梯度在 `bucket0` 中。当然，这个假设可能并不总是成立，当这种情况发生时，可能会损害 DDP 的反向传播速度，因为 `Reducer` 无法在最早可能的时间启动通信。除了分桶之外，`Reducer` 在构建期间还会注册 autograd 钩子，每个参数一个钩子。这些钩子将在反向传播过程中梯度准备就绪时触发。
- **前向传播**: DDP 接收输入并将其传递给本地模型，然后如果 `find_unused_parameters` 设置为 `True`，则分析本地模型的输出。此模式允许在模型的子图上运行反向传播，DDP 通过从模型输出遍历 autograd 图并标记所有未使用的参数为已准备好进行归约，从而找出哪些参数参与了反向传播。在反向传播过程中，`Reducer` 只会等待未就绪的参数，但它仍会归约所有桶。目前，将参数梯度标记为就绪并不能帮助 DDP 跳过桶，但它可以防止 DDP 在反向传播过程中永远等待不存在的梯度。请注意，遍历 autograd 图会引入额外的开销，因此应用程序应仅在必要时将 `find_unused_parameters` 设置为 `True`。
- **反向传播**: `backward()` 函数直接在损失 `Tensor` 上调用，这超出了 DDP 的控制范围，DDP 使用构建时注册的 autograd 钩子来触发梯度同步。当一个梯度准备就绪时，其对应的 DDP 钩子将在该梯度累加器上触发，然后 DDP 将该参数梯度标记为已准备好进行归约。当一个桶中的所有梯度都准备就绪时，`Reducer` 在该桶上启动异步 `allreduce` 以计算所有进程中梯度的平均值。当所有桶都准备就绪时，`Reducer` 将阻塞等待所有 `allreduce` 操作完成。完成后，平均梯度将写入所有参数的 `param.grad` 字段。因此，在反向传播之后，不同 DDP 进程中相同对应参数上的 [grad]{.title-ref} 字段应该是相同的。
- **优化器步骤**: 从优化器的角度来看，它正在优化一个本地模型。所有 DDP 进程上的模型副本可以保持同步，因为它们都从相同的状态开始，并且在每次迭代中都有相同的平均梯度。

![ddp_grad_sync.png](https://user-images.githubusercontent.com/16999635/72401724-d296d880-371a-11ea-90ab-737f86543df9.png){width="700px"}

 note
 title
Note


DDP 要求所有进程上的 `Reducer` 实例以完全相同的顺序调用 `allreduce`，这是通过始终按照桶索引顺序而不是实际桶就绪顺序运行 `allreduce` 来实现的。进程间不匹配的 `allreduce` 顺序可能导致错误结果或 DDP 反向传播挂起。


## 实现

以下是指向 DDP 实现组件的指针。堆叠图展示了代码的结构。

### ProcessGroup

- [ProcessGroup.hpp](https://github.com/pytorch/pytorch/blob/v1.7.0/torch/lib/c10d/ProcessGroup.hpp): 包含所有进程组实现的抽象 API。`c10d` 库开箱即用地提供了 3 种实现，即 [ProcessGroupGloo]{.title-ref}、\`ProcessGroupNCCL\` 和 [ProcessGroupMPI]{.title-ref}。 `DistributedDataParallel` 在初始化期间使用 `ProcessGroup::broadcast()` 将模型状态从 rank 0 进程发送到其他进程，并使用 `ProcessGroup::allreduce()` 对梯度求和。
- [Store.hpp](https://github.com/pytorch/pytorch/blob/v1.7.0/torch/lib/c10d/Store.hpp): 协助进程组实例的 rendezvous 服务相互发现。

### DistributedDataParallel

- [distributed.py](https://github.com/pytorch/pytorch/blob/v1.7.0/torch/nn/parallel/distributed.py): 是 DDP 的 Python 入口点。它实现了 `nn.parallel.DistributedDataParallel` 模块的初始化步骤和 `forward` 函数，这些函数会调用 C++ 库。其 `_sync_param` 函数在单个 DDP 进程处理多个设备时执行进程内参数同步，并且还会将模型缓冲区从 rank 0 进程广播到所有其他进程。进程间参数同步发生在 `Reducer.cpp` 中。
- [comm.h](https://github.com/pytorch/pytorch/blob/v1.7.0/torch/csrc/distributed/c10d/comm.h): 实现了聚合广播辅助函数，该函数在初始化期间被调用来广播模型状态，并在前向传播之前同步模型缓冲区。
- [reducer.h](https://github.com/pytorch/pytorch/blob/v1.7.0/torch/csrc/distributed/c10d/reducer.h): 为反向传播中的梯度同步提供了核心实现。它有三个入口点函数：
  - `Reducer`: 构造函数在 `distributed.py` 中被调用，它将 `Reducer::autograd_hook()` 注册到梯度累加器。
  - `autograd_hook()` 函数会在梯度准备就绪时被 autograd 引擎调用。
  - `prepare_for_backward()` 在 DDP 前向传播结束时于 `distributed.py` 中被调用。当 DDP 构造函数中 `find_unused_parameters` 设置为 `True` 时，它会遍历 autograd 图以查找未使用的参数。

![ddp_code.png](https://user-images.githubusercontent.com/16999635/72313120-4e7c1c80-3658-11ea-9c6d-44336b2daeac.png){width="400px"}

### TorchDynamo DDPOptimizer

DDP 的性能优势来自于在反向传播期间将 allreduce 集合操作与计算重叠。当与 TorchDynamo 一起用于编译整个前向和整个反向图时，AotAutograd 会阻止这种重叠，因为 allreduce 操作是在整个优化的反向计算 [完成之后]{#完成之后}\_ 由 autograd 钩子启动的。

TorchDynamo 的 DDPOptimizer 通过在反向传播期间，在 DDP 的 allreduce 桶的逻辑边界处打断前向图来提供帮助。注意：目标是在反向传播期间打断图，最简单的实现是打断前向图，然后对每个部分调用 AotAutograd 并进行编译。这使得 DDP 的 allreduce 钩子可以在反向传播的各个部分之间触发，并安排通信以与计算重叠。

有关更深入的解释和实验结果，请参阅 [此博客文章](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860/1)，或阅读 [torch/\_dynamo/optimizations/distributed.py](https://github.com/pytorch/pytorch/blob/bbc39b7bb48d28d67e3253a89cc82df3687ddd1b/torch/_dynamo/backends/distributed.py#L124) 中的文档和代码。

要调试 DDPOptimizer，请设置 [TORCH_LOGS=\'ddp_graphs\']{.title-ref} 以获取完整的图转储。对于不带图的日志，请将 \'dynamo\'、\'distributed\' 或 \'dist_ddp\' 中的任意一个添加到 [TORCH_LOGS]{.title-ref}（以获取关于桶边界的基本信息）。要禁用 DDPOptimizer，请设置 [torch.\_dynamo.config.optimize_ddp=False]{.title-ref}。即使没有 DDPOptimizer，DDP 和 TorchDynamo 仍应能正常工作，但性能会下降。
