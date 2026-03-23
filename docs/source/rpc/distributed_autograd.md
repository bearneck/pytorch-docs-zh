orphan

:   

# 分布式自动求导设计

本文档将详细介绍分布式自动求导的设计，并深入探讨其内部实现。在继续阅读之前，请确保您已熟悉 `autograd-mechanics` 和 `distributed-rpc-framework`。

## 背景

假设您有两个节点，并且一个非常简单的模型被划分在这两个节点上。这可以使用 `torch.distributed.rpc` 实现如下：

``` 
import torch
import torch.distributed.rpc as rpc

def my_add(t1, t2):
  return torch.add(t1, t2)

# 在 worker 0 上：
t1 = torch.rand((3, 3), requires_grad=True)
t2 = torch.rand((3, 3), requires_grad=True)

# 远程执行一些计算。
t3 = rpc.rpc_sync("worker1", my_add, args=(t1, t2))

# 基于远程结果本地执行一些计算。
t4 = torch.rand((3, 3), requires_grad=True)
t5 = torch.mul(t3, t4)

# 计算损失。
loss = t5.sum()
```

分布式自动求导的主要动机是能够在我们计算出的 `loss` 上对这样的分布式模型运行反向传播，并为所有需要梯度的张量记录相应的梯度。

## 前向传播期间的自动求导记录

PyTorch 在前向传播期间构建自动求导图，该图用于执行反向传播。更多细节请参见 `how-autograd-encodes-history`。

对于分布式自动求导，我们需要在前向传播期间跟踪所有 RPC 调用，以确保反向传播正确执行。为此，我们在执行 RPC 时，将 `send` 和 `recv` 函数附加到自动求导图中。

- `send` 函数附加在 RPC 的源节点上，其输出边指向 RPC 输入张量的自动求导函数。该函数在反向传播期间的输入是从目标节点接收到的，作为相应 `recv` 函数的输出。
- `recv` 函数附加在 RPC 的目标节点上，其输入是从使用输入张量在目标节点上执行的算子中获取的。该函数的输出梯度在反向传播期间被发送到源节点的相应 `send` 函数。
- 每个 `send-recv` 对被分配一个全局唯一的 `autograd_message_id`，以唯一标识该对。这在反向传播期间查找远程节点上的对应函数时非常有用。
- 对于 `rref`，每当我们调用 `torch.distributed.rpc.RRef.to_here` 时，我们都会为涉及的张量附加一个适当的 `send-recv` 对。

例如，我们上面示例的自动求导图将如下所示（为简化起见，省略了 t5.sum()）：

![image](../_static/img/distributed_autograd/send_recv_functions.png)

## 分布式自动求导上下文

每个使用分布式自动求导的前向和反向传播都被分配一个唯一的 `torch.distributed.autograd.context`，并且这个上下文有一个全局唯一的 `autograd_context_id`。此上下文会根据需要在每个节点上创建。

此上下文服务于以下目的：

1.  运行分布式反向传播的多个节点可能会在同一张量上累积梯度，因此在我们有机会运行优化器之前，张量的 `.grad` 字段将包含来自各种分布式反向传播的梯度。这类似于在本地多次调用 `torch.autograd.backward`。为了提供一种分离每次反向传播梯度的方法，梯度被累积在每次反向传播的 `torch.distributed.autograd.context` 中。
2.  在前向传播期间，我们将每次自动求导传播的 `send` 和 `recv` 函数存储在此上下文中。这确保我们持有自动求导图中适当节点的引用以保持其存活。除此之外，在反向传播期间更容易查找适当的 `send` 和 `recv` 函数。
3.  通常，我们还使用此上下文为每次分布式自动求导传播存储一些元数据。

| 

从用户的角度来看，自动求导上下文的设置如下：

``` 
import torch.distributed.autograd as dist_autograd
with dist_autograd.context() as context_id:
  loss = model.forward()
  dist_autograd.backward(context_id, loss)
```

需要注意的是，必须在分布式自动求导上下文管理器内调用模型的前向传播，因为需要一个有效的上下文来确保所有 `send` 和 `recv` 函数被正确存储，以便在所有参与节点上运行反向传播。

## 分布式反向传播

在本节中，我们将概述在分布式反向传播期间准确计算依赖关系所面临的挑战，并描述几种执行分布式反向传播的算法（及其权衡）。

### 计算依赖关系

考虑在单台机器上运行的以下代码

``` 
import torch
a = torch.rand((3, 3), requires_grad=True)
b = torch.rand((3, 3), requires_grad=True)
c = torch.rand((3, 3), requires_grad=True)
d = a + b
e = b * c
d.sum.().backward()
```

上述代码的自动求导图将如下所示：

![image](../_static/img/distributed_autograd/local_dependencies.png)

作为反向传播过程的第一步，autograd 引擎会计算 autograd 图中每个节点的依赖数量。这有助于 autograd 引擎了解图中节点何时可以执行。`add(1)` 和 `mul(0)` 括号中的数字表示依赖数量。如你所见，这意味着在反向传播过程中，`add` 节点需要 1 个输入，而 `mul` 节点不需要任何输入（换句话说，不需要执行）。本地 autograd 引擎通过从根节点（本例中为 `d`）开始遍历图来计算这些依赖。

autograd 图中某些节点在反向传播过程中可能不会被执行，这对分布式 autograd 提出了挑战。考虑以下使用 RPC 的代码片段。

``` 
import torch
import torch.distributed.rpc as rpc

a = torch.rand((3, 3), requires_grad=True)
b = torch.rand((3, 3), requires_grad=True)
c = torch.rand((3, 3), requires_grad=True)

d = rpc.rpc_sync("worker1", torch.add, args=(a, b))
e = rpc.rpc_sync("worker1", torch.mul, args=(b, c))
loss = d.sum()
```

上述代码对应的 autograd 图如下：

![image](../_static/img/distributed_autograd/distributed_dependencies.png)

计算这个分布式 autograd 图的依赖关系要困难得多，并且需要一些开销（无论是计算开销还是网络通信开销）。

对于性能敏感的应用，我们可以通过假设每个 `send` 和 `recv` 函数在反向传播过程中都是有效的（大多数应用不会执行未使用的 RPC）来避免大量开销。这简化了分布式 autograd 算法，并且效率更高，但代价是应用需要了解其局限性。该算法被称为 [FAST mode algorithm]()，将在下文详细描述。

在一般情况下，可能并非每个 `send` 和 `recv` 函数在反向传播过程中都是必需的。为了解决这个问题，我们提出了 [SMART mode algorithm]()，将在后续章节中描述。请注意，目前仅实现了 [FAST] 模式算法。

### FAST 模式算法

该算法的关键假设是，当我们运行反向传播时，每个 `send` 函数都具有 1 个依赖。换句话说，我们假设会通过 RPC 从另一个节点接收到梯度。

算法如下：

1.  我们从拥有反向传播根节点的工作节点开始（所有根节点必须是本地的）。
2.  查找当前 [Distributed Autograd Context]() 的所有 `send` 函数。
3.  从提供的根节点和我们检索到的所有 `send` 函数开始，在本地计算依赖关系。
4.  计算完依赖关系后，使用提供的根节点启动本地 autograd 引擎。
5.  当 autograd 引擎执行 `recv` 函数时，`recv` 函数会通过 RPC 将输入梯度发送到相应的工作节点。每个 `recv` 函数都知道目标工作节点的 ID，因为这在正向传播过程中已被记录。`recv` 函数还会将 `autograd_context_id` 和 `autograd_message_id` 发送到远程主机。
6.  当远程主机收到此请求时，我们使用 `autograd_context_id` 和 `autograd_message_id` 来查找相应的 `send` 函数。
7.  如果这是工作节点首次收到给定 `autograd_context_id` 的请求，它将按照上述第 1-3 点所述在本地计算依赖关系。
8.  在第 6 步中检索到的 `send` 函数随后会被放入该工作节点的本地 autograd 引擎的执行队列中。
9.  最后，我们不会将梯度累积在张量的 `.grad` 字段上，而是为每个 [Distributed Autograd Context]() 单独累积梯度。梯度存储在 `Dict[Tensor, Tensor]` 中，这基本上是一个从张量到其关联梯度的映射，可以使用 `torch.distributed.autograd.get_gradients` API 检索此映射。

| 

例如，使用分布式 autograd 的完整代码如下：

``` 
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc

def my_add(t1, t2):
  return torch.add(t1, t2)

# 在工作节点 0 上：

# 设置 autograd 上下文。参与分布式反向传播的计算必须位于分布式 autograd 上下文管理器内。
with dist_autograd.context() as context_id:
  t1 = torch.rand((3, 3), requires_grad=True)
  t2 = torch.rand((3, 3), requires_grad=True)

  # 远程执行一些计算。
  t3 = rpc.rpc_sync("worker1", my_add, args=(t1, t2))

  # 基于远程结果在本地执行一些计算。
  t4 = torch.rand((3, 3), requires_grad=True)
  t5 = torch.mul(t3, t4)

  # 计算损失。
  loss = t5.sum()

  # 运行反向传播。
  dist_autograd.backward(context_id, [loss])

  # 从上下文中检索梯度。
  dist_autograd.get_gradients(context_id)
```

包含依赖关系的分布式 autograd 图如下所示（为简化起见，省略了 t5.sum()）：

![image](../_static/img/distributed_autograd/distributed_dependencies_computed.png)

应用于上述示例的 [FAST mode algorithm]() 如下：

1.  在 `Worker 0` 上，我们从根节点 `loss` 和 `send1` 开始计算依赖关系。结果 `send1` 被标记为依赖 1，而 `Worker 0` 上的 `mul` 被标记为依赖 1。
2.  现在，我们在 `Worker 0` 上启动本地 autograd 引擎。首先执行 `mul` 函数，将其输出累积在 autograd 上下文中作为 `t4` 的梯度。然后，我们执行 `recv2`，它将梯度发送到 `Worker 1`。
3.  由于这是 `Worker 1` 第一次得知这个反向传播过程，它开始计算依赖关系，并适当地标记 `send2`、`add` 和 `recv1` 的依赖。
4.  接下来，我们将 `send2` 加入 `Worker 1` 的本地 autograd 引擎队列，该引擎随后执行 `add` 和 `recv1`。
5.  当 `recv1` 执行时，它将梯度发送到 `Worker 0`。
6.  由于 `Worker 0` 已经为此反向传播过程计算了依赖关系，它只是在本地将 `send1` 加入队列并执行。
7.  最后，`t1`、`t2` 和 `t4` 的梯度被累积在 [Distributed Autograd Context]() 中。

### SMART 模式算法

该算法的完整细节仍在开发中，但总体思路可以参考 [RFC](https://github.com/pytorch/pytorch/issues/23110) 中的 **Distributed Autograd Algorithm Smart mode** 部分。

## 分布式优化器

`torch.distributed.optim.DistributedOptimizer` 的操作如下：

1.  接收一个待优化的远程参数列表 (`torch.distributed.rpc.RRef`)。这些参数也可以是包装在本地 `RRef` 中的本地参数。
2.  接收一个 `torch.optim.Optimizer` 类作为本地优化器，在所有不同的 `RRef` 所有者上运行。
3.  分布式优化器在每个工作节点上创建本地 `Optimizer` 的实例，并持有指向它们的 `RRef`。
4.  当调用 `torch.distributed.optim.DistributedOptimizer.step` 时，分布式优化器使用 RPC 在相应的远程工作者上远程执行所有本地优化器。必须提供一个分布式 autograd 的 `context_id` 作为 `torch.distributed.optim.DistributedOptimizer.step` 的输入。本地优化器使用此 ID 来应用存储在对应上下文中的梯度。
5.  如果多个并发的分布式优化器在同一个工作者上更新相同的参数，这些更新会通过锁进行序列化。

## 简单的端到端示例

将所有内容整合起来，以下是一个使用分布式 autograd 和分布式优化器的简单端到端示例。如果代码保存在名为 \"dist_autograd_simple.py\" 的文件中，可以使用命令 `MASTER_ADDR="localhost" MASTER_PORT=29500 python dist_autograd_simple.py` 运行：

``` 
import torch
import torch.multiprocessing as mp
import torch.distributed.autograd as dist_autograd
from torch.distributed import rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer

def random_tensor():
    return torch.rand((3, 3), requires_grad=True)

def _run_process(rank, dst_rank, world_size):
    name = "worker{}".format(rank)
    dst_name = "worker{}".format(dst_rank)

    # 初始化 RPC。
    rpc.init_rpc(
        name=name,
        rank=rank,
        world_size=world_size
    )

    # 使用分布式 autograd 上下文。
    with dist_autograd.context() as context_id:
        # 前向传播（在远程节点上创建引用）。
        rref1 = rpc.remote(dst_name, random_tensor)
        rref2 = rpc.remote(dst_name, random_tensor)
        loss = rref1.to_here() + rref2.to_here()

        # 反向传播（运行分布式 autograd）。
        dist_autograd.backward(context_id, [loss.sum()])

        # 构建 DistributedOptimizer。
        dist_optim = DistributedOptimizer(
        optim.SGD,
        [rref1, rref2],
        lr=0.05,
        )

        # 运行分布式优化器步骤。
        dist_optim.step(context_id)

def run_process(rank, world_size):
    dst_rank = (rank + 1) % world_size
    _run_process(rank, dst_rank, world_size)
    rpc.shutdown()

if __name__ == '__main__':
  # 运行 world_size 个工作进程
  world_size = 2
  mp.spawn(run_process, args=(world_size,), nprocs=world_size)
```
