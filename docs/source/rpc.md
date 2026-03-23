# 分布式 RPC 框架

分布式 RPC 框架通过一组原语提供多机模型训练的机制，以实现远程通信，以及一个更高级的 API 来自动对跨多台机器分割的模型进行微分。

```{warning}
RPC 包中的 API 处于稳定和维护模式。
{warning}
CUDA 支持是一个 **测试版** 功能。
并非 RPC 包的所有功能都与 CUDA 支持兼容，因此不鼓励使用这些功能。这些不支持的功能包括：RRefs、JIT 兼容性、分布式自动求导和分布式优化器，以及性能分析。
{note}
请参阅 `PyTorch 分布式概述 <https://pytorch.org/tutorials/beginner/dist_overview.html>`__
以获取与分布式训练相关的所有功能的简要介绍。
```

## 基础

分布式 RPC 框架使得远程运行函数变得容易，支持引用远程对象而无需复制实际数据，并提供自动求导和优化器 API，以透明地跨 RPC 边界运行反向传播和更新参数。这些功能可以分为四组 API。

1) **远程过程调用 (RPC)** 支持在指定的目标工作节点上使用给定的参数运行函数，并获取返回值或创建对返回值的引用。主要有三个 RPC API：
   `torch.distributed.rpc.rpc_sync`（同步）、
   `torch.distributed.rpc.rpc_async`（异步）和
   `torch.distributed.rpc.remote`（异步并返回对远程返回值的引用）。
   如果用户代码在没有返回值的情况下无法继续执行，请使用同步 API。否则，使用异步 API 获取一个 future，并在调用者需要返回值时等待该 future。
   `torch.distributed.rpc.remote` API 在需要远程创建某些内容但永远不需要将其获取到调用者的情况下非常有用。想象一个驱动程序进程正在设置参数服务器和训练器的情况。驱动程序可以在参数服务器上创建一个嵌入表，然后与训练器共享对该嵌入表的引用，但驱动程序本身永远不会在本地使用该嵌入表。在这种情况下，
   `torch.distributed.rpc.rpc_sync` 和
   `torch.distributed.rpc.rpc_async` 不再适用，因为它们总是意味着返回值将立即或在未来返回给调用者。
2) **远程引用 (RRef)** 充当指向本地或远程对象的分布式共享指针。它可以与其他工作节点共享，并且引用计数将被透明处理。每个 RRef 只有一个所有者，并且对象仅存在于该所有者上。持有 RRef 的非所有者工作节点可以通过显式请求从所有者那里获取对象的副本。当一个工作节点需要访问某些数据对象，但它本身既不是对象的创建者（`torch.distributed.rpc.remote` 的调用者）也不是对象的所有者时，这非常有用。我们将在下面讨论的分布式优化器就是这种用例的一个例子。
3) **分布式自动求导** 将前向传播中涉及的所有工作节点上的本地自动求导引擎连接起来，并在反向传播期间自动访问它们以计算梯度。这在执行例如分布式模型并行训练、参数服务器训练等时，前向传播需要跨越多台机器的情况下特别有帮助。有了这个功能，用户代码不再需要担心如何跨 RPC 边界发送梯度，以及应该以何种顺序启动本地自动求导引擎，这在前向传播中存在嵌套和相互依赖的 RPC 调用时会变得相当复杂。
4) **分布式优化器** 的构造函数接受一个
   `torch.optim.Optimizer`（例如，`torch.optim.SGD`、
   `torch.optim.Adagrad` 等）和一个参数 RRef 列表，在每个不同的 RRef 所有者上创建一个
   `torch.optim.Optimizer` 实例，并在运行 ``step()`` 时相应地更新参数。当您有分布式前向和反向传播时，参数和梯度将分散在多个工作节点上，因此需要在每个涉及的工作节点上都有一个优化器。分布式优化器将所有那些本地优化器包装成一个，并提供一个简洁的构造函数和 ``step()`` API。


## RPC

在使用 RPC 和分布式自动求导原语之前，必须进行初始化。要初始化 RPC 框架，我们需要使用
`torch.distributed.rpc.init_rpc`，它将初始化 RPC 框架、RRef 框架和分布式自动求导。


以下 API 允许用户远程执行函数以及创建对远程数据对象的引用（RRef）。在这些 API 中，当传递一个
``Tensor`` 作为参数或返回值时，目标工作节点将尝试创建一个具有相同元数据（即形状、步幅等）的 ``Tensor``。
我们故意不允许传输 CUDA 张量，因为如果源工作节点和目标工作节点上的设备列表不匹配，可能会导致崩溃。在这种情况下，应用程序可以始终在调用者处将输入张量显式移动到 CPU，并在必要时在被调用者处将其移动到所需的设备。


RPC 包还提供了装饰器，允许应用程序指定在调用端应如何处理给定函数。


### 后端

RPC 模块可以利用不同的后端来执行节点间的通信。
要使用的后端可以通过传递 `torch.distributed.rpc.BackendType` 枚举的特定值，
在 `torch.distributed.rpc.init_rpc` 函数中指定。无论使用何种后端，
RPC API 的其余部分都不会改变。每个后端还定义了其自己的
`torch.distributed.rpc.RpcBackendOptions` 类的子类，
其实例也可以传递给 `torch.distributed.rpc.init_rpc`
以配置后端的行为。


#### TensorPipe 后端

默认的 TensorPipe 代理利用了 [TensorPipe 库](https://github.com/pytorch/tensorpipe)，
该库提供了一个原生的点对点通信原语，专门适用于机器学习，
从根本上解决了 Gloo 的一些限制。与 Gloo 相比，
它的优势在于异步性，允许大量传输同时进行，
每个传输以自己的速度进行，而不会相互阻塞。
它只在需要时按需在节点对之间打开管道，并且当一个节点失败时，
只有与其相关的管道会被关闭，而所有其他管道将继续正常工作。
此外，它能够支持多种不同的传输方式（当然包括 TCP，
还包括共享内存、NVLink、InfiniBand 等），
并能自动检测它们的可用性，并为每个管道协商使用最佳的传输方式。

TensorPipe 后端附带了一个基于 TCP 的传输方式，就像 Gloo 一样。
它还能够自动将大张量分块并在多个套接字和线程上进行多路复用，
以实现非常高的带宽。代理将能够自行选择最佳的传输方式，无需人工干预。

示例：

```{code-block} python
import os
from torch.distributed import rpc
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

rpc.init_rpc(
    "worker1",
    rank=0,
    world_size=2,
    rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=8,
        rpc_timeout=20 # 20 秒超时
    )
)

# 省略 worker2 上的 init_rpc 调用
{note}
RPC 框架不会自动重试任何
`torch.distributed.rpc.rpc_sync`、
`torch.distributed.rpc.rpc_async` 和
`torch.distributed.rpc.remote` 调用。
原因是 RPC 框架无法确定一个操作是否是幂等的，
以及重试是否安全。因此，应用程序有责任处理故障并在必要时重试。
RPC 通信基于 TCP，因此故障可能由于网络故障或间歇性网络连接问题而发生。
在这种情况下，应用程序需要以合理的退避策略进行适当重试，
以确保网络不会因激进的重试而拥塞。
```

## RRef

```{warning}
当前使用 CUDA 张量时不支持 RRef
```

``RRef``（远程引用）是对远程工作节点上某种类型 ``T``（例如 ``Tensor``）值的引用。
此句柄使引用的远程值在所有者上保持活动状态，但并不暗示该值将来会被传输到本地工作节点。
RRef 可用于多机训练，通过持有存在于其他工作节点上的 [nn.Modules](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的引用，
并在训练期间调用适当的函数来检索或修改其参数。更多细节请参见 *remote-reference-protocol*。


- [Rpc/rref](rpc/rref.md)


## RemoteModule

```{warning}
当前使用 CUDA 张量时不支持 RemoteModule
```

``RemoteModule`` 是一种在不同进程上远程创建 nn.Module 的简便方法。
实际的模块驻留在远程主机上，但本地主机有一个指向该模块的句柄，
并且可以像常规的 nn.Module 一样调用此模块。
然而，调用会引发对远程端的 RPC 调用，并且如果需要，
可以通过 RemoteModule 支持的额外 API 异步执行。


## 分布式自动求导框架

```{warning}
当前使用 CUDA 张量时不支持分布式自动求导
```

此模块提供了一个基于 RPC 的分布式自动求导框架，
可用于模型并行训练等应用。简而言之，
应用程序可以通过 RPC 发送和接收梯度记录张量。
在前向传播中，我们记录梯度记录张量何时通过 RPC 发送，
在反向传播中，我们使用此信息通过 RPC 执行分布式反向传播。
更多细节请参见 *distributed-autograd-design*。


- [Rpc/distributed Autograd](rpc/distributed_autograd.md)


## 分布式优化器

关于分布式优化器的文档，请参见 [torch.distributed.optim](https://pytorch.org/docs/main/distributed.optim.html) 页面。

## 设计说明

分布式自动求导设计说明涵盖了基于 RPC 的分布式自动求导框架的设计，
该框架对于模型并行训练等应用非常有用。

-  *distributed-autograd-design*

RRef 设计说明涵盖了框架用于引用远程工作节点上值的 *rref*（远程引用）协议的设计。

-  *remote-reference-protocol*

## 教程

RPC 教程向用户介绍 RPC 框架，提供了多个使用 torch.distributed.rpc API 的示例应用，并演示了如何使用 [性能分析器](https://pytorch.org/docs/stable/autograd.html#profiler) 来分析基于 RPC 的工作负载。

-  [分布式 RPC 框架入门](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html)
-  [使用分布式 RPC 框架实现参数服务器](https://pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html)
-  [将分布式数据并行与分布式 RPC 框架结合使用](https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html)（也涵盖 **RemoteModule**）
-  [实现批量 RPC 处理](https://pytorch.org/tutorials/intermediate/rpc_async_execution.html)
