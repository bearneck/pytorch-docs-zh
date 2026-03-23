

# 分布式通信包 - torch.distributed


> 📝 **注意**
> 关于分布式训练相关功能的简要介绍，请参阅 [PyTorch 分布式概述](https://pytorch.org/tutorials/beginner/dist_overview.html)。


## 后端

`torch.distributed` 支持四个内置后端，每个后端具有不同的功能。下表展示了每个后端在 CPU 或 GPU 上可用的函数。对于 NCCL，GPU 指的是 CUDA GPU，而对于 XCCL 则指 XPU GPU。

MPI 仅在用于构建 PyTorch 的实现支持 CUDA 时才支持 CUDA。


### PyTorch 自带的后端

PyTorch 分布式包支持 Linux（稳定版）、macOS（稳定版）和 Windows（原型版）。
在 Linux 上，默认情况下 Gloo 和 NCCL 后端会被构建并包含在 PyTorch 分布式包中（NCCL 仅在构建时使用 CUDA 的情况下包含）。MPI 是一个可选后端，只有在从源代码构建 PyTorch 时才能包含。（例如，在安装了 MPI 的主机上构建 PyTorch。）


> 📝 **注意**
> 从 PyTorch v1.8 开始，Windows 支持除 NCCL 之外的所有集体通信后端。如果 `init_process_group` 的 `init_method` 参数指向一个文件，它必须遵循以下模式：
>
> - 本地文件系统：`init_method="file:///d:/tmp/some_file"`
> - 共享文件系统：`init_method="file://////{machine_name}/{share_folder_name}/some_file"`
>
> 与 Linux 平台相同，您可以通过设置环境变量 MASTER_ADDR 和 MASTER_PORT 来启用 TcpStore。


### 应该使用哪个后端？

过去，我们经常被问到："我应该使用哪个后端？"。

- 经验法则

  - 使用 **GPU** CUDA 进行分布式训练时，使用 NCCL 后端。
  - 使用 **GPU** XPU 进行分布式训练时，使用 XCCL 后端。
  - 使用 **CPU** 进行分布式训练时，使用 Gloo 后端。

- 具有 InfiniBand 互连的 GPU 主机

  - 使用 NCCL，因为它是目前唯一支持 InfiniBand 和 GPUDirect 的后端。

- 具有以太网互连的 GPU 主机

  - 使用 NCCL，因为它目前提供了最佳的分布式 GPU 训练性能，特别是对于多进程单节点或多节点分布式训练。如果您遇到任何 NCCL 问题，请使用 Gloo 作为备选方案。（请注意，对于 GPU，Gloo 目前运行速度比 NCCL 慢。）

- 具有 InfiniBand 互连的 CPU 主机

  - 如果您的 InfiniBand 启用了 IP over IB，请使用 Gloo，否则请使用 MPI。我们计划在即将发布的版本中为 Gloo 添加 InfiniBand 支持。

- 具有以太网互连的 CPU 主机

  - 使用 Gloo，除非您有特定理由使用 MPI。

### 常用环境变量

#### 选择要使用的网络接口

默认情况下，NCCL 和 Gloo 后端都会尝试找到要使用的正确网络接口。如果自动检测到的接口不正确，您可以使用以下环境变量覆盖它（适用于相应的后端）：

- **NCCL_SOCKET_IFNAME**，例如 `export NCCL_SOCKET_IFNAME=eth0`
- **GLOO_SOCKET_IFNAME**，例如 `export GLOO_SOCKET_IFNAME=eth0`

如果您使用 Gloo 后端，可以通过用逗号分隔来指定多个接口，例如：`export GLOO_SOCKET_IFNAME=eth0,eth1,eth2,eth3`。后端将以轮询方式在这些接口上分发操作。所有进程必须在此变量中指定相同数量的接口。

#### 其他 NCCL 环境变量

**调试** - 如果 NCCL 失败，您可以设置 `NCCL_DEBUG=INFO` 来打印明确的警告信息以及基本的 NCCL 初始化信息。

您也可以使用 `NCCL_DEBUG_SUBSYS` 来获取 NCCL 特定方面的更多细节。例如，`NCCL_DEBUG_SUBSYS=COLL` 会打印集体调用的日志，这在调试挂起时可能很有帮助，特别是那些由集体操作类型或消息大小不匹配引起的挂起。在拓扑检测失败的情况下，设置 `NCCL_DEBUG_SUBSYS=GRAPH` 有助于检查详细的检测结果，并在需要 NCCL 团队进一步帮助时保存为参考。

**性能调优** - NCCL 基于其拓扑检测进行自动调优，以节省用户的调优工作量。在一些基于套接字的系统上，用户仍然可以尝试调优 `NCCL_SOCKET_NTHREADS` 和 `NCCL_NSOCKS_PERTHREAD` 来增加套接字网络带宽。这两个环境变量已由 NCCL 为一些云服务提供商（如 AWS 或 GCP）进行了预调优。

有关 NCCL 环境变量的完整列表，请参阅 [NVIDIA NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

您甚至可以使用 `torch.distributed.ProcessGroupNCCL.NCCLConfig` 和 `torch.distributed.ProcessGroupNCCL.Options` 进一步调优 NCCL 通信器。在解释器中使用 `help` 了解更多信息（例如 `help(torch.distributed.ProcessGroupNCCL.NCCLConfig)`）。


## 基础

`torch.distributed` 包为运行在一台或多台机器上的多个计算节点之间的多进程并行提供了 PyTorch 支持和通信原语。类 `torch.nn.parallel.DistributedDataParallel` 基于此功能构建，作为任何 PyTorch 模型的包装器，提供同步分布式训练。这与 `multiprocessing` 和 `torch.nn.DataParallel` 提供的并行类型不同，因为它支持多个网络连接的机器，并且用户必须为每个进程显式启动主训练脚本的单独副本。

在单机同步的情况下，`torch.distributed` 或 `torch.nn.parallel.DistributedDataParallel` 包装器相对于其他数据并行方法（包括 `torch.nn.DataParallel`）可能仍然具有优势：

- 每个进程维护自己的优化器，并在每次迭代中执行完整的优化步骤。虽然这看起来是冗余的，因为梯度已经在进程间收集并平均，因此对每个进程都是相同的，但这意味着不需要参数广播步骤，减少了在节点之间传输张量的时间。
- 每个进程包含一个独立的 Python 解释器，消除了从单个 Python 进程驱动多个执行线程、模型副本或 GPU 所带来的额外解释器开销和 "GIL 抖动"。这对于大量使用 Python 运行时的模型尤其重要，包括具有循环层或许多小组件的模型。

## 初始化

在调用任何其他方法之前，需要使用 `torch.distributed.init_process_group` 或 `torch.distributed.device_mesh.init_device_mesh` 函数初始化该包。两者都会阻塞直到所有进程加入。


> ⚠️ **警告**
> 初始化不是线程安全的。进程组的创建应在单个线程中执行，以防止跨不同秩的 'UUID' 分配不一致，并防止初始化期间的竞争条件导致挂起。


______________________________________________________________________

目前支持三种初始化方法：

### TCP 初始化

有两种使用 TCP 初始化的方式，都需要一个所有进程都可访问的网络地址和所需的 `world_size`。第一种方式需要指定一个属于秩为 0 的进程的地址。此初始化方法要求所有进程都手动指定了秩。

请注意，最新的分布式包不再支持多播地址。`group_name` 也已弃用。

```
import torch.distributed as dist

# 使用其中一台机器的地址
dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456',
                        rank=args.rank, world_size=4)
```

### 共享文件系统初始化

另一种初始化方法利用一个在组内所有机器上都可见的共享文件系统，以及期望的 `world_size`。URL 应以 `file://` 开头，并指向共享文件系统上一个不存在的文件（位于现有目录中）。文件系统初始化会在文件不存在时自动创建该文件，但不会删除文件。因此，您有责任确保在下一次对相同文件路径/名称调用 `init_process_group` 之前清理该文件。

请注意，最新的分布式包不再支持自动排名分配，并且 `group_name` 也已弃用。


> ⚠️ **警告**
> 此方法假设文件系统支持使用 `fcntl` 进行锁定——大多数本地系统和 NFS 都支持此功能。


> ⚠️ **警告**
> 此方法将始终创建文件，并尽力在程序结束时清理和删除文件。换句话说，每次使用文件初始化方法进行初始化都需要一个全新的空文件，初始化才能成功。如果再次使用先前初始化所用的同一文件（该文件恰巧未被清理），这是未预期的行为，通常会导致死锁和失败。因此，尽管此方法会尽力清理文件，但如果自动删除未能成功，您有责任确保在训练结束时移除文件，以防止下次再次重用同一文件。如果您计划在同一文件名上多次调用 `init_process_group`，这一点尤其重要。换句话说，如果文件未被移除/清理，而您再次对该文件调用 `init_process_group`，预计会出现失败。这里的经验法则是，确保每次调用 `init_process_group` 时文件不存在或为空。


```
import torch.distributed as dist

# rank 应始终指定
dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile',
                        world_size=4, rank=args.rank)
```

### 环境变量初始化

此方法将从环境变量读取配置，允许完全自定义信息的获取方式。需要设置的变量包括：

- `MASTER_PORT` - 必需；必须是 rank 0 机器上的空闲端口
- `MASTER_ADDR` - 必需（rank 0 除外）；rank 0 节点的地址
- `WORLD_SIZE` - 必需；可以在此处设置，也可以在初始化函数调用中设置
- `RANK` - 必需；可以在此处设置，也可以在初始化函数调用中设置

rank 0 的机器将用于建立所有连接。

这是默认方法，意味着无需指定 `init_method`（或可以设为 `env://`）。

### 改进初始化时间

- `TORCH_GLOO_LAZY_INIT` - 按需建立连接，而不是使用全连接网格，这可以显著改进非 all2all 操作的初始化时间。

## 初始化后

一旦运行了 `torch.distributed.init_process_group`，就可以使用以下函数。要检查进程组是否已初始化，请使用 `torch.distributed.is_initialized`。


## 关闭

通过调用 `destroy_process_group` 在退出时清理资源非常重要。

最简单的模式是在训练脚本中不再需要通信时（通常在 main() 函数末尾附近），通过为 `group` 参数使用默认值 None 调用 `destroy_process_group()` 来销毁每个进程组和后端。此调用应在每个训练器进程中执行一次，而不是在外部的进程启动器级别。

如果 pg 中的所有 rank 未在超时时间内调用 `destroy_process_group`，特别是在应用程序中有多个进程组时（例如用于 N-D 并行），退出时可能会出现挂起。这是因为 ProcessGroupNCCL 的析构函数会调用 ncclCommAbort，而该调用必须集体执行，但如果由 python 的 GC 调用 ProcessGroupNCCL 的析构函数，其调用顺序是不确定的。调用 `destroy_process_group` 有助于确保跨 rank 以一致的顺序调用 ncclCommAbort，并避免在 ProcessGroupNCCL 的析构函数中调用 ncclCommAbort。

### 重新初始化

`destroy_process_group` 也可用于销毁单个进程组。一个用例可能是容错训练，其中进程组可能在运行时被销毁，然后重新初始化。在这种情况下，关键是在调用 destroy 之后、随后初始化之前，使用除 torch.distributed 原语之外的某种方式同步训练器进程。由于实现此同步的困难，目前此行为不受支持/未经测试，被视为已知问题。如果此用例对您造成阻碍，请提交 github issue 或 RFC。

______________________________________________________________________

## 组

默认情况下，集合操作在默认组（也称为 world）上运行，并要求所有进程进入分布式函数调用。然而，某些工作负载可以从更细粒度的通信中受益。这就是分布式组发挥作用的地方。`~torch.distributed.new_group` 函数可用于创建新组，包含所有进程的任意子集。它返回一个不透明的组句柄，可以作为 `group` 参数传递给所有集合操作（集合操作是在某些众所周知的编程模式中交换信息的分布式函数）。


## DeviceMesh

DeviceMesh 是一个更高级别的抽象，用于管理进程组（或 NCCL 通信器）。
它允许用户轻松创建节点间和节点内的进程组，而无需担心如何为不同的子进程组正确设置秩，并有助于轻松管理这些分布式进程组。`~torch.distributed.device_mesh.init_device_mesh` 函数可用于创建新的 DeviceMesh，并通过一个描述设备拓扑的网格形状来实现。


## 点对点通信


`~torch.distributed.isend` 和 `~torch.distributed.irecv`
在使用时会返回分布式请求对象。通常，此对象的类型是未指定的，因为它们不应手动创建，但保证支持两种方法：

- `is_completed()` - 如果操作已完成则返回 True
- `wait()` - 将阻塞进程直到操作完成。一旦返回，`is_completed()` 保证返回 True。


## 同步和异步集体操作

每个集体操作函数都支持以下两种操作，具体取决于传递给集体的 `async_op` 标志的设置：

**同步操作** - 默认模式，当 `async_op` 设置为 `False` 时。
当函数返回时，保证集体操作已执行。对于 CUDA 操作，不保证 CUDA 操作已完成，因为 CUDA 操作是异步的。对于 CPU 集体操作，任何使用集体调用输出的后续函数调用都将按预期运行。对于 CUDA 集体操作，在同一 CUDA 流上使用输出的函数调用将按预期运行。用户必须在不同流下运行的场景中注意同步。有关 CUDA 语义（如流同步）的详细信息，请参阅 [CUDA 语义](https://pytorch.org/docs/stable/notes/cuda.html)。
请参阅以下脚本以了解 CPU 和 CUDA 操作在这些语义上的差异示例。

**异步操作** - 当 `async_op` 设置为 True 时。集体操作函数返回一个分布式请求对象。通常，您不需要手动创建它，并且保证支持两种方法：

- `is_completed()` - 对于 CPU 集体操作，如果完成则返回 `True`。对于 CUDA 操作，如果操作已成功排入 CUDA 流并且输出可以在默认流上使用而无需进一步同步，则返回 `True`。
- `wait()` - 对于 CPU 集体操作，将阻塞进程直到操作完成。对于 CUDA 集体操作，将阻塞当前活动的 CUDA 流直到操作完成（但不会阻塞 CPU）。
- `get_future()` - 返回 `torch._C.Future` 对象。支持 NCCL，也支持 GLOO 和 MPI 上的大多数操作，除了点对点操作。
  注意：随着我们继续采用 Futures 并合并 API，`get_future()` 调用可能会变得冗余。

**示例**

以下代码可以作为使用分布式集体操作时 CUDA 操作语义的参考。它展示了在不同 CUDA 流上使用集体输出时需要显式同步：

```
# 代码在每个秩上运行。
dist.init_process_group("nccl", rank=rank, world_size=2)
output = torch.tensor([rank]).cuda(rank)
s = torch.cuda.Stream()
handle = dist.all_reduce(output, async_op=True)
# Wait 确保操作已入队，但不一定完成。
handle.wait()
# 在非默认流上使用结果。
with torch.cuda.stream(s):
    s.wait_stream(torch.cuda.default_stream())
    output.add_(100)
if rank == 0:
    # 如果省略了对 wait_stream 的显式调用，下面的输出将是
    # 非确定性的 1 或 101，取决于 allreduce 是否在 add 完成后覆盖了值。
    print(output)
```

## 集体函数


## 分布式键值存储

distributed 包附带一个分布式键值存储，可用于在进程组之间共享信息，也可用于在 `torch.distributed.init_process_group` 中初始化分布式包（通过显式创建存储作为指定 `init_method` 的替代方案）。键值存储有 3 种选择：`~torch.distributed.TCPStore`、`~torch.distributed.FileStore` 和 `~torch.distributed.HashStore`。


## 分析集体通信

请注意，您可以使用 `torch.profiler`（推荐，仅在 1.8.1 之后可用）或 `torch.autograd.profiler` 来分析此处提到的集体通信和点对点通信 API。所有开箱即用的后端（`gloo`、`nccl`、`mpi`）都受支持，集体通信的使用情况将在分析输出/跟踪中按预期呈现。分析代码与任何常规 torch 操作相同：

```
import torch
import torch.distributed as dist
with torch.profiler():
    tensor = torch.randn(20, 10)
    dist.all_reduce(tensor)
```

有关分析器功能的完整概述，请参阅[分析器文档](https://pytorch.org/docs/main/profiler.html)。

## 使用对称内存进行优化

### 复制引擎集体操作

当在具有零 CTA 策略的对称内存张量上执行 NCCL 集体操作时，数据移动会卸载到 GPU 的复制引擎（DMA 引擎），而不是使用 CUDA 流式多处理器（SM）。这释放了 SM 用于计算工作，从而更好地实现通信和计算的重叠。

有关设置说明、要求和示例，请参阅对称内存文档中的[复制引擎集体操作](copy-engine-collectives)。

### 高精度归约

当 NCCL 集体操作（如 `reduce_scatter` 和 `all_reduce`）在对称内存张量上运行时，NCCL 的对称内核实现会自动以更高精度执行内部归约（例如，BF16/FP16 输入 → FP32 累加 → BF16/FP16 输出）。这提高了数值精度，而无需对集体调用进行任何代码更改。

有关范围、支持的域和版本要求的详细信息，请参阅对称内存文档中的[高精度归约](higher-precision-reduction)。

## 多 GPU 集体函数


> ⚠️ **警告**
> 多 GPU 函数（代表每个 CPU 线程有多个 GPU）已弃用。截至目前，PyTorch Distributed 的首选编程模型是每个线程一个设备，如本文档中的 API 所示。如果您是后端开发人员并希望支持每个线程多个设备，请联系 PyTorch Distributed 的维护者。


## 对象集体操作


> ⚠️ **警告**
> 对象集体操作有许多严重的限制。请继续阅读以确定它们是否适合您的用例。


对象集体操作是一组类似集体的操作，适用于任意 Python 对象，只要它们可以被 pickle。实现了各种集体模式（例如 broadcast、all_gather 等），但它们大致遵循以下模式：

1. 将输入对象转换为 pickle（原始字节），然后将其塞入字节张量
2. 将此字节张量的大小传达给对等体（第一个集体操作）
3. 分配适当大小的张量以执行真正的集体操作
4. 传达对象数据（第二个集体操作）
5. 将原始数据转换回 Python（unpickle）

对象集体操作有时具有令人惊讶的性能或内存特性，导致运行时间过长或内存不足（OOM），因此应谨慎使用。以下是一些常见问题。

**不对称的 pickle/unpickle 时间** - Pickle 对象可能很慢，具体取决于对象的数量、类型和大小。当集体操作具有扇入（例如 gather_object）时，接收方必须 unpickle 比发送方 pickle 多 N 倍的对象，这可能导致其他进程在下一个集体操作中超时。

**低效的张量通信** - 张量应通过常规集体 API 发送，而不是通过对象集体 API。可以通过对象集体 API 发送张量，但它们将被序列化和反序列化（包括非 CPU 张量的 CPU 同步和设备到主机复制），并且在几乎所有情况下（除了调试或故障排除代码），都值得花时间重构代码以使用非对象集体操作。

**意外的张量设备** - 如果您仍然想通过对象集体操作发送张量，还有另一个特定于 CUDA（可能还有其他加速器）张量的方面。如果您 pickle 一个当前在 `cuda:3` 上的张量，然后 unpickle 它，您将在 `cuda:3` 上获得另一个张量，*无论您在哪个进程中，或者该进程的“默认”设备是哪个 CUDA 设备*。使用常规张量集体 API，“输出张量”将始终位于相同的本地设备上，这通常是您所期望的。

如果这是进程第一次使用 GPU，unpickle 张量将隐式激活 CUDA 上下文，这可能会浪费大量 GPU 内存。可以通过在将张量作为输入传递给对象集体操作之前将其移动到 CPU 来避免此问题。

## 第三方后端

除了内置的 GLOO/MPI/NCCL 后端，PyTorch 分布式还支持通过运行时注册机制使用第三方后端。关于如何通过 C++ 扩展开发第三方后端的参考资料，请参阅 [教程 - 自定义 C++ 和 CUDA 扩展](https://pytorch.org/tutorials/advanced/cpp_extension.html) 和 `test/cpp_extensions/cpp_c10d_extension.cpp`。第三方后端的能力由其自身实现决定。

新的后端继承自 `c10d::ProcessGroup`，并在导入时通过 `torch.distributed.Backend.register_backend` 注册后端名称和实例化接口。

当手动导入此后端并使用相应的后端名称调用 `torch.distributed.init_process_group` 时，`torch.distributed` 包将在新后端上运行。


> ⚠️ **警告**
> 对第三方后端的支持是实验性的，可能会发生变化。


## 启动工具

`torch.distributed` 包还在 `torch.distributed.launch` 中提供了一个启动工具。这个辅助工具可用于在每个节点上启动多个进程以进行分布式训练。


## 生成工具

`multiprocessing-doc` 包还在 `torch.multiprocessing.spawn` 中提供了一个 `spawn` 函数。这个辅助函数可用于生成多个进程。它的工作原理是传入你想要运行的函数，然后生成 N 个进程来运行它。这也可以用于多进程分布式训练。

关于如何使用它的参考资料，请参阅 [PyTorch 示例 - ImageNet 实现](https://github.com/pytorch/examples/tree/master/imagenet)

请注意，此函数需要 Python 3.4 或更高版本。

## 调试 `torch.distributed` 应用程序

调试分布式应用程序可能具有挑战性，因为难以理解的挂起、崩溃或跨进程的不一致行为。`torch.distributed` 提供了一套工具，以自助方式帮助调试训练应用程序：

### Python 断点

在分布式环境中使用 Python 的调试器非常方便，但由于它不能开箱即用，许多人根本不使用它。PyTorch 提供了一个围绕 pdb 的自定义包装器，简化了这个过程。

`torch.distributed.breakpoint` 使这个过程变得容易。在内部，它以两种方式定制了 `pdb` 的断点行为，但在其他方面表现得像普通的 `pdb`。
1. 仅在一个进程（由用户指定）上附加调试器。
2. 通过使用 `torch.distributed.barrier()` 确保所有其他进程停止，该屏障将在被调试的进程发出 `continue` 时释放。
3. 重定向子进程的标准输入，使其连接到你的终端。

要使用它，只需在所有进程上调用 `torch.distributed.breakpoint(rank)`，并在每种情况下使用相同的 `rank` 值。

### 受监控的屏障

从 v1.10 开始，`torch.distributed.monitored_barrier` 作为 `torch.distributed.barrier` 的替代方案存在，当崩溃时（即并非所有进程在提供的超时时间内调用 `torch.distributed.monitored_barrier`），它会提供有关哪个进程可能出故障的有用信息。`torch.distributed.monitored_barrier` 使用 `send`/`recv` 通信原语在类似于确认的过程中实现了一个主机端屏障，允许进程 0 报告哪个（些）进程未能及时确认屏障。例如，考虑以下函数，其中进程 1 未能调用 `torch.distributed.monitored_barrier`（在实践中，这可能是由于应用程序错误或先前集合操作中的挂起）：

```
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank):
    dist.init_process_group("nccl", rank=rank, world_size=2)
    # monitored barrier requires gloo process group to perform host-side sync.
    group_gloo = dist.new_group(backend="gloo")
    if rank not in [1]:
        dist.monitored_barrier(group=group_gloo, timeout=timedelta(seconds=2))


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    mp.spawn(worker, nprocs=2, args=())
```

在进程 0 上会产生以下错误信息，允许用户确定哪个（些）进程可能出故障并进行进一步调查：

```
RuntimeError: Rank 1 failed to pass monitoredBarrier in 2000 ms
 Original exception:
[gloo/transport/tcp/pair.cc:598] Connection closed by peer [2401:db00:eef0:1100:3560:0:1c05:25d]:8594
```

### `TORCH_DISTRIBUTED_DEBUG`

当 `TORCH_CPP_LOG_LEVEL=INFO` 时，环境变量 `TORCH_DISTRIBUTED_DEBUG` 可用于触发额外的有用日志记录和集合同步检查，以确保所有进程都适当地同步。`TORCH_DISTRIBUTED_DEBUG` 可以根据所需的调试级别设置为 `OFF`（默认）、`INFO` 或 `DETAIL`。请注意，最详细的选项 `DETAIL` 可能会影响应用程序性能，因此应仅在调试问题时使用。

设置 `TORCH_DISTRIBUTED_DEBUG=INFO` 将在使用 `torch.nn.parallel.DistributedDataParallel` 训练的模型初始化时产生额外的调试日志记录，而 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 将额外记录选定迭代次数的运行时性能统计信息。这些运行时统计信息包括前向时间、后向时间、梯度通信时间等数据。例如，给定以下应用程序：

```
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
```

class TwoLinLayerNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(10, 10, bias=False)
        self.b = torch.nn.Linear(10, 1, bias=False)

    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        return (a, b)


def worker(rank):
    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    print("初始化模型")
    model = TwoLinLayerNet().cuda()
    print("初始化 ddp")
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    inp = torch.randn(10, 10).cuda()
    print("训练")

    for _ in range(20):
        output = ddp_model(inp)
        loss = output[0] + output[1]
        loss.sum().backward()


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"  # 设置为 DETAIL 以启用运行时日志记录。
    mp.spawn(worker, nprocs=2, args=())
```

以下日志在初始化时生成：

```
I0607 16:10:35.739390 515217 logger.cpp:173] [Rank 0]: DDP 初始化参数：
broadcast_buffers: 1
bucket_cap_bytes: 26214400
find_unused_parameters: 0
gradient_as_bucket_view: 0
is_multi_device_module: 0
iteration: 0
num_parameter_tensors: 2
output_device: 0
rank: 0
total_parameter_size_bytes: 440
world_size: 2
backend_name: nccl
bucket_sizes: 440
cuda_visible_devices: N/A
device_ids: 0
dtypes: float
master_addr: localhost
master_port: 29501
module_name: TwoLinLayerNet
nccl_async_error_handling: N/A
nccl_blocking_wait: N/A
nccl_debug: WARN
nccl_ib_timeout: N/A
nccl_nthreads: N/A
nccl_socket_ifname: N/A
torch_distributed_debug: INFO
```

以下日志在运行时生成（当设置了 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 时）：

```
I0607 16:18:58.085681 544067 logger.cpp:344] [Rank 1 / 2] 训练 TwoLinLayerNet unused_parameter_size=0
 平均前向计算时间：40838608
 平均反向计算时间：5983335
平均反向通信时间：4326421
 平均反向通信/计算重叠时间：4207652
I0607 16:18:58.085693 544066 logger.cpp:344] [Rank 0 / 2] 训练 TwoLinLayerNet unused_parameter_size=0
 平均前向计算时间：42850427
 平均反向计算时间：3885553
平均反向通信时间：2357981
 平均反向通信/计算重叠时间：2234674
```

此外，`TORCH_DISTRIBUTED_DEBUG=INFO` 增强了 `torch.nn.parallel.DistributedDataParallel` 中因模型存在未使用参数而导致的崩溃日志记录。目前，如果前向传播中可能存在未使用的参数，则必须在 `torch.nn.parallel.DistributedDataParallel` 初始化时传入 `find_unused_parameters=True`，并且从 v1.10 开始，要求所有模型输出都参与损失计算，因为 `torch.nn.parallel.DistributedDataParallel` 不支持反向传播中存在未使用的参数。这些约束对于大型模型尤其具有挑战性，因此当因错误而崩溃时，`torch.nn.parallel.DistributedDataParallel` 将记录所有未使用参数的完全限定名称。例如，在上述应用程序中，如果我们将 `loss` 的计算改为 `loss = output[1]`，那么 `TwoLinLayerNet.a` 在反向传播中不会收到梯度，从而导致 `DDP` 失败。崩溃时，用户会收到关于未使用参数的信息，这对于大型模型来说可能难以手动查找：

```
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing
 the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by
making sure all `forward` function outputs participate in calculating loss.
If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return va
lue of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
Parameters which did not receive grad for rank 0: a.weight
Parameter indices which did not receive grad for rank 0: 0
```

设置 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 将对用户直接或间接（例如 DDP 的 `allreduce`）发出的每个集体调用触发额外的**一致性**和**同步**检查。这是通过创建一个包装进程组来实现的，该包装进程组包装了 `torch.distributed.init_process_group` 和 `torch.distributed.new_group` API 返回的所有进程组。因此，这些 API 将返回一个包装进程组，可以像常规进程组一样使用，但在将集体操作分派到底层进程组之前会执行一致性检查。目前，这些检查包括一个 `torch.distributed.monitored_barrier`，它确保所有秩完成其未完成的集体调用，并报告卡住的秩。接下来，通过确保所有集体函数匹配且使用一致的张量形状调用，来检查集体操作本身的一致性。如果不是这种情况，应用程序崩溃时会包含详细的错误报告，而不是挂起或无信息的错误消息。例如，考虑以下函数，它在 `torch.distributed.all_reduce` 中使用了不匹配的输入形状：

```
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank):
    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    tensor = torch.randn(10 if rank == 0 else 20).cuda()
    dist.all_reduce(tensor)
    torch.cuda.synchronize(device=rank)
```

```python
if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    mp.spawn(worker, nprocs=2, args=())
```

使用 `NCCL` 后端时，此类应用程序很可能会导致挂起，这在复杂场景中可能难以定位根本原因。如果用户启用 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 并重新运行应用程序，以下错误信息会揭示根本原因：

```
work = default_pg.allreduce([tensor], opts)
RuntimeError: Error when verifying shape tensors for collective ALLREDUCE on rank 0. This likely indicates that input shapes into the collective are mismatched across ranks. Got shapes:  10
[ torch.LongTensor{1} ]
```


> 📝 **注意**
> 为了在运行时对调试级别进行细粒度控制，也可以使用函数 `torch.distributed.set_debug_level`、`torch.distributed.set_debug_level_from_env` 和 `torch.distributed.get_debug_level`。


此外，`TORCH_DISTRIBUTED_DEBUG=DETAIL` 可以与 `TORCH_SHOW_CPP_STACKTRACES=1` 结合使用，以便在检测到集合操作不同步时记录完整的调用堆栈。这些集合操作不同步检查适用于所有使用 `c10d` 集合调用的应用程序，这些调用由通过 `torch.distributed.init_process_group` 和 `torch.distributed.new_group` API 创建的进程组支持。


### torch.distributed.debug HTTP 服务器

`torch.distributed.debug` 模块提供了一个 HTTP 服务器，可用于调试分布式应用程序。可以通过调用 `torch.distributed.debug.start_debug_server` 来启动服务器。这允许用户在运行时跨所有工作进程收集数据。


## 日志记录

除了通过 `torch.distributed.monitored_barrier` 和 `TORCH_DISTRIBUTED_DEBUG` 提供的显式调试支持外，`torch.distributed` 的底层 C++ 库也会在不同级别输出日志消息。这些消息有助于理解分布式训练作业的执行状态，并排查诸如网络连接失败等问题。下表展示了如何通过组合 `TORCH_CPP_LOG_LEVEL` 和 `TORCH_DISTRIBUTED_DEBUG` 环境变量来调整日志级别。

| `TORCH_CPP_LOG_LEVEL` | `TORCH_DISTRIBUTED_DEBUG` | 有效日志级别 |
| --------------------- | ------------------------- | ------------------- |
| `ERROR`               | 忽略                      | 错误               |
| `WARNING`             | 忽略                      | 警告             |
| `INFO`                | 忽略                      | 信息                |
| `INFO`                | `INFO`                    | 调试               |
| `INFO`                | `DETAIL`                  | 跟踪（即全部）  |

分布式组件会引发从 `RuntimeError` 派生的自定义异常类型：

- `torch.distributed.DistError`：这是所有分布式异常的基类型。
- `torch.distributed.DistBackendError`：当发生特定于后端的错误时抛出此异常。例如，如果使用 `NCCL` 后端，并且用户尝试使用 `NCCL` 库无法访问的 GPU。
- `torch.distributed.DistNetworkError`：当网络库遇到错误时抛出此异常（例如：对端重置连接）。
- `torch.distributed.DistStoreError`：当存储（Store）遇到错误时抛出此异常（例如：TCPStore 超时）。


如果您正在运行单节点训练，可能希望以交互方式在脚本中设置断点。我们提供了一种方便地在单个排名（rank）上设置断点的方法：


% 缺少特定条目的分布式模块。

% 在此处添加它们以便跟踪，直到它们得到更永久的修复。


```{toctree}
:hidden:

distributed._dist2
```