```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# PyTorch 对称内存

:::{note}
`torch.distributed._symmetric_memory` 目前处于 alpha 状态，正在开发中。API 可能会发生变化。
:::

## 为什么需要对称内存？

随着并行化技术的快速发展，现有的框架和库往往难以跟上步伐，开发者越来越多地依赖直接调度通信和计算的自定义实现。近年来，我们见证了从主要依赖一维数据并行技术到多维并行技术的转变。后者对不同类型的通信有不同的延迟要求，因此需要精细地重叠计算和通信。

为了最小化计算干扰，它们还需要使用复制引擎和网络接口卡（NIC）来驱动通信。远程直接内存访问（RDMA）等网络传输协议通过实现处理器与内存之间直接、高速、低延迟的通信来提升性能。这种多样性的增加表明，需要比当前高级集合 API 提供的更细粒度的通信原语，这些原语将使开发者能够实现针对其用例定制的特定算法，例如低延迟集合操作、细粒度的计算-通信重叠或自定义融合。

此外，当今先进的人工智能系统通过高带宽链路（如 NVLink、InfiniBand 或 RoCE）连接 GPU，使得对等节点可以直接访问 GPU 全局内存。这种连接为程序员提供了一个绝佳的机会，可以将系统编程为一个具有巨大可访问内存的单一巨型 GPU，而不是编程单个的“GPU 孤岛”。

在本文档中，我们将展示如何使用 PyTorch 对称内存将现代 GPU 系统编程为“单个 GPU”，并实现细粒度的远程访问。

## PyTorch 对称内存解锁了什么？

PyTorch 对称内存解锁了三种新能力：

- **自定义通信模式**：内核编写灵活性的增加允许开发者编写自定义内核，实现其自定义的计算和通信，直接满足应用程序的需求。即使标准库中尚未支持，也可以轻松添加对新数据类型以及这些数据类型可能需要的特殊计算的支持。

- **内核内计算-通信融合**：设备发起的通信能力允许开发者编写同时包含计算和通信指令的内核，从而在尽可能小的粒度上融合计算和数据移动。

- **低延迟远程访问**：像 RDMA 这样的网络传输协议通过实现处理器与内存之间直接、高速、低延迟的通信，增强了网络环境中对称内存的性能。RDMA 消除了与传统网络栈和 CPU 参与相关的开销。它还将数据传输从计算卸载到 NIC，释放计算资源用于计算任务。

接下来，我们将展示 PyTorch 对称内存（SymmMem）如何利用上述能力实现新的应用。

## 一个“Hello World”示例

PyTorch SymmMem 编程模型涉及两个关键元素：

- 创建对称张量
- 创建 SymmMem 内核

要创建对称张量，可以使用 `torch.distributed._symmetric_memory` 包：

```python
import torch.distributed._symmetric_memory as symm_mem

t = symm_mem.empty(128, device=torch.device("cuda", rank))
hdl = symm_mem.rendezvous(t, group)
```

`symm_mem.empty` 函数创建一个由对称内存分配支持的张量。`rendezvous` 函数与组内的对等节点建立会合，并返回一个指向对称内存分配的句柄。该句柄提供了访问与对称内存分配相关信息的方法，例如指向对等节点等级上对称缓冲区的指针、多播指针（如果支持）以及信号垫。

组内所有节点必须按相同顺序调用 `empty` 和 `rendezvous` 函数。

然后，可以在这些张量上调用集合操作。例如，执行一次性全归约：

```python
# 大多数 SymmMem 操作都在 torch.ops.symm_mem 命名空间下
torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group)
```

请注意，`torch.ops.symm_mem` 是一个“操作命名空间”而不是一个 Python 模块。因此，你不能通过 `import torch.ops.symm_mem` 导入它，也不能通过 `from torch.ops.symm_mem import one_shot_all_reduce` 导入一个操作。你可以像上面的例子一样直接调用该操作。

## 编写你自己的内核

要编写你自己的使用对称内存进行通信的内核，你需要访问映射的对等缓冲区地址以及同步所需的信号垫。在内核中，你还需要执行正确的同步，以确保对等节点准备好进行通信，并向它们发出此 GPU 已准备好的信号。

PyTorch 对称内存提供了与 CUDA Graph 兼容的同步原语，这些原语在每个对称内存分配附带的信号垫上操作。使用对称内存的内核既可以用 CUDA 编写，也可以用 Triton 编写。以下是一个分配对称张量并交换句柄的示例：

```python
import torch.distributed._symmetric_memory as symm_mem

dist.init_process_group()
rank = dist.get_rank()

# 分配一个张量
t = symm_mem.empty(4096, device=f"cuda:{rank}")
# 建立对称内存并获取句柄
hdl = symm_mem.rendezvous(t, dist.group.WORLD)
```

通过以下方式提供对缓冲区指针、多内存指针和信号垫的访问：

```python
hdl.buffer_ptrs
hdl.multicast_ptr
hdl.signal_pad_ptrs
```

`buffer_ptrs` 所指向的数据可以像常规本地数据一样访问，任何必要的计算也可以按常规方式执行。与本地数据一样，您可以且应该使用向量化访问来提高效率。

对称内存在 Triton 中编写内核时特别方便。虽然之前 Triton 消除了编写高效 CUDA 代码的障碍，但现在可以轻松地将通信添加到 Triton 内核中。下面的内核展示了一个用 Triton 编写的低延迟、全归约内核。

```python
@triton.jit
def one_shot_all_reduce_kernel(
    buf_tuple,
    signal_pad_ptrs,
    output_ptr,
    numel: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasSubsequenceMemAccess=True
    )

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    while block_start < numel:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.bfloat16)

        for i in tl.static_range(world_size):
            buffer_rank = buf_tuple[i]
            x = tl.load(buffer_rank + offsets, mask=mask)
            acc += x

        tl.store(output_ptr + offsets, acc, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasPreviousMemAccess=True
    )
```

上述内核开头和结尾的同步保证了所有进程看到一致的数据。内核的主体部分是熟悉的 Triton 代码，Triton 会在幕后对其进行优化，确保通过向量化和展开以高效方式执行内存访问。与所有 Triton 内核一样，可以轻松修改它以添加额外的计算或更改通信算法。访问 https://github.com/meta-pytorch/kraken/blob/main/kraken 以查看更多实用工具和使用对称内存在 Triton 中实现常见模式的示例。

## 横向扩展

大型语言模型将专家分布在超过 8 个 GPU 上，因此需要多节点访问能力。支持 RDMA 的网卡应运而生。此外，像 NVSHMEM 或 rocSHMEM 这样的软件库通过比指针访问稍高级的原语（如 put 和 get）抽象了节点内访问和节点间访问的编程差异。

PyTorch 提供了 NVSHMEM 插件来增强 Triton 内核的跨节点能力。如下面的代码片段所示，可以在内核内发起跨节点的 put 命令。

```python
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
from torch.distributed._symmetric_memory._nvshmem_triton import requires_nvshmem

@requires_nvshmem
@triton.jit
def my_put_kernel(
    dest,
    src,
    nelems,
    pe,
):
    nvshmem.put(dest, src, nelems, pe)
```

`requires_nvshmem` 装饰器用于指示该内核需要 NVSHMEM 设备库作为外部依赖项。当 Triton 编译内核时，装饰器将在系统路径中搜索 NVSHMEM 设备库。如果可用，Triton 将包含必要的设备汇编代码以使用 NVSHMEM 函数。

## 使用内存池

内存池允许 PyTorch SymmMem 缓存已进行会合的内存分配，从而在创建新张量时节省时间。为方便起见，PyTorch SymmMem 添加了一个 `get_mem_pool` API 来返回一个对称内存池。用户可以将返回的 MemPool 与 `torch.cuda.use_mem_pool` 上下文管理器一起使用。在下面的示例中，张量 `x` 将从对称内存中创建：

```python
    import torch.distributed._symmetric_memory as symm_mem

    mempool = symm_mem.get_mem_pool(device)

    with torch.cuda.use_mem_pool(mempool):
        x = torch.arange(128, device=device)

    torch.ops.symm_mem.one_shot_all_reduce(x, "sum", group_name)
```

类似地，您可以将计算操作放在 MemPool 上下文中，结果张量也将从对称内存中创建。

```python
    dim = 1024
    w = torch.ones(dim, dim, device=device)
    x = torch.ones(1, dim, device=device)

    mempool = symm_mem.get_mem_pool(device)
    with torch.cuda.use_mem_pool(mempool):
        # y 将位于对称内存中
        y = torch.mm(x, w)
```

截至 torch 2.11，`CUDA` 和 `NVSHMEM` 后端支持 MemPool。`NCCL` 后端的 MemPool 支持正在进行中。

(copy-engine-collectives)=

## 复制引擎集合通信

:::{note}
复制引擎集合通信需要 NCCL 2.28 或更高版本，以及支持点对点（P2P）访问的 GPU。
:::

复制引擎（CE）集合通信是 NCCL 集合操作的一种优化，它将数据移动卸载到 GPU 的复制引擎（DMA 引擎），而不是使用 CUDA 流式多处理器（SM）。这释放了 SM 用于计算工作，从而在分布式训练期间实现更好的通信与计算重叠。

要使用 CE 集合通信，您需要：

1. 使用零 CTA 策略配置 NCCL 进程组
2. 使用 NCCL 后端设置对称内存
3. 使用对称内存分配张量
4. 通过会合将张量注册到对称内存

设置完成后，像 {func}`all_gather_into_tensor` 和 {func}`all_to_all_single` 这样的标准集合函数在操作对称内存张量时将自动使用复制引擎。

**示例**

```
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# 为 CE 集合通信使用零 CTA 策略初始化进程组
opts = dist.ProcessGroupNCCL.Options()
opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
device = torch.device("cuda", rank)
dist.init_process_group(backend="nccl", pg_options=opts, device_id=device)

# 使用 NCCL 后端设置对称内存
symm_mem.set_backend("NCCL")
group_name = dist.group.WORLD.group_name
```

# 使用对称内存分配张量
numel = 1024 * 1024
inp = symm_mem.empty(numel, device=device)
out = symm_mem.empty(numel * world_size, device=device)

# 为对称内存操作注册张量
symm_mem.rendezvous(inp, group=group_name)
symm_mem.rendezvous(out, group=group_name)

# 使用复制引擎执行集体操作
# 现在操作在 DMA 引擎上运行，而非 SM 上
work = dist.all_gather_into_tensor(out, inp, async_op=True)
work.wait()
```

**优势**

- **SM 卸载**：通信在复制引擎上运行，释放 SM 用于计算
- **更好的重叠**：实现更高效的计算/通信重叠
- **透明 API**：使用相同的集体 API，仅需使用对称内存张量

**要求与限制**

- NCCL 版本 2.28 或更高
- GPU 必须启用点对点（P2P）访问
- 张量必须使用 {func}`torch.distributed._symmetric_memory` 分配并进行 rendezvous 操作
- NCCL 进程组必须配置为 `NCCL_CTA_POLICY_ZERO` 或设置环境变量 `NCCL_CTA_POLICY` 为 2
- 截至 NCCL 2.28，CE 集体操作无法在默认流中运行，因此需要使用 `async_op=True` 标志来激活 `ProcessGroupNCCL` 的内部流，或自行创建侧流

(higher-precision-reduction)=

## 高精度规约

当张量使用对称内存分配时，NCCL 的对称内核实现支持内部使用更高精度进行规约。例如，对于 BF16 输入，NCCL 将在内部自动使用 FP32 进行累加，然后产生 BF16 输出（BF16 输入 → FP32 内部累加 → BF16 输出）。这提高了规约操作的数值精度，而无需更改集体调用。

**适用范围**

- **适用操作**：仅限 ``reduce_scatter`` 和 ``all_reduce``
- **域**：在 torch 2.9（NCCL 2.27）中仅限于 NVLink 域内；在 torch 2.11（NCCL 2.29）中，``reduce_scatter`` 支持 NVLink + 网络域
- **精度**：BF16/FP16 输入 → FP32 内部累加 → BF16/FP16 输出

**示例**

```python
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# 使用 NCCL 对称内存分配张量
symm_mem.set_backend("NCCL")
inp = symm_mem.empty(1024, 1024, device=device, dtype=torch.bfloat16)
symm_mem.rendezvous(inp, group_name)

# 对对称内存张量执行 reduce_scatter 和 all_reduce
# 自动受益于 FP32 内部累加
dist.all_reduce(inp)
```

:::{note}
当使用对称内存张量时，NCCL 会透明地启用此高精度累加。除了上述对称张量创建和 rendezvous 操作外，无需额外配置。目前这仅适用于支持域内的 ``reduce_scatter`` 和 ``all_reduce`` 操作；其他集体操作（例如 ``all_gather``）和节点间通信不受影响。
:::

## API 参考

```{eval-rst}
.. currentmodule:: torch.distributed._symmetric_memory
```

```{eval-rst}
.. autofunction:: empty
```

```{eval-rst}
.. autofunction:: rendezvous
```

```{eval-rst}
.. autofunction:: is_nvshmem_available
```

```{eval-rst}
.. autofunction:: set_backend
```

```{eval-rst}
.. autofunction:: get_backend
```

```{eval-rst}
.. autofunction:: get_mem_pool
```

## 操作参考
:::{note}
以下操作位于 `torch.ops.symm_mem` 命名空间中。您可以通过 `torch.ops.symm_mem.<op_name>` 直接调用它们。
:::

```{eval-rst}
.. currentmodule:: torch.ops.symm_mem
```

```{eval-rst}
.. py:function:: multimem_all_reduce_(input: Tensor, reduce_op: str, group_name: str) -> Tensor

    对输入张量执行 multimem 全规约操作。此操作需要硬件支持 multimem 操作。在 NVIDIA GPU 上，需要 NVLink SHARP。

    :param Tensor input: 要执行全规约的输入张量。必须是对称的。
    :param str reduce_op: 要执行的规约操作。目前仅支持 "sum"。
    :param str group_name: 要执行全规约的组名。


.. py:function:: multimem_all_gather_out(input: Tensor, group_name: str, out: Tensor) -> Tensor

    对输入张量执行 multimem 全收集操作。此操作需要硬件支持 multimem 操作。在 NVIDIA GPU 上，需要 NVLink SHARP。

    :param Tensor input: 要执行全收集的输入张量。
    :param str group_name: 要执行全收集的组名。
    :param Tensor out: 用于存储全收集操作结果的输出张量。必须是对称的。


.. py:function:: one_shot_all_reduce(input: Tensor, reduce_op: str, group_name: str) -> Tensor

    对输入张量执行单次全规约操作。

    :param Tensor input: 要执行全规约的输入张量。必须是对称的。
    :param str reduce_op: 要执行的规约操作。目前仅支持 "sum"。
    :param str group_name: 要执行全规约的组名。


.. py:function:: one_shot_all_reduce_out(input: Tensor, reduce_op: str, group_name: str, out: Tensor) -> Tensor

    基于输入张量执行单次全规约操作，并将结果写入输出张量。

    :param Tensor input: 要执行全规约的输入张量。必须是对称的。
    :param str reduce_op: 要执行的规约操作。目前仅支持 "sum"。
    :param str group_name: 要执行全规约的组名。
    :param Tensor out: 用于存储全规约操作结果的输出张量。可以是常规张量。


.. py:function:: two_shot_all_reduce_(input: Tensor, reduce_op: str, group_name: str) -> Tensor

    对输入张量执行两次全规约操作。

    :param Tensor input: 要执行全规约的输入张量。必须是对称的。
    :param str reduce_op: 要执行的规约操作。目前仅支持 "sum"。
    :param str group_name: 要执行全规约的组名。

.. py:function:: all_to_all_vdev(input: Tensor, out: Tensor, in_splits: Tensor, out_splits_offsets: Tensor, group_name: str) -> None

    使用 NVSHMEM 执行 all-to-all-v 操作，切分信息由设备提供。

    :param Tensor input: 要执行 all-to-all 操作的输入张量。必须是对称的。
    :param Tensor out: 用于存储 all-to-all 操作结果的输出张量。必须是对称的。
    :param Tensor in_splits: 包含要发送给每个对等体的数据切分的张量。必须是对称的。大小必须为 (group_size,)。切分单位是第一个维度中的元素数量。
    :param Tensor out_splits_offsets: 包含从每个对等体接收的数据的切分和偏移量的张量。必须是对称的。大小必须为 (2, group_size)。行依次为：输出切分和输出偏移量。
    :param str group_name: 要执行 all-to-all 操作的组名。


.. py:function:: all_to_all_vdev_2d(input: Tensor, out: Tensor, in_splits: Tensor, out_splits_offsets: Tensor, group_name: str, [major_align: int = None]) -> None

    使用 NVSHMEM 执行二维 all-to-all-v 操作，切分信息由设备提供。在专家混合模型中，此操作可用于分发令牌。

    :param Tensor input: 要执行 all-to-all 操作的输入张量。必须是对称的。
    :param Tensor out: 用于存储 all-to-all 操作结果的输出张量。必须是对称的。
    :param Tensor in_splits: 包含要发送给每个专家的数据切分的张量。必须是对称的。大小必须为 (group_size * ne,)，其中 ne 是每个秩的专家数量。切分单位是第一个维度中的元素数量。
    :param Tensor out_splits_offsets: 包含从每个对等体接收的数据的切分和偏移量的张量。必须是对称的。大小必须为 (2, group_size * ne)。行依次为：输出切分和输出偏移量。
    :param str group_name: 要执行 all-to-all 操作的组名。
    :param int major_align: 每个专家输出块主维度的可选对齐方式。如果未提供，则假定对齐为 1。任何对齐调整都会反映在输出偏移量中。

    二维 AllToAllv 混洗操作如下图所示：
    (world_size = 2, ne = 2, 专家总数 = 4)::

      源: |       秩 0       |       秩 1       |
          | c0 | c1 | c2 | c3 | d0 | d1 | d2 | d3 |

      目标: |       秩 0       |       秩 1       |
           | c0 | d0 | c1 | d1 | c2 | d2 | c3 | d3 |

    其中每个 `c_i` / `d_i` 是 `input` 张量的切片，目标专家为 `i`，长度由输入切分指示。也就是说，二维 AllToAllv 混洗实现了从输入时的秩主序到输出时的专家主序的转置。

    如果 `major_align` 不为 1，则 c1、c2、c3 的输出偏移量将向上对齐到此值。例如，如果 c0 长度为 5，d0 长度为 7（总计 12），并且 `major_align` 设置为 16，则 c1 的输出偏移量将为 16。c2 和 c3 同理。此值对次要维度（即 d0、d1、d2 和 d3）的偏移量没有影响。
    注意：由于 cutlass 不支持空桶，如果对齐长度为 0，我们将其设置为 `major_align`。参见 https://github.com/pytorch/pytorch/issues/152668。


.. py:function:: all_to_all_vdev_2d_offset(Tensor input, Tensor out, Tensor in_splits_offsets, Tensor out_splits_offsets, str group_name) -> None

    执行二维 AllToAllv 混洗操作，输入切分和偏移信息由设备提供。输入偏移量不需要是输入切分的精确前缀和，即允许在切分块之间存在填充。但是，填充不会被传输到对等秩。

    在专家混合模型中，此操作可用于合并由并行秩上的专家处理过的令牌。此操作可视为 `all_to_all_vdev_2d` 操作（将令牌混洗给专家）的“反向”操作。

    :param Tensor input: 要执行 all-to-all 操作的输入张量。必须是对称的。
    :param Tensor out: 用于存储 all-to-all 操作结果的输出张量。必须是对称的。
    :param Tensor in_splits_offsets: 包含要发送给每个专家的数据的切分和偏移量的张量。必须是对称的。大小必须为 (2, group_size * ne)，其中 `ne` 是专家数量。行依次为：输入切分和输入偏移量。切分单位是第一个维度中的元素数量。
    :param Tensor out_splits_offsets: 包含从每个对等体接收的数据的切分和偏移量的张量。必须是对称的。大小必须为 (2, group_size * ne)。行依次为：输出切分和输出偏移量。
    :param str group_name: 要执行 all-to-all 操作的组名。


.. py:function:: tile_reduce(in_tile: Tensor, out_tile: Tensor, root: int, group_name: str, [reduce_op: str = 'sum']) -> None

    将二维切片从组内所有秩归约到指定的根秩。

    :param Tensor in_tile: 要归约的输入二维张量。必须是对称分配的。
    :param Tensor out_tile: 包含归约结果的输出二维张量。必须是对称的，并且与 `in_tile` 具有相同的形状、数据类型和设备。
    :param int root: 指定组中将接收归约结果的进程的秩。
    :param str group_name: 要在其中执行归约操作的对称内存进程组的名称。
    :param str reduce_op: 要执行的归约操作。目前仅支持 ``"sum"``。默认为 ``"sum"``。

    此函数归约来自组内所有成员的 `in_tile` 张量，并将结果写入根秩的 `out_tile` 中。所有秩都必须参与，并提供相同的 `group_name` 和张量形状。

    示例::

>>> # doctest: +SKIP
>>> # 对张量的右下象限进行归约
>>> tile_size = full_size // 2
>>> full_inp = symm_mem.empty(full_size, full_size)
>>> full_out = symm_mem.empty(full_size, full_size)
>>> s = slice(tile_size, 2 * tile_size)
>>> in_tile = full_inp[s, s]
>>> out_tile = full_out[s, s]
>>> torch.ops.symm_mem.tile_reduce(in_tile, out_tile, root=0, group_name)


.. py:function:: multi_root_tile_reduce(in_tiles: list[Tensor], out_tile: Tensor, roots: list[int], group_name: str, [reduce_op: str = 'sum']) -> None

    并发执行多个分块归约操作，每个分块归约到一个独立的根节点。

    : param list[Tensor] in_tiles: 输入张量列表。
    : param Tensor out_tile: 用于存放归约后分块的输出张量。
    : param list[int] roots: 根节点排名列表，每个排名按相同顺序对应 `in_tiles` 中的一个输入分块。一个排名不能多次作为根节点。
    : param str group_name: 用于集合操作的组名。
    : param str reduce_op: 要执行的归约操作。目前仅支持 "sum"。

    示例::

        >>> # doctest: +SKIP
        >>> # 将张量的四个象限分别归约到不同的根节点
        >>> tile_size = full_size // 2
        >>> full_inp = symm_mem.empty(full_size, full_size)
        >>> s0 = slice(0, tile_size)
        >>> s1 = slice(tile_size, 2 * tile_size)
        >>> in_tiles = [ full_inp[s0, s0], full_inp[s0, s1], full_inp[s1, s0], full_inp[s1, s1] ]
        >>> out_tile = symm_mem.empty(tile_size, tile_size)
        >>> roots = [0, 1, 2, 3]
        >>> torch.ops.symm_mem.multi_root_tile_reduce(in_tiles, out_tile, roots, group_name)

```