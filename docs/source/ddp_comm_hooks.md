# DDP 通信钩子

DDP 通信钩子是一个通用接口，通过覆盖 [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) 中的标准 allreduce 操作，来控制如何在各个工作进程之间通信梯度。系统提供了若干内置的通信钩子，用户可以轻松应用其中任何一个来优化通信。此外，钩子接口也支持用户自定义的通信策略，以满足更高级的用例需求。

## 如何使用通信钩子？

要使用通信钩子，用户只需在训练循环开始前让 DDP 模型注册钩子，如下所示。

`torch.nn.parallel.DistributedDataParallel.register_comm_hook`

## 通信钩子操作什么？

通信钩子提供了一种灵活的方式来 allreduce 梯度。因此，它主要在 allreduce 之前操作每个副本上的梯度，这些梯度被分桶处理以增加通信与计算之间的重叠。具体来说，`torch.distributed.GradBucket` 表示一个待 allreduce 的梯度张量桶。


## 默认通信钩子

默认通信钩子是简单的**无状态**钩子，因此 `register_comm_hook` 中的输入状态要么是一个进程组，要么是 `None`。输入 `bucket` 是一个 `torch.distributed.GradBucket` 对象。


此外，还提供了一个通信钩子包装器，以支持将 `fp16_compress_hook` 或 `bf16_compress_hook` 作为包装器，这可以与其他通信钩子结合使用。


## PowerSGD 通信钩子

PowerSGD ([Vogels et al., NeurIPS 2019](https://arxiv.org/abs/1905.13727)) 是一种梯度压缩算法，它可以提供非常高的压缩率并加速受带宽限制的分布式训练。该算法需要维护一些超参数和内部状态。因此，PowerSGD 通信钩子是一个**有状态**的钩子，用户需要提供如下定义的状态对象。

### PowerSGD 状态


### PowerSGD 钩子

```{warning}
PowerSGD 通常需要与模型梯度大小相同的额外内存来启用误差反馈，这可以补偿有偏的压缩通信并提高准确性。
{warning}
PowerSGD 钩子可能与 [Apex 自动混合精度包](https://github.com/NVIDIA/apex) 冲突。请改用 PyTorch [原生自动混合精度包](https://pytorch.org/docs/stable/amp.html)。
```


## 调试通信钩子

顾名思义，调试通信钩子**仅**用于调试和性能优化目的。


```{warning}
调试通信钩子不一定输出正确的结果。
```


## 通信钩子的检查点


有状态的通信钩子可以作为模型检查点的一部分保存，以便支持训练器重启。为了使钩子可序列化，需要定义 `__setstate__` 和 `__getstate__` 方法。

```{warning}
`__getstate__` 应该从返回的字典中排除不可序列化的属性。
{warning}
`__setstate__` 应该正确初始化不可序列化的属性，这些属性在提供的 `state` 中被排除。
```

`PowerSGDState` 已经实现了 `__setstate__` 和 `__getstate__` 方法，可以作为参考。


以下是一个保存和重新加载 PowerSGD 状态及钩子的简单端到端示例。

```python

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24,24)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(24,12)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_demo(demo_fn, world_size):
    mp.spawn(
        demo_fn,
        args=(world_size,),
        nprocs=world_size,
        join=True)

def demo_serialization(rank, world_size):
    setup(rank, world_size)

    CHECKPOINT = tempfile.gettempdir() + "/checkpoint.pt"

    model = SimpleModel().to(rank)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])
```

powersgd_hook = powerSGD.powerSGD_hook
powersgd_state = powerSGD.PowerSGDState(process_group=None)

optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
ddp_model.register_comm_hook(powersgd_state, powersgd_hook)

state = {
    'state_dict': ddp_model.state_dict(),
    'comm_hook': powersgd_hook,
    'comm_hook_state': powersgd_state}

if rank == 0:
    torch.save(state, CHECKPOINT)

dist.barrier()
map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
checkpoint = torch.load(CHECKPOINT, map_location=map_location)

new_ddp_model = DistributedDataParallel(SimpleModel().to(rank), device_ids=[rank])
new_ddp_model.load_state_dict(checkpoint['state_dict'])
powersgd_hook = checkpoint['comm_hook']
powersgd_state = checkpoint['comm_hook_state']

new_ddp_model.register_comm_hook(powersgd_state, powersgd_hook)

if rank == 0:
    os.remove(CHECKPOINT)

cleanup()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_serialization, world_size)
```

## 致谢

衷心感谢 PowerSGD 论文作者 **Thijs Vogels** 对 PowerSGD 通信钩子的代码审查，以及提供的
[对比实验](https://observablehq.com/@tvogels/powersgd-benchmark)。
这些实验表明，PowerSGD 通信钩子的性能与原始
[论文](https://arxiv.org/abs/1905.13727) 中的实现相当。
