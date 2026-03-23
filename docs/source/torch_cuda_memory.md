(torch_cuda_memory)=

# 理解 CUDA 内存使用

为了调试 CUDA 内存使用，PyTorch 提供了一种生成内存快照的方法，该快照可以记录在任意时间点已分配的 CUDA 内存状态，并可选择记录导致该快照的分配事件历史。

生成的快照随后可以拖放到托管在 [pytorch.org/memory_viz](https://pytorch.org/memory_viz) 的交互式查看器中，用于探索快照内容。

```{note}
本文档描述的内存分析器和可视化工具仅能查看通过 PyTorch 分配器分配和管理的 CUDA 内存。任何直接从 CUDA API 分配的内存将不会在 PyTorch 内存分析器中可见。

NCCL（用于 CUDA 设备上的分布式通信）是一个常见示例，它分配了一些对 PyTorch 内存分析器不可见的 GPU 内存。更多信息请参阅 {ref}`non_pytorch_alloc`。
```

## 生成快照

记录快照的常见模式是：启用内存历史记录，运行要观察的代码，然后保存一个包含序列化快照的文件：

```python
# 启用内存历史记录，这将
# 为快照添加回溯跟踪和事件历史
torch.cuda.memory._record_memory_history()

run_your_code()
torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
```

## 使用可视化工具

打开 <https://pytorch.org/memory_viz> 并将序列化的快照文件拖放到可视化工具中。
可视化工具是一个在您计算机本地运行的 JavaScript 应用程序。它不会上传任何快照数据。

## 活动内存时间线

活动内存时间线显示了快照中特定 GPU 上随时间变化的所有存活张量。在图表上平移/缩放以查看较小的分配。
将鼠标悬停在已分配的块上可以查看该块分配时的堆栈跟踪，以及其地址等详细信息。可以调整细节滑块以减少渲染的分配数量，并在数据量很大时提高性能。

```{image} _static/img/torch_cuda_memory/active_memory_timeline.png
```

## 分配器状态历史

分配器状态历史在左侧的时间线中显示了个别的分配器事件。在时间线中选择一个事件，可以查看该事件发生时分配器状态的视觉摘要。此摘要显示了从 cudaMalloc 返回的每个独立段，以及它如何被分割成单个分配的块或空闲空间。将鼠标悬停在段和块上可以查看内存分配时的堆栈跟踪。将鼠标悬停在事件上可以查看事件发生时的堆栈跟踪，例如张量释放时。内存不足错误报告为 OOM 事件。查看 OOM 期间的内存状态可能有助于理解为何分配失败，即使预留内存仍然存在。

```{image} _static/img/torch_cuda_memory/allocator_state_history.png
```

堆栈跟踪信息还报告了分配发生时的地址。
地址 b7f064c000000_0 指的是地址为 7f064c000000 的 (b)lock，这是该地址第 "_0" 次被分配。
这个唯一的字符串可以在活动内存时间线中查找，并在活动状态历史中搜索，以检查张量分配或释放时的内存状态。

(non_pytorch_alloc)=
## 识别非 PyTorch 分配

如果您怀疑 CUDA 内存是在 PyTorch 之外分配的，您可以使用 pynvml 包收集原始的 CUDA 分配信息，并将其与 PyTorch 报告的内存分配进行比较。

要收集 PyTorch 之外的内存使用情况，请使用 {func}`device_memory_used`

```python
import torch
device_idx = ...
print(torch.cuda.device_memory_used(device_idx))
```

## 快照 API 参考

```{eval-rst}
.. currentmodule:: torch.cuda.memory
```

```{eval-rst}
.. autofunction:: _record_memory_history
```

```{eval-rst}
.. autofunction:: _snapshot
```

```{eval-rst}
.. autofunction:: _dump_snapshot
```