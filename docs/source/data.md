# torch.utils.data


PyTorch 数据加载工具的核心是 `torch.utils.data.DataLoader` 类。它代表了一个在数据集上的 Python 可迭代对象，支持：

- `映射式与可迭代式数据集 <dataset-types>`，
- `自定义数据加载顺序 <data-loading-order-and-sampler>`，
- `自动批处理 <loading-batched-and-non-batched-data>`，
- `单进程与多进程数据加载 <single-and-multi-process-data-loading>`，
- `自动内存固定 <memory-pinning>`。

这些选项通过 `~torch.utils.data.DataLoader` 的构造函数参数进行配置，其签名如下：

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

以下部分将详细描述这些选项的作用和用法。


## 数据集类型

`~torch.utils.data.DataLoader` 构造函数中最重要的参数是 `dataset`，它指示了要从中加载数据的数据集对象。PyTorch 支持两种不同类型的数据集：

- `映射式数据集 <map-style-datasets>`，
- `可迭代式数据集 <iterable-style-datasets>`。


### 映射式数据集

映射式数据集实现了 `__getitem__` 和 `__len__` 协议，并表示从（可能非整数的）索引/键到数据样本的映射。

例如，这样的数据集，当使用 `dataset[idx]` 访问时，可以从磁盘上的文件夹中读取第 `idx` 张图像及其对应的标签。

更多细节请参阅 `~torch.utils.data.Dataset`。


### 可迭代式数据集

可迭代式数据集是 `~torch.utils.data.IterableDataset` 子类的实例，它实现了 `__iter__` 协议，并表示数据样本上的一个可迭代对象。这种类型的数据集特别适用于随机读取成本高昂甚至不可能，以及批量大小取决于获取的数据的情况。

例如，这样的数据集，当调用 `iter(dataset)` 时，可以返回从数据库、远程服务器读取的数据流，甚至是实时生成的日志。

更多细节请参阅 `~torch.utils.data.IterableDataset`。


> 📝 **注意**
> 当将 `~torch.utils.data.IterableDataset` 与 `多进程数据加载 <multi-process-data-loading>` 一起使用时。同一个数据集对象会在每个工作进程上复制，因此必须对副本进行不同的配置以避免数据重复。请参阅 `~torch.utils.data.IterableDataset` 文档了解如何实现这一点。


## 数据加载顺序与 `~torch.utils.data.Sampler`

对于 `可迭代式数据集 <iterable-style-datasets>`，数据加载顺序完全由用户定义的可迭代对象控制。这使得实现分块读取和动态批量大小（例如，每次生成一个批处理样本）更加容易。

本节的其余部分涉及 `映射式数据集 <map-style-datasets>` 的情况。`torch.utils.data.Sampler` 类用于指定数据加载中使用的索引/键序列。它们表示数据集索引上的可迭代对象。例如，在常见的随机梯度下降（SGD）情况下，一个 `~torch.utils.data.Sampler` 可以随机打乱索引列表并每次生成一个索引，或者为小批量 SGD 生成少量索引。

将根据 `~torch.utils.data.DataLoader` 的 `shuffle` 参数自动构造一个顺序或随机采样器。或者，用户可以使用 `sampler` 参数来指定一个自定义的 `~torch.utils.data.Sampler` 对象，该对象每次生成下一个要获取的索引/键。

一个每次生成一批索引列表的自定义 `~torch.utils.data.Sampler` 可以作为 `batch_sampler` 参数传入。也可以通过 `batch_size` 和 `drop_last` 参数启用自动批处理。有关此内容的更多详细信息，请参阅 `下一节 <loading-batched-and-non-batched-data>`。


> 📝 **注意**
> `sampler` 和 `batch_sampler` 都与可迭代式数据集不兼容，因为此类数据集没有键或索引的概念。


## 加载批处理与非批处理数据

`~torch.utils.data.DataLoader` 支持通过参数 `batch_size`、`drop_last`、`batch_sampler` 和 `collate_fn`（它有一个默认函数）自动将单独获取的数据样本整理成批次。


### 自动批处理（默认）

这是最常见的情况，对应于获取一个数据小批量并将它们整理成批处理样本，即包含张量，其中一个维度是批次维度（通常是第一个维度）。

当 `batch_size`（默认为 `1`）不为 `None` 时，数据加载器会生成批处理样本，而不是单个样本。`batch_size` 和 `drop_last` 参数用于指定数据加载器如何获取数据集键的批次。对于映射式数据集，用户也可以指定 `batch_sampler`，它每次生成一个键列表。


> 📝 **注意**
> `batch_size` 和 `drop_last` 参数本质上用于从 `sampler` 构造一个 `batch_sampler`。对于映射式数据集，`sampler` 由用户提供或基于 `shuffle` 参数构造。对于可迭代式数据集，`sampler` 是一个虚拟的无限采样器。有关采样器的更多详细信息，请参阅 `本节 <data-loading-order-and-sampler>`。


> 📝 **注意**
> 当从 `可迭代式数据集 <iterable-style-datasets>` 使用 `多进程 <multi-process-data-loading>` 加载数据时，`drop_last` 参数会丢弃每个工作进程数据集副本的最后一个不完整的批次。


在使用采样器（sampler）提供的索引获取样本列表后，会调用作为 `collate_fn` 参数传入的函数，将样本列表整理成批次。

在这种情况下，从映射式数据集加载大致相当于：

```python
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])
```

而从可迭代式数据集加载大致相当于：

```python
dataset_iter = iter(dataset)
for indices in batch_sampler:
    yield collate_fn([next(dataset_iter) for _ in indices])
```

可以使用自定义的 `collate_fn` 来定制整理过程，例如，将序列数据填充到批次中的最大长度。更多关于 `collate_fn` 的信息，请参阅 `此部分 <dataloader-collate_fn>`。


### 禁用自动批处理

在某些情况下，用户可能希望在数据集代码中手动处理批处理，或者仅加载单个样本。例如，直接加载批处理数据可能更高效（例如，从数据库批量读取或读取连续的内存块），或者批大小依赖于数据，或者程序被设计为处理单个样本。在这些场景下，最好不要使用自动批处理（即使用 `collate_fn` 来整理样本），而是让数据加载器直接返回 `dataset` 对象的每个成员。

当 `batch_size` 和 `batch_sampler` 均为 `None` 时（`batch_sampler` 的默认值已经是 `None`），自动批处理将被禁用。从 `dataset` 获取的每个样本都会通过作为 `collate_fn` 参数传入的函数进行处理。

**当自动批处理被禁用时**，默认的 `collate_fn` 仅将 NumPy 数组转换为 PyTorch 张量，并保持其他所有内容不变。

在这种情况下，从映射式数据集加载大致相当于：

```python
for index in sampler:
    yield collate_fn(dataset[index])
```

而从可迭代式数据集加载大致相当于：

```python
for data in iter(dataset):
    yield collate_fn(data)
```

更多关于 `collate_fn` 的信息，请参阅 `此部分 <dataloader-collate_fn>`。


### 使用 `collate_fn`

`collate_fn` 的使用在启用或禁用自动批处理时略有不同。

**当自动批处理被禁用时**，`collate_fn` 会与每个单独的数据样本一起被调用，其输出会从数据加载器迭代器中产生。在这种情况下，默认的 `collate_fn` 仅将 NumPy 数组转换为 PyTorch 张量。

**当自动批处理被启用时**，`collate_fn` 每次会与一个数据样本列表一起被调用。它需要将输入样本整理成一个批次，以便从数据加载器迭代器中产生。本节的其余部分描述了默认 `collate_fn` (`~torch.utils.data.default_collate`) 的行为。

例如，如果每个数据样本由一个 3 通道图像和一个整数类标签组成，即数据集的每个元素返回一个元组 `(image, class_index)`，那么默认的 `collate_fn` 会将这样的元组列表整理成一个包含批处理图像张量和批处理类标签张量的单个元组。具体来说，默认的 `collate_fn` 具有以下特性：

- 它总是添加一个新的维度作为批次维度。
- 它会自动将 NumPy 数组和 Python 数值转换为 PyTorch 张量。
- 它会保留数据结构，例如，如果每个样本是一个字典，它会输出一个具有相同键集但值为批处理张量（如果值无法转换为张量，则为列表）的字典。对于 `list`、`tuple`、`namedtuple` 等也是如此。

用户可以使用自定义的 `collate_fn` 来实现自定义批处理，例如，沿第一个维度以外的维度进行整理、填充不同长度的序列，或添加对自定义数据类型的支持。

如果您遇到 `~torch.utils.data.DataLoader` 的输出维度或类型与您预期不同的情况，您可能需要检查您的 `collate_fn`。


## 单进程与多进程数据加载

`~torch.utils.data.DataLoader` 默认使用单进程数据加载。

在 Python 进程中，[全局解释器锁 (GIL)](https://wiki.python.org/moin/GlobalInterpreterLock) 会阻止跨线程真正完全并行化 Python 代码。为了避免数据加载阻塞计算代码，PyTorch 提供了一个简单的开关来执行多进程数据加载，只需将参数 `num_workers` 设置为一个正整数即可。


### 单进程数据加载（默认）

在此模式下，数据获取在与初始化 `~torch.utils.data.DataLoader` 相同的进程中完成。因此，数据加载可能会阻塞计算。然而，当用于进程间共享数据的资源（例如，共享内存、文件描述符）有限，或者整个数据集很小且可以完全加载到内存中时，可能更倾向于使用此模式。此外，单进程加载通常能显示更易读的错误跟踪信息，因此对调试很有用。


### 多进程数据加载

将参数 `num_workers` 设置为一个正整数，将启用具有指定数量加载器工作进程的多进程数据加载。


> ⚠️ **警告**
> 经过多次迭代后，加载器工作进程将消耗与父进程相同数量的 CPU 内存，用于父进程中所有被工作进程访问的 Python 对象。如果 Dataset 包含大量数据（例如，在 Dataset 构建时加载了一个非常大的文件名列表）和/或使用了大量工作进程（总内存使用量为 `工作进程数 * 父进程大小`），这可能会带来问题。最简单的解决方法是使用非引用计数的表示形式（如 Pandas、Numpy 或 PyArrow 对象）替换 Python 对象。有关此问题发生原因的更多详细信息以及如何解决这些问题的示例代码，请查看 [issue #13246](https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662)。


在此模式下，每次创建 `~torch.utils.data.DataLoader` 的迭代器时（例如，调用 `enumerate(dataloader)` 时），会创建 `num_workers` 个工作进程。此时，`dataset`、`collate_fn` 和 `worker_init_fn` 会传递给每个工作进程，用于初始化和获取数据。这意味着数据集访问及其内部 IO、转换（包括 `collate_fn`）都在工作进程中运行。

`torch.utils.data.get_worker_info()` 在工作进程中返回各种有用信息（包括工作进程 ID、数据集副本、初始种子等），在主进程中返回 `None`。用户可以在数据集代码和/或 `worker_init_fn` 中使用此函数来单独配置每个数据集副本，并确定代码是否在工作进程中运行。例如，这对于分片数据集特别有帮助。

对于映射式数据集，主进程使用 `sampler` 生成索引并将其发送给工作进程。因此，任何洗牌随机化都在主进程中完成，通过分配要加载的索引来指导加载。

对于可迭代式数据集，由于每个工作进程都获得 `dataset` 对象的副本，简单的多进程加载通常会导致数据重复。使用 `torch.utils.data.get_worker_info()` 和/或 `worker_init_fn`，用户可以独立配置每个副本。（有关如何实现此操作，请参阅 `~torch.utils.data.IterableDataset` 文档。）出于类似的原因，在多进程加载中，`drop_last` 参数会丢弃每个工作进程的可迭代式数据集副本的最后一个不完整的批次。

一旦迭代结束或迭代器被垃圾回收，工作进程就会被关闭。


> ⚠️ **警告**
> 通常不建议在多进程加载中返回 CUDA 张量，因为在多进程中使用 CUDA 和共享 CUDA 张量存在许多细微差别（参见 `multiprocessing-cuda-note`）。相反，我们建议使用 `自动内存固定 <memory-pinning>`（即设置 `pin_memory=True`），这样可以实现向支持 CUDA 的 GPU 快速传输数据。


#### 平台特定行为

由于工作进程依赖于 Python 的 {py:mod}`multiprocessing`，Windows 上的工作进程启动行为与 Unix 不同。

- 在 Unix 上，默认的 {py:mod}`multiprocessing` 启动方法对于 Python >= 3.14 是 `forkserver`；对于 Python < 3.14 是 `fork`。使用 `fork` 时，子工作进程通常可以通过克隆的地址空间直接访问 `dataset` 和 Python 参数函数。这可以快速启动，但可能导致多线程应用程序出现问题。在支持它的 Unix 平台上，`forkserver` 首先启动一个单独的服务器进程，然后由该服务器生成所有新的工作进程，提供比 `fork` 更安全的隔离（尤其是在处理线程时），同时避免了纯 `spawn` 的一些开销。
- 在 Windows 和 MacOS 上，`spawn` 是默认的 {py:mod}`multiprocessing` 启动方法。使用 `spawn` 时，会启动另一个解释器来运行您的主脚本，然后运行接收 `dataset`、`collate_fn` 和其他参数（通过 {py:mod}`pickle` 序列化）的内部工作函数。

这种单独的序列化意味着，在使用多进程数据加载时，您应采取两个步骤以确保与 Windows 兼容：

- 将主脚本的大部分代码包装在 `if __name__ == '__main__':` 块中，以确保每个工作进程启动时不会再次运行（很可能产生错误）。您可以在此处放置数据集和 `~torch.utils.data.DataLoader` 实例的创建逻辑，因为不需要在工作进程中重新执行。
- 确保任何自定义的 `collate_fn`、`worker_init_fn` 或 `dataset` 代码都声明为顶级定义，位于 `__main__` 检查之外。这确保它们在工作进程中可用。（这是必需的，因为函数仅作为引用进行 pickle，而不是 `bytecode`。）


#### 多进程数据加载中的随机性

默认情况下，每个工作进程的 PyTorch 种子将设置为 `base_seed + worker_id`，其中 `base_seed` 是主进程使用其 RNG（从而强制消耗一个 RNG 状态）或指定的 `generator` 生成的一个长整数。但是，其他库的种子可能在初始化工作进程时被复制，导致每个工作进程返回相同的随机数。（参见常见问题解答中的 `此部分 <dataloader-workers-random-seed>`。）

在 `worker_init_fn` 中，您可以通过 `torch.utils.data.get_worker_info().seed <torch.utils.data.get_worker_info>` 或 `torch.initial_seed()` 访问为每个工作进程设置的 PyTorch 种子，并在数据加载之前使用它为其他库设置种子。


## 内存固定

当数据来自固定（页锁定）内存时，从主机到 GPU 的复制速度要快得多。有关何时以及如何使用固定内存的更多详细信息，请参阅 `cuda-memory-pinning`。

在数据加载方面，向 `~torch.utils.data.DataLoader` 传递 `pin_memory=True` 参数会自动将获取的数据张量放入锁页内存中，从而能够更快地传输到支持 CUDA 的 GPU。

默认的内存锁定逻辑仅识别张量以及包含张量的映射和可迭代对象。默认情况下，如果锁定逻辑遇到自定义类型的批次（当您使用的 `collate_fn` 返回自定义批次类型时会发生这种情况），或者如果批次的每个元素都是自定义类型，锁定逻辑将无法识别它们，并会返回未锁定内存的批次（或元素）。要为自定义批次或数据类型启用内存锁定，请在自定义类型上定义 `pin_memory` 方法。

请参阅以下示例。

示例：

```python
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # 自定义类型上的内存锁定方法
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
dataset = TensorDataset(inps, tgts)

loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                    pin_memory=True)

for batch_ndx, sample in enumerate(loader):
    print(sample.inp.is_pinned())
    print(sample.tgt.is_pinned())
```


% 这些模块目前作为 torch/data 的一部分进行文档记录，在此列出
% 直到我们有更清晰的解决方案


