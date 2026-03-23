# 序列化语义

本文档描述了如何在 Python 中保存和加载 PyTorch 张量及模块状态，以及如何序列化 Python 模块以便在 C++ 中加载。

 contents
目录


## 保存和加载张量 {#saving-loading-tensors}

`torch.save`{.interpreted-text role="func"} 和 `torch.load`{.interpreted-text role="func"} 让您可以轻松地保存和加载张量：

    >>> t = torch.tensor([1., 2.])
    >>> torch.save(t, 'tensor.pt')
    >>> torch.load('tensor.pt')
    tensor([1., 2.])

按照惯例，PyTorch 文件通常使用 \'.pt\' 或 \'.pth\' 扩展名。

`torch.save`{.interpreted-text role="func"} 和 `torch.load`{.interpreted-text role="func"} 默认使用 Python 的 pickle 模块，因此您也可以将多个张量作为 Python 对象（如元组、列表和字典）的一部分进行保存：

    >>> d = {'a': torch.tensor([1., 2.]), 'b': torch.tensor([3., 4.])}
    >>> torch.save(d, 'tensor_dict.pt')
    >>> torch.load('tensor_dict.pt')
    {'a': tensor([1., 2.]), 'b': tensor([3., 4.])}

包含 PyTorch 张量的自定义数据结构，如果该数据结构可被 pickle 序列化，也可以被保存。

## 保存和加载张量会保留视图关系 {#preserve-storage-sharing}

保存张量会保留它们之间的视图关系：

    >>> numbers = torch.arange(1, 10)
    >>> evens = numbers[1::2]
    >>> torch.save([numbers, evens], 'tensors.pt')
    >>> loaded_numbers, loaded_evens = torch.load('tensors.pt')
    >>> loaded_evens *= 2
    >>> loaded_numbers
    tensor([ 1,  4,  3,  8,  5, 12,  7, 16,  9])

在底层，这些张量共享同一个"存储"。关于视图和存储的更多信息，请参阅 [Tensor Views](https://pytorch.org/docs/main/tensor_view.html)。

当 PyTorch 保存张量时，它会分别保存其存储对象和张量元数据。这是一个实现细节，未来可能会改变，但它通常可以节省空间，并让 PyTorch 轻松重建已加载张量之间的视图关系。例如，在上面的代码片段中，只有一个存储被写入 \'tensors.pt\' 文件。

然而，在某些情况下，保存当前的存储对象可能是不必要的，并且会创建过大的文件。在下面的代码片段中，一个比要保存的张量大得多的存储被写入文件：

    >>> large = torch.arange(1, 1000)
    >>> small = large[0:5]
    >>> torch.save(small, 'small.pt')
    >>> loaded_small = torch.load('small.pt')
    >>> loaded_small.storage().size()
    999

保存到 \'small.pt\' 的不是 [small]{.title-ref} 张量中的五个值，而是它与 [large]{.title-ref} 共享的存储中的 999 个值。

当保存的张量元素数量少于其存储对象时，可以通过先克隆张量来减小保存文件的大小。克隆张量会生成一个具有新存储对象的新张量，该存储对象仅包含张量中的值：

    >>> large = torch.arange(1, 1000)
    >>> small = large[0:5]
    >>> torch.save(small.clone(), 'small.pt')  # 保存 small 的一个克隆
    >>> loaded_small = torch.load('small.pt')
    >>> loaded_small.storage().size()
    5

然而，由于克隆的张量彼此独立，它们不再具有原始张量之间的视图关系。如果在保存小于其存储对象的张量时，文件大小和视图关系都很重要，那么在保存之前，必须小心地构造新的张量，使其存储对象的大小最小化，同时仍保持所需的视图关系。

## 保存和加载 torch.nn.Modules {#saving-loading-python-modules}

另请参阅：\`教程：保存和加载模块 \<https://pytorch.org/tutorials/beginner/saving_loading_models.html\>\`\_

在 PyTorch 中，模块的状态通常使用"状态字典"进行序列化。模块的状态字典包含其所有参数和持久化缓冲区：

    >>> bn = torch.nn.BatchNorm1d(3, track_running_stats=True)
    >>> list(bn.named_parameters())
    [('weight', Parameter containing: tensor([1., 1., 1.], requires_grad=True)),
     ('bias', Parameter containing: tensor([0., 0., 0.], requires_grad=True))]

    >>> list(bn.named_buffers())
    [('running_mean', tensor([0., 0., 0.])),
     ('running_var', tensor([1., 1., 1.])),
     ('num_batches_tracked', tensor(0))]

    >>> bn.state_dict()
    OrderedDict([('weight', tensor([1., 1., 1.])),
                 ('bias', tensor([0., 0., 0.])),
                 ('running_mean', tensor([0., 0., 0.])),
                 ('running_var', tensor([1., 1., 1.])),
                 ('num_batches_tracked', tensor(0))])

出于兼容性考虑，建议不要直接保存模块，而是只保存其状态字典。Python 模块甚至有一个函数 `~torch.nn.Module.load_state_dict`{.interpreted-text role="meth"}，可以从状态字典中恢复其状态：

    >>> torch.save(bn.state_dict(), 'bn.pt')
    >>> bn_state_dict = torch.load('bn.pt')
    >>> new_bn = torch.nn.BatchNorm1d(3, track_running_stats=True)
    >>> new_bn.load_state_dict(bn_state_dict)
    <All keys matched successfully>

请注意，状态字典首先使用 `torch.load`{.interpreted-text role="func"} 从文件中加载，然后使用 `~torch.nn.Module.load_state_dict`{.interpreted-text role="meth"} 恢复状态。

即使是自定义模块和包含其他模块的模块也有状态字典，并且可以使用这种模式：

    # 一个包含两个线性层的模块
    >>> class MyModule(torch.nn.Module):
          def __init__(self):
            super().__init__()
            self.l0 = torch.nn.Linear(4, 2)
            self.l1 = torch.nn.Linear(2, 1)

          def forward(self, input):
            out0 = self.l0(input)
            out0_relu = torch.nn.functional.relu(out0)
            return self.l1(out0_relu)

\>\>\> m = MyModule() \>\>\> m.state_dict() OrderedDict(\[(\'l0.weight\', tensor(\[\[ 0.1400, 0.4563, -0.0271, -0.4406\], \[-0.3289, 0.2827, 0.4588, 0.2031\]\])), (\'l0.bias\', tensor(\[ 0.0300, -0.1316\])), (\'l1.weight\', tensor(\[\[0.6533, 0.3413\]\])), (\'l1.bias\', tensor(\[-0.1112\]))\])

\>\>\> torch.save(m.state_dict(), \'mymodule.pt\') \>\>\> m_state_dict = torch.load(\'mymodule.pt\') \>\>\> new_m = MyModule() \>\>\> new_m.load_state_dict(m_state_dict) \<All keys matched successfully\>

## `torch.save` 的序列化文件格式 {#serialized-file-format}

自 PyTorch 1.6.0 起，除非用户设置 `_use_new_zipfile_serialization=False`，否则 `torch.save` 默认返回一个未压缩的 ZIP64 归档文件。

在此归档文件中，文件按以下顺序排列

``` text
checkpoint.pth
├── data.pkl
├── byteorder  # 在 PyTorch 2.1.0 中添加
├── data/
│   ├── 0
│   ├── 1
│   ├── 2
│   └── …
└── version
```

条目说明如下：

:   - `data.pkl` 是传递给 `torch.save` 的对象（排除其中包含的 `torch.Storage` 对象）的 pickle 结果
    - `byteorder` 包含保存时的 `sys.byteorder` 字符串（\"little\" 或 \"big\"）
    - `data/` 包含对象中的所有存储，每个存储都是一个单独的文件
    - `version` 包含保存时的版本号，可在加载时使用

保存时，PyTorch 将确保每个文件的本地文件头填充到 64 字节的倍数偏移量，从而确保每个文件的偏移量是 64 字节对齐的。

 note
 title
Note


某些设备（如 XLA）上的张量会被序列化为 pickled numpy 数组。因此，它们的存储不会被序列化。在这些情况下，检查点中可能不存在 `data/` 目录。


## 布局控制 {#layout-control}

`torch.load`{.interpreted-text role="func"} 中的 `mmap` 参数允许对张量存储进行惰性加载。

此外，还有一些高级功能允许对 `torch.save` 检查点进行更细粒度的控制和操作。

`torch.serialization.skip_data`{.interpreted-text role="class"} 上下文管理器支持：

:   - 使用 `torch.save` 保存一个检查点，其中包含为稍后写入数据字节预留的空位。
    - 使用 `torch.load` 加载一个检查点，并在稍后填充张量的数据字节。

要在不分配存储数据内存的情况下检查 `torch.save` 检查点中的张量元数据，请在 `FakeTensorMode` 上下文管理器中使用 `torch.load`。除了像上面的 `skip_data` 一样跳过加载存储数据外，它还会用它们在检查点内的偏移量标记存储，从而支持直接操作检查点。

``` python
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

m = nn.Linear(10, 10)
torch.save(m.state_dict(), "checkpoint.pt")

with FakeTensorMode() as mode:
    fake_sd = torch.load("checkpoint.pt")

for k, v in fake_sd.items():
    print(f"key={k}, dtype={v.dtype}, shape={v.shape}, stride={v.stride()}, storage_offset={v.storage_offset()}")
    # 存储体在检查点中的偏移量
    print(f"key={k}, checkpoint_offset={v.untyped_storage()._checkpoint_offset}")
```

更多信息，请参阅 [此教程](https://docs.pytorch.org/tutorials/prototype/gpu_direct_storage.html)，它提供了一个使用这些功能操作检查点的完整示例。

## 使用 `weights_only=True` 的 `torch.load` {#weights-only}

从版本 2.6 开始，如果未传递 `pickle_module` 参数，`torch.load` 将使用 `weights_only=True`。

### weights_only 安全性 {#weights-only-security}

正如 `torch.load`{.interpreted-text role="func"} 文档中所讨论的，`weights_only=True` 会限制 `torch.load` 中使用的 unpickler 仅执行普通 `torch.Tensors` 的 `state_dicts` 以及一些其他原始类型所需的函数/构建类。此外，与 `pickle` 模块提供的默认 `Unpickler` 不同，`weights_only` Unpickler 不允许在 unpickling 过程中动态导入任何内容。

`weights_only=True` 缩小了远程代码执行攻击的攻击面，但存在以下限制：

1.  `weights_only=True` 不能防止拒绝服务攻击。
2.  我们试图防止在 `torch.load(weights_only=True)` 期间发生内存损坏，但它们仍有可能发生。

请注意，即使内存损坏没有在 `torch.load` 本身期间发生，加载过程也可能为下游代码创建意外的对象，这也可能导致内存损坏（例如，用户代码中为稀疏张量创建的索引和值张量可能会越界写入/读取）。

### weights_only 允许列表 {#weights-only-allowlist}

如上所述，在使用 `torch.save` 时，保存模块的 `state_dict` 是最佳实践。如果要加载包含 `nn.Module` 的旧检查点，我们建议使用 `weights_only=False`。当加载包含张量子类的检查点时，很可能会有需要加入允许列表的函数/类，更多细节请参见下文。

如果 `weights_only` Unpickler 在 pickle 文件中遇到默认未加入允许列表的函数或类，您应该会看到类似以下的可操作错误信息

``` 
```

[pickle.UnpicklingError]{#pickle.unpicklingerror}: 仅加载权重失败。此文件仍可加载，为此您有两种选择，仅在您信任该检查点来源的情况下执行这些步骤。

:   1.  

        重新运行 [torch.load]{.title-ref} 并将 [weights_only]{.title-ref} 设置为 [False]{.title-ref} 可能会成功，

        :   但这可能导致任意代码执行。仅当您从受信任的来源获取该文件时才这样做。

    2.  或者，要使用 [weights_only=True]{.title-ref} 加载，请检查以下错误消息中推荐的步骤。 WeightsUnpickler 错误: 不支持的全局变量: GLOBAL {\_\_module\_\_}.{\_\_name\_\_} 默认不是允许的全局变量。 如果您信任此类/函数，请使用 [torch.serialization.add_safe_globals(\[{\_\_name\_\_}\])]{.title-ref} 或 [torch.serialization.safe_globals(\[{\_\_name\_\_}\])]{.title-ref} 上下文管理器将此全局变量加入允许列表。

请按照错误消息中的步骤操作，并且仅在您信任它们时才将函数或类加入允许列表。

要获取检查点中尚未加入允许列表的所有 GLOBAL（函数/类），您可以使用 `torch.serialization.get_unsafe_globals_in_checkpoint`{.interpreted-text role="func"}，它将返回一个格式为 `{__module__}.{__name__}` 的字符串列表。如果您信任这些函数/类，可以导入它们，并根据错误消息通过 `torch.serialization.add_safe_globals`{.interpreted-text role="func"} 或上下文管理器 `torch.serialization.safe_globals`{.interpreted-text role="class"} 将它们加入允许列表。

要访问用户允许列表中的函数/类列表，您可以使用 `torch.serialization.get_safe_globals`{.interpreted-text role="func"}， 要清除当前列表，请参阅 `torch.serialization.clear_safe_globals`{.interpreted-text role="func"}。

### 故障排除 `weights_only`

#### 获取不安全的全局变量

需要注意的是，`torch.serialization.get_unsafe_globals_in_checkpoint`{.interpreted-text role="func"} 是静态分析检查点， 某些类型可能在反序列化过程中动态构建，因此不会被 `torch.serialization.get_unsafe_globals_in_checkpoint`{.interpreted-text role="func"} 报告。 一个例子是 numpy 中的 `dtypes`。在 `numpy < 1.25` 版本中，将 `torch.serialization.get_unsafe_globals_in_checkpoint`{.interpreted-text role="func"} 报告的所有函数/类加入允许列表后，您可能会看到如下错误：

``` 
WeightsUnpickler 错误: 只能构建 Tensor、Parameter、OrderedDict 或通过 `add_safe_globals` 允许列表中的类型，
但得到了 <class 'numpy.dtype[float32]'>
```

这可以通过 `{add_}safe_globals([type(np.dtype(np.float32))])` 加入允许列表。

在 `numpy >=1.25` 版本中，您会看到：

``` 
WeightsUnpickler 错误: 只能构建 Tensor、Parameter、OrderedDict 或通过 `add_safe_globals` 允许列表中的类型，
但得到了 <class 'numpy.dtypes.Float32DType'>
```

这可以通过 `{add_}safe_globals([np.dtypes.Float32DType])` 加入允许列表。

#### 环境变量

有两个环境变量会影响 `torch.load` 的行为。如果无法访问 `torch.load` 的调用点，这些变量会很有帮助。

- `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1` 将覆盖所有 `torch.load` 调用点，强制使用 `weights_only=True`。
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 将使 `torch.load` 调用点使用 `weights_only=False` **仅当** `weights_only` 未作为参数传递时。

## 实用函数 {#utility functions}

以下实用函数与序列化相关：

 currentmodule
torch.serialization


 autofunction
register_package


 autofunction
get_crc32_options


 autofunction
set_crc32_options


 autofunction
get_default_load_endianness


 autofunction
set_default_load_endianness


 autofunction
get_default_mmap_options


 autofunction
set_default_mmap_options


 autofunction
add_safe_globals


 autofunction
clear_safe_globals


 autofunction
get_safe_globals


 autofunction
get_unsafe_globals_in_checkpoint


 autoclass
safe_globals


 autoclass
skip_data


## 配置 {#serialization config}

`torch.utils.serialization.config` 提供了一个全局配置，可以控制 `torch.save` 和 `torch.load` 的行为。

`torch.utils.serialization.config.save` 包含控制 `torch.save` 行为的选项。

> - `compute_crc32`: 是否计算并写入 zip 文件校验和（默认值: `True`）。 参见 `~torch.serialization.set_crc32_options`{.interpreted-text role="func"}。
> - `use_pinned_memory_for_d2h`: 对于传递给 `torch.save` 时位于加速器上的存储，是否在 `torch.save` 内将存储移动到 CPU 的固定内存或可分页内存。（默认值: `False`（即可分页））
> - `storage_alignment`: `torch.save` 期间检查点中存储的对齐方式（以字节为单位）。（默认值 `64`）

`torch.utils.serialization.config.load` 包含控制 `torch.load` 行为的选项。

> - `mmap`: 参见 `torch.load`{.interpreted-text role="func"} 中 `mmap` 参数的文档。 如果未显式传递给 `torch.load` 调用，此配置将设置 `torch.load` 的 `mmap` 行为（默认值: `False`）。
> - `endianness`: 参见 `~torch.serialization.set_default_load_endianness`{.interpreted-text role="func"}。 （默认值: `torch.serialization.LoadEndianness.NATIVE`）
> - `mmap_flags`: 参见 `~torch.serialization.set_default_mmap_options`{.interpreted-text role="class"}。 （默认值: `MAP_PRIVATE`）
> - `calculate_storage_offsets`: 如果此配置设置为 `True`，当使用 `torch.load(mmap=True)` 时，存储的偏移量将被计算，而不是通过随机读取来获取。这可以最小化随机读取，当通过网络加载文件时可能很有帮助。（默认值: `False`）
