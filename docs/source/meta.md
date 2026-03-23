# Meta 设备

"meta" 设备是一种抽象设备，它表示仅记录元数据但不包含实际数据的张量。Meta 张量有两个主要用途：

* 模型可以加载到 meta 设备上，允许您加载模型的表示形式，而无需将实际参数加载到内存中。如果您需要在加载实际数据之前对模型进行转换，这会很有帮助。

* 大多数操作都可以在 meta 张量上执行，产生新的 meta 张量，描述如果您在真实张量上执行该操作会得到什么结果。您可以使用它来进行抽象分析，而无需花费时间进行计算或占用空间来表示实际张量。由于 meta 张量没有真实数据，您无法执行依赖于数据的操作，例如 `torch.nonzero` 或 `~torch.Tensor.item`。在某些情况下，并非所有设备类型（例如 CPU 和 CUDA）对于某个操作都有完全相同的输出元数据；在这种情况下，我们通常倾向于忠实地表示 CUDA 的行为。

```{warning}
尽管原则上 meta 张量计算应该总是比等效的 CPU/CUDA 计算更快，但许多 meta 张量实现是用 Python 编写的，尚未移植到 C++ 以获得速度提升，因此您可能会发现，对于小型 CPU 张量，您获得的绝对框架延迟更低。
```

## 使用 meta 张量的惯用法

可以通过指定 `map_location='meta'` 将对象加载到 meta 设备上：

```python
>>> torch.save(torch.randn(2), 'foo.pt')
>>> torch.load('foo.pt', map_location='meta')
tensor(..., device='meta', size=(2,))
```

如果您有一些任意代码执行张量构造而没有明确指定设备，您可以使用 `torch.device` 上下文管理器覆盖它以在 meta 设备上构造：

```python
>>> with torch.device('meta'):
...     print(torch.randn(30, 30))
...
tensor(..., device='meta', size=(30, 30))
```

这对于神经网络模块构造尤其有帮助，因为您通常无法显式传递设备进行初始化：

```python
>>> from torch.nn.modules import Linear
>>> with torch.device('meta'):
...     print(Linear(20, 30))
...
Linear(in_features=20, out_features=30, bias=True)
```

您不能直接将 meta 张量转换为 CPU/CUDA 张量，因为 meta 张量不存储数据，我们不知道新张量的正确数据值是什么：

```python
>>> torch.ones(5, device='meta').to("cpu")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NotImplementedError: Cannot copy out of meta tensor; no data!
```

使用像 `torch.empty_like` 这样的工厂函数来明确指定您希望如何填充缺失的数据。

神经网络模块有一个便捷方法 `torch.nn.Module.to_empty`，允许您将模块移动到另一个设备，同时将所有参数保持未初始化状态。您需要手动显式地重新初始化参数：

```python
>>> from torch.nn.modules import Linear
>>> with torch.device('meta'):
...     m = Linear(20, 30)
>>> m.to_empty(device="cpu")
Linear(in_features=20, out_features=30, bias=True)
```

`torch._subclasses.meta_utils` 包含未文档化的实用程序，用于获取任意张量并构造具有高保真度的等效 meta 张量。这些 API 是实验性的，可能会随时以破坏向后兼容性的方式更改。