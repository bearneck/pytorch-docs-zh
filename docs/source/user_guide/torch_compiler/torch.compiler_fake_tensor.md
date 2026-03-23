# Fake tensor

代码：[fake_tensor.py](https://github.com/pytorch/pytorch/blob/db4572dbf18f1cf50cf662547e272d3117063747/torch/_subclasses/fake_tensor.py)

## 动机

在进行 Dynamo 符号求值和编译器传递时，我们通常希望能够运行张量操作以了解输出的大小/数据类型/设备信息，而无需实际运行这些操作（或破坏已有的张量），因为实际运行会较慢（如果进行大量计算）并占用大量内存（编译器在编译程序时使用 GPU 内存是不理想的）。Fake tensor 在所有方面都类似于真实张量，只是它实际上不包含任何数据。例如，当我们进行 Dynamo 追踪时，需要追踪用户的 Tensor 代码并回答关于中间结果的查询（例如，如果用户对中间张量进行条件判断）。如果没有 fake tensor，我们将无法为这些查询提供准确的信息。

类似地，假设您想为张量存储元数据，例如在 FX IR 节点上（meta['val']）。您可以改为直接在节点上存储一个 fake tensor，这将为您提供张量所需的所有元数据，包括您可能未处理的细微信息（例如，别名关系）。

## 相关工作

- Meta tensor 是 device='meta' 的张量。这实际上满足了 fake tensor 的许多需求，但 meta tensor 不模拟设备，并且有时步长行为因设备而异，因此 fake tensor 通过这种方式可以获得更准确的信息。此外，meta tensor 是“全局的”（它们独立存在，类似于 CPU/CUDA 张量的存在方式），而 fake tensor 的作用域限定在 FakeTensorMode 内。
- Tensor 子类允许您继承 torch.Tensor 并自定义其行为。Fake tensor 是作为张量子类实现的；这意味着其大部分实现都在 Python 中！有关张量子类的更简单示例，请查看 [subclass_zoo](https://github.com/albanD/subclass_zoo/)。
- 动态形状允许您创建具有符号大小而不仅仅是具体大小的张量，并通过操作符号传播这些大小。动态形状在 ShapeEnv 中维护状态，该环境始终与 FakeTensorMode 关联（因此 fake tensor 也负责管理符号大小）。通常，当我们使用 PT2 编译子图时，存在与此编译关联的追踪上下文，其中包含（除其他外）一个 FakeTensorMode 和（可能）一个 ShapeEnv。

## 整体架构

所有 fake tensor 都与一个 FakeTensorMode 关联。由于 fake tensor 的主要用例是对真实张量进行分析，一般工作流程是：您有一批真实张量，分配一个 FakeTensorMode，然后使用 from_real_tensor 将所有真实张量转换为 fake tensor，接着对这些 fake tensor 进行操作。特别是，FakeTensorMode 维护一个持久化的备忘录表，将张量（和存储）映射到相同的存储。如果您多次对同一张量进行 fakeify，您将获得相同的 fake tensor；如果您对两个相互别名的张量进行 fakeify，您将获得两个别名相同 fake 存储的 fake tensor。FakeTensor 是张量子类，因此如果您对它们进行操作，您将自动获得一个 fake tensor，但通常您希望在激活 FakeTensorMode 的情况下对 fake tensor 进行操作（例如，如果您正在运行 FX 传递）；张量操作将自动开启 fake tensor 模式并重试。

Fake tensor 表示为 meta tensor 的 \_\_torch_dispatch\_\_ 张量子类。这意味着在底层，fake tensor 是 meta 设备张量；然后它们使用额外的可扩展性钩子，特别是 dispatch_device，来谎报张量的实际设备。这是早期 fake tensor 中较易出错的部分之一：有时，fake tensor 过于擅长谎称自己是 CPU/CUDA 等设备，您最终会调用 CPU 内核并尝试解引用数据指针，这显然无法工作。如果您在 fake tensor 代码中遇到段错误，这是您应该首先检查的问题：C++ 回溯是在 CPU 内核中（意外！）还是在 meta 内核中（预期！）？Meta 内核类似于真实内核，但它所做的只是分配输出，不进行任何数据计算。

张量子类必须定义如何实现各种操作。以下是通用的 fake tensor 配方：

- 在输入的 fake tensor 上运行 meta 内核，将它们重新解释为 meta tensor。这是通过一个神奇的上下文管理器 in_kernel_invocation_manager 完成的，它指示 PyTorch 将所有 fake tensor 视为其底层的 meta tensor，而不是将 fake tensor “解包”为 meta tensor（fake tensor 就是 meta tensor）。Fake tensor 以这种方式表示，以避免必须保持两组元数据同步（meta tensor 的元数据和 fake tensor 的元数据）；这种“是”关系确保只有一份规范的元数据副本。
- 如果您是工厂函数，您将改为使用 device='meta' 调用底层的工厂函数。
- 将生成的 meta tensor 转换为 fake tensor，计算张量的输出设备应该是什么（这通常是简单的，但有时并非如此，例如 CPU 标量提升或设备转换操作）。

## API：重要部分

非 PT2 用法（更多示例请查看 test/test_fake_tensor.py）：

```python
# 创建 fake 模式
from torch._subclasses.fake_tensor import FakeTensorMode
fake_mode = FakeTensorMode()
converter = fake_mode.fake_tensor_converter
# 对某些真实张量进行 fakeify
fake_x = converter.from_real_tensor(fake_mode, x)
with fake_mode:
    # 对 fake tensor 进行一些操作
    fake_y = fake_x * 2
    # 工厂操作在上下文管理器中自动进行 fakeify
    fake_z = torch.empty(20)
```

问：为什么需要真实张量作为输入？

在 PT2 上下文中，这是因为你通常是在即时编译，所以对于正在编译的图的所有输入，你已经拥有"真实"的输入，因为你在执行程序的同时进行编译。

PT2 在 AOTAutograd 之前的使用方式（这种情况不常见，你可能不需要这样做）：

```python
# 未启用 Fake 模式！
from torch._guards import detect_fake_mode
fake_mode = detect_fake_mode(args)
# 如果 fake_mode 不为 None
converter = fake_mode.fake_tensor_converter
fake_args = [converter.from_real_tensor(fake_mode, arg) for arg in args]
with fake_mode:
    ... # 如果需要，使用 fake args 执行操作...
```

detect_fake_mode 会搜索多个位置以尝试找到与生命周期关联的"那个" fake tensor 模式。通常它会从追踪上下文中获取。

PT2 在 AOTAutograd 之后的使用方式：

```python
# Fake 模式已启用！example_inputs 通常已经是 fake 的
# TODO：我们可能想改变这一点
# 仍然这样做以访问 fake 模式
fake_mode = detect_fake_mode(example_inputs)
# 但通常你不需要手动开启它
```

其他有用的功能：

```python
from torch._subclasses.fake_tensor import unset_fake_temporarily
with unset_fake_temporarily():
    ... # 这里 fake 模式被禁用，你可以进行真实张量计算
```

什么时候你可能想要禁用 fake tensor 模式？通常你不需要这样做。我们发现一个有用的特定场景是在 fake tensors 上实现常量传播：在这种情况下，即使我们处于 fake tensor 模式，也需要进行一些实际的张量计算。

```python
import FakeTensorProp from torch.fx.passes.fake_tensor_prop
gm: GraphModule
real_inputs: List[Tensor]
FakeTensorProp(gm).propagate(*real_inputs)
# 这将在所有 FX 节点的 meta['val'] 中填充 fake tensor
# 或者如果你已有现成的 fake 模式，你应该使用它
FakeTensorProp(gm, mode=fake_mode).propagate(*real_inputs)
# 如果你的输入已经是 fake 的，还有 propagate_dont_convert_inputs 方法
fake_inputs: List[FakeTensor]
FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(*fake_inputs)
```

## 详细信息

是否自动转换？
最初，如果你尝试在 FakeTensorMode 区域内对真实张量进行计算，FakeTensorMode 不会自动将它们 fake 化。这样做的动机是为了防止以下陷阱：

```python
with FakeTensorMode():
    real_tensor.t_()
```

这段代码应该做什么？如果我们实际修改了真实张量的元数据，那会令人惊讶。但与此同时，也没有任何明显的机会来创建 FakeTensor。因此我们保守地决定让它引发错误："在 FakeTensorMode 中使用非 Fake Tensor 输入调用运算符尚不支持。请先将所有张量转换为 FakeTensors。"

这个错误在实践中相当烦人。例如，假设你有一个真实的 nn.Module，并且想要通过它传递 fake tensors。你需要以某种方式将 nn.Module fake 化。这促使了 FakeCopyMode 的出现。

最终，我们放弃了并添加了自动 fake 化功能。然而，在 FakeTensorMode 的许多使用场景中，这仍然默认未启用。

fake tensor 上的元数据突变
如果你有一个 fake tensor，并且对其执行 t_() 操作，fake tensor 上的元数据会发生变化。这本身是合理的，但有时你也想将 fake tensors 存储为 FX 节点上的元数据；突变 fake tensor 是不好的，因为这会使旧的元数据失效！

实际上，这里存在一个根本性的矛盾，即 fake tensors 维护着极其精确的张量元数据，甚至包括对象标识。如果对象元数据在 FX 图中随时间变化，实际上没有任何方法来表示这种随时间的变化。大多数情况下，我们严肃的 FX 分析是在功能化的图上进行的，这些图没有这个问题，但偶尔你需要在非功能化的图上进行分析。也许将 fake tensor 放在 meta['val'] 中是一个错误。

## 关于张量子类

Fake tensor 同时使用了子类和模式张量子类模式，其中 FakeTensor.\_\_torch_dispatch\_\_ 启用与 fake tensor 关联的 FakeTensorMode，然后重新分发（依赖 FakeTensorMode 来完成繁重的工作）。如果 fake tensor 操作收到一个它不认识的子类参数，它将返回 NotImplemented，给其他子类一个先运行的机会（希望将其转换为普通的张量操作），然后再试一次。这可能导致无限循环。

## 每个单独的运算符是如何实现的？

不幸的是，任何给定的运算符可能在一组相当复杂的地方实现。需要了解的一些重要情况：

- 如果元素数量非常少，张量子类支持有限的常量传播（这有助于处理一些我们立即对此类张量调用 item() 的情况）。
- 出于性能原因，我们对某些运算符有一些快速路径实现，这些实现完全在 fake tensor 中完成。
- 如果你使用 @custom_op 生成自定义张量，这些将直接向 fake tensor 注册 impl_abstract。
- Fake tensor 本身对设备转换操作有一些硬编码的特殊情况。
- 如果没有元实现也没有任何分解，我们将生成真实的零填充张量，并尝试直接运行运算符以找出结果。如果运算符尝试使用数据进行索引，这可能导致段错误，因此对于自定义操作，我们默认不启用此功能。

## 转换器是如何工作的？

由于 fake tensors 用于对张量的确切属性非常敏感的情况，fake tensors 非常小心地进行转换，保留叶节点性质、requires_grad 性质、别名以及许多其他属性。大部分繁重的工作都在 MetaConverter 中完成。

## 性能特征

您可能会认为伪张量（fake tensor）很快，因为它们不执行任何张量计算。但在小张量尺寸下，我们实际上完全受限于开销，而且，伪张量是用 Python 实现的，我们通常需要做大量工作来完成单个张量操作（因为它们被实现为分解）。因此，伪张量在实践中实际上相当慢，尤其是在涉及符号形状时。目前我们在伪张量中有两个重要的快速路径，在实践中产生了显著差异：

- 逐点操作（pointwise ops）不经过 PrimTorch 分解，而是我们手动编写了它们的传播规则。
- 如果可能，我们应该这样做。

## 伪张量的伪张量？

有人希望将伪张量作为用户输入送入 PT2 堆栈，这意味着我们需要能够创建伪张量的伪张量。目前这并不真正支持，但也许实现起来不会太困难。

## 与动态形状的交互

每个 FakeTensorMode 都包含一个 ShapeEnv，用于跟踪所有符号形状信息。它们的生命周期通常是绑定的：共存亡。

因为 FakeTensorMode 有一个 ShapeEnv（而元实现没有），所以依赖于数据且需要分配未绑定 SymInt 的元函数存在于伪张量中。伪张量还负责记忆未绑定的 SymInts，因此，例如，如果您在同一个伪张量上调用 nonzero() 两次，您会得到相同的符号大小。

## 其他资源

[使用 FakeTensor 确定最大批处理大小的 Colab 教程](https://colab.research.google.com/drive/1zjAisRrc8R6uixKsrs1DRm3lwz5MWN68)