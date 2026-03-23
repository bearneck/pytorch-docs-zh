# Tensor Creation API

本文档描述了如何在 PyTorch C++ API 中创建张量。它重点介绍了可用的工厂函数，这些函数根据特定算法填充新张量，并列出了可用于配置新张量形状、数据类型、设备和其他属性的选项。

## 工厂函数

*工厂函数* 是生成新张量的函数。PyTorch（包括 Python 和 C++）中有许多工厂函数，它们在返回新张量之前初始化张量的方式各不相同。所有工厂函数都遵循以下通用\"模式\"：

``` cpp
torch::<function-name>(<function-specific-options>, <sizes>, <tensor-options>)
```

让我们剖析这个\"模式\"的各个部分：

1.  `<function-name>` 是您要调用的函数名称，
2.  `<functions-specific-options>` 是特定工厂函数接受的任何必需或可选参数，
3.  `<sizes>` 是类型为 `IntArrayRef` 的对象，指定结果张量的形状，
4.  `<tensor-options>` 是 `TensorOptions` 的实例，配置结果张量的数据类型、设备、布局和其他属性。

### 选择工厂函数

截至撰写本文时，可用的工厂函数如下（超链接指向相应的 Python 函数，因为它们通常有更详细的文档------选项在 C++ 中是相同的）：

- [arange](https://pytorch.org/docs/stable/torch.html#torch.arange): 返回包含整数序列的张量，
- [empty](https://pytorch.org/docs/stable/torch.html#torch.empty): 返回包含未初始化值的张量，
- [eye](https://pytorch.org/docs/stable/torch.html#torch.eye): 返回单位矩阵，
- [full](https://pytorch.org/docs/stable/torch.html#torch.full): 返回填充单个值的张量，
- [linspace](https://pytorch.org/docs/stable/torch.html#torch.linspace): 返回在某个区间内线性间隔值的张量，
- [logspace](https://pytorch.org/docs/stable/torch.html#torch.logspace): 返回在某个区间内对数间隔值的张量，
- [ones](https://pytorch.org/docs/stable/torch.html#torch.ones): 返回填充全 1 的张量，
- [rand](https://pytorch.org/docs/stable/torch.html#torch.rand): 返回填充从 `[0, 1)` 均匀分布中抽取的值的张量。
- [randint](https://pytorch.org/docs/stable/torch.html#torch.randint): 返回填充从区间中随机抽取的整数的张量，
- [randn](https://pytorch.org/docs/stable/torch.html#torch.randn): 返回填充从单位正态分布中抽取的值的张量，
- [randperm](https://pytorch.org/docs/stable/torch.html#torch.randperm): 返回填充某个区间内整数随机排列的张量，
- [zeros](https://pytorch.org/docs/stable/torch.html#torch.zeros): 返回填充全 0 的张量。

### 指定大小

那些根据其填充张量的方式本质上不需要特定参数的函数，可以仅通过大小来调用。例如，以下代码创建一个包含 5 个分量的向量，初始值全部设置为 1：

``` cpp
torch::Tensor tensor = torch::ones(5);
```

如果我们想创建一个 `3 x 5` 矩阵，或者一个 `2 x 3 x 4` 张量呢？通常，`IntArrayRef`------工厂函数大小参数的类型------是通过在大括号中指定每个维度的大小来构造的。例如，`{2, 3}` 表示一个具有两行三列的张量（在这种情况下是矩阵），`{3, 4, 5}` 表示一个三维张量，`{2}` 表示一个具有两个分量的一维张量。在一维情况下，您可以省略大括号，像上面那样只传递单个整数。请注意，花括号只是构造 `IntArrayRef` 的一种方式。您也可以传递 `std::vector<int64_t>` 和其他几种类型。无论如何，这意味着我们可以通过以下方式构造一个填充了单位正态分布值的三维张量：

``` cpp
torch::Tensor tensor = torch::randn({3, 4, 5});
assert(tensor.sizes() == std::vector<int64_t>{3, 4, 5});
```

`tensor.sizes()` 返回一个 `IntArrayRef`，可以与 `std::vector<int64_t>` 进行比较，我们可以看到它包含我们传递给张量的大小。您也可以使用 `tensor.size(i)` 来访问单个维度，这等同于 `tensor.sizes()[i]`，但更推荐使用前者。

### 传递函数特定参数

`ones` 和 `randn` 都不接受任何额外参数来改变它们的行为。一个确实需要进一步配置的函数是 `randint`，它接受其生成的整数值的上限，以及一个可选的默认值为零的下限。这里我们创建一个 `5 x 5` 方阵，其整数在 0 到 10 之间：

``` cpp
torch::Tensor tensor = torch::randint(/*high=*/10, {5, 5});
```

这里我们将下限提高到 3：

``` cpp
torch::Tensor tensor = torch::randint(/*low=*/3, /*high=*/10, {5, 5});
```

行内注释 `/*low=*/` 和 `/*high=*/` 当然不是必需的，但它们有助于提高可读性，就像 Python 中的关键字参数一样。


> 💡 **提示**
> 主要要点是：大小参数总是跟在函数特定参数之后。


> ❗ **注意**
> 有时函数根本不需要大小参数。例如，`arange` 返回的张量的大小完全由其函数特定参数------整数范围的下限和上限------指定。在这种情况下，函数不接受 `size` 参数。
>
> ### 配置张量属性
>
> 上一节讨论了函数特定的参数。函数特定参数只能改变填充张量的值，有时也会改变张量的大小。它们从不改变所创建张量的数据类型（例如 [float32] 或 [int64]）或存储位置（CPU 或 GPU 内存）等属性。这些属性的指定留给了每个工厂函数的最后一个参数：一个 [TensorOptions] 对象，将在下文讨论。
>
> [TensorOptions] 是一个封装了张量构造轴的类。所谓\*构造轴\*，是指张量在构造前可以配置（有时在构造后可以更改）的特定属性。这些构造轴包括：
>
> - [dtype]（以前称为"标量类型"），控制存储在张量中元素的数据类型，
> - [layout]，可以是 strided（密集）或 sparse，
> - [device]，表示存储张量的计算设备（如 CPU 或 CUDA GPU），
> - [requires_grad] 布尔值，用于启用或禁用张量的梯度记录。
>
> 如果您熟悉 Python 中的 PyTorch，这些轴听起来会非常熟悉。目前这些轴允许的值为：
>
> - 对于 [dtype]：\`kUInt8\`、\`kInt8\`、\`kInt16\`、\`kInt32\`、\`kInt64\`、\`kFloat32\` 和 [kFloat64]，
> - 对于 [layout]：\`kStrided\` 和 [kSparse]，
> - 对于 [device]：\`kCPU\` 或 [kCUDA]（后者接受可选的设备索引），
> - 对于 [requires_grad]：\`true\` 或 [false]。


> 💡 **提示**
> 存在 dtype 的"Rust 风格"简写，例如用 [kF32] 代替 [kFloat32]。完整列表请参见 [此处](https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/types.h)。
>
> [TensorOptions] 的实例为每个轴存储一个具体值。以下是一个创建 [TensorOptions] 对象的示例，该对象表示一个 64 位浮点、strided 布局、需要梯度、存储在 CUDA 设备 1 上的张量：
>
> ``` cpp
> auto options =
>   torch::TensorOptions()
>     .dtype(torch::kFloat32)
>     .layout(torch::kStrided)
>     .device(torch::kCUDA, 1)
>     .requires_grad(true);
> ```
>
> 请注意，我们如何使用 [TensorOptions] 的"构建器"风格方法逐步构造对象。如果我们将此对象作为最后一个参数传递给工厂函数，新创建的张量将具有这些属性：
>
> ``` cpp
> torch::Tensor tensor = torch::full({3, 4}, /*value=*/123, options);
>
> assert(tensor.dtype() == torch::kFloat32);
> assert(tensor.layout() == torch::kStrided);
> assert(tensor.device().type() == torch::kCUDA); // 或 device().is_cuda()
> assert(tensor.device().index() == 1);
> assert(tensor.requires_grad());
> ```
>
> 现在，您可能会想：我是否真的需要为创建的每个新张量指定每个轴？幸运的是，答案是"不需要"，因为\*\*每个轴都有一个默认值\*\*。这些默认值为：
>
> - dtype 默认为 [kFloat32]，
> - layout 默认为 [kStrided]，
> - device 默认为 [kCPU]，
> - [requires_grad] 默认为 [false]。
>
> 这意味着在构造 [TensorOptions] 对象时，您省略的任何轴都将采用其默认值。例如，这是我们之前的 [TensorOptions] 对象，但 [dtype] 和 [layout] 使用了默认值：
>
> ``` cpp
> auto options = torch::TensorOptions().device(torch::kCUDA, 1).requires_grad(true);
> ```
>
> 实际上，我们甚至可以省略所有轴，得到一个完全使用默认值的 [TensorOptions] 对象：
>
> ``` cpp
> auto options = torch::TensorOptions(); // 或 `torch::TensorOptions options;`
> ```
>
> 一个很好的结果是，我们刚才讨论了很多的 [TensorOptions] 对象可以从任何张量工厂调用中完全省略：
>
> ``` cpp
> // 一个 32 位浮点、strided 布局、CPU 上、不需要梯度的张量。
> torch::Tensor tensor = torch::randn({3, 4});
> torch::Tensor range = torch::arange(5, 10);
> ```
>
> 但便利性更进一步：在目前介绍的 API 中，您可能已经注意到初始的 [torch::TensorOptions()] 写起来相当冗长。好消息是，对于每个构造轴（dtype、layout、device 和 [requires_grad]），在 [torch::] 命名空间中都有一个\*自由函数\*，您可以传递该轴的值。每个函数返回一个预先配置了该轴的 [TensorOptions] 对象，但允许通过上面展示的构建器风格方法进行进一步修改。例如，
>
> ``` cpp
> torch::ones(10, torch::TensorOptions().dtype(torch::kFloat32))
> ```
>
> 等价于
>
> ``` cpp
> torch::ones(10, torch::dtype(torch::kFloat32))
> ```
>
> 更进一步，代替
>
> ``` cpp
> torch::ones(10, torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided))
> ```
>
> 我们可以直接写
>
> ``` cpp
> torch::ones(10, torch::dtype(torch::kFloat32).layout(torch::kStrided))
> ```
>
> 这为我们节省了不少打字量。这意味着在实践中，您几乎不需要写出 [torch::TensorOptions]。而是使用 [torch::dtype()]、\`torch::device()\`、\`torch::layout()\` 和 [torch::requires_grad()] 函数。
>
> 最后一点便利是，\`TensorOptions\` 可以从单个值隐式构造。这意味着，每当函数有一个类型为 [TensorOptions] 的参数（所有工厂函数都是如此），我们可以直接传递像 [torch::kFloat32] 或 [torch::kStrided] 这样的值来代替完整的对象。因此，当我们只想改变单个轴相对于其默认值时，我们可以只传递该值。这样，
>
> ``` cpp
> torch::ones(10, torch::TensorOptions().dtype(torch::kFloat32))
> ```
>
> 变成了
>
> ``` cpp
> torch::ones(10, torch::dtype(torch::kFloat32))
> ```
>
> 并最终可以简化为
>
> ``` cpp
> torch::ones(10, torch::kFloat32)
> ```
>
> 当然，使用这种简写语法无法进一步修改 [TensorOptions] 实例的其他属性，但如果我们只需要更改一个属性，这已经相当实用了。
>
> 总而言之，我们现在可以比较 [TensorOptions] 的默认值，以及使用自由函数创建 [TensorOptions] 的简写 API，如何使 C++ 中的张量创建与 Python 中一样方便。比较以下 Python 调用：
>
> ``` python
> torch.randn(3, 4, dtype=torch.float32, device=torch.device('cuda', 1), requires_grad=True)
> ```
>
> 与等效的 C++ 调用：
>
> ``` cpp
> torch::randn({3, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 1).requires_grad(true))
> ```
>
> 非常接近！
>
> ## 转换
>
> 正如我们可以使用 [TensorOptions] 来配置应如何创建新张量一样，我们也可以使用 [TensorOptions] 将张量从一组属性转换为另一组属性。这种转换通常会创建一个新张量，并且不是原地进行的。例如，如果我们有一个通过以下方式创建的 [source_tensor]：
>
> ``` cpp
> torch::Tensor source_tensor = torch::randn({2, 3}, torch::kInt64);
> ```
>
> 我们可以将其从 [int64] 转换为 [float32]：
>
> ``` cpp
> torch::Tensor float_tensor = source_tensor.to(torch::kFloat32);
> ```


> ❗ **注意**
> 转换的结果 [float_tensor] 是一个指向新内存的新张量，与源张量 [source_tensor] 无关。
>
> 然后我们可以将其从 CPU 内存移动到 GPU 内存：
>
> ``` cpp
> torch::Tensor gpu_tensor = float_tensor.to(torch::kCUDA);
> ```
>
> 如果您有多个可用的 CUDA 设备，上述代码会将张量复制到\*默认\*的 CUDA 设备，您可以使用 [torch::DeviceGuard] 进行配置。如果没有设置 [DeviceGuard]，这将是 GPU 1。如果您想指定不同的 GPU 索引，可以将其传递给 [Device] 构造函数：
>
> ``` cpp
> torch::Tensor gpu_two_tensor = float_tensor.to(torch::Device(torch::kCUDA, 1));
> ```
>
> 在 CPU 到 GPU 复制及反向复制的情况下，我们还可以通过将 [/\*non_blocking=\*/false] 作为最后一个参数传递给 [to()] 来配置内存复制为\*异步\*：
>
> ``` cpp
> torch::Tensor async_cpu_tensor = gpu_tensor.to(torch::kCPU, /*non_blocking=*/true);
> ```
>
> ## 结论
>
> 希望本文档能让您很好地理解如何使用 PyTorch C++ API 以惯用的方式创建和转换张量。如果您有任何进一步的问题或建议，请使用我们的 [论坛](https://discuss.pytorch.org/) 或 [GitHub issues](https://github.com/pytorch/pytorch/issues) 与我们联系。

