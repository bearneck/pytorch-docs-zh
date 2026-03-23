
# 张量属性

每个 `torch.Tensor` 都拥有 `torch.dtype`、`torch.device` 和 `torch.layout` 属性。

## torch.dtype


`torch.dtype` 是一个表示 `torch.Tensor` 数据类型的对象。PyTorch 支持多种不同的数据类型：

**浮点数据类型**

  dtype                                                               描述
  ------------------------------------------------------------------- ----------------------------------------------------------------------------------------------------------------------------
  `torch.float32` 或 `torch.float`                                    32位浮点数，定义于 <https://en.wikipedia.org/wiki/IEEE_754>
  `torch.float64` 或 `torch.double`                                   64位浮点数，定义于 <https://en.wikipedia.org/wiki/IEEE_754>
  `torch.float16` 或 `torch.half`                                     16位浮点数，定义于 <https://en.wikipedia.org/wiki/IEEE_754，S-E-M> 1-5-10
  `torch.bfloat16`                                                    16位浮点数，有时称为 Brain 浮点数，S-E-M 1-8-7
  `torch.complex32` 或 `torch.chalf`                                  32位复数，包含两个 [float16] 分量
  `torch.complex64` 或 `torch.cfloat`                                 64位复数，包含两个 [float32] 分量
  `torch.complex128` 或 `torch.cdouble`                               128位复数，包含两个 [float64] 分量
  `torch.float8_e4m3fn` [\[shell\]](#shell),[^1]           8位浮点数，S-E-M 1-4-3，来自 <https://arxiv.org/abs/2209.05433>
  `torch.float8_e5m2` [\[shell\]](#shell)                  8位浮点数，S-E-M 1-5-2，来自 <https://arxiv.org/abs/2209.05433>
  `torch.float8_e4m3fnuz` [\[shell\]](#shell),[^2]         8位浮点数，S-E-M 1-4-3，来自 <https://arxiv.org/pdf/2206.02915>
  `torch.float8_e5m2fnuz` [\[shell\]](#shell),[^3]         8位浮点数，S-E-M 1-5-2，来自 <https://arxiv.org/pdf/2206.02915>
  `torch.float8_e8m0fnu` [\[shell\]](#shell),[^4]          8位浮点数，S-E-M 0-8-0，来自 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
  `torch.float4_e2m1fn_x2` [\[shell\]](#shell),[^5],[^6]   打包的4位浮点数，S-E-M 1-2-1，来自 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>

**整数数据类型**

  dtype                                                描述
  ---------------------------------------------------- --------------------
  `torch.uint8`                                        8位整数（无符号）
  `torch.int8`                                         8位整数（有符号）
  `torch.uint16` [\[shell\]](#shell),[^7]   16位整数（无符号）
  `torch.int16` 或 `torch.short`                       16位整数（有符号）
  `torch.uint32` [\[shell\]](#shell),[^8]   32位整数（无符号）
  `torch.int32` 或 `torch.int`                         32位整数（有符号）
  `torch.uint64` [\[shell\]](#shell),[^9]   64位整数（无符号）
  `torch.int64` 或 `torch.long`                        64位整数（有符号）
  `torch.bool`                                         布尔型

**注意**：遗留构造函数如 `torch.*.FloatTensor`、`torch.*.DoubleTensor`、`torch.*.HalfTensor`、 `torch.*.BFloat16Tensor`、`torch.*.ByteTensor`、`torch.*.CharTensor`、`torch.*.ShortTensor`、`torch.*.IntTensor`、 `torch.*.LongTensor`、`torch.*.BoolTensor` 仅为向后兼容而保留，不应再使用。

要判断一个 `torch.dtype` 是否为浮点数据类型，可以使用属性 `is_floating_point`，如果数据类型是浮点类型，则返回 `True`。

要判断一个 `torch.dtype` 是否为复数数据类型，可以使用属性 `is_complex`，如果数据类型是复数类型，则返回 `True`。


当算术操作（\`add\`、\`sub\`、\`div\`、\`mul\`）的输入数据类型不同时，我们会通过寻找满足以下规则的最小数据类型来进行类型提升：

- 如果标量操作数的类型类别高于张量操作数（类别顺序：复数 \> 浮点 \> 整数 \> 布尔值），我们会提升到足以容纳该类别所有标量操作数的类型。
- 如果零维张量操作数的类别高于有维度的操作数，我们会提升到具有足够大小和类别以容纳该类别所有零维张量操作数的类型。
- 如果没有更高类别的零维操作数，我们会提升到具有足够大小和类别以容纳所有有维度操作数的类型。

浮点标量操作数的 dtype 为 [torch.get_default_dtype()]，整数非布尔标量操作数的 dtype 为 [torch.int64]。与 numpy 不同，我们在确定操作数的最小 [dtypes] 时不检查值。目前不支持复数类型。未定义 shell dtypes 的提升规则。

提升示例:

    >>> float_tensor = torch.ones(1, dtype=torch.float)
    >>> double_tensor = torch.ones(1, dtype=torch.double)
    >>> complex_float_tensor = torch.ones(1, dtype=torch.complex64)
    >>> complex_double_tensor = torch.ones(1, dtype=torch.complex128)
    >>> int_tensor = torch.ones(1, dtype=torch.int)
    >>> long_tensor = torch.ones(1, dtype=torch.long)
    >>> uint_tensor = torch.ones(1, dtype=torch.uint8)
    >>> bool_tensor = torch.ones(1, dtype=torch.bool)
    # 零维张量
    >>> long_zerodim = torch.tensor(1, dtype=torch.long)
    >>> int_zerodim = torch.tensor(1, dtype=torch.int)

    >>> torch.add(5, 5).dtype
    torch.int64
    # 5 是 int64，但其类别不高于 int_tensor，因此不被考虑。
    >>> (int_tensor + 5).dtype
    torch.int32
    >>> (int_tensor + long_zerodim).dtype
    torch.int32
    >>> (long_tensor + int_tensor).dtype
    torch.int64
    >>> (bool_tensor + long_tensor).dtype
    torch.int64
    >>> (bool_tensor + uint_tensor).dtype
    torch.uint8
    >>> (float_tensor + double_tensor).dtype
    torch.float64
    >>> (complex_float_tensor + complex_double_tensor).dtype
    torch.complex128
    >>> (bool_tensor + int_tensor).dtype
    torch.int32
    # 由于 long 与 float 是不同类别，结果 dtype 只需足够大以容纳 float。
    >>> torch.add(long_tensor, float_tensor).dtype
    torch.float32

当指定了算术运算的输出张量时，我们允许强制转换到其 [dtype]，但以下情况除外：

:   - 整数输出张量不能接受浮点张量。
    - 布尔输出张量不能接受非布尔张量。
    - 非复数输出张量不能接受复数张量。

强制转换示例:

    # 允许：
    >>> float_tensor *= float_tensor
    >>> float_tensor *= int_tensor
    >>> float_tensor *= uint_tensor
    >>> float_tensor *= bool_tensor
    >>> float_tensor *= double_tensor
    >>> int_tensor *= long_tensor
    >>> int_tensor *= uint_tensor
    >>> uint_tensor *= int_tensor

    # 不允许（RuntimeError: 结果类型无法转换为所需的输出类型）：
    >>> int_tensor *= float_tensor
    >>> bool_tensor *= int_tensor
    >>> bool_tensor *= uint_tensor
    >>> float_tensor *= complex_float_tensor

## torch.device


`torch.device` 是一个表示 `torch.Tensor` 已分配或将要分配到的设备对象。

`torch.device` 包含一个设备类型（最常见的是 \"cpu\" 或 \"cuda\"，但也可能是 `"mps" <mps>`、`"xpu" <xpu>`、\`\"xla\" \<https://github.com/pytorch/xla/\>\`\_ 或 `"meta" <meta>`）以及可选的该设备类型的设备序号。如果未指定设备序号，此对象将始终表示该设备类型的当前设备，即使在调用 `torch.cuda.set_device()` 之后也是如此；例如，使用设备 `'cuda'` 构造的 `torch.Tensor` 等同于 `'cuda:X'`，其中 X 是 `torch.cuda.current_device()` 的结果。

可以通过 `Tensor.device` 属性访问 `torch.Tensor` 的设备。

`torch.device` 可以通过以下方式构造：

> - 设备字符串，即设备类型（可选包含设备序号）的字符串表示。
> - 设备类型和设备序号。
> - 设备序号，此时将使用当前的 `加速器<accelerators>` 类型。

通过设备字符串： :

    >>> torch.device('cuda:0')
    device(type='cuda', index=0)

    >>> torch.device('cpu')
    device(type='cpu')

    >>> torch.device('mps')
    device(type='mps')

    >>> torch.device('cuda')  # 隐式索引是“当前设备索引”
    device(type='cuda')

通过设备类型和设备序号：

    >>> torch.device('cuda', 0)
    device(type='cuda', index=0)

    >>> torch.device('mps', 0)
    device(type='mps', index=0)

    >>> torch.device('cpu', 0)
    device(type='cpu', index=0)

通过设备序号：


> 📝 **注意**
> 如果当前未检测到加速器，此方法将引发 RuntimeError。
>
>     >>> torch.device(0)  # 当前加速器是 cuda
>     device(type='cuda', index=0)
>
>     >>> torch.device(1)  # 当前加速器是 xpu
>     device(type='xpu', index=1)
>
>     >>> torch.device(0)  # 未检测到当前加速器
>     Traceback (most recent call last):
>       File "<stdin>", line 1, in <module>
>     RuntimeError: Cannot access accelerator device when none is available.
>
> 设备对象也可以用作上下文管理器，以更改分配张量的默认设备：
>
>     >>> with torch.device('cuda:1'):
>     ...     r = torch.randn(2, 3)
>     >>> r.device
>     device(type='cuda', index=1)
>
> 如果工厂函数被传递了显式的非 None 设备参数，此上下文管理器将不起作用。要全局更改默认设备，另请参阅 `torch.set_default_device`。


> ⚠️ **警告**
> 此函数会对每次调用 torch API（不仅仅是工厂函数）的 Python 调用施加轻微的性能开销。如果这给您带来了问题，请在 <https://github.com/pytorch/pytorch/issues/92701> 发表评论。


> 📝 **注意**
> 函数中的 `torch.device` 参数通常可以用字符串替代。 这有助于快速原型设计。
>
> \>\>\> \# 接受 torch.device 的函数示例 \>\>\> cuda1 = torch.device(\'cuda:1\') \>\>\> torch.randn((2,3), device=cuda1)
>
> \>\>\> \# 可以用字符串替代 torch.device \>\>\> torch.randn((2,3), device=\'cuda:1\')


> 📝 **注意**
> 接受设备参数的方法通常支持（格式正确的）字符串或整数设备序号，即以下写法完全等效：
>
> \>\>\> torch.randn((2,3), device=torch.device(\'cuda:1\')) \>\>\> torch.randn((2,3), device=\'cuda:1\') \>\>\> torch.randn((2,3), device=1) \# 如果当前加速器是 cuda，则等价于 \'cuda:1\'


> 📝 **注意**
> 张量永远不会在设备间自动移动，需要用户显式调用。标量张量（tensor.dim()==0）是此规则唯一的例外，它们在需要时会自动从 CPU 传输到 GPU，因为此操作可以\"免费\"完成。 示例：
>
> \>\>\> \# 两个标量 \>\>\> torch.ones(()) + torch.ones(()).cuda() \# 正常，标量自动从 CPU 传输到 GPU \>\>\> torch.ones(()).cuda() + torch.ones(()) \# 正常，标量自动从 CPU 传输到 GPU
>
> \>\>\> \# 一个标量（CPU），一个向量（GPU） \>\>\> torch.ones(()) + torch.ones(1).cuda() \# 正常，标量自动从 CPU 传输到 GPU \>\>\> torch.ones(1).cuda() + torch.ones(()) \# 正常，标量自动从 CPU 传输到 GPU
>
> \>\>\> \# 一个标量（GPU），一个向量（CPU） \>\>\> torch.ones(()).cuda() + torch.ones(1) \# 失败，标量不会自动从 GPU 传输到 CPU，且非标量不会自动从 CPU 传输到 GPU \>\>\> torch.ones(1) + torch.ones(()).cuda() \# 失败，标量不会自动从 GPU 传输到 CPU，且非标量不会自动从 CPU 传输到 GPU
>
> ## torch.layout


> **LAYOUT**


> ⚠️ **警告**
> `torch.layout` 类目前处于测试阶段，后续可能变更。
>
>
> `torch.layout` 是一个表示 `torch.Tensor` 内存布局的对象。目前我们支持 `torch.strided`（稠密张量），并对 `torch.sparse_coo`（稀疏 COO 张量）提供测试版支持。
>
> `torch.strided` 表示稠密张量，是最常用的内存布局。每个跨步张量都有一个关联的 `torch.Storage` 来存储其数据。这些张量提供了存储的多维 [跨步](https://en.wikipedia.org/wiki/Stride_of_an_array) 视图。跨步是一个整数列表：第 k 个跨步表示在张量的第 k 维中，从一个元素移动到下一个元素所需的内存跳跃量。这个概念使得许多张量操作能够高效执行。
>
> 示例:
>
>     >>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
>     >>> x.stride()
>     (5, 1)
>
>     >>> x.t().stride()
>     (1, 5)
>
> 关于 `torch.sparse_coo` 张量的更多信息，请参阅 `sparse-docs`。
>
> ## torch.memory_format
>
>
> `torch.memory_format` 是一个表示 `torch.Tensor` 已分配或将要分配的内存格式的对象。
>
> 可能的取值包括：
>
> - `torch.contiguous_format`： 张量已分配或将要分配在稠密非重叠内存中。跨步值按递减顺序表示。
> - `torch.channels_last`： 张量已分配或将要分配在稠密非重叠内存中。跨步值按 `strides[0] > strides[2] > strides[3] > strides[1] == 1` 即 NHWC 顺序表示。
> - `torch.channels_last_3d`： 张量已分配或将要分配在稠密非重叠内存中。跨步值按 `strides[0] > strides[2] > strides[3] > strides[4] > strides[1] == 1` 即 NDHWC 顺序表示。
> - `torch.preserve_format`： 用于 [clone] 等函数以保留输入张量的内存格式。如果输入张量分配在稠密非重叠内存中，输出张量的跨步将从输入复制。否则输出跨步将遵循 `torch.contiguous_format`。
>
>
> [shell]
>
> :   shell 数据类型是一种支持的操作和后端有限的特化数据类型。 具体来说，支持创建张量的操作（`torch.empty`、`torch.fill`、`torch.zeros`） 以及不窥探数据元素内部的操作（`torch.cat`、`torch.view`、`torch.reshape`）。 而窥探数据元素内部的操作，如类型转换、矩阵乘法、nan/inf 检查等，仅根据成熟度、硬件加速内核的存在性以及已确立的用例，在逐个案例的基础上提供支持。
>
> [^1]: \"fn\"、\"fnu\" 和 \"fnuz\" 数据类型后缀的含义： \"f\" - 仅有限值编码，无无穷大； \"n\" - nan 值编码与 IEEE 规范不同； \"uz\" - 仅\"无符号零\"，即无负零编码
>
> [^2]: \"fn\"、\"fnu\" 和 \"fnuz\" 数据类型后缀的含义： \"f\" - 仅有限值编码，无无穷大； \"n\" - nan 值编码与 IEEE 规范不同； \"uz\" - 仅\"无符号零\"，即无负零编码
>
> [^3]: \"fn\"、\"fnu\" 和 \"fnuz\" 数据类型后缀的含义： \"f\" - 仅有限值编码，无无穷大； \"n\" - nan 值编码与 IEEE 规范不同； \"uz\" - 仅\"无符号零\"，即无负零编码
>
> [^4]: \"fn\"、\"fnu\" 和 \"fnuz\" 数据类型后缀的含义： \"f\" - 仅有限值编码，无无穷大； \"n\" - nan 值编码与 IEEE 规范不同； \"uz\" - 仅\"无符号零\"，即无负零编码
>
> [^5]: \"fn\"、\"fnu\" 和 \"fnuz\" 数据类型后缀的含义： \"f\" - 仅有限值编码，无无穷大； \"n\" - nan 值编码与 IEEE 规范不同； \"uz\" - 仅\"无符号零\"，即无负零编码
>
> [^6]: [torch.float4_e2m1fn_x2] 数据类型表示两个打包在一个字节中的4位值。 请注意，修改张量形状/步幅的 PyTorch 操作（如转置）在字节边界上操作，并且\*\*不会\*\*解包/重新打包亚字节值。
>
> [^7]: 目前计划仅在 eager 模式下对除 `uint8` 之外的无符号类型提供有限支持（它们主要存在是为了辅助 torch.compile 的使用）；如果您需要 eager 模式支持且不需要额外的数值范围，我们建议使用其对应的有符号变体。更多详情请参阅 <https://github.com/pytorch/pytorch/issues/58734>。
>
> [^8]: 目前计划仅在 eager 模式下对除 `uint8` 之外的无符号类型提供有限支持（它们主要存在是为了辅助 torch.compile 的使用）；如果您需要 eager 模式支持且不需要额外的数值范围，我们建议使用其对应的有符号变体。更多详情请参阅 <https://github.com/pytorch/pytorch/issues/58734>。
>
> [^9]: 目前计划仅在 eager 模式下对除 `uint8` 之外的无符号类型提供有限支持（它们主要存在是为了辅助 torch.compile 的使用）；如果您需要 eager 模式支持且不需要额外的数值范围，我们建议使用其对应的有符号变体。更多详情请参阅 <https://github.com/pytorch/pytorch/issues/58734>。

