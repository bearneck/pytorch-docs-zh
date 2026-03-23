# 加速器钩子

## 背景

加速器钩子是将自定义加速器设备集成到 PyTorch 运行时的机制。

## 设计

下表列出了加速器供应商在集成新设备后端时应实现的钩子。这些钩子分为两个优先级级别：

- **高优先级钩子**：PyTorch 运行时直接依赖的核心 API。供应商应实现所有高优先级钩子，以确保核心兼容性和基本设备功能。

- **低优先级钩子**：PyTorch 不直接依赖的设备管理和实用程序 API。这些钩子可增强用户体验和多设备支持，是可选的。供应商可以根据特定需求和使用场景来实现它们。

### 高优先级钩子

| 钩子方法                        | 描述                                               | 应用场景                                                            |
| ------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------ |
| `init()`                           | 初始化加速器运行时和设备上下文   | 当 PyTorch 首次访问设备时设置必要的状态                    |
| `hasPrimaryContext(DeviceIndex)`   | 检查设备是否存在主上下文         | 确定设备初始化是否已发生                             |
| `getDefaultGenerator(DeviceIndex)` | 返回设备的默认随机数生成器  | 访问设备的主 RNG 以进行可重现的随机操作               |
| `getNewGenerator(DeviceIndex)`     | 创建新的独立随机数生成器         | 为并行操作创建隔离的 RNG 实例                            |
| `getDeviceFromPtr(void*)`          | 确定内存指针属于哪个设备       | 识别与内存分配关联的加速器设备              |
| `getPinnedMemoryAllocator()`       | 返回用于固定（页锁定）主机内存的分配器 | 分配可以高效传输到/从加速器的主机内存 |
| `isPinnedPtr(void*)`               | 检查指针是否指向固定内存               | 在执行操作前验证内存类型                               |

### 低优先级钩子

| 钩子方法                        | 描述                                                                  | 应用场景                                                |
| ------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------ |
| `isBuilt()`                        | 返回加速器后端是否已构建/编译到扩展中 | 检查加速器库在编译时是否可用   |
| `isAvailable()`                    | 返回加速器硬件在运行时是否可用             | 验证是否可以检测和初始化加速器设备   |
| `deviceCount()`                    | 返回可用加速器设备的数量                          | 枚举所有可用加速器设备以供设备选择     |
| `setCurrentDevice(DeviceIndex)`    | 设置当前线程的活动设备                                | 将当前线程的上下文切换到特定的加速器设备 |
| `getCurrentDevice()`               | 返回当前活动设备的索引                                    | 查询当前线程中哪个加速器设备处于活动状态       |
| `exchangeDevice(DeviceIndex)`      | 原子地交换当前设备并返回前一个设备         | 临时切换设备并在之后恢复先前的设备 |
| `maybeExchangeDevice(DeviceIndex)` | 仅在索引有效时有条件地交换设备                    | 通过验证安全地尝试设备切换                      |

## 实现

作为说明，OpenReg（开放注册）是一个 PyTorch 集成示例，它填补了树外加速器后端集成的空白。它演示了供应商如何通过实现钩子接口（参见 [`at::PrivateUse1HooksInterface`][PrivateUse1HooksInterface.h]）来注册自定义设备后端——而无需修改 PyTorch 核心。

我们以 `getDefaultGenerator` 为例：

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegHooks.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG HOOK EXAMPLES
    :end-before: LITERALINCLUDE END: OPENREG HOOK EXAMPLES
    :linenos:
```

在此实现中：

1. **重写基接口**：`getDefaultGenerator` 方法重写了来自 [`at::PrivateUse1HooksInterface`][PrivateUse1HooksInterface.h] 的虚方法。

2. **委托给设备特定的实现**：调用 `getDefaultOpenRegGenerator(device_index)`，它管理每个设备的生成器实例。

3. **返回设备特定的生成器**：返回的 `at::Generator` 包装了一个实现设备特定随机数生成的 `OpenRegGeneratorImpl`。

此模式适用于所有钩子：重写接口方法，验证输入，委托给您的设备特定 API，并以 PyTorch 预期的格式返回结果。

## 集成示例

以下部分演示了 PyTorch 在访问默认随机数生成器时如何与加速器钩子集成。该示例追踪了从面向用户的 Python 代码到设备特定实现的完整流程。

### 第 1 层：用户代码

用户代码通过调用 `manual_seed` 来设置确定性种子：

```python
import torch
torch.openreg.manual_seed(42)
```

### 第 2 层：扩展 Python API

Python API 层负责管理设备选择并调用 C++ 扩展（定义于 [`torch_openreg/openreg/random.py`][random.py]）：

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/random.py
    :language: python
    :start-after: LITERALINCLUDE START: OPENREG MANUAL SEED
    :end-before: LITERALINCLUDE END: OPENREG MANUAL SEED
    :linenos:
```

`manual_seed` 函数获取当前设备索引，调用 `torch_openreg._C._get_default_generator(idx)` 以获取设备特定的生成器，并设置其种子。

### 第三层：Python/C++ 桥接

C++ 扩展向 Python 暴露了 `_getDefaultGenerator`，它桥接到 PyTorch 核心：

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GET DEFAULT GENERATOR
    :end-before: LITERALINCLUDE END: OPENREG GET DEFAULT GENERATOR
    :linenos:
    :emphasize-lines: 10-11
```

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG MODULE METHODS
    :end-before: LITERALINCLUDE END: OPENREG MODULE METHODS
    :linenos:
    :emphasize-lines: 3
```

此函数从 Python 解包设备索引，创建一个 `PrivateUse1` 设备对象，并调用 `at::globalContext().defaultGenerator()`。随后 PyTorch 的上下文会分发到已注册的钩子。

### 第四层：PyTorch 核心上下文

PyTorch 的 `Context` 类将调用分发到相应的加速器钩子（[`aten/src/ATen/Context.h`][Context.h]）：

```{eval-rst}
.. literalinclude:: ../../../aten/src/ATen/Context.h
    :language: c++
    :lines: 60-103
    :linenos:
    :emphasize-lines: 8-9, 24-25
```

这种分层架构使 PyTorch 保持设备无关性，同时将硬件特定的操作委托给加速器实现。钩子在模块加载时一次性注册：

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegHooks.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG HOOK REGISTER
    :end-before: LITERALINCLUDE END: OPENREG HOOK REGISTER
    :linenos:
    :emphasize-lines: 4
```

### 第五层：加速器钩子

钩子接口提供了 PyTorch 用于委托给设备特定实现的抽象层：

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegHooks.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG HOOK EXAMPLES
    :end-before: LITERALINCLUDE END: OPENREG HOOK EXAMPLES
    :linenos:
```

`getDefaultGenerator` 钩子方法重写了基础接口并委托给 `getDefaultOpenRegGenerator`，后者管理实际的生成器实例。

### 第六层：设备特定实现

设备特定实现管理每个设备的生成器实例：

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGenerator.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GET DEFAULT GENERATOR IMPL
    :end-before: LITERALINCLUDE END: OPENREG GET DEFAULT GENERATOR IMPL
    :linenos:
```

此函数维护一个生成器的静态向量（每个设备一个），在首次访问时初始化它们，验证设备索引，并返回相应的生成器实例。

[random.py]: https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/random.py#L48-L53 "random.py"
[Context.h]: https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/Context.h#L61-L102 "Context.h"
[PrivateUse1HooksInterface.h]: https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/detail/PrivateUse1HooksInterface.h#L15-L72 "PrivateUse1HooksInterface.h"