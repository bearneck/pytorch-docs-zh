# Guard

## 背景

PyTorch 中的 Device Guard 抽象提供了基于 RAII 的设备与流管理机制，允许代码临时切换设备上下文，并在退出作用域时自动恢复原始设备。这对于需要确保在特定设备上执行的操作至关重要，无论当前全局设备状态如何。

对于自定义加速器，实现 `c10::impl::DeviceGuardImplInterface` 的 `CustomDeviceGuardImpl` 能够与 PyTorch 的设备管理基础设施无缝集成。例如，PyTorch 中的 OpenReg（开放注册）集成示例为 `PrivateUse1` 分发键提供了一个 `OpenRegGuardImpl`，具备以下能力：

- **设备管理**：保存、切换和恢复当前设备索引
- **流管理**：创建、查询和切换计算流
- **事件管理**：在流上记录事件并同步执行

通过向 PyTorch 注册 `OpenRegGuardImpl`，用户代码可以使用像 `torch.accelerator.device_index()` 这样的设备上下文管理器，并且操作能通过 guard 的 RAII 语义自动处理设备切换。

## 设计
Guard 接口类 [`c10::impl::DeviceGuardImplInterface`][DeviceGuardImplInterface] 提供三大类功能：

### 设备管理

设备管理支持在不同加速器设备之间切换并查询设备信息。这构成了 PyTorch 中设备上下文管理的基础。

| 功能                     | 描述                                                                 | 应用场景                                                                         |
| ------------------------ | -------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **设备查询与切换**       | `exchangeDevice`, `setDevice`, `getDevice`, `uncheckedSetDevice`     | 为操作更改当前设备；在 RAII guard 中保存/恢复设备上下文                          |
| **设备数量**             | `deviceCount` (noexcept)                                             | 查询可用设备数量；出错时必须返回 0 而不是抛出异常                                |
| **设备同步**             | `synchronizeDevice`                                                  | 等待设备上的所有操作完成                                                         |

### 流管理

流支持在加速器设备上异步执行操作。多个流允许在同一设备上并发执行独立操作。

| 功能                     | 描述                                                                 | 应用场景                                                         |
| ------------------------ | -------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **流创建/访问**          | `getStream`, `getDefaultStream`, `getNewStream`, `getStreamFromGlobalPool`, `exchangeStream` | 为异步执行创建和管理计算流                                       |
| **流同步**               | `queryStream`, `synchronizeStream`                                   | 检查流完成状态，等待流操作完成                                   |

### 事件管理

事件提供了用于协调跨流执行和测量执行时间的同步原语。它们标记了流执行中的特定点，这些点可以被等待或计时。

| 功能                     | 描述                                                                 | 应用场景                                                                      |
| ------------------------ | -------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **事件记录**             | `record`                                                             | 标记流执行中的一个点，用于后续同步或计时                                      |
| **事件阻塞**             | `block`                                                              | 使一个流等待来自另一个流的事件（流间依赖）                                    |
| **事件同步**             | `queryEvent`, `synchronizeEvent`                                     | 检查事件完成状态，等待事件完成                                                |
| **事件生命周期**         | `destroyEvent`                                                       | 在不再需要时释放事件资源                                                      |
| **事件计时**             | `elapsedTime`                                                        | 测量两个事件之间的时间，用于性能分析                                          |

## 实现

- {ref}`Device Guard 实现 <device-guard>`
- Stream Guard 实现（即将推出）
- Event Guard 实现（即将推出）

[OpenReg Guard]: https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.h "OpenReg Guard"
[DeviceGuardImplInterface]: https://github.com/pytorch/pytorch/blob/main/c10/core/impl/DeviceGuardImplInterface.h "DeviceGuardImplInterface"