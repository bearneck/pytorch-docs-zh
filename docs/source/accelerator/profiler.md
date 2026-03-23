# 性能分析器集成

## 背景

PyTorch 内置了一个设备无关的性能分析器，它能够检测 CPU 侧的算子分发、与加速器收集器协调、捕获 Python 调用栈，并导出聚合统计信息或 Chrome/Perfetto 跟踪数据。关于核心架构，请参阅 [`torch/csrc/profiler/README.md`][PyTorch Profiler README]。

对于加速器，主要有两种集成路径：

1. 传统 autograd 性能分析器：
    - 可以通过 `ProfilerStubs` 附加后端特定的钩子来记录设备事件并计算耗时。
    - 无需 Kineto 即可工作；适用于希望采用最小化、自包含路径的 PrivateUse1 后端。

2. 基于 Kineto 的时间线：
    - 桥接到 Kineto，后者通过供应商库（例如，用于 CUDA 的 CUPTI）聚合设备时间线。
    - 提供丰富的活动跟踪和高级导出/可视化功能，但需要一个支持 Kineto 的后端。

本文档重点介绍路径（1）：一个 `PrivateUse1` 加速器如何暴露最少的钩子以接入传统的 autograd 性能分析器，从而正确地将 ATen 算子和 `record_function` 范围归因于设备活动。

## 设计

### 架构概览

| 层级                 | 职责                                                                                                                                      | 源代码位置                       |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| Python 控制平面      | 管理性能分析器的生命周期（`prepare → start → stop → step`）并暴露用户 API，例如 `torch.autograd.profiler.profile`。                         | `torch/autograd/profiler.py`    |
| 性能分析器存根       | 实现 `torch::profiler::impl::ProfilerStubs`，以便性能分析器可以记录设备事件、同步、迭代设备并计算耗时。                                    | `torch/csrc/profiler/stubs/`    |
| 设备运行时           | 提供存根使用的流、事件和设备守卫；实现是后端特定的。                                                                                      | 后端扩展（供应商代码）            |

这种分层设计保持了 PyTorch 的设备无关性：Python 负责管理会话，`ProfilerStubs` 将性能分析器请求转换为后端运行时调用，而运行时则与加速器交互。

### 关键约定

*   **记录钩子**：`record()` 必须捕获（可选的）设备索引，分配一个后端事件，可选地存储一个 CPU 时间戳，并将事件入队到活动流中。
*   **耗时计算**：`elapsed()` 负责同步各个事件并以微秒为单位返回持续时间。
*   **同步钩子**：`synchronize()` 和 `onEachDevice()` 确保阶段转换（例如，预热 → 活动）在所有设备上对齐。
*   **注解**：可以实现 `mark`、`rangePush` 和 `rangePop` 来丰富跟踪信息；否则，它们可以作为空操作保留。

## 实现（传统方式）

这里我们使用 OpenReg（开放注册）来说明一个 `PrivateUse1` 加速器需要暴露的最少钩子集，以便性能分析器能够将 ATen 算子、`record_function` 范围和用户代码归因于设备活动。OpenReg 通过将性能分析器请求转换为其运行时调用，保持了上游代码不变，这反映了一个生产级加速器在树外扩展中会实现的内容。

OpenReg 目前依赖于传统的性能分析器（`torch.autograd.profiler.profile`）接口，而不是现代接口（`torch.profiler.profile`），因为后者强制要求 `use_kineto=True`。

### 性能分析器存根（C++）

[`torch::profiler::impl::OpenRegMethods`][openreg-stubs] 继承自 `ProfilerStubs` 并连接上述钩子：

| 方法                         | 目的                                                                                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `record`                       | 获取当前的 `OpenRegStream`，创建一个 `orEvent`，通过 `c10::getTime()` 捕获可选的 CPU 时间戳，并在流上记录该事件。 |
| `elapsed`                      | 同步两个事件，调用 `orEventElapsedTime`，并将毫秒转换为微秒以供性能分析器使用。                                      |
| `onEachDevice`                 | 使用 `c10::DeviceGuard(DeviceType::PrivateUse1)` 遍历 `torch.openreg.device_count()`，以便调度器可以运行每个设备的设置或清理操作。    |
| `synchronize`                  | 调用 `orDeviceSynchronize()` 以使设备工作与 CPU 调度阶段对齐。                                                                         |
| `enabled` 和注解填充函数 | 报告可用性并为 mark/push/pop 提供占位符实现。                                                                         |

构造函数通过 `registerPrivateUse1Methods(&methods);` 一次性注册这些方法，使得每当性能分析器启用 `use_device="openreg"` 时都能发现它们。

### Python 控制平面

在 Python 侧，不需要新的入口点——开发者使用标准的 autograd 性能分析器：

```python
from torch.autograd.profiler import profile as autograd_profile
from torch.profiler import record_function

with autograd_profile(use_device="openreg", record_shapes=True) as prof:
    with record_function("matmul"):
        x = torch.randn(512, 512, device="openreg")
        y = torch.randn(512, 512, device="openreg")
        z = x @ y

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
prof.export_chrome_trace("openreg_trace.json")
```

### 数据捕获流程

1. 用户代码进入 `autograd_profile(use_device="openreg")`。
2. 分析器切换到 `ProfilerState.KINETO_PRIVATEUSE1_FALLBACK` 状态。
3. 分析器请求当前活跃的后端 `record()` 一个事件。
4. OpenReg 存根分配 `orEvent` 对象，将其附加到当前流，并记录 CPU 时间戳。
5. 当事件结束时，分析器调用 `elapsed()` 来计算持续时间。


[PyTorch Profiler README]: https://github.com/pytorch/pytorch/blob/main/torch/csrc/profiler/README.md "PyTorch Profiler README"
[openreg-stubs]: https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/profiler/stubs/openreg.cpp "OpenReg profiler stubs"
