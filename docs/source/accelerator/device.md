# 设备管理

## 背景

设备管理涵盖诸如查询可用设备数量、在设备间切换等基础功能。加速器后端封装其设备运行时 API 并将其暴露给 PyTorch。

## 设计

加速器供应商应实现以下核心函数：

| 函数名                  | 描述                                                         | 应用场景                                                                                          |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `device_count()`        | 查询系统中可用设备的总数                                     | - 应用程序初始化<br>- 多设备工作负载分配<br>- 使用前验证设备索引                                   |
| `current_device()`      | 获取调用线程当前活动的设备                                   | - 调试和日志记录<br>- 确定张量放置位置<br>- 防护实现                                              |
| `set_device()`          | 为后续操作更改活动设备                                       | - 在设备间切换上下文<br>- 初始化特定设备资源<br>- 多 GPU 训练循环                                 |
| `exchange_device()`     | 原子地交换设备并返回之前的设备                               | - 实现设备防护<br>- 临时切换设备上下文<br>- 基于 RAII 的设备管理                                  |
| `maybe_exchange_device()` | 仅在索引有效时（允许 -1）有条件地交换设备                     | - 使用可选索引的安全设备切换<br>- 支持可空设备值的防护实现                                        |

这些函数是流、事件和内存管理的基础构建块。请正确验证输入并处理错误。

## 实现

本节以 `set_device` 为例说明设备管理的实现。实现需要：
1. 设备运行时的 C++ 包装器
2. 暴露 C++ 函数的 Python 绑定
3. 用户友好的 Python API

作为示例，OpenReg（开放注册）是一个 PyTorch 集成示例，它为树外加速器后端集成填补了空白。其实现（[`OpenRegFunctions.h/cpp`][OpenReg Device Management]）展示了如何清晰地包装第三方运行时。这些函数在后端中被重复使用——用于流、事件、生成器和 Python 绑定。

### C++ 端

包装设备运行时 API 并添加错误处理。`SetDevice` 函数展示了这种模式：


### 绑定

使用 pybind11 将 C++ 函数暴露给 Python：


### Python 端

用用户友好的 Python 函数包装 C++ 绑定：


以下是 C++ 到 Python 的完整映射：

| C++ 绑定函数       | C++ 绑定 API (pybind11)               | Python 用户 API                  | 描述                                   |
| ------------------ | ------------------------------------- | -------------------------------- | -------------------------------------- |
| `_getDeviceCount`  | `torch_openreg._C._get_device_count()` | `torch.openreg.device_count()`   | 返回设备总数                           |
| `_getDevice`       | `torch_openreg._C._get_device()`       | `torch.openreg.current_device()` | 返回当前活动设备索引                   |
| `_setDevice`       | `torch_openreg._C._set_device(idx)`    | `torch.openreg.set_device(idx)`  | 设置活动设备                           |
| `_exchangeDevice`  | `torch_openreg._C._exchange_device(idx)` | N/A（仅内部使用）                | 原子地交换设备并返回之前的设备         |


## 防护

设备防护提供具有异常安全性的自动设备切换。它们类似于 C++ 锁防护——在构造时切换设备，在析构时恢复。

实现 `DeviceGuardImplInterface` 以集成到 PyTorch 的防护系统中：


这使得该防护可用于 PyTorch 的 `PrivateUse1` 设备类型；用户随后可以将标准的 PyTorch 设备防护与自定义后端一起使用。

[OpenReg 设备管理]: https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegFunctions.cpp "OpenReg Device Management"