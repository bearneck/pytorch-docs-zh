# 自动加载机制

PyTorch 中的**自动加载**机制通过支持运行时自动发现和初始化，简化了自定义后端的集成。这消除了显式导入或手动初始化的需求，使开发者能够将新的加速器或后端无缝集成到 PyTorch 中。

## 背景

PyTorch 中的**自动加载设备扩展**提案旨在改进对各种硬件后端设备的支持，特别是那些作为树外扩展（不属于 PyTorch 主代码库的一部分）实现的设备。目前，用户必须手动导入或加载这些设备特定的扩展才能使用它们，这使体验复杂化并增加了认知负担。

相比之下，树内设备（PyTorch 官方支持的设备）是无缝集成的——用户不需要额外的导入或步骤。自动加载的目标是使树外设备同样易于使用，因此用户可以遵循标准的 PyTorch 设备编程模型，而无需显式加载或代码更改。这将允许现有的 PyTorch 应用程序无需任何修改即可在新设备上运行，使硬件支持更加用户友好并降低采用门槛。

有关**自动加载**背景的更多信息，请参阅其 [RFC](https://github.com/pytorch/pytorch/issues/122468)。

## 设计

**自动加载**的核心思想是利用 Python 的插件发现机制（入口点），使 PyTorch 在导入 torch 时自动加载树外设备扩展——无需用户显式导入。

有关**自动加载**设计的更多说明，请参阅 [**工作原理**](https://docs.pytorch.org/tutorials/unstable/python_extension_autoload.html#how-it-works)。

## 实现

本教程将以 **OpenReg** 作为新的树外设备，指导您完成启用和使用**自动加载**机制的步骤。

### 入口点设置

要启用**自动加载**，请在 [setup.py](https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/setup.py) 文件中将 `_autoload` 函数注册为入口点。


### 后端设置

在 [torch_openreg](https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/__init__.py) 中为后端初始化定义初始化钩子 `_autoload`。此钩子将在 PyTorch 启动期间自动调用。


## 结果

设置好入口点和后端后，构建并安装您的后端。现在，我们可以在不显式导入的情况下使用新的加速器。
