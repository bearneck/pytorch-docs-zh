# MPS 后端 {#MPS-Backend}

`mps`{.interpreted-text role="mod"} 设备支持在配备 Metal 编程框架的 macOS 设备上实现高性能 GPU 训练。它引入了一种新的设备，能够将机器学习计算图和基本运算分别映射到高效的 Metal Performance Shaders Graph 框架以及由 Metal Performance Shaders 框架提供的优化内核上。

新的 MPS 后端扩展了 PyTorch 生态系统，并为现有脚本提供了在 GPU 上设置和运行运算的能力。

要开始使用，只需将您的张量和模型移动到 `mps` 设备：

``` python
# 检查 MPS 是否可用
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS 不可用，因为当前安装的 PyTorch 未启用 MPS 支持。")
    else:
        print("MPS 不可用，因为当前 macOS 版本低于 14.0+ 和/或此机器上没有支持 MPS 的设备。")

else:
    mps_device = torch.device("mps")

    # 直接在 mps 设备上创建张量
    x = torch.ones(5, device=mps_device)
    # 或者
    x = torch.ones(5, device="mps")

    # 所有运算都在 GPU 上进行
    y = x * 2

    # 将模型移至 mps 设备，就像其他设备一样
    model = YourFavoriteNet()
    model.to(mps_device)

    # 现在每次调用都在 GPU 上运行
    pred = model(x)
```
