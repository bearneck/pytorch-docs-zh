(miscellaneous_environment_variables)=

# 杂项环境变量

| 变量名                              | 描述 |
|---------------------------------------|-------------|
| `TORCH_FORCE_WEIGHTS_ONLY_LOAD`       | 如果设置为 [`1`, `y`, `yes`, `true`] 之一，`torch.load` 将使用 `weights_only=True`。即使调用时传入了 `weights_only=False`，此设置也会生效。更多文档请参阅 [`torch.load`](https://pytorch.org/docs/stable/generated/torch.load.html)。 |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`    | 如果设置为 [`1`, `y`, `yes`, `true`] 之一，并且调用时未传递 `weights_only` 变量，`torch.load` 将使用 `weights_only=False`。更多文档请参阅 [`torch.load`](https://pytorch.org/docs/stable/generated/torch.load.html)。 |
| `TORCH_AUTOGRAD_SHUTDOWN_WAIT_LIMIT`  | 在某些情况下，autograd 线程可能在程序关闭时挂起，因此我们不会无限期地等待它们关闭，而是依赖一个默认设置为 `10` 秒的超时机制。此环境变量可用于设置超时时间（单位：秒）。 |
| `TORCH_DEVICE_BACKEND_AUTOLOAD`       | 如果设置为 `1`，在运行 `import torch` 时将自动导入树外后端扩展。 |