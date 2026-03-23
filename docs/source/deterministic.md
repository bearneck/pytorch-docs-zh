# torch.utils.deterministic

> 模块：`torch.utils.deterministic`


## `fill_uninitialized_memory`

一个 `bool` 类型的属性，如果设置为 True，当 `torch.use_deterministic_algorithms()` 设置为 `True` 时，会导致未初始化的内存被填充为一个已知值。浮点数和复数值被设置为 NaN，整数值被设置为最大值。
