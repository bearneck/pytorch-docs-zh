# 修补批归一化

## 问题描述
批归一化需要对与输入大小相同的 running_mean 和 running_var 进行原地更新。
Functorch 不支持对接收批处理张量的常规张量进行原地更新（即不允许 `regular.add_(batched)`）。因此，当对单个模块的输入批次进行 vmap 操作时，我们会遇到此错误。

## 解决方案
最受支持的解决方案之一是将 BatchNorm 替换为 GroupNorm。选项 1 和 2 支持此方法。

所有这些方案都假设您不需要运行统计信息。如果您使用的是模块，这意味着假设您不会在评估模式下使用批归一化。如果您有需要在评估模式下使用 vmap 运行批归一化的用例，请提交问题报告。

### 选项 1：更改 BatchNorm
如果您想更改为 GroupNorm，请将所有 BatchNorm 替换为：

```python
BatchNorm2d(C, G, track_running_stats=False)
```

这里的 `C` 与原始 BatchNorm 中的 `C` 相同。`G` 是将 `C` 分成的组数。因此，`C % G == 0`，作为备用方案，您可以设置 `C == G`，这意味着每个通道将被单独处理。

如果您必须使用 BatchNorm 并且您自己构建了模块，可以将模块更改为不使用运行统计信息。换句话说，在任何有 BatchNorm 模块的地方，将 `track_running_stats` 标志设置为 False：

```python
BatchNorm2d(64, track_running_stats=False)
```

### 选项 2：torchvision 参数
一些 torchvision 模型，如 resnet 和 regnet，可以接受 `norm_layer` 参数。如果默认设置，这些通常默认为 BatchNorm2d。

您可以将其设置为 GroupNorm：

```python
import torchvision
from functools import partial
torchvision.models.resnet18(norm_layer=lambda c: GroupNorm(num_groups=g, c))
```

这里再次强调，`c % g == 0`，因此作为备用方案，设置 `g = c`。

如果您坚持使用 BatchNorm，请确保使用不使用运行统计信息的版本：

```python
import torchvision
from functools import partial
torchvision.models.resnet18(norm_layer=partial(BatchNorm2d, track_running_stats=False))
```

### 选项 3：functorch 的修补功能
functorch 添加了一些功能，允许快速、原地修补模块以不使用运行统计信息。更改归一化层更脆弱，因此我们没有提供该功能。如果您有一个希望 BatchNorm 不使用运行统计信息的网络，可以运行 `replace_all_batch_norm_modules_` 来原地更新模块以不使用运行统计信息：

```python
from torch.func import replace_all_batch_norm_modules_
replace_all_batch_norm_modules_(net)
```

### 选项 4：评估模式
在评估模式下运行时，running_mean 和 running_var 不会被更新。因此，vmap 可以支持此模式：

```python
model.eval()
vmap(model)(x)
model.train()
```
