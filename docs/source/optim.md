# torch.optim


## 如何使用优化器

要使用 `torch.optim`，你需要构造一个优化器对象，该对象将保存当前状态，并根据计算出的梯度更新参数。

### 构造优化器

要构造一个 `Optimizer`，你需要给它一个包含待优化参数（所有参数都应为 `~torch.nn.Parameter` 类型）的可迭代对象，或者包含命名参数（(str, `~torch.nn.Parameter`) 元组）的可迭代对象。然后，你可以指定优化器特定的选项，例如学习率、权重衰减等。

示例：
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)
```

命名参数示例：

```python
optimizer = optim.SGD(model.named_parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([('layer0', var1), ('layer1', var2)], lr=0.0001)
```

### 每个参数组的选项

`Optimizer` 也支持指定每个参数组的选项。为此，不要传递 `~torch.autograd.Variable` 的可迭代对象，而是传递 `dict` 的可迭代对象。每个字典将定义一个独立的参数组，并且应包含一个 `params` 键，其中包含属于该组的参数列表。其他键应与优化器接受的关键字参数匹配，并将用作该组的优化选项。

例如，当想要指定每层的学习率时，这非常有用：

```python
optim.SGD([
    {'params': model.base.parameters(), 'lr': 1e-2},
    {'params': model.classifier.parameters()}
], lr=1e-3, momentum=0.9)

optim.SGD([
    {'params': model.base.named_parameters(), 'lr': 1e-2},
    {'params': model.classifier.named_parameters()}
], lr=1e-3, momentum=0.9)
```

这意味着 `model.base` 的参数将使用 `1e-2` 的学习率，而 `model.classifier` 的参数将保持默认的 `1e-3` 学习率。最后，所有参数都将使用 `0.9` 的动量。

```{note}
你仍然可以将选项作为关键字参数传递。它们将用作默认值，适用于那些没有覆盖这些选项的组。当你只想改变一个选项，同时保持所有参数组之间的其他选项一致时，这很有用。
```

还要考虑以下与参数不同惩罚相关的示例。请记住，`~torch.nn.Module.parameters` 返回一个包含所有可学习参数的可迭代对象，包括偏置项和其他可能偏好不同惩罚的参数。为了解决这个问题，可以为每个参数组指定单独的惩罚权重：

```python
bias_params = [p for name, p in self.named_parameters() if 'bias' in name]
others = [p for name, p in self.named_parameters() if 'bias' not in name]

optim.SGD([
    {'params': others},
    {'params': bias_params, 'weight_decay': 0}
], weight_decay=1e-2, lr=1e-2)
```

通过这种方式，偏置项与非偏置项被隔离开来，并且为偏置项专门设置了 `weight_decay` 为 `0`，以避免对该组进行任何惩罚。

### 执行优化步骤

所有优化器都实现了一个 `~Optimizer.step` 方法，用于更新参数。它可以通过两种方式使用：

#### `optimizer.step()`

这是大多数优化器支持的简化版本。在使用例如 `~torch.autograd.Variable.backward` 计算出梯度后，可以调用此函数。

示例：

```python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

#### `optimizer.step(closure)`

一些优化算法，如共轭梯度法和 LBFGS，需要多次重新评估函数，因此你必须传入一个闭包，允许它们重新计算你的模型。该闭包应清除梯度，计算损失，并返回损失。

示例：
```python
for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)
```


## 基类


## 算法


我们的许多算法都有针对性能、可读性和/或通用性进行优化的各种实现，因此如果用户没有指定特定的实现，我们会尝试默认为当前设备上通常最快的实现。

我们有三大类实现：for-loop、foreach（多张量）和 fused。最直接的实现是对参数进行 for 循环，并包含大块的计算。For-loop 通常比我们的 foreach 实现慢，后者将参数组合成一个多张量，并一次性运行大块的计算，从而节省了许多顺序内核调用。我们的一些优化器甚至有更快的 fused 实现，它将大块的计算融合到一个内核中。我们可以将 foreach 实现视为水平融合，而 fused 实现则是在此基础上的垂直融合。

通常来说，三种实现的性能排序为：融合实现 > foreach 实现 > for 循环实现。
因此在适用的情况下，我们默认使用 foreach 实现而非 for 循环实现。适用条件包括：foreach 实现可用、用户未指定任何实现特定的参数（例如 fused、foreach、differentiable），且所有张量都是原生类型。需要注意的是，虽然融合实现应该比 foreach 实现更快，但这些实现较新，我们希望在全范围启用前给予它们更多的测试时间。我们在下面的第二个表格中总结了每种实现的稳定性状态，欢迎您尝试使用！

以下是显示各算法可用实现及默认实现的表格：


下表显示了融合实现的稳定性状态：


## 如何调整学习率

`torch.optim.lr_scheduler.LRScheduler` 提供了多种基于训练轮次调整学习率的方法。`torch.optim.lr_scheduler.ReduceLROnPlateau` 允许基于某些验证指标动态降低学习率。

学习率调度应在优化器更新之后应用；例如，您应该按以下方式编写代码：

示例：
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = ExponentialLR(optimizer, gamma=0.9)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

大多数学习率调度器可以连续调用（也称为链式调度器）。这样每个调度器会依次对前一个调度器得到的学习率进行处理。

示例：
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler1 = ExponentialLR(optimizer, gamma=0.9)
scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler1.step()
    scheduler2.step()
```

在文档的许多地方，我们将使用以下模板来指代调度器算法。

```python
>>> scheduler = ...
>>> for epoch in range(100):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()
```

```{warning}
在 PyTorch 1.1.0 之前，学习率调度器预期在优化器更新之前调用；1.1.0 版本以破坏向后兼容的方式改变了这一行为。如果您在优化器更新（调用 `optimizer.step()`）之前使用学习率调度器（调用 `scheduler.step()`），将会跳过学习率调度计划的第一个值。如果在升级到 PyTorch 1.1.0 后无法复现结果，请检查是否在错误的时间调用了 `scheduler.step()`。
```


## 如何利用命名参数加载优化器状态字典

函数 `~Optimizer.load_state_dict` 会在加载的状态字典包含可选的 `param_names` 内容时将其存储。然而，加载优化器状态的过程不受影响，因为参数的顺序对于保持兼容性很重要（以防顺序不同）。要利用从加载的状态字典中获取的参数名称，需要根据所需行为实现自定义的 `register_load_state_dict_pre_hook`。

例如，当模型架构发生变化但权重和优化器状态需要保持不变时，这非常有用。以下示例演示了如何实现此自定义功能。

示例：
```python
class OneLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 4)
```

def forward(self, x):
        return self.fc(x)

model = OneLayerModel()
optimizer = optim.SGD(model.named_parameters(), lr=0.01, momentum=0.9)
# 训练过程..
torch.save(optimizer.state_dict(), PATH)
```

假设 `model` 实现了一个专家（MoE），我们想要复制它并恢复训练两个专家，这两个专家都以与 `fc` 层相同的方式初始化。对于下面的 `model2`，我们创建两个与 `fc` 相同的层，并通过将 `model` 的模型权重和优化器状态加载到 `model2` 的 `fc1` 和 `fc2` 中（并相应调整它们）来恢复训练：

```python
class TwoLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(3, 4)

    def forward(self, x):
        return (self.fc1(x) + self.fc2(x)) / 2

model2 = TwoLayerModel()
# 调整并加载模型权重..
optimizer2 = optim.SGD(model2.named_parameters(), lr=0.01, momentum=0.9)
```

为了使用先前优化器的状态字典加载 `optimizer2` 的状态字典，使得 `fc1` 和 `fc2` 都使用 `fc` 优化器状态的副本进行初始化（以便从 `fc` 恢复每个层的训练），我们可以使用以下钩子：

```python
def adapt_state_dict_ids(optimizer, state_dict):
    adapted_state_dict = deepcopy(optimizer.state_dict())
    # 复制设置参数（lr、weight_decay 等），以防加载的状态字典中这些参数不同。
    for k, v in state_dict['param_groups'][0].items():
        if k not in ['params', 'param_names']:
            adapted_state_dict['param_groups'][0][k] = v

    lookup_dict = {
        'fc1.weight': 'fc.weight',
        'fc1.bias': 'fc.bias',
        'fc2.weight': 'fc.weight',
        'fc2.bias': 'fc.bias'
    }
    clone_deepcopy = lambda d: {k: (v.clone() if isinstance(v, torch.Tensor) else deepcopy(v)) for k, v in d.items()}
    for param_id, param_name in zip(
            optimizer.state_dict()['param_groups'][0]['params'],
            optimizer.state_dict()['param_groups'][0]['param_names']):
        name_in_loaded = lookup_dict[param_name]
        index_in_loaded_list = state_dict['param_groups'][0]['param_names'].index(name_in_loaded)
        id_in_loaded = state_dict['param_groups'][0]['params'][index_in_loaded_list]
        # 复制对应参数的状态
        if id_in_loaded in state_dict['state']:
            adapted_state_dict['state'][param_id] = clone_deepcopy(state_dict['state'][id_in_loaded])

    return adapted_state_dict

optimizer2.register_load_state_dict_pre_hook(adapt_state_dict_ids)
optimizer2.load_state_dict(torch.load(PATH)) # 先前优化器保存的 state_dict
```

这确保了在模型加载过程中将使用为 `model2` 的各层提供正确状态的适配后状态字典。
请注意，此代码是专门为此示例设计的（例如，假设只有一个参数组），其他情况可能需要不同的适配。

以下示例展示了当模型结构发生变化时，如何处理加载的 `state dict` 中缺失的参数。
`Model_bypass` 添加了一个新的 `bypass` 层，该层在原始的 `Model1` 中不存在。
为了恢复训练，使用了一个自定义的 `adapt_state_dict_missing_param` 钩子来适配优化器的 `state_dict`，确保现有参数被正确映射，而缺失的参数（如 bypass 层）保持不变（如本示例中的初始化状态）。
这种方法使得尽管模型发生变化，优化器状态仍能顺利加载和恢复。
新的 bypass 层将从零开始训练：

```python
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        return self.fc(x) + x


model = Model1()
optimizer = optim.SGD(model.named_parameters(), lr=0.01, momentum=0.9)
# 训练过程..
torch.save(optimizer.state_dict(), PATH)

class Model_bypass(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)
        self.bypass = nn.Linear(5, 5, bias=False)
        torch.nn.init.eye_(self.bypass.weight)

    def forward(self, x):
        return self.fc(x) + self.bypass(x)

model2 = Model_bypass()
optimizer2 = optim.SGD(model2.named_parameters(), lr=0.01, momentum=0.9)

def adapt_state_dict_missing_param(optimizer, state_dict):
    adapted_state_dict = deepcopy(optimizer.state_dict())
    # 复制设置参数（lr、weight_decay 等），以防加载的状态字典中这些参数不同。
    for k, v in state_dict['param_groups'][0].items():
        if k not in ['params', 'param_names']:
            adapted_state_dict['param_groups'][0][k] = v

    lookup_dict = {
        'fc.weight': 'fc.weight',
        'fc.bias': 'fc.bias',
        'bypass.weight': None,
    }

    clone_deepcopy = lambda d: {k: (v.clone() if isinstance(v, torch.Tensor) else deepcopy(v)) for k, v in d.items()}
    for param_id, param_name in zip(
            optimizer.state_dict()['param_groups'][0]['params'],
            optimizer.state_dict()['param_groups'][0]['param_names']):
        name_in_loaded = lookup_dict[param_name]
        if name_in_loaded in state_dict['param_groups'][0]['param_names']:
            index_in_loaded_list = state_dict['param_groups'][0]['param_names'].index(name_in_loaded)
            id_in_loaded = state_dict['param_groups'][0]['params'][index_in_loaded_list]
            # 复制对应参数的状态
            if id_in_loaded in state_dict['state']:
                adapted_state_dict['state'][param_id] = clone_deepcopy(state_dict['state'][id_in_loaded])

    return adapted_state_dict

optimizer2.register_load_state_dict_pre_hook(adapt_state_dict_ids)
optimizer2.load_state_dict(torch.load(PATH)) # 先前优化器保存的 state_dict
```


作为第三个示例，此钩子可用于根据参数名称加载状态，而不是根据参数顺序加载（默认方法）：

```python
def names_matching(optimizer, state_dict):
    assert len(state_dict['param_groups']) == len(optimizer.state_dict()['param_groups'])
    adapted_state_dict = deepcopy(optimizer.state_dict())
    for g_ind in range(len(state_dict['param_groups'])):
        assert len(state_dict['param_groups'][g_ind]['params']) == len(
            optimizer.state_dict()['param_groups'][g_ind]['params'])

        for k, v in state_dict['param_groups'][g_ind].items():
            if k not in ['params', 'param_names']:
                adapted_state_dict['param_groups'][g_ind][k] = v

        for param_id, param_name in zip(
                optimizer.state_dict()['param_groups'][g_ind]['params'],
                optimizer.state_dict()['param_groups'][g_ind]['param_names']):
            index_in_loaded_list = state_dict['param_groups'][g_ind]['param_names'].index(param_name)
            id_in_loaded = state_dict['param_groups'][g_ind]['params'][index_in_loaded_list]
            # 复制对应参数的状态
            if id_in_loaded in state_dict['state']:
                adapted_state_dict['state'][param_id] = deepcopy(state_dict['state'][id_in_loaded])

    return adapted_state_dict
```


## 权重平均（SWA 和 EMA）

`torch.optim.swa_utils.AveragedModel` 实现了随机权重平均（SWA）和指数移动平均（EMA），
`torch.optim.swa_utils.SWALR` 实现了 SWA 学习率调度器，
`torch.optim.swa_utils.update_bn` 是一个实用函数，用于在训练结束时更新 SWA/EMA 的批量归一化统计量。

SWA 在论文 [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407) 中提出。

EMA 是一种广为人知的技术，通过减少所需的权重更新次数来缩短训练时间。它是 [Polyak averaging](https://paperswithcode.com/method/polyak-averaging) 的一种变体，但使用指数权重而非迭代间的等权重。

### 构建平均模型

`AveragedModel` 类用于计算 SWA 或 EMA 模型的权重。

你可以通过以下方式创建 SWA 平均模型：

```python
>>> averaged_model = AveragedModel(model)
```

EMA 模型通过指定 `multi_avg_fn` 参数来构建，如下所示：

```python
>>> decay = 0.999
>>> averaged_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))
```

衰减（decay）是一个介于 0 和 1 之间的参数，控制平均参数的衰减速度。如果未提供给 `torch.optim.swa_utils.get_ema_multi_avg_fn`，默认值为 0.999。衰减值应接近 1.0，因为较小的值可能导致优化收敛问题。

`torch.optim.swa_utils.get_ema_multi_avg_fn` 返回一个函数，该函数将以下 EMA 方程应用于权重：

```{math}
W_0^{\text{EMA}} = W_0^{\text{model}}
```

```{math}
W_{t+1}^{\text{EMA}} = \text{decay} \times W_t^{\text{EMA}} + (1 - \text{decay}) \times W_{t+1}^{\text{model}}
```
其中 `W_t^{\text{EMA}}` 是步骤 `t` 的 EMA 参数，`W_t^{\text{model}}` 是步骤 `t` 的模型参数，decay 是 EMA 衰减率（默认：0.999）。

这里的模型 `model` 可以是任意的 `torch.nn.Module` 对象。`averaged_model` 将跟踪 `model` 参数的运行平均值。要更新这些平均值，应在 `optimizer.step()` 后使用 `update_parameters` 函数：

```python
>>> averaged_model.update_parameters(model)
```

对于 SWA 和 EMA，此调用通常在优化器 `step()` 之后立即执行。在 SWA 的情况下，通常在训练开始阶段跳过若干步。

### 自定义平均策略

默认情况下，`torch.optim.swa_utils.AveragedModel` 计算所提供参数的运行等权重平均值，但你也可以使用 `avg_fn` 或 `multi_avg_fn` 参数定义自定义平均函数：

- `avg_fn` 允许定义一个作用于每个参数元组（平均参数，模型参数）的函数，并应返回新的平均参数。
- `multi_avg_fn` 允许定义更高效的操作，同时作用于参数列表元组（平均参数列表，模型参数列表），例如使用 `torch._foreach*` 函数。此函数必须原地更新平均参数。

在以下示例中，`ema_model` 使用 `avg_fn` 参数计算指数移动平均：

```python
>>> ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
>>>         0.9 * averaged_model_parameter + 0.1 * model_parameter
>>> ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
```

在以下示例中，`ema_model` 使用更高效的 `multi_avg_fn` 参数计算指数移动平均：

```python
>>> ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9))
```

### SWA 学习率调度

通常，在 SWA 中学习率设置为较高的常数值。`SWALR` 是一种学习率调度器，它将学习率退火到一个固定值，然后保持恒定。例如，以下代码创建一个调度器，在每个参数组内，将学习率在 5 个周期内从其初始值线性退火到 0.05：

```python
>>> swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, \
>>>         anneal_strategy="linear", anneal_epochs=5, swa_lr=0.05)
```

你也可以通过设置 `anneal_strategy="cos"` 使用余弦退火到固定值，而不是线性退火。

### 处理批量归一化

`update_bn` 是一个实用函数，允许在训练结束时，在给定的数据加载器 `loader` 上为 SWA 模型计算批量归一化统计量：

```python
>>> torch.optim.swa_utils.update_bn(loader, swa_model)
```

`update_bn` 将 `swa_model` 应用于数据加载器中的每个元素，并计算模型中每个批量归一化层的激活统计量。

```{warning}
`update_bn` 假设数据加载器 `loader` 中的每个批次要么是一个张量，要么是一个张量列表，
其中第一个元素是网络 `swa_model` 应该应用到的张量。
如果你的数据加载器具有不同的结构，可以通过使用 `swa_model` 对数据集中的每个元素进行前向传递来更新 `swa_model` 的批归一化统计量。
```


### 完整示例：SWA

在下面的示例中，`swa_model` 是累积权重平均值的 SWA 模型。
我们总共训练模型 300 个周期，并在第 160 个周期切换到 SWA 学习率调度并开始收集参数的 SWA 平均值：

```python
>>> loader, optimizer, model, loss_fn = ...
>>> swa_model = torch.optim.swa_utils.AveragedModel(model)
>>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
>>> swa_start = 160
>>> swa_scheduler = SWALR(optimizer, swa_lr=0.05)
>>>
>>> for epoch in range(300):
>>>       for input, target in loader:
>>>           optimizer.zero_grad()
>>>           loss_fn(model(input), target).backward()
>>>           optimizer.step()
>>>       if epoch > swa_start:
>>>           swa_model.update_parameters(model)
>>>           swa_scheduler.step()
>>>       else:
>>>           scheduler.step()
>>>
>>> # 最后更新 swa_model 的批归一化统计量
>>> torch.optim.swa_utils.update_bn(loader, swa_model)
>>> # 使用 swa_model 对测试数据进行预测
>>> preds = swa_model(test_input)
```

### 完整示例：EMA

在下面的示例中，`ema_model` 是累积权重指数衰减平均值的 EMA 模型，衰减率为 0.999。
我们总共训练模型 300 个周期，并立即开始收集 EMA 平均值。

```python
>>> loader, optimizer, model, loss_fn = ...
>>> ema_model = torch.optim.swa_utils.AveragedModel(model, \
>>>             multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
>>>
>>> for epoch in range(300):
>>>       for input, target in loader:
>>>           optimizer.zero_grad()
>>>           loss_fn(model(input), target).backward()
>>>           optimizer.step()
>>>           ema_model.update_parameters(model)
>>>
>>> # 最后更新 ema_model 的批归一化统计量
>>> torch.optim.swa_utils.update_bn(loader, ema_model)
>>> # 使用 ema_model 对测试数据进行预测
>>> preds = ema_model(test_input)
```


<!-- 此模块需要文档记录。暂时添加在此处以供跟踪 -->


