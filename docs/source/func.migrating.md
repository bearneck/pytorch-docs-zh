# 从 functorch 迁移到 torch.func

torch.func，之前称为 "functorch"，是 PyTorch 中类似 [JAX](https://github.com/google/jax) 的可组合函数变换。

functorch 最初是 [pytorch/functorch](https://github.com/pytorch/functorch) 仓库中的一个树外库。我们的目标一直是直接将 functorch 上游化到 PyTorch 中，并将其作为 PyTorch 的核心库提供。

作为上游化的最后一步，我们决定从顶层包 (`functorch`) 迁移为 PyTorch 的一部分，以反映函数变换如何直接集成到 PyTorch 核心中。从 PyTorch 2.0 开始，我们弃用 `import functorch`，并要求用户迁移到最新的 API，我们将持续维护这些 API。`import functorch` 将保留一段时间以保持向后兼容性，持续几个版本。

## 函数变换

以下 API 是以下 [functorch API](https://pytorch.org/functorch/1.13/functorch.html) 的直接替代品。它们完全向后兼容。

| functorch API                      | PyTorch API (自 PyTorch 2.0 起)                |
| ----------------------------------- | ---------------------------------------------- |
| functorch.vmap                      | `torch.vmap` 或 `torch.func.vmap`              |
| functorch.grad                      | `torch.func.grad`                              |
| functorch.vjp                       | `torch.func.vjp`                               |
| functorch.jvp                       | `torch.func.jvp`                               |
| functorch.jacrev                    | `torch.func.jacrev`                            |
| functorch.jacfwd                    | `torch.func.jacfwd`                            |
| functorch.hessian                   | `torch.func.hessian`                           |
| functorch.functionalize             | `torch.func.functionalize`                     |

此外，如果您正在使用 torch.autograd.functional API，请尝试使用 `torch.func` 的等效功能。`torch.func` 函数变换在许多情况下更具可组合性和更高性能。

| torch.autograd.functional API               | torch.func API (自 PyTorch 2.0 起)                |
| ------------------------------------------- | ---------------------------------------------- |
| `torch.autograd.functional.vjp`             | `torch.func.grad` 或 `torch.func.vjp`           |
| `torch.autograd.functional.jvp`             | `torch.func.jvp`                                |
| `torch.autograd.functional.jacobian`        | `torch.func.jacrev` 或 `torch.func.jacfwd`      |
| `torch.autograd.functional.hessian`         | `torch.func.hessian`                            |

## 神经网络模块工具

我们更改了在神经网络模块上应用函数变换的 API，以使其更好地适应 PyTorch 的设计理念。新的 API 有所不同，因此请仔细阅读本节。

### functorch.make_functional

`torch.func.functional_call` 是 [functorch.make_functional](https://pytorch.org/functorch/1.13/generated/functorch.make_functional.html#functorch.make_functional) 和 [functorch.make_functional_with_buffers](https://pytorch.org/functorch/1.13/generated/functorch.make_functional_with_buffers.html#functorch.make_functional_with_buffers) 的替代品。然而，它并非直接替代品。

如果您时间紧迫，可以使用 [此 gist 中的辅助函数](https://gist.github.com/zou3519/7769506acc899d83ef1464e28f22e6cf) 来模拟 functorch.make_functional 和 functorch.make_functional_with_buffers 的行为。我们建议直接使用 `torch.func.functional_call`，因为它是一个更明确和灵活的 API。

具体来说，functorch.make_functional 返回一个功能模块和参数。该功能模块接受模型的参数和输入作为参数。`torch.func.functional_call` 允许使用新的参数、缓冲区和输入调用现有模块的前向传播。

以下是一个使用 functorch 与 `torch.func` 计算模型参数梯度的示例：

```python
# ---------------
# 使用 functorch
# ---------------
import torch
import functorch
inputs = torch.randn(64, 3)
targets = torch.randn(64, 3)
model = torch.nn.Linear(3, 3)

fmodel, params = functorch.make_functional(model)

def compute_loss(params, inputs, targets):
    prediction = fmodel(params, inputs)
    return torch.nn.functional.mse_loss(prediction, targets)

grads = functorch.grad(compute_loss)(params, inputs, targets)

# ------------------------------------
# 使用 torch.func (自 PyTorch 2.0 起)
# ------------------------------------
import torch
inputs = torch.randn(64, 3)
targets = torch.randn(64, 3)
model = torch.nn.Linear(3, 3)

params = dict(model.named_parameters())

def compute_loss(params, inputs, targets):
    prediction = torch.func.functional_call(model, params, (inputs,))
    return torch.nn.functional.mse_loss(prediction, targets)

grads = torch.func.grad(compute_loss)(params, inputs, targets)
```

以下是一个计算模型参数雅可比矩阵的示例：

```python
# ---------------
# 使用 functorch
# ---------------
import torch
import functorch
inputs = torch.randn(64, 3)
model = torch.nn.Linear(3, 3)

fmodel, params = functorch.make_functional(model)
jacobians = functorch.jacrev(fmodel)(params, inputs)

# ------------------------------------
# 使用 torch.func (自 PyTorch 2.0 起)
# ------------------------------------
import torch
from torch.func import jacrev, functional_call
inputs = torch.randn(64, 3)
model = torch.nn.Linear(3, 3)

params = dict(model.named_parameters())
# jacrev 默认计算 argnums=0 的雅可比矩阵。
# 我们将其设置为 1 以计算参数的雅可比矩阵
jacobians = jacrev(functional_call, argnums=1)(model, params, (inputs,))
```

请注意，对于内存消耗而言，重要的是您应该只保留一份参数副本。`model.named_parameters()` 不会复制参数。如果在模型训练中您就地更新模型的参数，那么作为模型的 `nn.Module` 拥有参数的唯一副本，一切正常。

然而，如果您想在字典中携带参数并进行非就地更新，那么就会存在两份参数副本：一份在字典中，另一份在 `model` 中。在这种情况下，您应该通过 `model.to('meta')` 将模型转换为元设备，使其不占用内存。

### functorch.combine_state_for_ensemble

请使用 `torch.func.stack_module_state` 替代
[functorch.combine_state_for_ensemble](https://pytorch.org/functorch/1.13/generated/functorch.combine_state_for_ensemble.html)
`torch.func.stack_module_state` 返回两个字典，一个包含堆叠的参数，另一个包含堆叠的缓冲区，然后可以与 `torch.vmap` 和 `torch.func.functional_call`
一起用于集成学习。

例如，以下是一个如何对非常简单的模型进行集成学习的示例：

```python
import torch
num_models = 5
batch_size = 64
in_features, out_features = 3, 3
models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
data = torch.randn(batch_size, 3)

# ---------------
# 使用 functorch
# ---------------
import functorch
fmodel, params, buffers = functorch.combine_state_for_ensemble(models)
output = functorch.vmap(fmodel, (0, 0, None))(params, buffers, data)
assert output.shape == (num_models, batch_size, out_features)

# ------------------------------------
# 使用 torch.func（自 PyTorch 2.0 起）
# ------------------------------------
import copy

# 通过将张量放在元设备上，构建一个不占用内存的模型版本。
base_model = copy.deepcopy(models[0])
base_model.to('meta')

params, buffers = torch.func.stack_module_state(models)

# 可以直接对 torch.func.functional_call 进行 vmap 操作，
# 但将其包装在函数中可以使过程更清晰。
def call_single_model(params, buffers, data):
    return torch.func.functional_call(base_model, (params, buffers), (data,))

output = torch.vmap(call_single_model, (0, 0, None))(params, buffers, data)
assert output.shape == (num_models, batch_size, out_features)
```

## functorch.compile

我们不再支持将 functorch.compile（也称为 AOTAutograd）
作为 PyTorch 中编译的前端；我们已经将 AOTAutograd
集成到 PyTorch 的编译体系中。如果您是用户，请使用
`torch.compile` 替代。
