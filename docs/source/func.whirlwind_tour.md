# torch.func 快速入门

## 什么是 torch.func？

```{eval-rst}
.. currentmodule:: torch.func
```

torch.func，原名 functorch，是一个为 PyTorch 提供类似 [JAX](https://github.com/google/jax) 风格的可组合函数变换的库。

- “函数变换”是一个高阶函数，它接受一个数值函数并返回一个计算不同量的新函数。
- torch.func 包含自动微分变换（`grad(f)` 返回一个计算 `f` 梯度的函数）、向量化/批处理变换（`vmap(f)` 返回一个在输入批次上计算 `f` 的函数）等。
- 这些函数变换可以任意相互组合。例如，组合 `vmap(grad(f))` 可以计算一种称为“逐样本梯度”的量，这是当前标准 PyTorch 无法高效计算的。

## 为什么需要可组合的函数变换？

当前在 PyTorch 中有一些用例实现起来比较棘手：

- 计算逐样本梯度（或其他逐样本量）
- 在单台机器上运行模型集成
- 在 MAML 的内循环中高效地批处理任务
- 高效计算雅可比矩阵和海森矩阵
- 高效计算批处理雅可比矩阵和海森矩阵

组合 {func}`vmap`、{func}`grad`、{func}`vjp` 和 {func}`jvp` 变换，使我们能够表达上述内容，而无需为每个用例设计单独的子系统。

## 有哪些变换？

### {func}`grad`（梯度计算）

`grad(func)` 是我们的梯度计算变换。它返回一个新函数，用于计算 `func` 的梯度。它假设 `func` 返回一个单元素张量，并且默认计算 `func` 输出相对于第一个输入的梯度。

```python
import torch
from torch.func import grad
x = torch.randn([])
cos_x = grad(lambda x: torch.sin(x))(x)
assert torch.allclose(cos_x, x.cos())

# 二阶梯度
neg_sin_x = grad(grad(lambda x: torch.sin(x)))(x)
assert torch.allclose(neg_sin_x, -x.sin())
```

### {func}`vmap`（自动向量化）

注意：{func}`vmap` 对其可使用的代码施加了限制。更多详情，请参阅 {ref}`ux-limitations`。

`vmap(func)(*inputs)` 是一个变换，它为 `func` 中的所有张量操作添加一个维度。`vmap(func)` 返回一个新函数，该函数将 `func` 映射到输入中每个张量的某个维度（默认为第 0 维）。

vmap 对于隐藏批次维度非常有用：可以编写一个在单个样本上运行的函数 `func`，然后使用 `vmap(func)` 将其提升为可以处理样本批次的函数，从而实现更简洁的建模体验：

```python
import torch
from torch.func import vmap
batch_size, feature_size = 3, 5
weights = torch.randn(feature_size, requires_grad=True)

def model(feature_vec):
    # 带有激活函数的简单线性模型
    assert feature_vec.dim() == 1
    return feature_vec.dot(weights).relu()

examples = torch.randn(batch_size, feature_size)
result = vmap(model)(examples)
```

当与 {func}`grad` 组合时，{func}`vmap` 可用于计算逐样本梯度：

```python
from torch.func import vmap
batch_size, feature_size = 3, 5

def model(weights,feature_vec):
    # 带有激活函数的简单线性模型
    assert feature_vec.dim() == 1
    return feature_vec.dot(weights).relu()

def compute_loss(weights, example, target):
    y = model(weights, example)
    return ((y - target) ** 2).mean()  # MSELoss

weights = torch.randn(feature_size, requires_grad=True)
examples = torch.randn(batch_size, feature_size)
targets = torch.randn(batch_size)
inputs = (weights,examples, targets)
grad_weight_per_example = vmap(grad(compute_loss), in_dims=(None, 0, 0))(*inputs)
```

### {func}`vjp`（向量-雅可比积）

{func}`vjp` 变换将 `func` 应用于 `inputs`，并返回一个新函数，该函数在给定一些 `cotangents` 张量的情况下计算向量-雅可比积（vjp）。

```python
from torch.func import vjp

inputs = torch.randn(3)
func = torch.sin
cotangents = (torch.randn(3),)

outputs, vjp_fn = vjp(func, inputs); vjps = vjp_fn(*cotangents)
```

### {func}`jvp`（雅可比-向量积）

{func}`jvp` 变换计算雅可比-向量积，也称为“前向模式自动微分”。与大多数其他变换不同，它不是高阶函数，但它返回 `func(inputs)` 的输出以及 jvp。

```python
from torch.func import jvp
x = torch.randn(5)
y = torch.randn(5)
f = lambda x, y: (x * y)
_, out_tangent = jvp(f, (x, y), (torch.ones(5), torch.ones(5)))
assert torch.allclose(out_tangent, x + y)
```

### {func}`jacrev`、{func}`jacfwd` 和 {func}`hessian`

{func}`jacrev` 变换返回一个新函数，该函数接受 `x` 并使用反向模式自动微分返回函数相对于 `x` 的雅可比矩阵。

```python
from torch.func import jacrev
x = torch.randn(5)
jacobian = jacrev(torch.sin)(x)
expected = torch.diag(torch.cos(x))
assert torch.allclose(jacobian, expected)
```

{func}`jacrev` 可以与 {func}`vmap` 组合以产生批处理雅可比矩阵：

```python
x = torch.randn(64, 5)
jacobian = vmap(jacrev(torch.sin))(x)
assert jacobian.shape == (64, 5, 5)
```

{func}`jacfwd` 是 jacrev 的直接替代品，它使用前向模式自动微分计算雅可比矩阵：

```python
from torch.func import jacfwd
x = torch.randn(5)
jacobian = jacfwd(torch.sin)(x)
expected = torch.diag(torch.cos(x))
assert torch.allclose(jacobian, expected)
```

将 {func}`jacrev` 与自身或 {func}`jacfwd` 组合可以产生海森矩阵：

```python
def f(x):
    return x.sin().sum()

x = torch.randn(5)
hessian0 = jacrev(jacrev(f))(x)
hessian1 = jacfwd(jacrev(f))(x)
```

{func}`hessian` 是一个结合了 jacfwd 和 jacrev 的便捷函数：

```python
from torch.func import hessian

def f(x):
    return x.sin().sum()

x = torch.randn(5)
hess = hessian(f)(x)
```