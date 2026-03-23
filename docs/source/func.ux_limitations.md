```{eval-rst}
.. currentmodule:: torch.func
```

(ux-limitations)=

# 用户体验限制

torch.func 与 [JAX](https://github.com/google/jax) 类似，对于可转换的内容存在限制。一般来说，JAX 的限制在于变换仅适用于纯函数：即输出完全由输入决定且不涉及副作用（如数据修改）的函数。

我们提供类似的保证：我们的变换能很好地处理纯函数。然而，我们确实支持某些原地操作。一方面，编写与函数变换兼容的代码可能需要改变你编写 PyTorch 代码的方式；另一方面，你可能会发现我们的变换让你能够表达以前在 PyTorch 中难以表达的内容。

## 通用限制

所有 torch.func 变换都有一个共同限制：函数不应为全局变量赋值。相反，函数的所有输出都必须从函数返回。这个限制源于 torch.func 的实现方式：每个变换都将张量输入包装在特殊的 torch.func 张量子类中，以促进变换。

因此，不要这样做：

```python
import torch
from torch.func import grad

# 不要这样做
intermediate = None

def f(x):
  global intermediate
  intermediate = x.sin()
  z = intermediate.sin()
  return z

x = torch.randn([])
grad_x = grad(f)(x)
```

请将 `f` 重写为返回 `intermediate`：

```python
def f(x):
  intermediate = x.sin()
  z = intermediate.sin()
  return z, intermediate

grad_x, intermediate = grad(f, has_aux=True)(x)
```

## torch.autograd API

如果你尝试在由 {func}`vmap` 或 torch.func 的自动微分变换（{func}`vjp`、{func}`jvp`、{func}`jacrev`、{func}`jacfwd`）转换的函数内部使用 `torch.autograd` API（如 `torch.autograd.grad` 或 `torch.autograd.backward`），变换可能无法对其进行转换。如果无法转换，你将收到错误信息。

这是 PyTorch 自动微分支持实现方式的一个基本设计限制，也是我们设计 torch.func 库的原因。请改用 torch.func 中对应的 `torch.autograd` API：
- `torch.autograd.grad`、`Tensor.backward` -> `torch.func.vjp` 或 `torch.func.grad`
- `torch.autograd.functional.jvp` -> `torch.func.jvp`
- `torch.autograd.functional.jacobian` -> `torch.func.jacrev` 或 `torch.func.jacfwd`
- `torch.autograd.functional.hessian` -> `torch.func.hessian`

## vmap 限制

:::{note}
{func}`vmap` 是我们限制最严格的变换。
与梯度相关的变换（{func}`grad`、{func}`vjp`、{func}`jvp`）没有这些限制。{func}`jacfwd`（以及使用 {func}`jacfwd` 实现的 {func}`hessian`）是 {func}`vmap` 和 {func}`jvp` 的组合，因此也具有这些限制。
:::

`vmap(func)` 是一个变换，它返回一个函数，该函数将 `func` 映射到每个输入张量的某个新维度上。vmap 的心理模型类似于运行 for 循环：对于纯函数（即没有副作用的情况），`vmap(f)(x)` 等价于：

```python
torch.stack([f(x_i) for x_i in x.unbind(0)])
```

### 修改：Python 数据结构的任意修改

存在副作用时，{func}`vmap` 不再表现得像在运行 for 循环。例如，以下函数：

```python
def f(x, list):
  list.pop()
  print("hello!")
  return x.sum(0)

x = torch.randn(3, 1)
lst = [0, 1, 2, 3]

result = vmap(f, in_dims=(0, None))(x, lst)
```

只会打印一次 "hello!" 并从 `lst` 中弹出一个元素。

{func}`vmap` 只执行一次 `f`，因此所有副作用只发生一次。

这是 vmap 实现方式的结果。torch.func 有一个特殊的内部 BatchedTensor 类。`vmap(f)(*inputs)` 获取所有张量输入，将它们转换为 BatchedTensors，并调用 `f(*batched_tensor_inputs)`。BatchedTensor 重写了 PyTorch API，为每个 PyTorch 操作符产生批处理（即向量化）行为。

### 修改：原地 PyTorch 操作

你可能因为收到关于 vmap 不兼容的原地操作错误而来到这里。如果 {func}`vmap` 遇到不支持的 PyTorch 原地操作，它将引发错误；否则会成功。不支持的操作是那些会导致将更多元素的张量写入到元素较少的张量的操作。以下是一个可能发生这种情况的示例：

```python
def f(x, y):
  x.add_(y)
  return x

x = torch.randn(1)
y = torch.randn(3, 1)  # 当被 vmap 处理时，看起来形状为 [1]

# 引发错误，因为 `x` 的元素比 `y` 少。
vmap(f, in_dims=(None, 0))(x, y)
```

`x` 是一个有一个元素的张量，`y` 是一个有三个元素的张量。`x + y` 有三个元素（由于广播），但尝试将三个元素写回 `x`（它只有一个元素）会引发错误，因为尝试将三个元素写入只有一个元素的张量。

如果被写入的张量在 {func}`~torch.vmap` 下被批处理（即它正在被 vmap 处理），则没有问题。

```python
def f(x, y):
  x.add_(y)
  return x

x = torch.randn(3, 1)
y = torch.randn(3, 1)
expected = x + y

# 不会引发错误，因为 x 正在被 vmap 处理。
vmap(f, in_dims=(0, 0))(x, y)
assert torch.allclose(x, expected)
```

一个常见的解决方法是使用对应的 "new\_\*" 方法替换工厂函数的调用。例如：

- 将 {func}`torch.zeros` 替换为 {meth}`Tensor.new_zeros`
- 将 {func}`torch.empty` 替换为 {meth}`Tensor.new_empty`

要理解为什么这有帮助，请考虑以下示例。

```python
def diag_embed(vec):
  assert vec.dim() == 1
  result = torch.zeros(vec.shape[0], vec.shape[0])
  result.diagonal().copy_(vec)
  return result

vecs = torch.tensor([[0., 1, 2], [3., 4, 5]])

# RuntimeError: vmap: inplace arithmetic(self, *extra_args) is not possible ...
vmap(diag_embed)(vecs)
```

在 {func}`~torch.vmap` 内部，`result` 是一个形状为 [3, 3] 的张量。
然而，尽管 `vec` 看起来形状是 [3]，但实际上 `vec` 的底层形状是 [2, 3]。
无法将 `vec` 复制到形状为 [3] 的 `result.diagonal()` 中，因为它的元素数量过多。

```python
def diag_embed(vec):
  assert vec.dim() == 1
  result = vec.new_zeros(vec.shape[0], vec.shape[0])
  result.diagonal().copy_(vec)
  return result

vecs = torch.tensor([[0., 1, 2], [3., 4, 5]])
vmap(diag_embed)(vecs)
```

将 {func}`torch.zeros` 替换为 {meth}`Tensor.new_zeros` 后，`result` 的底层张量形状变为 [2, 3, 3]，因此现在可以将底层形状为 [2, 3] 的 `vec` 复制到 `result.diagonal()` 中。

### 修改：PyTorch 操作中的 out= 参数

{func}`vmap` 不支持 PyTorch 操作中的 `out=` 关键字参数。
如果在代码中遇到此参数，它将优雅地报错。

这不是一个根本性的限制；理论上我们未来可以支持此功能，但目前选择暂不支持。

### 数据依赖的 Python 控制流

我们尚不支持对数据依赖的控制流进行 `vmap` 操作。数据依赖的控制流是指 if 语句、while 循环或 for 循环的条件是一个正在被 `vmap` 操作的张量。例如，以下代码将引发错误：

```python
def relu(x):
  if x > 0:
    return x
  return 0

x = torch.randn(3)
vmap(relu)(x)
```

但是，任何不依赖于 `vmap` 操作张量中值的控制流都可以正常工作：

```python
def custom_dot(x):
  if x.dim() == 1:
    return torch.dot(x, x)
  return (x * x).sum()

x = torch.randn(3)
vmap(custom_dot)(x)
```

JAX 支持使用特殊的控制流操作符（例如 `jax.lax.cond`、`jax.lax.while_loop`）对[数据依赖的控制流](https://jax.readthedocs.io/en/latest/jax.lax.html#control-flow-operators)进行变换。
我们正在研究为 PyTorch 添加等效的操作符。

### 数据依赖的操作 (.item())

我们不支持（并且将来也不会支持）对调用张量 `.item()` 方法的用户定义函数进行 vmap 操作。例如，以下代码将引发错误：

```python
def f(x):
  return x.item()

x = torch.randn(3)
vmap(f)(x)
```

请尝试重写代码，避免使用 `.item()` 调用。

您也可能遇到关于使用 `.item()` 的错误信息，但您可能并未使用它。在这些情况下，可能是 PyTorch 内部调用了 `.item()` —— 请在 GitHub 上提交问题，我们将修复 PyTorch 内部实现。

### 动态形状操作 (nonzero 及类似操作)

`vmap(f)` 要求 `f` 应用于输入中的每个"示例"时返回具有相同形状的张量。因此，不支持诸如 `torch.nonzero`、`torch.is_nonzero` 等操作，并且会报错。

要理解原因，请考虑以下示例：

```python
xs = torch.tensor([[0, 1, 2], [0, 0, 3]])
vmap(torch.nonzero)(xs)
```

`torch.nonzero(xs[0])` 返回一个形状为 2 的张量；
但 `torch.nonzero(xs[1])` 返回一个形状为 1 的张量。
我们无法构造一个单一的张量作为输出；
输出将需要是一个不规则张量（而 PyTorch 目前还没有不规则张量的概念）。

## 随机性

调用随机操作时，用户的意图可能不明确。具体来说，一些用户可能希望跨批次的随机行为相同，而另一些用户可能希望跨批次的随机行为不同。为了解决这个问题，`vmap` 接受一个随机性标志。

该标志只能传递给 vmap，可以取 3 个值："error"、"different" 或 "same"，默认为 error。在 "error" 模式下，任何对随机函数的调用都会产生错误，要求用户根据其用例使用其他两个标志之一。

在 "different" 随机性下，批次中的元素产生不同的随机值。例如，

```python
def add_noise(x):
  y = torch.randn(())  # y 在批次中会不同
  return x + y

x = torch.ones(3)
result = vmap(add_noise, randomness="different")(x)  # 我们得到 3 个不同的值
```

在 "same" 随机性下，批次中的元素产生相同的随机值。例如，

```python
def add_noise(x):
  y = torch.randn(())  # y 在批次中会相同
  return x + y

x = torch.ones(3)
result = vmap(add_noise, randomness="same")(x)  # 我们得到相同的值，重复 3 次
```

:::{warning}
我们的系统只能确定 PyTorch 操作符的随机性行为，无法控制其他库（如 numpy）的行为。这与 JAX 解决方案的局限性类似。
:::

:::{note}
使用任一支持的随机性类型进行多次 vmap 调用不会产生相同的结果。与标准 PyTorch 一样，用户可以通过在 vmap 外部使用 `torch.manual_seed()` 或使用生成器来实现随机性可复现性。
:::

:::{note}
最后，我们的随机性与 JAX 不同，因为我们没有使用无状态 PRNG，部分原因是 PyTorch 不完全支持无状态 PRNG。相反，我们引入了一个标志系统，以允许我们所见的最常见的随机性形式。如果您的用例不符合这些随机性形式，请提交问题。
:::