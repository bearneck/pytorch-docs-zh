(complex_numbers-doc)=

# 复数

复数是可以表示为 {math}`a + bj` 形式的数，其中 a 和 b 是实数，*j* 称为虚数单位，满足方程 {math}`j^2 = -1`。复数在数学和工程领域经常出现，尤其是在信号处理等主题中。传统上，许多用户和库（例如 TorchAudio）通过使用形状为 {math}`(..., 2)` 的浮点张量来表示数据来处理复数，其中最后一个维度包含实部和虚部值。

复数数据类型的张量为处理复数提供了更自然的用户体验。对复数张量的操作（例如 {func}`torch.mv`、{func}`torch.matmul`）可能比模拟它们的浮点张量操作更快且内存效率更高。PyTorch 中涉及复数的操作经过优化，可以使用向量化汇编指令和专用内核（例如 LAPACK、cuBlas）。

```{note}
[torch.fft 模块](https://pytorch.org/docs/stable/fft.html#torch-fft) 中的频谱操作支持原生复数张量。
```

```{warning}
复数张量是测试版功能，可能会发生变化。
```

## 创建复数张量

我们支持两种复数数据类型：`torch.cfloat` 和 `torch.cdouble`

```python
>>> x = torch.randn(2,2, dtype=torch.cfloat)
>>> x
tensor([[-0.4621-0.0303j, -0.2438-0.5874j],
     [ 0.7706+0.1421j,  1.2110+0.1918j]])
```

```{note}
复数张量的默认数据类型由默认浮点数据类型决定。
如果默认浮点数据类型是 `torch.float64`，则复数被推断为 `torch.complex128` 数据类型，否则它们被假定为 `torch.complex64` 数据类型。
```

除了 {func}`torch.linspace`、{func}`torch.logspace` 和 {func}`torch.arange` 之外的所有工厂函数都支持复数张量。

## 从旧表示法过渡

当前使用形状为 {math}`(..., 2)` 的实数张量来弥补缺乏复数张量的用户，可以使用 {func}`torch.view_as_complex` 和 {func}`torch.view_as_real` 轻松切换到在代码中使用复数张量。请注意，这些函数不执行任何复制，而是返回输入张量的视图。

```python
>>> x = torch.randn(3, 2)
>>> x
tensor([[ 0.6125, -0.1681],
     [-0.3773,  1.3487],
     [-0.0861, -0.7981]])
>>> y = torch.view_as_complex(x)
>>> y
tensor([ 0.6125-0.1681j, -0.3773+1.3487j, -0.0861-0.7981j])
>>> torch.view_as_real(y)
tensor([[ 0.6125, -0.1681],
     [-0.3773,  1.3487],
     [-0.0861, -0.7981]])
```

## 访问实部和虚部

复数张量的实部和虚部值可以使用 {attr}`real` 和 {attr}`imag` 属性访问。

```{note}
访问 `real` 和 `imag` 属性不会分配任何内存，并且对 `real` 和 `imag` 张量的原地更新将更新原始复数张量。此外，返回的 `real` 和 `imag` 张量不是连续的。
```

```python
>>> y.real
tensor([ 0.6125, -0.3773, -0.0861])
>>> y.imag
tensor([-0.1681,  1.3487, -0.7981])

>>> y.real.mul_(2)
tensor([ 1.2250, -0.7546, -0.1722])
>>> y
tensor([ 1.2250-0.1681j, -0.7546+1.3487j, -0.1722-0.7981j])
>>> y.real.stride()
(2,)
```

## 角度和绝对值

复数张量的角度和绝对值可以使用 {func}`torch.angle` 和 {func}`torch.abs` 计算。

```python
>>> x1=torch.tensor([3j, 4+4j])
>>> x1.abs()
tensor([3.0000, 5.6569])
>>> x1.angle()
tensor([1.5708, 0.7854])
```

## 线性代数

许多线性代数操作，如 {func}`torch.matmul`、{func}`torch.linalg.svd`、{func}`torch.linalg.solve` 等，都支持复数。如果您希望请求我们目前不支持的操作，请[搜索](https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+complex)是否已有相关 issue 提交，如果没有，请[提交一个](https://github.com/pytorch/pytorch/issues/new/choose)。

## 序列化

复数张量可以被序列化，允许数据以复数值形式保存。

```python
>>> torch.save(y, 'complex_tensor.pt')
>>> torch.load('complex_tensor.pt')
tensor([ 0.6125-0.1681j, -0.3773+1.3487j, -0.0861-0.7981j])
```

## 自动微分

PyTorch 支持复数张量的自动微分。计算出的梯度是共轭 Wirtinger 导数，其负方向正是梯度下降算法中使用的最陡下降方向。因此，所有现有的优化器都可以直接用于复数参数。更多详细信息，请查看说明 {ref}`complex_autograd-doc`。

## 优化器

在语义上，我们将使用复数参数的 PyTorch 优化器进行步进定义为等同于在复数参数的 {func}`torch.view_as_real` 等效形式上进行相同优化器的步进。更具体地说：

```python
>>> params = [torch.rand(2, 3, dtype=torch.complex64) for _ in range(5)]
>>> real_params = [torch.view_as_real(p) for p in params]

>>> complex_optim = torch.optim.AdamW(params)
>>> real_optim = torch.optim.AdamW(real_params)
```

`real_optim` 和 `complex_optim` 将在参数上计算相同的更新，尽管两个优化器之间可能存在轻微的数值差异，类似于 foreach 与 forloop 优化器以及可捕获与默认优化器之间的数值差异。更多详细信息，请参阅[数值精度](https://pytorch.org/docs/stable/notes/numerical_accuracy.html)。

具体来说，虽然您可以将我们的优化器对复数张量的处理视为分别优化其 `p.real` 和 `p.imag` 部分，但实现细节并非完全如此。请注意，等效的 {func}`torch.view_as_real` 会将复数张量转换为形状为 {math}`(..., 2)` 的实数张量，而将复数张量拆分为两个张量会得到两个大小为 {math}`(...)` 的张量。这种区别对逐点优化器（如 AdamW）没有影响，但会在执行全局归约的优化器（如 LBFGS）中导致轻微差异。目前我们没有执行逐张量归约的优化器，因此尚未定义此行为。如果您有需要精确定义此行为的用例，请提交问题。

我们尚未完全支持以下子系统：

* 量化
* JIT
* 稀疏张量
* 分布式

如果其中任何一项对您的用例有帮助，请[搜索](https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+complex)是否已有相关问题被提交，如果没有，请[提交一个问题](https://github.com/pytorch/pytorch/issues/new/choose)。