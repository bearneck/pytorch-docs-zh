# 梯度检查机制

本文档概述了 `torch.autograd.gradcheck` 和 `torch.autograd.gradgradcheck` 函数的工作原理。

内容将涵盖实值函数和复值函数的前向模式与反向模式自动微分，以及高阶导数。本文档同时介绍了梯度检查的默认行为以及传递 `fast_mode=True` 参数时的情况（下文称为快速梯度检查）。


local

:   

## 符号与背景信息

在本文档中，我们将使用以下约定：

1.  $x$, $y$, $a$, $b$, $v$, $u$, $ur$ 和 $ui$ 是实值向量，而 $z$ 是复值向量，可以表示为两个实值向量 $z = a + i b$。
2.  $N$ 和 $M$ 是两个整数，分别表示输入和输出空间的维度。
3.  $f: \mathcal{R}^N \to \mathcal{R}^M$ 是我们的基本实到实函数，满足 $y = f(x)$。
4.  $g: \mathcal{C}^N \to \mathcal{R}^M$ 是我们的基本复到实函数，满足 $y = g(z)$。

对于简单的实到实情况，我们将与 $f$ 关联的雅可比矩阵记为 $J_f$，其大小为 $M \times N$。该矩阵包含所有偏导数，使得位置 $(i, j)$ 处的元素为 $\frac{\partial y_i}{\partial x_j}$。反向模式自动微分则针对给定的尺寸为 $M$ 的向量 $v$ 计算量 $v^T J_f$。另一方面，前向模式自动微分针对给定的尺寸为 $N$ 的向量 $u$ 计算量 $J_f u$。

对于包含复数值的函数，情况要复杂得多。我们在此仅提供概要，完整描述可参阅 `complex_autograd-doc`。

满足复可微性（柯西-黎曼方程）的约束条件对所有实值损失函数来说过于严格，因此我们选择使用 Wirtinger 微积分。在 Wirtinger 微积分的基本设定中，链式法则需要同时访问 Wirtinger 导数（下文称为 $W$）和共轭 Wirtinger 导数（下文称为 $CW$）。尽管名称如此，但通常两者并非彼此的复共轭，因此 $W$ 和 $CW$ 都需要传播。

为了避免必须同时传播两个值，对于反向模式自动微分，我们始终假设正在计算导数的函数是实值函数或是某个更大实值函数的一部分。这一假设意味着我们在反向传播过程中计算的所有中间梯度也都与实值函数相关联。实际上，在进行优化时，此假设并不具有限制性，因为此类问题需要实值目标函数（因为复数没有自然的序关系）。

在此假设下，使用 $W$ 和 $CW$ 的定义，我们可以证明 $W = CW^*$（此处使用 $*$ 表示复共轭），因此实际上只需要对两个值中的一个进行"沿图反向传播"，因为另一个可以轻松恢复。为了简化内部计算，PyTorch 使用 $2 * CW$ 作为其反向传播并在用户请求梯度时返回的值。与实值情况类似，当输出实际上在 $\mathcal{R}^M$ 中时，反向模式自动微分并不计算 $2 * CW$，而是针对给定的向量 $v \in \mathcal{R}^M$ 仅计算 $v^T (2 * CW)$。

对于前向模式自动微分，我们采用类似的逻辑，在这种情况下，假设该函数是某个输入在 $\mathcal{R}$ 中的更大函数的一部分。在此假设下，我们可以做出类似的断言，即每个中间结果都对应于一个输入在 $\mathcal{R}$ 中的函数，并且在这种情况下，使用 $W$ 和 $CW$ 的定义，我们可以证明对于中间函数有 $W = CW$。为了确保在一维函数的基本情况下，前向模式和反向模式计算相同的量，前向模式也计算 $2 * CW$。与实值情况类似，当输入实际上在 $\mathcal{R}^N$ 中时，前向模式自动微分并不计算 $2 * CW$，而是针对给定的向量 $u \in \mathcal{R}^N$ 仅计算 $(2 * CW) u$。

## 默认反向模式梯度检查行为

### 实到实函数

为了测试函数 $f: \mathcal{R}^N \to \mathcal{R}^M, x \to y$，我们通过两种方式重建完整的雅可比矩阵 $J_f$（尺寸为 $M \times N$）：解析法和数值法。解析版本使用我们的反向模式自动微分，而数值版本使用有限差分。然后逐元素比较两个重建的雅可比矩阵是否相等。

#### 默认实输入数值评估

如果我们考虑一维函数（$N = M = 1$）的基本情况，那么我们可以使用 [维基百科文章](https://en.wikipedia.org/wiki/Finite_difference) 中的基本有限差分公式。我们使用"中心差分"以获得更好的数值特性：

$$\frac{\partial y}{\partial x} \approx \frac{f(x + eps) - f(x - eps)}{2 * eps}$$

该公式可以轻松推广到多输出（$M \gt 1$）的情况，此时 $\frac{\partial y}{\partial x}$ 是一个尺寸为 $M \times 1$ 的列向量，类似于 $f(x + eps)$。在这种情况下，上述公式可以原样复用，并且仅通过两次用户函数评估（即 $f(x + eps)$ 和 $f(x - eps)$）来近似完整的雅可比矩阵。

处理多输入（$N \gt 1$）的情况在计算上更为昂贵。在此场景中，我们依次遍历所有输入，并对 $x$ 的每个元素依次应用 $eps$ 扰动。这使我们能够逐列重建 $J_f$ 矩阵。

#### 默认实输入解析求值

对于解析求值，我们利用上述事实，即反向模式 AD 计算 $v^T J_f$。 对于单输出函数，我们简单地使用 $v = 1$，通过一次反向传播即可恢复完整的雅可比矩阵。

对于多输出函数，我们采用一个循环来遍历输出，其中每个 $v$ 是一个独热向量，依次对应每个输出。这允许我们逐行重建 $J_f$ 矩阵。

### 复到实函数

为了测试函数 $g: \mathcal{C}^N \to \mathcal{R}^M, z \to y$，其中 $z = a + i b$，我们重建包含 $2 * CW$ 的（复值）矩阵。

#### 默认复输入数值求值

首先考虑基本情形，其中 $N = M = 1$。我们从 [这篇研究论文](https://arxiv.org/pdf/1701.00392.pdf) 的（第3章）得知：

$$CW := \frac{\partial y}{\partial z^*} = \frac{1}{2} * (\frac{\partial y}{\partial a} + i \frac{\partial y}{\partial b})$$

注意，上式中 $\frac{\partial y}{\partial a}$ 和 $\frac{\partial y}{\partial b}$ 是 $\mathcal{R} \to \mathcal{R}$ 导数。 为了对这些进行数值求值，我们使用上述针对实到实情况描述的方法。 这使我们能够计算 $CW$ 矩阵，然后将其乘以 $2$。

请注意，截至撰写时，代码以一种略微复杂的方式计算此值：

``` python
# 代码来自 https://github.com/pytorch/pytorch/blob/58eb23378f2a376565a66ac32c93a316c45b6131/torch/autograd/gradcheck.py#L99-L105
# 此代码块中的符号变更：
# 这里的 s 对应上面的 y
# 这里的 x, y 对应上面的 a, b

ds_dx = compute_gradient(eps)
ds_dy = compute_gradient(eps * 1j)
# 共轭 Wirtinger 导数
conj_w_d = 0.5 * (ds_dx + ds_dy * 1j)
# Wirtinger 导数
w_d = 0.5 * (ds_dx - ds_dy * 1j)
d[d_idx] = grad_out.conjugate() * conj_w_d + grad_out * w_d.conj()

# 由于 grad_out 始终为 1，且 W 和 CW 互为复共轭，最后一行最终计算的结果正是 `conj_w_d + w_d.conj() = conj_w_d + conj_w_d = 2 * conj_w_d`。
```

#### 默认复输入解析求值

由于反向模式 AD 已经精确计算了两倍的 $CW$ 导数，我们在此简单地使用与实到实情况相同的技巧，并在存在多个实输出时逐行重建矩阵。

### 具有复输出的函数

在这种情况下，用户提供的函数不遵循自动微分（autograd）的假设，即我们计算反向 AD 的函数是实值的。 这意味着直接对此函数使用自动微分是没有明确定义的。 为了解决这个问题，我们将用两个函数 $hr$ 和 $hi$ 来替换对函数 $h: \mathcal{P}^N \to \mathcal{C}^M$（其中 $\mathcal{P}$ 可以是 $\mathcal{R}$ 或 $\mathcal{C}$）的测试，这两个函数定义为：

$$\begin{aligned}
\begin{aligned}
hr(q) &:= real(f(q)) \\
hi(q) &:= imag(f(q))
\end{aligned}
\end{aligned}$$

其中 $q \in \mathcal{P}$。 然后，我们根据 $\mathcal{P}$ 的情况，使用上述的实到实或复到实情况，对 $hr$ 和 $hi$ 分别进行基本的梯度检查。

请注意，截至撰写时，代码并未显式创建这些函数，而是通过向不同函数传递 $\text{grad\_out}$ 参数，手动结合 $real$ 或 $imag$ 函数进行链式法则计算。 当 $\text{grad\_out} = 1$ 时，我们考虑的是 $hr$。 当 $\text{grad\_out} = 1j$ 时，我们考虑的是 $hi$。

## 快速反向模式梯度检查

虽然上述梯度检查的表述在确保正确性和可调试性方面都很出色，但由于它重建了完整的雅可比矩阵，因此速度非常慢。 本节介绍一种在不影响正确性的情况下执行梯度检查的更快方法。当我们检测到错误时，可以通过添加特殊逻辑来恢复可调试性。在这种情况下，我们可以运行重建完整矩阵的默认版本，向用户提供完整的详细信息。

这里的高层策略是找到一个标量，该标量可以通过数值和解析方法高效计算，并且能够足够好地代表慢速梯度检查计算的完整矩阵，以确保它能捕捉到雅可比矩阵中的任何差异。

### 实到实函数的快速梯度检查

我们在此要计算的标量是 $v^T J_f u$，其中 $v \in \mathcal{R}^M$ 是一个给定的随机向量，$u \in \mathcal{R}^N$ 是一个随机单位范数向量。

对于数值求值，我们可以高效地计算

$$J_f u \approx \frac{f(x + u * eps) - f(x - u * eps)}{2 * eps}.$$

然后，我们计算此向量与 $v$ 的点积，得到感兴趣的标量值。

对于解析版本，我们可以使用反向模式 AD 直接计算 $v^T J_f$。然后我们计算其与 $u$ 的点积，得到期望值。

### 复到实函数的快速梯度检查

与实到实情况类似，我们希望执行完整矩阵的约简。但 $2 * CW$ 矩阵是复值的，因此在这种情况下，我们将比较复标量。

由于数值计算中可高效计算的内容存在一些限制，并且为了将数值评估次数保持在最低水平，我们计算以下（尽管令人惊讶的）标量值：

$$s := 2 * v^T (real(CW) ur + i * imag(CW) ui)$$

其中 $v \in \mathcal{R}^M$、$ur \in \mathcal{R}^N$ 和 $ui \in \mathcal{R}^N$。

#### 快速复数输入数值评估

我们首先考虑如何用数值方法计算 $s$。为此，请记住我们考虑的是 $g: \mathcal{C}^N \to \mathcal{R}^M, z \to y$，其中 $z = a + i b$，并且 $CW = \frac{1}{2} * (\frac{\partial y}{\partial a} + i \frac{\partial y}{\partial b})$，我们将其重写如下：

$$\begin{aligned}
\begin{aligned}
s &= 2 * v^T (real(CW) ur + i * imag(CW) ui) \\
&= 2 * v^T (\frac{1}{2} * \frac{\partial y}{\partial a} ur + i * \frac{1}{2} * \frac{\partial y}{\partial b} ui) \\
&= v^T (\frac{\partial y}{\partial a} ur + i * \frac{\partial y}{\partial b} ui) \\
&= v^T ((\frac{\partial y}{\partial a} ur) + i * (\frac{\partial y}{\partial b} ui))
\end{aligned}
\end{aligned}$$

在此公式中，我们可以看到 $\frac{\partial y}{\partial a} ur$ 和 $\frac{\partial y}{\partial b} ui$ 可以用与实到实情况的快速版本相同的方式评估。 一旦计算出这些实值量，我们就可以重建右侧的复向量，并与实值向量 $v$ 进行点积运算。

#### 快速复数输入解析评估

对于解析情况，事情更简单，我们将公式重写为：

$$\begin{aligned}
\begin{aligned}
s &= 2 * v^T (real(CW) ur + i * imag(CW) ui) \\
&= v^T real(2 * CW) ur + i * v^T imag(2 * CW) ui) \\
&= real(v^T (2 * CW)) ur + i * imag(v^T (2 * CW)) ui
\end{aligned}
\end{aligned}$$

因此，我们可以利用反向模式自动微分为我们提供计算 $v^T (2 * CW)$ 的高效方法这一事实，然后在重建最终的复标量 $s$ 之前，将其实部与 $ur$ 进行点积运算，虚部与 $ui$ 进行点积运算。

#### 为什么不使用复数 $u$

此时，您可能想知道为什么我们没有选择复数 $u$ 并直接执行约简 $2 * v^T CW u'$。 为了深入探讨这一点，在本段中，我们将使用 $u$ 的复数版本，记为 $u' = ur' + i ui'$。 使用这样的复数 $u'$，问题在于在进行数值评估时，我们需要计算：

$$\begin{aligned}
\begin{aligned}
2*CW u' &= (\frac{\partial y}{\partial a} + i \frac{\partial y}{\partial b})(ur' + i ui') \\
&= \frac{\partial y}{\partial a} ur' + i \frac{\partial y}{\partial a} ui' + i \frac{\partial y}{\partial b} ur' - \frac{\partial y}{\partial b} ui'
\end{aligned}
\end{aligned}$$

这将需要四次实到实有限差分的评估（是上述方法的两倍）。 由于这种方法没有更多的自由度（相同数量的实值变量），并且我们在此尝试获得尽可能快的评估，因此我们使用上述另一种公式。

### 具有复数输出的函数的快速梯度检查

与慢速情况类似，我们考虑两个实值函数，并为每个函数使用上述适当的规则。

## 梯度梯度检查实现

PyTorch 还提供了一个实用程序来验证二阶梯度。此处的目标是确保反向实现也是可微的，并且计算正确。

此功能通过考虑函数 $F: x, v \to v^T J_f$ 并在该函数上使用上述定义的梯度检查来实现。 请注意，此处的 $v$ 只是一个与 $f(x)$ 类型相同的随机向量。

梯度梯度检查的快速版本是通过在同一函数 $F$ 上使用梯度检查的快速版本来实现的。
