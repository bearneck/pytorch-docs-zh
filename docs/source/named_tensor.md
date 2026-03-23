```{eval-rst}
.. currentmodule:: torch
```

(named_tensors-doc)=

# 命名张量

命名张量允许用户为张量的维度指定显式名称。在大多数情况下，接受维度参数的操作将接受维度名称，从而无需通过位置来跟踪维度。此外，命名张量使用名称在运行时自动检查 API 是否被正确使用，提供了额外的安全性。名称还可用于重新排列维度，例如，支持"按名称广播"而非"按位置广播"。

```{warning}
    命名张量 API 是一个原型功能，可能会发生变化。
```

## 创建命名张量

工厂函数现在接受一个新的 {attr}`names` 参数，该参数将名称与每个维度关联起来。

```
    >>> torch.zeros(2, 3, names=('N', 'C'))
    tensor([[0., 0., 0.],
            [0., 0., 0.]], names=('N', 'C'))
```

命名维度与常规张量维度一样是有序的。``tensor.names[i]`` 是 ``tensor`` 第 ``i`` 维的名称。

以下工厂函数支持命名张量：

- {func}`torch.empty`
- {func}`torch.rand`
- {func}`torch.randn`
- {func}`torch.ones`
- {func}`torch.tensor`
- {func}`torch.zeros`

## 命名维度

关于张量名称的限制，请参阅 {attr}`~Tensor.names`。

使用 {attr}`~Tensor.names` 访问张量的维度名称，并使用 {meth}`~Tensor.rename` 重命名维度。

```
    >>> imgs = torch.randn(1, 2, 2, 3 , names=('N', 'C', 'H', 'W'))
    >>> imgs.names
    ('N', 'C', 'H', 'W')

    >>> renamed_imgs = imgs.rename(H='height', W='width')
    >>> renamed_imgs.names
    ('N', 'C', 'height', 'width)
```

命名张量可以与未命名张量共存；命名张量是 {class}`torch.Tensor` 的实例。未命名张量的维度名称为 ``None``。命名张量并不要求所有维度都被命名。

```
    >>> imgs = torch.randn(1, 2, 2, 3 , names=(None, 'C', 'H', 'W'))
    >>> imgs.names
    (None, 'C', 'H', 'W')
```

## 名称传播语义

命名张量使用名称在运行时自动检查 API 是否被正确调用。这发生在称为*名称推断*的过程中。更正式地说，名称推断包括以下两个步骤：

- **检查名称**：操作符可以在运行时执行自动检查，验证某些维度名称必须匹配。
- **传播名称**：名称推断将名称传播到输出张量。

所有支持命名张量的操作都会传播名称。

```
    >>> x = torch.randn(3, 3, names=('N', 'C'))
    >>> x.abs().names
    ('N', 'C')
```

(match_semantics-doc)=
### 匹配语义

如果两个名称相等（字符串相等）或者至少有一个是 ``None``，则它们*匹配*。None 本质上是一种特殊的"通配符"名称。

``unify(A, B)`` 决定将名称 ``A`` 和 ``B`` 中的哪一个传播到输出。如果它们匹配，则返回两者中更*具体*的名称。如果名称不匹配，则会报错。

```{note}
在实践中，使用命名张量时，应避免存在未命名维度，因为它们的处理可能很复杂。建议使用 {meth}`~Tensor.refine_names` 将所有未命名维度提升为命名维度。
```

### 基本名称推断规则

让我们看看在没有广播的情况下，对两个一维张量进行加法运算时，``match`` 和 ``unify`` 是如何用于名称推断的。

```
    x = torch.randn(3, names=('X',))
    y = torch.randn(3)
    z = torch.randn(3, names=('Z',))
```

**检查名称**：检查两个张量的名称是否*匹配*。

对于以下示例：

```
    >>> # x + y  # match('X', None) is True
    >>> # x + z  # match('X', 'Z') is False
    >>> # x + x  # match('X', 'X') is True

    >>> x + z
    Error when attempting to broadcast dims ['X'] and dims ['Z']: dim 'X' and dim 'Z' are at the same position from the right but do not match.
```

**传播名称**：*unify* 名称以选择要传播的名称。在 ``x + y`` 的情况下，``unify('X', None) = 'X'``，因为 ``'X'`` 比 ``None`` 更具体。

```
    >>> (x + y).names
    ('X',)
    >>> (x + x).names
    ('X',)
```

有关名称推断规则的完整列表，请参阅 {ref}`name_inference_reference-doc`。以下是两个常见操作，可能有助于理解：

- 二元算术运算：{ref}`unifies_names_from_inputs-doc`
- 矩阵乘法运算：{ref}`contracts_away_dims-doc`

## 按名称显式对齐

使用 {meth}`~Tensor.align_as` 或 {meth}`~Tensor.align_to` 按名称将张量维度对齐到指定的顺序。这对于执行"按名称广播"非常有用。

```
    # 此函数对 `input` 的维度顺序不敏感，
    # 只要它有一个 `C` 维度即可。
    def scale_channels(input, scale):
        scale = scale.refine_names('C')
        return input * scale.align_as(input)

    >>> num_channels = 3
    >>> scale = torch.randn(num_channels, names=('C',))
    >>> imgs = torch.rand(3, 3, 3, num_channels, names=('N', 'H', 'W', 'C'))
    >>> more_imgs = torch.rand(3, num_channels, 3, 3, names=('N', 'C', 'H', 'W'))
    >>> videos = torch.randn(3, num_channels, 3, 3, 3, names=('N', 'C', 'H', 'W', 'D')

    >>> scale_channels(imgs, scale)
    >>> scale_channels(more_imgs, scale)
    >>> scale_channels(videos, scale)
```

## 操作维度

使用 {meth}`~Tensor.align_to` 可以置换大量维度，而无需像 {meth}`~Tensor.permute` 那样提及所有维度。

```
    >>> tensor = torch.randn(2, 2, 2, 2, 2, 2)
    >>> named_tensor = tensor.refine_names('A', 'B', 'C', 'D', 'E', 'F')

    # 将 F 维度（第 5 维）和 E 维度（第 4 维）移到前面，同时保持其余维度顺序不变
    >>> tensor.permute(5, 4, 0, 1, 2, 3)
    >>> named_tensor.align_to('F', 'E', ...)
```

分别使用 {meth}`~Tensor.flatten` 和 {meth}`~Tensor.unflatten` 来展平和展开维度。这些方法比 {meth}`~Tensor.view` 和 {meth}`~Tensor.reshape` 更冗长，但对于阅读代码的人来说具有更多的语义含义。

```
    >>> imgs = torch.randn(32, 3, 128, 128)
    >>> named_imgs = imgs.refine_names('N', 'C', 'H', 'W')

    >>> flat_imgs = imgs.view(32, -1)
    >>> named_flat_imgs = named_imgs.flatten(['C', 'H', 'W'], 'features')
    >>> named_flat_imgs.names
    ('N', 'features')

    >>> unflattened_named_imgs = named_flat_imgs.unflatten('features', [('C', 3), ('H', 128), ('W', 128)])
    >>> unflattened_named_imgs.names
    ('N', 'C', 'H', 'W')
```

(named_tensors_autograd-doc)=
## 自动求导支持

目前自动求导对命名张量的支持有限：自动求导会忽略所有张量上的名称。梯度计算仍然是正确的，但我们失去了名称提供的安全保障。

```
    >>> x = torch.randn(3, names=('D',))
    >>> weight = torch.randn(3, names=('D',), requires_grad=True)
    >>> loss = (x - weight).abs()
    >>> grad_loss = torch.randn(3)
    >>> loss.backward(grad_loss)
    >>> weight.grad  # 目前未命名。未来会命名
    tensor([-1.8107, -0.6357,  0.0783])

    >>> weight.grad.zero_()
    >>> grad_loss = grad_loss.refine_names('C')
    >>> loss = (x - weight).abs()
    # 理想情况下我们会检查 loss 和 grad_loss 的名称是否匹配，但目前尚未实现。
    >>> loss.backward(grad_loss)
    >>> weight.grad
    tensor([-1.8107, -0.6357,  0.0783])
```

## 当前支持的操作和子系统

### 运算符

有关支持的 torch 和张量操作的完整列表，请参阅 {ref}`name_inference_reference-doc`。以下操作目前尚未支持（链接中未涵盖）：

- 索引、高级索引。

对于 ``torch.nn.functional`` 运算符，我们支持以下内容：

- {func}`torch.nn.functional.relu`
- {func}`torch.nn.functional.softmax`
- {func}`torch.nn.functional.log_softmax`
- {func}`torch.nn.functional.tanh`
- {func}`torch.nn.functional.sigmoid`
- {func}`torch.nn.functional.dropout`

### 子系统

自动求导已支持，请参阅 {ref}`named_tensors_autograd-doc`。
由于梯度目前未命名，优化器可能可以工作但未经测试。

NN 模块目前不支持。这可能导致在使用命名张量输入调用模块时出现以下情况：

- NN 模块参数未命名，因此输出可能部分命名。
- NN 模块的前向传播代码不支持命名张量，会相应地报错。

以下子系统我们也不支持，尽管有些可能开箱即用：

- 分布
- 序列化 ({func}`torch.load`, {func}`torch.save`)
- 多进程
- JIT
- 分布式
- ONNX

如果其中任何一项对您的用例有帮助，请[搜索是否已有相关 issue 提交](https://github.com/pytorch/pytorch/issues?q=is%3Aopen+is%3Aissue+label%3A%22module%3A+named+tensor%22)，如果没有，请[提交一个](https://github.com/pytorch/pytorch/issues/new/choose)。

## 命名张量 API 参考

在本节中，请查阅命名张量特定 API 的文档。有关名称如何通过其他 PyTorch 运算符传播的完整参考，请参阅 {ref}`name_inference_reference-doc`。

```{eval-rst}
.. class:: Tensor()
   :noindex:

   .. autoattribute:: names

   .. automethod:: rename

   .. automethod:: rename_

   .. automethod:: refine_names

   .. automethod:: align_as

   .. automethod:: align_to

   .. py:method:: flatten(dims, out_dim) -> Tensor
      :noindex:

      将 :attr:`dims` 展平为具有名称 :attr:`out_dim` 的单个维度。

      `dims` 中的所有维度在 :attr:`self` 张量中必须是顺序连续的，但在内存中不一定是连续的。

      示例::

          >>> imgs = torch.randn(32, 3, 128, 128, names=('N', 'C', 'H', 'W'))
          >>> flat_imgs = imgs.flatten(['C', 'H', 'W'], 'features')
          >>> flat_imgs.names, flat_imgs.shape
          (('N', 'features'), torch.Size([32, 49152]))

      .. warning::
          命名张量 API 是实验性的，可能会发生变化。
```