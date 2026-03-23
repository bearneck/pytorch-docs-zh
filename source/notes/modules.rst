.. _modules:

模块
=======

PyTorch 使用模块来表示神经网络。模块具有以下特点：

* **状态化计算的基本构建单元。**
  PyTorch 提供了丰富的模块库，并使得定义新的自定义模块变得简单，从而能够轻松构建复杂的多层神经网络。
* **与 PyTorch 的**
  `autograd <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>`_
  **系统紧密集成。** 模块使得为 PyTorch 优化器指定可学习参数以进行更新变得简单。
* **易于使用和转换。** 模块可以方便地保存和恢复，在 CPU / GPU / TPU 设备之间传输，进行剪枝、量化等操作。

本文档描述了模块，面向所有 PyTorch 用户。由于模块是 PyTorch 的基础，本文档中的许多主题在其他文档或教程中有详细阐述，这里也提供了许多相关文档的链接。

.. contents:: :local:

一个简单的自定义模块
----------------------

首先，让我们看一个 PyTorch :class:`~torch.nn.Linear` 模块的简化自定义版本。该模块对其输入应用仿射变换。

.. code-block:: python

   import torch
   from torch import nn

   class MyLinear(nn.Module):
     def __init__(self, in_features, out_features):
       super().__init__()
       self.weight = nn.Parameter(torch.randn(in_features, out_features))
       self.bias = nn.Parameter(torch.randn(out_features))

     def forward(self, input):
       return (input @ self.weight) + self.bias

这个简单的模块具有模块的以下基本特征：

* **它继承自基础 Module 类。**
  所有模块都应继承 :class:`~torch.nn.Module` 以与其他模块组合使用。
* **它定义了一些用于计算的"状态"。**
  这里，状态由随机初始化的 ``weight`` 和 ``bias`` 张量组成，它们定义了仿射变换。由于它们都被定义为 :class:`~torch.nn.parameter.Parameter`，它们会被*注册*到模块中，并在调用 :func:`~torch.nn.Module.parameters` 时自动被跟踪和返回。参数可以被视为模块计算中"可学习"的部分（稍后会详细说明）。注意，模块不一定需要有状态，也可以是无状态的。
* **它定义了一个执行计算的 forward() 函数。** 对于这个仿射变换模块，输入与 ``weight`` 参数进行矩阵乘法（使用 ``@`` 简写符号），然后加上 ``bias`` 参数以产生输出。更一般地说，模块的 ``forward()`` 实现可以执行任意涉及任意数量输入和输出的计算。

这个简单的模块展示了模块如何将状态和计算打包在一起。可以构造并调用该模块的实例：

.. code-block:: python

   m = MyLinear(4, 3)
   sample_input = torch.randn(4)
   m(sample_input)
   : tensor([-0.3037, -1.0413, -4.2057], grad_fn=<AddBackward0>)

注意模块本身是可调用的，调用它会执行其 ``forward()`` 函数。这个名称指的是"前向传播"和"反向传播"的概念，它们适用于每个模块。"前向传播"负责将模块所表示的计算应用于给定的输入（如上所示）。"反向传播"计算模块输出相对于其输入的梯度，这些梯度可用于通过梯度下降方法"训练"参数。PyTorch 的 autograd 系统会自动处理反向传播计算，因此无需为每个模块手动实现 ``backward()`` 函数。通过连续的前向/反向传播训练模块参数的过程在 :ref:`Neural Network Training with Modules` 中有详细说明。

模块注册的所有参数可以通过调用 :func:`~torch.nn.Module.parameters` 或 :func:`~torch.nn.Module.named_parameters` 进行迭代，后者包含每个参数的名称：

.. code-block:: python

   for parameter in m.named_parameters():
     print(parameter)
   : ('weight', Parameter containing:
   tensor([[ 1.0597,  1.1796,  0.8247],
           [-0.5080, -1.2635, -1.1045],
           [ 0.0593,  0.2469, -1.4299],
           [-0.4926, -0.5457,  0.4793]], requires_grad=True))
   ('bias', Parameter containing:
   tensor([ 0.3634,  0.2015, -0.8525], requires_grad=True))

通常，模块注册的参数是模块计算中应该被"学习"的方面。本文档后面部分将展示如何使用 PyTorch 的优化器之一来更新这些参数。但在讨论之前，让我们先看看模块如何相互组合。

模块作为构建块
--------------------------

模块可以包含其他模块，这使得它们成为开发更复杂功能的有用构建块。最简单的方法是使用 :class:`~torch.nn.Sequential` 模块。它允许我们将多个模块链接在一起：

.. code-block:: python

   net = nn.Sequential(
     MyLinear(4, 3),
     nn.ReLU(),
     MyLinear(3, 1)
   )

   sample_input = torch.randn(4)
   net(sample_input)
   : tensor([-0.6749], grad_fn=<AddBackward0>)

注意 :class:`~torch.nn.Sequential` 自动将第一个 ``MyLinear`` 模块的输出作为输入传递给 :class:`~torch.nn.ReLU`，并将其输出作为输入传递给第二个 ``MyLinear`` 模块。如上所示，它仅限于具有单个输入和输出的模块的顺序链接。

通常，对于最简单的用例之外的任何情况，建议定义一个自定义模块，因为这样可以完全灵活地控制子模块如何用于模块的计算。

例如，以下是一个作为自定义模块实现的简单神经网络：

.. code-block:: python

   import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = MyLinear(4, 3)
        self.l1 = MyLinear(3, 1)
    def forward(self, x):
        x = self.l0(x)
        x = F.relu(x)
        x = self.l1(x)
        return x

该模块由两个"子模块"或"子模块"（\ ``l0`` 和 ``l1``\ ）组成，它们定义了神经网络的层，并在模块的 ``forward()`` 方法中用于计算。可以通过调用 :func:`~torch.nn.Module.children` 或 :func:`~torch.nn.Module.named_children` 来遍历模块的直接子模块：

.. code-block:: python

   net = Net()
   for child in net.named_children():
     print(child)
   : ('l0', MyLinear())
   ('l1', MyLinear())

要深入到不仅仅是直接子模块，:func:`~torch.nn.Module.modules` 和 :func:`~torch.nn.Module.named_modules` 会*递归地*遍历一个模块及其所有子模块：

.. code-block:: python

   class BigNet(nn.Module):
     def __init__(self):
       super().__init__()
       self.l1 = MyLinear(5, 4)
       self.net = Net()
     def forward(self, x):
       return self.net(self.l1(x))

   big_net = BigNet()
   for module in big_net.named_modules():
     print(module)
   : ('', BigNet(
     (l1): MyLinear()
     (net): Net(
       (l0): MyLinear()
       (l1): MyLinear()
     )
   ))
   ('l1', MyLinear())
   ('net', Net(
     (l0): MyLinear()
     (l1): MyLinear()
   ))
   ('net.l0', MyLinear())
   ('net.l1', MyLinear())

有时，模块需要动态定义子模块。:class:`~torch.nn.ModuleList` 和 :class:`~torch.nn.ModuleDict` 模块在此很有用；它们可以从列表或字典中注册子模块：

.. code-block:: python

   class DynamicNet(nn.Module):
     def __init__(self, num_layers):
       super().__init__()
       self.linears = nn.ModuleList(
         [MyLinear(4, 4) for _ in range(num_layers)])
       self.activations = nn.ModuleDict({
         'relu': nn.ReLU(),
         'lrelu': nn.LeakyReLU()
       })
       self.final = MyLinear(4, 1)
     def forward(self, x, act):
       for linear in self.linears:
         x = linear(x)
         x = self.activations[act](x)
       x = self.final(x)
       return x

   dynamic_net = DynamicNet(3)
   sample_input = torch.randn(4)
   output = dynamic_net(sample_input, 'relu')

对于任何给定的模块，其参数包括其直接参数以及所有子模块的参数。这意味着调用 :func:`~torch.nn.Module.parameters` 和 :func:`~torch.nn.Module.named_parameters` 将递归地包含子参数，从而方便地优化网络中的所有参数：

.. code-block:: python

   for parameter in dynamic_net.named_parameters():
     print(parameter)
   : ('linears.0.weight', Parameter containing:
   tensor([[-1.2051,  0.7601,  1.1065,  0.1963],
           [ 3.0592,  0.4354,  1.6598,  0.9828],
           [-0.4446,  0.4628,  0.8774,  1.6848],
           [-0.1222,  1.5458,  1.1729,  1.4647]], requires_grad=True))
   ('linears.0.bias', Parameter containing:
   tensor([ 1.5310,  1.0609, -2.0940,  1.1266], requires_grad=True))
   ('linears.1.weight', Parameter containing:
   tensor([[ 2.1113, -0.0623, -1.0806,  0.3508],
           [-0.0550,  1.5317,  1.1064, -0.5562],
           [-0.4028, -0.6942,  1.5793, -1.0140],
           [-0.0329,  0.1160, -1.7183, -1.0434]], requires_grad=True))
   ('linears.1.bias', Parameter containing:
   tensor([ 0.0361, -0.9768, -0.3889,  1.1613], requires_grad=True))
   ('linears.2.weight', Parameter containing:
   tensor([[-2.6340, -0.3887, -0.9979,  0.0767],
           [-0.3526,  0.8756, -1.5847, -0.6016],
           [-0.3269, -0.1608,  0.2897, -2.0829],
           [ 2.6338,  0.9239,  0.6943, -1.5034]], requires_grad=True))
   ('linears.2.bias', Parameter containing:
   tensor([ 1.0268,  0.4489, -0.9403,  0.1571], requires_grad=True))
   ('final.weight', Parameter containing:
   tensor([[ 0.2509], [-0.5052], [ 0.3088], [-1.4951]], requires_grad=True))
   ('final.bias', Parameter containing:
   tensor([0.3381], requires_grad=True))

使用 :func:`~torch.nn.Module.to` 也可以轻松地将所有参数移动到不同的设备或更改其精度：

.. code-block:: python

   # 将所有参数移动到 CUDA 设备
   dynamic_net.to(device='cuda')

   # 更改所有参数的精度
   dynamic_net.to(dtype=torch.float64)

   dynamic_net(torch.randn(5, device='cuda', dtype=torch.float64))
   : tensor([6.5166], device='cuda:0', dtype=torch.float64, grad_fn=<AddBackward0>)

更一般地，可以通过使用 :func:`~torch.nn.Module.apply` 函数将任意函数递归地应用于模块及其子模块。例如，对模块及其子模块的参数应用自定义初始化：

.. code-block:: python

   # 定义一个初始化 Linear 权重的函数。
   # 注意这里使用了 no_grad() 来避免在自动求导图中跟踪此计算。
   @torch.no_grad()
   def init_weights(m):
     if isinstance(m, nn.Linear):
       nn.init.xavier_normal_(m.weight)
       m.bias.fill_(0.0)

   # 在模块及其子模块上递归地应用该函数。
   dynamic_net.apply(init_weights)

这些示例展示了如何通过模块组合形成复杂的神经网络，并进行便捷的操作。为了以最少的样板代码快速轻松地构建神经网络，PyTorch 在 :mod:`torch.nn` 命名空间中提供了一个包含大量高性能模块的库，这些模块执行常见的神经网络操作，如池化、卷积、损失函数等。

在下一节中，我们将给出一个完整的神经网络训练示例。

更多信息，请查看：

* PyTorch 提供的模块库：`torch.nn <https://pytorch.org/docs/stable/nn.html>`_
* 定义神经网络模块：https://pytorch.org/tutorials/beginner/examples_nn/polynomial_module.html

.. _Neural Network Training with Modules:

使用模块进行神经网络训练
------------------------------------

网络构建完成后，需要对其进行训练，其参数可以通过 :mod:`torch.optim` 中的 PyTorch 优化器轻松优化：

.. code-block:: python

   # 创建网络（来自上一节）和优化器
   net = Net()
   optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)

   # 运行一个示例训练循环，"教导"网络输出常数零函数
   for _ in range(10000):
     input = torch.randn(4)
     output = net(input)
     loss = torch.abs(output)
     net.zero_grad()
     loss.backward()
     optimizer.step()

   # 训练后，将模块切换到评估模式以进行推理、计算性能指标等。
   # （关于训练和评估模式的描述，请参见下文讨论）
   ...
   net.eval()
   ...

在这个简化的示例中，网络学会简单地输出零，因为任何非零输出都会通过使用 :func:`torch.abs` 作为损失函数而根据其绝对值受到"惩罚"。虽然这不是一个非常有趣的任务，但训练的关键部分都已涵盖：

* 创建了一个网络。
* 创建了一个优化器（在本例中是随机梯度下降优化器），并将网络的参数与之关联。
* 一个训练循环...
    * 获取输入，
    * 运行网络，
    * 计算损失，
    * 将网络参数的梯度清零，
    * 调用 loss.backward() 来更新参数的梯度，
    * 调用 optimizer.step() 将梯度应用于参数。

运行上述代码片段后，请注意网络的参数已经改变。特别是，检查 ``l1`` 的 ``weight`` 参数的值会发现其值现在更接近 0（正如预期的那样）：

.. code-block:: python

   print(net.l1.weight)
   : Parameter containing:
   tensor([[-0.0013],
           [ 0.0030],
           [-0.0008]], requires_grad=True)

请注意，上述过程完全是在网络模块处于"训练模式"下完成的。模块默认处于训练模式，可以使用 :func:`~torch.nn.Module.train` 和 :func:`~torch.nn.Module.eval` 在训练模式和评估模式之间切换。它们的行为可能因所处模式而异。例如，:class:`~torch.nn.BatchNorm` 模块在训练期间维护一个运行均值和方差，当模块处于评估模式时不会更新这些值。通常，模块在训练期间应处于训练模式，仅在推理或评估时切换到评估模式。以下是一个自定义模块的示例，它在两种模式下行为不同：

.. code-block:: python

   class ModalModule(nn.Module):
     def __init__(self):
       super().__init__()

     def forward(self, x):
       if self.training:
         # 仅在训练模式下添加一个常数。
         return x + 1.
       else:
         return x


   m = ModalModule()
   x = torch.randn(4)

   print('training mode output: {}'.format(m(x)))
   : tensor([1.6614, 1.2669, 1.0617, 1.6213, 0.5481])

   m.eval()
   print('evaluation mode output: {}'.format(m(x)))
   : tensor([ 0.6614,  0.2669,  0.0617,  0.6213, -0.4519])

训练神经网络通常可能很棘手。更多信息，请查看：

* 使用优化器：https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_optim.html。
* 神经网络训练：https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
* 自动求导简介：https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

模块状态
------------

在上一节中，我们演示了训练模块的"参数"，即可学习的计算方面。现在，如果我们想将训练好的模型保存到磁盘，可以通过保存其 ``state_dict``（即"状态字典"）来实现：

.. code-block:: python

   # 保存模块
   torch.save(net.state_dict(), 'net.pt')

   ...

   # 稍后加载模块
   new_net = Net()
   new_net.load_state_dict(torch.load('net.pt'))
   : <All keys matched successfully>

模块的 ``state_dict`` 包含影响其计算的状态。这包括但不限于模块的参数。对于某些模块，可能拥有超出参数之外的状态，这些状态影响模块计算但不可学习，这可能很有用。对于这种情况，PyTorch 提供了"缓冲区"的概念，包括"持久"和"非持久"缓冲区。以下是模块可以拥有的各种状态类型的概述：

* **参数**：可学习的计算方面；包含在 ``state_dict`` 中
* **缓冲区**：不可学习的计算方面

  * **持久**缓冲区：包含在 ``state_dict`` 中（即在保存和加载时序列化）
  * **非持久**缓冲区：不包含在 ``state_dict`` 中（即在序列化时被排除）

作为使用缓冲区的一个动机示例，考虑一个维护运行均值的简单模块。我们希望运行均值的当前值被视为模块 ``state_dict`` 的一部分，以便在加载模块的序列化形式时能够恢复它，但我们不希望它是可学习的。以下代码片段展示了如何使用 :func:`~torch.nn.Module.register_buffer` 来实现这一点：

.. code-block:: python

   class RunningMean(nn.Module):
     def __init__(self, num_features, momentum=0.9):
       super().__init__()
       self.momentum = momentum
       self.register_buffer('mean', torch.zeros(num_features))
     def forward(self, x):
       self.mean = self.momentum * self.mean + (1.0 - self.momentum) * x
       return self.mean

现在，运行均值的当前值被视为模块 ``state_dict`` 的一部分，并在从磁盘加载模块时将正确恢复：

.. code-block:: python

   m = RunningMean(4)
   for _ in range(10):
     input = torch.randn(4)
     m(input)

   print(m.state_dict())
   : OrderedDict([('mean', tensor([ 0.1041, -0.1113, -0.0647,  0.1515]))]))

   # 序列化形式将包含 'mean' 张量
   torch.save(m.state_dict(), 'mean.pt')

   m_loaded = RunningMean(4)
   m_loaded.load_state_dict(torch.load('mean.pt'))
   assert(torch.all(m.mean == m_loaded.mean))

如前所述，可以通过将缓冲区标记为非持久化，使其不包含在模块的 ``state_dict`` 中：

.. code-block:: python

   self.register_buffer('unserialized_thing', torch.randn(5), persistent=False)

持久化和非持久化缓冲区都会受到通过 :func:`~torch.nn.Module.to` 应用的模型级设备/数据类型更改的影响：

.. code-block:: python

   # 将所有模块参数和缓冲区移动到指定的设备/数据类型
   m.to(device='cuda', dtype=torch.float64)

可以使用 :func:`~torch.nn.Module.buffers` 或 :func:`~torch.nn.Module.named_buffers` 迭代模块的缓冲区。

.. code-block:: python

   for buffer in m.named_buffers():
     print(buffer)

以下类演示了在模块中注册参数和缓冲区的各种方式：

.. code-block:: python

   class StatefulModule(nn.Module):
     def __init__(self):
       super().__init__()
       # 将 nn.Parameter 设置为模块的属性会自动将该张量注册为模块的参数。
       self.param1 = nn.Parameter(torch.randn(2))

       # 基于字符串的替代方法来注册参数。
       self.register_parameter('param2', nn.Parameter(torch.randn(3)))

       # 保留 "param3" 属性作为参数，防止其被设置为参数以外的任何值。
       # 像这样的 "None" 条目不会出现在模块的 state_dict 中。
       self.register_parameter('param3', None)

       # 注册参数列表。
       self.param_list = nn.ParameterList([nn.Parameter(torch.randn(2)) for i in range(3)])

       # 注册参数字典。
       self.param_dict = nn.ParameterDict({
         'foo': nn.Parameter(torch.randn(3)),
         'bar': nn.Parameter(torch.randn(4))
       })

       # 注册持久化缓冲区（会出现在模块的 state_dict 中）。
       self.register_buffer('buffer1', torch.randn(4), persistent=True)

       # 注册非持久化缓冲区（不会出现在模块的 state_dict 中）。
       self.register_buffer('buffer2', torch.randn(5), persistent=False)

       # 保留 "buffer3" 属性作为缓冲区，防止其被设置为缓冲区以外的任何值。
       # 像这样的 "None" 条目不会出现在模块的 state_dict 中。
       self.register_buffer('buffer3', None)

       # 添加子模块会将其参数注册为模块的参数。
       self.linear = nn.Linear(2, 3)

   m = StatefulModule()

   # 保存和加载 state_dict。
   torch.save(m.state_dict(), 'state.pt')
   m_loaded = StatefulModule()
   m_loaded.load_state_dict(torch.load('state.pt'))

   # 注意，非持久化缓冲区 "buffer2" 以及保留的属性 "param3" 和 "buffer3" 不会出现在 state_dict 中。
   print(m_loaded.state_dict())
   : OrderedDict([('param1', tensor([-0.0322,  0.9066])),
                  ('param2', tensor([-0.4472,  0.1409,  0.4852])),
                  ('buffer1', tensor([ 0.6949, -0.1944,  1.2911, -2.1044])),
                  ('param_list.0', tensor([ 0.4202, -0.1953])),
                  ('param_list.1', tensor([ 1.5299, -0.8747])),
                  ('param_list.2', tensor([-1.6289,  1.4898])),
                  ('param_dict.bar', tensor([-0.6434,  1.5187,  0.0346, -0.4077])),
                  ('param_dict.foo', tensor([-0.0845, -1.4324,  0.7022])),
                  ('linear.weight', tensor([[-0.3915, -0.6176],
                                            [ 0.6062, -0.5992],
                                            [ 0.4452, -0.2843]])),
                  ('linear.bias', tensor([-0.3710, -0.0795, -0.3947]))])

更多信息，请查看：

* 保存和加载模型：https://pytorch.org/tutorials/beginner/saving_loading_models.html
* 序列化语义：https://pytorch.org/docs/main/notes/serialization.html
* 什么是 state dict？https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html

模块初始化
---------------------

默认情况下，:mod:`torch.nn` 提供的模块的参数和浮点缓冲区会在模块实例化期间，使用历史上对该模块类型表现良好的初始化方案，在 CPU 上初始化为 32 位浮点值。对于某些用例，可能希望使用不同的数据类型、设备（例如 GPU）或初始化技术进行初始化。

示例：

.. code-block:: python

   # 直接在 GPU 上初始化模块。
   m = nn.Linear(5, 3, device='cuda')

   # 使用 16 位浮点参数初始化模块。
   m = nn.Linear(5, 3, dtype=torch.half)

   # 跳过默认参数初始化并执行自定义（例如正交）初始化。
   m = torch.nn.utils.skip_init(nn.Linear, 5, 3)
   nn.init.orthogonal_(m.weight)

请注意，上面演示的设备（device）和数据类型（dtype）选项也适用于为模块注册的任何浮点缓冲区：

.. code-block:: python

   m = nn.BatchNorm2d(3, dtype=torch.half)
   print(m.running_mean)
   : tensor([0., 0., 0.], dtype=torch.float16)

虽然模块编写者可以在其自定义模块中使用任何设备或数据类型来初始化参数，但良好的做法是默认也使用 ``dtype=torch.float`` 和 ``device='cpu'``。或者，您可以通过遵循上述所有 :mod:`torch.nn` 模块都遵循的约定，为您的自定义模块在这些方面提供完全的灵活性：

* 提供一个 ``device`` 构造函数关键字参数，该参数适用于模块注册的任何参数/缓冲区。
* 提供一个 ``dtype`` 构造函数关键字参数，该参数适用于模块注册的任何参数/浮点缓冲区。
* 仅在模块构造函数内对参数和缓冲区使用初始化函数（即来自 :mod:`torch.nn.init` 的函数）。请注意，这仅在使用 :func:`~torch.nn.utils.skip_init` 时需要；有关解释，请参阅 `此页面 <https://pytorch.org/tutorials/prototype/skip_param_init.html#updating-modules-to-support-skipping-initialization>`_。

更多信息，请查看：

* 跳过模块参数初始化：https://pytorch.org/tutorials/prototype/skip_param_init.html

模块钩子
------------

在 :ref:`使用模块进行神经网络训练` 中，我们演示了模块的训练过程，该过程迭代执行前向和后向传播，每次迭代更新模块参数。为了对此过程进行更多控制，PyTorch 提供了"钩子"，可以在前向或后向传播期间执行任意计算，甚至可以根据需要修改传播的执行方式。此功能的一些有用示例包括调试、可视化激活、深入检查梯度等。钩子可以添加到您自己未编写的模块中，这意味着此功能可以应用于第三方或 PyTorch 提供的模块。

PyTorch 为模块提供两种类型的钩子：

* **前向钩子** 在前向传播期间调用。可以使用 :func:`~torch.nn.Module.register_forward_pre_hook` 和 :func:`~torch.nn.Module.register_forward_hook` 为给定模块安装这些钩子。这些钩子将分别在调用前向函数之前和之后立即调用。或者，可以使用类似的 :func:`~torch.nn.modules.module.register_module_forward_pre_hook` 和 :func:`~torch.nn.modules.module.register_module_forward_hook` 函数为所有模块全局安装这些钩子。
* **后向钩子** 在后向传播期间调用。可以使用 :func:`~torch.nn.Module.register_full_backward_pre_hook` 和 :func:`~torch.nn.Module.register_full_backward_hook` 安装这些钩子。这些钩子将在计算完此模块的后向传播时调用。:func:`~torch.nn.Module.register_full_backward_pre_hook` 允许用户访问输出的梯度，而 :func:`~torch.nn.Module.register_full_backward_hook` 允许用户访问输入和输出的梯度。或者，可以使用 :func:`~torch.nn.modules.module.register_module_full_backward_hook` 和 :func:`~torch.nn.modules.module.register_module_full_backward_pre_hook` 为所有模块全局安装它们。

所有钩子都允许用户返回一个更新后的值，该值将在剩余的计算中使用。因此，这些钩子可用于在常规模块前向/后向传播过程中执行任意代码，或者修改某些输入/输出，而无需更改模块的 ``forward()`` 函数。

以下是一个演示前向和后向钩子用法的示例：

.. code-block:: python

   torch.manual_seed(1)

   def forward_pre_hook(m, inputs):
     # 允许在前向传播之前检查和修改输入。
     # 注意输入总是包装在元组中。
     input = inputs[0]
     return input + 1.

   def forward_hook(m, inputs, output):
     # 允许在前向传播之后检查输入/输出并修改输出。
     # 注意输入总是包装在元组中，而输出则按原样传递。

     # 类似 ResNet 的残差计算。
     return output + inputs[0]

   def backward_hook(m, grad_inputs, grad_outputs):
     # 允许检查 grad_inputs / grad_outputs 并修改在剩余后向传播中使用的 grad_inputs。
     # 注意 grad_inputs 和 grad_outputs 总是包装在元组中。
     new_grad_inputs = [torch.ones_like(gi) * 42. for gi in grad_inputs]
     return new_grad_inputs

   # 创建示例模块和输入。
   m = nn.Linear(3, 3)
   x = torch.randn(2, 3, requires_grad=True)

   # ==== 演示前向钩子。 ====
   # 在添加钩子前后通过模块运行输入。
   print('output with no forward hooks: {}'.format(m(x)))
   : output with no forward hooks: tensor([[-0.5059, -0.8158,  0.2390],
                                           [-0.0043,  0.4724, -0.1714]], grad_fn=<AddmmBackward>)

   # 注意修改后的输入导致不同的输出。
   forward_pre_hook_handle = m.register_forward_pre_hook(forward_pre_hook)
   print('output with forward pre hook: {}'.format(m(x)))
   : output with forward pre hook: tensor([[-0.5752, -0.7421,  0.4942],
                                           [-0.0736,  0.5461,  0.0838]], grad_fn=<AddmmBackward>)

   # 注意修改后的输出。
   forward_hook_handle = m.register_forward_hook(forward_hook)
   print('output with both forward hooks: {}'.format(m(x)))
   : output with both forward hooks: tensor([[-1.0980,  0.6396,  0.4666],
                                             [ 0.3634,  0.6538,  1.0256]], grad_fn=<AddBackward0>)

   # 移除钩子；注意此处的输出与添加钩子前的输出匹配。
   forward_pre_hook_handle.remove()
   forward_hook_handle.remove()
   print('output after removing forward hooks: {}'.format(m(x)))
   : output after removing forward hooks: tensor([[-0.5059, -0.8158,  0.2390],
                                                  [-0.0043,  0.4724, -0.1714]], grad_fn=<AddmmBackward>)

   # ==== 演示后向钩子。 ====
   m(x).sum().backward()
   print('x.grad with no backwards hook: {}'.format(x.grad))
   : x.grad with no backwards hook: tensor([[ 0.4497, -0.5046,  0.3146],
                                            [ 0.4497, -0.5046,  0.3146]])

   # 在再次运行后向传播之前清除梯度。
   m.zero_grad()
   x.grad.zero_()

m.register_full_backward_hook(backward_hook)
m(x).sum().backward()
print('x.grad with backwards hook: {}'.format(x.grad))
: x.grad with backwards hook: tensor([[42., 42., 42.],
                                      [42., 42., 42.]])

高级功能
-----------------

PyTorch 还提供了更多旨在与模块协同工作的高级功能。所有这些功能都可用于自定义编写的模块，但需要注意某些功能可能要求模块符合特定约束才能获得支持。关于这些功能及其相应要求的深入讨论，请参阅以下链接。

分布式训练
********************

PyTorch 中存在多种分布式训练方法，既可用于使用多个 GPU 扩展训练规模，也可用于跨多台机器进行训练。请查阅
`分布式训练概述页面 <https://pytorch.org/tutorials/beginner/dist_overview.html>`_ 以获取关于如何利用这些方法的详细信息。

性能分析
*********************

`PyTorch Profiler <https://pytorch.org/tutorials/beginner/profiler.html>`_ 可用于识别模型内部的性能瓶颈。它会测量并输出内存使用情况和时间消耗的性能特征。

通过量化提升性能
***************************************

对模块应用量化技术可以通过使用比浮点精度更低的位宽来提升性能和内存使用效率。请查阅 PyTorch 提供的各种量化机制
`此处 <https://pytorch.org/docs/stable/quantization.html>`_。

通过剪枝改善内存使用
***********************************

大型深度学习模型通常参数过多，导致内存使用量高。为了解决这个问题，PyTorch 提供了模型剪枝机制，这有助于在保持任务准确性的同时减少内存使用。
`剪枝教程 <https://pytorch.org/tutorials/intermediate/pruning_tutorial.html>`_ 描述了如何利用 PyTorch 提供的剪枝技术或根据需要定义自定义剪枝技术。

参数化
****************

对于某些应用，在模型训练期间约束参数空间可能是有益的。例如，强制学习参数的正交性可以改善 RNN 的收敛性。PyTorch 提供了一种应用此类
`参数化 <https://pytorch.org/tutorials/intermediate/parametrizations.html>`_ 的机制，并进一步允许定义自定义约束。

使用 FX 转换模块
****************************

PyTorch 的 `FX <https://pytorch.org/docs/stable/fx.html>`_ 组件提供了一种灵活的方式来转换模块，即直接对模块计算图进行操作。这可用于以编程方式为广泛的用例生成或操作模块。要探索 FX，请查看这些使用 FX 进行
`卷积 + 批归一化融合 <https://pytorch.org/tutorials/intermediate/fx_conv_bn_fuser.html>`_ 和
`CPU 性能分析 <https://pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html>`_ 的示例。