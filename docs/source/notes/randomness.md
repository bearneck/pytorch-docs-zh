# 可复现性 {#reproducibility}

在 PyTorch 的不同版本、独立提交或不同平台之间，无法保证完全可复现的结果。此外，即使使用相同的随机种子，CPU 和 GPU 执行之间的结果也可能无法复现。

然而，您可以采取一些步骤来限制特定平台、设备和 PyTorch 版本中非确定性行为的来源。首先，您可以控制可能导致应用程序多次执行行为不同的随机性来源。其次，您可以配置 PyTorch 以避免对某些操作使用非确定性算法，从而确保在给定相同输入的情况下，多次调用这些操作会产生相同的结果。

 warning
 title
Warning


确定性操作通常比非确定性操作更慢，因此您的模型在单次运行中的性能可能会下降。然而，确定性可以通过促进实验、调试和回归测试来节省开发时间。


## 控制随机性来源

### PyTorch 随机数生成器

您可以使用 `torch.manual_seed()`{.interpreted-text role="meth"} 为所有设备（CPU 和 CUDA）设置随机数生成器（RNG）的种子：:

> import torch torch.manual_seed(0)

一些 PyTorch 操作可能在内部使用随机数。例如，`torch.svd_lowrank()`{.interpreted-text role="meth"} 就是如此。因此，使用相同的输入参数连续多次调用它可能会得到不同的结果。但是，只要在应用程序开始时将 `torch.manual_seed()`{.interpreted-text role="meth"} 设置为一个常数，并且消除了所有其他非确定性来源，每次在相同环境中运行应用程序时都会生成相同系列的随机数。

通过在后续调用之间将 `torch.manual_seed()`{.interpreted-text role="meth"} 设置为相同的值，也可以从使用随机数的操作中获得相同的结果。

### Python

对于自定义操作符，您可能还需要设置 Python 的种子：:

> import random random.seed(0)

### 其他库中的随机数生成器

如果您或您使用的任何库依赖于 NumPy，您可以使用以下方式设置全局 NumPy RNG 的种子：:

> import numpy as np np.random.seed(0)

但是，一些应用程序和库可能使用 NumPy 随机生成器对象，而不是全局 RNG (<https://numpy.org/doc/stable/reference/random/generator.html>)，这些也需要一致地设置种子。

如果您使用任何其他使用随机数生成器的库，请参考这些库的文档，了解如何为它们设置一致的种子。

### CUDA 卷积基准测试

CUDA 卷积操作使用的 cuDNN 库可能是导致应用程序多次执行之间非确定性的一个来源。当使用一组新的尺寸参数调用 cuDNN 卷积时，一个可选功能可以运行多个卷积算法，对它们进行基准测试以找到最快的算法。然后，在后续过程中，对于相应的尺寸参数集，将一致地使用最快的算法。由于基准测试的噪声和不同的硬件，即使在相同的机器上，后续运行中基准测试也可能选择不同的算法。

通过设置 `torch.backends.cudnn.benchmark = False` 来禁用基准测试功能会导致 cuDNN 确定性地选择一个算法，但可能会以降低性能为代价。

然而，如果您不需要在应用程序的多次执行之间实现可复现性，那么通过设置 `torch.backends.cudnn.benchmark = True` 启用基准测试功能可能会提高性能。

请注意，此设置与下面讨论的 `torch.backends.cudnn.deterministic` 设置不同。

## 避免非确定性算法

`torch.use_deterministic_algorithms`{.interpreted-text role="meth"} 允许您配置 PyTorch 在可用时使用确定性算法而不是非确定性算法，并且如果已知某个操作是非确定性的（且没有确定性替代方案），则会抛出错误。

请查看 `torch.use_deterministic_algorithms()`{.interpreted-text role="meth"} 的文档以获取受影响操作的完整列表。如果某个操作未按照文档正确运行，或者如果您需要一个没有确定性实现的操作的确定性实现，请提交问题： <https://github.com/pytorch/pytorch/issues?q=label:%22module:%20determinism%22>

例如，运行 `torch.Tensor.index_add_`{.interpreted-text role="meth"} 的非确定性 CUDA 实现将抛出错误：:

> \>\>\> import torch \>\>\> torch.use_deterministic_algorithms(True) \>\>\> torch.randn(2, 2).cuda().index_add\_(0, torch.tensor(\[0, 1\]), torch.randn(2, 2)) Traceback (most recent call last): File \"\<stdin\>\", line 1, in \<module\> RuntimeError: [index_add_cuda]() does not have a deterministic implementation, but you set \'torch.use_deterministic_algorithms(True)\'. \...

当使用稀疏-稠密 CUDA 张量调用 `torch.bmm`{.interpreted-text role="meth"} 时，它通常使用非确定性算法，但当启用确定性标志时，将使用其替代的确定性实现：:

> \>\>\> import torch \>\>\> torch.use_deterministic_algorithms(True) \>\>\> torch.bmm(torch.randn(2, 2, 2).to_sparse().cuda(), torch.randn(2, 2, 2).cuda()) tensor(\[\[\[ 1.1900, -2.3409\], \[ 0.4796, 0.8003\]\], \[\[ 0.1509, 1.8027\], \[ 0.0333, -1.1444\]\]\], device=\'cuda:0\')

### CUDA 卷积确定性

虽然禁用 CUDA 卷积基准测试（如上所述）能确保 CUDA 在每次运行应用程序时选择相同的算法，但该算法本身可能仍是非确定性的，除非设置 `torch.use_deterministic_algorithms(True)` 或 `torch.backends.cudnn.deterministic = True`。后一个设置仅控制此行为，而 `torch.use_deterministic_algorithms`{.interpreted-text role="meth"} 还会使其他 PyTorch 操作也具有确定性。

### CUDA RNN 和 LSTM

在某些 CUDA 版本中，RNN 和 LSTM 网络可能具有非确定性行为。详情及解决方法请参阅 `torch.nn.RNN`{.interpreted-text role="meth"} 和 `torch.nn.LSTM`{.interpreted-text role="meth"}。

### 填充未初始化内存

像 `torch.empty`{.interpreted-text role="meth"} 和 `torch.Tensor.resize_`{.interpreted-text role="meth"} 这样的操作可能返回包含未定义值的未初始化内存张量。如果需要确定性，将此类张量用作另一个操作的输入是无效的，因为输出将是非确定性的。但实际上没有任何机制能阻止运行此类无效代码。因此为了安全起见，默认将 `torch.utils.deterministic.fill_uninitialized_memory`{.interpreted-text role="attr"} 设置为 `True`，如果设置了 `torch.use_deterministic_algorithms(True)`，这将用已知值填充未初始化内存。这将防止此类非确定性行为的发生。

然而，填充未初始化内存会损害性能。因此，如果你的程序是有效的且不使用未初始化内存作为操作的输入，则可以关闭此设置以获得更好的性能。

## DataLoader

DataLoader 将按照 `data-loading-randomness`{.interpreted-text role="ref"} 算法重新设置工作进程种子。使用 `worker_init_fn`{.interpreted-text role="meth"} 和 [generator]{.title-ref} 来保持可重现性:

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
