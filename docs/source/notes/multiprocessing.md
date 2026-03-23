# 多进程最佳实践

`torch.multiprocessing` 是 Python `python:multiprocessing` 模块的直接替代品。它支持完全相同的操作，但进行了扩展，使得所有通过 `python:multiprocessing.Queue` 发送的张量，都会将其数据移动到共享内存中，并且只发送一个句柄给另一个进程。


> 📝 **注意**
> 当一个 `torch.Tensor` 被发送到另一个进程时，该 `torch.Tensor` 的数据是共享的。如果 `torch.Tensor.grad` 不是 `None`，它也会被共享。当一个没有 `torch.Tensor.grad` 字段的 `torch.Tensor` 被发送到另一个进程后，它会创建一个进程特定的标准 `.grad` `torch.Tensor`，该张量不会像 `torch.Tensor` 的数据那样在所有进程间自动共享。
>
> 这使得可以实现各种训练方法，如 Hogwild、A3C 或任何其他需要异步操作的方法。
>
> ## 多进程中的毒化 fork
>
> 当将多进程与 `加速器<accelerators>` 一起使用时，可能会出现一个称为"毒化 fork"的已知问题。当加速器的运行时不是 fork 安全的，并且在进程 fork 之前被初始化时，就会发生这种情况，导致子进程中出现运行时错误。
>
> 为防止此类错误：
>
> :   - 避免在 fork 子进程之前在主进程中初始化加速器。
>     - 使用替代的进程启动方法，例如 `spawn` 或 `forkserver`，这可以确保每个进程的干净初始化。
>
> ## 多进程中的 CUDA
>
> 当使用 `fork` 启动方法时，CUDA 运行时存在 `multiprocessing-poison-fork-note` 中描述的限制；要在子进程中使用 CUDA，需要使用 `spawn` 或 `forkserver` 启动方法。


> 📝 **注意**
> 启动方法可以通过使用 `multiprocessing.get_context(...)` 创建上下文来设置，或者直接使用 `multiprocessing.set_start_method(...)`。
>
> 与 CPU 张量不同，只要接收进程保留张量的副本，发送进程就需要保留原始张量。这是在底层实现的，但要求用户遵循最佳实践以确保程序正确运行。例如，只要消费者进程持有对张量的引用，发送进程就必须保持存活，并且如果消费者进程通过致命信号异常退出，引用计数也无法挽救。请参阅 `此部分 <multiprocessing-cuda-sharing-details>`。
>
> 另请参阅：`cuda-nn-ddp-instead`
>
> ## 最佳实践和技巧
>
> ### 避免和应对死锁
>
> 当生成新进程时，很多事情都可能出错，最常见的死锁原因是后台线程。如果有任何线程持有锁或导入模块，并且调用了 `fork`，那么子进程很可能处于损坏状态，并会以不同的方式死锁或失败。请注意，即使你没有这样做，Python 内置库也会这样做------无需看 `python:multiprocessing` 之外的东西。`python:multiprocessing.Queue` 实际上是一个非常复杂的类，它会生成多个用于序列化、发送和接收对象的线程，这些线程也可能导致上述问题。如果你发现自己处于这种情况，请尝试使用 `python:multiprocessing.queues.SimpleQueue`，它不使用任何额外的线程。
>
> 我们正在尽最大努力使其变得简单，并确保这些死锁不会发生，但有些事情是我们无法控制的。如果你有任何暂时无法解决的问题，请尝试在论坛上寻求帮助，我们会看看这是否是我们能够修复的问题。
>
> ### 重用通过队列传递的缓冲区
>
> 请记住，每次你将一个 `torch.Tensor` 放入 `python:multiprocessing.Queue` 时，它都必须被移动到共享内存中。如果它已经共享，则无需操作，否则将产生额外的内存拷贝，从而减慢整个进程。即使你有一个进程池向单个进程发送数据，也要让它将缓冲区发送回来------这几乎是免费的，并且可以让你在发送下一批数据时避免拷贝。
>
> ### 异步多进程训练（例如 Hogwild）
>
> 使用 `torch.multiprocessing`，可以异步训练模型，参数可以始终共享，也可以定期同步。在前一种情况下，我们建议发送整个模型对象，而在后一种情况下，我们建议仅发送 `torch.nn.Module.state_dict`。
>
> 我们建议使用 `python:multiprocessing.Queue` 在进程之间传递各种 PyTorch 对象。例如，当使用 `fork` 启动方法时，可以继承已经在共享内存中的张量和存储，但这非常容易出错，应谨慎使用，并且仅由高级用户使用。队列，尽管有时是一种不太优雅的解决方案，但在所有情况下都能正常工作。


> ⚠️ **警告**
> 你应该注意那些没有用 `if __name__ == '__main__'` 保护的全局语句。如果使用了 `fork` 以外的启动方法，它们将在所有子进程中执行。
>
> #### Hogwild
>
> 可以在 [examples repository](https://github.com/pytorch/examples/tree/master/mnist_hogwild) 中找到具体的 Hogwild 实现，但为了展示代码的整体结构，下面也有一个最小示例：:
>
> > import torch.multiprocessing as mp from model import MyModel
> >
> > def train(model):
> >
> > :   \# 构造 data_loader, optimizer 等 for data, labels in data_loader: optimizer.zero_grad() loss_fn(model(data), labels).backward() optimizer.step() \# 这将更新共享参数
> >
> > if \_\_name\_\_ == \'\_\_main\_\_\':
> >
> > :   num_processes = 4 model = MyModel() \# 注意：这是 `fork` 方法正常工作所必需的 model.share_memory() processes = \[\] for rank in range(num_processes): p = mp.Process(target=train, args=(model,)) p.start() processes.append(p) for p in processes: p.join()
>
> ## 多进程中的 CPU
>
> 不恰当的多进程处理可能导致 CPU 过载，使得不同进程竞争 CPU 资源，从而导致效率低下。
>
> 本教程将解释什么是 CPU 过载以及如何避免它。
>
> ### CPU 过载
>
> CPU 过载是一个技术术语，指的是分配给系统的虚拟 CPU 总数超过了硬件上可用的虚拟 CPU 总数的情况。
>
> 这会导致对 CPU 资源的激烈竞争。在这种情况下，进程之间会频繁切换，从而增加了进程切换的开销并降低了整体系统效率。
>
> 关于 CPU 过载的示例，请参见 [示例仓库](https://github.com/pytorch/examples/tree/main/mnist_hogwild) 中 Hogwild 实现里的代码示例。
>
> 当在 CPU 上使用 4 个进程运行以下命令执行训练示例时：
>
> ``` bash
> python main.py --num-processes 4
> ```
>
> 假设机器上有 N 个可用的虚拟 CPU，执行上述命令将生成 4 个子进程。每个子进程将为自己分配 N 个虚拟 CPU，总共需要 4\*N 个虚拟 CPU。然而，机器上只有 N 个可用的虚拟 CPU。因此，不同的进程将竞争资源，导致频繁的进程切换。
>
> 以下现象表明存在 CPU 过载：
>
> 1.  高 CPU 利用率：使用 `htop` 命令，可以观察到 CPU 利用率持续处于高位，经常达到或超过其最大容量。这表明对 CPU 资源的需求超过了可用的物理核心，导致进程之间为 CPU 时间而竞争和争夺。
> 2.  频繁的上下文切换与低系统效率：在 CPU 过载的场景下，进程竞争 CPU 时间，操作系统需要快速在不同进程之间切换以公平地分配资源。这种频繁的上下文切换增加了开销并降低了整体系统效率。
>
> ### 避免 CPU 过载
>
> 避免 CPU 过载的一个好方法是进行适当的资源分配。确保并发运行的进程或线程数量不超过可用的 CPU 资源。
>
> 在这种情况下，一个解决方案是在子进程中指定适当数量的线程。这可以通过在子进程中使用 `torch.set_num_threads(int)` 函数为每个进程设置线程数来实现。
>
> 假设机器上有 N 个虚拟 CPU，并且将生成 M 个进程，那么每个进程使用的最大 `num_threads` 值应为 `floor(N/M)`。为了避免 mnist_hogwild 示例中的 CPU 过载，需要对 [示例仓库](https://github.com/pytorch/examples/tree/main/mnist_hogwild) 中的 `train.py` 文件进行以下修改。
>
> ``` python
> def train(rank, args, model, device, dataset, dataloader_kwargs):
>     torch.manual_seed(args.seed + rank)
>
>     #### 定义当前子进程中使用的线程数
>     torch.set_num_threads(floor(N/M))
>
>     train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
>
>     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
>     for epoch in range(1, args.epochs + 1):
>         train_epoch(epoch, args, model, device, train_loader, optimizer)
> ```
>
> 使用 `torch.set_num_threads(floor(N/M))` 为每个进程设置 `num_thread`。其中，你需要将 N 替换为可用的虚拟 CPU 数量，将 M 替换为选择的进程数。合适的 `num_thread` 值会根据具体任务而变化。然而，作为一般准则，为了避免 CPU 过载，`num_thread` 的最大值应为 `floor(N/M)`。在 [mnist_hogwild](https://github.com/pytorch/examples/tree/main/mnist_hogwild) 训练示例中，避免 CPU 过载后，你可以获得高达 30 倍的性能提升。

