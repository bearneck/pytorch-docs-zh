# 常见问题

## 我的模型报告 \"cuda runtime error(2): out of memory\"

正如错误信息所示，您的 GPU 内存已耗尽。由于在 PyTorch 中我们经常处理大量数据，小的错误可能迅速导致程序耗尽所有 GPU 内存；幸运的是，这些情况的修复通常很简单。以下是一些需要检查的常见事项：

**不要在训练循环中累积历史记录。** 默认情况下，涉及需要梯度的变量的计算会保留历史记录。这意味着您应避免在训练循环之外的计算中使用此类变量，例如在跟踪统计信息时。相反，您应该分离变量或访问其底层数据。

有时，可微分变量的出现可能并不明显。考虑以下训练循环（摘自 [来源](https://discuss.pytorch.org/t/high-memory-usage-while-training/162)）：

``` python
total_loss = 0
for i in range(10000):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output)
    loss.backward()
    optimizer.step()
    total_loss += loss
```

这里，\`total_loss\` 在训练循环中累积历史记录，因为 [loss] 是一个具有 autograd 历史记录的可微分变量。您可以通过改为写入 [total_loss += float(loss)] 来修复此问题。

此问题的其他实例： [1](https://discuss.pytorch.org/t/resolved-gpu-out-of-memory-error-with-batch-size-1/3719)。

**不要保留不需要的张量和变量。** 如果您将 Tensor 或 Variable 分配给局部变量，Python 将不会释放它，直到该局部变量超出作用域。您可以通过使用 [del x] 来释放此引用。类似地，如果您将 Tensor 或 Variable 分配给对象的成员变量，它将不会释放，直到该对象超出作用域。如果您不保留不需要的临时变量，您将获得最佳的内存使用情况。

局部变量的作用域可能比您预期的要大。例如：

``` python
for i in range(5):
    intermediate = f(input[i])
    result += g(intermediate)
output = h(result)
return output
```

这里，\`intermediate\` 即使在 [h] 执行时仍然存活，因为它的作用域延伸到了循环结束之后。要更早地释放它，您应该在完成后使用 [del intermediate]。

**避免在过长的序列上运行 RNN。** 通过 RNN 进行反向传播所需的内存量与 RNN 输入的长度成线性比例；因此，如果您尝试向 RNN 输入过长的序列，您将耗尽内存。

这种现象的技术术语是 [随时间反向传播](https://en.wikipedia.org/wiki/Backpropagation_through_time)，并且有许多关于如何实现截断 BPTT 的参考资料，包括在 [word language model](https://github.com/pytorch/examples/tree/master/word_language_model) 示例中；截断由 [repackage] 函数处理，如 [此论坛帖子](https://discuss.pytorch.org/t/help-clarifying-repackage-hidden-in-word-language-model/226) 中所述。

**不要使用过大的线性层。** 线性层 [nn.Linear(m, n)] 使用 $O(nm)$ 内存：也就是说，权重的内存需求与特征数量成二次方比例。这种方式很容易 [耗尽内存](https://github.com/pytorch/pytorch/issues/958)（并且记住您至少需要两倍于权重大小的内存，因为您还需要存储梯度。）

**考虑使用检查点。** 您可以通过使用 [checkpoint](https://pytorch.org/docs/stable/checkpoint.html) 来以计算换取内存。

## 我的 GPU 内存未正确释放

PyTorch 使用缓存内存分配器来加速内存分配。因此，\`nvidia-smi\` 中显示的值通常不反映真实的内存使用情况。有关 GPU 内存管理的更多详细信息，请参阅 `cuda-memory-management`。

如果即使在 Python 退出后您的 GPU 内存仍未释放，很可能是一些 Python 子进程仍然存活。您可以通过 [ps -elf \| grep python] 找到它们，并使用 [kill -9 \[pid\]] 手动终止它们。

## 我的内存不足异常处理程序无法分配内存

您可能有一些代码试图从内存不足错误中恢复。

``` python
try:
    run_model(batch_size)
except RuntimeError: # Out of memory
    for _ in range(batch_size):
        run_model(1)
```

但发现当您确实耗尽内存时，您的恢复代码也无法分配内存。这是因为 python 异常对象持有对引发错误的堆栈帧的引用。这阻止了原始张量对象的释放。解决方案是将您的 OOM 恢复代码移到 [except] 子句之外。

``` python
oom = False
try:
    run_model(batch_size)
except RuntimeError: # Out of memory
    oom = True

if oom:
    for _ in range(batch_size):
        run_model(1)
```

## 我的数据加载器工作进程返回相同的随机数

您可能正在使用其他库在数据集中生成随机数，并且工作子进程是通过 [fork] 启动的。有关如何使用其 `worker_init_fn` 选项在工作进程中正确设置随机种子，请参阅 `torch.utils.data.DataLoader` 的文档。

## 我的循环网络在使用数据并行时无法工作

在使用 `torch.nn.DataParallel` 或 `torch.nn.parallel.data_parallel` 时，在 `torch.nn.Module` 中采用 `打包序列 -> 循环网络 -> 解包序列` 模式存在一个细微之处。每个设备上 `forward` 方法的输入将只是整个输入的一部分。由于解包操作 `torch.nn.utils.rnn.pad_packed_sequence` 默认只填充到它看到的最长输入，即该特定设备上的最长序列，因此在结果被收集到一起时会发生尺寸不匹配。因此，你可以转而利用 `torch.nn.utils.rnn.pad_packed_sequence` 的 `total_length` 参数来确保 `forward` 调用返回相同长度的序列。例如，你可以这样写:

    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    class MyModule(nn.Module):
        # ... __init__, 其他方法等

        # padded_input 的形状为 [B x T x *] (batch_first 模式) 并且包含
        # 按长度排序的序列
        #   B 是批次大小
        #   T 是最大序列长度
        def forward(self, padded_input, input_lengths):
            total_length = padded_input.size(1)  # 获取最大序列长度
            packed_input = pack_padded_sequence(padded_input, input_lengths,
                                                batch_first=True)
            packed_output, _ = self.my_lstm(packed_input)
            output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                            total_length=total_length)
            return output


    m = MyModule().cuda()
    dp_m = nn.DataParallel(m)

此外，当批次维度是维度 `1`（即 `batch_first=False`）并使用数据并行时，需要格外小心。在这种情况下，pack_padded_sequence 的第一个参数 `padding_input` 的形状将是 `[T x B x *]`，并且应该沿着维度 `1` 进行分散，但第二个参数 `input_lengths` 的形状将是 `[B]`，并且应该沿着维度 `0` 进行分散。将需要额外的代码来操作张量形状。
