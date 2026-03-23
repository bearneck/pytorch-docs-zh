# torch.distributed.fsdp.fully_shard

## PyTorch FSDP2 (`fully_shard`)

PyTorch FSDP2 ([RFC](<https://github.com/pytorch/pytorch/issues/114299>)) 提供了一种完全分片数据并行 (FSDP) 的实现，旨在实现高性能的即时执行模式，同时使用按参数分片以提高可用性。

- 更多信息请参阅 [FSDP2 入门教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)。

- 如果您当前正在使用 FSDP1，请考虑使用我们的 [迁移指南](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#fsdp1-to-fsdp2-migration-guide) 迁移到 FSDP2。

`fully_shard(model)` 的用户约定如下：

- 对于模型初始化，`fully_shard` 将 `model.parameters()` 从普通的 `torch.Tensor` 原地转换为 `DTensor`。参数会根据设备网格移动到适当的设备上。

- 在前向和后向传播之前，前向/后向钩子负责对所有参数进行全收集，并将 `model.parameters()` 从 `DTensor` 转换为普通的 `torch.Tensor`。

- 在前向和后向传播之后，后前向/后向钩子会释放未分片的参数（无需通信），并将 `model.parameters()` 从普通的 `torch.Tensor` 转换回 `DTensor`。

- 对于优化器，必须使用 `DTensor` 类型的 `model.parameters()` 进行初始化，并且优化器步骤应在 `DTensor` 参数上执行。

- 调用 `model(input)` 而不是 `model.forward(input)` 来触发前向钩子以全收集参数。为了使 `model.forward(input)` 正常工作，用户必须显式调用 `model.unshard()` 或使用 `register_fsdp_forward_method(model, "forward")` 来注册需要挂钩的前向方法。

- `fully_shard` 将参数分组以进行单次全收集。用户应以自底向上的方式应用 `fully_shard`。例如，在 Transformer 模型中，应在将 `fully_shard` 应用于根模型之前，先将其应用于每个层。当应用于根模型时，`fully_shard` 会排除每个层的 `model.parameters()`，并将剩余的参数（例如，嵌入层、输出投影层）分组到一个全收集组中。

- `type(model)` 会原地与 `FSDPModule` "合并"。例如，如果 `model` 原本是 `nn.Linear` 类型，那么 `fully_shard` 会将 `type(model)` 从 `nn.Linear` 原地更改为 `FSDPLinear`。`FSDPLinear` 既是 `nn.Linear` 的实例，也是 `FSDPModule` 的实例。它保留了 `nn.Linear` 的所有方法，同时在 `FSDPModule` 下暴露了 FSDP2 特有的 API，例如 `reshard()` 和 `unshard()`。

- 参数的完全限定名称 (FQN) 保持不变。如果我们调用 `model.state_dict()`，应用 `fully_shard` 前后的 FQN 是相同的。这是因为 `fully_shard` 不会包装模块，而只是向原始模块注册钩子。

### 通信分组与调度

每次调用 `fully_shard` 都会创建一个**通信组**，其中包含模块中所有尚未通过先前在子模块上的调用分配到组中的参数。每个组的参数在前向传播之前通过一次集合操作进行全收集，并在后向传播之后通过一次集合操作进行归约分散。与 DDP 不同，FSDP2 没有 `bucket_cap_mb` 参数——通信边界完全由您对哪些模块应用 `fully_shard` 决定。

考虑一个包含四个子模块的模型，其中 `a`、`b`、`c` 和 `d` 表示每个子模块中的参数数量：

```
model[ m1[a] -> m2[b] -> m3[c] -> m4[d] ]
```

**如果只调用** `fully_shard(model)` **（仅根模块）**，所有参数都在一个组中。这意味着整个前向和后向传播看起来像：

```
all-gather(a+b+c+d) -> forward(m1,m2,m3,m4) -> backward(m4,m3,m2,m1) -> reduce-scatter(a+b+c+d)
```

所有通信都作为两个大型阻塞操作进行，与计算没有重叠。这几乎从来都不是您想要的。

**如果对每个子模块应用** `fully_shard` —— 例如，调用 `fully_shard(m2)`、`fully_shard(m3)`，然后调用 `fully_shard(model)` —— 剩余的参数（`a` 和 `d`）形成根组，而 `m2` 和 `m3` 各自拥有自己的组。

在**前向传播**中，全收集操作在单独的 CUDA 流上运行，因此下一个模块的全收集可以与当前模块的前向计算重叠。每个模块的前向钩子会发起自己的全收集操作，并在运行模块之前等待其完成。由于 CPU 通常领先于 GPU，下一个模块的全收集会在 AG 流上发起，而当前模块的前向传播仍在计算流上执行：

```
              时间 ──────────────────────────────────────────────►

计算流:      [等待] [ fwd(m1)   | fwd(m2)    | fwd(m3,m4)     ]
AG 流:    [AG(a,d)]  [AG(b)  |    AG(c)   ]
```

当 `fwd(m1)` 在计算流上运行时，CPU 触发 `m2` 的前向钩子，该钩子在 AG 流上发起 `AG(b)`。为了使这种重叠更稳健（例如，当 CPU 端开销减少了领先时间时），可以使用 `set_modules_to_forward_prefetch` 来更早地发起下一个全收集——在当前模块的前向钩子内部，而不是等待下一个模块的钩子触发。

在**后向传播**中，FSDP2 还会显式预取下一个模块的全收集，并在单独的 CUDA 流上运行归约分散操作，所有这些都无需任何额外配置：

```
              时间 ──────────────────────────────────────────────►

计算流:      [ bwd(m4,m3)     | bwd(m2)        | bwd(m1)       ]
AG 流:    [AG(c)] [ AG(b)  |   AG(a,d)      ]
RS 流:                     |[RS(c)]  [ RS(b)|     RS(a,d)   ]
```

当 `bwd(m4,m3)` 在计算流上运行时，`b` 的全收集（`m2` 所需）会在 AG 流上预取。当 `bwd(m2)` 运行时，`AG(a,d)` 和 `RS(c)` 都与计算重叠。这种流水线操作就是为什么推荐模式是在将 `fully_shard` 应用于根模块之前，自底向上地将其应用于每个层。

要控制每个通信组的大小，请选择要包装的模块：
包装更细粒度的模块会产生更小、可重叠性更高的组
（类似于较小的 DDP 桶），而包装较少的模块会产生更大的组。
这里没有自动分桶机制——分组是显式的，由模块结构决定。

与 PyTorch FSDP1 (`FullyShardedDataParallel`) 相比：

- FSDP2 使用基于 `DTensor` 的按参数 dim-0 分片，相比 FSDP1 的扁平参数分片，提供了更简单的分片表示，同时保持了相似的吞吐性能。具体来说，FSDP2 将每个参数在 dim-0 上跨数据并行工作进程进行分块（使用 `torch.chunk(dim=0)`），而 FSDP1 则将一组张量展平、拼接并一起分块，这使得理解每个工作进程上存在哪些数据以及向不同并行模式重新分片变得复杂。按参数分片提供了更直观的用户体验，放宽了对冻结参数的限制，并允许实现无通信（分片）的状态字典，而这在 FSDP1 中需要全收集操作。
- FSDP2 实现了一种不同的内存管理方法来处理多流使用，避免了 `torch.Tensor.record_stream`。这确保了确定性和预期的内存使用，并且不需要像 FSDP1 的 `limit_all_gathers=True` 那样阻塞 CPU。
- FSDP2 提供了用于手动控制预取和集体调度的 API，允许高级用户进行更多自定义。详情请参见下文 `FSDPModule` 的方法。
- FSDP2 简化了部分 API 接口：例如，FSDP2 不直接支持完整状态字典。相反，用户可以使用 `DTensor` API（如 `DTensor.full_tensor()`）或使用更高级的 API（如 [PyTorch 分布式检查点](https://pytorch.org/docs/stable/distributed.checkpoint.html) 的分布式状态字典 API）自行将包含 `DTensor` 的分片状态字典重新分片为完整状态字典。此外，一些其他参数已被移除；详情请参见[此处](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)。


前端 API 是 `fully_shard`，可以在 `module` 上调用：


