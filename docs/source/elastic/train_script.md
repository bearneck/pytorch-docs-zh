# 训练脚本

如果你的训练脚本能与 `torch.distributed.launch` 配合工作，那么它也能继续与 `torchrun` 配合工作，但存在以下差异：

1.  无需手动传递 `RANK`、`WORLD_SIZE`、`MASTER_ADDR` 和 `MASTER_PORT`。
2.  可以提供 `rdzv_backend` 和 `rdzv_endpoint`。对于大多数用户，这将设置为 `c10d`（参见 [集合点](rendezvous.html)）。默认的 `rdzv_backend` 创建一个非弹性的集合点，其中 `rdzv_endpoint` 保存主节点地址。
3.  确保你的脚本中包含 `load_checkpoint(path)` 和 `save_checkpoint(path)` 的逻辑。当任意数量的工作进程失败时，我们会使用相同的程序参数重新启动所有工作进程，因此你将丢失最近一次检查点之后的所有进度（参见 [弹性启动](run.html)）。
4.  `use_env` 标志已被移除。如果你之前通过解析 `--local-rank` 选项来获取本地排名，现在需要从环境变量 `LOCAL_RANK` 中获取本地排名（例如 `int(os.environ["LOCAL_RANK"])`）。

下面是一个说明性的训练脚本示例，它在每个训练周期（epoch）都进行模型检查点保存，因此在最坏情况下，失败时最多丢失一个完整周期的训练进度。

``` python
def main():
     args = parse_args(sys.argv[1:])
     state = load_checkpoint(args.checkpoint_path)
     initialize(state)

     # torch.distributed.run 通过导出初始化进程组所需的所有环境变量来确保此操作有效
     torch.distributed.init_process_group(backend=args.backend)

     for i in range(state.epoch, state.total_num_epochs)
          for batch in iter(state.dataset)
              train(batch, state.model)

          state.epoch += 1
          save_checkpoint(state)
```

要查看符合 torchelastic 规范的训练脚本的具体示例，请访问我们的 [示例](examples.html) 页面。
