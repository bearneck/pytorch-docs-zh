```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# 分布式优化器

:::{warning}
当前在使用 CUDA 张量时，分布式优化器不受支持。
:::

```{eval-rst}
.. automodule:: torch.distributed.optim
    :members: DistributedOptimizer, PostLocalSGDOptimizer, ZeroRedundancyOptimizer
```