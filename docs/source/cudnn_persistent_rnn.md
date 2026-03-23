 note
 title
Note


如果满足以下条件： 1) 已启用 cudnn， 2) 输入数据位于 GPU 上 3) 输入数据具有 `torch.float16` 数据类型 4) 使用 V100 GPU， 5) 输入数据不是 `PackedSequence` 格式 则可以选择持久性算法以提高性能。

