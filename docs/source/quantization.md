# 量化


torch.ao.quantization


torch.ao.quantization.fx

我们正在将所有与量化相关的开发集中到 [torchao](https://github.com/pytorch/ao)，请查看我们的新文档页面：https://docs.pytorch.org/ao/stable/index.html

现有量化流程的计划： 1. Eager 模式量化 (torch.ao.quantization.quantize, torch.ao.quantization.quantize_dynamic)，请迁移到使用 torchao eager 模式 [quantize\_](https://docs.pytorch.org/ao/main/generated/torchao.quantization.quantize_.html#torchao.quantization.quantize_) API。

2.  FX 图模式量化 (torch.ao.quantization.quantize_fx.prepare_fx, torch.ao.quantization.quantize_fx.convert_fx)，请迁移到使用 torchao pt2e 量化 API ([torchao.quantization.pt2e.quantize_pt2e.prepare_pt2e], [torchao.quantization.pt2e.quantize_pt2e.convert_pt2e])。
3.  pt2e 量化已迁移到 torchao (<https://github.com/pytorch/ao/tree/main/torchao/quantization/pt2e>)，更多详情请参阅 <https://github.com/pytorch/ao/issues/2259>。

如果没有阻碍，我们计划在 2.10 版本中删除 [torch.ao.quantization]，或者在所有阻碍清除后的最早 PyTorch 版本中删除。

## 量化 API 参考（由于 API 仍为公开，故保留）

`量化 API 参考 <quantization-support>` 包含量化 API 的文档，例如量化过程、量化张量操作以及支持的量化模块和函数。


quantization-support


torch.ao.ns.fx.utils.compute_sqnr(x, y)


torch.ao.ns.fx.utils.compute_normalized_l2_error(x, y)


torch.ao.ns.fx.utils.compute_cosine_similarity(x, y)
