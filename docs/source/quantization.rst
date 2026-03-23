.. _quantization-doc:

量化
============

.. automodule:: torch.ao.quantization
.. automodule:: torch.ao.quantization.fx

我们正在将所有与量化相关的开发集中到 `torchao <https://github.com/pytorch/ao>`__，请查看我们的新文档页面：https://docs.pytorch.org/ao/stable/index.html

现有量化流程的计划：
1. Eager 模式量化 (torch.ao.quantization.quantize, torch.ao.quantization.quantize_dynamic)，请迁移到使用 torchao eager 模式 `quantize_ <https://docs.pytorch.org/ao/main/generated/torchao.quantization.quantize_.html#torchao.quantization.quantize_>`__ API。

2. FX 图模式量化 (torch.ao.quantization.quantize_fx.prepare_fx, torch.ao.quantization.quantize_fx.convert_fx)，请迁移到使用 torchao pt2e 量化 API (`torchao.quantization.pt2e.quantize_pt2e.prepare_pt2e`, `torchao.quantization.pt2e.quantize_pt2e.convert_pt2e`)。

3. pt2e 量化已迁移到 torchao (https://github.com/pytorch/ao/tree/main/torchao/quantization/pt2e)，更多详情请参阅 https://github.com/pytorch/ao/issues/2259。

如果没有阻碍，我们计划在 2.10 版本中删除 `torch.ao.quantization`，或者在所有阻碍清除后的最早 PyTorch 版本中删除。

量化 API 参考（由于 API 仍为公开，故保留）
-----------------------------------------------------------------

:doc:`量化 API 参考 <quantization-support>` 包含量化 API 的文档，例如量化过程、量化张量操作以及支持的量化模块和函数。

.. toctree::
    :hidden:

    quantization-support

.. torch.ao 缺少文档。由于此处提到了它的一部分，暂时将它们添加在这里。
.. 它们在此处用于跟踪目的，直到它们得到更永久的修复。
.. py:module:: torch.ao
.. py:module:: torch.ao.nn
.. py:module:: torch.ao.nn.quantizable
.. py:module:: torch.ao.nn.quantizable.modules
.. py:module:: torch.ao.nn.quantized
.. py:module:: torch.ao.nn.quantized.reference
.. py:module:: torch.ao.nn.quantized.reference.modules
.. py:module:: torch.ao.nn.sparse
.. py:module:: torch.ao.nn.sparse.quantized
.. py:module:: torch.ao.nn.sparse.quantized.dynamic
.. py:module:: torch.ao.ns
.. py:module:: torch.ao.ns.fx
.. py:module:: torch.ao.quantization.backend_config
.. py:module:: torch.ao.pruning
.. py:module:: torch.ao.pruning.scheduler
.. py:module:: torch.ao.pruning.sparsifier
.. py:module:: torch.ao.nn.intrinsic.modules.fused
.. py:module:: torch.ao.nn.intrinsic.qat.modules.linear_fused
.. py:module:: torch.ao.nn.intrinsic.qat.modules.linear_relu
.. py:module:: torch.ao.nn.intrinsic.quantized.dynamic.modules.linear_relu
.. py:module:: torch.ao.nn.intrinsic.quantized.modules.bn_relu
.. py:module:: torch.ao.nn.intrinsic.quantized.modules.conv_add
.. py:module:: torch.ao.nn.intrinsic.quantized.modules.linear_relu
.. py:module:: torch.ao.nn.qat.dynamic.modules.linear
.. py:module:: torch.ao.nn.qat.modules.conv
.. py:module:: torch.ao.nn.qat.modules.embedding_ops
.. py:module:: torch.ao.nn.qat.modules.linear
.. py:module:: torch.ao.nn.quantizable.modules.activation
.. py:module:: torch.ao.nn.quantizable.modules.rnn
.. py:module:: torch.ao.nn.quantized.dynamic.modules.conv
.. py:module:: torch.ao.nn.quantized.dynamic.modules.linear
.. py:module:: torch.ao.nn.quantized.dynamic.modules.rnn
.. py:module:: torch.ao.nn.quantized.modules.activation
.. py:module:: torch.ao.nn.quantized.modules.batchnorm
.. py:module:: torch.ao.nn.quantized.modules.conv
.. py:module:: torch.ao.nn.quantized.modules.dropout
.. py:module:: torch.ao.nn.quantized.modules.embedding_ops
.. py:module:: torch.ao.nn.quantized.modules.functional_modules
.. py:module:: torch.ao.nn.quantized.modules.linear
.. py:module:: torch.ao.nn.quantized.modules.normalization
.. py:module:: torch.ao.nn.quantized.modules.rnn
.. py:module:: torch.ao.nn.quantized.modules.utils
.. py:module:: torch.ao.nn.quantized.reference.modules.conv
.. py:module:: torch.ao.nn.quantized.reference.modules.linear
.. py:module:: torch.ao.nn.quantized.reference.modules.rnn
.. py:module:: torch.ao.nn.quantized.reference.modules.sparse
.. py:module:: torch.ao.nn.quantized.reference.modules.utils
.. py:module:: torch.ao.nn.sparse.quantized.dynamic.linear
.. py:module:: torch.ao.nn.sparse.quantized.linear
.. py:module:: torch.ao.nn.sparse.quantized.utils
.. py:module:: torch.ao.ns.fx.graph_matcher
.. py:module:: torch.ao.ns.fx.graph_passes
.. py:module:: torch.ao.ns.fx.mappings
.. py:module:: torch.ao.ns.fx.n_shadows_utils
.. py:module:: torch.ao.ns.fx.ns_types
.. py:module:: torch.ao.ns.fx.pattern_utils
.. py:module:: torch.ao.ns.fx.qconfig_multi_mapping
.. py:module:: torch.ao.ns.fx.weight_utils
.. py:module:: torch.ao.ns.fx.utils
.. py:module:: torch.ao.pruning.scheduler.base_scheduler
.. py:module:: torch.ao.pruning.scheduler.cubic_scheduler
.. py:module:: torch.ao.pruning.scheduler.lambda_scheduler
.. py:module:: torch.ao.pruning.sparsifier.base_sparsifier
.. py:module:: torch.ao.pruning.sparsifier.nearly_diagonal_sparsifier
.. py:module:: torch.ao.pruning.sparsifier.utils
.. py:module:: torch.ao.pruning.sparsifier.weight_norm_sparsifier
.. py:module:: torch.ao.quantization.backend_config.backend_config
.. py:module:: torch.ao.quantization.backend_config.executorch
.. py:module:: torch.ao.quantization.backend_config.fbgemm
.. py:module:: torch.ao.quantization.backend_config.native
.. py:module:: torch.ao.quantization.backend_config.onednn
.. py:module:: torch.ao.quantization.backend_config.qnnpack
.. py:module:: torch.ao.quantization.backend_config.tensorrt
.. py:module:: torch.ao.quantization.backend_config.utils
.. py:module:: torch.ao.quantization.backend_config.x86
.. py:module:: torch.ao.quantization.fake_quantize
.. py:module:: torch.ao.quantization.fuser_method_mappings
.. py:module:: torch.ao.quantization.fuse_modules
.. py:module:: torch.ao.quantization.fx.convert
.. py:module:: torch.ao.quantization.fx.custom_config
.. py:module:: torch.ao.quantization.fx.fuse
.. py:module:: torch.ao.quantization.fx.fuse_handler
.. py:module:: torch.ao.quantization.fx.graph_module
.. py:module:: torch.ao.quantization.fx.lower_to_fbgemm
.. py:module:: torch.ao.quantization.fx.lower_to_qnnpack
.. py:module:: torch.ao.quantization.fx.lstm_utils
.. py:module:: torch.ao.quantization.fx.match_utils
.. py:module:: torch.ao.quantization.fx.pattern_utils
.. py:module:: torch.ao.quantization.fx.prepare
.. py:module:: torch.ao.quantization.fx.qconfig_mapping_utils
.. py:module:: torch.ao.quantization.fx.quantize_handler
.. py:module:: torch.ao.quantization.fx.tracer
.. py:module:: torch.ao.quantization.fx.utils
.. py:module:: torch.ao.quantization.observer
.. py:module:: torch.ao.quantization.qconfig
.. py:module:: torch.ao.quantization.qconfig_mapping
.. py:module:: torch.ao.quantization.quant_type
.. py:module:: torch.ao.quantization.quantize_jit
.. py:module:: torch.ao.quantization.stubs
.. py:module:: torch.nn.intrinsic.modules.fused
.. py:module:: torch.nn.intrinsic.qat.modules.conv_fused
.. py:module:: torch.nn.intrinsic.qat.modules.linear_fused
.. py:module:: torch.nn.intrinsic.qat.modules.linear_relu
.. py:module:: torch.nn.intrinsic.quantized.dynamic.modules.linear_relu
.. py:module:: torch.nn.intrinsic.quantized.modules.bn_relu
.. py:module:: torch.nn.intrinsic.quantized.modules.conv_relu
.. py:module:: torch.nn.intrinsic.quantized.modules.linear_relu
.. py:module:: torch.nn.qat.dynamic.modules.linear
.. py:module:: torch.nn.qat.modules.conv
.. py:module:: torch.nn.qat.modules.embedding_ops
.. py:module:: torch.nn.qat.modules.linear
.. py:module:: torch.nn.quantizable.modules.activation
.. py:module:: torch.nn.quantizable.modules.rnn
.. py:module:: torch.nn.quantized.dynamic.modules.conv
.. py:module:: torch.nn.quantized.dynamic.modules.linear
.. py:module:: torch.nn.quantized.dynamic.modules.rnn
.. py:module:: torch.nn.quantized.functional
.. py:module:: torch.nn.quantized.modules.activation
.. py:module:: torch.nn.quantized.modules.batchnorm
.. py:module:: torch.nn.quantized.modules.conv
.. py:module:: torch.nn.quantized.modules.dropout
.. py:module:: torch.nn.quantized.modules.embedding_ops
.. py:module:: torch.nn.quantized.modules.functional_modules
.. py:module:: torch.nn.quantized.modules.linear
.. py:module:: torch.nn.quantized.modules.normalization
.. py:module:: torch.nn.quantized.modules.rnn
.. py:module:: torch.nn.quantized.modules.utils
.. py:module:: torch.quantization.fake_quantize
.. py:module:: torch.quantization.fuse_modules
.. py:module:: torch.quantization.fuser_method_mappings
.. py:module:: torch.quantization.fx.convert
.. py:module:: torch.quantization.fx.fuse
.. py:module:: torch.quantization.fx.fusion_patterns
.. py:module:: torch.quantization.fx.graph_module
.. py:module:: torch.quantization.fx.match_utils
.. py:module:: torch.quantization.fx.pattern_utils
.. py:module:: torch.quantization.fx.prepare
.. py:module:: torch.quantization.fx.quantization_patterns
.. py:module:: torch.quantization.fx.quantization_types
.. py:module:: torch.quantization.fx.utils
.. py:module:: torch.quantization.observer
.. py:module:: torch.quantization.qconfig
.. py:module:: torch.quantization.quant_type
.. py:module:: torch.quantization.quantization_mappings
.. py:module:: torch.quantization.quantize
.. py:module:: torch.quantization.quantize_fx
.. py:module:: torch.quantization.quantize_jit
.. py:module:: torch.quantization.stubs
.. py:module:: torch.quantization.utils

.. currentmodule:: torch.ao.ns.fx.utils
.. autofunction:: torch.ao.ns.fx.utils.compute_sqnr(x, y)
.. autofunction:: torch.ao.ns.fx.utils.compute_normalized_l2_error(x, y)
.. autofunction:: torch.ao.ns.fx.utils.compute_cosine_similarity(x, y)