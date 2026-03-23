# 量化 API 参考

## torch.ao.quantization

此模块包含 Eager 模式量化 API。


### 顶层 API


### 为量化准备模型


### 实用函数


## torch.ao.quantization.utils


## torch.ao.quantization.quantize_fx

此模块包含 FX 图模式量化 API（原型）。


## torch.ao.quantization.qconfig_mapping

此模块包含用于配置 FX 图模式量化的 QConfigMapping。


## torch.ao.quantization.backend_config

此模块包含 BackendConfig，这是一个定义后端如何支持量化的配置对象。目前仅由 FX 图模式量化使用，但我们也可能扩展 Eager 模式量化以与此配合使用。


## torch.ao.quantization.backend_config.utils


## torch.ao.quantization.fx.custom_config

此模块包含一些在 Eager 模式和 FX 图模式量化中使用的 CustomConfig 类。


## torch.ao.quantization.fx.utils


## torch (量化相关函数)

此处描述了 `torch` 命名空间的量化相关函数。


## torch.Tensor (量化相关方法)

量化张量支持常规全精度张量的有限子集数据操作方法。


## torch.ao.quantization.observer

此模块包含观察器，用于收集在校准（PTQ）或训练（QAT）期间观察到的值的统计信息。


## torch.ao.quantization.fake_quantize

此模块实现在 QAT 期间执行伪量化的模块。


## torch.ao.quantization.qconfig

此模块定义了 `QConfig` 对象，用于配置单个操作的量化设置。


## torch.ao.quantization.quantization_mappings


## torch.ao.nn.intrinsic


此模块实现了组合（融合）模块 conv + relu，这些模块随后可以被量化。


## torch.ao.nn.intrinsic.qat


此模块实现了量化感知训练所需的那些融合操作的版本。


## torch.ao.nn.intrinsic.quantized


此模块实现了融合操作（如 conv + relu）的量化版本。不包含 BatchNorm 变体，因为在推理时它通常会被折叠到卷积中。


## torch.ao.nn.intrinsic.quantized.dynamic


此模块实现了融合操作（如 linear + relu）的量化动态版本。


## torch.ao.nn.qat


此模块实现了关键 nn 模块 **Conv2d()** 和 **Linear()** 的版本，这些模块在 FP32 下运行，但应用了舍入以模拟 INT8 量化的效果。


## torch.ao.nn.qat.dynamic


此模块实现了关键 nn 模块（如 **Linear()**）的版本，这些模块在 FP32 下运行，但应用了舍入以模拟 INT8 量化的效果，并将在推理期间动态量化。


## torch.ao.nn.quantized


此模块实现了 nn 层（如 `torch.nn.Conv2d` 和 `torch.nn.ReLU`）的量化版本。


## torch.ao.nn.quantized.functional


## torch.ao.nn.quantizable

此模块实现了部分 nn 层的可量化版本。
这些模块可与自定义模块机制结合使用，通过为 prepare 和 convert 函数提供 ``custom_module_config`` 参数来实现。


## torch.ao.nn.quantized.dynamic


动态量化的 `torch.nn.Linear`、`torch.nn.LSTM`、
`torch.nn.LSTMCell`、`torch.nn.GRUCell` 和
`torch.nn.RNNCell`。


## 量化数据类型与量化方案

请注意，当前算子实现仅支持 **conv** 和 **linear** 算子的权重进行逐通道量化。此外，输入数据与量化数据之间的映射关系如下：


其中 :math:`[x_\text{min}, x_\text{max}]` 表示输入数据的范围，而 :math:`Q_\text` 和 :math:`Q_\text` 分别是量化数据类型的最小值和最大值。

请注意，:math:`s` 和 :math:`z` 的选择意味着，只要零在输入数据范围内或使用对称量化，零的表示就不会引入量化误差。

可以通过 `自定义算子机制 <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_ 实现额外的数据类型和量化方案。
