# 量化 API 参考

## torch.ao.quantization

此模块包含 Eager 模式量化 API。

```{eval-rst}
.. currentmodule:: torch.ao.quantization
```

### 顶层 API

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    quantize
    quantize_dynamic
    quantize_qat
    prepare
    prepare_qat
    convert
```

### 为量化准备模型

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    fuse_modules.fuse_modules
    QuantStub
    DeQuantStub
    QuantWrapper
    add_quant_dequant
```

### 实用函数

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    swap_module
    propagate_qconfig_
    default_eval_fn
```

## torch.ao.quantization.utils

```{eval-rst}
.. automodule:: torch.ao.quantization.utils
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    activation_is_dynamically_quantized
    activation_is_int32_quantized
    activation_is_int8_quantized
    activation_is_statically_quantized

    determine_qparams
    check_min_max_valid
    calculate_qmin_qmax
    validate_qmin_qmax
```

## torch.ao.quantization.quantize_fx

此模块包含 FX 图模式量化 API（原型）。

```{eval-rst}
.. currentmodule:: torch.ao.quantization.quantize_fx
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    prepare_fx
    prepare_qat_fx
    convert_fx
    fuse_fx
```

## torch.ao.quantization.qconfig_mapping

此模块包含用于配置 FX 图模式量化的 QConfigMapping。

```{eval-rst}
.. currentmodule:: torch.ao.quantization.qconfig_mapping
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    QConfigMapping
    get_default_qconfig_mapping
    get_default_qat_qconfig_mapping
```

## torch.ao.quantization.backend_config

此模块包含 BackendConfig，这是一个定义后端如何支持量化的配置对象。目前仅由 FX 图模式量化使用，但我们也可能扩展 Eager 模式量化以与此配合使用。

```{eval-rst}
.. currentmodule:: torch.ao.quantization.backend_config
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    BackendConfig
    BackendPatternConfig
    DTypeConfig
    DTypeWithConstraints
    ObservationType
```

## torch.ao.quantization.backend_config.utils
```{eval-rst}
.. currentmodule:: torch.ao.quantization.backend_config.utils
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    entry_to_pretty_str
    pattern_to_human_readable
    remove_boolean_dispatch_from_name

```

## torch.ao.quantization.fx.custom_config

此模块包含一些在 Eager 模式和 FX 图模式量化中使用的 CustomConfig 类。

```{eval-rst}
.. currentmodule:: torch.ao.quantization.fx.custom_config
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    FuseCustomConfig
    PrepareCustomConfig
    ConvertCustomConfig
    StandaloneModuleConfigEntry
```

## torch.ao.quantization.fx.utils

```{eval-rst}
.. currentmodule:: torch.ao.quantization.fx.utils
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    all_node_args_except_first
    all_node_args_have_no_tensors
    collect_producer_nodes
    create_getattr_from_value
    create_node_from_old_node_preserve_meta
    graph_module_from_producer_nodes
    maybe_get_next_module
    node_arg_is_bias
    node_arg_is_weight
    return_arg_list
```

## torch (量化相关函数)

此处描述了 `torch` 命名空间的量化相关函数。

```{eval-rst}
.. currentmodule:: torch
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    quantize_per_tensor
    quantize_per_channel
    dequantize
```

## torch.Tensor (量化相关方法)

量化张量支持常规全精度张量的有限子集数据操作方法。

```{eval-rst}
.. currentmodule:: torch.Tensor
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    view
    as_strided
    expand
    flatten
    select
    ne
    eq
    ge
    le
    gt
    lt
    copy_
    clone
    dequantize
    equal
    int_repr
    max
    mean
    min
    q_scale
    q_zero_point
    q_per_channel_scales
    q_per_channel_zero_points
    q_per_channel_axis
    resize_
    sort
    topk
```

## torch.ao.quantization.observer

此模块包含观察器，用于收集在校准（PTQ）或训练（QAT）期间观察到的值的统计信息。

```{eval-rst}
.. currentmodule:: torch.ao.quantization.observer
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ObserverBase
    MinMaxObserver
    MovingAverageMinMaxObserver
    PerChannelMinMaxObserver
    MovingAveragePerChannelMinMaxObserver
    HistogramObserver
    PlaceholderObserver
    RecordingObserver
    NoopObserver
    get_observer_state_dict
    load_observer_state_dict
    default_observer
    default_placeholder_observer
    default_debug_observer
    default_weight_observer
    default_histogram_observer
    default_per_channel_weight_observer
    default_dynamic_quant_observer
    default_float_qparams_observer
    AffineQuantizedObserverBase
    Granularity
    MappingType
    PerAxis
    PerBlock
    PerGroup
    PerRow
    PerTensor
    PerToken
    TorchAODType
    ZeroPointDomain
    get_block_size
```

## torch.ao.quantization.fake_quantize

此模块实现在 QAT 期间执行伪量化的模块。

```{eval-rst}
.. currentmodule:: torch.ao.quantization.fake_quantize
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    FakeQuantizeBase
    FakeQuantize
    FixedQParamsFakeQuantize
    FusedMovingAvgObsFakeQuantize
    default_fake_quant
    default_weight_fake_quant
    default_per_channel_weight_fake_quant
    default_histogram_fake_quant
    default_fused_act_fake_quant
    default_fused_wt_fake_quant
    default_fused_per_channel_wt_fake_quant
    disable_fake_quant
    enable_fake_quant
    disable_observer
    enable_observer
```

## torch.ao.quantization.qconfig

此模块定义了 `QConfig` 对象，用于配置单个操作的量化设置。

```{eval-rst}
.. currentmodule:: torch.ao.quantization.qconfig
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    QConfig
    default_qconfig
    default_debug_qconfig
    default_per_channel_qconfig
    default_dynamic_qconfig
    float16_dynamic_qconfig
    float16_static_qconfig
    per_channel_dynamic_qconfig
    float_qparams_weight_only_qconfig
    default_qat_qconfig
    default_weight_only_qconfig
    default_activation_only_qconfig
    default_qat_qconfig_v2
```

## torch.ao.quantization.quantization_mappings

```{eval-rst}
.. automodule:: torch.ao.quantization.quantization_mappings
```

```{eval-rst}
.. currentmodule:: torch.ao.quantization.quantization_mappings
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    no_observer_set
```

## torch.ao.nn.intrinsic

```{eval-rst}
.. automodule:: torch.ao.nn.intrinsic
.. automodule:: torch.ao.nn.intrinsic.modules
```

此模块实现了组合（融合）模块 conv + relu，这些模块随后可以被量化。

```{eval-rst}
.. currentmodule:: torch.ao.nn.intrinsic
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ConvReLU1d
    ConvReLU2d
    ConvReLU3d
    LinearReLU
    ConvBn1d
    ConvBn2d
    ConvBn3d
    ConvBnReLU1d
    ConvBnReLU2d
    ConvBnReLU3d
    BNReLU2d
    BNReLU3d
```

## torch.ao.nn.intrinsic.qat

```{eval-rst}
.. automodule:: torch.ao.nn.intrinsic.qat
.. automodule:: torch.ao.nn.intrinsic.qat.modules
```

此模块实现了量化感知训练所需的那些融合操作的版本。

```{eval-rst}
.. currentmodule:: torch.ao.nn.intrinsic.qat
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    LinearReLU
    ConvBn1d
    ConvBnReLU1d
    ConvBn2d
    ConvBnReLU2d
    ConvReLU2d
    ConvBn3d
    ConvBnReLU3d
    ConvReLU3d
    update_bn_stats
    freeze_bn_stats
```

## torch.ao.nn.intrinsic.quantized

```{eval-rst}
.. automodule:: torch.ao.nn.intrinsic.quantized
.. automodule:: torch.ao.nn.intrinsic.quantized.modules
```

此模块实现了融合操作（如 conv + relu）的量化版本。不包含 BatchNorm 变体，因为在推理时它通常会被折叠到卷积中。

```{eval-rst}
.. currentmodule:: torch.ao.nn.intrinsic.quantized
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    BNReLU2d
    BNReLU3d
    ConvReLU1d
    ConvReLU2d
    ConvReLU3d
    LinearReLU
```

## torch.ao.nn.intrinsic.quantized.dynamic

```{eval-rst}
.. automodule:: torch.ao.nn.intrinsic.quantized.dynamic
.. automodule:: torch.ao.nn.intrinsic.quantized.dynamic.modules
```

此模块实现了融合操作（如 linear + relu）的量化动态版本。

```{eval-rst}
.. currentmodule:: torch.ao.nn.intrinsic.quantized.dynamic
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    LinearReLU
```

## torch.ao.nn.qat

```{eval-rst}
.. automodule:: torch.ao.nn.qat
.. automodule:: torch.ao.nn.qat.modules
```

此模块实现了关键 nn 模块 **Conv2d()** 和 **Linear()** 的版本，这些模块在 FP32 下运行，但应用了舍入以模拟 INT8 量化的效果。

```{eval-rst}
.. currentmodule:: torch.ao.nn.qat
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Conv2d
    Conv3d
    Linear
```

## torch.ao.nn.qat.dynamic

```{eval-rst}
.. automodule:: torch.ao.nn.qat.dynamic
.. automodule:: torch.ao.nn.qat.dynamic.modules
```

此模块实现了关键 nn 模块（如 **Linear()**）的版本，这些模块在 FP32 下运行，但应用了舍入以模拟 INT8 量化的效果，并将在推理期间动态量化。

```{eval-rst}
.. currentmodule:: torch.ao.nn.qat.dynamic
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Linear
```

## torch.ao.nn.quantized

```{eval-rst}
.. automodule:: torch.ao.nn.quantized
   :noindex:
.. automodule:: torch.ao.nn.quantized.modules
```

此模块实现了 nn 层（如 `~torch.nn.Conv2d` 和 `torch.nn.ReLU`）的量化版本。

```{eval-rst}
.. currentmodule:: torch.ao.nn.quantized
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ReLU6
    Hardswish
    ELU
    LeakyReLU
    Sigmoid
    BatchNorm2d
    BatchNorm3d
    Conv1d
    Conv2d
    Conv3d
    ConvTranspose1d
    ConvTranspose2d
    ConvTranspose3d
    Embedding
    EmbeddingBag
    FloatFunctional
    FXFloatFunctional
    QFunctional
    Linear
    LayerNorm
    GroupNorm
    InstanceNorm1d
    InstanceNorm2d
    InstanceNorm3d
```

## torch.ao.nn.quantized.functional

```{eval-rst}
.. automodule:: torch.ao.nn.quantized.functional
```

```{eval-rst}
此模块实现了功能层（如 `~torch.nn.functional.conv2d` 和 `torch.nn.functional.relu`）的量化版本。注意：
:math:`~torch.nn.functional.relu` 支持量化输入。
```

```{eval-rst}
.. currentmodule:: torch.ao.nn.quantized.functional
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    avg_pool2d
    avg_pool3d
    adaptive_avg_pool2d
    adaptive_avg_pool3d
    conv1d
    conv2d
    conv3d
    interpolate
    linear
    max_pool1d
    max_pool2d
    celu
    leaky_relu
    hardtanh
    hardswish
    threshold
    elu
    hardsigmoid
    clamp
    upsample
    upsample_bilinear
    upsample_nearest
```

## torch.ao.nn.quantizable

此模块实现了部分 nn 层的可量化版本。
这些模块可与自定义模块机制结合使用，通过为 prepare 和 convert 函数提供 ``custom_module_config`` 参数来实现。

```{eval-rst}
.. currentmodule:: torch.ao.nn.quantizable
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    LSTM
    MultiheadAttention
```

## torch.ao.nn.quantized.dynamic

```{eval-rst}
.. automodule:: torch.ao.nn.quantized.dynamic
.. automodule:: torch.ao.nn.quantized.dynamic.modules
```

动态量化的 {class}`~torch.nn.Linear`、{class}`~torch.nn.LSTM`、
{class}`~torch.nn.LSTMCell`、{class}`~torch.nn.GRUCell` 和
{class}`~torch.nn.RNNCell`。

```{eval-rst}
.. currentmodule:: torch.ao.nn.quantized.dynamic
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Linear
    LSTM
    GRU
    RNNCell
    LSTMCell
    GRUCell
```

## 量化数据类型与量化方案

请注意，当前算子实现仅支持 **conv** 和 **linear** 算子的权重进行逐通道量化。此外，输入数据与量化数据之间的映射关系如下：

```{eval-rst}
    .. math::

        \begin{aligned}
            \text{量化:}&\\
            &Q_\text{out} = \text{clamp}(x_\text{input}/s+z, Q_\text{min}, Q_\text{max})\\
            \text{反量化:}&\\
            &x_\text{out} = (Q_\text{input}-z)*s
        \end{aligned}
```

```{eval-rst}
其中 :math:`\text{clamp}(.)` 与 :func:`~torch.clamp` 函数相同，而比例因子 :math:`s` 和零点 :math:`z` 的计算方式如 :class:`~torch.ao.quantization.observer.MinMaxObserver` 所述，具体为：
```

```{eval-rst}
    .. math::

        \begin{aligned}
            \text{若为对称量化:}&\\
            &s = 2 \max(|x_\text{min}|, x_\text{max}) /
                \left( Q_\text{max} - Q_\text{min} \right) \\
            &z = \begin{cases}
                0 & \text{若数据类型为 qint8} \\
                128 & \text{其他情况}
            \end{cases}\\
            \text{否则:}&\\
                &s = \left( x_\text{max} - x_\text{min}  \right ) /
                    \left( Q_\text{max} - Q_\text{min} \right ) \\
                &z = Q_\text{min} - \text{round}(x_\text{min} / s)
        \end{aligned}
```

其中 :math:`[x_\text{min}, x_\text{max}]` 表示输入数据的范围，而 :math:`Q_\text{min}` 和 :math:`Q_\text{max}` 分别是量化数据类型的最小值和最大值。

请注意，:math:`s` 和 :math:`z` 的选择意味着，只要零在输入数据范围内或使用对称量化，零的表示就不会引入量化误差。

可以通过 `自定义算子机制 <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_ 实现额外的数据类型和量化方案。

```{eval-rst}
* :attr:`torch.qscheme` — 用于描述张量量化方案的类型。
  支持的类型：

  * :attr:`torch.per_tensor_affine` — 逐张量，非对称
  * :attr:`torch.per_channel_affine` — 逐通道，非对称
  * :attr:`torch.per_tensor_symmetric` — 逐张量，对称
  * :attr:`torch.per_channel_symmetric` — 逐通道，对称

* ``torch.dtype`` — 用于描述数据的类型。支持的类型：

  * :attr:`torch.quint8` — 8 位无符号整数
  * :attr:`torch.qint8` — 8 位有符号整数
  * :attr:`torch.qint32` — 32 位有符号整数
```

```{eval-rst}
.. 这些模块缺少文档。在此添加仅用于跟踪
.. automodule:: torch.ao.nn.quantizable.modules
   :noindex:
.. automodule:: torch.ao.nn.quantized.reference
   :noindex:
.. automodule:: torch.ao.nn.quantized.reference.modules
   :noindex:

.. automodule:: torch.nn.quantizable
.. automodule:: torch.nn.qat.dynamic.modules
.. automodule:: torch.nn.qat.modules
.. automodule:: torch.nn.qat
.. automodule:: torch.nn.intrinsic.qat.modules
.. automodule:: torch.nn.quantized.dynamic
.. automodule:: torch.nn.intrinsic
.. automodule:: torch.nn.intrinsic.quantized.modules
.. automodule:: torch.quantization.fx
.. automodule:: torch.nn.intrinsic.quantized.dynamic
.. automodule:: torch.nn.qat.dynamic
.. automodule:: torch.nn.intrinsic.qat
.. automodule:: torch.nn.quantized.modules
.. automodule:: torch.nn.intrinsic.quantized
.. automodule:: torch.nn.quantizable.modules
.. automodule:: torch.nn.quantized
.. automodule:: torch.nn.intrinsic.quantized.dynamic.modules
.. automodule:: torch.nn.quantized.dynamic.modules
.. automodule:: torch.quantization
.. automodule:: torch.nn.intrinsic.modules
```

```{eval-rst}
.. toctree::
    :hidden:

    quantization-support.aliases.md
```