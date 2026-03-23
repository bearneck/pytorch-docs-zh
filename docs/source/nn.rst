.. role:: hidden
    :class: hidden-section

.. toctree::
   :maxdepth: 2
   :hidden:

   nn.aliases.rst

torch.nn
===================================
.. automodule:: torch.nn

这些是构建计算图的基本模块：

.. contents:: torch.nn
    :depth: 2
    :local:
    :backlinks: top


.. currentmodule:: torch.nn


.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~parameter.Buffer
    ~parameter.Parameter
    ~parameter.UninitializedParameter
    ~parameter.UninitializedBuffer

容器
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Module
    Sequential
    ModuleList
    ModuleDict
    ParameterList
    ParameterDict

模块的全局钩子

.. currentmodule:: torch.nn.modules.module
.. autosummary::
    :toctree: generated
    :nosignatures:

    register_module_forward_pre_hook
    register_module_forward_hook
    register_module_backward_hook
    register_module_full_backward_pre_hook
    register_module_full_backward_hook
    register_module_buffer_registration_hook
    register_module_module_registration_hook
    register_module_parameter_registration_hook

.. currentmodule:: torch

卷积层
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Conv1d
    nn.Conv2d
    nn.Conv3d
    nn.ConvTranspose1d
    nn.ConvTranspose2d
    nn.ConvTranspose3d
    nn.LazyConv1d
    nn.LazyConv2d
    nn.LazyConv3d
    nn.LazyConvTranspose1d
    nn.LazyConvTranspose2d
    nn.LazyConvTranspose3d
    nn.Unfold
    nn.Fold

池化层
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.MaxPool1d
    nn.MaxPool2d
    nn.MaxPool3d
    nn.MaxUnpool1d
    nn.MaxUnpool2d
    nn.MaxUnpool3d
    nn.AvgPool1d
    nn.AvgPool2d
    nn.AvgPool3d
    nn.FractionalMaxPool2d
    nn.FractionalMaxPool3d
    nn.LPPool1d
    nn.LPPool2d
    nn.LPPool3d
    nn.AdaptiveMaxPool1d
    nn.AdaptiveMaxPool2d
    nn.AdaptiveMaxPool3d
    nn.AdaptiveAvgPool1d
    nn.AdaptiveAvgPool2d
    nn.AdaptiveAvgPool3d

填充层
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.ReflectionPad1d
    nn.ReflectionPad2d
    nn.ReflectionPad3d
    nn.ReplicationPad1d
    nn.ReplicationPad2d
    nn.ReplicationPad3d
    nn.ZeroPad1d
    nn.ZeroPad2d
    nn.ZeroPad3d
    nn.ConstantPad1d
    nn.ConstantPad2d
    nn.ConstantPad3d
    nn.CircularPad1d
    nn.CircularPad2d
    nn.CircularPad3d

非线性激活函数（加权求和，非线性变换）
---------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.ELU
    nn.Hardshrink
    nn.Hardsigmoid
    nn.Hardtanh
    nn.Hardswish
    nn.LeakyReLU
    nn.LogSigmoid
    nn.MultiheadAttention
    nn.PReLU
    nn.ReLU
    nn.ReLU6
    nn.RReLU
    nn.SELU
    nn.CELU
    nn.GELU
    nn.Sigmoid
    nn.SiLU
    nn.Mish
    nn.Softplus
    nn.Softshrink
    nn.Softsign
    nn.Tanh
    nn.Tanhshrink
    nn.Threshold
    nn.GLU

非线性激活函数（其他）
------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Softmin
    nn.Softmax
    nn.Softmax2d
    nn.LogSoftmax
    nn.AdaptiveLogSoftmaxWithLoss

归一化层
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.BatchNorm1d
    nn.BatchNorm2d
    nn.BatchNorm3d
    nn.LazyBatchNorm1d
    nn.LazyBatchNorm2d
    nn.LazyBatchNorm3d
    nn.GroupNorm
    nn.SyncBatchNorm
    nn.InstanceNorm1d
    nn.InstanceNorm2d
    nn.InstanceNorm3d
    nn.LazyInstanceNorm1d
    nn.LazyInstanceNorm2d
    nn.LazyInstanceNorm3d
    nn.LayerNorm
    nn.LocalResponseNorm
    nn.RMSNorm

循环层
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.RNNBase
    nn.RNN
    nn.LSTM
    nn.GRU
    nn.RNNCell
    nn.LSTMCell
    nn.GRUCell

Transformer 层
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Transformer
    nn.TransformerEncoder
    nn.TransformerDecoder
    nn.TransformerEncoderLayer
    nn.TransformerDecoderLayer

线性层
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Identity
    nn.Linear
    nn.Bilinear
    nn.LazyLinear

Dropout 层
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Dropout
    nn.Dropout1d
    nn.Dropout2d
    nn.Dropout3d
    nn.AlphaDropout
    nn.FeatureAlphaDropout

稀疏层
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Embedding
    nn.EmbeddingBag

距离函数
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.CosineSimilarity
    nn.PairwiseDistance

损失函数
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.L1Loss
    nn.MSELoss
    nn.CrossEntropyLoss
    nn.CTCLoss
    nn.NLLLoss
    nn.PoissonNLLLoss
    nn.GaussianNLLLoss
    nn.KLDivLoss
    nn.BCELoss
    nn.BCEWithLogitsLoss
    nn.MarginRankingLoss
    nn.HingeEmbeddingLoss
    nn.MultiLabelMarginLoss
    nn.HuberLoss
    nn.SmoothL1Loss
    nn.SoftMarginLoss
    nn.MultiLabelSoftMarginLoss
    nn.CosineEmbeddingLoss
    nn.MultiMarginLoss
    nn.TripletMarginLoss
    nn.TripletMarginWithDistanceLoss

视觉层
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.PixelShuffle
    nn.PixelUnshuffle
    nn.Upsample
    nn.UpsamplingNearest2d
    nn.UpsamplingBilinear2d

Shuffle Layers
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.ChannelShuffle

DataParallel Layers (multi-GPU, distributed)
--------------------------------------------
.. automodule:: torch.nn.parallel
.. currentmodule:: torch

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.DataParallel
    nn.parallel.DistributedDataParallel

Utilities
---------
.. automodule:: torch.nn.utils

来自 ``torch.nn.utils`` 模块：

用于裁剪参数梯度的实用函数。

.. currentmodule:: torch.nn.utils
.. autosummary::
    :toctree: generated
    :nosignatures:

    clip_grad_norm_
    clip_grad_norm
    clip_grad_value_
    get_total_norm
    clip_grads_with_norm_

用于将 Module 参数展平为单个向量以及从单个向量恢复的实用函数。

.. autosummary::
    :toctree: generated
    :nosignatures:

    parameters_to_vector
    vector_to_parameters

用于融合 Module 与 BatchNorm 模块的实用函数。

.. autosummary::
    :toctree: generated
    :nosignatures:

    fuse_conv_bn_eval
    fuse_conv_bn_weights
    fuse_linear_bn_eval
    fuse_linear_bn_weights

用于转换 Module 参数内存格式的实用函数。

.. autosummary::
    :toctree: generated
    :nosignatures:

    convert_conv2d_weight_memory_format
    convert_conv3d_weight_memory_format

用于对 Module 参数应用和移除权重归一化的实用函数。

.. autosummary::
    :toctree: generated
    :nosignatures:

    weight_norm
    remove_weight_norm
    spectral_norm
    remove_spectral_norm

用于初始化 Module 参数的实用函数。

.. autosummary::
    :toctree: generated
    :nosignatures:

    skip_init

用于剪枝 Module 参数的实用类和函数。

.. autosummary::
    :toctree: generated
    :nosignatures:

    prune.BasePruningMethod
    prune.PruningContainer
    prune.Identity
    prune.RandomUnstructured
    prune.L1Unstructured
    prune.RandomStructured
    prune.LnStructured
    prune.CustomFromMask
    prune.identity
    prune.random_unstructured
    prune.l1_unstructured
    prune.random_structured
    prune.ln_structured
    prune.global_unstructured
    prune.custom_from_mask
    prune.remove
    prune.is_pruned

使用 :func:`torch.nn.utils.parameterize.register_parametrization` 中的新参数化功能实现的参数化。

.. autosummary::
    :toctree: generated
    :nosignatures:

    parametrizations.orthogonal
    parametrizations.weight_norm
    parametrizations.spectral_norm

用于在现有 Module 上参数化张量的实用函数。
请注意，这些函数可用于参数化给定的 Parameter 或 Buffer，给定一个从输入空间映射到参数化空间的特定函数。它们不是将对象转换为参数的参数化。有关如何实现自己的参数化的更多信息，请参阅
`参数化教程 <https://pytorch.org/tutorials/intermediate/parametrizations.html>`_。

.. autosummary::
    :toctree: generated
    :nosignatures:

    parametrize.register_parametrization
    parametrize.remove_parametrizations
    parametrize.cached
    parametrize.is_parametrized
    parametrize.transfer_parametrizations_and_params
    parametrize.type_before_parametrizations

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    parametrize.ParametrizationList

用于以无状态方式调用给定 Module 的实用函数。

.. autosummary::
    :toctree: generated
    :nosignatures:

    stateless.functional_call

其他模块中的实用函数

.. currentmodule:: torch
.. autosummary::
    :toctree: generated
    :nosignatures:

    nn.utils.rnn.PackedSequence
    nn.utils.rnn.pack_padded_sequence
    nn.utils.rnn.pad_packed_sequence
    nn.utils.rnn.pad_sequence
    nn.utils.rnn.pack_sequence
    nn.utils.rnn.unpack_sequence
    nn.utils.rnn.unpad_sequence
    nn.utils.rnn.invert_permutation
    nn.parameter.is_lazy
    nn.factory_kwargs

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.flatten.Flatten
    nn.modules.flatten.Unflatten

Quantized Functions
--------------------

量化指的是以低于浮点精度的位宽执行计算和存储张量的技术。PyTorch 支持每张量和每通道的非对称线性量化。要了解更多关于如何在 PyTorch 中使用量化函数的信息，请参阅 :ref:`quantization-doc` 文档。

Lazy Modules Initialization
---------------------------

.. currentmodule:: torch
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.lazy.LazyModuleMixin

.. 此模块需要编写文档。暂时添加于此
.. 以便跟踪
.. py:module:: torch.nn.backends
.. py:module:: torch.nn.utils.stateless
.. py:module:: torch.nn.backends.thnn
.. py:module:: torch.nn.common_types
.. py:module:: torch.nn.cpp
.. py:module:: torch.nn.functional
.. py:module:: torch.nn.grad
.. py:module:: torch.nn.init
.. py:module:: torch.nn.modules.activation
.. py:module:: torch.nn.modules.adaptive
.. py:module:: torch.nn.modules.batchnorm
.. py:module:: torch.nn.modules.channelshuffle
.. py:module:: torch.nn.modules.container
.. py:module:: torch.nn.modules.conv
.. py:module:: torch.nn.modules.distance
.. py:module:: torch.nn.modules.dropout
.. py:module:: torch.nn.modules.flatten
.. py:module:: torch.nn.modules.fold
.. py:module:: torch.nn.modules.instancenorm
.. py:module:: torch.nn.modules.lazy
.. py:module:: torch.nn.modules.linear
.. py:module:: torch.nn.modules.loss
.. py:module:: torch.nn.modules.module
.. py:module:: torch.nn.modules.normalization
.. py:module:: torch.nn.modules.padding
.. py:module:: torch.nn.modules.pixelshuffle
.. py:module:: torch.nn.modules.pooling
.. py:module:: torch.nn.modules.rnn
.. py:module:: torch.nn.modules.sparse
.. py:module:: torch.nn.modules.transformer
.. py:module:: torch.nn.modules.upsampling
.. py:module:: torch.nn.modules.utils
.. py:module:: torch.nn.parallel.comm
.. py:module:: torch.nn.parallel.distributed
.. py:module:: torch.nn.parallel.parallel_apply
.. py:module:: torch.nn.parallel.replicate
.. py:module:: torch.nn.parallel.scatter_gather
.. py:module:: torch.nn.parameter
.. py:module:: torch.nn.utils.clip_grad
.. py:module:: torch.nn.utils.convert_parameters
.. py:module:: torch.nn.utils.fusion
.. py:module:: torch.nn.utils.init
.. py:module:: torch.nn.utils.memory_format
.. py:module:: torch.nn.utils.parametrizations
.. py:module:: torch.nn.utils.parametrize
.. py:module:: torch.nn.utils.prune
.. py:module:: torch.nn.utils.rnn