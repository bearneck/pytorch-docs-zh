# torch.nn 中的别名

以下是嵌套命名空间中 ``torch.nn`` 内对应组件的别名。

以下是 ``torch.nn.modules`` 命名空间中 ``torch.nn`` 内对应组件的别名。


## torch.nn.modules


## 容器 (别名)


| API | 说明 |
|-----|------|
| `container.Sequential` | — |
| `container.ModuleList` | — |
| `container.ModuleDict` | — |
| `container.ParameterList` | — |
| `container.ParameterDict` | — |

## 卷积层 (别名)


| API | 说明 |
|-----|------|
| `conv.Conv1d` | — |
| `conv.Conv2d` | — |
| `conv.Conv3d` | — |
| `conv.ConvTranspose1d` | — |
| `conv.ConvTranspose2d` | — |
| `conv.ConvTranspose3d` | — |
| `conv.LazyConv1d` | — |
| `conv.LazyConv2d` | — |
| `conv.LazyConv3d` | — |
| `conv.LazyConvTranspose1d` | — |
| `conv.LazyConvTranspose2d` | — |
| `conv.LazyConvTranspose3d` | — |
| `fold.Unfold` | — |
| `fold.Fold` | — |

## 池化层 (别名)


| API | 说明 |
|-----|------|
| `pooling.MaxPool1d` | — |
| `pooling.MaxPool2d` | — |
| `pooling.MaxPool3d` | — |
| `pooling.MaxUnpool1d` | — |
| `pooling.MaxUnpool2d` | — |
| `pooling.MaxUnpool3d` | — |
| `pooling.AvgPool1d` | — |
| `pooling.AvgPool2d` | — |
| `pooling.AvgPool3d` | — |
| `pooling.FractionalMaxPool2d` | — |
| `pooling.FractionalMaxPool3d` | — |
| `pooling.LPPool1d` | — |
| `pooling.LPPool2d` | — |
| `pooling.LPPool3d` | — |
| `pooling.AdaptiveMaxPool1d` | — |
| `pooling.AdaptiveMaxPool2d` | — |
| `pooling.AdaptiveMaxPool3d` | — |
| `pooling.AdaptiveAvgPool1d` | — |
| `pooling.AdaptiveAvgPool2d` | — |
| `pooling.AdaptiveAvgPool3d` | — |

## 填充层 (别名)


| API | 说明 |
|-----|------|
| `padding.ReflectionPad1d` | — |
| `padding.ReflectionPad2d` | — |
| `padding.ReflectionPad3d` | — |
| `padding.ReplicationPad1d` | — |
| `padding.ReplicationPad2d` | — |
| `padding.ReplicationPad3d` | — |
| `padding.ZeroPad1d` | — |
| `padding.ZeroPad2d` | — |
| `padding.ZeroPad3d` | — |
| `padding.ConstantPad1d` | — |
| `padding.ConstantPad2d` | — |
| `padding.ConstantPad3d` | — |
| `padding.CircularPad1d` | — |
| `padding.CircularPad2d` | — |
| `padding.CircularPad3d` | — |

## 非线性激活函数 (加权求和，非线性) (别名)


| API | 说明 |
|-----|------|
| `activation.ELU` | — |
| `activation.Hardshrink` | — |
| `activation.Hardsigmoid` | — |
| `activation.Hardtanh` | — |
| `activation.Hardswish` | — |
| `activation.LeakyReLU` | — |
| `activation.LogSigmoid` | — |
| `activation.MultiheadAttention` | — |
| `activation.PReLU` | — |
| `activation.ReLU` | — |
| `activation.ReLU6` | — |
| `activation.RReLU` | — |
| `activation.SELU` | — |
| `activation.CELU` | — |
| `activation.GELU` | — |
| `activation.Sigmoid` | — |
| `activation.SiLU` | — |
| `activation.Mish` | — |
| `activation.Softplus` | — |
| `activation.Softshrink` | — |
| `activation.Softsign` | — |
| `activation.Tanh` | — |
| `activation.Tanhshrink` | — |
| `activation.Threshold` | — |
| `activation.GLU` | — |

## 非线性激活函数 (其他) (别名)


| API | 说明 |
|-----|------|
| `activation.Softmin` | — |
| `activation.Softmax` | — |
| `activation.Softmax2d` | — |
| `activation.LogSoftmax` | — |
| `adaptive.AdaptiveLogSoftmaxWithLoss` | — |

## 归一化层 (别名)


| API | 说明 |
|-----|------|
| `batchnorm.BatchNorm1d` | — |
| `batchnorm.BatchNorm2d` | — |
| `batchnorm.BatchNorm3d` | — |
| `batchnorm.LazyBatchNorm1d` | — |
| `batchnorm.LazyBatchNorm2d` | — |
| `batchnorm.LazyBatchNorm3d` | — |
| `normalization.GroupNorm` | — |
| `batchnorm.SyncBatchNorm` | — |
| `instancenorm.InstanceNorm1d` | — |
| `instancenorm.InstanceNorm2d` | — |
| `instancenorm.InstanceNorm3d` | — |
| `instancenorm.LazyInstanceNorm1d` | — |
| `instancenorm.LazyInstanceNorm2d` | — |
| `instancenorm.LazyInstanceNorm3d` | — |
| `normalization.LayerNorm` | — |
| `normalization.LocalResponseNorm` | — |
| `normalization.RMSNorm` | — |

## 循环层 (别名)


| API | 说明 |
|-----|------|
| `rnn.RNNBase` | — |
| `rnn.RNN` | — |
| `rnn.LSTM` | — |
| `rnn.GRU` | — |
| `rnn.RNNCell` | — |
| `rnn.LSTMCell` | — |
| `rnn.GRUCell` | — |

## Transformer 层 (别名)


| API | 说明 |
|-----|------|
| `transformer.Transformer` | — |
| `transformer.TransformerEncoder` | — |
| `transformer.TransformerDecoder` | — |
| `transformer.TransformerEncoderLayer` | — |
| `transformer.TransformerDecoderLayer` | — |

## 线性层 (别名)


| API | 说明 |
|-----|------|
| `linear.Identity` | — |
| `linear.Linear` | — |
| `linear.Bilinear` | — |
| `linear.LazyLinear` | — |

## Dropout 层 (别名)


| API | 说明 |
|-----|------|
| `dropout.Dropout` | — |
| `dropout.Dropout1d` | — |
| `dropout.Dropout2d` | — |
| `dropout.Dropout3d` | — |
| `dropout.AlphaDropout` | — |
| `dropout.FeatureAlphaDropout` | — |

## 稀疏层 (别名)


| API | 说明 |
|-----|------|
| `sparse.Embedding` | — |
| `sparse.EmbeddingBag` | — |

## 距离函数 (别名)


| API | 说明 |
|-----|------|
| `distance.CosineSimilarity` | — |
| `distance.PairwiseDistance` | — |

## 损失函数 (别名)


| API | 说明 |
|-----|------|
| `loss.L1Loss` | — |
| `loss.MSELoss` | — |
| `loss.CrossEntropyLoss` | — |
| `loss.CTCLoss` | — |
| `loss.NLLLoss` | — |
| `loss.PoissonNLLLoss` | — |
| `loss.GaussianNLLLoss` | — |
| `loss.KLDivLoss` | — |
| `loss.BCELoss` | — |
| `loss.BCEWithLogitsLoss` | — |
| `loss.MarginRankingLoss` | — |
| `loss.HingeEmbeddingLoss` | — |
| `loss.MultiLabelMarginLoss` | — |
| `loss.HuberLoss` | — |
| `loss.SmoothL1Loss` | — |
| `loss.SoftMarginLoss` | — |
| `loss.MultiLabelSoftMarginLoss` | — |
| `loss.CosineEmbeddingLoss` | — |
| `loss.MultiMarginLoss` | — |
| `loss.TripletMarginLoss` | — |
| `loss.TripletMarginWithDistanceLoss` | — |

## 视觉层（别名）


| API | 说明 |
|-----|------|
| `pixelshuffle.PixelShuffle` | — |
| `pixelshuffle.PixelUnshuffle` | — |
| `upsampling.Upsample` | — |
| `upsampling.UpsamplingNearest2d` | — |
| `upsampling.UpsamplingBilinear2d` | — |

## 混洗层（别名）


| API | 说明 |
|-----|------|
| `channelshuffle.ChannelShuffle` | — |

## torch.nn.utils


| API | 说明 |
|-----|------|
| `clip_grad.clip_grad_norm_` | — |
| `clip_grad.clip_grad_norm` | — |
| `clip_grad.clip_grad_value_` | — |
| `convert_parameters.parameters_to_vector` | — |
| `convert_parameters.vector_to_parameters` | — |
| `fusion.fuse_conv_bn_eval` | — |
| `fusion.fuse_conv_bn_weights` | — |
| `fusion.fuse_linear_bn_eval` | — |
| `fusion.fuse_linear_bn_weights` | — |
| `memory_format.convert_conv2d_weight_memory_format` | — |
| `memory_format.convert_conv3d_weight_memory_format` | — |
| `weight_norm.weight_norm` | — |
| `weight_norm.remove_weight_norm` | — |
| `spectral_norm.spectral_norm` | — |
| `spectral_norm.remove_spectral_norm` | — |
| `init.skip_init` | — |
| `以下是嵌套命名空间中 ``torch.nn.utils`` 对应功能的别名。` | — |
| `torch.nn.utils.用于裁剪参数梯度的实用函数。` | — |
| `torch.nn.utils.用于将 Module 参数展平为单个向量以及从单个向量恢复的实用函数。` | — |
| `torch.nn.utils.用于融合 Module 与 BatchNorm 模块的实用函数。` | — |
| `torch.nn.utils.用于转换 Module 参数内存格式的实用函数。` | — |
| `torch.nn.utils.用于对 Module 参数应用和移除权重归一化的实用函数。` | — |
| `torch.nn.utils.用于初始化 Module 参数的实用函数。` | — |
