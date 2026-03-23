# torch.nn.functional


## 卷积函数


conv1d conv2d conv3d conv_transpose1d conv_transpose2d conv_transpose3d unfold fold

## 池化函数


avg_pool1d avg_pool2d avg_pool3d max_pool1d max_pool2d max_pool3d max_unpool1d max_unpool2d max_unpool3d lp_pool1d lp_pool2d lp_pool3d adaptive_max_pool1d adaptive_max_pool2d adaptive_max_pool3d adaptive_avg_pool1d adaptive_avg_pool2d adaptive_avg_pool3d fractional_max_pool2d fractional_max_pool3d

## 注意力机制

`torch.nn.attention.bias` 模块包含设计用于与 scaled_dot_product_attention 一起使用的注意力偏置。


scaled_dot_product_attention

## 非线性激活函数


threshold [threshold]() relu [relu]() hardtanh [hardtanh]() hardswish relu6 elu [elu]() selu celu leaky_relu [leaky_relu]() prelu rrelu [rrelu]() glu gelu logsigmoid hardshrink tanhshrink softsign softplus softmin softmax softshrink gumbel_softmax log_softmax tanh sigmoid hardsigmoid silu mish batch_norm group_norm instance_norm layer_norm local_response_norm rms_norm normalize

## 线性函数


linear bilinear

## Dropout 函数


dropout alpha_dropout feature_alpha_dropout dropout1d dropout2d dropout3d

## 稀疏函数


embedding embedding_bag one_hot

## 距离函数


pairwise_distance cosine_similarity pdist

## 损失函数


binary_cross_entropy binary_cross_entropy_with_logits poisson_nll_loss cosine_embedding_loss cross_entropy ctc_loss gaussian_nll_loss hinge_embedding_loss kl_div l1_loss mse_loss margin_ranking_loss multilabel_margin_loss multilabel_soft_margin_loss multi_margin_loss nll_loss huber_loss smooth_l1_loss soft_margin_loss triplet_margin_loss triplet_margin_with_distance_loss

## 视觉函数


pixel_shuffle pixel_unshuffle pad interpolate upsample upsample_nearest upsample_bilinear grid_sample affine_grid

## DataParallel 函数 (多 GPU，分布式)

### [data_parallel]


torch.nn.parallel.data_parallel

## 低精度函数


ScalingType SwizzleType grouped_mm scaled_mm scaled_grouped_mm
