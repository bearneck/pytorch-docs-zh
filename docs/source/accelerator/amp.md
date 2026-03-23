# 自动混合精度

## 背景

自动混合精度（AMP）允许在训练或推理过程中同时使用单精度（32位）和半精度（16位）浮点类型。

关键组件包括：

- [**Autocast**](https://docs.pytorch.org/docs/stable/amp.html#autocasting)：自动将操作转换为较低精度（例如 float16 或 bfloat16），以提高性能，同时保持准确性。
- [**梯度缩放**](https://docs.pytorch.org/docs/stable/amp.html#gradient-scaling)：在反向传播过程中动态缩放梯度，以防止使用混合精度训练时出现下溢。

## 设计

### 类型转换策略

[`CastPolicy`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/autocast_mode.h#L416-L438) 用于定义类型转换规则。每个枚举值代表一组操作符的类型转换要求，确保对优先考虑精度或性能的操作进行一致处理。

| 策略                     | 说明                                                                          |
| :---                     | :---                                                                                 |
| **`lower_precision_fp`** | 在执行操作前将所有输入转换为 `lower_precision_fp`。                       |
| **`fp32`**               | 在运行操作前将所有输入转换为 `at::kFloat`。                               |
| **`fp32_set_opt_dtype`** | 在 `at::kFloat` 中执行，同时如果用户提供了输出数据类型，则予以尊重。 |
| **`fp32_append_dtype`**  | 将 at::kFloat 附加到参数中，并重新分派到类型感知的重载函数              |
| **`promote`**            | 在执行前将所有输入提升到“最宽”的数据类型。                           |

### 操作符列表

PyTorch 为上述每种类型转换策略定义了一个通用的操作符列表，作为新加速器开发者的参考。

| 策略                     | 操作符列表                                                                                    |
| :---                     | :---                                                                                              |
| **`lower_precision_fp`** | [列表链接](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/autocast_mode.h#L819-L852) |
| **`fp32`**               | [列表链接](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/autocast_mode.h#L854-L912) |
| **`fp32_set_opt_dtype`** | [列表链接](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/autocast_mode.h#L914-L931) |
| **`fp32_append_dtype`**  | [列表链接](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/autocast_mode.h#L933-L958) |
| **`promote`**            | [列表链接](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/autocast_mode.h#L960-L971) |

## 实现

### Python 集成

实现 `get_amp_supported_dtype` 方法，以返回新加速器在 AMP 上下文中支持的数据类型。


### C++ 集成

本节展示 AMP 如何为 `AutocastPrivateUse1` 分发键注册自动转换内核。

- 注册一个后备处理程序，使未处理的操作回退到其正常实现。
- 使用 `KERNEL_PRIVATEUSEONE` 辅助宏在 `AutocastPrivateUse1` 下注册特定的 aten 内核，该宏将操作映射到所需的精度实现（使用枚举 `at::autocast::CastPolicy`）

