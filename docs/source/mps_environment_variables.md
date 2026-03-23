(mps_environment_variables)=
# MPS 环境变量

**PyTorch 环境变量**


| 变量名                           | 描述 |
|----------------------------------|-------------|
| `PYTORCH_DEBUG_MPS_ALLOCATOR`   | 如果设置为 `1`，将分配器日志级别设为详细模式。 |
| `PYTORCH_MPS_LOG_PROFILE_INFO`  | 为 `MPSProfiler` 设置日志选项位掩码。参见 `aten/src/ATen/mps/MPSProfiler.h` 中的 `LogOptions` 枚举。 |
| `PYTORCH_MPS_TRACE_SIGNPOSTS`   | 为 `MPSProfiler` 设置性能分析和标记位掩码。参见 `ProfileOptions` 和 `SignpostTypes`。 |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | MPS 分配器的高水位线比率。默认值为 1.7。 |
| `PYTORCH_MPS_LOW_WATERMARK_RATIO` | MPS 分配器的低水位线比率。默认值为 1.4（统一内存）或 1.0（独立内存）。 |
| `PYTORCH_MPS_FAST_MATH`         | 如果为 `1`，则为 MPS 内核启用快速数学运算。参见 [Metal 着色语言规范](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) 第 1.6.3 节。 |
| `PYTORCH_MPS_PREFER_METAL`      | 如果为 `1`，则使用 Metal 内核而非 MPS Graph API。用于矩阵乘法。 |
| `PYTORCH_ENABLE_MPS_FALLBACK`   | 如果为 `1`，当 MPS 操作不被支持时回退到 CPU。 |

```{note}
**高水位线比率** 是允许的总分配量的硬性限制

- `0.0` : 禁用高水位线限制（如果发生系统范围的 OOM 可能导致系统故障）
- `1.0` : 推荐的最大分配大小（即 device.recommendedMaxWorkingSetSize）
- `>1.0`: 允许限制超过 device.recommendedMaxWorkingSetSize

例如，值 0.95 表示我们最多分配推荐最大分配大小的 95%；超过此值，分配将因 OOM 错误而失败。

**低水位线比率** 是一个软性限制，旨在通过垃圾回收或更频繁地提交命令缓冲区（也称为自适应提交）来尝试将内存分配限制在较低水位线水平。值介于 0 到 m_high_watermark_ratio 之间（设置为 0.0 会禁用自适应提交和垃圾回收）。例如，值 0.9 表示我们“尝试”将分配限制在推荐最大分配大小的 90% 以内。
```