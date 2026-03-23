# 报告问题

如果提供的变通方案不足以让 `torch.compile` 正常工作，那么您应该考虑向 PyTorch 报告该问题。但有几件事您可以做，以显著减轻我们的工作负担。

## 隔离测试

使用 `torch.compile` 的 `backend=` 选项检查 `torch.compile` 堆栈的哪个组件导致了问题。具体来说，尝试：

- `torch.compile(fn, backend="eager")`：仅运行 TorchDynamo，即 `torch.compile` 的图捕获组件。
- `torch.compile(fn, backend="aot_eager")`：运行 TorchDynamo 和 AOTAutograd，后者在编译期间额外生成反向图。
- `torch.compile(fn, backend="aot_eager_decomp_partition")`：运行 TorchDynamo 和 AOTAutograd，并启用算子分解/分区。
- `torch.compile(fn, backend="inductor")`：运行 TorchDynamo、AOTAutograd 和 TorchInductor，后者是生成编译内核的后端 ML 编译器。

如果仅在 Inductor 后端失败，您还可以测试各种 Inductor 模式：

- `torch.compile(fn, backend="inductor", mode="default")`
- `torch.compile(fn, backend="inductor", mode="reduce-overhead")`
- `torch.compile(fn, backend="inductor", mode="max-autotune")`

您还可以检查动态形状是否在任何后端导致问题：

- `torch.compile(fn, dynamic=True)` （始终使用动态形状）
- `torch.compile(fn, dynamic=False)` （从不使用动态形状）
- `torch.compile(fn, dynamic=None)` （自动动态形状）

## 二分查找

您是否尝试过最新的 nightly 版本？过去是否正常工作但现在不再工作？您能否通过二分查找来确定问题首次出现的 nightly 版本？二分查找对于性能、精度或编译时间退化问题尤其有帮助，因为这些问题通常无法立即定位根源。

## 创建复现脚本

创建复现脚本需要大量工作，如果您没有时间做，这完全可以理解。但是，如果您是一位积极但不熟悉 `torch.compile` 内部机制的用户，创建一个独立的复现脚本对我们修复错误的能力有巨大影响。如果没有复现脚本，您的错误报告必须包含足够的信息，以便我们能够识别问题的根本原因并从头编写复现脚本。

以下是有用的复现脚本列表，按从最优先到最不优先排序：

1.  **独立、小型复现脚本**：一个没有外部依赖、少于 100 行代码的脚本，运行时能复现问题。
2.  **独立、大型复现脚本**：即使规模较大，但独立性强是一个巨大的优势！
3.  **依赖项可控的非独立复现脚本**：例如，如果在 `pip install transformers` 后运行脚本能复现问题，这是可控的。我们很可能可以运行并调查。
4.  **需要大量设置的非独立复现脚本**：这可能涉及下载数据集、多个环境设置步骤，或需要特定系统库版本以至于需要 Docker 镜像。设置越复杂，我们重建环境的难度就越大。


> 📝 **注意**
> Docker 简化了设置，但使环境变更复杂化，因此它不是一个完美的解决方案，尽管必要时我们会使用它。


如果可能，尽量使您的复现脚本是单进程的，因为单进程比多进程的复现脚本更容易调试。

此外，以下是在您的问题中可能需要检查并尝试在复现脚本中复现的方面（非穷举列表）：

- **自动梯度**。您的张量输入是否设置了 `requires_grad=True`？您是否在输出上调用了 `backward()`？
- **动态形状**。您是否设置了 `dynamic=True`？或者您是否使用不同形状多次运行了测试代码？
- **自定义算子**。真实工作流中是否涉及自定义算子？您能否使用 Python 自定义算子 API 复现其某些重要特性？
- **配置**。您是否设置了所有相同的配置？这包括 `torch._dynamo.config` 和 `torch._inductor.config` 设置，以及 `torch.compile` 的参数，如 `backend` / `mode`。
- **上下文管理器**。您是否复现了任何活动的上下文管理器？这可能是 `torch.no_grad`、自动混合精度、`TorchFunctionMode` / `TorchDispatchMode`、激活检查点、编译自动梯度等。
- **张量子类**。是否涉及张量子类？