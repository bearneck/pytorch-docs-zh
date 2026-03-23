# 推理模式

`c10::InferenceMode` 是一个新的 RAII 守卫，类似于 `NoGradMode`，用于当你确定你的操作不会与自动求导（例如模型训练）交互时。与 `NoGradMode` 相比，在此模式下运行的代码通过禁用与自动求导相关的工作（如视图跟踪和版本计数器递增）来获得更好的性能。然而，在 `c10::InferenceMode` 内部创建的张量与自动求导系统交互时也有更多限制。

`InferenceMode` 可以为给定的代码块启用。在 `InferenceMode` 内部，所有新分配的（非视图）张量都被标记为推理张量。推理张量：

- 没有版本计数器，因此如果你尝试读取它们的版本（例如，因为你保存了这个张量用于反向传播），将会引发错误。
- 在 `InferenceMode` 外部是不可变的。因此，如果你尝试在 InferenceMode 外部进行以下操作，将会引发错误：
  - 在 InferenceMode 外部修改它们的数据。

  \- 在 InferenceMode 外部将它们修改为 `requires_grad=True`。 解决方法是在 `InferenceMode` 外部克隆以在修改前获得一个普通张量。

一个非视图张量是推理张量，当且仅当它是在 `InferenceMode` 内部分配的。 一个视图张量是推理张量，当且仅当它是一个推理张量的视图。

在 `InferenceMode` 块内部，我们提供以下性能保证：

- 与 `NoGradMode` 类似，所有操作即使其输入具有 `requires_grad=True`，也不会记录 `grad_fn`。这适用于推理张量和普通张量。
- 对推理张量的视图操作不进行视图跟踪。视图和非视图推理张量是无法区分的。
- 对推理张量的原地操作保证不会进行版本递增。

有关 `InferenceMode` 的更多实现细节，请参阅 [RFC-0011-InferenceMode](https://github.com/pytorch/rfcs/pull/17)。

## 从 `AutoNonVariableTypeMode` 迁移指南

在 PyTorch 用于推理工作负载的生产使用中，我们已经看到 C++ 守卫 `AutoNonVariableTypeMode`（现在称为 `AutoDispatchBelowADInplaceOrView`）的广泛使用，它禁用了自动求导、视图跟踪和版本计数器递增。不幸的是，当前这种守卫用于推理工作负载的用法是不安全的：使用 `AutoNonVariableTypeMode` 可能会绕过 PyTorch 的安全检查，导致静默的错误结果，例如，当保存用于反向传播的张量随后被修改时，PyTorch 会抛出错误，但在 `AutoNonVariableTypeMode` 内部发生的修改会静默绕过检查并向用户返回错误的梯度。

当 `AutoNonVariableTypeMode` 的当前用户考虑迁移时，以下步骤可能有助于你决定最佳替代方案：

1.  尝试在仅推理模式下运行工作负载的用户（例如加载预训练的 JIT 模型并在 C++ 运行时中运行推理）应该添加 `c10::InferenceMode guard` 来保护所有张量上的操作（包括模型加载）。请参见下面的推理工作负载示例：

``` cpp
c10::InferenceMode guard;
model.load_jit(saved_model);
auto inputs = preprocess_tensors(data);
auto out = model.forward(inputs);
auto outputs = postprocess_tensors(out);
```

注意 `c10::InferenceMode` 提供了 `AutoNonVariableTypeMode` 的直接替代品，保留了 `AutoNonVariableTypeMode` 的性能特性。但它们也有一些不同之处，用户应额外注意：

> - 两个守卫都影响张量执行过程以跳过与推理无关的工作，但 `InferenceMode` 还影响张量创建，而 `AutoNonVariableTypeMode` 不影响。换句话说，在 `InferenceMode` 内部创建的张量被标记为推理张量，以便在退出 `InferenceMode` 后可以应用某些限制。
> - `InferenceMode` 的启用/禁用状态可以嵌套，而 `AutoNonVariableTypeMode` 只允许启用状态。

``` cpp
{
  InferenceMode guard(true);
  // InferenceMode 开启
  {
    InferenceMode guard(false);
    // InferenceMode 关闭
  }
  // InferenceMode 开启
}
// InferenceMode 关闭
```

2.  尝试实现自定义内核并希望在 `Autograd` 分发键下重新分发的用户应该使用 `AutoDispatchBelowADInplaceOrView` 代替。注意 `AutoDispatchBelowADInplaceOrView` 只是 `AutoNonVariableTypeMode` 的新名称，因为它更好地解释了守卫的功能。我们正在弃用 `AutoNonVariableTypeMode`，它将在 1.10 版本中移除。请参见 `pytorch/vision` 中的自定义内核 `ROIAlignFunction` 示例：

``` cpp
class ROIAlignFunction : public torch::autograd::Function<ROIAlignFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input,
      const torch::autograd::Variable& rois,
      double spatial_scale,
      int64_t pooled_height,
      int64_t pooled_width,
      int64_t sampling_ratio,
      bool aligned) {
    ctx->saved_data["spatial_scale"] = spatial_scale;
    ctx->saved_data["pooled_height"] = pooled_height;
    ctx->saved_data["pooled_width"] = pooled_width;
    ctx->saved_data["sampling_ratio"] = sampling_ratio;
    ctx->saved_data["aligned"] = aligned;
    ctx->saved_data["input_shape"] = input.sizes();
    ctx->save_for_backward({rois});
    // 以前是 at::AutoNonVariableTypeMode g;
    at::AutoDispatchBelowADInplaceOrView guard;
    auto result = roi_align(
        input, rois, spatial_scale, pooled_height,
        pooled_width, sampling_ratio, aligned);
    return {result};
  }
```

自定义的原地和视图内核除了上述守卫外，还需要一些特殊处理，更多细节请参阅 [自定义内核教程](https://pytorch.org/tutorials/advanced/cpp_extension.html#backward-pass)。
