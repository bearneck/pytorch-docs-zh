# MaybeOwned\<Tensor\>

`MaybeOwned<Tensor>` 是一个 C++ 智能指针类，它动态地编码一个 Tensor 是\*被拥有\*还是\*被借用\*。它被用于某些对性能敏感的场景中，以避免不必要地增加 Tensor 的引用计数（代价是额外的间接寻址带来少量开销）。

 warning
 title
Warning


必须\*\*极其\*\*小心地使用 MaybeOwned。关于（非）所有权的声明没有静态检查，错误可能导致引用计数不足和释放后使用（use-after-free）的崩溃。

由于缺乏这种安全网，我们不鼓励在已知对性能高度敏感之外的代码路径中使用 MaybeOwned。但是，如果您在想要修改的代码中遇到已有的 MaybeOwned 用法，理解如何正确使用它是至关重要的。


`MaybeOwned<Tensor>` 的主要使用场景是这样一个函数或方法：它动态地选择是返回其参数之一（通常来自直通或"无操作"代码路径）还是返回一个新构造的 Tensor。这样的函数在两种情况下都会返回一个 `MaybeOwned<Tensor>`，前者通过调用 `MaybeOwned<Tensor>::borrowed()` 处于"借用"状态，后者通过调用 `MaybeOwned<Tensor>::owned()` 处于"拥有"状态。

典型的例子是 `Tensor` 的 `expect_contiguous` 方法，当已经连续时，它会短路并返回一个借用的自引用：

``` cpp
inline c10::MaybeOwned<Tensor> Tensor::expect_contiguous(MemoryFormat memory_format) const & {
  if (is_contiguous(memory_format)) {
    return c10::MaybeOwned<Tensor>::borrowed(*this);
  } else {
    return c10::MaybeOwned<Tensor>::owned(__dispatch_contiguous(memory_format));
  }
}
```

使用生命周期的术语，借用的基本安全要求是：被借用的 Tensor 必须比任何对它的借用引用存活得更久。例如在这里，我们可以安全地借用 `*this`，但由 `__dispatch_contiguous()` 返回的 Tensor 是新创建的，借用其引用实际上会使其失去所有者。

因此，一般的经验法则是：

- 如有疑问，完全不要使用 `MaybeOwned<Tensor>` ------ 特别是，优先避免在尚未使用它的代码中使用它。新的使用方式只应在能带来关键（且可证明的）性能提升时才被引入。
- 当修改或调用已经使用 `MaybeOwned<Tensor>` 的代码时，请记住，通过调用 `MaybeOwned<Tensor>::owned()` 从手头的 Tensor 生成一个 `MaybeOwned<Tensor>` 总是安全的。这可能会导致不必要的引用计数，但绝不会导致错误行为 ------ 因此这总是更安全的选择，除非你想要包装的 Tensor 的生命周期非常清晰。

更多细节和实现代码可以在 \<https://github.com/pytorch/pytorch/blob/main/c10/util/MaybeOwned.h\> 和 \<https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/templates/TensorBody.h\> 找到。
