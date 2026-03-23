
# Backed 与 Unbacked Symints

Backed `SymInts` 是具有具体值或关联"提示"的符号整数。这意味着 torch 可以使用这些值来进行控制流决策，例如确定要执行的代码分支。它们通常源自大小或值已知或可推断的操作。

Unbacked `SymInts` 是没有具体值或提示的符号整数。它们通常出现在数据依赖的操作中，例如 `.nonzero()` 或 `.item()`，这些操作的大小或值在编译时无法确定。由于缺乏具体值，它们不能用于控制流决策，尝试这样做会导致图中断。

Unbacked `SymInts` 使用*无感知大小推理*，这在处理 `0/1 特化重编译问题 <zero-one-specialization>` 时特别有用。

总之，backed `SymInts` 具有可用于决策的已知值，而 unbacked `SymInts` 则没有，需要特殊处理以避免图中断。

Unbacked 符号整数可能限制性过强，导致大多数 PyTorch 程序失败。为了解决这个问题，您可以使用以下方法和 API 作为变通方案：

* 使用更高级的 API，如 `empty` 而不是 `empty_strided` 来创建张量。这确保了张量是非重叠且密集的，避免了不必要的步幅排序和守卫创建，从而避免这些属性的不必要重新计算。

* 修改您的代码，使预计算的属性变为*惰性*。这确保了对 unbacked 符号整数的守卫仅在必要时应用，减少了计算开销。

## 如何使用 unbacked
要使用 unbacked API，请将 `mark_dynamic` 替换为 `mark_unbacked`，并将 `TORCH_COMPILE_DYNAMIC_SOURCES` 替换为 `TORCH_COMPILE_UNBACKED_SOURCES`。这会告诉编译器将输入视为 unbacked。

```{seealso}
* `dynamic_shapes`
* `torch.export`
* `what_is_a_specialization`
```