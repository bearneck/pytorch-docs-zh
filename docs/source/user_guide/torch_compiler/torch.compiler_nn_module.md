# PyTorch 2.0 NNModule 支持

**作者**: [Will Constable](https://github.com/wconstab)

`torch.compile` 对 torch.nn.Module 对象有特殊处理，其追踪方式与追踪任意 Python 类不同，目的是通过对结构做出假设来生成更快的代码。

本文档描述了由于这种特殊化而产生的一些权衡取舍或边界情况。

## NNModule Hooks 支持

此前，`torch.compile` 不支持 nn.Module 上的 hooks，如果注册了 hooks，它们在编译后的程序中会被简单地忽略。确实，许多用户根本不使用 nn.Module hooks，或者仅将其用于调试工作流，但将 nn.Module hooks 与 `torch.compile` 结合使用存在有效的用例。

通过 nn.Module.__call__ 实现来编排的 hooks 包括 `_forward_pre_hooks`、`forward_hooks`、`_backward_pre_hooks` 和 `_backward_hooks`，它们将被称为“调用 hooks”。`torch.compile` 部分支持这些 hooks，具体限制如下所述。

另一类 hooks 包括 `_state_dict_hooks` 及其 `pre` 和 `load_` 变体，`torch.compile` 目前仍不支持这些 hooks。

## `nn.Module.__call__` Hooks 使用与限制

默认情况下，`torch.compile` 会追踪 `nn.Module.__call__` 的内容，这意味着它会遇到并运行 forward/pre-forward hooks。如果在调用 `torch.compile` 之前安装了 hooks，并且之后不移除或更改这些 hooks，那么您的用例默认应该得到支持。

Backward/Pre-backward hooks 通常也受支持，但有类似的注意事项：目前 dynamo 在访问 backward_hooks 字典时会发生 graph-breaks，这或许可以通过一些工作来避免。Graph-breaks 也会影响 backward hooks 的触发时机，因为 graph-segments 是作为 autograd-functions 运行的，它们会同时产生所有梯度。假设 dynamo 能够不因 backward-hooks 的存在而发生 graph-break，我们仍然期望一系列模块的 backward hooks 在整个编译图的 backward 运行后一起触发。

**'allowed modules' 上的 hooks**
`torch.compile` 将常见模块（如 torch.conv）以及难以追踪的模块特殊对待，允许它们在 dynamo 图中不透明地调用，而不是由 dynamo 追踪进去。对于此类模块，hooks 目前会触发 graph-break，以便受影响的模块在 dynamo 之外运行。根据模型的不同，这可能会引入显著的性能回归，需要额外的工作来改进此支持。

**skip_nnmodule_hook_guards**
默认情况下，`torch._dynamo.config.skip_nnmodule_hook_guards` 设置为 True，这意味着不会在每个 nn.Module hook 字典上安装 guards，通过减少 guard 执行时间来改善运行时性能，代价是如果在编译后更改了任何 hook 字典，系统将无法察觉。

如果您希望在编译后能够移除或修改 hooks，并让 `torch.compile` 做出适当反应（通过重新编译），那么您需要设置 `skip_nnmodule_hook_guards=False`，并预期会因添加的 guards 而产生运行时性能损失。

TODO: 确认 backward/pre_backward hooks 是否正常工作并相应记录

## state_dict Hooks

`torch.compile` 目前尚未支持 state dict hooks。

TODO: 如果 hooks 导致 graph-breaking，则 warn_once。如果存在 hooks，则 warn_once 并指向本文档。
