# 使用 `fullgraph=False`
虽然 `fullgraph=False` 是 `torch.compile` 的默认设置，但在遇到图中断时恢复编译的语义更为复杂。
您可以在以下小节中找到关于 `fullgraph=False` 语义的详细信息。

使用 `torch.compile(fullgraph=False)` 的策略如下：

1. [确定放置 `torch.compile` 的理想位置](programming_model.where_to_apply_compile)。通常，这是不会导致过多图中断的最高层函数。
   执行大量预处理或 I/O 操作的函数就是会导致许多图中断且无法从 `torch.compile` 中显著受益的示例。
   a. 您可以通过先编译单个函数/模块，再编译整个模型来隔离问题。
2. [对编译区域内会导致大量图中断且无法从编译中受益的函数应用 `torch.compiler.disable`](programming_model.compiler_disable)。在这种情况下，一个图中断比可能数十或数百个更好。
3. [使用 `TORCH_LOGS="graph_breaks"` 或 tlparse 来调查剩余的图中断。](programming_model.observability)
   使用与在 `fullgraph=True` 编程模型下处理图中断相同的方法来解决这些图中断。并非所有图中断都需要移除——有些可能比其他图中断对性能的影响更大。一般规则是关注发生在模型计算期间的图中断。
   a. 我们建议在调试图中断时使用 `torch.compile(backend='eager')`，以获得更快的调试迭代周期。

```{toctree}
programming_model.where_to_apply_compile
programming_model.compiler_disable
programming_model.error_on_graph_break
programming_model.nested_graph_breaks
programming_model.skipped_functions
```