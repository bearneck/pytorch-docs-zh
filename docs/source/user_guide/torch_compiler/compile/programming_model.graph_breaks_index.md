# 处理图中断

您可能还记得 [Dynamo 核心概念](programming_model.dynamo_core_concepts) 中提到，当 Dynamo 遇到无法追踪的代码时会执行图中断。在默认的 `torch.compile` 设置中，Dynamo 会编译到该点为止已确定的 FX 图，在常规 Python 中执行不受支持的代码，然后恢复追踪。

图中断使 Dynamo 能够追踪任意 Python 代码，并提取出可以单独优化的功能子图。

然而，图中断可能导致 `torch.compile` 出现意外的性能下降。如果您没有看到预期的加速效果，我们建议检查并移除图中断。

以下部分概述了处理图中断的策略。

- [Programming Model Fullgraph True](programming_model.fullgraph_true.md)
- [Programming Model Common Graph Breaks](programming_model.common_graph_breaks.md)
- [Programming Model Dynamo Nonstrict Trace](programming_model.dynamo_nonstrict_trace.md)
- [Programming Model Custom Ops](programming_model.custom_ops.md)
- [Programming Model Fullgraph False](programming_model.fullgraph_false.md)
