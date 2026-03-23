
# torch.compile 编程模型

`torch.compile` 编程模型：
1. 阐明 `torch.compile` 的一些内部行为，以便用户能更好地预测编译器对代码的处理方式；
2. 提供更细粒度控制 `torch.compile` 的方法。

通过理解 `torch.compile` 编程模型，用户可以在遇到 `torch.compile` 相关问题时，系统性地找到解决方案。

```{toctree}
programming_model.dynamo_core_concepts
programming_model.graph_breaks_index
programming_model.non_strict_tracing_model
programming_model.recompilation
programming_model.observability
programming_model.reporting_issues
```