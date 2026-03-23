# torch.monitor

统计接口设计用于跟踪高级别指标，这些指标会定期记录输出，用于监控系统性能。由于统计信息会以特定窗口大小进行聚合，您可以在关键循环中记录它们，而对性能的影响最小。

对于更不频繁的事件或值，例如损失、准确率、使用情况跟踪，可以直接使用事件接口。


## API 参考


| API | 说明 |
|-----|------|
| `torch.monitor.log_event` | — |
| `torch.monitor.register_event_handler` | — |
| `torch.monitor.unregister_event_handler` | — |

**类**


| 类 | 说明 |
|-----|------|
| `torch.monitor.Aggregation` | — |
| `torch.monitor.Stat` | — |
| `torch.monitor.data_value_t` | — |
| `torch.monitor.Event` | — |
| `torch.monitor.EventHandlerHandle` | — |
| `torch.monitor.TensorboardEventHandler` | — |
