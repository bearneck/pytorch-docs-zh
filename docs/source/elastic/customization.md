# 自定义

本节介绍如何自定义 TorchElastic 以满足您的需求。

## 启动器

TorchElastic 自带的启动器程序应能满足大多数用例（参见 `launcher-api`{.interpreted-text role="ref"}）。 您可以通过编程方式创建代理并为其传递工作进程的规格说明来实现自定义启动器，如下所示。

``` python
# my_launcher.py

if __name__ == "__main__":
  args = parse_args(sys.argv[1:])
  rdzv_handler = RendezvousHandler(...)
  spec = WorkerSpec(
      local_world_size=args.nproc_per_node,
      fn=trainer_entrypoint_fn,
      args=(trainer_entrypoint_fn args.fn_args,...),
      rdzv_handler=rdzv_handler,
      max_restarts=args.max_restarts,
      monitor_interval=args.monitor_interval,
  )

  agent = LocalElasticAgent(spec, start_method="spawn")
  try:
      run_result = agent.run()
      if run_result.is_failed():
          print(f"worker 0 failed with: run_result.failures[0]")
      else:
          print(f"worker 0 return value is: run_result.return_values[0]")
  except Exception ex:
      # 处理异常
```

## 集合处理器

要实现您自己的集合机制，请扩展 `torch.distributed.elastic.rendezvous.RendezvousHandler` 并实现其方法。

 warning
 title
Warning


集合处理器的实现较为复杂。在开始之前，请确保您完全理解集合的属性。更多信息请参考 `rendezvous-api`{.interpreted-text role="ref"}。


实现后，您可以在创建代理时将自定义的集合处理器传递给工作进程规格说明。

``` python
spec = WorkerSpec(
    rdzv_handler=MyRendezvousHandler(params),
    ...
)
elastic_agent = LocalElasticAgent(spec, start_method=start_method)
elastic_agent.run(spec.role)
```

## 指标处理器

TorchElastic 会发出平台级别的指标（参见 `metrics-api`{.interpreted-text role="ref"}）。 默认情况下，指标会输出到 [/dev/null]{.title-ref}，因此您将看不到它们。 若要将指标推送到您基础设施中的指标处理服务，请实现一个 [torch.distributed.elastic.metrics.MetricHandler]{.title-ref} 并在您的自定义启动器中 [configure]{.title-ref} 它。

``` python
# my_launcher.py

import torch.distributed.elastic.metrics as metrics

class MyMetricHandler(metrics.MetricHandler):
    def emit(self, metric_data: metrics.MetricData):
        # 将 metric_data 推送到您的指标接收器

def main():
  metrics.configure(MyMetricHandler())

  spec = WorkerSpec(...)
  agent = LocalElasticAgent(spec)
  agent.run()
```

## 事件处理器

TorchElastic 支持事件记录（参见 `events-api`{.interpreted-text role="ref"}）。 事件模块定义了允许您记录事件并实现自定义 EventHandler 的 API。EventHandler 用于将 torchelastic 执行期间产生的事件发布到不同的源，例如 AWS CloudWatch。 默认情况下，它使用 [torch.distributed.elastic.events.NullEventHandler]{.title-ref}，该处理器会忽略事件。要配置自定义事件处理器，您需要实现 [torch.distributed.elastic.events.EventHandler]{.title-ref} 接口，并在您的自定义启动器中 [configure]{.title-ref} 它。

``` python
# my_launcher.py

import torch.distributed.elastic.events as events

class MyEventHandler(events.EventHandler):
    def record(self, event: events.Event):
        # 处理事件

def main():
  events.configure(MyEventHandler())

  spec = WorkerSpec(...)
  agent = LocalElasticAgent(spec)
  agent.run()
```
