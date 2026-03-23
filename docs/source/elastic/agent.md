# 弹性代理


torch.distributed.elastic.agent


## 服务器


torch.distributed.elastic.agent.server

下图展示了一个管理本地工作进程组的代理。

![image](agent_diagram.jpg)

## 概念

本节描述了理解 [agent] 在 torchelastic 中作用所需的高级类和概念。


ElasticAgent


WorkerSpec


WorkerState


Worker


WorkerGroup

## 实现

以下是 torchelastic 提供的代理实现。


LocalElasticAgent

## 扩展代理

要扩展代理，你可以直接实现 `ElasticAgent`，但我们建议你改为扩展 `SimpleElasticAgent`，它提供了大部分框架，只留下几个特定的抽象方法需要实现。


SimpleElasticAgent


torch.distributed.elastic.agent.server.api.RunResult

## 代理中的看门狗

如果在 `LocalElasticAgent` 进程中定义了值为 1 的环境变量 `TORCHELASTIC_ENABLE_FILE_TIMER`，则可以在 `LocalElasticAgent` 中启用基于命名管道的看门狗。 可选地，可以设置另一个环境变量 `TORCHELASTIC_TIMER_FILE`，为其指定一个唯一的命名管道文件名。如果未设置环境变量 `TORCHELASTIC_TIMER_FILE`，`LocalElasticAgent` 将在内部创建一个唯一的文件名，并将其设置为环境变量 `TORCHELASTIC_TIMER_FILE`，此环境变量将传播到工作进程，使它们能够连接到 `LocalElasticAgent` 使用的同一命名管道。

## 健康检查服务器

如果在 `LocalElasticAgent` 进程中定义了环境变量 `TORCHELASTIC_HEALTH_CHECK_PORT`，则可以在 `LocalElasticAgent` 中启用健康检查监控服务器。 通过添加健康检查服务器的接口，可以在指定端口号上启动 tcp/http 服务器来扩展该功能。 此外，健康检查服务器将具有回调功能来检查看门狗是否存活。


torch.distributed.elastic.agent.server.health_check_server


HealthCheckServer


torch.distributed.elastic.agent.server.health_check_server.create_healthcheck_server
