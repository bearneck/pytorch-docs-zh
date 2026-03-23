
# PYTORCH ProcessGroupNCCL 环境变量

有关环境变量的更多信息，请参阅 [ProcessGroupNCCL Environment Variables](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp)。

```{list-table}
:header-rows: 1

* - **变量**
  - **描述**
* - ``TORCH_NCCL_ASYNC_ERROR_HANDLING``
  - 控制在 watchdog 中观察到异常时，如何处理 NCCL 的异步错误。如果设置为 0，则不处理异步 NCCL 错误。如果设置为 1，则在出错时中止 NCCL 通信器并终止进程。如果设置为 2，则仅中止 NCCL 通信器；如果设置为 3，则在不中止 NCCL 通信器的情况下终止进程。默认值为 3。
* - ``TORCH_NCCL_HIGH_PRIORITY``
  - 控制是否为 NCCL 通信器使用高优先级流。
* - ``TORCH_NCCL_BLOCKING_WAIT``
  - 控制 wait() 是阻塞式还是非阻塞式。
* - ``TORCH_NCCL_DUMP_ON_TIMEOUT``
  - 控制在检测到 watchdog 超时或异常时是否转储调试信息。此变量必须与 TORCH_NCCL_TRACE_BUFFER_SIZE 大于 0 一起设置。
* - ``TORCH_NCCL_DESYNC_DEBUG``
  - 控制是否启用 Desync Debug。这有助于找出导致集合操作不同步的罪魁祸首 rank。
* - ``TORCH_NCCL_ENABLE_TIMING``
  - 如果设置为 ``1``，则为所有 ProcessGroupNCCL 集合操作启用记录开始事件，并计算每个集合操作的精确计时。
* - ``TORCH_NCCL_ENABLE_MONITORING``
  - 如果设置为 ``1``，则启用监控线程，当 ProcessGroupNCCL Watchdog 线程卡住且在 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC 后未检测到心跳时，该线程将中止进程。这可能由于调用可能挂起的 CUDA/NCCL API 而发生。这有助于防止作业不必要地长时间卡住，从而占用集群资源。
* - ``TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC``
  - 控制 watchdog 心跳超时时间，超过此时间后监控线程将中止进程。
* - ``TORCH_NCCL_TRACE_BUFFER_SIZE``
  - 飞行记录器环形缓冲区中存储的最大事件数。例如，一个事件可能是一个集合操作的开始或结束。设置为 0 以禁用 tracebuffer 和调试信息转储。
* - ``TORCH_NCCL_TRACE_CPP_STACK``
  - 是否为飞行记录器收集 cpp 堆栈跟踪。默认值为 False。
* - ``TORCH_NCCL_COORD_CHECK_MILSEC``
  - 控制监控线程内部检查来自其他 ranks 的协调信号（例如，转储调试信息）的间隔。默认值为 1000 毫秒。
* - ``TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC``
  - 控制在退出并抛出超时异常之前，为转储调试信息额外等待多长时间。
* - ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE``
  - 调试信息将被转储到的文件。
* - ``TORCH_NCCL_DEBUG_INFO_PIPE_FILE``
  - 用于手动触发调试转储的管道文件，向管道写入任何内容都会触发转储。
* - ``TORCH_NCCL_NAN_CHECK``
  - 控制是否对输入启用 NAN 检查，如果检测到 NAN 将抛出错误。
```