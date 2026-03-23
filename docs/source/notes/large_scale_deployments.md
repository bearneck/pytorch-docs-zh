# 大规模部署特性


local

:   

本文档讨论在大型系统中运行 PyTorch 或在大规模组织中使用 PyTorch 操作多个系统时，可能有用的一些扩展点和技巧。

本文档假设您所在组织从源代码构建 PyTorch，或者能够静态链接在 PyTorch 使用时加载的额外代码。因此，许多钩子都作为 C++ API 暴露，可以在集中位置（例如在静态初始化代码中）一次性触发。

## 全范围算子性能分析

PyTorch 内置了 `torch.autograd.profiler`，能够按需测量单个算子的执行时间。您可以使用相同的机制对运行 PyTorch 的任何进程进行\"始终开启\"的测量。这对于收集在给定进程或整个机器集群中运行的 PyTorch 工作负载信息非常有用。

可以通过 `torch::addGlobalCallback` 添加新的算子调用回调。钩子函数将接收描述调用上下文（例如 [name]）的 `torch::RecordFunction` 结构体。如果启用，`RecordFunction::inputs()` 包含以 `torch::IValue` 变体类型表示的函数参数。请注意，输入日志记录的开销相对较大，因此需要显式启用。

算子回调还可以访问 `c10::ThreadLocalDebugInfo::get()` 接口，该接口返回指向包含调试信息的结构体的指针。此调试信息可以通过使用 `at::DebugInfoGuard` 对象提前设置。调试信息会通过前向传播（包括异步 `fork` 任务）和后向传播传递，对于将有关执行环境（例如模型 ID）的额外信息从应用程序的高层传递到算子回调非常有用。

调用回调会增加一些开销，因此通常只需随机采样算子调用即可。这可以通过向 `torch::addGlobalCallback` 传递可选采样率来基于每个回调启用。

请注意，`addGlobalCallback` 不是线程安全的，只能在没有任何 PyTorch 算子运行时调用。通常，在初始化期间调用它们是个好主意。

以下是一个示例：

``` cpp
// 在程序开始处某处调用
void init() {
    // 随机采样百分之一的算子运行
    addGlobalCallback(
      RecordFunctionCallback(
        &onFunctionEnter,
        &onFunctionExit)
      .needsInputs(true)
      .samplingProb(0.01)
    );
    // 注意：要在模型调用线程中启用观察器，
    // 需在运行模型前在该线程中调用 enableRecordFunction()
}

void onFunctionEnter(const RecordFunction& fn) {
    std::cerr << "Before function " << fn.name()
              << " with " << fn.inputs().size() << " inputs" << std::endl;
}

void onFunctionExit(const RecordFunction& fn) {
    std::cerr << "After function " << fn.name();
}
```

## API 使用情况日志记录

在更广泛的生态系统中运行时，例如在托管作业调度器中，跟踪哪些二进制文件调用特定 PyTorch API 通常很有用。PyTorch 在几个重要的 API 点注入了简单的检测机制，会触发给定的回调。由于 PyTorch 通常在一次性 Python 脚本中调用，因此对于每个 API，回调在每个进程中仅触发一次。

`c10::SetAPIUsageHandler` 可用于注册 API 使用情况检测处理器。传递的参数将是一个标识使用点的\"api key\"，例如 PyTorch 扩展导入的 `python.import`。

``` cpp
SetAPIUsageLogger([](const std::string& event_name) {
    std::cerr << "API was used: " << event_name << std::endl;
});
```

给开发者的说明：可以在代码中使用 C++ 的 `C10_LOG_API_USAGE_ONCE("my_api")` 或 Python 的 `torch._C._log_api_usage_once("my.api")` 添加新的 API 触发点。

## 常见扩展点

PyTorch API 通常是松散耦合的，很容易用专门版本替换组件。常见的扩展点包括：

- 用 C++ 实现的自定义算子 - 详见 [教程](https://pytorch.org/tutorials/advanced/cpp_extension.html)。
- 自定义数据读取通常可以通过调用相应的 Python 库直接集成。可以通过扩展 `torch.utils.data.Dataset` 或 `torch.utils.data.IterableDataset` 来利用 `torch.utils.data` 的现有功能。
