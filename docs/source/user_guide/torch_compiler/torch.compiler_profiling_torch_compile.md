# 使用 torch.compile 性能分析

## torch.profiler 的用途：

torch.profiler 有助于在核函数级别粒度上理解程序的性能——例如，它可以在程序级别显示图中断和资源利用率。分析器提供的数据通常能帮助用户了解应进一步调查哪些方面以理解模型性能。

要理解核函数级别的性能，可以使用其他工具，例如 [Nvidia Nsight compute 工具](https://developer.nvidia.com/nsight-compute)、[AMD Omnitrace](https://rocm.docs.amd.com/projects/omnitrace/en/latest/)、Intel® VTune™ Profiler 或 [inductor 的分析工具](https://docs.pytorch.org/docs/stable/torch.compiler_inductor_profiling.html#torchinductor-gpu-profiling)。

另请参阅 [通用 PyTorch 分析器指南](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。

## 使用 torch.profiler 和查看追踪的基础知识

**示例程序**：我们将使用这个分析 resnet18 的示例。注意此示例程序的以下部分：

* 包含预热运行以等待编译完成（这将预热诸如 CUDA 缓存分配器之类的系统）
* 使用 `torch.profiler.profile()` 上下文来分析我们感兴趣的部分
* 使用 `prof.export_chrome_trace("trace.json")` 导出分析产物。

```python

    import torch
    from torchvision.models import resnet18

    device = 'cuda'      # or 'cpu', 'xpu', etc.
    model = resnet18().to(device)

    inputs = [torch.randn((5, 3, 224, 224), device=device) for _ in range(10)]

    model_c = torch.compile(model)

    def fwd_bwd(inp):
        out = model_c(inp)
        out.sum().backward()

    # warm up
    fwd_bwd(inputs[0])

    with torch.profiler.profile() as prof:
        for i in range(1, 4):
            fwd_bwd(inputs[i])
            prof.step()

    prof.export_chrome_trace("trace.json")
```

**查看 Chrome 追踪**：在 Chrome 浏览器中，打开 chrome://tracing 并加载 json 文件。使用“w”和“s”键进行放大和缩小，使用“a”和“d”键左右滚动。“?”将显示包含快捷键列表的“帮助”屏幕。

```{figure} ../../_static/img/profiling_torch_compile/basic_chrome_trace.png
:alt: 在 chrome://tracing 查看器中可视化的基础 Chrome 追踪示例
```

在这里，我们观察到：
* CompiledFunction 和 CompiledFunctionBackward 事件，它们对应于 dynamo 编译的区域。
* 顶部的 CPU 事件和底部的 GPU 事件。

**CPU 与加速器事件之间的流**

加速器上的每个内核都是在 CPU 上运行的代码启动后发生的。分析器可以在加速器和 CPU 事件之间绘制连接（即“流”）以显示哪个 CPU 事件启动了加速器内核。这特别有帮助，因为除了少数例外，加速器内核是异步启动的。

要查看流连接，请单击 GPU 内核并单击“ac2g”：

```{figure}  ../../_static/img/profiling_torch_compile/ac2g.png
:alt: 在 chrome://trace 查看器中的可视化，显示内核与其启动位置之间的异步流。
```

或者，使用顶部的“Flow events”下拉菜单打开*所有*流。

## 解决 CUDA Graph 分析问题

当启用 CUDA 图时，某些 CUDA 配置（驱动程序版本低于 525.85.12 或 CUDA < 12）可能会在分析工具和 CUDA 图之间遇到问题。要解决这些问题，请在程序顶部添加一个空的分析上下文：

```python

    import torch

    torch.profiler._utils._init_for_cuda_graphs()

    # ... rest of program
```

## 理解编译时间

要理解为什么编译花费很长时间，可以分析首次调用 torch.compile 程序的执行情况。请注意，编译的分析追踪可能比典型分析失真更严重，因为编译工作负载可能与典型的 PyTorch 工作负载有很大不同。在某些情况下，追踪文件也可能非常大。大于 1GB 的追踪文件可能难以用 Chrome 追踪工具打开。

注意：大致相同的信息也可以通过非图形格式使用 :code:`torch._dynamo.utils.compile_times()` 获得。此实用程序不会显示编译步骤何时发生，但会显示每个步骤花费的时间——并且时间不会受到任何分析开销的影响。

请参见以下示例：

```python

    import torch
    from torchvision.models import resnet18

    # user can switch between cuda and xpu
    device = 'cuda'
    model = resnet18().to(device)
    inputs = [torch.randn((5, 3, 224, 224), device=device) for _ in range(10)]

    model_c = torch.compile(model)

    def fwd_bwd(inp):
        out = model_c(inp)
        out.sum().backward()

    def warmup_compile():
        def fn(x):
            return x.sin().relu()

        x = torch.rand((2, 2), device=device, requires_grad=True)
        fn_c = torch.compile(fn)
        out = fn_c(x)
        out.sum().backward()

    with torch.profiler.profile() as prof:
        with torch.profiler.record_function("warmup compile"):
            warmup_compile()

        with torch.profiler.record_function("resnet18 compile"):
            fwd_bwd(inputs[0])

    prof.export_chrome_trace("trace_compile.json")
```

```{figure} ../../_static/img/profiling_torch_compile/compilation_profiling.png
:alt: 在 chrome://trace 查看器中的可视化，显示 dynamo 和 inductor 编译步骤
```

注意几点：

* 第一次调用应*在*分析期间发生，以便捕获编译过程
* 添加预热编译以初始化任何需要延迟初始化的系统。

# 查找图中断："Torch-Compiled Region" 和 "CompiledFunction"

尽管有用于识别图中断的日志记录工具，但分析器提供了一种快速的可视化方法来识别 :ref:`图中断 <torch.compiler_graph_breaks>`。有两个分析器事件需要注意：**Torch-Compiled Region** 和 **CompiledFunction**。

**Torch-Compiled Region** - 在 PyTorch 2.2 中引入 - 是一个覆盖整个编译区域的分析器事件。图中断（graph break）的表现几乎总是一样的：嵌套的 "Torch-Compiled Region" 事件。从 PyTorch 2.5 开始，分析器事件还将包含帧 ID 和帧编译 ID。帧 ID 是帧的唯一标识符，帧编译 ID 表示该帧已被编译的次数。

如果你运行两个独立的函数，并且每个函数都独立应用了 `torch.compile()`，通常应该会看到两个相邻（即**不是**堆叠/嵌套）的 Torch-Compiled 区域。同时，如果你遇到图中断（或 `disable()`/跳过的区域），则会出现嵌套的 "Torch-Compiled Region" 事件。

**CompiledFunction** - 在 PyTorch 2.0 中引入 - 是一个分析器事件，当任何输入需要梯度时会出现。每个图中断都会打断一个 CompiledFunction 块，将其一分为二。CompiledFunction 事件仅在涉及 Autograd 时出现，即图中某些输入张量的 `requires_grad=True`。

当跟踪中出现 CompiledFunction 事件时，通常在反向传播过程中会有一个 CompiledFunctionBackward 事件与之配对。如果调用了反向函数，跟踪中应该会出现一个 "fwd-bwd link" 连接这两者。

如果你的用例包含一个不需要梯度且不包含 "Torch-Compiled Region" 事件的图，可能更难判断 `torch.compile` 是否正确应用。一个线索可以是 Inductor 生成的 Triton 内核的存在。

请参见下面的合成示例进行演示：

```python

    import torch
    import torch._dynamo
    # user can switch between cuda and xpu
    device = 'cuda'

    class ModelWithBreaks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            def create_sequential():
                return torch.nn.Sequential(
                    torch.nn.Linear(128, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),
                    torch.nn.ReLU(),
                )
            self.mod1 = create_sequential()
            self.mod2 = create_sequential()
            self.mod3 = create_sequential()
            self.mod4 = create_sequential()

        def forward(self, inp):
            mod1 = self.mod1(inp)
            torch._dynamo.graph_break()
            mod2 = self.mod2(mod1)
            torch._dynamo.graph_break()
            mod3 = self.mod3(mod2)
            torch._dynamo.graph_break()
            mod4 = self.mod4(mod3)
            return mod4

    model = ModelWithBreaks().to(device)
    inputs = [torch.randn((128, 128), device=device) for _ in range(10)]

    model_c = torch.compile(model)

    def fwd_bwd(inp):
        out = model_c(inp)
        out.sum().backward()

    # warm up
    fwd_bwd(inputs[0])

    with torch.profiler.profile() as prof:
        for i in range(1, 4):
            fwd_bwd(inputs[i])
            prof.step()

    prof.export_chrome_trace("trace_break.json")
```

```{figure} ../../_static/img/profiling_torch_compile/graph_breaks_with_torch_compiled_region.png
:alt: 在 chrome://trace 查看器中的可视化，显示嵌套的 Torch-Compiled Region 事件和多个 CompiledFunction 事件 - 表明存在图中断。
```

## 算子内核

当一个算子被启动时，我们期望看到几个事件：

1. CPU 端事件
2. 内核启动（如果处理的是 GPU 内核）
3. GPU 端事件

```{figure} ../../_static/img/profiling_torch_compile/kernel_launch_labeled.png
:alt: 在 chrome://trace 查看器中的可视化，显示三种类型的事件 - CPU 端事件、内核启动和 GPU 端事件
```

**Inductor 生成的 Triton 内核：**
1. **CPU 端事件** 应显示为以 "triton\_" 为前缀的事件。这些事件目前信息量很少 - 只有内核名称和启动信息，但比典型的 aten 内核启动（包含输入形状、类型等）的信息要少。
2. **内核启动** 应显示为 cuLaunchKernel 而不是 cudaLaunchKernel（cudaLaunchKernel 是 aten 算子的典型情况）
3. **GPU 端事件** 应该出现，其名称的描述性取决于 inductor 配置中的 `unique_kernel_names`

```{figure} ../../_static/img/profiling_torch_compile/triton_kernel_launch.png
```

**非 Inductor 生成的 Triton 内核：**

1. **CPU 端** 事件可能不会出现在跟踪中；自动插入分析器事件的机制目前是在 Inductor 层面实现的，因此绕过 Inductor 的 Triton 内核可能不会出现在跟踪中，除非用户手动标注了它们。
2. **内核启动** 应显示为 cuLaunchKernel 而不是 cudaLaunchKernel（cudaLaunchKernel 是 aten 算子的典型情况）
3. **GPU 端** 事件应该出现，其命名方式与编写的 triton 内核类似。

```{figure} ../../_static/img/profiling_torch_compile/noninductor_triton_kernel.png
```

**Inductor 生成的 CPU 内核：**

1. **CPU 端事件** 不会出现在跟踪中；我们尚未为此添加分析功能。
2. **内核启动** 和 **GPU 端事件** 不存在

**非 Triton 内核**（即 aten 内核或自定义算子）有时也可能会出现在跟踪中。有时，Inductor 会回退到原始算子的实现，在这种情况下，你会看到对 aten 算子的调用。

## 启动开销

一个常见的问题是 GPU 利用率低。一个快速的识别方法是查看 GPU 上的内核之间是否存在大的间隙：

```{figure} ../../_static/img/profiling_torch_compile/cpu_bound.png
:alt: 在 chrome://trace 查看器中的可视化，显示 GPU 内核之间存在大的间隙。这表明模型受 CPU 限制，很可能是由于内核启动期间的开销造成的。
```

这通常是 CPU 开销的结果，例如，如果内核启动之间在 CPU 上花费的时间大于 GPU 处理内核所花费的时间。这个问题在小批量大小的情况下更常见。

在使用 inductor 时，当启动开销成为关注点时，启用 CUDA 图通常有助于提升性能。