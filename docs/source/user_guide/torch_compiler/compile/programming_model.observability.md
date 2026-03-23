# tlparse / TORCH_TRACE

tlparse / `TORCH_TRACE` 是一对工具，用于生成类似[这样](https://web.mit.edu/~ezyang/Public/bhack-20240609-tlparse/index.html)的编译报告。

收集追踪信息相当简单。要收集追踪信息，请按如下方式运行您的模型：

```bash
TORCH_TRACE="/tmp/tracedir" python foo.py
pip install tlparse
tlparse /tmp/tracedir --latest
```

`--latest` 标志会处理目录中最新的日志文件。您也可以使用 `tlparse <log_file>` 处理特定的日志文件。

默认情况下，输出存储在 `tl_out` 文件夹中。您也可以使用 `-o my_folder` 指定输出文件夹。

即使您正在运行分布式作业，此方法也适用，它会为每个 rank 提供追踪信息。
它将在您的浏览器中打开类似于上面生成的 HTML 文件。
如果您正在为一个没有独立复现的复杂问题提交错误报告，您仍然可以通过以下方式极大地帮助 PyTorch 开发者：

1) 附上在 `/tmp/tracedir` 中生成的追踪日志，或者
2) 附上一个**包含 tlparse 所有输出的 zip 文件**（例如 `tl_out` 中的所有文件）。请不要只附上 index.html 文件，因为它只包含输出文件的目录，而不是真正的输出内容。

```{warning}
追踪日志包含您所有的模型代码。
如果您正在处理的模型是敏感的，请不要分享追踪日志。追踪日志**不包含**权重。
```

```{raw} html
    <style>
        .red {background-color:#ff0000;}
        .green {background-color:#00ff00;}
        .dark-green {background-color:#027f02;}
    </style>
```

```{eval-rst}
.. role:: red
.. role:: green
.. role:: dark-green
```

`tlparse` 的输出主要面向 PyTorch 开发者，其日志格式便于在 GitHub 上上传和分享。
然而，作为非 PyTorch 开发者，您仍然可以从中提取有用的信息。
我们建议从报告中的内联帮助文本开始，它解释了报告的内容。
以下是一些您可以从 `tlparse` 中获得的信息：

- 通过查看堆栈字典树，了解哪些模型代码被编译了？
  如果您不熟悉正在编译的代码库，这尤其有用！
- 存在多少个图中断 / 不同的编译区域？
  （每个不同的编译都是一个颜色编码的块，如 {dark-green}`[0/0]`）。
  可能发生图中断的帧显示为浅绿色 {green}`[2/4]`。
  如果有很多帧，这是可疑的，表明您可能遇到了灾难性的图中断，或者您的代码不太适合 `torch.compile`。
- 我重新编译特定帧了多少次？频繁重新编译的帧看起来像：
  {dark-green}`[10/0]` {dark-green}`[10/1]` {dark-green}`[10/2]`
  \- 如果某个东西被频繁重新编译，这非常可疑，值得调查，即使它不是您问题的根本原因。
- 是否存在编译错误？出错的帧看起来像 {red}`[0/1]`。
- 我为给定帧生成了哪些中间编译器产物？
  例如，您可以查看生成的高级 FX 图或生成的 Triton 代码。
- 特定帧是否有相关信息？您可以在 `compilation_metrics` 中找到这些信息。

以下是一些文件名及其描述。根据您的具体程序，您可能不会看到所有这些文件。

```{eval-rst}
.. list-table::
    :widths: 25 50
    :header-rows: 1

    * - 文件名
      - 描述
    * - dynamo_output_graph
      - 来自 Dynamo 前端图捕获的输出图
    * - before_pre_grad_graph
      - 运行任何预自动微分图传递之前的 FX 图
    * - after_pre_grad_graph
      - 运行所有预自动微分图传递之后的 FX 图
    * - aot_autograd_cache_miss / aot_autograd_cache_hit
      - aot_autograd_cache 的缓存键，以及我们是否发生了缓存未命中或命中
    * - aot_inference_graph
      - 当不需要自动微分时（例如，没有张量需要梯度），分解后的 FX 图。
    * - aot_joint_graph
      - 经过自动微分和分解后的联合前向-反向图
    * - aot_forward_graph
      - 分割 aot_joint_graph 后的前向图
    * - aot_backward_graph
      - 分割 aot_joint_graph 后的反向图
    * - before_joint_graph
      - 运行任何联合图传递之前的 FX 图
    * - after_joint_graph
      - 运行所有联合图传递之后的 FX 图
    * - before_post_grad_graph
      - 运行任何后自动微分图传递之前的 FX 图
    * - inductor_post_grad_graph
      - 运行所有后自动微分图传递之后的 FX 图
    * - fx_graph_runnable
      - 与 before_post_grad_graph 基本相同的图，但它是一个可运行的 Python 脚本。它还包含 torch 配置和一些包装代码，因此您可以使用虚拟输入运行该图。
    * - inductor_output_code
      - 由 Inductor 生成的代码
    * - fx_graph_cache_miss/ fx_graph_cache_hit
      - fx 图缓存的缓存键，以及我们是否发生了缓存未命中或命中
    * - dynamo_cpp_guards_str
      - 来自 dynamo 的守卫信息
```


## TORCH_LOGS

您可以使用 `TORCH_LOGS` 环境变量选择性地启用 `torch.compile` 堆栈的某些部分进行日志记录。
`TORCH_LOGS` 实际上是 `tlparse` 的日志来源。`TORCH_LOGS` 环境变量的格式如下：

```bash
TORCH_LOGS="<option1>,<option2>,..." python foo.py
```

您也可以使用 `torch._logging.set_logs` 以编程方式设置日志选项：

```python
import logging
torch._logging.set_logs(graph_breaks=True, dynamic=logging.DEBUG)
```

最有用的选项是：

- `graph_breaks`：记录用户代码中图中断的位置以及图中断的原因
- `guards`：记录生成的守卫
- `recompiles`：记录哪些函数重新编译以及导致重新编译的失败守卫
- `dynamic`：与动态形状相关的日志
- `output_code`：记录由 Inductor 生成的代码

一些更有用的 `TORCH_LOGS` 选项包括：

```{eval-rst}
.. list-table::
    :widths: 25 50
    :header-rows: 1

    * - Option
      - Description
    * - +all
      - 输出所有 ``torch.compile`` 组件的调试日志
    * - +dynamo
      - 输出 TorchDynamo 的调试日志
    * - +aot
      - 输出 AOTAutograd 的调试日志
    * - +inductor
      - 输出 TorchInductor 的调试日志
    * - dynamic
      - 输出动态形状相关的日志
    * - graph_code
      - 输出 Dynamo 生成的 FX 图的 Python 代码
    * - graph_sizes
      - 输出 Dynamo 生成的 FX 图的张量尺寸
    * - trace_bytecode
      - 输出 Dynamo 正在追踪的字节码指令以及 Dynamo 正在跟踪的符号解释器堆栈
    * - trace_source
      - 输出 Dynamo 当前正在追踪的原始源代码行
    * - bytecode
      - 输出 Dynamo 生成的字节码
    * - guards
      - 输出生成的守卫
    * - recompiles
      - 输出重新编译的原因（仅输出第一个失败的守卫检查）
    * - recompiles_verbose
      - 输出重新编译发生时所有失败的守卫检查
    * - aot_graphs
      - 输出 AOTAutograd 生成的图
    * - aot_joint_graphs
      - 输出 AOTAutograd 生成的联合前向-反向图
    * - output_code
      - 输出 Inductor 生成的代码
    * - kernel_code
      - 按内核输出 Inductor 生成的代码
    * - schedule
      - 输出 Inductor 调度日志
    * - perf_hints
      - 输出 Inductor 性能提示日志
    * - fusion
      - 输出 Inductor 融合日志
```

有关完整选项列表，请参阅 [torch.\_logging](https://pytorch.org/docs/stable/logging.html)
和 [torch.\_logging.set_logs](https://pytorch.org/docs/stable/generated/torch._logging.set_logs.html#torch._logging.set_logs)。

## tlparse 与 TORCH_LOGS

通常，我们建议在遇到问题时首先使用 `tlparse`。
`tlparse` 非常适合调试大型模型，并获取模型编译过程的高级概览。
另一方面，当我们已经知道是哪个 `torch.compile` 组件导致问题时，`TORCH_LOGS` 更适合用于小型示例和细粒度的调试细节。