# TorchInductor GPU 性能分析

本节列出了有助于深入分析模型在 TorchInductor 中性能的有用命令和工作流程。当模型运行速度未达预期时，您可能需要检查模型的各个内核。通常，那些占用 GPU 时间大部分的内核是最值得关注的。之后，您可能还想直接运行单个内核并检查其性能。PyTorch 提供了覆盖上述所有内容的工具。

## 相关环境变量

您可以在分析中使用以下环境变量：

- ``TORCHINDUCTOR_UNIQUE_KERNEL_NAMES``

   - 默认情况下，TorchInductor 将 Triton 内核命名为 ``'triton_'``。启用此环境变量后，inductor 会在跟踪中生成更具意义的 kernel 名称，例如 ``triton_poi_fused_cat_155``，其中包含内核类别（``poi`` 表示 pointwise）和原始的 ATen 算子。默认情况下禁用此配置以提高编译缓存命中的机会。

- ``TORCHINDUCTOR_BENCHMARK_KERNEL``

   - 启用此选项将使 inductor 代码生成工具对单个 triton 内核进行基准测试。

- ``TORCHINDUCTOR_MAX_AUTOTUNE``

   - Inductor 自动调优器将基准测试更多的 ``triton.Configs`` 并选择性能最佳的结果。这将增加编译时间，以期提高性能。

## 分解模型 GPU 时间

以下是将模型的执行时间分解为各个内核的步骤。我们以 ``mixnet_l`` 为例。

1. 运行模型的基准测试脚本：

   ```bash
      TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 TORCHINDUCTOR_BENCHMARK_KERNEL=1
      python -u benchmarks/dynamo/timm_models.py –backend inductor –amp
      –performance –dashboard –only mixnet_l –disable-cudagraphs –training
   ```
   ```{note}
   该工具依赖内核名称来决定其类别。启用 ``TORCHINDUCTOR_UNIQUE_KERNEL_NAMES`` 对此至关重要。
   ```
2. 在输出日志中，查找以下行：

   ```bash
      **Compiled module path:
      /tmp/torchinductor_shunting/qz/cqz7hvhood7y3psp7fy6msjxsxyli7qiwiybizdwtjw6ffyq5wwd.py**
   ```

每个已编译模块对应一行。如果没有额外的图中断，我们会在日志中看到 2 个这样的行，一个用于前向图，一个用于反向图。

对于我们的示例命令，我们分别获得前向图和反向图的以下已编译模块：

-  [前向图已编译模块](https://gist.github.com/shunting314/c2a4d8a28b00fcb5586d0e9d9bf77f9f)
-  [反向图已编译模块](https://gist.github.com/shunting314/48efc83b12ec3ead950052e4a0220b10)

3. 现在我们可以深入分析每个已编译模块的性能。为了说明，我们选择前向图的模块。为了方便，我将其命名为 ``fwd.py``。使用 ``-p`` 参数直接运行它：

   ```bash
      **> python fwd.py -p**
   ```

在此 [示例 gist](https://gist.github.com/shunting314/8243734a38b5733ea78479209c0ae893) 中查看完整的输出日志。

在输出中，您可以注意到以下内容：

* 我们为性能分析写入了一个 chrome 跟踪文件，以便我们可以加载跟踪并与之交互。在日志中，查找以下行以找到跟踪文件的路径。

  **Chrome trace for the profile is written to /tmp/compiled_module_profile.json**

   将跟踪加载到 Chrome 中（在 chrome 浏览器中访问 chrome://tracing 并按 UI 提示加载文件）将显示如下 UI：

   ```{image} ../../_static/img/inductor_profiling/trace.png
   ```

   您可以放大和缩小以检查性能分析。

* 我们通过如下日志行报告 GPU 时间相对于挂钟时间的百分比：

  **Percent of time when GPU is busy: 102.88%**

  有时您可能会看到大于 100% 的值。原因是 PyTorch 在启用性能分析时使用内核执行时间，而在禁用性能分析时使用挂钟时间。性能分析可能会稍微扭曲内核执行时间。但总体而言，这应该不是大问题。

  如果我们以较小的批处理大小运行像 ``densenet121`` 这样的模型，我们会看到 GPU 繁忙时间的百分比很低：

   ```bash
     (Forward graph) Percent of time when GPU is busy: 32.69%
   ```

  这意味着模型有很多 CPU 开销。这与启用 cudagraphs 能显著提高 densenet121 性能的事实一致。

* 我们可以将 GPU 时间分解为不同类别的内核。在 ``mixnet_l`` 示例中，我们看到

  -  pointwise 内核占 28.58%
  -  reduction 内核占 13.85%
  -  persistent reduction 内核占 3.89%
  -  其余是用于 mm/conv 的 cutlass/cudnn 内核，占 56.57%

  此信息可以在每个内核类别报告的最后一行（摘要行）中找到。

* 我们还可以放大查看特定类别的内核。例如，让我们检查 reduction 内核：

  ```{image} ../../_static/img/inductor_profiling/kernel_breakdown.png
  ```

  我们可以看到每个单独的 reduction 内核执行时间的有序表格。我们还看到一个内核被执行了多少次。这很有帮助，原因如下：

  - 如果一个内核只占用极少时间，例如 0.1%，改进它最多只能带来 0.1% 的整体增益。不值得为此花费大量精力。
  - 如果一个内核占用 2% 的时间，将其改进 2 倍将带来 1% 的整体增益，这证明了付出的努力是合理的。

## 基准测试单个 Triton 内核

假设我们想更仔细地查看 ``triton_red_fused\__native_batch_norm_legit_functional_16``，这是最昂贵的 reduction 内核，占前向图总挂钟时间的 2.19%。

我们可以在 ``fwd.py`` 中查找内核名称，并找到如下注释：

**# kernel path:
/tmp/torchinductor_shunting/jk/cjk2vm3446xrk7rth7hr6pun7xxo3dnzubwcn6ydrpifal4eykrz.py**

```{image} ../../_static/img/inductor_profiling/inductor_code.png
```

为了方便，我将其重命名为 k.py。这是该[文件](https://gist.github.com/shunting314/96a0afef9dce53d6357bf1633094f358)的内容。

``k.py`` 是一个独立的 Python 模块，包含内核代码及其基准测试。

直接运行 ``k.py`` 将报告其执行时间和带宽：

 ```{image} ../../_static/img/inductor_profiling/terminal_printout.png
 ```

我们可以通过运行以下命令来检查 max-autotune 是否对该内核有帮助：

```bash
   **TORCHINDUCTOR_MAX_AUTOTUNE=1 python /tmp/k.py**
```
我们也可以临时添加更多的归约启发式方法，并再次运行脚本来检查这对内核有何帮助。