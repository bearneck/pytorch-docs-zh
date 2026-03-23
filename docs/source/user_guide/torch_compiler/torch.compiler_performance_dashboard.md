# PyTorch 2.0 性能仪表板

**作者:** [Bin Bao](https://github.com/desertfire) 和 [Huy Do](https://github.com/huydhn)

PyTorch 2.0 的性能每晚在此[仪表板](https://hud.pytorch.org/benchmark/compilers)上进行跟踪。
性能收集工作每晚在 12 个 GCP A100 节点上运行。每个节点包含一个 40GB 的 A100 Nvidia GPU 和一个 6 核 2.2GHz 的 Intel Xeon CPU。对应的 CI 工作流文件可以在[这里](https://github.com/pytorch/pytorch/blob/main/.github/workflows/inductor-perf-test-nightly.yml)找到。

## 如何阅读仪表板？

登录页面显示了我们测量的三个基准测试套件的表格：``TorchBench``、``Huggingface`` 和 ``TIMM``，以及一个基准测试套件在默认设置下的图表。例如，默认图表目前显示了过去 7 天 ``TorchBench`` 的 AMP 训练性能趋势。页面顶部的下拉列表可以选择查看不同选项的表格和图表。除了通过率，仪表板上还报告了 3 个关键性能指标：``几何平均加速比``、``平均编译时间`` 和 ``峰值内存占用压缩比``。
``几何平均加速比`` 和 ``峰值内存占用压缩比`` 都是与 PyTorch eager 模式性能进行比较的，数值越大越好。表格上的每个单独性能数字都可以点击，点击后会跳转到一个视图，显示该特定基准测试套件中所有测试的详细数字。

## 仪表板上测量了什么？

所有仪表板测试都在这个[函数](https://github.com/pytorch/pytorch/blob/3e18d3958be3dfcc36d3ef3c481f064f98ebeaf6/.ci/pytorch/test.sh#L305)中定义。具体的测试配置可能会发生变化，但目前，我们测量了三个基准测试套件在 AMP 精度下的推理和训练性能。我们还测量了 TorchInductor 的不同设置，包括 ``default``、``with_cudagraphs (default + cudagraphs)`` 和 ``dynamic (default + dynamic_shapes)``。

## 我能在合并前检查我的 PR 是否影响仪表板上的 TorchInductor 性能吗？

可以通过点击[这里](https://github.com/pytorch/pytorch/actions/workflows/inductor-perf-test-nightly.yml)的 ``Run workflow`` 按钮并选择你的 PR 分支来手动触发单个仪表板运行。这将启动一个包含你 PR 更改的完整仪表板运行。完成后，你可以在性能仪表板 UI 上选择相应的分支名称和提交 ID 来查看结果。请注意，这是一个昂贵的 CI 运行。由于资源有限，请明智地使用此功能。

## 如何在本地运行任何性能测试？

完整仪表板运行期间使用的确切命令行可以在任何最近的 CI 运行日志中找到。[工作流页面](https://github.com/pytorch/pytorch/actions/workflows/inductor-perf-test-nightly.yml)是查找最近一些运行日志的好地方。在这些日志中，你可以搜索类似
`python benchmarks/dynamo/huggingface.py --performance --cold-start-latency --inference --amp --backend inductor --disable-cudagraphs --device cuda`
的行，如果你有一个能与 PyTorch 2.0 配合使用的 GPU，可以在本地运行它们。
``python benchmarks/dynamo/huggingface.py -h`` 将为你提供基准测试脚本选项的详细说明。
