(export.draft_export)=

# 草案导出

:::{warning}
此功能不应用于生产环境，其设计目的是作为调试 torch.export 追踪错误的工具。
:::

草案导出（draft-export）是 export 的一个新版本，其设计目标是始终生成一个计算图，即使存在潜在的正确性问题，并生成一份报告，列出 export 在追踪过程中遇到的所有问题，并提供额外的调试信息。对于没有 fake kernel 的自定义算子，它还会生成一个配置文件，您可以注册该文件以自动生成一个 fake kernel。

您是否曾尝试使用 {func}`torch.export.export` 导出一个模型，却遇到了数据依赖问题？您修复了它，但随后又遇到了缺少 fake kernel 的问题。解决之后，又遇到了另一个数据依赖问题。您可能会想，要是有一种方法能直接得到一个计算图来调试，并且能在一个地方查看所有问题以便后续修复就好了……

`draft_export` 来拯救您了！

`draft_export` 是 export 的一个版本，即使存在潜在的正确性问题，它也会始终成功导出一个计算图。这些问题随后会被编译成一份报告以便更清晰地查看，并可以在后续修复。

## 它能捕获哪些类型的错误？

草案导出有助于捕获和调试以下错误：

- 数据依赖错误上的守卫（Guard）
- 约束违反错误
- 缺少 fake kernel
- 编写错误的 fake kernel

## 它是如何工作的？

在正常的 export 中，我们会将示例输入转换为 FakeTensors，并使用它们来记录操作并将程序追踪成一个计算图。可以改变的输入张量形状（通过 `dynamic_shapes` 标记）或张量内部的值（通常来自 `.item()` 调用）将被表示为符号形状（`SymInt`）而不是具体的整数。然而，在追踪过程中可能会出现一些问题——我们可能会遇到无法评估的守卫，例如我们想检查张量中的某个项是否大于 0（`u0 >= 0`）。由于追踪器对 `u0` 的值一无所知，它将抛出一个数据依赖错误。如果模型使用了自定义算子但尚未为其定义 fake kernel，那么我们将遇到 `fake_tensor.UnsupportedOperatorException` 错误，因为 export 不知道如何在 `FakeTensors` 上应用此操作。如果自定义算子的 fake kernel 实现不正确，export 将静默地生成一个与 eager 行为不匹配的错误计算图。

为了解决上述错误，草案导出使用*真实张量追踪*来指导我们在追踪时如何继续。当我们使用 fake tensors 追踪模型时，对于发生在 fake tensor 上的每一个操作，草案导出也会在存储的真实张量上运行该算子，这些真实张量来自传递给 export 的示例输入。这使我们能够处理上述错误：当我们遇到一个无法评估的守卫，例如 `u0 >= 0` 时，我们将使用存储的真实张量值来评估此守卫。运行时断言将被添加到计算图中，以确保计算图断言了与我们在追踪时假设的相同守卫。如果我们遇到一个没有 fake kernel 的自定义算子，我们将使用存储的真实张量运行该算子的正常 kernel，并返回一个具有相同秩但未绑定形状的 fake tensor。由于我们拥有每个操作的真实张量输出，我们将将其与来自 fake kernel 的 fake tensor 输出进行比较。如果 fake kernel 实现不正确，我们将捕获此行为并生成一个更正确的 fake kernel。

## 如何使用草案导出？

假设您正在尝试导出这段代码：

```python
class M(torch.nn.Module):
    def forward(self, x, y, z):
        res = torch.ops.mylib.foo2(x, y)

        a = res.item()
        a = -a
        a = a // 3
        a = a + 5

        z = torch.cat([z, z])

        torch._check_is_size(a)
        torch._check(a < z.shape[0])

        return z[:a]

inp = (torch.tensor(3), torch.tensor(4), torch.ones(3, 3))

ep = torch.export.export(M(), inp)
```

由于 `mylib.foo2` 缺少 fake kernel，以及使用未绑定的符号整数 `a` 对 `z` 进行切片，这会遇到“缺少 fake kernel”错误和 `GuardOnDataDependentExpression` 错误。

要调用 `draft-export`，我们可以将 `torch.export` 行替换为以下内容：

```python
ep = torch.export.draft_export(M(), inp)
```

`ep` 是一个有效的 ExportedProgram，现在可以传递给后续环境了！

## 使用草案导出进行调试

在草案导出的终端输出中，您应该看到以下消息：

```
#########################################################################################
警告：在导出过程中发现 2 个问题，无法正确生成计算图。
要查看 HTML 页面中的失败报告，请运行命令：
    `tlparse /tmp/export_angelayi/dedicated_log_torch_trace_axpofwe2.log --export`
或者，您可以通过检查 `print(ep._report)` 在 Python 中查看错误。
########################################################################################
```

草案导出会自动为 `tlparse` 转储日志。您可以通过 `print(ep._report)` 查看追踪错误，或者将日志传递给 `tlparse` 以生成 HTML 报告。

在终端中运行 `tlparse` 命令将生成一个 [tlparse](https://github.com/pytorch/tlparse) HTML 报告。以下是 `tlparse` 报告的示例：

```{image} ../../../_static/img/export/draft_export_report.png
```

点击进入数据依赖错误，我们将看到以下页面，其中包含有助于调试此错误的信息。具体来说，它包含：

- 发生此错误时的堆栈跟踪
- 局部变量及其形状的列表
- 此守卫如何创建的信息

```{image} ../../../_static/img/export/draft_export_report_dde.png
```

## 返回的 Exported Program

由于草稿导出（draft-export）会根据示例输入对代码路径进行特化，因此通过草稿导出得到的导出程序（exported program）**至少**对于给定的示例输入保证是可运行且结果正确的。其他输入只要满足草稿导出时捕获的相同约束条件（guards），也可能正常工作。

例如，如果我们有一个根据某个值是否大于5进行分支的图，在草稿导出时，如果示例输入大于5，那么返回的 `ExportedProgram` 将特化于该分支，并断言该值大于5。这意味着，如果你传入另一个大于5的值，程序将成功运行；但如果你传入一个小于5的值，程序将失败。这比 `torch.jit.trace` 更可靠，因为 `torch.jit.trace` 会静默地特化于某个分支。`torch.export` 支持两个分支的正确方式是使用 `torch.cond` 重写代码，从而捕获两个分支。

由于图中存在运行时断言，返回的导出程序也可以通过 `torch.export` 或 `torch.compile` 重新追踪（retraceable），在缺少自定义算子（custom operator）伪内核（fake kernel）的情况下需要稍作处理。

## 生成伪内核

如果一个自定义算子没有包含伪实现（fake implementation），目前草稿导出将使用真实张量传播（real-tensor propagation）来获取该算子的输出并继续追踪。然而，如果我们使用伪张量（fake tensors）运行导出的程序，或者重新追踪导出的模型，仍然会失败，因为仍然没有伪内核实现。

为了解决这个问题，在草稿导出之后，我们将为遇到的每个自定义算子调用生成一个算子配置文件（operator profile），并将其存储在附加到导出程序的报告（report）中：`ep._report.op_profiles`。然后，用户可以使用上下文管理器 `torch._library.fake_profile.unsafe_generate_fake_kernels` 基于这些算子配置文件生成并注册一个伪实现。这样，未来的伪张量重新追踪就能正常工作。

工作流程大致如下：

```python
class M(torch.nn.Module):
    def forward(self, a, b):
        res = torch.ops.mylib.foo(a, b)  # 没有伪实现
        return res

ep = draft_export(M(), (torch.ones(3, 4), torch.ones(3, 4)))

with torch._library.fake_profile.unsafe_generate_fake_kernels(ep._report.op_profiles):
    decomp = ep.run_decompositions()

new_inp = (
    torch.ones(2, 3, 4),
    torch.ones(2, 3, 4),
)

# 将配置文件保存到 yaml 文件并检入代码库
save_op_profiles(ep._report.op_profiles, "op_profile.yaml")
# 加载 yaml 文件
loaded_op_profile = load_op_profiles("op_profile.yaml")
```

算子配置文件是一个字典，将算子名称映射到一组描述算子输入和输出的配置文件。这些配置文件可以手动编写，保存到 yaml 文件中，并检入代码库。以下是一个 `mylib.foo.default` 的配置文件示例：

```python
"mylib.foo.default": {
    OpProfile(
        args_profile=(
            TensorMetadata(
                rank=2,
                dtype=torch.float32,
                device=torch.device("cpu"),
                layout=torch.strided,
            ),
            TensorMetadata(
                rank=2,
                dtype=torch.float32,
                device=torch.device("cpu"),
                layout=torch.strided,
            ),
        ),
        out_profile=TensorMetadata(
            rank=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
            layout=torch.strided,
        ),
    )
}
```

`mylib.foo.default` 的配置文件只包含一个配置项，它表示对于两个秩（rank）为2、数据类型（dtype）为 `torch.float32`、设备（device）为 `cpu` 的输入张量，我们将返回一个秩为2、数据类型为 `torch.float32`、设备为 `cpu` 的张量。使用上下文管理器后，将生成一个伪内核，当给定两个秩为2的输入张量（以及其他张量元数据）时，输出一个秩为2的张量（以及其他张量元数据）。

如果该算子还支持其他输入秩，我们可以通过手动将其添加到现有配置文件中，或者使用新的输入重新运行草稿导出来获取新的配置文件，从而将新的配置项添加到这个配置列表中。这样生成的伪内核将支持更多的输入类型。否则会报错。

## 接下来该做什么？

现在我们已经使用草稿导出成功创建了一个 `ExportedProgram`，我们可以使用进一步的编译器（如 `AOTInductor`）来优化其性能并生成可运行的产物。这个优化后的版本可以用于部署。同时，我们可以利用草稿导出生成的报告来识别和修复遇到的 `torch.export` 错误，从而使原始模型能够直接被 `torch.export` 追踪。