# TorchInductor 与 AOTInductor 溯源追踪

本节介绍如何在 `tlparse` 中使用 TorchInductor 和 AOTInductor 的溯源追踪功能。 溯源追踪可帮助您可视化输入 GraphModule 与 (AOT)Inductor 生成的优化代码之间的关系。此功能允许您追踪原始操作在编译过程中是如何被转换的。

下方展示了溯源追踪工具的一些示例截图。 该工具可视化了输入图（面板1）、梯度后图（面板2）和 Inductor 生成代码（面板3）中节点之间的映射关系。

**加粗** 的行表示当前溯源追踪功能所覆盖的节点/内核。 我们目前覆盖 triton 内核、cpp 内核和组合内核。 黄色高亮显示了节点/内核的溯源。

TorchInductor 溯源追踪工具的示例截图：

:   ![image](../../_static/img/inductor_provenance/provenance_jit_inductor.png)

AOTInductor 溯源追踪工具的示例截图：

:   ![image](../../_static/img/inductor_provenance/provenance_aot_inductor.png)

## 使用溯源追踪高亮器

请按照以下步骤在您的 PyTorch 项目中启用并使用溯源追踪：

1.  通过 `cargo install tlparse` 安装 `tlparse`。如果您没有 `cargo`，请参阅 [The Cargo Book](https://doc.rust-lang.org/cargo/getting-started/installation.html) 获取安装说明。

2.  使用所需的标志运行您的程序：

    ``` bash
    TORCH_TRACE=~/my_trace_log_dir INDUCTOR_PROVENANCE=1 python your_program.py
    ```

    这将在 `/my_trace_log_dir` 中生成一个日志文件。该日志文件将被 tlparse 用来生成溯源追踪高亮器。

3.  使用 `--inductor-provenance` 标志对日志运行 `tlparse`。例如：

    ``` bash
    tlparse log_file_name.log --inductor-provenance
    ```

    - 即使您不添加 `--inductor-provenance` 标志，您也应该能在 `index.html` tlparse 输出的 `inductor_provenance_tracking_node_mappings_<number>.json` 文件中看到 JSON 格式的映射关系。

    - 直接在日志文件上运行 `tlparse`。如果您运行 \"tlparse parse \<folder_name\> \--inductor-provenance\" 可能无法工作。

    - 溯源追踪高亮器使用的 `tlparse` 工件包括：

      > - `before_pre_grad_graph.txt`
      > - `after_post_grad_graph.txt`
      > - `inductor_aot_wrapper_code.txt`
      > - `inductor_output_code.txt`
      > - `inductor_provenance_tracking_node_mappings.json`

运行 `tlparse <file_name> --inductor-provenance` 后，您应该在 tlparse 输出中看到一个额外的 \"Provenance Tracking\" 部分。点击链接即可访问溯源追踪工具。 有关演示，请参阅：https://github.com/pytorch/tlparse/pull/93

> ![image](../../_static/img/inductor_provenance/index.png)

## 每个 Inductor 内核对应的源代码

设置 `INDUCTOR_PROVENANCE=1` 后，您还可以在 tlparse 中查看每个 Inductor 内核对应的源代码。要访问它，请点击 tlparse 输出中 \"inductor_provenance_tracking_kernel_stack_traces.json\" 旁边的 \"readable_html\" 链接。

> ![image](../../_static/img/inductor_provenance/index_2.png)

以下是一些示例截图。内核名称末尾的 `:1` 和 `:467` 后缀用于区分对同一内核的不同调用。我们将这些后缀称为调试句柄。

> ![image](../../_static/img/inductor_provenance/kernel_source_1.png)
>
> ![image](../../_static/img/inductor_provenance/kernel_source_2.png)

您也可以在内核源代码的注释中找到调试句柄。

> ![image](../../_static/img/inductor_provenance/kernel_source_3.png)

## 另请参阅

`tlparse` 是一个用 Rust 编写的工具。

- tlparse GitHub 仓库链接：https://github.com/pytorch/tlparse
- 在 `torch.compiler_troubleshooting` 中了解更多关于 `tlparse` 的信息
