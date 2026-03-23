```{eval-rst}
:orphan:
```

# AOTInductor 中间值调试打印器

本文档是关于如何使用 AOT Inductor 中间值调试打印器工具的用户手册。该工具是一个实用程序，可在使用 AOT Inductor 编译 PyTorch 模型时，帮助定位 CUDA IMA 内核或数值差异问题。

该工具的主要功能是自动打印或转储 AOT Inductor 中每个内核启动调用前后所有中间张量参数的值信息。

## 使用方法

调试打印器可通过环境变量进行配置。以下标志既支持在内部 fbcode buck 命令中使用，也支持在开源项目中使用。

所有配置定义在此处：[torch/_inductor/config.py](https://github.com/pytorch/pytorch/blob/768361e67f0eb36491d7b763ef38d7c928ebefe6/torch/_inductor/config.py#L1493-L1505)

```
    # aot inductor 中间张量值调试打印/保存的选项

    0: 禁用调试转储
    1: 启用保存中间张量值
    2: 启用打印中间张量值
    3: 仅启用打印内核名称（有助于定位有问题的内核）
```

1. 启用**默认**模式调试打印：

    - 添加标志 `AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=2`（PRINT_ONLY 模式）以默认打印所有支持的内核张量参数值。

    - 添加标志 `AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT={kernel_name_1, kernel_name_2,...}` 以选择性打印与指定内核关联的张量值。（建议先运行一次生成完整打印日志）

    示例命令：

    ```
    AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT="aoti_torch_cuda_addmm_out" AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=2 TORCH_LOGS="+inductor, output_code" python test/inductor/test_aot_inductor.py -k test_addmm_cuda
    ```

2. 启用**仅定位**有问题的内核名称：（在 CUDA IMA 调试中特别有用）

   - 添加标志 `AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=3`（PRINT_KERNEL_NAME_ONLY 模式），不会转储任何张量数值。

   示例命令：

   ```
   AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=3 TORCH_LOGS="+inductor, output_code" python test/inductor/test_aot_inductor.py -k test_addmm_cuda
   ```

3. 启用**保存**中间张量值：

    - 当您想在独立的内核调试复现中重现错误时非常有用。保存的中间张量值可用作有问题的内核的调试输入。
    - 设置 `AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=1`（SAVE_ONLY 模式）以默认将所有支持的内核张量参数值保存到临时文件夹中的 `.pt` 文件。
    - 类似地，添加 `AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT={kernel_name_1, kernel_name_2,...}` 以选择性保存与指定内核关联的张量值。

   示例命令：
    ```
    AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT="triton_poi_fused_0" AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=1 TORCH_LOGS="+inductor, output_code" python test/inductor/test_aot_inductor.py -k test_addmm_cuda
    ```

    保存的张量值将以以下格式转储：`<before/after_launch>_<kernel_name>_<arg_name>_<device>.pt`

    转储的 `.pt` 张量可以进一步加载和使用，如下所示：
    ```
        def _load_tensor(path):
            return torch.load(path, weights_only=True)
        tensor = _load_tensor("../tmp/aoti_torch/before_launch_aoti_torch_cuda_addmm_out_buf1_cuda:0.pt")

        # 简单地打印张量以查看完整值
        print(tensor)
    ```

## 示例输出

启动前的张量统计信息：

![示例图片 1](_static/img/aoti_debug_printer/before_launch.png)

启动后的张量统计信息：

![示例图片 2](_static/img/aoti_debug_printer/after_launch.png)