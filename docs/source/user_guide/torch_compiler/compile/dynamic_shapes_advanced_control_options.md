(dynamic_shapes_advanced_control_options)=
# 控制动态行为的高级选项

PyTorch 提供了几个高级选项来控制动态行为。
这些选项需要对 PyTorch 内部机制有深入理解，并且可能涉及设置额外的工具。这些选项包括：

* **基于性能分析的优化 (PGO)** 是一种允许编译器保存自动动态决策并在不同作业中复用的技术。
* **编译器集合** 是一个用于修改自动动态形状行为的功能，它通过推断输入的大小是否在不同 rank 间变化来判断该输入是否为动态。

## 基于性能分析的优化 (PGO)

基于性能分析的优化 (PGO) 通过在模型的不同运行之间共享性能分析决策来增强自动动态功能。具体来说，它将自动动态所做的所有选择序列化到磁盘上的一个文件中。然后，你可以复制此文件——或将其存储在像 S3 这样的集中式元数据服务中——并在其他机器上重复使用，以确保跨环境的行为一致性。

在本教程的其余部分，你可以使用以下环境变量在本地开启 PGO：`TORCH_COMPILE_JOB_ID=1 TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO=1`

(identifying-dynamic-elements-marked-by-pgo)=
### 识别由 PGO 标记的动态元素

使用 `tlparse` 查找感兴趣的行号并检查输入是否观察到多个值。

要确定哪些元素被基于性能分析的优化 (PGO) 标记为动态，请按照以下步骤使用 `tlparse`：

1. 在 `tlparse` 输出中，识别感兴趣帧的行号。示例：

   ```{image} ../../../_static/img/dynamic_shapes/tlparse4_pgo.png
   ```

2. 使用 `put_local_code_state_` 或 `put_remote_code_state_` 打开最新帧（例如，6/1）的 `local_code`。

   每个 `?` 表示已观察到该输入有多个值。

   例如，以下输出显示输入 `L['m']` 在 `size[0]` 处观察到多个大小，但其步幅始终为 1：

   ```
   /data/users/bobren/a/pytorch/r2.py:2:func:
   L['m']: fully dynamic scalar or tensor
   L['x']: tensor size=[?] stride=[1]
   L['y']: tensor size=[?] stride=[1]
   L['z']: tensor size=[?] stride=[1]
   ```

```{note}
如果一个元素被 PGO 标记为动态，这并不能保证它在计算图中将保持动态。特化操作可以将其恢复为静态状态。
```

## 编译器集合

不同的 rank 可以相互通信以共享观察到的大小。在第二次迭代中，自动动态使用此信息，根据所有 rank 上观察到的输入来决定将哪些元素标记为动态。查看此 [PR](https://github.com/pytorch/pytorch/pull/130935) 了解更多细节。
要启用此功能，请将 `enable_compiler_collectives=True` 与 `@config.patch` 装饰器一起使用。

```python
@config.patch(enable_compiler_collectives=True)
```

```{note}
此功能允许在编译期间使用集合操作来同步不同 rank 间的行为。目前，它用于修改自动动态形状行为，通过推断输入的大小是否在不同 rank 间变化来判断该输入是否为动态。由于此同步使用集合操作，所有 rank 必须同时运行编译；rank 之间不能因图中断而出现分歧。最可靠的方法是确保 torch 仅在 SPMD 程序上运行。违反此不变量可能导致 NCCL 死锁并遇到 NCCL 超时。
```

## 逐步减少编译次数

如果你有一个可以在主作业上运行的模型并且拥有 `tlparse`，以下是接下来应该做的步骤：

### 步骤 1：标记动态元素

第一步是减少那些最终会被自动动态或 PGO 优化掉的初始编译。这很简单，因为我们事先知道它会起作用。如果在一次运行中，一个帧从静态图开始并收敛到动态图，并且如果你注意到在第二次（热）启用 PGO 的运行中编译的帧数减少了，那么这很可能是由于此优化。

这是一个两步过程：

1. 找到被 PGO 或自动动态标记为动态的元素。
2. 使用 {ref}`user_annotations` 之一将它们标记为动态。

#### 如何识别要标记为动态的元素

遵循以下指南：

1. **PGO 产物：** 按照 {ref}`identifying-dynamic-elements-marked-by-pgo` 中的步骤操作。
2. **动态日志：** 如果你有一个启用了 `TORCH_LOGS="+dynamic"` 的运行，每次分配新的动态维度时，都会有一条调试行指定它以及输入名称。
3. **比较计算图：** 对于在不同运行间编译次数减少的帧，检查第二次运行或冷运行中最新运行的 Dynamo 图。查找那些图中标记为动态的元素。具体来说，找到相似的计算图（一个是特化版本，一个是动态版本）。

即使没有热运行，你也可以检查特定帧的所有计算图，看看是否有相似的计算图并收敛到一个动态版本。

例如，在下面的 `tlparse` 快照中，Dynamo 图 20/0、20/1 和 20/2 除了大小不同（例如，图 20/0 与图 20/2）外是相似的。在 20/2 的 Dynamo 图中，大小 `s0`、`s1` 和 `s5` 用于 `rotary_pos_emb_` 和 `x`。

```{image} ../../../_static/img/dynamic_shapes/tlparse5_dynamic_shapes.png
```

```{tip}
如果两个计算图具有相同的 torch 操作调用序列和相同的张量输入，则认为它们是相似的。差异可能存在于整数输入中（这些整数输入可能在特化版本中被内联），或者存在于算术计算中（这些计算可能由于静态版本中的内联而仅存在于动态版本中）。
```

### 步骤 2：调试：识别错过的优化机会

调试的复杂性可能因你遇到的问题而有很大差异。最终结果通常是找到一个 bug、启用一个标志或修改用户/框架代码。

#### 查找相似的计算图

首先识别一组可能希望合并到单个动态图中的相似图，如上一节关于比较图的讨论所述。如果找不到任何相似图，则此步骤无需进一步操作。

#### 快速检查：快速失败

找到相似图后，需要理解它们为何重新编译。请检查以下内容：

1. **检查重新编译原因：** 对于你认为相似的图，在 `tlparse` 输出中点击较晚图的 `recompile_reason`。确保原因是尺寸相关的，而非其他因素。例如，在下述截图中重新编译原因是尺寸相关的：

```{image} ../../../_static/img/dynamic_shapes/tlparse6_size_related_recompilations.png
```

而在下图中则不是，这表明动态形状无法解决此问题：

```{image} ../../../_static/img/dynamic_shapes/tlparse7_not_size_related_recompilations.png
:width: 500px
:align: center
```

2. **比较 Guards 文件：** 确保不存在非尺寸相关元素上的守卫，这些守卫存在于一个图中但不存在于其他图中。

3. **早期检查自定义 Triton 内核：** 检查你的模型是否调用了带有 `tl.constexpr` 参数的自定义 Triton 内核，因为这些参数总是被特化的。如果你的模型接收这些参数的不同值，这可能是重新编译的一个来源。

## **识别并修复重新编译原因**

1. **是否有应标记为动态但未标记的元素？** 确定一个输入是否被标记为动态但被特化了，或者根本没有被标记为动态。你可以通过以下方式识别：

    * 检查 Dynamo 图 - 查找 `Sym(number)`。例如：

      ```
      Sym(256) vs Sym(s0)
      ```

    * 使用动态日志：

      ```
      ["TORCH_LOGS=+dynamic"]
      create_symbol s2 = 2 for L['self']._modules['cle ...
      ```

    * 审查 guards 文件。如果张量尺寸是动态的，它将显示为 `None`：

      ```
      TENSOR_MATCH:check_tensor(L['self'].x._parameters['weight']], Parameter, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[None, None], stride=[None, 1])
      ```

2. **为何未标记为动态？** 如果你确定某个元素未标记为动态，请考虑：

    * 检查它是否是 `nn` 模块属性、参数或字段。验证以下标志的设置：
      * `force_parameter_static_shapes = True`
      * `force_nn_module_property_static_shapes = True`
      * `allow_unspec_int_on_nn_module = False`
      * 或者使用动态允许列表将其标记为动态，这应具有最高优先级。

    ```{tip}
    逐个标记元素可能很耗时。最初，可以翻转这些标志以识别任何阻碍性的特化，然后在过程结束时决定如何将它们标记为动态。
    ```

    * 如果你认为这可能是错误，请提交错误报告并标记为 `module: dynamic shapes` 标签。请在[此列表](https://github.com/pytorch/pytorch/issues?q=sort%3Aupdated-desc+state%3Aopen+label%3A%22module%3A+dynamic+shapes%22)中查看已知问题列表。

3. **动态元素是否被特化了？** 确定它为何被特化。原因可能是用户代码（如 `if` 条件）、框架代码或对 Triton 内核的调用。要识别特化的原因：

    * **使用 tlparse：** 检查 `compilation_metrics` 中的特化部分，该部分将指示发生了什么特化以及发生时的用户和框架堆栈。示例：

    ```{image} ../../../_static/img/dynamic_shapes/tlparse8_compilation_metrics.png
    ```

    上面的日志表明 `s0` 被特化为 `33`，原因如下代码：

    ```
    `if self.x ==33` at example4.py line 16.
    ```

    * **+Dynamic 日志：** 传递 `["TORCH_LOGS=+dynamic"]`。查找第一个特化，因为一旦变量被特化，所有依赖变量也会被特化。

    示例日志：

    ```
    torch/fx/experimental/symbolic_shapes.py:6557] [0/2] eval Eq(s0, 33) [guard added] if self.x ==33:  # example4.py:16 in forward (_dynamo/variables/tensor.py:1242 in evaluate_expr), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="Eq(s0, 33)"
    V0228 12:04:24.190000 2990033 torch/fx/experimental/symbolic_shapes.py:6000] [0/2] _update_var_to_range s0 = VR[33, 33] (update)
    ```

    上面的日志表明 `s0` 被特化为 `33`，原因如下代码：
    ```
    if self.x ==33. At example4.py like 16.
    ```