```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# torch._logging

PyTorch 拥有一个可配置的日志系统，可以为不同的组件设置不同的日志级别。例如，可以完全禁用某个组件的日志消息，而将另一个组件的日志消息设置为最高详细程度。

:::{warning}
此功能目前处于测试阶段，未来可能会有破坏兼容性的更改。
:::

:::{warning}
此功能尚未扩展到控制 PyTorch 中所有组件的日志消息。
:::

有两种方式可以配置日志系统：通过环境变量 `TORCH_LOGS` 或 Python API `torch._logging.set_logs`。

```{eval-rst}
.. automodule:: torch._logging
.. currentmodule:: torch._logging
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    set_logs
```

环境变量 `TORCH_LOGS` 是一个以逗号分隔的 `[+-]<component>` 对列表，其中 `<component>` 是下文指定的组件。前缀 `+` 会降低组件的日志级别，显示更多日志消息；而前缀 `-` 会提高组件的日志级别，显示更少的日志消息。默认设置是当组件未在 `TORCH_LOGS` 中指定时的行为。除了组件，还有产物。产物是与组件关联的特定调试信息片段，要么显示要么不显示，因此给产物加上 `+` 或 `-` 前缀将不起作用。由于它们与组件关联，启用该组件通常也会启用该产物，除非该产物被指定为 `off_by_default`。此选项在 _registrations.py 中指定，适用于那些过于冗长、仅应在显式启用时显示的产物。以下组件和产物可通过 `TORCH_LOGS` 环境变量进行配置（Python API 请参见 torch._logging.set_logs）：

```{eval-rst}
组件：
        ``all``
            特殊组件，用于配置所有组件的默认日志级别。默认值：``logging.WARN``

        ``dynamo``
            TorchDynamo 组件的日志级别。默认值：``logging.WARN``

        ``aot``
            AOTAutograd 组件的日志级别。默认值：``logging.WARN``

        ``inductor``
            TorchInductor 组件的日志级别。默认值：``logging.WARN``

        ``your.custom.module``
            任意未注册模块的日志级别。提供完全限定名称，该模块将被启用。默认值：``logging.WARN``
```

```{eval-rst}
产物：
        ``bytecode``
            是否输出 TorchDynamo 的原始和生成的字节码。
            默认值：``False``

        ``aot_graphs``
            是否输出 AOTAutograd 生成的图。默认值：``False``

        ``aot_joint_graph``
            是否输出 AOTAutograd 生成的联合前向-反向图。默认值：``False``

        ``compiled_autograd``
            是否输出 compiled_autograd 的日志。默认值：``False``

        ``ddp_graphs``
            是否输出 DDPOptimizer 生成的图。默认值：``False``

        ``graph``
            是否以表格格式输出 TorchDynamo 捕获的图。
            默认值：``False``

        ``graph_code``
            是否输出 TorchDynamo 捕获的图的 Python 源代码。
            默认值：``False``

        ``graph_breaks``
            在 TorchDynamo 追踪期间遇到唯一图中断时，是否输出消息。
            默认值：``False``

        ``guards``
            是否输出 TorchDynamo 为每个编译函数生成的守卫。
            默认值：``False``

        ``recompiles``
            每次 TorchDynamo 重新编译函数时，是否输出守卫失败的原因和消息。
            默认值：``False``

        ``output_code``
            是否输出 TorchInductor 的输出代码。默认值：``False``

        ``schedule``
            是否输出 TorchInductor 的调度。默认值：``False``
```

```{eval-rst}
示例：
    ``TORCH_LOGS="+dynamo,aot"`` 会将 TorchDynamo 的日志级别设置为 ``logging.DEBUG``，并将 AOT 的日志级别设置为 ``logging.INFO``

    ``TORCH_LOGS="-dynamo,+inductor"`` 会将 TorchDynamo 的日志级别设置为 ``logging.ERROR``，并将 TorchInductor 的日志级别设置为 ``logging.DEBUG``

    ``TORCH_LOGS="aot_graphs"`` 将启用 ``aot_graphs`` 产物

    ``TORCH_LOGS="+dynamo,schedule"`` 会将 TorchDynamo 的日志级别设置为 ``logging.DEBUG`` 并启用 ``schedule`` 产物

    ``TORCH_LOGS="+some.random.module,schedule"`` 会将 some.random.module 的日志级别设置为 ``logging.DEBUG`` 并启用 ``schedule`` 产物
```