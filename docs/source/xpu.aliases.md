# torch.xpu 中的别名

以下是嵌套命名空间中定义的、对应 ``torch.xpu`` 中功能的别名。对于这些 API 中的任何一个，您都可以使用 ``torch.xpu`` 中的顶层版本，例如 ``torch.xpu.seed``，或使用嵌套版本 ``torch.xpu.random.seed``。

```{eval-rst}
.. automodule:: torch.xpu.random
.. currentmodule:: torch.xpu.random
.. autosummary::
    :toctree: generated
    :nosignatures:

    get_rng_state
    get_rng_state_all
    initial_seed
    manual_seed
    manual_seed_all
    seed
    seed_all
    set_rng_state
    set_rng_state_all
```

```{eval-rst}
.. automodule:: torch.xpu.graphs
.. currentmodule:: torch.xpu.graphs
.. autosummary::
    :toctree: generated
    :nosignatures:

    is_current_stream_capturing
    graph_pool_handle
    XPUGraph
    graph
    make_graphed_callables
```

```{eval-rst}
.. automodule:: torch.xpu.streams
.. currentmodule:: torch.xpu.streams
.. autosummary::
    :toctree: generated
    :nosignatures:

    Event
    Stream
```