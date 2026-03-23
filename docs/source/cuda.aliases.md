# torch.cuda 中的别名

以下是在其定义的嵌套命名空间中，对应 ``torch.cuda`` 中功能的别名。对于这些 API 中的任何一个，您都可以使用 ``torch.cuda`` 中的顶层版本，例如 ``torch.cuda.seed``，或者使用嵌套版本 ``torch.cuda.random.seed``。

```{eval-rst}
.. automodule:: torch.cuda.random
.. currentmodule:: torch.cuda.random
.. autosummary::
    :toctree: generated
    :nosignatures:

    get_rng_state
    get_rng_state_all
    set_rng_state
    set_rng_state_all
    manual_seed
    manual_seed_all
    seed
    seed_all
    initial_seed
```

```{eval-rst}
.. automodule:: torch.cuda.graphs
.. currentmodule:: torch.cuda.graphs
.. autosummary::
    :toctree: generated
    :nosignatures:

    is_current_stream_capturing
    graph_pool_handle
    CUDAGraph
    graph
    make_graphed_callables
```

```{eval-rst}
.. automodule:: torch.cuda.streams
.. currentmodule:: torch.cuda.streams
.. autosummary::
    :toctree: generated
    :nosignatures:

    Stream
    ExternalStream
    Event
```