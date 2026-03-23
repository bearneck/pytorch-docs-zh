# torch.mtia

MTIA 后端是在树外实现的，此处仅定义接口。

```{eval-rst}
.. automodule:: torch.mtia
```

```{eval-rst}
.. currentmodule:: torch.mtia
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    StreamContext
    current_device
    current_stream
    default_stream
    device_count
    init
    is_available
    is_bf16_supported
    is_initialized
    memory_stats
    get_device_capability
    empty_cache
    record_memory_history
    snapshot
    attach_out_of_memory_observer
    set_device
    set_stream
    stream
    synchronize
    device
    set_rng_state
    get_rng_state
    set_rng_state_all
    get_rng_state_all
    manual_seed
    manual_seed_all
    seed
    seed_all
    initial_seed
    DeferredMtiaCallError
```

## 流和事件

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    Event
    Stream
```