# torch.mps

```{eval-rst}
.. automodule:: torch.mps
```

```{eval-rst}
.. currentmodule:: torch.mps
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    is_available
    device_count
    synchronize
    get_rng_state
    set_rng_state
    manual_seed
    seed
    empty_cache
    set_per_process_memory_fraction
    current_allocated_memory
    driver_allocated_memory
    recommended_max_memory
    compile_shader
```

## MPS 性能分析器

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    profiler.start
    profiler.stop
    profiler.profile

    profiler.is_capturing_metal
    profiler.is_metal_capture_enabled
    profiler.metal_capture
```

## MPS 事件

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    event.Event

```

% 此模块需要编写文档。暂时添加在此处

% 以便跟踪

```{eval-rst}
.. py:module:: torch.mps.event
```

```{eval-rst}
.. py:module:: torch.mps.profiler
```