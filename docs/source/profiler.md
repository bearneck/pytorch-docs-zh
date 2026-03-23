```{eval-rst}
.. currentmodule:: torch.profiler
```

# torch.profiler

## 概述
```{eval-rst}
.. automodule:: torch.profiler
```

## API 参考
```{eval-rst}
.. autoclass:: torch.profiler._KinetoProfile
  :members:

.. autoclass:: torch.profiler.profile
  :members:

.. autoclass:: torch.profiler.ProfilerAction
  :members:

.. autoclass:: torch.profiler.ProfilerActivity
  :members:

.. autofunction:: torch.profiler.schedule

.. autofunction:: torch.profiler.tensorboard_trace_handler
```

## Intel 插桩与追踪技术 API

```{eval-rst}
.. autofunction:: torch.profiler.itt.is_available

.. autofunction:: torch.profiler.itt.mark

.. autofunction:: torch.profiler.itt.range_push

.. autofunction:: torch.profiler.itt.range_pop
```

<!-- 此模块需要编写文档。暂时添加在此处以供追踪 -->
```{eval-rst}
.. py:module:: torch.profiler.itt
.. py:module:: torch.profiler.profiler
.. py:module:: torch.profiler.python_tracer
```