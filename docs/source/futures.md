```{eval-rst}
.. currentmodule:: torch.futures
```

(futures-docs)=

# torch.futures

本模块提供了一个 {class}`~torch.futures.Future` 类型，用于封装异步执行，并提供了一组工具函数来简化对 {class}`~torch.futures.Future` 对象的操作。目前，{class}`~torch.futures.Future` 类型主要被 {ref}`distributed-rpc-framework` 使用。

```{eval-rst}
.. automodule:: torch.futures
```

```{eval-rst}
.. autoclass:: Future
    :inherited-members:
```

```{eval-rst}
.. autofunction:: collect_all
```

```{eval-rst}
.. autofunction:: wait_all
```