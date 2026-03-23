```{role} hidden
---
class: hidden-section
---

```
# 通用 Join 上下文管理器

通用 Join 上下文管理器旨在促进不均匀输入下的分布式训练。本页概述了相关类的 API：{class}`Join`、{class}`Joinable` 和 {class}`JoinHook`。如需教程，请参阅[使用 Join 上下文管理器进行不均匀输入的分布式训练](https://pytorch.org/tutorials/advanced/generic_join.html)。

```{eval-rst}
.. autoclass:: torch.distributed.algorithms.Join
    :members:

.. autoclass:: torch.distributed.algorithms.Joinable
    :members:

.. autoclass:: torch.distributed.algorithms.JoinHook
    :members:

```