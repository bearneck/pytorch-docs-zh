# torch 中的别名

以下是 ``torch`` 中定义的嵌套命名空间对应功能的别名。您可以使用 ``torch`` 中的顶层版本（例如 ``torch.broadcast_tensors()``）或嵌套版本 ``torch.functional.broadcast_tensors()``。

```{eval-rst}
.. automodule:: torch.functional
.. currentmodule:: torch.functional
.. autosummary::
   :toctree: generated
   :nosignatures:

    align_tensors
    atleast_1d
    atleast_2d
    atleast_3d
    block_diag
    broadcast_shapes
    broadcast_tensors
    cartesian_prod
    cdist
    chain_matmul
    einsum
    lu
    meshgrid
    norm
    split
    stft
    tensordot
    unique
    unique_consecutive
    unravel_index
```