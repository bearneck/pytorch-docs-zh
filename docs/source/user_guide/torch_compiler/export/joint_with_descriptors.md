# 联合描述符

联合描述符是一个实验性 API，用于导出一个追踪的联合计算图，该计算图全面支持 torch.compile 的所有功能，并且在处理后可以转换回一个可微分的可调用对象，能够正常执行。例如，它被用于实现自动并行系统（autoparallel），该系统接收一个模型并重新分片输入和参数，使其成为一个分布式 SPMD 程序。

```{eval-rst}
.. currentmodule:: torch._functorch.aot_autograd
.. autofunction:: aot_export_joint_with_descriptors
.. autofunction:: aot_compile_joint_with_descriptors
```

## 描述符

```{eval-rst}
.. currentmodule:: torch._functorch._aot_autograd.descriptors

.. autoclass:: AOTInput
  :members:

.. autoclass:: AOTOutput
  :members:

.. autoclass:: BackwardTokenAOTInput
  :members:

.. autoclass:: BackwardTokenAOTOutput
  :members:

.. autoclass:: BufferAOTInput
  :members:

.. autoclass:: DummyAOTInput
  :members:

.. autoclass:: DummyAOTOutput
  :members:

.. autoclass:: GradAOTOutput
  :members:

.. autoclass:: InputMutationAOTOutput
  :members:

.. autoclass:: IntermediateBaseAOTOutput
  :members:

.. autoclass:: ParamAOTInput
  :members:

.. autoclass:: PhiloxBackwardBaseOffsetAOTInput
  :members:

.. autoclass:: PhiloxBackwardSeedAOTInput
  :members:

.. autoclass:: PhiloxForwardBaseOffsetAOTInput
  :members:

.. autoclass:: PhiloxForwardSeedAOTInput
  :members:

.. autoclass:: PhiloxUpdatedBackwardOffsetAOTOutput
  :members:

.. autoclass:: PhiloxUpdatedForwardOffsetAOTOutput
  :members:

.. autoclass:: PlainAOTInput
  :members:

.. autoclass:: PlainAOTOutput
  :members:

.. autoclass:: SavedForBackwardsAOTOutput
  :members:

.. autoclass:: SubclassGetAttrAOTInput
  :members:

.. autoclass:: SubclassGetAttrAOTOutput
  :members:

.. autoclass:: SubclassSizeAOTInput
  :members:

.. autoclass:: SubclassSizeAOTOutput
  :members:

.. autoclass:: SubclassStrideAOTInput
  :members:

.. autoclass:: SubclassStrideAOTOutput
  :members:

.. autoclass:: SyntheticBaseAOTInput
  :members:

.. autoclass:: ViewBaseAOTInput
  :members:
```

## FX 工具

```{eval-rst}
.. automodule:: torch._functorch._aot_autograd.fx_utils
  :members:
```