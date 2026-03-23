# 联合描述符

联合描述符是一个实验性 API，用于导出一个追踪的联合计算图，该计算图全面支持 torch.compile 的所有功能，并且在处理后可以转换回一个可微分的可调用对象，能够正常执行。例如，它被用于实现自动并行系统（autoparallel），该系统接收一个模型并重新分片输入和参数，使其成为一个分布式 SPMD 程序。


> 模块：`torch._functorch.aot_autograd`


| API | 说明 |
|-----|------|
| `torch._functorch.aot_autograd.aot_export_joint_with_descriptors` | — |
| `torch._functorch.aot_autograd.aot_compile_joint_with_descriptors` | — |

## 描述符


| 类 | 说明 |
|-----|------|
| `torch._functorch._aot_autograd.descriptors.AOTInput` | — |
| `torch._functorch._aot_autograd.descriptors.AOTOutput` | — |
| `torch._functorch._aot_autograd.descriptors.BackwardTokenAOTInput` | — |
| `torch._functorch._aot_autograd.descriptors.BackwardTokenAOTOutput` | — |
| `torch._functorch._aot_autograd.descriptors.BufferAOTInput` | — |
| `torch._functorch._aot_autograd.descriptors.DummyAOTInput` | — |
| `torch._functorch._aot_autograd.descriptors.DummyAOTOutput` | — |
| `torch._functorch._aot_autograd.descriptors.GradAOTOutput` | — |
| `torch._functorch._aot_autograd.descriptors.InputMutationAOTOutput` | — |
| `torch._functorch._aot_autograd.descriptors.IntermediateBaseAOTOutput` | — |
| `torch._functorch._aot_autograd.descriptors.ParamAOTInput` | — |
| `torch._functorch._aot_autograd.descriptors.PhiloxBackwardBaseOffsetAOTInput` | — |
| `torch._functorch._aot_autograd.descriptors.PhiloxBackwardSeedAOTInput` | — |
| `torch._functorch._aot_autograd.descriptors.PhiloxForwardBaseOffsetAOTInput` | — |
| `torch._functorch._aot_autograd.descriptors.PhiloxForwardSeedAOTInput` | — |
| `torch._functorch._aot_autograd.descriptors.PhiloxUpdatedBackwardOffsetAOTOutput` | — |
| `torch._functorch._aot_autograd.descriptors.PhiloxUpdatedForwardOffsetAOTOutput` | — |
| `torch._functorch._aot_autograd.descriptors.PlainAOTInput` | — |
| `torch._functorch._aot_autograd.descriptors.PlainAOTOutput` | — |
| `torch._functorch._aot_autograd.descriptors.SavedForBackwardsAOTOutput` | — |
| `torch._functorch._aot_autograd.descriptors.SubclassGetAttrAOTInput` | — |
| `torch._functorch._aot_autograd.descriptors.SubclassGetAttrAOTOutput` | — |
| `torch._functorch._aot_autograd.descriptors.SubclassSizeAOTInput` | — |
| `torch._functorch._aot_autograd.descriptors.SubclassSizeAOTOutput` | — |
| `torch._functorch._aot_autograd.descriptors.SubclassStrideAOTInput` | — |
| `torch._functorch._aot_autograd.descriptors.SubclassStrideAOTOutput` | — |
| `torch._functorch._aot_autograd.descriptors.SyntheticBaseAOTInput` | — |
| `torch._functorch._aot_autograd.descriptors.ViewBaseAOTInput` | — |

## FX 工具
