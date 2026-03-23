# 算子注册

对于新的加速器而言，集成中最重要和基础的方面之一是支持高性能算子。为了便于用户和加速器开发者适配算子，PyTorch 提供了多种在 `Python` 和 `C++` 中开发和注册算子的方法。以下章节详细介绍了 PyTorch 用于算子注册的一些基础能力。

```{note}
`Dispatch Key` 用于在 PyTorch 内部唯一标识加速器，例如 `CPU`、`CUDA`、`MPS` 和 `PrivateUse1`。理论上，所有后续的新加速器都将共享 `PrivateUse1`，利用其内置的全面脚手架能力来完成新加速器的集成。如果您对调度器感兴趣，请参考 [Let's talk about the PyTorch dispatcher](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)。
```


## 算子集合

PyTorch 目前有超过 3500 个内置算子（包括相关的算子变体）。从任何角度来看，这都代表着巨大的工作量，在短时间内支持如此庞大的算子数量并非易事。因此，作为开发新后端算子的第一步，我们的目标应该是专注于核心算子。对于其他算子，我们可以首先使用社区的 fallback 机制来优先支持功能。之后，再逐步完成其他算子以提高新后端的性能。

所需的算子集合如下所示，主要由工厂函数所需的低级算子和 fallback 算子组成：

| 算子名称                          | Dispatch Key | 描述                                                                                                        |
| :---:                              | :---:        | :---:                                                                                                              |
| empty.memory_format                | PrivateUse1  | 创建具有指定形状和内存布局的未初始化张量（步幅自动计算） |
| empty_strided                      | PrivateUse1  | 创建具有指定形状和步幅的未初始化张量（自由度更高）                         |
| as_strided                         | PrivateUse1  | 使用新的形状、步幅和偏移量创建输入张量的共享视图（不分配新内存）        |
| view                               | PrivateUse1  | 使用新形状创建输入张量的共享视图，但原始张量必须是内存连续的         |
| _reshape_alias                     | PrivateUse1  | 创建无安全检查的共享视图（reshape 的内部版本）                                           |
| resize_                            | PrivateUse1  | 原地修改张量的形状，如果容量不足则重新分配内存                          |
| _copy_from                         | PrivateUse1  | Tensor.copy_ 的底层核心函数，负责实际的跨设备数据复制               |
| _copy_from_and_resize              | PrivateUse1  | 结合 `resize_` 和 `_copy_from`，先调整大小再复制                                                   |
| _local_scalar_dense                | PrivateUse1  | `.item()` 的底层实现，从张量中提取值到 CPU 标量                           |
| set_.source_Tensor                 | PrivateUse1  | 使用指定的张量设置当前张量                                                                  |
| set_.source_Storage                | PrivateUse1  | 使用指定的 Storage 设置当前张量                                                                 |
| set_.source_Storage_storage_offset | PrivateUse1  | 使用具有存储偏移量的指定 Storage 设置当前张量                                         |
| fallback                           | PrivateUse1  | 回退到 CPU                                                                                                    |

## 基础

既然我们已经定义了算子支持的初始范围，就可以开始开发算子适配了。本节将根据实际场景，在 `Python` 和 `C++` 中解释这些实现。


### 步骤 1

`上面提到的算子 <operator-set>` 有一个共同特点：它们都是具有已定义 `命名空间` 和 `Schemas` 的内置 PyTorch 算子，并且这些算子的内置加速器（`CPU`、`CUDA` 等）已经实现。我们接下来要做的就是为新的加速器实现这些算子。


以 `empty.memory_format` 算子为例，我们首先需要在 `native_functions.yaml` 中查询该算子的 `schema` 信息，其中包含详细的签名信息。然后，我们可以根据新加速器的能力来实现该算子。

```Yaml
- func: empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
dispatch:
    CPU: empty_cpu
    CUDA: empty_cuda
    ...
```


完成 `wrapper_empty_memory_format` 后，我们可以通过 `TORCH_LIBRARY_IMPL` 为 `PrivateUse1` 注册 `aten::empty.memory_format`。

### 步骤 2

遵循 `步骤 1<step-one>`，我们可以完成除 `fallback` 外所有算子的开发和注册。接下来，为了支持运算相关的算子（例如数学运算、卷积运算等），我们需要实现回退语义的注册。这是 PyTorch 框架提供的内置能力，可以将新加速器不支持的部分操作回退到 CPU 上执行。对于开发中的新后端，这是一种以性能为代价保证功能的极其有效的方式。


`wrapper_cpu_fallback` 包装了 PyTorch 提供的 `at::native::cpu_fallback` 方法，并通过 `TORCH_LIBRARY_IMPL` 在 PyTorch 中为 `PrivateUse1` 注册。后续新后端不支持的运算将自动回退到 CPU 执行，执行完毕后结果会传回新后端。

## 进阶

### 选择性回退

仅对部分算子启用回退机制，而其他算子遵循 PyTorch 的默认行为（如果加速器没有对应的算子实现则会报错），这也是一个非常合理的场景。


按算子回退与全局回退非常相似，唯一的区别在于注册方式：调用 `m.impl` 是为特定算子注册一个实现，而 `m.fallback` 是为所有算子注册一个默认实现。


当然，全局回退也可以结合回退黑名单使用，这是一种常见的方法，特别是当只有少数算子不支持回退时。

### PyTorch STUB

PyTorch 还为内置算子提供了另一种方法：`STUB`。这种方法本质上基于 `步骤 1<step-one>` 的方法，但增加了二次调度的能力（例如，基于 CPU 特性进行调度）。

```{note}
`STUB` 方法目前仅支持有限的算子集合。对于新的加速器设备，`STUB` 方法的优势在于以微小的性能开销为代价，显著降低了开发成本。PyTorch 目前没有明确列出可以通过 `STUB` 注册的算子集合。由于相关算子数量众多，这里仅提供支持的算子列表的查询方法。
```

```shell
pushd ${TORCH_ROOT}

find aten -type f -a -name "*.h" | xargs -I {} grep -wl "^DECLARE_DISPATCH" {}

popd
```

`DECLARE_DISPATCH` 是一个用于显式声明 `STUB` 的宏。它目前分布在 `aten` 目录中。基于此宏，你可以找到所有可以使用 `STUB` 方法集成的算子。

```text
...
aten/src/ATen/native/Activation.h
aten/src/ATen/native/FusedSGD.h
aten/src/ATen/native/nested/NestedTensorBinaryOps.h
aten/src/ATen/native/TensorCompare.h
aten/src/ATen/native/Sorting.h
...
```

```c++
using unary_fn = void(*)(TensorIteratorBase&);

DECLARE_DISPATCH(unary_fn, abs_stub)
```

上面的列表包含了声明 `STUB` 算子的文件，在这里你可以清晰地看到 STUB 名称和关联的函数签名。接下来，我们将以 `abs_stub` 为例，简要介绍通过 `STUB` 支持算子的路径。


从签名可以看出，`abs_stub` 的输入是 `TensorIteratorBase`，这是 PyTorch 提供的一个强大的辅助类，包含了所有输入和输出操作符，以及一些其他辅助方法。基于此，我们可以开发 `abs_kernel` 操作符，然后调用 `REGISTER_PRIVATEUSE1_DISPATCH` 来指定 `abs_stub` 以完成注册。

### 自定义操作符

除了 PyTorch 内置的操作符外，自定义加速器操作符在特定场景下提升性能也非常常见。这些主要可以分为三种方法：

* 仅前向
* 前向和反向：分开注册
* 前向和反向：使用 `torch.autograd.Function` 实现

```{note}
PyTorch 教程中有更多细节，如果您感兴趣，请参考 [PyTorch 自定义操作符](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)。
```

#### 仅前向

这里，我们将简要介绍自定义操作符的实现过程，重点介绍仅前向的方法。实现可以总结为以下三点：

1. **定义模式：**

    

    

    

    

    

    * 命名空间名称：`openreg`
    * 函数名称：`custom_abs`
    * 输入参数：
        * 类型：`Tensor`
        * 名称：`input`
    * 输出类型：`Tensor`

2. **注册操作符**

    

    

    {eval-rst}
    .. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/meta.py
        :language: python
        :start-after: LITERALINCLUDE START: CUSTOM OPERATOR META
        :end-before: LITERALINCLUDE END: CUSTOM OPERATOR META
        :linenos:
    ```

    

    PyTorch 支持在 C++ 和 Python 中注册 `Meta`。由于 Python 注册更简单，这里以 Python 为例。类似于 C++ 中的 `TORCH_LIBRARY_IMPL` 函数，Python 提供了更用户友好的 `torch.library.impl` 装饰器。

## 工具

PyTorch 中的操作符注册非常复杂，注册方法多样，场景众多。因此，PyTorch 社区提供了一系列工具来帮助开发者快速理解底层原理并协助排查问题。这里我们简要介绍几个常用的工具：

### 命令

PyTorch 围绕其 Dispatch 特性提供了一组以 `torch._C._dispatch_` 为前缀的命令。您可以使用以下命令查询所有相关接口：

```Shell
python -c 'import torch; print("\n".join([x for x in dir(torch._C) if x.startswith("_dispatch_")]))'

...
_dispatch_dump
_dispatch_dump_table
_dispatch_has_kernel
_dispatch_has_kernel_for_any_dispatch_key
_dispatch_has_kernel_for_dispatch_key
_dispatch_isTensorSubclassLike
_dispatch_is_alias_key
_dispatch_is_included_in_alias
_dispatch_is_main_interpreter
_dispatch_kernel_for_dispatch_key_is_fallthrough
_dispatch_key_for_device
_dispatch_key_name
_dispatch_key_parse
_dispatch_key_set
...
```

以下是几个常用命令的解释：

* `torch._C._dispatch_key_set`：

    显示当前张量的 DispatchKey，优先级从左到右递增。

    ```Python
    >>> import torch
    >>> a = torch.randn(3,3,device="cuda")
    >>> torch._C._dispatch_key_set(a)
    'DispatchKeySet(CUDA, ADInplaceOrView, AutogradCUDA, AutocastCUDA)'
    ```

* `torch._C._dispatch_dump_table`：

    查询给定操作符在不同 Dispatch Key 下的支持状态，便于定位对应的实现代码。

```Python
>>> import torch
>>> print(torch._C._dispatch_dump_table("aten::add.Tensor"))
>>> ...
    CPU: registered at ./build/aten/src/ATen/RegisterCPU_0.cpp:1309 [kernel]
    CUDA: registered at ./build/aten/src/ATen/RegisterCUDA_0.cpp:2420 [kernel]
    HIP: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
    MPS: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
    IPU: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
    XPU: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
    HPU: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
    VE: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
    MTIA: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
    MAIA: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
    PrivateUse1: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
    ...
```

你可以轻松查询 `aten::add.Tensor` 算子在其它平台上的对应实现，从而可以从源码层面追踪整个算子的调用过程。

### 环境变量

PyTorch 还提供了一些与分发器相关的环境变量，可以帮助学习和快速定位问题。

* TORCH_SHOW_DISPATCH_TRACE

    显示 PyTorch 执行过程中详细的内部分发键调度信息。

    ```Bash
    export TORCH_SHOW_DISPATCH_TRACE=1
    ```

    ```Python
    >>> import torch
    >>> a = torch.randn(3,3)
     [call] op=[aten::randn], key=[BackendSelect]
       [redispatch] op=[aten::randn], key=[CPU]
         [call] op=[aten::empty.memory_format], key=[BackendSelect]
           [redispatch] op=[aten::empty.memory_format], key=[CPU]
         [call] op=[aten::normal_], key=[CPU]
    ```

    你可以清晰地看到 PyTorch 内部 Python 层级算子所调用的所有底层算子：包括算子名称、调用层级以及对应的 `Dispatch Key`。