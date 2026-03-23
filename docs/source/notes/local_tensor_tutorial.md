# LocalTensor 教程：单进程 SPMD 调试 {#local_tensor_tutorial}

本教程介绍 `LocalTensor`，这是一个强大的调试工具，用于开发和测试分布式张量操作，而无需多个进程或 GPU。

 {.contents local="" depth="2"}
目录


## 什么是 LocalTensor？

`LocalTensor` 是 `torch.Tensor` 的一个子类，它在单个进程上模拟分布式 SPMD（单程序多数据）计算。它在内部维护从 rank ID 到其对应本地张量分片的映射，使您能够在没有基础设施开销的情况下调试和测试分布式代码。

### 主要优势

1.  **无需多进程设置**：在单个 CPU/GPU 上测试分布式算法
2.  **更快的调试周期**：无需启动多个进程即可快速迭代
3.  **完全可见性**：直接检查每个 rank 的张量状态
4.  **CI 友好**：在单进程 CI 流水线中运行分布式测试
5.  **DTensor 集成**：无缝地在本地测试 DTensor 代码

 note
 title
Note


`LocalTensor` 仅用于\*\*调试和测试\*\*，不适用于生产环境。在本地模拟多个 rank 的开销是显著的。


## 安装与设置

`LocalTensor` 是 PyTorch 分布式包的一部分。除了 PyTorch 本身之外，无需额外安装。

## 使用示例

以下示例演示了使用 `LocalTensor` 的核心模式。每个示例的代码都直接包含自源文件，这些文件也经过测试以确保正确性。测试直接调用这些相同的函数。

### 示例 1：基本的 LocalTensor 创建与操作

**从每个 rank 的张量创建 LocalTensor：**

 {.literalinclude language="python" start-after="# [core_create_local_tensor]" end-before="# [end_core_create_local_tensor]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_01_basic_operations.py


**算术运算（按 rank 应用）：**

 {.literalinclude language="python" start-after="# [core_arithmetic_operations]" end-before="# [end_core_arithmetic_operations]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_01_basic_operations.py


**在所有分片相同时提取张量：**

 {.literalinclude language="python" start-after="# [core_reconcile]" end-before="# [end_core_reconcile]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_01_basic_operations.py


**使用 LocalTensorMode 自动创建 LocalTensor：**

 {.literalinclude language="python" start-after="# [core_local_tensor_mode]" end-before="# [end_core_local_tensor_mode]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_01_basic_operations.py


**完整源代码：** [example_01_basic_operations.py](https://github.com/pytorch/pytorch/blob/main/test/distributed/local_tensor_tutorial_examples/example_01_basic_operations.py)

### 示例 2：模拟集合通信操作

无需多个进程即可测试 `all_reduce`、`broadcast` 和 `all_gather` 等集合通信操作。

**使用 SUM 的 All-reduce：**

 {.literalinclude language="python" start-after="# [core_all_reduce]" end-before="# [end_core_all_reduce]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_02_collective_operations.py


**从源 rank 进行广播：**

 {.literalinclude language="python" start-after="# [core_broadcast]" end-before="# [end_core_broadcast]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_02_collective_operations.py


**All-gather 以收集所有 rank 的张量：**

 {.literalinclude language="python" start-after="# [core_all_gather]" end-before="# [end_core_all_gather]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_02_collective_operations.py


**完整源代码：** [example_02_collective_operations.py](https://github.com/pytorch/pytorch/blob/main/test/distributed/local_tensor_tutorial_examples/example_02_collective_operations.py)

### 示例 3：与 DTensor 协同工作

`LocalTensor` 与 DTensor 集成，用于测试分布式张量并行。

**分发张量并验证重构：**

 {.literalinclude language="python" start-after="# [core_dtensor_distribute]" end-before="# [end_core_dtensor_distribute]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_03_dtensor_integration.py


**分布式矩阵乘法：**

 {.literalinclude language="python" start-after="# [core_dtensor_matmul]" end-before="# [end_core_dtensor_matmul]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_03_dtensor_integration.py


**模拟分布式线性层：**

 {.literalinclude language="python" start-after="# [core_dtensor_nn_layer]" end-before="# [end_core_dtensor_nn_layer]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_03_dtensor_integration.py


**完整源代码：** [example_03_dtensor_integration.py](https://github.com/pytorch/pytorch/blob/main/test/distributed/local_tensor_tutorial_examples/example_03_dtensor_integration.py)

### 示例 4：处理不均匀分片

现实世界的分布式系统通常在不同 rank 之间具有不均匀的数据分布。`LocalTensor` 使用 `LocalIntNode` 来处理这种情况。

**创建每个 rank 大小不同的 LocalTensor：**

 {.literalinclude language="python" start-after="# [core_uneven_shards]" end-before="# [end_core_uneven_shards]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_04_uneven_sharding.py


**LocalIntNode 算术运算：**

 {.literalinclude language="python" start-after="# [core_local_int_node]" end-before="# [end_core_local_int_node]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_04_uneven_sharding.py


**维度无法均匀划分的 DTensor：**

 {.literalinclude language="python" start-after="# [core_dtensor_uneven]" end-before="# [end_core_dtensor_uneven]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_04_uneven_sharding.py


**完整源代码：** [example_04_uneven_sharding.py](https://github.com/pytorch/pytorch/blob/main/test/distributed/local_tensor_tutorial_examples/example_04_uneven_sharding.py)

### 示例 5：特定秩的计算

有时需要在不同的秩上执行不同的操作。

**使用 rank_map() 创建每个秩的特定值：**

 {.literalinclude language="python" start-after="# [core_rank_map]" end-before="# [end_core_rank_map]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_05_rank_specific.py


**使用 tensor_map() 按秩转换分片：**

 {.literalinclude language="python" start-after="# [core_tensor_map]" end-before="# [end_core_tensor_map]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_05_rank_specific.py


**临时退出 LocalTensorMode：**

 {.literalinclude language="python" start-after="# [core_disable_mode]" end-before="# [end_core_disable_mode]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_05_rank_specific.py


**完整源代码：** [example_05_rank_specific.py](https://github.com/pytorch/pytorch/blob/main/test/distributed/local_tensor_tutorial_examples/example_05_rank_specific.py)

### 示例 6：多维设备网格

使用 2D/3D 设备网格实现混合并行（例如，数据并行 + 张量并行）。

**创建 2D 网格：**

 {.literalinclude language="python" start-after="# [core_2d_mesh]" end-before="# [end_core_2d_mesh]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_06_multidim_mesh.py


**混合并行（DP + TP）：**

 {.literalinclude language="python" start-after="# [core_hybrid_parallel]" end-before="# [end_core_hybrid_parallel]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_06_multidim_mesh.py


**用于 DP + TP + PP 的 3D 网格：**

 {.literalinclude language="python" start-after="# [core_3d_mesh]" end-before="# [end_core_3d_mesh]" dedent="0"}
../../../test/distributed/local_tensor_tutorial_examples/example_06_multidim_mesh.py


**完整源代码：** [example_06_multidim_mesh.py](https://github.com/pytorch/pytorch/blob/main/test/distributed/local_tensor_tutorial_examples/example_06_multidim_mesh.py)

## 测试教程示例

本教程中的所有示例都经过测试以确保正确性。测试套件直接调用上面包含的相同函数：

``` python
# 来自 test_local_tensor_tutorial_examples.py
from example_01_basic_operations import create_local_tensor

def test_create_local_tensor(self):
    lt = create_local_tensor()
    self.assertIsInstance(lt, LocalTensor)
    self.assertEqual(lt.shape, torch.Size([2, 2]))
```

**测试套件：** [test_local_tensor_tutorial_examples.py](https://github.com/pytorch/pytorch/blob/main/test/distributed/local_tensor_tutorial_examples/test_local_tensor_tutorial_examples.py)

## API 参考

### 核心类

 {.autoclass members="reconcile, is_contiguous, contiguous, tolist, numpy"}
torch.distributed.\_local_tensor.LocalTensor


 {.autoclass members="disable, rank_map, tensor_map"}
torch.distributed.\_local_tensor.LocalTensorMode


 autoclass
torch.distributed.\_local_tensor.LocalIntNode


### 实用函数

 autofunction
torch.distributed.\_local_tensor.local_tensor_mode


 autofunction
torch.distributed.\_local_tensor.enabled_local_tensor_mode


 autofunction
torch.distributed.\_local_tensor.maybe_run_for_local_tensor


 autofunction
torch.distributed.\_local_tensor.maybe_disable_local_tensor_mode


## 最佳实践

1.  **仅用于测试**：LocalTensor 有显著的开销，不应在生产代码中使用。
2.  **初始化进程组**：即使对于本地测试，也需要初始化一个进程组（使用 \"fake\" 后端）。
3.  **避免内部张量设置 requires_grad**：LocalTensor 期望内部张量不设置 `requires_grad=True`。应在 LocalTensor 包装器上设置梯度。
4.  **使用 reconcile 进行断言**：当所有秩应具有相同值时（例如，执行 all-reduce 后），使用 `reconcile()` 提取单个张量。
5.  **通过直接访问进行调试**：通过 `tensor._local_tensors[rank]` 访问单个分片以进行调试。

## 常见陷阱

1.  **忘记上下文管理器**：在 `LocalTensorMode` 之外对 LocalTensor 进行操作仍然有效，但不会从工厂函数创建新的 LocalTensor。
2.  **秩不匹配**：确保一个操作中的所有 LocalTensor 具有兼容的秩。
3.  **内部张量梯度**：从设置了 `requires_grad=True` 的张量创建 LocalTensor 会引发错误。
