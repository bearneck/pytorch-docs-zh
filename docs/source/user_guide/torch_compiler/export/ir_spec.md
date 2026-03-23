
# torch.export IR 规范

Export IR 是一种面向编译器的中间表示（IR），它与 [MLIR](https://mlir.llvm.org/) 和 TorchScript 有相似之处。它专门设计用于表达 PyTorch 程序的语义。Export IR 主要以精简的操作列表来表示计算，对动态性（如控制流）的支持有限。

要创建 Export IR 图，可以使用一个前端，通过追踪特化机制来可靠地捕获 PyTorch 程序。生成的 Export IR 随后可以由后端进行优化和执行。目前可以通过 `torch.export.export` 来实现。

本文档将涵盖的关键概念包括：

- ExportedProgram：包含 Export IR 程序的数据结构
- Graph：由节点列表组成。
- Nodes：表示操作、控制流以及存储在此节点上的元数据。
- Values 由节点产生和消费。
- Types 与值和节点相关联。
- 值的大小和内存布局也有定义。

## 前提假设

本文档假设读者对 PyTorch 有足够的了解，特别是熟悉 `torch.fx` 及其相关工具。因此，对于 `torch.fx` 文档和论文中已存在的内容，本文将不再赘述。

## 什么是 Export IR

Export IR 是 PyTorch 程序的基于图的中间表示（IR）。Export IR 是在 `torch.fx.Graph` 之上实现的。换句话说，**所有 Export IR 图也都是有效的 FX 图**，并且如果使用标准的 FX 语义进行解释，Export IR 可以被可靠地解释。这意味着导出的图可以通过标准的 FX 代码生成转换为有效的 Python 程序。

本文档将主要侧重于强调 Export IR 在严格性方面与 FX 的不同之处，而跳过与 FX 相似的部分。

## ExportedProgram

顶层的 Export IR 结构是 `torch.export.ExportedProgram` 类。它将 PyTorch 模型（通常是 `torch.nn.Module`）的计算图与此模型消费的参数或权重捆绑在一起。

`torch.export.ExportedProgram` 类的一些重要属性包括：

- `graph_module` (`torch.fx.GraphModule`)：包含 PyTorch 模型扁平化计算图的数据结构。可以通过 `ExportedProgram.graph` 直接访问该图。
- `graph_signature` (`torch.export.ExportGraphSignature`)：图签名，用于指定图中使用和修改的参数和缓冲区名称。参数和缓冲区不是作为图的属性存储，而是被提升为图的输入。`graph_signature` 用于跟踪这些参数和缓冲区的附加信息。
- `state_dict` (`Dict[str, Union[torch.Tensor, torch.nn.Parameter]]`)：包含参数和缓冲区的数据结构。
- `range_constraints` (`Dict[sympy.Symbol, RangeConstraint]`)：对于导出时具有数据依赖行为的程序，每个节点上的元数据将包含符号形状（看起来像 `s0`、`i0`）。此属性将符号形状映射到它们的下限/上限范围。

## Graph

Export IR Graph 是以 DAG（有向无环图）形式表示的 PyTorch 程序。图中的每个节点代表一个特定的计算或操作，图的边由节点之间的引用组成。

我们可以将 Graph 视为具有以下模式：

```python
class Graph:
  nodes: List[Node]
```

实际上，Export IR 的图是作为 `torch.fx.Graph` Python 类实现的。

一个 Export IR 图包含以下节点（节点将在下一节中详细描述）：

- 0 个或多个操作类型为 `placeholder` 的节点
- 0 个或多个操作类型为 `call_function` 的节点
- 恰好 1 个操作类型为 `output` 的节点

**推论：** 最小的有效 Graph 将只有一个节点。即，节点列表永远不会为空。

**定义：**
Graph 的 `placeholder` 节点集合代表 GraphModule 的 Graph 的**输入**。Graph 的 `output` 节点代表 GraphModule 的 Graph 的**输出**。

示例：

```python
import torch
from torch import nn

class MyModule(nn.Module):

    def forward(self, x, y):
      return x + y

example_args = (torch.randn(1), torch.randn(1))
mod = torch.export.export(MyModule(), example_args)
print(mod.graph)
```

```python
graph():
  %x : [num_users=1] = placeholder[target=x]
  %y : [num_users=1] = placeholder[target=y]
  %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %y), kwargs = {})
  return (add,)
```

以上是 Graph 的文本表示，每一行代表一个节点。

## Node

Node 代表一个特定的计算或操作，在 Python 中使用 `torch.fx.Node` 类表示。节点之间的边通过 Node 类的 `args` 属性表示为对其他节点的直接引用。使用相同的 FX 机制，我们可以表示计算图通常需要的以下操作，例如运算符调用、占位符（即输入）、条件语句和循环。

Node 具有以下模式：

```python
class Node:
  name: str # 节点名称
  op_name: str  # 操作类型

  # 下面字段的解释取决于 op_name
  target: [str|Callable]
  args: List[object]
  kwargs: Dict[str, object]
  meta: Dict[str, object]
```

**FX 文本格式**

如上例所示，请注意每一行都有以下格式：

```
%<name>:[...] = <op_name>[target=<target>](args = (%arg1, %arg2, arg3, arg4, …)), kwargs = {"keyword": arg5})
```

这种格式以紧凑的形式捕获了 Node 类中除 `meta` 之外的所有内容。

具体来说：

- **&lt;name&gt;** 是节点在 `node.name` 中显示的名称。
- **&lt;op_name&gt;** 是 `node.op` 字段，必须是以下之一：
  `<call_function>`、`<placeholder>`、
  `<get_attr>` 或 `<output>`。
- **&lt;target&gt;** 是节点的目标，即 `node.target`。此字段的含义取决于 `op_name`。
- **args1, … args 4…** 是 `node.args` 元组中列出的内容。如果列表中的值是 `torch.fx.Node`，则会特别用前缀 **%** 表示。

例如，调用加法运算符将显示为：

```
%add1 = call_function[target = torch.op.aten.add.Tensor](args = (%x, %y), kwargs = {})
```

其中 `%x`、`%y` 是另外两个名为 x 和 y 的节点。值得注意的是，字符串 `torch.op.aten.add.Tensor` 表示实际存储在目标字段中的可调用对象，而不仅仅是其字符串名称。

此文本格式的最后一行是：

```
return [add]
```

这是一个 `op_name = output` 的节点，表示我们返回这一个元素。

### call_function

`call_function` 节点表示对运算符的调用。

**定义**

- **函数式：** 如果一个可调用对象满足以下所有要求，我们称其为“函数式”：
  - 非变异：运算符不会改变其输入的值（对于张量，这包括元数据和数据）。
  - 无副作用：运算符不会改变从外部可见的状态，例如更改模块参数的值。

- **运算符：** 是具有预定义模式的函数式可调用对象。此类运算符的示例包括函数式 ATen 运算符。

**在 FX 中的表示**

```
%name = call_function[target = operator](args = (%x, %y, …), kwargs = {})
```

**与普通 FX call_function 的区别**

1. 在 FX 图中，call_function 可以引用任何可调用对象，而在导出 IR 中，我们将其限制为仅限 ATen 运算符、自定义运算符和控制流运算符的选定子集。
2. 在导出 IR 中，常量参数将嵌入在图中。
3. 在 FX 图中，get_attr 节点可以表示读取存储在图形模块中的任何属性。然而，在导出 IR 中，这仅限于仅读取子模块，因为所有参数/缓冲区都将作为输入传递给图形模块。

#### 元数据

`Node.meta` 是附加到每个 FX 节点的字典。然而，FX 规范并未指定可以或将会存在哪些元数据。导出 IR 提供了更强的约定，具体来说，所有 `call_function` 节点将保证具有且仅具有以下元数据字段：

- `node.meta["stack_trace"]` 是一个字符串，包含引用原始 Python 源代码的 Python 堆栈跟踪。堆栈跟踪示例如下：

  ```
  File "my_module.py", line 19, in forward
  return x + dummy_helper(y)
  File "helper_utility.py", line 89, in dummy_helper
  return y + 1
  ```

- `node.meta["val"]` 描述运行操作的输出。它可以是 `<symint>`、`<FakeTensor>`、`List[Union[FakeTensor, SymInt]]` 或 `None` 类型。

- `node.meta["nn_module_stack"]` 描述节点来源的 `torch.nn.Module` 的“堆栈跟踪”（如果它来自 `torch.nn.Module` 调用）。例如，如果包含 `addmm` 操作的节点是从 `torch.nn.Sequential` 模块内的 `torch.nn.Linear` 模块调用的，则 `nn_module_stack` 将类似于：

  ```
  {'self_linear': ('self.linear', <class 'torch.nn.Linear'>), 'self_sequential': ('self.sequential', <class 'torch.nn.Sequential'>)}
  ```

- `node.meta["source_fn_stack"]` 包含此节点在分解之前被调用的 torch 函数或叶子 `torch.nn.Module` 类。例如，来自 `torch.nn.Linear` 模块调用的包含 `addmm` 操作的节点将在其 `source_fn` 中包含 `torch.nn.Linear`，而来自 `torch.nn.functional.Linear` 模块调用的包含 `addmm` 操作的节点将在其 `source_fn` 中包含 `torch.nn.functional.Linear`。

### placeholder

占位符表示图的输入。其语义与 FX 中完全相同。占位符节点必须是图中节点列表的前 N 个节点。N 可以为零。

**在 FX 中的表示**

```python
%name = placeholder[target = name](args = ())
```

目标字段是一个字符串，即输入的名称。

`args` 如果非空，大小应为 1，表示此输入的默认值。

**元数据**

与 `call_function` 节点类似，占位符节点也有 `meta['val']`。在这种情况下，`val` 字段表示图为此输入参数预期接收的输入形状/数据类型。

### output

输出调用表示函数中的返回语句；因此它终止当前图。有且只有一个输出节点，并且它始终是图的最后一个节点。

**在 FX 中的表示**

```
output[](args = (%something, …))
```

这与 `torch.fx` 中的语义完全相同。`args` 表示要返回的节点。

**元数据**

输出节点具有与 `call_function` 节点相同的元数据。

### get_attr

`get_attr` 节点表示从封装的 `torch.fx.GraphModule` 中读取子模块。与来自 `torch.fx.symbolic_trace` 的普通 FX 图不同，在普通 FX 图中，`get_attr` 节点用于从顶级 `torch.fx.GraphModule` 读取属性和缓冲区等属性，而参数和缓冲区是作为输入传递给图形模块的，并存储在顶级 `torch.export.ExportedProgram` 中。

**在 FX 中的表示**

```python
%name = get_attr[target = name](args = ())
```

**示例**

考虑以下模型：

```python
from functorch.experimental.control_flow import cond

def true_fn(x):
    return x.sin()

def false_fn(x):
    return x.cos()

def f(x, y):
    return cond(y, true_fn, false_fn, [x])
```

图：

graph():
    %x_1 : [num_users=1] = placeholder[target=x_1]
    %y_1 : [num_users=1] = placeholder[target=y_1]
    %true_graph_0 : [num_users=1] = get_attr[target=true_graph_0]
    %false_graph_0 : [num_users=1] = get_attr[target=false_graph_0]
    %conditional : [num_users=1] = call_function[target=torch.ops.higher_order.cond](args = (%y_1, %true_graph_0, %false_graph_0, [%x_1]), kwargs = {})
    return conditional
```

行 `%true_graph_0 : [num_users=1] = get_attr[target=true_graph_0]` 读取了包含 `sin` 算子的子模块 `true_graph_0`。

## 参考

### SymInt

SymInt 是一个对象，它既可以是一个字面整数，也可以是一个表示整数的符号（在 Python 中由 `sympy.Symbol` 类表示）。当 SymInt 是一个符号时，它描述了一个在编译时对图来说是未知的整数类型变量，也就是说，它的值只在运行时才知道。

### FakeTensor

FakeTensor 是一个包含张量元数据的对象。可以将其视为具有以下元数据。

```python
class FakeTensor:
  size: List[SymInt]
  dtype: torch.dtype
  device: torch.device
  dim_order: List[int]  # 此字段尚不存在
```

FakeTensor 的 size 字段是一个整数或 SymInt 的列表。如果存在 SymInt，则意味着该张量具有动态形状。如果存在整数，则假定该张量将具有该确切的静态形状。TensorMeta 的秩（rank）永远不会是动态的。dtype 字段表示该节点输出的数据类型。在 Edge IR 中没有隐式的类型提升。FakeTensor 中没有步幅（strides）信息。

换句话说：

- 如果 node.target 中的算子返回一个张量，那么 `node.meta['val']` 就是一个描述该张量的 FakeTensor。
- 如果 node.target 中的算子返回一个 n 元组的张量，那么 `node.meta['val']` 就是一个描述每个张量的 n 元组 FakeTensor。
- 如果 node.target 中的算子返回一个在编译时已知的 int/float/标量，那么 `node.meta['val']` 为 None。
- 如果 node.target 中的算子返回一个在编译时未知的 int/float/标量，那么 `node.meta['val']` 的类型为 SymInt。

例如：

- `aten::add` 返回一个张量；因此其规范将是一个 FakeTensor，其 dtype 和 size 由该算子返回的张量决定。
- `aten::sym_size` 返回一个整数；因此其 val 将是一个 SymInt，因为其值仅在运行时可用。
- `max_pool2d_with_indexes` 返回一个 (Tensor, Tensor) 元组；因此规范也将是一个包含 2 个 FakeTensor 对象的 2 元组，第一个 TensorMeta 描述返回值的第一个元素，依此类推。

Python 代码：

```python
def add_one(x):
  return torch.ops.aten(x, 1)
```

图：

```
graph():
  %ph_0 : [#users=1] = placeholder[target=ph_0]
  %add_tensor : [#users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%ph_0, 1), kwargs = {})
  return [add_tensor]
```

FakeTensor：

```python
FakeTensor(dtype=torch.int, size=[2,], device=CPU)
```

### 可 Pytree 化的类型

我们定义一个类型为“可 Pytree 化的”，如果它要么是叶子类型，要么是包含其他可 Pytree 化类型的容器类型。

注意：

> pytree 的概念与 JAX 文档中记录的相同：
> [此处](https://jax.readthedocs.io/en/latest/pytrees.html)

以下类型被定义为**叶子类型**：


以下类型被定义为**容器类型**：

