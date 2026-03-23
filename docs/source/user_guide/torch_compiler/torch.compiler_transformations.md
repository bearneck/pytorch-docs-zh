# 在 ATen IR 上编写图变换

## 变换过程

由于 ATen IR 位于 FX Graph/GraphModule 层级，任何为 FX 图编写的变换都可以轻松应用于 ATen IR。如果您熟悉编写 FX 图变换，那么这将完全相同。

编写变换最直接的方式是遍历给定的图并直接操作图中的节点。

例如，假设我们想要将 `torch.ops.aten.add.Tensor()` 调用替换为 `torch.ops.aten.mul.Tensor()` 调用：

```python
import torch

def replace_add_with_mul(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
            node.target = torch.ops.aten.mul.Tensor
```

我们也可以通过 FX 工具函数来删除和添加新节点，这些函数可以在 [Graph](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) 文档中找到。例如，如果我们想在 `add` 调用后插入一个 `torch.ops.aten.relu.default()`：

```python
import torch

def insert_relu_after_add(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:

            # 指定插入点。在此作用域内添加到图中的任何节点都将插入在 `node` 之后
            with gm.graph.inserting_after(node):
                # 插入一个新的 `call_function` 节点，操作为 `torch.ops.aten.relu.default`
                new_relu_node = gm.graph.call_function(torch.ops.aten.relu.default, args=(node,))
                # 将所有使用 `node` 的地方替换为使用 `new_relu_node`
                node.replace_all_uses_with(new_relu_node)
```

通常，变换可以大致按几个维度分类：

维度 A：1. 创建一对多映射（例如分解） 2. 创建多对一映射（例如融合）

维度 B：1. 进行前向迭代（例如形状传播） 2. 进行反向迭代（例如死代码消除）

维度 C：1. 依赖于局部节点信息（例如输出变体转换） 2. 依赖于全局图信息（例如内存规划）

我们对这些用例频率的预测是：1. A.1, B.1, C.1 2. A.2 3. B.2, C.2

虽然我们可以通过直接操作图来进行所有图变换，但我们也为第 1 级和第 2 级用例提供了一些辅助工具以方便使用。

### Transformer

对于第 1 级用例（创建一对多映射、进行前向迭代以及查看局部节点信息），我们可以利用 [Transformer](https://pytorch.org/docs/stable/fx.html#torch.fx.Transformer) 类来执行每个节点并重新创建图，但应用指定的变换。

#### 一对一变换

对于一对一映射的示例，如果我们想将操作 A 替换为另一个操作 B，我们可以运行 GraphModule，每次看到操作 A 时，返回操作 B。

示例如下：

```python
class ReplaceAddWithMul(torch.fx.Transformer):
    def call_function(self, target, args, kwargs):
        if target != torch.ops.aten.add.Tensor:
            return super().call_function(target, args, kwargs)
        return super().call_function(torch.ops.aten.mul.Tensor, args, kwargs)

transformed_graph_module = ReplaceAddWithMul(graph_module).transform()
```

`super().call_function(target, args, kwargs, meta)` 调用会创建一个 `call_function` FX 节点，并返回使用给定参数运行操作符的结果。

#### 一对多变换

如果我们想进行一对多映射，例如将操作 A 替换为另外两个操作 B 和 C，那么我们将进行两次 `super().call_function` 调用来创建两个 FX 节点，一个用于操作 B，另一个用于操作 C，并返回运行操作 C 的结果。

例如：

```python
class ReplaceAddWithMulSub(torch.fx.Transformer):
    """
    原始：
        def f(x, y):
            return x + y

    变换后：
        def f(x, y):
            z = x * y
            return z - y
    """
    def call_function(self, target, args, kwargs):
        if target != torch.ops.aten.add.Tensor:
            return super().call_function(target, args, kwargs)

        x, y = args

        mul_res = super().call_function(torch.ops.aten.mul.Tensor, args, {})
        return super().call_function(torch.ops.aten.sub.Tensor, (mul_res, y), {})

transformed_graph_module = ReplaceAddWithMulSub(graph_module).transform()
```

#### 一对零变换

如果我们想删除一个操作，可以直接返回传入函数的值：

```python
class RemoveDetachPass(torch.fx.Transformer):
    def call_function(self, target, args, kwargs):
        if target not in (
            torch.ops.aten.detach.default,
            torch.ops.aten.detach_copy.default,
        ):
            return super().call_function(target, args, kwargs, meta)

        assert len(args) == 1
        return args[0]

transformed_graph_module = RemoveDetachPass(graph_module).transform()
```

#### 利用局部信息

利用局部节点信息的一个例子是，如果我们想将图中的所有标量转换为张量，我们可以运行给定的 `fx.GraphModule`，对于每个包含标量的参数，将其转换为张量。可能如下所示：

```python
def args_map(target, fn, args, kwargs):
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    args = list(args)
    kwargs = kwargs.copy()

    # 根据传入的函数更新参数
    def update(key, args, schema):
        args[key] = fn(args[key], schema)

    # 更新模式中的每个参数
    for i, schema in enumerate(target._schema.arguments):
        if schema.name in kwargs:
            update(schema.name, kwargs, schema)
        elif not schema.kwarg_only and i < len(args):
            update(i, args, schema)
    return tuple(args), kwargs
```

# torch.compiler 变换

## 概述

`torch.compiler` 提供了多种变换，可用于修改 FX 图。这些变换包括：

- **FX 变换器**：用于修改 FX 图的基类。
- **子图重写器**：用于将图中的子图替换为其他子图。
- **Pass 管理器**：用于在图上运行一系列变换。
- **分区器**：用于将图划分为多个子图。

## FX 变换器

`torch.fx.Transformer` 是一个基类，可用于修改 FX 图。它提供了一种通过重写节点来变换图的方法。要使用它，请继承 `torch.fx.Transformer` 并重写 `call_function`、`call_module` 和 `call_method` 方法。

例如，以下变换器将所有标量输入转换为张量：

```python
class ScalarToTensorPass(torch.fx.Transformer):
    def call_function(self, target, args, kwargs):
        breakpoint()
        def try_coerce(value, arg):
            return (
                torch.tensor(value)
                if isinstance(value, (float, int, bool))
                and type(arg.type) == torch.TensorType
                else value
            )

        args, kwargs = args_map(target, try_coerce, args, kwargs)
        return super().call_function(target, args, kwargs)

transformed_graph_module = ScalarToTensorPass(graph_module).transform()
```

### 子图重写器

对于创建多对一的映射，我们可以利用 FX 的[子图重写器](https://github.com/pytorch/pytorch/blob/main/torch/fx/subgraph_rewriter.py)。给定一个 `pattern`，它会创建一个与模式匹配的运算符子图，然后用 `replacement` 替换每个匹配的子图。

注意：

```
这是一个原地操作。
```

`pattern` 和 `replacement` 输入必须是可调用函数或包含图中使用的相同运算符（ATen 操作）的 GraphModules，以便子图重写器能够在图中找到正确的模式。模式/替换可调用对象的输入在匹配时将被视为通配符。

示例：

```python
from torch.fx import subgraph_rewriter

def replace_patterns(graph_module):
    def pattern(x, y):
        x = torch.ops.aten.add.Tensor(x, y)
        x = torch.ops.aten.mul.Tensor(x, y)
        return x

    def replacement(x, y):
        return torch.ops.aten.sub.Tensor(x, y)

replaced_patterns = subgraph_rewriter.replace_pattern_with_filters(
    traced_module, pattern, replacement
)
```

子图重写器返回一个 `ReplacedPatterns` 列表：

```python
@dataclass
class ReplacedPatterns:
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node]
    # List of nodes that were added into the graph
    replacements: List[Node]
```

注意：

```
子图重写器创建的节点将不包含匹配节点中填充的元数据，但您可以使用 `ReplacedPatterns.nodes_map` 来查找原始图中匹配的节点，并使用 `ReplacedPatterns.replacements` 来查找变换图中被替换的节点。
```

## Pass 管理器

[PassManager](https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/infra/pass_manager.py) 是一个用于在给定图模块上运行多个 pass 的类。初始化 `PassManager` 实例时，我们传入要运行的 pass 列表并设置几个标志。要在图模块上运行这组 pass，我们可以直接将图模块传递给 `PassManager` 实例。

示例：

```python
from torch.fx.passes.infra.pass_manager import PassManager

pm = PassManager(
    passes=[replace_add_with_div, replace_div_with_mul],
    run_checks_after_each_pass=True,
    suppress_check_failures=False,
)
graph_module_out = pm(graph_module)
```

要添加在每个 pass 后运行的通用检查集，我们可以调用函数 `set_checks(check: Callable)`，该函数接受一个可调用函数作为输入。如果设置了 `run_checks_after_each_pass` 标志，则 `check` 将在每个 pass 在图模块上运行后被调用。

示例：

```python
pm = PassManager(passes=[replace_add_with_div, replace_div_with_mul])

def check_div_target(graph_module):
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target != torch.div:
            raise ValueError("Target should be div!")

pm.add_checks(check_div_target)

pm(graph_module)    # raises ValueError after replace_div_with_mul pass
```

## 分区器

有几个常用的基于 FX 图的分区器可用于对图进行分区。

### 子图匹配器

为了在图中查找匹配特定模式的子图，我们可以利用 FX 的 [`SubgraphMatcher`](https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/utils/matcher_utils.py)。

类属性：

- `pattern (Graph)`：目标匹配模式。图中的占位符节点在匹配时将被视为通配符。
- `match_output (bool)`：如果为 True，模式图中的输出节点将被视为目标模式的一部分。如果为 False，输出节点在匹配期间将被忽略。
- `match_placeholder (bool)`：如果为 True，模式图中的占位符节点将被视为目标模式的一部分。如果为 False，占位符节点将被用作通配符。
- `remove_overlapping_matches (bool)`：如果为 True，在重叠匹配的情况下，只返回第一个匹配。
- `ignore_literals (bool)`：如果为 True，将不检查字面量是否相等，而是将它们视为通配符。

示例：

```python
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

class LargeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._weight = torch.nn.Parameter(torch.ones(3, 3))
        self._bias = torch.nn.Parameter(torch.ones(3, 3))

    def forward(self, x):
        return torch.ops.aten.addmm.default(self._bias, x, self._weight)

large_model_graph = torch.export(LargeModel(), inputs).graph

class PatternModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._weight_1 = torch.nn.Parameter(torch.ones(5, 5))
        self._bias_1 = torch.nn.Parameter(torch.ones(5, 5))

    def forward(self, x):
        return torch.ops.aten.addmm.default(self._bias_1, x, self._weight_1)

pattern_graph = torch.export(PatternModel(), inputs).graph

subgraph_matcher = SubgraphMatcher(pattern_graph)
match_result = subgraph_matcher.match(large_model_graph)
```

`match` 函数返回一个 `InternalMatch` 列表：

```python
@dataclass
class InternalMatch():
    # 从中找到匹配的节点
    anchors: List[Node]
    # 将模式子图中的节点映射到更大图中的节点
    nodes_map: Dict[Node, Node] = field(default_factory=dict)
    # 目标图中与模式中占位符匹配的节点
    placeholder_nodes: List[Node] = field(default_factory=list)
    # 匹配子图中由输出返回的节点
    returning_nodes: List[Node] = field(default_factory=list)
```

### 基于能力的划分器

为了找到支持特定不变量的最大节点子图，我们可以利用 FX 的
[`CapabilityBasedPartitioner`](https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/infra/partitioner.py#L34)。

类属性

- `graph_module (torch.fx.GraphModule)`：我们正在划分的图模块。
- `operator_support (OperatorSupportBase)`：用于确定图中节点是否在划分中受支持的对象。
- `allows_single_node_partition (bool)`：如果为 True，则允许形成单节点划分。
- `non_compute_ops (Optional[Sequence[str]])`：被视为“非计算”操作的一组操作（例如 `torch.ops.aten.view` 和 `_operator.getitem`），这样划分器就不会创建仅包含这些非计算操作的图。
- `allowed_single_node_partition_ops (Optional[Sequence[str]])`：允许出现在单节点划分中的一组操作。

划分器使用
[`OperatorSupportBase`](https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/operator_support.py#LL28C1-L28C1)
类来确定图中的特定节点是否属于该划分。这是通过重写 `is_node_supported` 函数来实现的。您可以通过使用
[`chain`](https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/operator_support.py#L150)（如果任何 OperatorSupportBase 返回 False，则返回 False）和
[`any_chain`](https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/operator_support.py#L164)
（如果任何 OperatorSupportBase 返回 True，则返回 True）来链接多个 `OperatorSupportBase`。

示例：

```python
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import any_chain, OperatorSupportBase

class AddMulOperatorSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target in [
            torch.ops.aten.add.Tensor, torch.ops.aten.mul.Tensor,
        ]

capability_partitioner = CapabilityBasedPartitioner(
    graph_module,
    op_support,
)

# 返回划分列表（每个划分中属于的节点列表）
partition_list = capability_partitioner.propose_partitions()
# 将划分融合为图模块，并在图中插入 `call_module` 节点
fused_graph_module = capability_partitioner.fuse_partitions(partition_list)
```
