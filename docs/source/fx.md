


# torch.fx

## 概述


## 编写变换

什么是 FX 变换？本质上，它是一个如下所示的函数。

```python

import torch
import torch.fx

def transform(m: nn.Module,
                tracer_class : type = torch.fx.Tracer) -> torch.nn.Module:
    # 步骤 1：获取表示 `m` 中代码的 Graph

    # 注意：torch.fx.symbolic_trace 是对 fx.Tracer.trace 调用和构造 GraphModule 的封装。
    # 我们将在变换中将其拆分，以允许调用者自定义追踪行为。
    graph : torch.fx.Graph = tracer_class().trace(m)

    # 步骤 2：修改此 Graph 或创建一个新的 Graph
    graph = ...

    # 步骤 3：构造要返回的 Module
    return torch.fx.GraphModule(m, graph)
```

你的变换将接收一个 `torch.nn.Module`，从中获取一个 `Graph`，进行一些修改，然后返回一个新的 `torch.nn.Module`。你应该将 FX 变换返回的 `torch.nn.Module` 视为与常规 `torch.nn.Module` 完全相同——你可以将其传递给另一个 FX 变换，也可以运行它。确保 FX 变换的输入和输出是 `torch.nn.Module` 将保证可组合性。

```{note}

也可以修改现有的 `GraphModule` 而不是创建一个新的，如下所示：

```python
import torch
import torch.fx

def transform(m : nn.Module) -> nn.Module:
    gm : torch.fx.GraphModule = torch.fx.symbolic_trace(m)

    # 修改 gm.graph
    # <...>

    # 从其 Graph 重新编译 `gm` 的 forward() 方法
    gm.recompile()

    return gm
```

注意，你必须调用 `GraphModule.recompile` 以使 `GraphModule` 上生成的 `forward()` 方法与修改后的 `Graph` 保持同步。

既然你已经传入了一个已被追踪为 `Graph` 的 `torch.nn.Module`，现在有两种主要方法可以构建一个新的 `Graph`。

### Graph 快速入门

关于图语义的完整介绍可以在 `Graph` 文档中找到，但我们将在此介绍基础知识。`Graph` 是一种表示 `GraphModule` 上方法的数据结构。这需要的信息是：

- 方法的输入是什么？
- 方法内部运行哪些操作？
- 方法的输出（即返回值）是什么？

这三个概念都用 `Node` 实例表示。让我们通过一个简短的示例来理解其含义：

```python

import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return torch.topk(torch.sum(
            self.linear(x + self.linear.weight).relu(), dim=-1), 3)

m = MyModule()
gm = torch.fx.symbolic_trace(m)

gm.graph.print_tabular()
```

这里我们定义了一个用于演示的模块 `MyModule`，实例化它，进行符号追踪，然后调用 `Graph.print_tabular` 方法打印出此 `Graph` 节点的表格：

| opcode | name | target | args | kwargs |
|--------|------|--------|------|--------|
| placeholder | x | x | () | {} |
| get_attr | linear_weight | linear.weight | () | {} |
| call_function | add_1 | | (x, linear_weight) | {} |
| call_module | linear_1 | linear | (add_1,) | {} |
| call_method | relu_1 | relu | (linear_1,) | {} |
| call_function | sum_1 | <built-in method sum ...> | (relu_1,) | {'dim': -1} |
| call_function | topk_1 | <built-in method topk ...> | (sum_1, 3) | {} |
| output | output | output | (topk_1,) | {} |

我们可以使用这些信息来回答上面提出的问题。

- 方法的输入是什么？在 FX 中，方法输入通过特殊的 `placeholder` 节点指定。在本例中，我们有一个 `target` 为 `x` 的 `placeholder` 节点，这意味着我们有一个名为 x 的（非 self）参数。
- 方法内部的操作是什么？`get_attr`、`call_function`、`call_module` 和 `call_method` 节点表示方法中的操作。所有这些节点的完整语义可以在 `Node` 文档中找到。
- 方法的返回值是什么？`Graph` 中的返回值由特殊的 `output` 节点指定。

既然我们现在了解了代码在 FX 中表示的基础知识，现在可以探索如何编辑 `Graph`。

### 图操作

#### 直接图操作

构建这个新 `Graph` 的一种方法是直接操作旧的 Graph。为此，我们可以简单地获取从符号追踪得到的 `Graph` 并修改它。例如，假设我们希望将 `torch.add` 调用替换为 `torch.mul` 调用。

```python

import torch
import torch.fx

# 示例模块
class M(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)

def transform(m: torch.nn.Module,
                tracer_class : type = fx.Tracer) -> torch.nn.Module:
    graph : fx.Graph = tracer_class().trace(m)
    # FX 将其 Graph 表示为一个有序的节点列表，因此我们可以遍历它们。
    for node in graph.nodes:
        # 检查我们是否在调用一个函数（例如：torch.add）
        if node.op == 'call_function':
            # target 属性是 call_function 调用的函数。
            if node.target == torch.add:
                node.target = torch.mul

    graph.lint() # 执行一些检查以确保 Graph 格式正确。

    return fx.GraphModule(m, graph)
```

我们还可以进行更复杂的 `Graph` 重写，例如删除或追加节点。为了辅助这些转换，FX 提供了一些用于变换图的实用函数，可以在 `Graph` 文档中找到。下面是一个使用这些 API 来追加一个 `torch.relu` 调用的示例。

```python
# 指定插入点。在此作用域内添加到 Graph 的任何节点都将在 `node` 之后插入
with traced.graph.inserting_after(node):
    # 插入一个新的 `call_function` 节点，调用 `torch.relu`
    new_node = traced.graph.call_function(
        torch.relu, args=(node,))

    # 我们希望所有使用 `node` 值的地方现在都使用我们添加的 `relu` 调用之后的值。
    # 我们使用 `replace_all_uses_with` API 来实现这一点。
    node.replace_all_uses_with(new_node)
```

对于仅包含替换的简单转换，您也可以使用 [子图重写器。](https://github.com/pytorch/pytorch/blob/main/torch/fx/subgraph_rewriter.py)

#### 使用 replace_pattern() 进行子图重写

FX 在直接图操作之上还提供了另一层自动化。`replace_pattern` API 本质上是一个用于编辑 `Graph` 的“查找/替换”工具。它允许您指定一个 `pattern` 和一个 `replacement` 函数，然后它会追踪这些函数，在 `pattern` 图中找到该组操作的实例，并用 `replacement` 图的副本替换这些实例。这有助于极大地自动化繁琐的图操作代码，随着转换变得更加复杂，这些代码可能会变得难以管理。

#### 图操作示例

-  [替换一个操作](https://github.com/pytorch/examples/blob/master/fx/replace_op.py)
-  [卷积/批量归一化融合](https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L50)
-  [replace_pattern：基本用法](https://github.com/pytorch/examples/blob/master/fx/subgraph_rewriter_basic_use.py)
-  [量化](https://pytorch.org/docs/main/quantization.html#prototype-fx-graph-mode-quantization)
-  [反转变换](https://github.com/pytorch/examples/blob/master/fx/invert.py)

### Proxy/重新追踪

另一种操作 `Graph` 的方法是重用符号追踪中使用的 `Proxy` 机制。例如，假设我们想编写一个将 PyTorch 函数分解为更小操作的转换。它将把每个 `F.relu(x)` 调用转换为 `(x > 0) * x`。一种可能的方法是执行必要的图重写，在 `F.relu` 之后插入比较和乘法操作，然后清理原始的 `F.relu`。但是，我们可以通过使用 `Proxy` 对象自动将操作记录到 `Graph` 中来自动化这个过程。

要使用此方法，我们将要插入的操作编写为常规的 PyTorch 代码，并使用 `Proxy` 对象作为参数调用该代码。这些 `Proxy` 对象将捕获对它们执行的操作，并将其追加到 `Graph` 中。

```python
# 请注意，这个分解规则可以像常规 Python 代码一样阅读
def relu_decomposition(x):
    return (x > 0) * x

decomposition_rules = {}
decomposition_rules[F.relu] = relu_decomposition

def decompose(model: torch.nn.Module,
                tracer_class : type = fx.Tracer) -> torch.nn.Module:
    """
    将 `model` 分解为更小的组成操作。
    目前，这仅支持将 ReLU 分解为其数学定义：(x > 0) * x
    """
    graph : fx.Graph = tracer_class().trace(model)
    new_graph = fx.Graph()
    env = {}
    tracer = torch.fx.proxy.GraphAppendingTracer(new_graph)
    for node in graph.nodes:
        if node.op == 'call_function' and node.target in decomposition_rules:
            # 通过用代理包装参数，
            # 我们可以分派到适当的分解规则，
            # 并通过符号追踪将其隐式添加到 Graph 中。
            proxy_args = [
                fx.Proxy(env[x.name], tracer) if isinstance(x, fx.Node) else x for x in node.args]
            output_proxy = decomposition_rules[node.target](*proxy_args)

            # 对 `Proxy` 的操作总是产生新的 `Proxy`，
            # 我们的分解规则的返回值也不例外。
            # 我们需要从 `Proxy` 中提取底层的 `Node`，
            # 以便在此转换的后续迭代中使用它。
            new_node = output_proxy.node
            env[node.name] = new_node
        else:
            # 默认情况：我们没有此节点的分解规则，
            # 因此只需将节点复制到新图中。
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
    return fx.GraphModule(model, new_graph)
```

除了避免显式的图操作之外，使用 `Proxy` 还允许您将重写规则指定为原生 Python 代码。对于需要大量重写规则的转换（例如 vmap 或 grad），这通常可以提高规则的可读性和可维护性。请注意，在调用 `Proxy` 时，我们还传递了一个指向底层变量 `graph` 的追踪器。这样做是为了确保如果图中的操作是 n 元的（例如 add 是二元运算符），对 `Proxy` 的调用不会创建多个图追踪器实例，否则可能导致意外的运行时错误。我们推荐这种使用 `Proxy` 的方法，尤其是在不能安全地假设底层运算符是一元的情况下。

一个使用 `Proxy` 进行 `Graph` 操作的实际示例可以在[这里](https://github.com/pytorch/examples/blob/master/fx/proxy_based_graph_creation.py)找到。

### 解释器模式

FX 中一个有用的代码组织模式是循环遍历 `Graph` 中的所有 `Node` 并执行它们。这可以用于多种用途，包括运行时分析流经图的值，或通过使用 `Proxy` 重新跟踪来转换代码。例如，假设我们想要运行一个 `GraphModule`，并在运行时记录节点上看到的 `torch.Tensor` 形状和 dtype 属性。这可能如下所示：

```python

import torch
import torch.fx
from torch.fx.node import Node

from typing import Dict

class ShapeProp:
    """
    形状传播。此类接收一个 `GraphModule`。
    然后，其 `propagate` 方法使用给定的参数逐节点执行 `GraphModule`。
    当每个操作执行时，ShapeProp 类将每个操作输出值的形状和元素类型
    存储在该操作 `Node` 的 `shape` 和 `dtype` 属性上。
    """
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':

                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            # 这是特定于形状传播的唯一代码。
            # 你可以删除这个 `if` 分支，它就会变成一个
            # 通用的 GraphModule 解释器。
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype

            env[node.name] = result

        return load_arg(self.graph.result)
```

如你所见，一个完整的 FX 解释器并不复杂，但非常有用。为了简化这种模式的使用，我们提供了 `Interpreter` 类，它封装了上述逻辑，使得解释器执行的某些方面可以通过方法重写来覆盖。

除了执行操作，我们还可以通过向解释器提供 `Proxy` 值来生成一个新的 `Graph`。类似地，我们提供了 `Transformer` 类来封装这种模式。`Transformer` 的行为类似于 `Interpreter`，但你不是调用 `run` 方法从模块获取具体的输出值，而是调用 `Transformer.transform` 方法来返回一个新的 `GraphModule`，该模块已受到你作为重写方法安装的任何转换规则的影响。

#### 解释器模式示例

-  [ShapePropagation](https://github.com/pytorch/pytorch/blob/master/torch/fx/passes/shape_prop.py)
-  [Performance Profiler](https://github.com/pytorch/tutorials/pull/1319)


## 调试

### 简介

在编写转换的过程中，我们的代码常常不会完全正确。在这种情况下，我们可能需要进行一些调试。关键是要逆向工作：首先，检查调用生成模块的结果，以证明或反驳正确性。然后，检查并调试生成的代码。接着，调试导致生成代码的转换过程。

如果你不熟悉调试器，请参阅辅助章节 `Available-Debuggers`。

### 转换编写中的常见陷阱

* 非确定性的 `set` 迭代顺序。在 Python 中，`set` 数据类型是无序的。例如，使用 `set` 来包含像 `Node` 这样的对象集合，可能会导致意外的非确定性。一个例子是迭代一组 `Node` 以将它们插入到 `Graph` 中。因为 `set` 数据类型是无序的，输出程序中操作的顺序将是非确定性的，并且可能在程序调用之间发生变化。推荐的替代方法是使用 `dict` 数据类型，该类型从 Python 3.7（以及 cPython 3.6）开始是[插入有序的](https://mail.python.org/pipermail/python-dev/2017-December/151283.html)。可以通过将要去重的值存储在 `dict` 的键中，来等效地使用 `dict` 作为集合。

### 检查模块的正确性

由于大多数深度学习模块的输出由浮点型 `torch.Tensor` 实例组成，检查两个 `torch.nn.Module` 的结果之间的等价性并不像进行简单的相等性检查那样直接。为了说明这一点，我们使用一个例子：

```python

import torch
import torch.fx
import torchvision.models as models

def transform(m : torch.nn.Module) -> torch.nn.Module:
    gm = torch.fx.symbolic_trace(m)

    # 假设我们在这里进行一些转换
    # <...>

    gm.recompile()

    return gm

resnet18 = models.resnet18()
transformed_resnet18 = transform(resnet18)

input_image = torch.randn(5, 3, 224, 224)

assert resnet18(input_image) == transformed_resnet18(input_image)
"""
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
"""
```

这里，我们尝试使用 `==` 相等运算符来检查两个深度学习模型的值是否相等。然而，这并不明确，既因为该运算符返回的是张量而非布尔值，也因为浮点数值的比较应使用误差容限（或 epsilon）来考虑浮点运算的非交换性（更多细节请参见[此处](https://floating-point-gui.de/errors/comparison/)）。我们可以改用 `torch.allclose`，它会根据相对和绝对容限阈值给出近似比较：

```python
assert torch.allclose(resnet18(input_image), transformed_resnet18(input_image))
```
这是我们工具箱中第一个用于检查转换后的模块是否与参考实现行为一致的工具。

### 调试生成的代码

由于 FX 在 `GraphModule` 上生成 `forward()` 函数，使用传统的调试技术如 `print` 语句或 `pdb` 就不那么直接了。幸运的是，我们有几种技术可以用来调试生成的代码。

#### 使用 `pdb`
调用 `pdb` 进入运行中的程序。尽管表示 `Graph` 的代码不在任何源文件中，但当执行前向传播时，我们仍然可以使用 `pdb` 手动进入它。

```python

import torch
import torch.fx
import torchvision.models as models

def my_pass(inp: torch.nn.Module, tracer_class : type = fx.Tracer) -> torch.nn.Module:
    graph = tracer_class().trace(inp)
    # 转换逻辑写在这里
    # <...>

    # 返回新的 Module
    return fx.GraphModule(inp, graph)

my_module = models.resnet18()
my_module_transformed = my_pass(my_module)

input_value = torch.randn(5, 3, 224, 224)

# 当这一行在运行时执行时，我们将进入一个交互式的 `pdb` 提示符。我们可以使用 `step` 或 `s` 命令进入下一行的执行
import pdb; pdb.set_trace()

my_module_transformed(input_value)
```

#### 打印生成的代码
如果你想多次运行相同的代码，那么使用 `pdb` 逐步执行到正确的代码可能会有点繁琐。在这种情况下，一种方法是简单地将生成的 `forward` 传递代码复制粘贴到你的代码中，然后在那里进行检查。

```python

# 假设 `traced` 是一个经过若干次转换的 GraphModule

# 复制此代码供后续使用
print(traced)
# 打印符号追踪生成的代码。输出如下：
"""
def forward(self, y):
    x = self.x
    add_1 = x + y;  x = y = None
    return add_1
"""

# 子类化原始 Module
class SubclassM(M):
    def __init__(self):
        super().__init__()

    # 将生成的 `forward` 函数（我们上面打印并复制的那个）粘贴到这里
    def forward(self, y):
        x = self.x
        add_1 = x + y;  x = y = None
        return add_1

# 创建原始、未追踪的 Module 实例。然后，创建带有复制 `forward` 函数的 Module 实例。我们现在可以比较原始版本和追踪版本的输出。
pre_trace = M()
post_trace = SubclassM()
```
#### 使用 `GraphModule` 中的 `to_folder` 函数
`GraphModule.to_folder` 是 `GraphModule` 中的一个方法，允许你将生成的 FX 代码转储到一个文件夹中。尽管如 `打印生成的代码` 中所示，将前向传递代码复制到代码中通常就足够了，但使用 `to_folder` 可能更容易检查模块和参数。

```python

m = symbolic_trace(M())
m.to_folder("foo", "Bar")
from foo import Bar
y = Bar()
```
运行上述示例后，我们可以查看 `foo/module.py` 中的代码，并根据需要修改它（例如添加 `print` 语句或使用 `pdb`）来调试生成的代码。

### 调试转换过程

既然我们已经确定某个转换正在生成错误的代码，那么是时候调试转换本身了。首先，我们将查阅文档中的 `符号追踪的限制` 部分。一旦我们确认追踪按预期工作，目标就变成了找出在 `GraphModule` 转换过程中出了什么问题。`编写转换` 中可能有一个快速的答案，但如果没有，有几种方法可以检查我们追踪的模块：

```python

# 示例 Module
class M(torch.nn.Module):
    def forward(self, x, y):
        return x + y

# 创建 `M` 的实例
m = M()

# 符号追踪 `M` 的实例（返回一个 GraphModule）。在这个例子中，我们只讨论如何检查 GraphModule，因此为了简洁起见，不展示任何示例转换。
traced = symbolic_trace(m)

# 打印追踪模块生成的代码。
print(traced)
# 生成的 `forward` 函数是：
"""
def forward(self, x, y):
    add = x + y;  x = y = None
    return add
"""

# 打印内部 Graph。
print(traced.graph)
# 此打印输出返回：
"""
graph():
    %x : [num_users=1] = placeholder[target=x]
    %y : [num_users=1] = placeholder[target=y]
    %add : [num_users=1] = call_function[target=operator.add](args = (%x, %y), kwargs = {})
    return add
"""

# 打印内部 Graph 的表格表示。
traced.graph.print_tabular()
# 这给我们：
"""
opcode         name    target                   args    kwargs
-------------  ------  -----------------------  ------  --------
placeholder    x       x                        ()      {}
placeholder    y       y                        ()      {}
call_function  add     <built-in function add>  (x, y)  {}
output         output  output                   (add,)  {}
"""
```
使用上述实用函数，我们可以比较应用转换前后追踪的 Module。有时，简单的视觉比较就足以追踪到错误。如果仍然不清楚哪里出了问题，像 `pdb` 这样的调试器可能是下一步的好选择。

基于上面的例子，考虑以下代码：

```python

# 用户自定义函数示例
def transform_graph(module: torch.nn.Module, tracer_class : type = fx.Tracer) -> torch.nn.Module:
    # 从跟踪的 Module 中获取 Graph
    g = tracer_class().trace(module)

    """
    在此处对 `g` 进行变换
    """

    return fx.GraphModule(module, g)

# 变换 Graph
transformed = transform_graph(traced)

# 打印变换后的新代码。检查是否符合预期
print(transformed)
```
使用上述示例，假设调用 `print(traced)` 显示我们的变换存在错误。我们希望通过调试器找出问题所在。我们启动一个 `pdb` 会话。可以通过在 `transform_graph(traced)` 处设置断点，然后按 `s` 键“步入” `transform_graph(traced)` 调用来观察变换过程中的情况。

我们也可以通过编辑 `print_tabular` 方法来打印 Graph 中节点的不同属性。（例如，我们可能希望查看节点的 `input_nodes` 和 `users`。）


### 可用的调试器

最常用的 Python 调试器是 [pdb](https://docs.python.org/3/library/pdb.html)。可以通过在命令行输入 `python -m pdb FILENAME.py` 以“调试模式”启动程序，其中 `FILENAME` 是要调试的文件名。之后，可以使用 [pdb 调试器命令](https://docs.python.org/3/library/pdb.html#debugger-commands)逐步执行正在运行的程序。通常的做法是在启动 `pdb` 时设置断点（`b LINE-NUMBER`），然后调用 `c` 运行程序直到该断点。这样可以避免必须逐步执行每一行代码（使用 `s` 或 `n`）才能到达要检查的代码部分。或者，可以在要中断的行之前写入 `import pdb; pdb.set_trace()`。如果添加了 `pdb.set_trace()`，程序在运行时将自动以调试模式启动。（换句话说，只需在命令行输入 `python FILENAME.py` 而不是 `python -m pdb FILENAME.py`。）一旦以调试模式运行文件，就可以使用特定命令逐步执行代码并检查程序的内部状态。网上有许多优秀的 `pdb` 教程，包括 RealPython 的 [“Python Debugging With Pdb”](https://realpython.com/python-debugging-pdb/)。

像 PyCharm 或 VSCode 这样的 IDE 通常内置了调试器。在 IDE 中，可以选择：a) 通过在 IDE 中打开终端窗口（例如 VSCode 中的 View → Terminal）使用 `pdb`；或者 b) 使用内置调试器（通常是 `pdb` 的图形化包装器）。


## 符号跟踪的局限性

FX 使用**符号跟踪**（也称为[符号执行](https://en.wikipedia.org/wiki/Symbolic_execution)）系统，以可变换/可分析的形式捕获程序的语义。该系统是**跟踪**的，因为它执行程序（实际上是 `torch.nn.Module` 或函数）来记录操作。它是**符号**的，因为在此执行期间流经程序的数据不是真实数据，而是符号（FX 术语中的 `Proxy`）。

尽管符号跟踪适用于大多数神经网络代码，但它有一些局限性。

### 动态控制流

符号跟踪的主要局限性是它目前不支持*动态控制流*。即，条件可能依赖于程序输入值的循环或 `if` 语句。

例如，让我们检查以下程序：

```python

def func_to_trace(x):
    if x.sum() > 0:
        return torch.relu(x)
    else:

        return torch.neg(x)

traced = torch.fx.symbolic_trace(func_to_trace)
"""
    <...>
    File "dyn.py", line 6, in func_to_trace
    if x.sum() > 0:
    File "pytorch/torch/fx/proxy.py", line 155, in __bool__
    return self.tracer.to_bool(self)
    File "pytorch/torch/fx/proxy.py", line 85, in to_bool
    raise TraceError('symbolically traced variables cannot be used as inputs to control flow')
torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow
"""
```
`if` 语句的条件依赖于 `x.sum()` 的值，而 `x.sum()` 又依赖于函数输入 `x` 的值。由于 `x` 可以改变（即，如果将新的输入张量传递给跟踪函数），这就是*动态控制流*。回溯会向上追溯代码，显示这种情况发生的位置。

### 静态控制流

另一方面，支持所谓的*静态控制流*。静态控制流是指其值在多次调用中不会改变的循环或 `if` 语句。通常，在 PyTorch 程序中，这种控制流出现在基于超参数对模型架构做出决策的代码中。具体示例如下：

```python

import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self, do_activation : bool = False):
        super().__init__()
        self.do_activation = do_activation
        self.linear = torch.nn.Linear(512, 512)

    def forward(self, x):
        x = self.linear(x)
        # 这个 if 语句就是所谓的静态控制流。
        # 它的条件不依赖于任何输入值
        if self.do_activation:
            x = torch.relu(x)
        return x

without_activation = MyModule(do_activation=False)
with_activation = MyModule(do_activation=True)

traced_without_activation = torch.fx.symbolic_trace(without_activation)
print(traced_without_activation.code)
"""
def forward(self, x):
    linear_1 = self.linear(x);  x = None
    return linear_1
"""
```

traced_with_activation = torch.fx.symbolic_trace(with_activation)
print(traced_with_activation.code)
"""
import torch
def forward(self, x):
    linear_1 = self.linear(x);  x = None
    relu_1 = torch.relu(linear_1);  linear_1 = None
    return relu_1
"""
```
if 语句 `if self.do_activation` 不依赖于任何函数输入，因此它是静态的。`do_activation` 可以被视为一个超参数，并且具有不同参数值的 `MyModule` 的不同实例的追踪会产生不同的代码。这是一种受符号追踪支持的有效模式。

许多动态控制流的实例在语义上是静态控制流。通过移除对输入值的数据依赖，可以使这些实例支持符号追踪，例如将值移动到 `Module` 属性中，或者在符号追踪期间将具体值绑定到参数上：

```python

def f(x, flag):
    if flag: return x
    else: return x*2

fx.symbolic_trace(f) # 失败！

fx.symbolic_trace(f, concrete_args={'flag': True})
```
对于真正的动态控制流，包含此代码的程序部分可以追踪为对方法（参见 `自定义追踪`）或函数（参见 `wrap`）的调用，而不是直接追踪其内部代码。

### 非 `torch` 函数

FX 使用 `__torch_function__` 作为其拦截调用的机制（有关此机制的更多信息，请参阅[技术概述](https://github.com/pytorch/pytorch/blob/main/torch/fx/README.md#technical-details)）。某些函数，例如内置的 Python 函数或 `math` 模块中的函数，不在 `__torch_function__` 的覆盖范围内，但我们仍然希望在符号追踪中捕获它们。例如：

```python

import torch
import torch.fx
from math import sqrt

def normalize(x):
    """
    通过批次维度的大小对 `x` 进行归一化
    """
    return x / sqrt(len(x))

# 这是有效的 Python 代码
normalize(torch.rand(3, 4))

traced = torch.fx.symbolic_trace(normalize)
"""
    <...>
    File "sqrt.py", line 9, in normalize
    return x / sqrt(len(x))
    File "pytorch/torch/fx/proxy.py", line 161, in __len__
    raise RuntimeError("'len' is not supported in symbolic tracing by default. If you want "
RuntimeError: 'len' is not supported in symbolic tracing by default. If you want this call to be recorded, please call torch.fx.wrap('len') at module scope
"""
```
错误信息告诉我们内置函数 `len` 不受支持。我们可以使用 `wrap` API 使此类函数在追踪中被记录为直接调用：

```python

torch.fx.wrap('len')
torch.fx.wrap('sqrt')

traced = torch.fx.symbolic_trace(normalize)

print(traced.code)
"""
import math
def forward(self, x):
    len_1 = len(x)
    sqrt_1 = math.sqrt(len_1);  len_1 = None
    truediv = x / sqrt_1;  x = sqrt_1 = None
    return truediv
"""
```

### 使用 `Tracer` 类自定义追踪

`Tracer` 类是 `symbolic_trace` 实现的基础。可以通过子类化 Tracer 来自定义追踪行为，如下所示：

```python

class MyCustomTracer(torch.fx.Tracer):
    # 在这里，你可以重写各种方法
    # 来自定义追踪。请参阅 `Tracer` API
    # 参考文档
    pass


# 让我们使用这个自定义追踪器来追踪这个模块
class MyModule(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + torch.ones(3, 4)

mod = MyModule()

traced_graph = MyCustomTracer().trace(mod)
# trace() 返回一个 Graph。让我们将其包装在
# GraphModule 中使其可运行
traced = torch.fx.GraphModule(mod, traced_graph)
```
## 叶模块

叶模块是在符号追踪中作为调用出现而不是被追踪通过的模块。默认的叶模块集合是标准 `torch.nn` 模块实例的集合。例如：

```python

class MySpecialSubmodule(torch.nn.Module):
    def forward(self, x):
        return torch.neg(x)

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.submod = MySpecialSubmodule()

    def forward(self, x):
        return self.submod(self.linear(x))

traced = torch.fx.symbolic_trace(MyModule())
print(traced.code)
# `linear` 被保留为一个调用，而 `submod` 被追踪通过。
# 这是因为默认的“叶模块”集合包含了所有
# 标准的 `torch.nn` 模块。
"""
import torch
def forward(self, x):
    linear_1 = self.linear(x);  x = None
    neg_1 = torch.neg(linear_1);  linear_1 = None
    return neg_1
"""
```
叶模块的集合可以通过重写 `Tracer.is_leaf_module` 来自定义。

### 杂项

-   张量构造函数（例如 `torch.zeros`、`torch.ones`、`torch.rand`、`torch.randn`、`torch.sparse_coo_tensor`）目前不可追踪。

    -   确定性构造函数（`zeros`、`ones`）可以使用，并且它们产生的值将作为常量嵌入到追踪中。仅当这些构造函数的参数引用动态输入大小时，这才会成为问题。在这种情况下，`ones_like` 或 `zeros_like` 可能是可行的替代方案。
    -   非确定性构造函数（`rand`、`randn`）将有一个单一的随机值嵌入到追踪中。这可能不是预期的行为。一种解决方法是把 `torch.randn` 包装在一个 `torch.fx.wrap` 函数中并调用该函数。

    ```python

    @torch.fx.wrap
    def torch_randn(x, shape):
        return torch.randn(shape)

    def f(x):
        return x + torch_randn(x, 5)
    fx.symbolic_trace(f)
    ```
    -   此行为可能在未来的版本中修复。

-   类型注解

-  Python 3 风格的类型注解（例如 `func(x : torch.Tensor, y : int) -> torch.Tensor`）受支持，并且会被符号追踪保留。
-  Python 2 风格的注释类型注解 `# type: (torch.Tensor, int) -> torch.Tensor` 目前不受支持。
-  函数内局部名称的注解目前不受支持。

-  关于 `training` 标志和子模块的注意事项

   -  当使用像 `torch.nn.functional.dropout` 这样的函数时，通常会将训练参数作为 `self.training` 传入。在 FX 追踪期间，这很可能会被烘焙为一个常量值。

    ```python

    import torch
    import torch.fx

    class DropoutRepro(torch.nn.Module):
        def forward(self, x):
        return torch.nn.functional.dropout(x, training=self.training)


    traced = torch.fx.symbolic_trace(DropoutRepro())
    print(traced.code)
    """
    def forward(self, x):
        dropout = torch.nn.functional.dropout(x, p = 0.5, training = True, inplace = False);  x = None
        return dropout
    """

    traced.eval()

    x = torch.randn(5, 3)
    torch.testing.assert_close(traced(x), x)
    """
    AssertionError: Tensor-likes are not close!

    Mismatched elements: 15 / 15 (100.0%)
    Greatest absolute difference: 1.6207983493804932 at index (0, 2) (up to 1e-05 allowed)
    Greatest relative difference: 1.0 at index (0, 0) (up to 0.0001 allowed)
    """
    ```
   - 然而，当使用标准的 `nn.Dropout()` 子模块时，训练标志被封装起来，并且由于保留了 `nn.Module` 对象模型，因此可以更改。

    ```python

    class DropoutRepro2(torch.nn.Module):
        def __init__(self):
        super().__init__()
        self.drop = torch.nn.Dropout()

        def forward(self, x):
        return self.drop(x)

    traced = torch.fx.symbolic_trace(DropoutRepro2())
    print(traced.code)
    """
    def forward(self, x):
        drop = self.drop(x);  x = None
        return drop
    """

    traced.eval()

    x = torch.randn(5, 3)
    torch.testing.assert_close(traced(x), x)
    ```
  - 由于这种差异，请考虑将与 `training` 标志动态交互的模块标记为叶子模块。

## API 参考


## torch.fx.annotate


## torch.fx.node


## torch.fx.operator_schemas


## torch.fx.traceback


<!-- experimental 和 passes 子模块缺少文档。 -->
<!-- 在此处添加以覆盖，但这不会为渲染的文档添加任何内容。 -->
