(export.programming_model)=

# torch.export 编程模型

本文档旨在解释 {func}`torch.export.export` 的行为和功能。它旨在帮助您建立对 {func}`torch.export.export` 如何处理代码的直观理解。

## 追踪基础

{func}`torch.export.export` 通过在"示例"输入上追踪模型的执行，并记录沿追踪路径观察到的 PyTorch 操作和条件，来捕获表示模型的图。只要满足相同的条件，该图就可以在不同的输入上运行。

{func}`torch.export.export` 的基本输出是一个包含关联元数据的 PyTorch 操作单图。此输出的确切格式在 {ref}`导出 IR 规范 <export.ir_spec>` 中介绍。

(non-strict-export)=

### 严格追踪与非严格追踪

{func}`torch.export.export` 提供两种追踪模式。

在*非严格模式*下，我们使用普通的 Python 解释器追踪程序。您的代码完全按照在即时执行模式下的方式执行；唯一的区别是所有张量都被替换为[伪张量](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_fake_tensor.html)（**这些张量具有形状和其他形式的元数据，但没有实际数据**），并包装在记录所有操作的[代理对象](https://pytorch.org/docs/main/fx.html)中，这些操作被记录到图中。我们还会捕获[张量形状上的条件](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#the-guard-model)，**这些条件保护生成代码的正确性**。

在*严格模式*下，我们首先使用 {ref}`TorchDynamo <torch.compiler_dynamo_deepdive>`（一个 Python 字节码分析引擎）追踪程序。TorchDynamo 实际上并不执行您的 Python 代码。相反，它进行符号分析并根据结果构建图。一方面，这种分析允许 {func}`torch.export.export` 在 Python 级别安全性上提供额外的保证（除了捕获张量形状上的条件，如非严格模式）。另一方面，并非所有 Python 功能都受此分析支持。

尽管目前默认的追踪模式是严格模式，但**我们强烈建议使用非严格模式**，该模式很快将成为默认模式。对于大多数模型，张量形状上的条件足以保证正确性，而 Python 级别安全性的额外保证没有影响；同时，在 TorchDynamo 中遇到不受支持的 Python 功能会带来不必要的风险。

在本文档的其余部分，我们假设在[非严格模式](https://pytorch.org/docs/main/export.html#non-strict-export)下进行追踪；特别是，我们假设**所有 Python 功能都受支持**。

## 值：静态与动态

理解 {func}`torch.export.export` 行为的一个关键概念是*静态*值和*动态*值之间的区别。

### 静态值

*静态*值是一个**在导出时固定且在导出程序的不同执行之间不能改变**的值。在追踪过程中遇到该值时，我们将其视为常量并将其硬编码到图中。

当执行一个操作（例如 `x + y`）且所有输入都是静态时，该操作的输出会直接硬编码到图中，并且该操作不会显示出来（即它被"常量折叠"了）。

当一个值被硬编码到图中时，我们说该图已*特化*到该值。例如：

```python
import torch

class MyMod(torch.nn.Module):
    def forward(self, x, y):
        z = y + 7
        return x + z

m = torch.export.export(MyMod(), (torch.randn(1), 3))
print(m.graph_module.code)

"""
def forward(self, arg0_1, arg1_1):
    add = torch.ops.aten.add.Tensor(arg0_1, 10);  arg0_1 = None
    return (add,)

"""
```

这里，我们提供 `3` 作为 `y` 的追踪值；它被视为静态值并与 `7` 相加，将静态值 `10` 固化到图中。

### 动态值

*动态*值是一个**可以在不同运行之间改变**的值。它的行为就像"普通"的函数参数：您可以传递不同的输入，并期望您的函数执行正确的操作。

### 哪些值是静态的，哪些是动态的？

一个值是静态还是动态取决于其类型：

- 对于张量：

  - 张量*数据*被视为动态。

  - 张量*形状*可以被系统视为静态或动态。

    - 默认情况下，所有输入张量的形状被视为静态。用户可以通过为任何输入张量指定[动态形状](https://pytorch.org/docs/main/export.html#expressing-dynamism)来覆盖此行为。
    - 作为模块状态一部分的张量（即参数和缓冲区）始终具有静态形状。

  - 张量的其他形式*元数据*（例如 `device`、`dtype`）是静态的。

- Python*基本类型*（`int`、`float`、`bool`、`str`、`None`）是静态的。

  - 某些基本类型有动态变体（`SymInt`、`SymFloat`、`SymBool`）。通常用户无需处理它们。
  - 用户可以通过为整数输入指定[动态形状](https://pytorch.org/docs/main/export.html#expressing-dynamism)来将其指定为动态。

- 对于 Python*标准容器*（`list`、`tuple`、`dict`、`namedtuple`）：

  - 结构（即 `list` 和 `tuple` 值的长度，以及 `dict` 和 `namedtuple` 值的键序列）是静态的。
  - 包含的元素递归应用这些规则（基本上是[PyTree](https://jax.readthedocs.io/en/latest/pytrees.html)方案），其中叶子节点要么是张量，要么是基本类型。

- 其他*类*（包括数据类）可以向 PyTree 注册（见下文），并遵循与标准容器相同的规则。

## 输入类型

根据其类型（如上所述），输入将被视为静态或动态。

- 静态输入将被硬编码到图中，在运行时传递不同的值会导致错误。请注意，这些主要是基本类型的值。
- 动态输入的行为类似于“普通”函数输入。请注意，这些主要是张量类型的值。

默认情况下，程序中可使用的输入类型包括：

- Tensor
- Python 基本类型（`int`、`float`、`bool`、`str`、`None`）
- Python 标准容器（`list`、`tuple`、`dict`、`namedtuple`）

### 自定义输入类型（PyTree）

此外，您也可以定义自己的（自定义）类并将其用作输入类型，但需要将此类注册为 PyTree。

以下是一个使用工具注册数据类作为输入类型的示例。

```python
@dataclass
class Input:
    f: torch.Tensor
    p: torch.Tensor

import torch.utils._pytree as pytree
pytree.register_dataclass(Input)

class M(torch.nn.Module):
    def forward(self, x: Input):
        return x.f + 1

torch.export.export(M(), (Input(f=torch.ones(10, 4), p=torch.zeros(10, 4)),))
```

### 可选输入类型

对于程序中未传递的可选输入，{func}`torch.export.export` 将特化为其默认值。因此，导出的程序将要求用户显式传递所有参数，并失去默认行为。例如：

```python
class M(torch.nn.Module):
    def forward(self, x, y=None):
        if y is not None:
            return y * x
        return x + x

# 传递了可选输入
ep = torch.export.export(M(), (torch.randn(3, 3), torch.randn(3, 3)))
print(ep)
"""
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[3, 3]", y: "f32[3, 3]"):
            # File: /data/users/angelayi/pytorch/moo.py:15 in forward, code: return y * x
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(y, x);  y = x = None
            return (mul,)
"""

# 未传递可选输入
ep = torch.export.export(M(), (torch.randn(3, 3),))
print(ep)
"""
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[3, 3]", y):
            # File: /data/users/angelayi/pytorch/moo.py:16 in forward, code: return x + x
            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(x, x);  x = None
            return (add,)
"""
```

## 控制流：静态与动态

{func}`torch.export.export` 支持控制流。控制流的行为取决于您分支所依据的值是静态还是动态的。

### 静态控制流

**支持透明地处理基于静态值的 Python 控制流**。（请注意，静态值包括静态形状，因此基于静态形状的控制流也属于此情况。）

如上所述，我们“固化”静态值，因此导出的图永远不会看到任何基于静态值的控制流。

对于 `if` 语句，我们将继续跟踪导出时执行的分支。对于 `for` 或 `while` 语句，我们将通过展开循环来继续跟踪。

### 动态控制流：形状依赖与数据依赖

当控制流中涉及的值是动态时，它可能依赖于动态形状或动态数据。鉴于编译器在跟踪时使用的是形状信息而非数据信息，这两种情况对编程模型的影响是不同的。

#### 动态形状依赖控制流

当控制流中涉及的值是[动态形状](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html)时，在大多数情况下**我们在跟踪期间也会知道动态形状的具体值**：有关编译器如何跟踪此信息的更多细节，请参阅下一节。

在这些情况下，我们说控制流是形状依赖的。**我们使用动态形状的具体值来评估条件**为 `True` 或 `False` 并继续跟踪（如上所述），同时发出与刚刚评估的条件相对应的保护。

否则，控制流被视为数据依赖。我们无法将条件评估为 `True` 或 `False`，因此无法继续跟踪，必须在导出时引发错误。请参阅下一节。

#### 动态数据依赖控制流

**支持基于动态值的数据依赖控制流，但您必须使用 PyTorch 的显式运算符之一**来继续跟踪。不允许在动态值上使用 Python 控制流语句，因为编译器无法评估继续跟踪所需的条件，因此必须在导出时引发错误。

我们提供了**用于表达基于动态值的一般条件语句和循环的运算符**，例如 `torch.cond`、`torch.map`。请注意，只有在确实需要*数据依赖控制流*时才需要使用这些运算符。

以下是一个基于数据依赖条件 `x.sum() > 0`（其中 `x` 是输入张量）的 `if` 语句示例，使用 `torch.cond` 重写。现在无需决定跟踪哪个分支，而是同时跟踪两个分支。

```python
class M_old(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x.sin()
        else:
            return x.cos()

class M_new(torch.nn.Module):
    def forward(self, x):
        return torch.cond(
            pred=x.sum() > 0,
            true_fn=lambda x: x.sin(),
            false_fn=lambda x: x.cos(),
            operands=(x,),
        )
```

数据依赖控制流的一个特殊情况是涉及[数据依赖动态形状](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#unbacked-symints)：通常，某些中间张量的形状依赖于输入数据而非输入形状（因此不是形状依赖）。在这种情况下，您无需使用控制流运算符，而是可以提供一个断言来决定条件是 `True` 还是 `False`。给定这样的断言，我们可以继续跟踪，并如上所述发出保护。

我们提供了**用于表达动态形状断言的运算符**，例如 `torch._check`。请注意，仅当存在依赖于数据的动态形状上的控制流时才需要使用此功能。

以下是一个在涉及数据依赖动态形状的条件 `nz.shape[0] > 0` 上使用 `if` 语句的示例，其中 `nz` 是调用 {func}`torch.nonzero` 的结果，该运算符的输出形状依赖于输入数据。您无需重写代码，只需使用 `torch._check` 添加断言即可有效决定追踪哪个分支。

```python
class M_old(torch.nn.Module):
    def forward(self, x):
        nz = x.nonzero()
        if nz.shape[0] > 0:
            return x.sin()
        else:
            return x.cos()

class M_new(torch.nn.Module):
    def forward(self, x):
        nz = x.nonzero()
        torch._check(nz.shape[0] > 0)
        if nz.shape[0] > 0:
            return x.sin()
        else:
            return x.cos()
```

## 符号形状基础

在追踪过程中，动态张量形状及其相关条件被编码为“符号表达式”。（相比之下，静态张量形状及其相关条件则只是简单的 `int` 和 `bool` 值。）

*符号*类似于变量；它描述动态张量形状。

随着追踪的进行，中间张量的形状可能由更通用的表达式描述，通常涉及整数算术运算符。这是因为**对于大多数 PyTorch 运算符，输出张量的形状可以描述为输入张量形状的函数**。例如，{func}`torch.cat` 的输出形状是其输入形状的总和。

此外，当我们在程序中遇到控制流时，会创建布尔表达式（通常涉及关系运算符）来描述追踪路径上的条件。这些**表达式会被求值以决定追踪程序的哪条路径**，并记录在[形状环境](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#overall-architecture)中，以保护追踪路径的正确性并评估后续创建的表达式。

接下来我们简要介绍这些子系统。

### PyTorch 运算符的伪实现

回顾在追踪过程中，我们使用[伪张量](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_fake_tensor.html)执行程序，这些张量没有实际数据。通常我们无法使用伪张量调用 PyTorch 运算符的实际实现。因此，每个运算符都需要一个额外的伪（或称“元”）实现，该实现输入和输出伪张量，并在形状和伪张量携带的其他元数据方面与实际实现的行为匹配。

例如，请注意 {func}`torch.index_select` 的伪实现如何利用输入形状计算输出形状（同时忽略输入数据并返回空的输出数据）。

```python
def meta_index_select(self, dim, index):
    result_size = list(self.size())
    if self.dim() > 0:
        result_size[dim] = index.numel()
    return self.new_empty(result_size)
```

#### 形状传播：有支撑与无支撑动态形状

形状传播通过 PyTorch 运算符的伪实现进行。

理解动态形状传播的一个关键概念是*有支撑*和*无支撑*动态形状之间的区别：我们知道前者的具体值，但不知道后者的具体值。

形状传播（包括追踪有支撑和无支撑动态形状）的过程如下：

- 表示输入的张量形状可以是静态或动态的。当为动态时，它们由符号描述；此外，**由于我们在导出时通过用户提供的“真实”示例输入也知晓其具体值，因此这类符号是有支撑的**。

- 运算符的输出形状由其伪实现计算，可以是静态或动态的。当为动态时，通常由符号表达式描述。此外：
  - 如果输出形状仅依赖于输入形状，则当所有输入形状均为静态或有支撑动态时，输出形状要么是静态的，要么是有支撑动态的。
  - 另一方面，**如果输出形状依赖于输入数据**，则它必然是动态的，并且**由于我们无法知道其具体值，因此它是无支撑的**。

### 控制流：守卫与断言

当遇到形状条件时，它要么仅涉及静态形状（此时为 `bool` 值），要么涉及动态形状（此时为符号布尔表达式）。对于后者：

- 当条件仅涉及有支撑动态形状时，我们可以使用这些动态形状的具体值将条件求值为 `True` 或 `False`。然后，我们可以向形状环境添加一个守卫，声明相应的符号布尔表达式为 `True` 或 `False`，并继续追踪。
- 否则，条件涉及无支撑动态形状。通常，在没有额外信息的情况下我们无法评估此类条件；因此我们无法继续追踪，必须在导出时引发错误。用户应使用显式的 PyTorch 运算符来继续追踪。此信息将作为守卫添加到形状环境中，并可能有助于将后续遇到的其他条件求值为 `True` 或 `False`。

模型导出后，**所有基于已绑定动态形状的约束条件均可理解为对输入动态形状的限制**。这些约束会与动态形状规范进行验证，该规范必须在导出时提供，用于描述动态形状需满足的条件——不仅示例输入需满足，所有未来输入也必须满足，以确保生成的代码正确运行。更准确地说，动态形状规范必须在逻辑上蕴含生成的约束条件，否则导出时将引发错误（同时会提供修改动态形状规范的建议）。另一方面，当不存在基于已绑定动态形状的约束时（特别是所有形状均为静态时），导出时无需提供动态形状规范。通常，动态形状规范会被转换为生成代码输入端的运行时断言。

最后，**所有基于未绑定动态形状的约束会被转换为“内联”运行时断言**。这些断言会被插入到生成代码中创建未绑定动态形状的位置：通常是在数据依赖的算子调用之后。

## 允许的 PyTorch 算子

所有 PyTorch 算子均被允许。

### 自定义算子

此外，您可以定义并使用[自定义算子](https://pytorch.org/tutorials/advanced/python_custom_ops#python-custom-ops-tutorial)。定义自定义算子需要为其定义伪实现，就像其他 PyTorch 算子一样（参见前一节）。

以下是一个包装 NumPy 的自定义 `sin` 算子示例及其注册的（简单）伪实现：

```python
@torch.library.custom_op("mylib::sin", mutates_args=())
def sin(x: Tensor) -> Tensor:
    x_np = x.numpy()
    y_np = np.sin(x_np)
    return torch.from_numpy(y_np)

@torch.library.register_fake("mylib::sin")
def _(x: Tensor) -> Tensor:
    return torch.empty_like(x)
```

**有时自定义算子的伪实现会涉及数据依赖的形状**。以下是一个自定义 `nonzero` 算子的伪实现示例：

```python
...

@torch.library.register_fake("mylib::custom_nonzero")
def _(x):
    nnz = torch.library.get_ctx().new_dynamic_size()
    shape = [nnz, x.dim()]
    return x.new_empty(shape, dtype=torch.int64)
```

## 模块状态：读取与更新

模块状态包括参数、缓冲区和常规属性。

- 常规属性可以是任意类型。
- 而参数和缓冲区始终是张量。

根据上述类型划分，模块状态可以是动态或静态的。例如，`self.training` 是 `bool` 类型，这意味着它是静态的；而任何参数或缓冲区都是动态的。

模块状态中包含的任何张量的**形状**不能是动态的，即这些形状在导出时是固定的，在导出程序的多次执行之间不能改变。

### 访问规则

**所有模块状态必须被初始化**。访问未初始化的模块状态会在导出时引发错误。

**读取模块状态始终被允许**。

更新模块状态是可能的，但必须遵循以下规则：

- **静态常规属性**（例如基本类型）**可以被更新**。读取和更新可以自由交错进行，且如预期那样，任何读取操作都将始终看到最新更新的值。由于这些属性是静态的，我们也会将其值固化，因此生成的代码不会包含实际“获取”或“设置”此类属性的指令。
- **动态常规属性**（例如张量类型）**不能被更新**。若要更新，必须在模块初始化时将其注册为缓冲区。
- **缓冲区可以被更新**，更新可以是原地操作（例如 `self.buffer[:] = ...`）或非原地操作（例如 `self.buffer = ...`）。
- **参数不能被更新**。通常参数仅在训练期间更新，推理期间不更新。建议使用 {func}`torch.no_grad` 进行导出，以避免导出时的参数更新。

### 函数化处理的影响

任何被读取和/或更新的动态模块状态会被“提升”为生成代码的输入和/或输出。

导出程序会与生成的代码一起存储参数和缓冲区的初始值以及其他张量属性的常量值。