---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

(torch.export)=

# torch.export

## 概述

{func}`torch.export.export` 接收一个 {class}`torch.nn.Module`，并以提前编译（AOT）的方式生成一个仅表示函数张量计算过程的追踪图。该图随后可以用不同的输入执行或进行序列化。

```{code-cell}
import torch
from torch.export import export, ExportedProgram

class Mod(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b

example_args = (torch.randn(10, 10), torch.randn(10, 10))

exported_program: ExportedProgram = export(Mod(), args=example_args)
print(exported_program)
```

`torch.export` 生成一个简洁的中间表示（IR），并遵循以下不变性。关于 IR 的更多规范可以在 {ref}`此处 <export.ir_spec>` 找到。

- **正确性**：它保证是原始程序的一个正确表示，并保持原始程序相同的调用约定。
- **规范化**：图中不包含 Python 语义。原始程序中的子模块被内联，形成一个完全扁平化的计算图。
- **图属性**：默认情况下，图可能同时包含功能性和非功能性操作符（包括突变操作）。要获得一个纯粹的功能性图，可以使用 `run_decompositions()` 来移除突变和别名。
- **元数据**：图中包含追踪期间捕获的元数据，例如来自用户代码的堆栈跟踪。

在底层，`torch.export` 利用了以下最新技术：

- **TorchDynamo (torch._dynamo)** 是一个内部 API，它使用一个名为帧评估 API 的 CPython 功能来安全地追踪 PyTorch 图。这极大地改善了图捕获体验，为了完全追踪 PyTorch 代码所需的代码重写大大减少。
- **AOT Autograd** 确保图被分解/降级到 ATen 操作符集。当使用 `run_decompositions()` 时，它还可以提供功能化。
- **Torch FX (torch.fx)** 是图的底层表示，允许基于 Python 的灵活转换。

### 现有框架

{func}`torch.compile` 也使用与 `torch.export` 相同的 PT2 技术栈，但略有不同：

- **JIT 与 AOT**：{func}`torch.compile` 是一个 JIT 编译器，其目的不是用于在部署环境之外生成编译产物。
- **部分与完整图捕获**：当 {func}`torch.compile` 遇到模型中不可追踪的部分时，它会“图中断”并回退到在即时 Python 运行时中运行程序。相比之下，`torch.export` 旨在获取 PyTorch 模型的完整图表示，因此当遇到不可追踪的内容时会报错。由于 `torch.export` 生成一个与任何 Python 特性或运行时分离的完整图，因此该图可以保存、加载并在不同的环境和语言中运行。
- **可用性权衡**：由于 {func}`torch.compile` 在遇到不可追踪的内容时能够回退到 Python 运行时，因此它灵活得多。而 `torch.export` 则需要用户提供更多信息或重写代码以使其可追踪。

与 {func}`torch.fx.symbolic_trace` 相比，`torch.export` 使用 TorchDynamo 进行追踪，后者在 Python 字节码级别操作，使其能够追踪任意 Python 结构，不受 Python 操作符重载支持的限制。此外，`torch.export` 对张量元数据进行细粒度跟踪，因此对张量形状等条件的判断不会导致追踪失败。总的来说，`torch.export` 预期能在更多用户程序上工作，并生成更低级别的图（在 `torch.ops.aten` 操作符级别）。请注意，用户仍然可以在 `torch.export` 之前使用 {func}`torch.fx.symbolic_trace` 作为预处理步骤。

与 {func}`torch.jit.script` 相比，`torch.export` 不捕获 Python 控制流或数据结构，除非使用显式的 {ref}`控制流操作符 <higher_order_ops>`，但由于其对 Python 字节码的全面覆盖，它支持更多的 Python 语言特性。生成的图更简单，除了显式的控制流操作符外，只有直线控制流。

与 {func}`torch.jit.trace` 相比，`torch.export` 是正确的：它可以追踪对尺寸进行整数计算的代码，并记录所有必要的边界条件，以确保特定追踪对于其他输入是有效的。

## 导出 PyTorch 模型

主要入口点是通过 {func}`torch.export.export`，它接收一个 {class}`torch.nn.Module` 和示例输入，并将计算图捕获到一个 {class}`torch.export.ExportedProgram` 中。示例如下：

```{code-cell}
import torch
from torch.export import export, ExportedProgram

# 用于演示的简单模块
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

    def forward(self, x: torch.Tensor, *, constant=None) -> torch.Tensor:
        a = self.conv(x)
        a.add_(constant)
        return self.maxpool(self.relu(a))

example_args = (torch.randn(1, 3, 256, 256),)
example_kwargs = {"constant": torch.ones(1, 16, 256, 256)}

exported_program: ExportedProgram = export(
    M(), args=example_args, kwargs=example_kwargs
)
print(exported_program)

# 要运行导出的程序，我们可以使用 `module()` 方法
print(exported_program.module()(torch.randn(1, 3, 256, 256), constant=torch.ones(1, 16, 256, 256)))
```

检查 `ExportedProgram`，我们可以注意到以下几点：

- {class}`torch.fx.Graph` 包含原始程序的计算图，并附带原始代码记录以便于调试。
- 图中仅包含 [此处](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml) 找到的 `torch.ops.aten` 运算符和自定义运算符。
- 参数（卷积的权重和偏置）被提升为图的输入，因此图中不存在 `get_attr` 节点，而这类节点先前存在于 {func}`torch.fx.symbolic_trace` 的结果中。
- {class}`torch.export.ExportGraphSignature` 对输入和输出签名进行建模，并指定哪些输入是参数。
- 图中每个节点产生的张量的最终形状和数据类型会被记录。例如，`conv2d` 节点将产生一个数据类型为 `torch.float32`、形状为 (1, 16, 256, 256) 的张量。

## 表达动态性

默认情况下，`torch.export` 会假设所有输入形状都是**静态的**来追踪程序，并将导出的程序专门化到这些维度。这样做的一个后果是，在运行时，程序将无法处理具有不同形状的输入，即使这些输入在即时执行模式下是有效的。

示例：

```{code-cell}
import torch
import traceback as tb

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = torch.nn.Sequential(
            torch.nn.Linear(64, 32), torch.nn.ReLU()
        )
        self.branch2 = torch.nn.Sequential(
            torch.nn.Linear(128, 64), torch.nn.ReLU()
        )
        self.buffer = torch.ones(32)

    def forward(self, x1, x2):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        return (out1 + self.buffer, out2)

example_args = (torch.randn(32, 64), torch.randn(32, 128))

ep = torch.export.export(M(), example_args)
print(ep)

example_args2 = (torch.randn(64, 64), torch.randn(64, 128))
try:
    ep.module()(*example_args2)  # 失败
except Exception:
    tb.print_exc()
```

然而，某些维度（例如批次维度）可以是动态的，并且在每次运行时变化。必须使用 {func}`torch.export.Dim()` API 创建这些维度，并通过 `dynamic_shapes` 参数将它们传递给 {func}`torch.export.export()` 来指定这些维度。

```{code-cell}
import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = torch.nn.Sequential(
            torch.nn.Linear(64, 32), torch.nn.ReLU()
        )
        self.branch2 = torch.nn.Sequential(
            torch.nn.Linear(128, 64), torch.nn.ReLU()
        )
        self.buffer = torch.ones(32)

    def forward(self, x1, x2):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        return (out1 + self.buffer, out2)

example_args = (torch.randn(32, 64), torch.randn(32, 128))

# 创建一个动态批次大小
batch = torch.export.Dim("batch")
# 指定每个输入的第一个维度是该批次大小
dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

ep = torch.export.export(
    M(), args=example_args, dynamic_shapes=dynamic_shapes
)
print(ep)

example_args2 = (torch.randn(64, 64), torch.randn(64, 128))
ep.module()(*example_args2)  # 成功
```

需要注意的一些额外事项：

- 通过 {func}`torch.export.Dim` API 和 `dynamic_shapes` 参数，我们指定了每个输入的第一个维度是动态的。查看输入 `x1` 和 `x2`，它们具有符号形状 `(s0, 64)` 和 `(s0, 128)`，而不是我们作为示例输入传入的形状为 `(32, 64)` 和 `(32, 128)` 的张量。`s0` 是一个符号，表示该维度可以取一系列值。
- `exported_program.range_constraints` 描述了图中出现的每个符号的取值范围。在本例中，我们看到 `s0` 的范围是 [0, int_oo]。由于此处难以解释的技术原因，它们被假定不为 0 或 1。这不是一个错误，也不一定意味着导出的程序无法处理维度 0 或 1。有关此主题的深入讨论，请参阅 [0/1 专门化问题](https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ#heading=h.ez923tomjvyk)。

在示例中，我们使用 `Dim("batch")` 创建了一个动态维度。这是指定动态性最明确的方式。我们也可以使用 `Dim.DYNAMIC` 和 `Dim.AUTO` 来指定动态性。我们将在下一节中介绍这两种方法。

### 命名维度

对于每个使用 `Dim("name")` 指定的维度，我们将分配一个符号形状。使用相同名称指定 `Dim` 将生成相同的符号。这允许用户指定为每个输入维度分配哪些符号。

```python
batch = Dim("batch")
dynamic_shapes = {"x1": {0: dim}, "x2": {0: batch}}
```

对于每个 `Dim`，我们可以指定最小值和最大值。我们还允许在单变量线性表达式中指定 `Dim` 之间的关系：`A * dim + B`。这允许用户为动态维度指定更复杂的约束，例如整数可除性。这些功能允许用户对生成的 `ExportedProgram` 的动态行为施加明确的限制。

```python
dx = Dim("dx", min=4, max=256)
dh = Dim("dh", max=512)
dynamic_shapes = {
    "x": (dx, None),
    "y": (2 * dx, dh),
}
```

但是，如果在追踪过程中我们发出的守卫与给定的关系或静态/动态规范冲突，则会引发 `ConstraintViolationErrors`。例如，在上述规范中，断言了以下内容：

* `x.shape[0]` 的范围为 `[4, 256]`，并且与 `y.shape[0]` 的关系为 `y.shape[0] == 2 * x.shape[0]`。
* `x.shape[1]` 是静态的。
* `y.shape[1]` 的范围为 `[0, 512]`，并且与其他任何维度无关。

如果在追踪过程中发现这些断言不正确（例如 `x.shape[0]` 是静态的，或 `y.shape[1]` 的范围更小，或 `y.shape[0] != 2 * x.shape[0]`），则会引发 `ConstraintViolationError`，用户需要更改其 `dynamic_shapes` 规范。

### 维度提示

除了使用 `Dim("name")` 显式指定动态性外，我们还可以让 `torch.export` 使用 `Dim.DYNAMIC` 来推断动态值的范围和关系。当您不确定动态值具体*如何*动态变化时，这也是一种更便捷的指定动态性的方式。

```python
dynamic_shapes = {
    "x": (Dim.DYNAMIC, None),
    "y": (Dim.DYNAMIC, Dim.DYNAMIC),
}
```

我们还可以为 `Dim.DYNAMIC` 指定最小/最大值，这些值将作为导出时的提示。但如果追踪过程中发现范围不同，导出将自动更新范围而不会引发错误。我们也不能指定动态值之间的关系。相反，这将由导出推断，并通过检查图中的断言向用户公开。在这种指定动态性的方法中，**仅当**推断出指定值为**静态**时才会引发 `ConstraintViolationErrors`。

指定动态性的一种更便捷的方法是使用 `Dim.AUTO`，它的行为类似于 `Dim.DYNAMIC`，但如果推断出维度是静态的，则**不会**引发错误。当您完全不知道动态值是什么，并希望以“尽力而为”的动态方式导出程序时，这非常有用。

### ShapesCollection

通过 `dynamic_shapes` 指定哪些输入是动态时，我们必须指定每个输入的动态性。例如，给定以下输入：

```python
args = {"x": tensor_x, "others": [tensor_y, tensor_z]}
```

我们需要指定 `tensor_x`、`tensor_y` 和 `tensor_z` 的动态性以及动态形状：

```python
# 使用命名 Dim
dim = torch.export.Dim(...)
dynamic_shapes = {"x": {0: dim, 1: dim + 1}, "others": [{0: dim * 2}, None]}

torch.export(..., args, dynamic_shapes=dynamic_shapes)
```

然而，这特别复杂，因为我们需要以与输入参数相同的嵌套输入结构来指定 `dynamic_shapes` 规范。相反，指定动态形状的一种更简单的方法是使用辅助工具 {class}`torch.export.ShapesCollection`，在这里我们无需指定每个单独输入的动态性，而是可以直接分配哪些输入维度是动态的。

```{code-cell}
import torch

class M(torch.nn.Module):
    def forward(self, inp):
        x = inp["x"] * 1
        y = inp["others"][0] * 2
        z = inp["others"][1] * 3
        return x, y, z

tensor_x = torch.randn(3, 4, 8)
tensor_y = torch.randn(6)
tensor_z = torch.randn(6)
args = {"x": tensor_x, "others": [tensor_y, tensor_z]}

dim = torch.export.Dim("dim")
sc = torch.export.ShapesCollection()
sc[tensor_x] = (dim, dim + 1, 8)
sc[tensor_y] = {0: dim * 2}

print(sc.dynamic_shapes(M(), (args,)))
ep = torch.export.export(M(), (args,), dynamic_shapes=sc)
print(ep)
```

### AdditionalInputs

如果您不知道输入有多动态，但拥有充足的测试或性能分析数据集，可以为模型提供具有代表性的输入，那么可以使用 {class}`torch.export.AdditionalInputs` 代替 `dynamic_shapes`。您可以指定用于追踪程序的所有可能输入，`AdditionalInputs` 将根据哪些输入形状发生变化来推断哪些输入是动态的。

示例：

```{code-cell}
import dataclasses
import torch
import torch.utils._pytree as pytree

@dataclasses.dataclass
class D:
    b: bool
    i: int
    f: float
    t: torch.Tensor

pytree.register_dataclass(D)

class M(torch.nn.Module):
    def forward(self, d: D):
        return d.i + d.f + d.t

input1 = (D(True, 3, 3.0, torch.ones(3)),)
input2 = (D(True, 4, 3.0, torch.ones(4)),)
ai = torch.export.AdditionalInputs()
ai.add(input1)
ai.add(input2)

print(ai.dynamic_shapes(M(), input1))
ep = torch.export.export(M(), input1, dynamic_shapes=ai)
print(ep)
```

## 序列化

要保存 `ExportedProgram`，用户可以使用 {func}`torch.export.save` 和 {func}`torch.export.load` API。生成的文件是一个具有特定结构的 zip 文件。该结构的详细信息在 {ref}`PT2 归档规范 <export.pt2_archive>` 中定义。

示例：

```python
import torch

class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 10

exported_program = torch.export.export(MyModule(), (torch.randn(5),))

torch.export.save(exported_program, 'exported_program.pt2')
saved_exported_program = torch.export.load('exported_program.pt2')
```

(training-export)=

## 导出 IR：训练与推理

`torch.export` 生成的图返回一个仅包含 [ATen 运算符](https://pytorch.org/cppdocs/#aten) 的图，这些运算符是 PyTorch 中的基本计算单元。根据您的使用场景，导出提供了不同的 IR 级别：

| IR 类型 | 如何获取 | 特性 | 运算符数量 | 使用场景 |
|---------|---------------|------------|----------------|----------|
| 训练 IR | `torch.export.export()`（默认） | 可能包含突变 | ~3000 | 使用自动微分的训练 |
| 推理 IR | `ep.run_decompositions(decomp_table={})` | 纯函数式 | ~2000 | 推理部署 |
| 核心 ATen IR | `ep.run_decompositions(decomp_table=None)` | 纯函数式，高度分解 | ~180 | 最小后端支持 |

### 训练 IR（默认）

默认情况下，导出生成一个**训练 IR**，其中包含所有 ATen 运算符，包括函数式和非函数式（突变）运算符。函数式运算符是指不包含任何输入突变或别名的运算符，而非函数式运算符可能会就地修改其输入。您可以在[此处](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)找到所有 ATen 运算符的列表，并通过检查 `op._schema.is_mutable` 来检查运算符是否为函数式。

此训练 IR（可能包含突变操作）专为训练场景设计，可与即时执行模式的 PyTorch Autograd 配合使用。

```{code-cell}
import torch

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = torch.export.export(M(), (torch.randn(1, 1, 3, 3),))
print(ep_for_training.graph_module.print_readable(print_output=False))
```

### 推理 IR（通过 run_decompositions）

要获得适用于部署的**推理 IR**，请使用 {func}`ExportedProgram.run_decompositions` API。该方法会自动：
1. 功能化图（移除所有突变操作并转换为等效功能化版本）
2. 根据提供的分解表选择性地分解 ATen 算子

这将生成一个纯粹功能化的图，非常适合推理场景。

通过指定空分解表（`decomp_table={}`），您将仅获得功能化转换而不进行额外分解。这会生成包含约 2000 个功能化算子的推理 IR（相比之下训练 IR 包含 3000+ 个算子）。

```{code-cell}
import torch

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = torch.export.export(M(), (torch.randn(1, 1, 3, 3),))
with torch.no_grad():
    ep_for_inference = ep_for_training.run_decompositions(decomp_table={})
print(ep_for_inference.graph_module.print_readable(print_output=False))
```

如我们所见，原先的就地操作符 `torch.ops.aten.add_.default` 现在已被替换为功能化操作符 `torch.ops.aten.add.default`。

### 核心 ATen IR

我们可以进一步将推理 IR 降级到
`Core ATen Operator Set <https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_ir.html#core-aten-ir>`__，
该集合仅包含约 180 个算子。这是通过向 `run_decompositions()` 传递 `decomp_table=None`（使用默认分解表）实现的。这种 IR 对于希望最小化需要实现的操作符数量的后端来说是最优选择。

```{code-cell}
import torch

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = torch.export.export(M(), (torch.randn(1, 1, 3, 3),))
with torch.no_grad():
    core_aten_ir = ep_for_training.run_decompositions(decomp_table=None)
print(core_aten_ir.graph_module.print_readable(print_output=False))
```

现在我们可以看到 `torch.ops.aten.conv2d.default` 已被分解为 `torch.ops.aten.convolution.default`。这是因为 `convolution` 是一个更"核心"的算子，因为像 `conv1d` 和 `conv2d` 这样的操作可以使用同一个算子实现。

我们也可以指定自定义的分解行为：

```{code-cell}
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = torch.export.export(M(), (torch.randn(1, 1, 3, 3),))

my_decomp_table = torch.export.default_decompositions()

def my_awesome_custom_conv2d_function(x, weight, bias, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1):
    return 2 * torch.ops.aten.convolution(x, weight, bias, stride, padding, dilation, False, [0, 0], groups)

my_decomp_table[torch.ops.aten.conv2d.default] = my_awesome_custom_conv2d_function
my_ep = ep_for_training.run_decompositions(my_decomp_table)
print(my_ep.graph_module.print_readable(print_output=False))
```

请注意，`torch.ops.aten.conv2d.default` 现在不是被分解为 `torch.ops.aten.convolution.default`，而是被分解为 `torch.ops.aten.convolution.default` 和 `torch.ops.aten.mul.Tensor`，这与我们的自定义分解规则相匹配。

(limitations-of-torch-export)=

## torch.export 的限制

由于 `torch.export` 是从 PyTorch 程序捕获计算图的一次性过程，它最终可能会遇到程序中无法追踪的部分，因为几乎不可能支持追踪所有 PyTorch 和 Python 特性。在 `torch.compile` 的情况下，不支持的操作会导致"图中断"，不支持的操作将通过默认的 Python 求值运行。相比之下，`torch.export` 将要求用户提供额外信息或重写部分代码以使其可追踪。

{ref}`Draft-export <export.draft_export>` 是一个很好的资源，列出了追踪程序时会遇到的图中断，以及解决这些错误的额外调试信息。

{ref}`ExportDB <torch.export_db>` 也是一个很好的资源，用于了解支持和不受支持的程序类型，以及重写程序使其可追踪的方法。

### TorchDynamo 不支持

当使用 `torch.export` 并设置 `strict=True` 时，这将使用 TorchDynamo 在 Python 字节码级别评估程序以追踪成图。与之前的追踪框架相比，使程序可追踪所需的改写会显著减少，但仍会有一些 Python 特性不受支持。解决这些图中断的一个选项是通过将 `strict` 标志更改为 `strict=False` 来使用 {ref}`非严格导出 <non-strict-export>`。

(data-shape-dependent-control-flow)=

### 数据/形状依赖的控制流

当未对形状进行特化时，数据依赖的控制流（例如 `if x.shape[0] > 2`）也可能遇到图中断，因为追踪编译器无法在不生成组合爆炸路径数量的代码的情况下处理此类情况。在这种情况下，用户需要使用特殊的控制流运算符重写代码。目前，我们支持 {ref}`高阶运算符 <higher_order_ops>` 来表达条件、映射、扫描和循环等控制流模式。

你也可以参考这个
[教程](https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html#data-dependent-errors)
了解处理数据依赖错误的更多方法。

### 运算符缺少 Fake/Meta 内核

在追踪过程中，所有运算符都需要一个 FakeTensor 内核（也称为 meta 内核）。这用于推断该运算符的输入/输出形状。

更多详情请参阅此
[教程](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)。

如果不幸你的模型使用了尚未实现 FakeTensor 内核的 ATen 运算符，请提交一个问题。

## 延伸阅读

```{toctree}
:caption: 面向导出用户的附加链接
:maxdepth: 1

export/api_reference
export/programming_model
export/ir_spec
export/pt2_archive
export/draft_export
export/joint_with_descriptors
../../higher_order_ops/index
../../generated/exportdb/index
torch.compiler_aot_inductor
torch.compiler_ir
```

```{toctree}
:caption: 面向 PyTorch 开发者的深入探讨
:maxdepth: 1

torch.compiler_dynamic_shapes
torch.compiler_fake_tensor
torch.compiler_transformations
```