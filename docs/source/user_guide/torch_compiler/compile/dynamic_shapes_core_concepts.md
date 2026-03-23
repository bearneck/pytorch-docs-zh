
# 动态形状核心概念

本节描述了 PyTorch 中动态形状的核心概念。旨在为从事 PyTorch 编译器栈工作的工程师以及任何希望理解动态形状内部工作原理的人员提供参考。

## 符号整数
符号整数（Symints）用于表示可以跨越一定范围的变量。例如：
```python
x = torch.randn(5, 5) # 此张量的形状为 [5, 5]
torch._dynamo.decorators.mark_dynamic(x, 0)
x = torch.randn(5, 5) # 此张量的形状为 [s0, 5]
y = torch.cat([x, x], dim=0) # 此张量的形状为 [2*s0, 5]
```

然而，`z = x * y` 会抛出错误，因为我们知道像乘法这样的逐点操作必须在相同大小的张量上进行，但我们静态地知道 `s0 != 2 * s0`。敏锐的读者可能会指出，当 `s0 == 0` 时这不成立，而此处这无关紧要的原因在 `zero-one-specialization` 中有所描述。

## 守卫

在 `torch.compile` 中，守卫是一种用于确保已编译代码图有效性的机制。默认情况下，当你使一个变量变为动态时，其范围可以是 `[-inf, inf]`。例如：

```python
def foo(x): return x / 2

这对于任何动态的 x 都有效。但如果你的代码是：

def foo(x)
    if x > 5:
        return x / 2
    return x / 3
```
如果你调用 `foo(6)`，它会返回 `x / 2` 并添加一个守卫 `x > 5`。随后调用 `foo(4)` 将需要重新编译，因为守卫被打破了。

## 运行时断言
当你已知某些事实时，例如批次大小小于 100，可以使用运行时断言来提供提示：

```python
def foo(batch_size):
    torch._check(batch_size < 100)
    if batch_size < 100:
        return do_something
    return do_something_else()
```

## "提示"值

在 `torch.compile` 的上下文中，"提示值"指的是编译过程中已知的实际值，这些值帮助 JIT 编译器对表达式做出决策。提示值对于处理动态形状特别有用，因为它们提供了具体信息来指导编译，而无需为变化的维度重新编译。

## 动态行为概述

PyTorch 默认假设静态形状。当检测到大小变化时，它会尝试使用动态输入重新编译，但如果存在条件分支或缺少对动态形状的支持，则可能会失败。要诊断过度特化，可以设置 `TORCH_LOGS=dynamic` 来查看指示何时以及为何添加守卫的 "eval" 条目。

如果你预期某个维度将是动态的，可以提前使用 `torch._dynamo.mark_dynamic(tensor, dim)` 来标记它，并在已知的情况下指定 `min` 和 `max` 值。使用 `torch.compile(dynamic=False)` 会禁用自动动态形状，导致为每个唯一大小重新编译。相反，`torch.compile(dynamic=True)` 旨在尽可能使用动态形状，这对于小型模型最有用，但由于潜在的崩溃或性能问题，可能不适合大型模型。

你可以使用 `TORCH_COMPILE_DYNAMIC_SOURCES` 环境变量或 `torch.compiler.config.dynamic_sources` 将特定源列入白名单以标记为动态。这对于具有图中断的大型模型特别有用，因为源名称保持一致，你可以在图中断之间保持动态性。你也可以使用此功能将整数标记为动态。格式是以逗号分隔的源名称列表，例如 `"L['x'], L['y']"`。
你也可以使用正则表达式，例如 `"L\['x.*'\], L\['y.*'\]")`。
此白名单优先于其他标志，如 `dynamic=False`、`force_nn_module_property_static_shapes` 和 `force_parameter_static_shapes`。

有时，找到正确的输入标记为动态可能很繁琐。如果你愿意为第一批次承担性能损失，我们提供的另一个可行选项是 `eager_then_compile` 姿态，它会为你推导动态性。
有关更多详细信息，请参阅 `torch.compiler.set_stance`。

## 整体架构

符号形状工作流程：

1. 在 Dynamo 中编译帧时，我们分配一个 `ShapeEnv`（附加到 `FakeTensorMode`）来跟踪符号形状。
2. 我们根据策略决策，在入口处为张量分配符号大小。
3. 我们通过运算符传播符号大小，同时维护用于符号计算导出的 FX IR 和用于推理的 Sympy 表达式。
4. 我们在 Dynamo 追踪或 Inductor 优化期间，基于条件语句添加守卫，这些守卫由 Python 和 C++ 共同引发。
5. 守卫可以简化符号变量。例如，断言 `s0 == 4` 允许将所有出现的 `s0` 替换为 `4`。
6. 在追踪和优化之后，我们将所有守卫与编译后的代码一起安装，确保仅当所有守卫评估为真时才可重用。

## 内部 API 类层次结构

### Python 类

- **`SymInt`/`SymFloat`/`SymBool`**：用户可见的类，模拟其对应的 `int`/`float`/`bool`。将两个 `SymInts` 相加会产生一个新的 `SymInt`，该 `SymInt` 以符号方式跟踪整数加法。

- **`SymNode`**：内部结构（可通过 `symint.node` 访问），保存实际的符号跟踪信息。`SymNode` 是类型擦除的，便于表示混合类型的操作。

- **`ShapeEnv`**：每次编译的上下文状态，跟踪迄今为止累积的所有自由符号和守卫。每个 `SymNode` 记录其 `ShapeEnv`（但反之则不然；`SymNodes` 仅在参与守卫时使用）。

### C++ 对应类

- **`c10::SymInt`/`SymFloat`/`SymBool`**：用户可见的类，模拟 `int`/`float`/`bool`
- **`c10::SymNode`/`SymNodeImpl`**：类似于 Python 的 `SymNode`
- **无 C++ `ShapeEnv`**：为了便于调试，整个符号推理装置保留在 Python 中

在编写可通过 `make_fx` 追踪的代码时，它必须处理流经其中的 `SymInt`/`SymFloat`/`SymBool`。

## 值范围和约束

符号变量维护着**值范围**，用于指定可能的取值集合。默认情况下：
- 尺寸类无后端 `SymInts` 的值范围为 `[0, Inf]`
- 常规无后端 `SymInts` 的值范围为 `[-Inf, Inf]`

当进行断言时（例如 `torch._check(x == y)`），系统会：
1. 尝试用等效表达式替换无后端符号
2. 基于断言细化值范围
3. 记录始终为真的布尔表达式

重要文件：

- C++ SymInt API: `c10/core/SymInt.h`, `SymFloat.h`, `SymBool.h`
- Python SymInt API: `torch/__init__.py` (查找 `SymInt/SymFloat/SymBool`)
- C++ 底层实现: `c10/core/SymNodeImpl.h`, `torch/csrc/utils/python_symnode.h`, `torch/csrc/jit/python/init.cpp`
- Python 基础设施: `torch/fx/experimental/symbolic_shapes.py`
- 其他重要文件: `torch/_subclasses/fake_tensor.py`, `torch/_meta_registrations.py`, 分解函数, PrimTorch 参考

```{seealso}
* `dynamic_shapes`
* `dynamic_shapes_troubleshooting`
```