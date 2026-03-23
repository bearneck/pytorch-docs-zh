(torch.compiler_troubleshooting)=

# torch.compile 故障排除

您正尝试在 PyTorch 模型上使用 `torch.compile` 来提升性能，但效果不如预期。可能是性能没有改善、出现崩溃，或者编译时间过长。本文提供了一些技巧、变通方法和调试工具，帮助您应对这些挑战。

**目录**

```{contents}
:local: true
```

## 设定预期

`torch.compile` 被设计为一个通用的 PyTorch 编译器。与之前的编译器解决方案 TorchScript 不同，`torch.compile` 需要更少的代码改动，这意味着通常不需要从头重写模型。它也能更优雅地处理不支持的代码——不支持的代码只会导致优化机会的丢失，而不会引发崩溃。

在理想情况下，我们可以简单地将 `torch.compile` 应用于任何 PyTorch 模型，并享受自动加速带来的好处。然而，现实中代码的复杂性可能导致以下三种情况之一：

1.  `torch.compile` 无缝工作，提供加速效果。
2.  需要进行一些代码修改。`torch.compile` 不会崩溃或耗时过长，但您可能看不到显著的性能提升。
3.  需要对代码进行大量修改。

我们预计大多数代码会属于情况 (1) 和 (2)。本文档提供了按参与程度排列的技巧，以帮助解决情况 (2) 中的代码问题。

### 编译时间

`torch.compile` 作为一个即时编译器运行，因此编译函数的初始一两次运行预计会明显变慢。在某些条件下（详见下文）发生的重新编译也会使运行变慢。`torch.compile` 的各个组件会缓存结果，以减少未来调用（甚至在不同进程中）的编译时间。对于常见或基准测试模型，冷启动（未缓存）编译时间通常在几秒到几分钟之间。更大的模型可能需要 30 分钟到几个小时。

## 术语

以下术语与 `torch.compile` 问题排查相关。

### 图中断

`torch.compile` 会追踪您的代码，并尝试将 PyTorch 代码捕获到一个单一的 PyTorch 操作符计算图（FX 图）中。然而，这并非总是可行。当遇到无法追踪的代码时，就会发生“图中断”。图中断涉及编译到目前为止已确定的 FX 图，运行不支持的代码，然后在不受支持的代码之后使用新的 FX 图恢复追踪。由于计算图被分割，我们失去了优化机会，因此模型代码应尽可能避免图中断。图中断发生在以下情况：

-   依赖于数据的 if 语句
-   许多 Python 内置函数
-   C 函数

下面是一个由于 Python 内置库中的 `copy.deepcopy` 函数导致图中断的示例（实际输出可能有所不同）。

```py
import torch

@torch.compile
def fn(x):
    x = x + 1
    with open("test.txt", "r") as f:
        return x + len(f.read())

fn(torch.ones(3, 3))
```

```
$TORCH_LOGS="graph_breaks" python playground.py
Graph break in user code at /data/users/williamwen/pytorch/playground.py:7
Reason: Unsupported: builtin: open [<class 'torch._dynamo.variables.constant.ConstantVariable'>, <class 'torch._dynamo.variables.constant.ConstantVariable'>] False
User code traceback:
File "/data/users/williamwen/pytorch/playground.py", line 7, in fn
    with open("test.txt", "r") as f:
Traceback (most recent call last):
File "/data/users/williamwen/pytorch/torch/_dynamo/symbolic_convert.py", line 635, in wrapper
    return inner_fn(self, inst)
        ^^^^^^^^^^^^^^^^^^^^
File "/data/users/williamwen/pytorch/torch/_dynamo/symbolic_convert.py", line 2414, in CALL
    self._call(inst)
File "/data/users/williamwen/pytorch/torch/_dynamo/symbolic_convert.py", line 2408, in _call
    self.call_function(fn, args, kwargs)
File "/data/users/williamwen/pytorch/torch/_dynamo/symbolic_convert.py", line 962, in call_function
    self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/data/users/williamwen/pytorch/torch/_dynamo/variables/builtin.py", line 997, in call_function
    return handler(tx, args, kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^
File "/data/users/williamwen/pytorch/torch/_dynamo/variables/builtin.py", line 831, in <lambda>
    return lambda *args: unimplemented(error_msg)
                        ^^^^^^^^^^^^^^^^^^^^^^^^
File "/data/users/williamwen/pytorch/torch/_dynamo/exc.py", line 313, in unimplemented
    raise Unsupported(msg, case_name=case_name)
torch._dynamo.exc.Unsupported: builtin: open [<class 'torch._dynamo.variables.constant.ConstantVariable'>, <class 'torch._dynamo.variables.constant.ConstantVariable'>] False
```

### 守卫

`torch.compile` 在追踪代码时会对运行时值做出一些假设。在追踪过程中，我们会生成“守卫”，这些是对这些假设的运行时检查。在后续调用编译函数时，会运行守卫以确定是否可以重用先前编译的代码。运行时检查的示例包括常量值、类型和对象 ID。

以下是生成守卫的示例。`TENSOR_MATCH` 守卫检查输入的类型、设备、数据类型、形状等。

```py
import torch

@torch.compile
def fn(x):
    return x + 1

fn(torch.ones(3, 3))
```

```
$ TORCH_LOGS="guards" python playground.py
GUARDS:
```

TREE_GUARD_MANAGER:
+- RootGuardManager
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:471 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['x'], accessed_by=DictGetItemGuardAccessor(x)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3, 3], stride=[3, 1])  # return x + 1  # playground.py:6 in fn
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # return x + 1  # playground.py:6 in fn
```

### 重新编译

如果所有先前编译代码实例的守卫检查都失败，那么 `torch.compile` 必须"重新编译"该函数，需要再次追踪原始代码。

在下面的示例中，由于检查张量参数形状的守卫失败，重新编译是必要的。

```py
import torch

@torch.compile
def fn(x):
    return x + 1

fn(torch.ones(3, 3))
fn(torch.ones(4, 4))
```

```
$ TORCH_LOGS="recompiles" python playground.py
Recompiling function fn in /data/users/williamwen/pytorch/playground.py:3
    triggered by the following guard failure(s):
    - 0/0: tensor 'L['x']' size mismatch at index 0. expected 3, actual 4
```

### 动态形状

`torch.compile` 最初假设张量形状是静态/常量，并基于这些假设设置守卫。
通过使用"动态形状"，我们可以让 `torch.compile` 生成能够接受不同形状张量输入的编译代码——我们避免了每次形状不同时都重新编译。
默认情况下，自动动态形状是启用的 `torch.compile(dynamic=None)`——如果由于形状不匹配导致编译失败，将尝试使用动态形状重新编译。
动态形状也可以完全启用 `dynamic=True` 或禁用 `dynamic=False`。

下面，我们启用动态形状，并注意到不再需要重新编译。

```py
import torch

@torch.compile(dynamic=True)
def fn(x):
    return x + 1

fn(torch.ones(3, 3))
fn(torch.ones(4, 4))
```

```
$ TORCH_LOGS="dynamic,recompiles" python playground.py
create_symbol s0 = 3 for L['x'].size()[0] [2, int_oo] at playground.py:5 in fn (_dynamo/variables/builder.py:2718 in <lambda>), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="s0"
produce_guards
produce_guards
```

有关动态形状的更多信息，请参阅 [动态形状手册](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.fh8zzonyw8ng)。

## 日志记录工具

(tlparse-torch-trace)=

### tlparse / TORCH_TRACE

`tlparse` / `TORCH_TRACE` 是一对工具，用于生成如下所示的编译报告：
<https://web.mit.edu/~ezyang/Public/bhack-20240609-tlparse/index.html>。

追踪非常容易收集。要收集追踪，请使用以下命令运行你的复现命令：

```
TORCH_TRACE="/tmp/tracedir" python foo.py
pip install tlparse
tlparse /tmp/tracedir
```

这种方法即使你在运行分布式作业时也有效，会为每个 rank 提供一个追踪。
它将在你的浏览器中打开类似于上面生成的 HTML 页面。
如果你正在为一个你没有独立复现的复杂问题提交错误报告，你仍然可以通过附上在 `/tmp/tracedir` 中生成的追踪日志来极大地帮助 PyTorch 开发人员。

```{warning}
追踪日志包含你所有的模型代码。
如果你正在处理的模型是敏感的，请不要共享追踪日志。追踪日志**不包含**权重。
```

```{raw} html
    <style>
        .red {background-color:#ff0000;}
        .green {background-color:#00ff00;}
        .dark-green {background-color:#027f02;}
    </style>
```

```{eval-rst}
.. role:: red

.. role:: green

.. role:: dark-green
```

`tlparse` 的输出主要面向 PyTorch 开发人员，其日志格式易于在 GitHub 上上传和共享。
然而，作为非 PyTorch 开发人员，你仍然可以从中提取有用的信息。
我们建议从报告中的内联帮助文本开始，它解释了其内容。
以下是一些你可以从 `tlparse` 中获得的信息：

-   通过查看堆栈树，可以了解编译了哪些模型代码？
    如果你不熟悉正在编译的代码库，这尤其有用！
-   存在多少个图中断 / 不同的编译区域？
    （每个不同的编译都是其自己颜色编码的块，如 {dark-green}`[0/0]`）。
    可能发生图中断的帧显示为浅绿色 {green}`[2/4]`。
    如果有很多帧，那是可疑的，表明你可能有一些灾难性的图中断，或者你的代码可能不太适合 `torch.compile`。
-   我重新编译特定帧多少次了？重新编译很多次的帧看起来像：
    {dark-green}`[10/0]` {dark-green}`[10/1]` {dark-green}`[10/2]`
    \- 如果某个东西被重新编译了很多次，那是非常可疑的，值得调查，即使它不是问题的根本原因。
-   是否存在编译错误？出错的帧看起来像 {red}`[0/1]`。
-   我为给定帧生成了哪些中间编译器产物？
    例如，你可以查看生成的高级 FX 图或生成的 Triton 代码。
-   特定帧是否有相关信息？你可以在 `compilation_metrics` 中找到这些信息。

(torch-logs)=

### TORCH_LOGS

你可以使用 `TORCH_LOGS` 环境变量来有选择地启用 `torch.compile` 堆栈的某些部分进行日志记录。
`TORCH_LOGS` 实际上是 `tlparse` 的日志来源。`TORCH_LOGS` 环境变量的格式如下：

```
TORCH_LOGS="<option1>,<option2>,..." python foo.py
```

有用的高级选项包括：

- `graph_breaks`：记录用户代码中发生图中断的位置及中断原因
- `guards`：记录生成的守卫条件
- `recompiles`：记录重新编译的函数以及导致重新编译的失败守卫条件
- `dynamic`：记录与动态形状相关的信息

此外，您也可以通过编程方式使用 `torch._logging.set_logs` 设置日志选项：

```py
import logging
torch._logging.set_logs(graph_breaks=True)
...
```

更多 `TORCH_LOGS` 选项请参阅 {ref}`troubleshooting-torch-logs-options`。
完整选项列表请查看 [torch.\_logging](https://pytorch.org/docs/stable/logging.html)
和 [torch.\_logging.set_logs](https://pytorch.org/docs/stable/generated/torch._logging.set_logs.html#torch._logging.set_logs)。

### tlparse 与 TORCH_LOGS 对比

通常，我们建议在遇到问题时首先使用 `tlparse`。
`tlparse` 非常适合调试大型模型并获取模型编译过程的高层概览。
另一方面，当我们已经知道是哪个 `torch.compile` 组件导致问题时，
`TORCH_LOGS` 更适合用于小型示例和细粒度的调试细节。

## 简单解决方案

这里我们描述一些涉及少量代码修改或更改 `torch.compile` 设置的解决方案。

### 在哪里应用 torch.compile？

我们建议在不会导致过多问题的最高层级函数上应用 `torch.compile`。
通常，这是指包含优化器但不包含循环的训练或评估步骤、顶层的 `nn.Module`，
或者某些子 `nn.Module`。`torch.compile` 特别不擅长处理像 DDP 或 FSDP 这样的分布式包装器模块，
因此考虑将 `torch.compile` 应用于传递给包装器的内部模块。

```py
# 推理
model = ...
opt_model = torch.compile(model)

for _ in range(N_ITERS):
    inp = ...
    out = opt_model(inp)
```

```py
# 训练
model = ...
opt = torch.optim.Adam(model.parameters())

@torch.compile
def train(mod, data):
    opt.zero_grad(True)
    pred = mod(data[0])
    loss = torch.nn.CrossEntropyLoss()(pred, data[1])
    loss.backward()
    opt.step()

for _ in range(N_ITERS):
    inp = ...
    train(model, inp)
```

```py
# DistributedDataParallel
model = ...
opt_model = torch.compile(model)
model_ddp = DistributedDataParallel(opt_model, ...)

for _ in range(N_ITERS):
    inp = ...
    out = model_ddp(inp)
```

### 禁用和抑制错误

对于某些模型架构，模型的部分内容特别难以编译——要么存在许多图中断，要么会发生崩溃。
您可能希望显式禁用模型中这些有问题的部分，以便将 `torch.compile` 应用于可正常工作的部分。
您可以使用 `@torch.compiler.disable` 装饰器来实现这一点。当 `torch.compile` 尝试调用被禁用的函数时，
它会中断图并跳过对被禁用函数的追踪，在调用后恢复追踪。
默认情况下，从被禁用函数进行的所有递归调用也会被禁用。使用 `recursive=False` 选项可以允许对递归调用进行编译。

```py
def bad1_inner(...):
    # 被跳过

@torch.compiler.disable
def bad1_outer(...):
    # 被跳过
    bad1_inner(...)

def bad2_inner(...)
    # 被追踪

@torch.compiler.disable(recursive=False)
def bad2_outer(...):
    # 被跳过
    bad2_inner(...)

@torch.compile
def fn(...):
    # 图中断
    bad1_outer(...)
        ...
    # 图中断
    bad2_outer(...)
```

例如，我们在推荐模型中使用 `torch.compiler.disable` 来禁用稀疏架构上的 `torch.compile`，
因为稀疏架构难以编译。预处理和日志记录函数是其他通常会导致大量图中断且从编译中获益不大的函数示例。

如果您遇到编译器崩溃但希望继续执行，可以设置 `torch._dynamo.config.suppress_errors = True`。
当编译器崩溃时，我们将跳过追踪该函数并稍后重试。这不是最佳实践——最好根据需要手动添加禁用注解。

### 解决图中断问题

为了最大化优化机会，减少图中断的数量非常重要。
请记住，您可以使用 `tlparse` 或 `TORCH_LOGS="graph_breaks"` 查看发生的图中断。
通常，图中断由以下原因之一引起：

1. 您尝试执行的操作本质上无法被追踪，例如数据依赖的控制流。
2. 您尝试执行的操作尚未得到支持。
   例如，目前对使用内置 Python `inspect` 模块的代码追踪支持有限。
3. 您的代码中存在错误。例如，您可能尝试使用错误数量的参数调用函数。

图中断日志会告诉您用户代码位置和图中断的原因。
不幸的是，如果没有对 Dynamo 的深入理解，许多图中断是无法解决的。
甚至确定三种原因中哪一种是导致图中断的真正原因都可能具有挑战性。
我们正在努力使图中断信息更具可操作性。

此外，不同图中断导致的优化机会损失影响也不同。
例如，发生在模型 `forward` 方法中间的图中断可能比发生在 `forward` 开头预处理部分的图中断影响更负面。
因此，防止*每一个*中断并不关键，关键是防止那些导致显著性能损失的中断。

如果图中断信息没有提示任何操作，您怀疑图中断的原因是 (2)，
并且您认为图中断正在导致性能损失，请将图中断作为问题报告。
如果一个函数有许多图中断，考虑在该函数上禁用编译，因为图中断的开销成本可能会变得过高。

以下是一些常见的图中断及其解决方案。

#### 数据依赖操作

`torch.compile` 会在数据依赖操作上发生图中断，例如数据依赖的控制流（if 语句、涉及张量的循环）和直接的张量数据访问（`.item`、`.data_ptr`）。

```py
import torch

@torch.compile
def fn(x):
    y = x.sum()
    if y > 0:
        return x + y.item()
    return x - y.item()

fn(torch.ones(3, 3))
```

```
$ TORCH_LOGS="graph_breaks" python playground.py
Graph break in user code at /data/users/williamwen/pytorch/playground.py:6
Reason: Data-dependent jump
User code traceback:
File "/data/users/williamwen/pytorch/playground.py", line 6, in fn
    if y > 0:

Graph break in user code at /data/users/williamwen/pytorch/playground.py:7
Reason: Unsupported: Tensor.item
User code traceback:
File "/data/users/williamwen/pytorch/playground.py", line 7, in torch_dynamo_resume_in_fn_at_6
    return x + y.item()
Traceback (most recent call last):
File "/data/users/williamwen/pytorch/torch/_dynamo/symbolic_convert.py", line 616, in wrapper
    return inner_fn(self, inst)
        ^^^^^^^^^^^^^^^^^^^^
File "/data/users/williamwen/pytorch/torch/_dynamo/symbolic_convert.py", line 2288, in CALL
    self._call(inst)
File "/data/users/williamwen/pytorch/torch/_dynamo/symbolic_convert.py", line 2282, in _call
    self.call_function(fn, args, kwargs)
File "/data/users/williamwen/pytorch/torch/_dynamo/symbolic_convert.py", line 838, in call_function
    self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/data/users/williamwen/pytorch/torch/_dynamo/variables/misc.py", line 1038, in call_function
    return self.obj.call_method(tx, self.name, args, kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/data/users/williamwen/pytorch/torch/_dynamo/variables/tensor.py", line 527, in call_method
    result = handler_method(*args, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/data/users/williamwen/pytorch/torch/_dynamo/variables/tensor.py", line 773, in method_item
    unimplemented("Tensor.item")
File "/data/users/williamwen/pytorch/torch/_dynamo/exc.py", line 304, in unimplemented
    raise Unsupported(msg, case_name=case_name)
torch._dynamo.exc.Unsupported: Tensor.item
```

解决这些图中断的通用方法是避免执行数据依赖操作。一些具体的解决方法包括：

- 如果你的控制流实际上并不依赖于数据值，考虑修改你的代码，使其在常量上执行控制流。

```py
# 旧代码
x = torch.randn(3, 3)
@torch.compile
def fn(y):
    if x.sum() > 0:
        return y + x
    else:
        return y - x

# 新代码
x = torch.randn(3, 3)
cond = (x.sum() > 0).item()
@torch.compile
def fn(y):
    if cond:
        return y + x
    else:
        return y - x
```

- 使用高阶操作符如 `torch.cond` (<https://pytorch.org/docs/main/cond.html>) 来代替数据依赖的控制流

```py
# 旧代码
@torch.compile
def fn(x):
    if x.sum() > 0:
        return x + 1
    return x - 1

# 新代码
@torch.compile
def fn(x):
    return torch.cond(
        x.sum() > 0,
        lambda x: x + 1,
        lambda x: x - 1,
        (x,),
    )
```

- 如果你有 `.item()` 调用，尝试设置 `torch._dynamo.config.capture_scalar_outputs = True` 或 `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`
- 将函数中有问题的部分包装在自定义操作符中

#### 自定义操作符

如果你的代码由于缺少支持或根本上的不兼容性导致 `torch.compile` 难以追踪，你可以考虑将有问题的代码包装在自定义操作符中。

自定义操作符需要一些额外的工作才能使其与 `torch.compile` 兼容。
更多详情请参阅 <https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html>。

#### 打印

打印/日志记录/发出警告将导致图中断。如果你的函数进行了许多日志调用，例如一个记录训练迭代数据的函数，考虑对其应用 `torch.compiler.disable`。

或者，你可以尝试使用 `torch._dynamo.config.reorderable_logging_functions`。
此配置用于重新排序日志函数，使它们在跟踪函数的末尾被调用，从而避免图中断。但是，如果发生突变等情况，记录的内容可能会有所不同。

```py
import torch

torch._dynamo.config.reorderable_logging_functions.add(print)

@torch.compile
def fn(x):
    x += 1
    print("log!")
    return torch.sin(x)

fn(torch.ones(3, 3))
```

```
$ TORCH_LOGS="graph_breaks" python playground.py
log!
```

#### 错误的代码

你的代码可能是错误的，或者正遇到来自 `torch.compile` 外部的错误。
在下面的代码中，我们在 `torch.sin` 调用中打错了字，提供了一个额外的参数。

```py
import torch

@torch.compile
def fn(x):
    y = torch.sin(x, x)
    return y

fn(torch.ones(3, 3))
```

```
$ TORCH_LOGS="graph_breaks" python playground.py
Graph break in user code at /data/users/williamwen/pytorch/playground.py:5
Reason: Unsupported: TypeError <built-in method sin of type object at 0x7fd6fd764600>: sin() takes 1 positional argument but 2 were given
User code traceback:
File "/data/users/williamwen/pytorch/playground.py", line 5, in fn
    y = torch.sin(x, x)
...
```

从日志中很难判断错误是由你的代码引起的还是由 `torch.compile` 的 bug 引起的。
为了区分，我们建议尝试在不使用 `torch.compile` 的情况下运行你的代码，看看是否仍然出现错误。

### 处理重新编译

你可以使用 `tlparse` 或 `TORCH_LOGS=recompiles` 来查看重新编译及其原因。

#### 是否启用了动态形状？

由于形状不匹配导致的重新编译形式如下：

```
tensor 'L['x']' size mismatch at index 0. expected 3, actual 4
```

请确保 `torch.compile` 的 `dynamic` 选项没有设置为 `False`。
默认选项 `dynamic=None` 只会在第一次编译后尝试动态形状。
你可以设置 `dynamic=True` 来尽可能地进行动态编译。

有关动态形状的更多信息，请参阅[动态形状手册](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.fh8zzonyw8ng)。

#### 更改缓存大小限制

函数可以重新编译的次数是有限制的，由 `torch._dynamo.config.recompile_limit` 和 `torch._dynamo.config.accumulated_recompile_limit` 决定。
如果任一限制被超过，我们将不再尝试编译该函数，而是以即时执行的方式运行该函数。
`torch.compile` 还会发出一个警告，其中包含受影响的函数以及触发了哪个限制。
在下面的示例中，每次函数调用都会导致重新编译尝试。
当我们达到缓存大小限制（8）时，我们停止尝试重新编译。

```py
import torch

@torch.compile(dynamic=False)
def fn(x):
    return x + 1

for i in range(1, 10):
    fn(torch.ones(i))
```

```
$ python playground.py
torch._dynamo hit config.recompile_limit (8)
    function: 'fn' (/data/users/williamwen/pytorch/playground.py:5)
    last reason: 0/0: tensor 'L['x']' size mismatch at index 0. expected 1, actual 9
```

如果您知道重新编译的次数有一个合理的恒定上限，您可以提高缓存大小限制。
如果重新编译的成本超过了编译带来的好处，那么您可以考虑降低缓存大小限制。

#### 用张量包装常量

默认情况下，`int` / `float` 变量被视为常量并受到相应的保护。
在下面的示例中，每次函数调用都会导致重新编译。

```py
import torch

@torch.compile
def fn(x, c):
    return x + c

for i in range(1, 10):
    fn(torch.ones(i), 0.5 + i)
```

```
$ TORCH_LOGS="recompiles" python playground.py
Recompiling function fn in /data/users/williamwen/pytorch/playground.py:3
    triggered by the following guard failure(s):
    - 0/7: L['c'] == 8.5
    - 0/6: L['c'] == 7.5
    - 0/5: L['c'] == 6.5
    - 0/4: L['c'] == 5.5
    - 0/3: L['c'] == 4.5
    - 0/2: L['c'] == 3.5
    - 0/1: L['c'] == 2.5
    - 0/0: L['c'] == 1.5
torch._dynamo hit config.recompile_limit (8)
    function: 'fn' (/data/users/williamwen/pytorch/playground.py:3)
    last reason: 0/0: L['c'] == 1.5
```

特别是对于学习率调度器，使用常量初始化可能导致重新编译：

```py
import torch

mod = torch.nn.Linear(3, 3)
opt = torch.optim.Adam(mod.parameters(), lr=0.01)
sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9)

@torch.compile
def fn(inp):
    opt.zero_grad(True)
    out = mod(inp).sum()
    out.backward()
    opt.step()
    sched.step()

for i in range(1, 10):
    fn(torch.ones(3, 3))
```

```
$ TORCH_LOGS="recompiles" python playground.py
Recompiling function step in /data/users/williamwen/pytorch/torch/optim/adam.py:189
    triggered by the following guard failure(s):
    - 3/7: L['self'].param_groups[0]['lr'] == 0.004782969000000002
    - 3/6: L['self'].param_groups[0]['lr'] == 0.005314410000000002
    - 3/5: L['self'].param_groups[0]['lr'] == 0.005904900000000002
    - 3/4: L['self'].param_groups[0]['lr'] == 0.006561000000000002
    - 3/3: L['self'].param_groups[0]['lr'] == 0.007290000000000001
    - 3/2: L['self'].param_groups[0]['lr'] == 0.008100000000000001
    - 3/1: L['self'].param_groups[0]['lr'] == 0.009000000000000001
    - 3/0: L['self'].param_groups[0]['lr'] == 0.01
torch._dynamo hit config.recompile_limit (8)
    function: 'step' (/data/users/williamwen/pytorch/torch/optim/adam.py:189)
    last reason: 3/0: L['self'].param_groups[0]['lr'] == 0.01
```

在这两个示例中，我们可以将浮点变量包装在张量中，以防止重新编译。

```py
# 第一个示例
for i in range(1, 10):
    fn(torch.ones(i), torch.tensor(0.5 + i))

# 第二个示例
opt = torch.optim.Adam(mod.parameters(), lr=torch.tensor(0.01))
sched = torch.optim.lr_scheduler.ExponentialLR(opt, torch.tensor(0.9))
```

## 报告问题

如果上面提供的解决方法不足以让 `torch.compile` 正常工作，那么您应该考虑向 PyTorch 报告该问题。
但是，您可以做一些事情来大大简化我们的工作。

### 消融测试

使用 `torch.compile` 的 `backend=` 选项检查 `torch.compile` 堆栈的哪个组件导致了问题。
特别是，尝试：

- `torch.compile(fn, backend="eager")`，它只运行 TorchDynamo，即 `torch.compile` 的图捕获组件。
- `torch.compile(fn, backend="aot_eager")`，它运行 TorchDynamo 和 AOTAutograd，后者在编译期间额外生成反向图。
- `torch.compile(fn, backend="aot_eager_decomp_partition")`，它运行 TorchDynamo 和 AOTAutograd，并带有算子分解/分区。
- `torch.compile(fn, backend="inductor")`，它运行 TorchDynamo、AOTAutograd 和 TorchInductor，后者是生成编译内核的后端 ML 编译器。

如果您仅在 Inductor 后端失败，您还可以测试各种 Inductor 模式：

- `torch.compile(fn, backend="inductor", mode="default")`
- `torch.compile(fn, backend="inductor", mode="reduce-overhead")`
- `torch.compile(fn, backend="inductor", mode="max-autotune")`

您还可以检查动态形状是否在任何后端导致问题：

- `torch.compile(fn, dynamic=True)` （始终使用动态形状）
- `torch.compile(fn, dynamic=False)` （从不使用动态形状）
- `torch.compile(fn, dynamic=None)` （自动动态形状）

### 二分查找

您是否尝试过最新的 nightly 版本？过去是否正常工作但现在不再工作？
您能否通过二分查找来确定问题首次出现的 nightly 版本？
二分查找对于性能、精度或编译时间回归特别有帮助，因为问题的根源可能并不明显。

### 创建复现示例

创建可复现示例需要大量工作，如果您没有时间完成，这完全可以理解。
然而，如果您是一位积极但不熟悉 `torch.compile` 内部机制的用户，
创建一个独立的可复现示例对我们修复错误的能力会产生巨大影响。
如果没有可复现示例，您的错误报告必须包含足够的信息，以便我们能够识别问题的根本原因并从头编写复现示例。

以下是有用的可复现示例列表，按优先级从高到低排列：

1. **独立、小型可复现示例：** 一个没有外部依赖、代码少于 100 行的脚本，运行时能复现问题。
2. **独立、大型可复现示例：** 即使规模较大，但独立性强是一个巨大优势！
3. **依赖项可控的非独立可复现示例：**
   例如，如果在 `pip install transformers` 后运行脚本能复现问题，
   这是可控的。我们很可能能够运行并进行调查。
4. **需要大量设置的非独立可复现示例：** 这可能涉及下载数据集、
   多个环境设置步骤，或需要特定系统库版本而要求使用 Docker 镜像。
   设置越复杂，我们重建环境的难度就越大。

   :::{note}
       Docker 简化了设置，但使环境变更复杂化，因此它并非完美解决方案，不过必要时我们会使用它。
   :::

在某种程度上正交的是，可以在单个进程中运行的可复现示例优于
需要多进程训练的可复现示例（但再次强调，如果您只有多进程复现示例，我们也会接受！）。

此外，以下是您可以在问题中检查并尝试在复现示例中复现的方面（非详尽列表）：

- **自动求导**。您的张量输入是否设置了 `requires_grad=True`？您是否在输出上调用了 `backward()`？
- **动态形状**。您是否设置了 `dynamic=True`？或者您是否使用不同形状多次运行了测试代码？
- **自定义运算符**。实际工作流中是否涉及自定义运算符？
  您能否使用 Python 自定义运算符 API 复现其某些重要特性？
- **配置**。您是否设置了所有相同的配置？
  这包括 `torch._dynamo.config` 和 `torch._inductor.config` 设置，
  以及 `torch.compile` 的参数，如 `backend` / `mode`。
- **上下文管理器**。您是否复现了任何活动的上下文管理器？
  这可能是 `torch.no_grad`、自动混合精度、`TorchFunctionMode` / `TorchDispatchMode`、
  激活检查点、编译自动求导等。
- **张量子类**。是否涉及张量子类？

### 最小化工具

最小化工具是 `torch.compile` 的早期工具，给定一个在尝试运行或编译时崩溃的 FX 图，
它会找到一个同样会崩溃的子图，并输出执行该子图操作的代码。
本质上，最小化工具为特定类别的 `torch.compile` 相关崩溃找到最小复现示例。
这假设我们能够成功追踪代码。

不幸的是，目前大多数情况下，最小化工具无法按预期工作，可能需要替代方法。
这可能是因为可以通过这种方式自动复现的错误通常更容易修复，
并且已经得到解决，留下的是更复杂且不易复现的问题。
然而，尝试使用最小化工具很简单，因此即使可能不会成功，也值得一试。

操作最小化工具的说明可以在[此处](https://pytorch.org/docs/stable/torch.compiler_troubleshooting_old.html)找到。
如果编译器崩溃，您可以设置 `TORCHDYNAMO_REPRO_AFTER="dynamo"` 或 `TORCHDYNAMO_REPRO_AFTER="aot"`
`aot` 选项更可能成功，尽管它可能无法识别 `AOTAutograd` 问题。这将生成 `repro.py` 文件，可能有助于诊断问题。
对于与精度相关的问题，考虑设置 `TORCHDYNAMO_REPRO_LEVEL=4`。请注意，这可能无法总是成功识别有问题的子图。

## 深入调试

本节提供了用于独立调试 `torch.compile` 问题
或更深入理解 `torch.compile` 堆栈的工具和技术。
这些方法比上面介绍的方法更复杂，PyTorch 开发人员经常使用它们
来调试实际的 `torch.compile` 问题。

以下是堆栈的高级概述：

![Torch Dynamo 堆栈](../../_static/img/dynamo/td_stack.png)

该堆栈包含三个主要组件：TorchDynamo、AOTAutograd 和 Inductor。
我们的调试策略首先确定错误发生在哪个组件，
然后单独调试该组件。要确定导致问题的组件，
请参阅上面 `报告问题` 下的 `消融` 部分。有关调试特定组件的指导，请参考以下部分。

### TorchDynamo

#### 记录 Dynamo 正在追踪的内容

`TORCH_LOGS=trace_bytecode` 选项使您能够查看 Dynamo 正在追踪的确切字节码指令，
以及 Python 解释器堆栈的符号表示。当遇到图中断或崩溃时，
建议检查最后几条追踪的字节码指令。

您还可以使用 `TORCH_LOGS=trace_source` 来查看 Dynamo 正在追踪的源代码行。
这与 `trace_bytecode` 结合使用很有用，可以查看每条追踪的字节码指令对应的源代码行。

最后，您可以使用 `TORCH_LOGS=graph_code` 来查看表示 Dynamo 追踪的 FX 图的 Python 代码。
您可以查看此代码以双重检查是否追踪了正确的操作。

```py
import torch

def g(x, y):
    return x + y

@torch.compile(backend="eager")
def f(x):
    x = torch.sin(x)
    x = g(x, x)
    return x

f(torch.ones(3, 3))
```

```
$ TORCH_LOGS="trace_bytecode,trace_source,graph_code" python playground.py
TRACE starts_line /data/users/williamwen/pytorch/playground.py:6 in f ()
    @torch.compile(backend="eager")
TRACE RESUME 0 []
TRACE starts_line /data/users/williamwen/pytorch/playground.py:8 in f (f)
        x = torch.sin(x)
TRACE LOAD_GLOBAL torch []
TRACE LOAD_ATTR sin [NullVariable(), PythonModuleVariable(<module 'torch' from '/data/users/williamwen/pytorch/torch/__init__.py'>)]
TRACE LOAD_FAST x [NullVariable(), TorchInGraphFunctionVariable(<built-in method sin of type object at 0x7f00f6964600>)]
TRACE CALL 1 [NullVariable(), TorchInGraphFunctionVariable(<built-in method sin of type object at 0x7f00f6964600>), LazyVariableTracker()]
TRACE STORE_FAST x [TensorVariable()]
TRACE starts_line /data/users/williamwen/pytorch/playground.py:9 in f (f)
        x = g(x, x)
TRACE LOAD_GLOBAL g []
TRACE LOAD_FAST x [NullVariable(), UserFunctionVariable()]
TRACE LOAD_FAST x [NullVariable(), UserFunctionVariable(), TensorVariable()]
TRACE CALL 2 [NullVariable(), UserFunctionVariable(), TensorVariable(), TensorVariable()]
TRACE starts_line /data/users/williamwen/pytorch/playground.py:3 in g (g) (inline depth: 1)
    def g(x, y):
TRACE RESUME 0 []
TRACE starts_line /data/users/williamwen/pytorch/playground.py:4 in g (g) (inline depth: 1)
        return x + y
TRACE LOAD_FAST x []
TRACE LOAD_FAST y [TensorVariable()]
TRACE BINARY_OP 0 [TensorVariable(), TensorVariable()]
TRACE RETURN_VALUE None [TensorVariable()]
TRACE STORE_FAST x [TensorVariable()]
TRACE starts_line /data/users/williamwen/pytorch/playground.py:10 in f (f)
        return x
TRACE LOAD_FAST x []
TRACE RETURN_VALUE None [TensorVariable()]
TRACED GRAPH
===== __compiled_fn_1 =====
/data/users/williamwen/pytorch/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3][3, 1]cpu"):
        l_x_ = L_x_

        # File: /data/users/williamwen/pytorch/playground.py:8 in f, code: x = torch.sin(x)
        x: "f32[3, 3][3, 1]cpu" = torch.sin(l_x_);  l_x_ = None

        # File: /data/users/williamwen/pytorch/playground.py:4 in g, code: return x + y
        x_1: "f32[3, 3][3, 1]cpu" = x + x;  x = None
        return (x_1,)
```

#### 在 Dynamo 追踪过程中设置断点

有时，在 Dynamo/用户代码中插入断点有助于查看在追踪用户代码时 Dynamo 的状态。
不幸的是，以常规 Python 方式插入断点会导致 TorchDynamo 中出现图中断，因此我们将无法在预期设置断点的位置查看 Dynamo 的状态。

设置断点的第一种方法是在 Dynamo 源代码中插入断点。推荐放置断点的三个位置是：

- 在 `torch/_dynamo/symbolic_convert.py` 中，在以有问题的字节码指令命名的函数处设置断点，
  例如 `def CALL_FUNCTION` 和 `def STORE_ATTR`。您可以根据输入条件设置断点，
  例如，指令的 `argval`，或者栈顶对象的名称，因为某些字节码操作码使用频繁。
- 在图中断或错误起源处设置断点。通常，图中断是通过调用 `unimplemented(...)` 发出的。
- 在 `torch/_dynamo/variables/builder.py` 的 `_wrap` 函数中设置断点。您可能需要根据输入条件设置断点。
  此函数决定如何符号化表示给定的值。如果您怀疑某个值的表示不正确，请考虑在此处设置断点。

第二种插入断点的方法是使用 `torch._dynamo.comptime.comptime.breakpoint`：

```py
from torch._dynamo.comptime import comptime

@torch.compile
def f(...):
    ...
    comptime.breakpoint()
    ...
```

comptime 断点非常方便，因为它允许您在正在追踪的用户代码中的特定位置检查 Dynamo 状态。
它不需要您在 Dynamo 源代码中插入断点，也不需要根据变量条件设置断点。

当 comptime 断点被触发时，您可以执行以下操作：

- `ctx.print_bt()` 打印用户堆栈跟踪
- `ctx.print_locals()` 打印所有当前局部变量
- `ctx.print_graph()` 打印当前追踪的图
- `ctx.disas()` 打印当前追踪函数的字节码
- 使用标准的 `pdb` 命令，例如 `bt/u/d/n/s/r` - 您可以向上查看 `pdb` 堆栈以检查更多 Dynamo 内部状态

```py
import torch
from torch._dynamo.comptime import comptime

@torch.compile(backend="eager")
def f(x):
    y = x + 1
    comptime.breakpoint()
    y = y + 1
    return y

f(torch.ones(3, 3))
```

```
$ python playground.py
--Return--
> /data/users/williamwen/pytorch/torch/_dynamo/comptime.py(392)inner()->None
-> builtins.breakpoint()
(Pdb) ctx.print_bt()
File "/data/users/williamwen/pytorch/playground.py", line 7, in f
    comptime.breakpoint()

(Pdb) ctx.print_locals()
x = FakeTensor(..., size=(3, 3))
y = FakeTensor(..., size=(3, 3))
(Pdb) bt
...
/data/users/williamwen/pytorch/torch/_dynamo/symbolic_convert.py(826)call_function()
-> self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
/data/users/williamwen/pytorch/torch/_dynamo/variables/misc.py(331)call_function()
-> func(ComptimeContext(tx))
> /data/users/williamwen/pytorch/torch/_dynamo/comptime.py(392)inner()->None
-> builtins.breakpoint()
(Pdb) ctx.print_graph()



def forward(self, L_x_: "f32[3, 3]"):
    l_x_ = L_x_

    # File: /data/users/williamwen/pytorch/playground.py:6 in f, code: y = x + 1
    y: "f32[3, 3]" = l_x_ + 1;  l_x_ = y = None
```

% TODO(uncomment/update once we improve this API)
% 调试大型模型
% ^^^^^^^^^^^^^^^^^^^^^^
%
% 在大型模型上调试 TorchDynamo 可能比较棘手，主要是因为 Dynamo 会追踪大量代码。
% 很难找到有问题的函数，或者确定在哪里设置断点。
% 即使我们找到了有问题的函数，我们也不想处理大量的日志输出。
% 幸运的是，你可以使用 ``TORCHDYNAMO_DEBUG_FUNCTION=<function name>``，它将 dynamo 追踪限制在仅具有特定名称（精确匹配）的函数上。
% 这将允许你将模型中的所有函数过滤到感兴趣的函数。
% 结合上述调试策略使用此功能。

#### 字节码生成错误

虽然不常见，但 Dynamo 可能会生成错误的字节码。如果你确定以下情况，则可能发生此错误：

- 消融实验表明错误发生在 TorchDynamo 层面
- 错误并非来自 TorchDynamo 堆栈帧
- 错误看起来更像是用户错误而非 Dynamo 错误，或者是段错误
- 在不使用 `torch.compile` 时不会发生此错误

字节码生成错误通常很难修复，我们建议提交问题而不是尝试自己修复。
如果你有兴趣查看 Dynamo 生成的字节码，可以使用 `TORCH_LOGS=bytecode`。
你可以在此处查看 Dynamo 生成的字节码的高级概述 [here](https://docs.google.com/presentation/d/1tMZOoAoNKF32CAm1C-WfzdVVgoEvJ3lp/edit?usp=sharing&ouid=114922067987692817315&rtpof=true&sd=true)。

### AOTAutograd

AOTAutograd 错误通常难以调试——我们建议直接提交问题。
AOTAutograd 日志输出主要用于查看 Inductor 的输入是什么。

% TODO
% TorchInductor
% -------------

% TODO

(troubleshooting-torch-logs-options)=

### TORCH_LOGS 选项摘要

以下是有用的 `TORCH_LOGS` 选项摘要：

```{eval-rst}
.. list-table::
    :widths: 25 50
    :header-rows: 1

    * - 选项
      - 描述
    * - +all
      - 输出所有 ``torch.compile`` 组件的调试日志
    * - +dynamo
      - 输出 TorchDynamo 的调试日志
    * - +aot
      - 输出 AOTAutograd 的调试日志
    * - +inductor
      - 输出 TorchInductor 的调试日志
    * - dynamic
      - 输出动态形状的日志
    * - graph_code
      - 输出 Dynamo 生成的 FX 图的 Python 代码
    * - graph_sizes
      - 输出 Dynamo 生成的 FX 图的张量大小
    * - trace_bytecode
      - 输出 Dynamo 正在追踪的字节码指令以及 Dynamo 正在跟踪的符号解释器堆栈
    * - trace_source
      - 输出 Dynamo 当前正在追踪的原始源代码行
    * - bytecode
      - 输出 Dynamo 生成的字节码
    * - guards
      - 输出生成的守卫
    * - recompiles
      - 输出重新编译原因（仅输出第一个失败的守卫检查）
    * - recompiles_verbose
      - 输出重新编译发生时所有失败的守卫检查
    * - aot_graphs
      - 输出 AOTAutograd 生成的图
    * - aot_joint_graphs
      - 输出 AOTAutograd 生成的联合前向-反向图
    * - output_code
      - 输出 Inductor 生成的代码
    * - kernel_code
      - 按内核输出 Inductor 生成的代码
    * - schedule
      - 输出 Inductor 调度日志
    * - perf_hints
      - 输出 Inductor 性能提示日志
    * - fusion
      - 输出 Inductor 融合日志
```

有关完整选项列表，请参阅 [torch.\_logging](https://pytorch.org/docs/stable/logging.html)
和 [torch.\_logging.set_logs](https://pytorch.org/docs/stable/generated/torch._logging.set_logs.html#torch._logging.set_logs)。

## 相关文章

- [torch.compile 教程](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [torch.compile 细粒度 API](https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html)
- [torch.compile 常见问题解答](https://pytorch.org/docs/stable/torch.compiler_faq.html)
- [torch.compiler 命名空间概述](https://pytorch.org/docs/stable/torch.compiler.html#torch-compiler-overview)
- [torch.compiler API 参考](https://pytorch.org/docs/stable/torch.compiler_api.html)
- [分析 torch.compile](https://pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html)
- [torch.compile 缺失手册](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?usp=sharing)
- [动态形状手册](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.fh8zzonyw8ng)
- [TorchInductor 缓存教程](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)