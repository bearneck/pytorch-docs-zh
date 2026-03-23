(debugging-tlparse-torch-logs)=
# 使用 `tlparse` 和 `TORCH_LOGS=dynamic` 进行调试

`tlparse` 是一个用于分析和理解 PyTorch 编译过程的工具，特别是在处理动态形状时。它有助于识别代码中发生守卫（guards）和特化（specializations）的位置。

`TORCH_LOGS=dynamic` 是一个环境变量设置，用于启用动态形状操作的详细日志记录，从而深入了解执行过程中符号形状是如何被处理的。

本节将指导你如何使用 `tlparse` 和 `TORCH_LOGS=dynamic` 来排查代码中的动态形状问题，包括调试特化、守卫等。

# 调试特化

在以下示例中，`x.shape[0]` 是动态的，但由于乘法操作而变得特化：

```python
import torch

@torch.compile
def fn(x, y):
    return x * y

x = torch.randn(5)
y = torch.randn(5)
torch._dynamo.decorators.mark_dynamic(x, 0)

fn(x, y)
```

通过使用 `TORCH_LOGS=dynamic`，你可以在日志中观察到这种特化：

```xml
TORCH_LOGS=dynamic python tl.py
I0721 11:10:00.950000 845259 torch/fx/experistic/symbolic_shapes.py:3776] [0/0] create_env
I0721 11:10:01.030000 845259 torch/fx/experimental/symbolic_shapes.py:5117] [0/0] create_symbol s77 = 5 for L['x'].size()[0] [2, int_oo] return x * y  # tl.py:5 in fn (_dynamo/variables/builder.py:3466 in <lambda>), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="s77" or to suppress this message run with TORCHDYNAMO_EXTENDED_ADVICE="0"
I0721 11:10:01.038000 845259 torch/fx/experimental/symbolic_shapes.py:7211] [0/0] eval Eq(s77, 5) [guard added] return x * y  # tl.py:5 in fn (_subclasses/fake_impls.py:922 in infer_size), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="Eq(s77, 5)"
```

`eval Eq(s77, 5) [guard added] return x * y # tl.py:5` 这一行指示了特化的发生。

## 调试守卫

考虑以下代码，它可能由于动态形状而导致重新编译：

```python
import torch

@torch.compile
def fn(x, y):
    if x.shape[0] < 10:
        return x * y

x = torch.randn(5)
y = torch.randn(5)
torch._dynamo.decorators.mark_dynamic(x, 0)
torch._dynamo.decorators.mark_dynamic(y, 0)

fn(x, y)
```

要识别动态形状守卫的来源，请使用 `tlparse`。以下是一个 tlparse 输出示例：

```{image} ../../../_static/img/dynamic_shapes/tlparse9_debugging_guards.png
```

通过点击 `dynamo_cpp_guards` 链接，你可以查看编译中的所有守卫，包括符号形状守卫 `L['x'].size()[0] <= 9`。

细心的读者会注意到 0/1 特化，其中我们对 `L['x'].size()[0] >= 2` 进行了守卫。通过修改代码以使用无背景符号（unbacked symbols），可以移除这个守卫：

```python
import torch

@torch.compile
def fn(x, y):
    # 必要的运行时断言，因为我们无法对无背景符号进行守卫
    torch._check(x.shape[0] < 10)
    if x.shape[0] < 10:
        return x * y

x = torch.randn(5)
y = torch.randn(5)
torch._dynamo.decorators.mark_unbacked(x, 0)
torch._dynamo.decorators.mark_unbacked(y, 0)

fn(x, y)
```

现在，这个编译后的区域可以用于大小为 0 和 1 的输入：

```{image} ../../../_static/img/dynamic_shapes/tlparse10_debugging_guards_unbacked.png
```

```{seealso}
* {ref}`dynamic_shapes`
* {ref}`troubleshooting_guardondatadependentsymnode_errors`
```