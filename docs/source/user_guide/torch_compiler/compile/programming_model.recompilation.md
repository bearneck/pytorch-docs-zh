---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

```{code-cell}
:tags: [remove-cell]
import torch

import header_code

torch._logging.set_logs(recompiles=True)
```

# 处理重新编译

重新编译对于 `torch.compile` 的正确性是必要的，但可能导致编译时间显著增加。
因此，在保持正确性的同时最小化重新编译对于减少编译时间至关重要。

您可以使用 tlparse 或 `TORCH_LOGS=recompiles` 查看重新编译及其原因。

## 是否启用了动态形状？

在下面的示例中，由于形状不匹配，我们进行了重新编译：

```{code-cell}
@torch.compile
def fn(x):
    return x + 1
fn(torch.ones(3))
fn(torch.ones(4))
```

请确保 `torch.compile` 的 dynamic 选项未设置为 `False`。
默认选项 `dynamic=None` 仅在第一次编译后尝试动态形状。
您可以设置 `dynamic=True` 以尽可能提前进行动态编译：

```{code-cell}
@torch.compile(dynamic=True)
def gn(x):
    return x + 1
gn(torch.ones(3))
gn(torch.ones(4))
```

有关动态形状的更多信息，包括处理因动态形状导致的错误/重新编译，请参阅 [动态形状手册](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit?tab=t.0#heading=h.fh8zzonyw8ng)。

## 使用张量包装常量
默认情况下，`int` / `float` 变量被视为常量，并基于其精确值进行保护。
在下面的示例中，每次函数调用都会导致一次重新编译。

```{code-cell}
@torch.compile
def fn(x, c):
    return x + c
for i in range(5):
    fn(torch.ones(i), 0.5 + i)
```

特别是对于学习率调度器，使用常量初始化可能导致重新编译：

```{code-cell}
mod = torch.nn.Linear(3, 3)
opt = torch.optim.Adam(mod.parameters(), lr=0.01)
sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9)
@torch.compile
def gn(inp):
    opt.zero_grad(True)
    out = mod(inp).sum()
    out.backward()
    opt.step()
    sched.step()
for i in range(5):
    gn(torch.ones(3, 3))
```

在这两个示例中，我们可以将 `float` 变量包装在张量中以防止重新编译。

```{code-cell}
:tags: [remove-cell]
torch._dynamo.reset()
```

```{code-cell}
# 第一个示例
for i in range(5):
    fn(torch.ones(i), torch.tensor(0.5 + i))
# 第二个示例
opt = torch.optim.Adam(mod.parameters(), lr=torch.tensor(0.01))
sched = torch.optim.lr_scheduler.ExponentialLR(opt, torch.tensor(0.9))
for i in range(5):
    gn(torch.ones(3, 3))
```


## 更改缓存大小限制

函数可以重新编译的次数是有限制的，
由 `torch._dynamo.config.cache_size_limit` 和 `torch._dynamo.config.accumulated_cache_size_limit` 决定
（这两个值之间的确切区别详见 [`torch/_dynamo/cache_size.py`](https://github.com/pytorch/pytorch/blob/4ce6e6ec8890a3f6ee604c9efb3ff153825ce575/torch/_dynamo/cache_size.py#L14)）。
如果达到 Dynamo 缓存限制，那么所有未来的编译尝试**将导致函数被跳过（以即时模式运行）**。
如果保护条件通过，Dynamo 仍将尝试为未来的函数调用使用先前编译的字节码。
请注意，在达到重新编译限制的情况下，**所有嵌套的函数调用都将被跳过**
（Dynamo 将尝试为嵌套函数使用先前编译的字节码）。
Dynamo 还会发出一个警告，包含受影响的函数以及达到了哪个限制。
在下面的示例中，每次函数调用都会导致重新编译尝试。
当我们达到缓存大小限制（默认为 8）时，我们停止尝试重新编译。
（请注意，我们为了演示目的设置了 `dynamic=False`，以强制每次重新编译）。

```{code-cell}
@torch.compile(dynamic=False)
def fn(x):
    return x + 1
for i in range(1, 10):
    # 由于 dynamic=False，每次都会重新编译
    fn(torch.ones(i))
```

如果您知道重新编译的次数有一个合理的常数上限，可以提高缓存大小限制。
如果重新编译的成本超过了编译的好处，那么可以考虑降低缓存大小限制。

```{code-cell}
torch._dynamo.config.cache_size_limit = 16
@torch.compile(dynamic=False)
def gn(x):
    return x + 1
for i in range(1, 10):
    gn(torch.ones(i))
```

## 通过图中断减少重新编译成本
如果一个大型图正在重新编译并导致高编译时间，您可以有意引入
一个图中断以减少重新编译成本，代价是引入性能损失。

```{code-cell}
def very_large_function(x):
    return x + 1

@torch.compile(dynamic=False)
def fn(x, c):
    y = very_large_function(x)  # 每次都会重新编译
    return y + c

for i in range(1, 5):
    fn(torch.ones(3), i)

@torch.compile(dynamic=False)
def gn(x, c):
    y = very_large_function(x)  # 仅编译一次
    torch._dynamo.graph_break()
    return y + c  # 每次都会重新编译

for i in range(1, 5):
    gn(torch.ones(3), i)
```