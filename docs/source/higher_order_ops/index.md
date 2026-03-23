---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---


# 控制流运算符

控制流运算符是 PyTorch 中的结构化运算符，能够以兼容 `torch.compile` 和 `torch.export` 的方式表达复杂的控制流模式。与常规的 Python 控制流不同，这些运算符在 torch.compile 和 torch.export 过程中会保留其语义，从而支持在追踪程序中实现数据依赖的控制流。

```{warning}
控制流运算符是 PyTorch 中的原型功能。它们可能对某些输入/输出类型的支持有限，并且某些运算符可能不完全支持训练。请阅读以下链接了解更多关于功能分类的信息：
https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## 为何使用控制流运算符？

PyTorch 允许您使用原生 Python 编写模型，包括 `if` 语句、`for` 循环和 `while` 循环等控制流。这为灵活性带来了极大便利，但也给编译带来了挑战。

考虑以下简单示例：

```python
if mod.static_config == 0:
    return f(x)
return g(x)
```

两个分支可能包含完全不同的操作。如果我们每次遇到 `if` 语句都尝试编译两个分支，代码路径的数量将呈指数级爆炸式增长，并迅速变得难以处理。

为了解决这个问题，`torch.compile` 使用了**特化和守卫**。在追踪模型时，编译器会根据谓词的当前值选择一个代码路径（特化），然后添加一个守卫在运行时检查该假设。如果守卫失败，则重新编译。

循环的处理方式类似：编译器会展开循环并守卫迭代次数。这会生成一个易于优化的直线型计算图。

这种方法对于静态控制流效果很好，但在以下几种情况下会失效：

- **数据依赖的控制流**：当谓词依赖于张量的*值*时，编译器无法在编译时选择分支，因为值尚未知晓。同样，如果迭代次数依赖于张量值，编译器也无法展开 `while` 循环。编译器通过图中断并回退到 Python 来处理这种情况，这也使得模型无法在没有 Python 运行时的情况下运行（例如，在边缘设备上）。

- **动态形状依赖的控制流**：当循环迭代次数或分支谓词依赖于动态张量大小时，特化意味着编译后的代码仅适用于该特定大小。每当大小发生变化时，编译器都必须重新编译。

- **大型计算图**：即使迭代次数是静态的，展开大型循环也会创建一个随迭代次数线性增长的图，即使每次迭代执行的操作相同。这会导致编译时间过长和内存使用量过高。

控制流运算符通过将控制流表示为编译器理解的显式运算符来解决这些问题。这些运算符不会特化掉控制流，而是将其保留在编译后的图中。

## 可用运算符

- [Cond](cond.md)
- [While Loop](while_loop.md)
- [Scan](scan.md)
- [Associative Scan](associative_scan.md)
- [Map](map.md)


### 快速比较

| 运算符 | 用例 | 示例 |
|----------|----------|---------|
| [cond](cond.md) | 如果 `pred` 为 True，则返回 `true_fn(*operands)`，否则返回 `false_fn(*operands)`。 | `cond(pred, true_fn, false_fn, operands)` |
| [while_loop](while_loop.md) | 当 `cond_fn(*operands)` 为 True 时，执行 `body_fn(*operands)`，该函数返回下一次迭代的操作数。 | `while_loop(cond_fn, body_fn, operands)` |
| [scan](scan.md) | 对 `xs` 应用累积操作，并携带状态 | `scan(combine_fn, init, xs)` |
| [associative_scan](associative_scan.md) | 类似于 `scan`，但要求 `combine_fn` 具有结合律，以允许进行更多优化。 | `associative_scan(add, xs, dim=0)` |
| [map](map.md) | 在 `xs` 的每个切片上计算 `fn`，并返回堆叠的输出。 | `map(fn, xs)` |
