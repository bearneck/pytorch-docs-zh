# 在何处应用 torch.compile？

我们建议将 `torch.compile` 应用于不会导致过度问题的最高层级函数。
通常，这包括：
- 包含优化器但不包含循环的 `train` 或 `eval` 步骤，
- 顶层的 `nn.Module`
- 或某些子 `nn.Module`。

`torch.compile` 特别不擅长处理像 DDP 或 FSDP 这样的分布式包装器模块，
因此请考虑将 `torch.compile` 应用于传递给包装器的内部模块。

```python
# 推理
model = ...
model.compile()

for _ in range(N_ITERS):
    inp = ...
    out = model(inp)
```

```python
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

```python
# DistributedDataParallel
model = ...
model.compile()
model_ddp = DistributedDataParallel(model, ...)

for _ in range(N_ITERS):
    inp = ...
    out = model_ddp(inp)
```

<!-- TODO 为特定模型领域添加示例，compile(model) 与 model.compile()-->

## `compile(model)` 与 `model.compile()`

由于 `torch.compile` 与 `nn.Module` 实例交互的细微差别，
如果您希望将 `nn.Module` 实例作为顶层函数进行编译，我们建议使用其 `.compile()` 方法。
嵌套模块的调用将被正确追踪 -
在这种情况下无需调用 `.compile()`。

```python
# 不要这样做
model = MyModel()
model = torch.compile(model)
model(inp)

# 这样做
model = MyModel()
model.compile()
model(inp)

# 这也是可以接受的
@torch.compile
def fn(model, inp):
    return model(inp)
model = MyModel()
fn(model, inp)
```