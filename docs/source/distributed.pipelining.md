# 流水线并行


> 📝 **注意**
> `torch.distributed.pipelining` 目前处于 alpha 阶段，正在开发中。API 可能会有变更。它从 [PiPPy](https://github.com/pytorch/PiPPy) 项目迁移而来。


## 为什么需要流水线并行？

流水线并行是深度学习的**基本**并行方式之一。它允许将模型的**执行过程**进行划分，使得多个**微批次**可以并发地执行模型代码的不同部分。流水线并行在以下场景中是一种有效的技术：

- 大规模训练
- 带宽受限的集群
- 大模型推理

以上场景有一个共同点：每个设备上的计算无法掩盖传统并行方式（例如 FSDP 的权重全收集）的通信开销。

## 什么是 `torch.distributed.pipelining`？

尽管流水线并行在扩展性方面前景广阔，但实现起来通常很困难，因为它除了需要划分模型权重外，还需要**划分模型的执行过程**。执行过程的划分通常需要对模型代码进行侵入式修改。复杂性的另一个方面来自于**分布式环境中微批次的调度**，同时需要考虑**数据流依赖关系**。

`pipelining` 包提供了一个工具包，能够**自动**完成上述工作，从而可以轻松地在**通用**模型上实现流水线并行。

它由两部分组成：一个**划分前端**和一个**分布式运行时**。划分前端接收你的模型代码（无需修改），将其拆分为“模型分区”，并捕获数据流关系。分布式运行时在不同的设备上并行执行流水线阶段，处理诸如微批次划分、调度、通信和梯度传播等事务。

总的来说，`pipelining` 包提供以下功能：

- 基于简单规范对模型代码进行划分。
- 对流水线调度方案的丰富支持，包括 GPipe、1F1B、交错 1F1B 和循环 BFS，并提供编写自定义调度方案的基础设施。
- 对跨主机流水线并行的首要支持，因为这是流水线并行通常使用的场景（在较慢的互连上）。
- 与其他 PyTorch 并行技术（如数据并行（DDP、FSDP）或张量并行）的可组合性。[TorchTitan](https://github.com/pytorch/torchtitan) 项目展示了在 Llama 模型上的“3D 并行”应用。

## 步骤 1：构建 `PipelineStage`

在使用 `PipelineSchedule` 之前，我们需要创建 `PipelineStage` 对象来包装在该阶段运行的模型部分。`PipelineStage` 负责分配通信缓冲区并创建发送/接收操作以与其对等节点通信。它管理中间缓冲区，例如尚未被消耗的前向输出，并提供一个实用程序来运行阶段模型的反向传播。

`PipelineStage` 需要知道阶段模型的输入和输出形状，以便正确分配通信缓冲区。形状必须是静态的，例如，在运行时，形状不能每一步都发生变化。如果运行时形状与预期形状不匹配，将引发 `PipeliningShapeError` 异常。当与其他并行技术组合或应用混合精度时，必须考虑这些技术，以便 `PipelineStage` 知道运行时阶段模块输出的正确形状（和数据类型）。

用户可以通过传入一个代表应在该阶段运行的模型部分的 `nn.Module` 来直接构造 `PipelineStage` 实例。这可能需要对原始模型代码进行修改。请参阅 *option_1_manual* 中的示例。

或者，划分前端可以使用图划分技术自动将你的模型拆分为一系列 `nn.Module`。此技术要求模型可以通过 `torch.Export` 进行追踪。生成的 `nn.Module` 与其他并行技术的可组合性尚处于实验阶段，可能需要一些变通方法。如果用户无法轻松更改模型代码，使用此前端可能更具吸引力。更多信息请参阅 *option_2_tracer*。

## 步骤 2：使用 `PipelineSchedule` 执行

现在我们可以将 `PipelineStage` 附加到流水线调度方案，并使用输入数据运行该调度方案。以下是一个 GPipe 示例：

```python
from torch.distributed.pipelining import ScheduleGPipe

# 创建调度方案
schedule = ScheduleGPipe(stage, n_microbatches)

# 输入数据（整个批次）
x = torch.randn(batch_size, in_dim, device=device)

# 使用输入 `x` 运行流水线
# `x` 将自动划分为微批次
if rank == 0:
    schedule.step(x)
else:
    output = schedule.step()
```

请注意，上述代码需要在每个工作进程上启动，因此我们使用启动服务来启动多个进程：

```bash
torchrun --nproc_per_node=2 example.py
```

## 划分模型的选项


### 选项 1：手动划分模型

要直接构造 `PipelineStage`，用户需要负责提供一个单独的 `nn.Module` 实例，该实例拥有相关的 `nn.Parameters` 和 `nn.Buffers`，并定义一个 `forward()` 方法来执行该阶段相关的操作。例如，Torchtitan 中定义的 Transformer 类的精简版本展示了一种构建易于划分模型的模式。

```python
class Transformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.tok_embeddings = nn.Embedding(...)

        # 使用 ModuleDict 允许我们在不影响名称的情况下删除层，
        # 确保检查点能够正确保存和加载。
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(...)

        self.output = nn.Linear(...)
python
def forward(self, tokens: torch.Tensor):
    # 在运行时处理层为 'None' 的情况，便于流水线切分
    h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

    for layer in self.layers.values():
        h = layer(h, self.freqs_cis)

    h = self.norm(h) if self.norm else h
    output = self.output(h).float() if self.output else h
    return output
```

以这种方式定义的模型可以轻松按阶段配置，方法是首先初始化整个模型（使用元设备以避免 OOM 错误），删除该阶段不需要的层，然后创建一个包装模型的 PipelineStage。例如：

```python
with torch.device("meta"):
    assert num_stages == 2, "这是一个简单的 2 阶段示例"

    # 我们构建整个模型，然后删除此阶段不需要的部分
    # 在实践中，可以使用一个辅助函数自动在各阶段间划分层。
    model = Transformer()

    if stage_index == 0:
        # 准备第一阶段模型
        del model.layers["1"]
        model.norm = None
        model.output = None

    elif stage_index == 1:
        # 准备第二阶段模型
        model.tok_embeddings = None
        del model.layers["0"]

    from torch.distributed.pipelining import PipelineStage
    stage = PipelineStage(
        model,
        stage_index,
        num_stages,
        device,
    )
```

当与其他数据或模型并行技术结合使用时，如果模型块的输出形状/数据类型会受到影响，则可能还需要 `output_args`。


### 选项 2：自动切分模型

如果你有一个完整的模型，并且不想花时间将其修改为一系列“模型分区”，那么 `pipeline` API 可以帮你。以下是一个简要示例：

```python
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(10, 3)
        self.layers = torch.nn.ModuleList(
            Layer() for _ in range(2)
        )
        self.lm = LMHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.lm(x)
        return x
```

如果我们打印模型，可以看到多个层次结构，这使得手动切分变得困难：

```python
Model(
  (emb): Embedding(10, 3)
  (layers): ModuleList(
    (0-1): 2 x Layer(
      (lin): Linear(in_features=3, out_features=3, bias=True)
    )
  )
  (lm): LMHead(
    (proj): Linear(in_features=3, out_features=3, bias=True)
  )
)
```

让我们看看 `pipeline` API 是如何工作的：

```python
from torch.distributed.pipelining import pipeline, SplitPoint

# 一个示例微批次输入
x = torch.LongTensor([1, 2, 4, 5])

pipe = pipeline(
    module=mod,
    mb_args=(x,),
    split_spec={
        "layers.1": SplitPoint.BEGINNING,
    }
)
```

`pipeline` API 根据给定的 `split_spec` 切分你的模型，其中 `SplitPoint.BEGINNING` 表示在 `forward` 函数中执行某个子模块*之前*添加一个切分点，类似地，`SplitPoint.END` 表示在*之后*添加切分点。

如果我们 `print(pipe)`，可以看到：

```python
GraphModule(
  (submod_0): GraphModule(
    (emb): InterpreterModule()
    (layers): Module(
      (0): InterpreterModule(
        (lin): InterpreterModule()
      )
    )
  )
  (submod_1): GraphModule(
    (layers): Module(
      (1): InterpreterModule(
        (lin): InterpreterModule()
      )
    )
    (lm): InterpreterModule(
      (proj): InterpreterModule()
    )
  )
)

def forward(self, x):
    submod_0 = self.submod_0(x);  x = None
    submod_1 = self.submod_1(submod_0);  submod_0 = None
    return (submod_1,)
```

“模型分区”由子模块（`submod_0`、`submod_1`）表示，每个子模块都使用原始模型的操作、权重和层次结构重建。此外，还重建了一个“根级别”的 `forward` 函数来捕获这些分区之间的数据流。这种数据流稍后将以分布式方式由流水线运行时重放。

`Pipe` 对象提供了一个方法来获取“模型分区”：

```python
stage_mod : nn.Module = pipe.get_stage_module(stage_idx)
```

返回的 `stage_mod` 是一个 `nn.Module`，你可以用它来创建优化器、保存或加载检查点，或应用其他并行技术。

`Pipe` 还允许你在给定 `ProcessGroup` 的设备上创建分布式阶段运行时：

```python
stage = pipe.build_stage(stage_idx, device, group)
```

或者，如果你希望对 `stage_mod` 进行一些修改后再构建阶段运行时，可以使用 `build_stage` API 的函数式版本。例如：

```python
from torch.distributed.pipelining import build_stage
from torch.nn.parallel import DistributedDataParallel

dp_mod = DistributedDataParallel(stage_mod)
info = pipe.info()
stage = build_stage(dp_mod, stage_idx, info, device, group)
```


> 📝 **注意**
> `pipeline` 前端使用一个追踪器（`torch.export`）将你的模型捕获为单个图。如果你的模型无法完全图化，可以使用下面介绍的手动前端。


## Hugging Face 示例

在这个包最初创建的 [PiPPy](https://github.com/pytorch/PiPPy) 仓库中，我们保留了基于未修改的 Hugging Face 模型的示例。请参阅 [examples/huggingface](https://github.com/pytorch/PiPPy/tree/main/examples/huggingface) 目录。

示例包括：

- [GPT2](https://github.com/pytorch/PiPPy/tree/main/examples/huggingface/pippy_gpt2.py)
- [Llama](https://github.com/pytorch/PiPPy/tree/main/examples/llama)

## 技术深度解析

### `pipeline` API 如何切分模型？

首先，`pipeline` API 通过追踪模型将其转换为有向无环图（DAG）。它使用 `torch.export`（一个 PyTorch 2 全图捕获工具）来追踪模型。

然后，它将一个阶段所需的**操作和参数**分组到一个重建的子模块中：`submod_0`、`submod_1`、...

与传统子模块访问方法（如 `Module.children()`）不同，`pipeline` API 不仅切割模型的模块结构，还切割模型的 **forward** 函数。

这是必要的，因为像 `Module.children()` 这样的模型结构仅捕获 `Module.__init__()` 期间的信息，而不捕获任何关于 `Module.forward()` 的信息。换句话说，`Module.children()` 缺少以下对流水线化至关重要的信息：

- 子模块在 `forward` 中的执行顺序
- 子模块之间的激活流
- 子模块之间是否存在任何函数式操作符（例如，`relu` 或 `add` 操作不会被 `Module.children()` 捕获）。

相反，`pipeline` API 确保真正保留了 `forward` 行为。它还捕获分区之间的激活流，帮助分布式运行时无需人工干预即可进行正确的发送/接收调用。

`pipeline` API 的另一个灵活性是，分割点可以位于模型层次结构中的任意级别。在分割后的分区中，与该分区相关的原始模型层次结构将被重建，而无需您付出额外成本。因此，指向子模块或参数的完全限定名称（FQN）仍然有效，并且依赖 FQN 的服务（如 FSDP、TP 或检查点）仍然可以在您的分区模块上运行，几乎无需代码更改。

## 实现自定义调度

您可以通过扩展以下两个类之一来实现自定义流水线调度：

- `PipelineScheduleSingle`
- `PipelineScheduleMulti`

`PipelineScheduleSingle` 适用于每个 rank 仅分配 *一个* 阶段的调度。
`PipelineScheduleMulti` 适用于每个 rank 分配多个阶段的调度。

例如，`ScheduleGPipe` 和 `Schedule1F1B` 是 `PipelineScheduleSingle` 的子类。
而 `ScheduleInterleaved1F1B`、`ScheduleLoopedBFS`、`ScheduleInterleavedZeroBubble` 和 `ScheduleZBVZeroBubble` 是 `PipelineScheduleMulti` 的子类。

## 日志记录

您可以使用 [torch.\_logging](https://pytorch.org/docs/main/logging.html#module-torch._logging) 中的 `TORCH_LOGS` 环境变量开启额外的日志记录：

- `TORCH_LOGS=+pp` 将显示 `logging.DEBUG` 及以上级别的消息。
- `TORCH_LOGS=pp` 将显示 `logging.INFO` 及以上级别的消息。
- `TORCH_LOGS=-pp` 将显示 `logging.WARNING` 及以上级别的消息。

## API 参考


### 模型分割 API

以下一组 API 将您的模型转换为流水线表示。


### 微批次工具


### 流水线阶段


### 流水线调度
