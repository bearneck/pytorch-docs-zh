# 入门指南

在阅读本节之前，请确保已阅读 *torch.compiler_overview*

让我们从一个简单的 `torch.compile` 示例开始，该示例演示了如何将 `torch.compile` 用于推理。这个示例展示了 `torch.cos()` 和 `torch.sin()` 功能，它们是逐点运算符的示例，因为它们对向量进行逐元素操作。此示例可能不会显示出显著的性能提升，但应有助于您直观地理解如何在您自己的程序中使用 `torch.compile`。


> 📝 **注意**
> 要运行此脚本，您的机器上至少需要有一个 GPU。
> 如果您没有 GPU，可以移除下面代码片段中的 `.to(device="cuda:0")` 代码，它将在 CPU 上运行。您也可以将设备设置为 `xpu:0` 以在 Intel® GPU 上运行。


```python
import torch
def fn(x):
   a = torch.cos(x)
   b = torch.sin(a)
   return b
new_fn = torch.compile(fn, backend="inductor")
input_tensor = torch.randn(10000).to(device="cuda:0")
a = new_fn(input_tensor)
```

您可能想使用的一个更著名的逐点运算符是类似 `torch.relu()` 的运算符。在即时执行模式下，逐点操作不是最优的，因为每个操作都需要从内存中读取张量，进行一些更改，然后将这些更改写回。Inductor 执行的最重要的优化是融合。在上面的示例中，我们可以将 2 次读取（`x`, `a`）和 2 次写入（`a`, `b`）转换为 1 次读取（`x`）和 1 次写入（`b`），这对于较新的 GPU 尤其关键，因为瓶颈是内存带宽（向 GPU 发送数据的速度）而不是计算能力（GPU 处理浮点运算的速度）。

Inductor 提供的另一个主要优化是自动支持 CUDA 图。
CUDA 图有助于消除从 Python 程序启动单个内核的开销，这对于较新的 GPU 尤其相关。

TorchDynamo 支持许多不同的后端，但 TorchInductor 专门通过生成 [Triton](https://github.com/openai/triton) 内核来工作。让我们将上面的示例保存到一个名为 `example.py` 的文件中。我们可以通过运行 `TORCH_COMPILE_DEBUG=1 python example.py` 来检查生成的 Triton 内核代码。当脚本执行时，您应该会看到 `DEBUG` 消息打印到终端。在日志接近结尾处，您应该会看到一个包含 `torchinductor_<您的用户名>` 的文件夹路径。在该文件夹中，您可以找到包含生成的内核代码的 `output_code.py` 文件，类似于以下内容：

```python
@pointwise(size_hints=[16384], filename=__file__, triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
   xnumel = 10000
   xoffset = tl.program_id(0) * XBLOCK
   xindex = xoffset + tl.arange(0, XBLOCK)[:]
   xmask = xindex < xnumel
   x0 = xindex
   tmp0 = tl.load(in_ptr0 + (x0), xmask, other=0.0)
   tmp1 = tl.cos(tmp0)
   tmp2 = tl.sin(tmp1)
   tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
```


> 📝 **注意**
> 上面的代码片段是一个示例。根据您的硬件，您可能会看到生成不同的代码。


您可以验证 `cos` 和 `sin` 确实发生了融合，因为 `cos` 和 `sin` 操作发生在单个 Triton 内核内，并且临时变量保存在访问速度非常快的寄存器中。

有关 Triton 性能的更多信息，请阅读[此处](https://openai.com/blog/triton/)。由于代码是用 Python 编写的，即使您没有编写过那么多 CUDA 内核，也相当容易理解。

接下来，让我们尝试一个真实的模型，例如来自 PyTorch hub 的 resnet50。

```python
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
opt_model = torch.compile(model, backend="inductor")
opt_model(torch.randn(1,3,64,64))
```

这还不是唯一可用的后端，您可以在 REPL 中运行 `torch.compiler.list_backends()` 来查看所有可用的后端。接下来可以尝试 `cudagraphs` 作为灵感。

## 使用预训练模型

PyTorch 用户经常利用来自 [transformers](https://github.com/huggingface/transformers) 或 [TIMM](https://github.com/rwightman/pytorch-image-models) 的预训练模型，TorchDynamo 和 TorchInductor 的设计目标之一就是能够开箱即用地与人们想要编写的任何模型一起工作。

让我们直接从 HuggingFace hub 下载一个预训练模型并优化它：

```python
import torch
from transformers import BertTokenizer, BertModel
# 从这里复制粘贴 https://huggingface.co/bert-base-uncased
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").to(device="cuda:0")
model = torch.compile(model, backend="inductor") # 这是我们更改的唯一一行代码
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt').to(device="cuda:0")
output = model(**encoded_input)
```

如果您从模型和 `encoded_input` 中移除 `to(device="cuda:0")`，那么 Triton 将生成针对在您的 CPU 上运行而优化的 C++ 内核。您可以检查 BERT 的 Triton 或 C++ 内核。它们比我们上面尝试的三角函数示例更复杂，但您可以类似地浏览它，看看是否理解 PyTorch 的工作原理。

同样，让我们尝试一个 TIMM 示例：

```python
import timm
import torch
model = timm.create_model('resnext101_32x8d', pretrained=True, num_classes=2)
opt_model = torch.compile(model, backend="inductor")
opt_model(torch.randn(64,3,7,7))
```

## 后续步骤

在本节中，我们回顾了几个推理示例，并对 torch.compile 的工作原理有了基本的了解。以下是您接下来可以查看的内容：

- [torch.compile 训练教程](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- *torch.compiler_api*
- *torchdynamo_fine_grain_tracing*
