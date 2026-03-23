# 在 Intel GPU 上开始使用

## 硬件先决条件

### Intel 数据中心 GPU

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  设备                                                      Red Hat\* Enterprise Linux\* 9.2   SUSE Linux Enterprise Server\* 15 SP5   Ubuntu\* Server 22.04 (\>= 5.15 LTS 内核)
  --------------------------------------------------------- ---------------------------------- --------------------------------------- -------------------------------------------
  Intel® Data Center GPU Max Series (代号: Ponte Vecchio)   是                                 是                                      是

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Intel 客户端 GPU

+------------------------------------------------------------------------------------------------------------------------------------------+
| 支持的操作系统 \| 已验证的硬件 \|                                                                                                        |
+=====================================+====================================================================================================+
| Windows 11 & Ubuntu 24.04/25.04     | | Intel® Arc A-Series Graphics (代号: Alchemist) \|                                                |
|                                     | | Intel® Arc B-Series Graphics (代号: Battlemage) \|                                               |
|                                     | | Intel® Core™ Ultra Processors with Intel® Arc™ Graphics (代号: Meteor Lake-H) \|                 |
|                                     | | Intel® Core™ Ultra Processors (Series 2) with Intel® Arc™ Graphics (代号: Arrow Lake-H) \|       |
|                                     | | Intel® Core™ Ultra Mobile Processors (Series 2) with Intel® Arc™ Graphics (代号: Lunar Lake) \|  |
+-------------------------------------+----------------------------------------------------------------------------------------------------+
| Windows 11 & Ubuntu 25.10           | | Intel® Core™ Ultra Mobile Processors (Series 3) with Intel® Arc™ Graphics (代号: Panther Lake)\| |
+-------------------------------------+----------------------------------------------------------------------------------------------------+

从 PyTorch\* 2.5 版本开始，Intel GPUs 支持（原型）已就绪，适用于 Linux 和 Windows 上的 Intel® 客户端 GPU 和 Intel® Data Center GPU Max Series。这使 Intel GPU 和 SYCL\* 软件栈进入了官方的 PyTorch 生态，提供一致的用户体验，以拥抱更多 AI 应用场景。

## 软件先决条件

要在 Intel GPU 上使用 PyTorch，您需要先安装 Intel GPU 驱动程序。有关安装指南，请访问 [Intel GPUs 驱动程序安装](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html)。

如果您通过二进制文件安装，请跳过 Intel® Deep Learning Essentials 安装部分。对于从源代码构建，请参考 [PyTorch Installation Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html) 以获取 Intel GPU 驱动程序和 Intel® Deep Learning Essentials 的安装说明。

## 安装

### 二进制文件

现在我们已经安装了 [Intel GPU Driver](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html)，使用以下命令安装 `pytorch`、`torchvision`、`torchaudio`。

#### 稳定版本

安装适用于 Intel GPU (XPU) 的最新稳定版本 wheel 包：

``` bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

#### 夜间构建版本

安装最新的预览版/夜间构建 wheel 包：

``` bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
```

#### 历史版本

**v2.10.0**

``` bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/xpu
```

**v2.9.1**

``` bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/xpu
```

 note
 title
Note


对于更旧的 wheel 包，请参考 [历史版本](https://pytorch.org/get-started/previous-versions/) 页面，并确保使用 `xpu` 索引 URL。


### 从源代码构建

现在我们已经安装了 [Intel GPU Driver and Intel® Deep Learning Essentials](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html)。请按照指南从源代码构建 `pytorch`、`torchvision`、`torchaudio`。

从源代码构建 `torch` 请参考 [PyTorch Installation Build from source](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source)。

从源代码构建 `torchvision` 请参考 [Torchvision Installation Build from source](https://github.com/pytorch/vision/blob/main/CONTRIBUTING.md#development-installation)。

从源代码构建 `torchaudio` 请参考 [Torchaudio Installation Build from source](https://github.com/pytorch/audio/blob/main/CONTRIBUTING.md#building-torchaudio-from-source)。

## 检查 Intel GPU 可用性

要检查您的 Intel GPU 是否可用，通常使用以下代码：

``` python
import torch
print(torch.xpu.is_available())  # torch.xpu 是用于 Intel GPU 支持的 API
```

如果输出是 `False`，请仔细检查 Intel GPU 的驱动程序安装。

## 最小代码改动

如果您正在从 `cuda` 迁移代码，您需要将引用从 `cuda` 更改为 `xpu`。例如：

``` python
# CUDA 代码
tensor = torch.tensor([1.0, 2.0]).to("cuda")

# Intel GPU 代码
tensor = torch.tensor([1.0, 2.0]).to("xpu")
```

以下几点概述了 PyTorch 对 Intel GPU 的支持和限制：

1.  支持训练和推理工作流。
2.  支持即时执行模式和 `torch.compile`。从 PyTorch\* 2.7 开始，Windows 平台上的 Intel GPU 也支持 `torch.compile` 功能，请参阅 [如何在 Windows CPU/XPU 上使用 torch.compile](https://pytorch.org/tutorials/unstable/inductor_windows.html)。
3.  支持 FP32、BF16、FP16 等数据类型以及自动混合精度（AMP）。

## 示例

本节包含推理和训练工作流的使用示例。

### 推理示例

以下是几个推理工作流示例。

#### 使用 FP32 进行推理

``` python
import torch
import torchvision.models as models

model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()
data = torch.rand(1, 3, 224, 224)

model = model.to("xpu")
data = data.to("xpu")

with torch.no_grad():
    model(data)

print("Execution finished")
```

#### 使用 AMP 进行推理

``` python
import torch
import torchvision.models as models

model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()
data = torch.rand(1, 3, 224, 224)

model = model.to("xpu")
data = data.to("xpu")

with torch.no_grad():
    d = torch.rand(1, 3, 224, 224)
    d = d.to("xpu")
    # 设置 dtype=torch.bfloat16 以使用 BF16
    with torch.autocast(device_type="xpu", dtype=torch.float16, enabled=True):
        model(data)

print("Execution finished")
```

#### 使用 `torch.compile` 进行推理

``` python
import torch
import torchvision.models as models
import time

model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()
data = torch.rand(1, 3, 224, 224)
ITERS = 10

model = model.to("xpu")
data = data.to("xpu")

for i in range(ITERS):
    start = time.time()
    with torch.no_grad():
        model(data)
        torch.xpu.synchronize()
    end = time.time()
    print(f"Inference time before torch.compile for iteration {i}: {(end-start)*1000} ms")

model = torch.compile(model)
for i in range(ITERS):
    start = time.time()
    with torch.no_grad():
        model(data)
        torch.xpu.synchronize()
    end = time.time()
    print(f"Inference time after torch.compile for iteration {i}: {(end-start)*1000} ms")

print("Execution finished")
```

### 训练示例

以下是几个训练工作流示例。

#### 使用 FP32 进行训练

``` python
import torch
import torchvision

LR = 0.001
DOWNLOAD = True
DATA = "datasets/cifar10/"

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
train_dataset = torchvision.datasets.CIFAR10(
    root=DATA,
    train=True,
    transform=transform,
    download=DOWNLOAD,
)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128)
train_len = len(train_loader)

model = torchvision.models.resnet50()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
model.train()
model = model.to("xpu")
criterion = criterion.to("xpu")

print(f"Initiating training")
for batch_idx, (data, target) in enumerate(train_loader):
    data = data.to("xpu")
    target = target.to("xpu")
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if (batch_idx + 1) % 10 == 0:
        iteration_loss = loss.item()
        print(f"Iteration [{batch_idx+1}/{train_len}], Loss: {iteration_loss:.4f}")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "checkpoint.pth",
)

print("Execution finished")
```

#### 使用 AMP 进行训练

 note
 title
Note


使用 `GradScaler` 进行训练需要硬件支持 `FP64`。Intel® Arc™ A 系列显卡原生不支持 `FP64`。如果您在 Intel® Arc™ A 系列显卡上运行工作负载，请禁用 `GradScaler`。


``` python
import torch
import torchvision

LR = 0.001
DOWNLOAD = True
DATA = "datasets/cifar10/"

use_amp=True

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
train_dataset = torchvision.datasets.CIFAR10(
    root=DATA,
    train=True,
    transform=transform,
    download=DOWNLOAD,
)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128)
train_len = len(train_loader)

model = torchvision.models.resnet50()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
scaler = torch.amp.GradScaler(device="xpu", enabled=use_amp)

model.train()
model = model.to("xpu")
criterion = criterion.to("xpu")

print(f"开始训练")
for batch_idx, (data, target) in enumerate(train_loader):
    data = data.to("xpu")
    target = target.to("xpu")
    # 对于 BF16，设置 dtype=torch.bfloat16
    with torch.autocast(device_type="xpu", dtype=torch.float16, enabled=use_amp):
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    if (batch_idx + 1) % 10 == 0:
        iteration_loss = loss.item()
        print(f"迭代 [{batch_idx+1}/{train_len}], 损失: {iteration_loss:.4f}")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "checkpoint.pth",
)

print("执行完成")
```

#### 使用 `torch.compile` 进行训练

``` python
import torch
import torchvision

LR = 0.001
DOWNLOAD = True
DATA = "datasets/cifar10/"

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
train_dataset = torchvision.datasets.CIFAR10(
    root=DATA,
    train=True,
    transform=transform,
    download=DOWNLOAD,
)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128)
train_len = len(train_loader)

model = torchvision.models.resnet50()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
model.train()
model = model.to("xpu")
criterion = criterion.to("xpu")
model = torch.compile(model)

print(f"使用 torch compile 开始训练")
for batch_idx, (data, target) in enumerate(train_loader):
    data = data.to("xpu")
    target = target.to("xpu")
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if (batch_idx + 1) % 10 == 0:
        iteration_loss = loss.item()
        print(f"迭代 [{batch_idx+1}/{train_len}], 损失: {iteration_loss:.4f}")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "checkpoint.pth",
)

print("执行完成")
```
