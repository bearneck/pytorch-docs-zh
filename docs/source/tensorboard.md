# torch.utils.tensorboard

 automodule
torch.utils.tensorboard


在深入之前，可以在 <https://www.tensorflow.org/tensorboard/> 找到关于 TensorBoard 的更多详细信息。

安装 TensorBoard 后，这些实用工具允许你将 PyTorch 模型和指标记录到一个目录中，以便在 TensorBoard UI 中进行可视化。对于 PyTorch 模型和张量以及 Caffe2 网络和 blob，都支持标量、图像、直方图、图和嵌入可视化。

SummaryWriter 类是你记录数据以供 TensorBoard 使用和可视化的主要入口。例如：

``` python
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Writer 默认会输出到 ./runs/ 目录
writer = SummaryWriter()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
model = torchvision.models.resnet50(False)
# 让 ResNet 模型接收灰度图而非 RGB 图
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
images, labels = next(iter(trainloader))

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()
```

然后可以使用 TensorBoard 进行可视化，TensorBoard 应该可以通过以下方式安装和运行：:

> pip install tensorboard tensorboard \--logdir=runs

一次实验可以记录大量信息。为了避免 UI 杂乱并获得更好的结果聚类，我们可以通过分层命名来对绘图进行分组。例如，\"Loss/train\" 和 \"Loss/test\" 将被分组在一起，而 \"Accuracy/train\" 和 \"Accuracy/test\" 将在 TensorBoard 界面中被单独分组。

``` python
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
```

预期结果：

![image](_static/img/tensorboard/hier_tags.png)

## \|

 currentmodule
torch.utils.tensorboard.writer


 autoclass
SummaryWriter

 automethod
\_\_init\_\_


 automethod
add_scalar


 automethod
add_scalars


 automethod
add_histogram


 automethod
add_image


 automethod
add_images


 automethod
add_figure


 automethod
add_video


 automethod
add_audio


 automethod
add_text


 automethod
add_graph


 automethod
add_embedding


 automethod
add_pr_curve


 automethod
add_custom_scalars


 automethod
add_mesh


 automethod
add_hparams


 automethod
flush


 automethod
close


