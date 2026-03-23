# torch.hub

PyTorch Hub 是一个预训练模型仓库，旨在促进研究可复现性。

## 发布模型

PyTorch Hub 支持通过添加一个简单的 `hubconf.py` 文件，将预训练模型（模型定义和预训练权重）发布到 GitHub 仓库。

`hubconf.py` 可以有多个入口点。每个入口点被定义为一个 Python 函数（例如：您想要发布的预训练模型）。

```python
  def entrypoint_name(*args, **kwargs):
      # args & kwargs 是可选的，用于接受位置参数/关键字参数的模型。
      ...
```

### 如何实现一个入口点？

以下是一个代码片段，展示了如果我们扩展 `pytorch/vision/hubconf.py` 中的实现，如何为 `resnet18` 模型指定一个入口点。
在大多数情况下，在 `hubconf.py` 中导入正确的函数就足够了。这里我们只是希望使用扩展版本作为示例来说明其工作原理。
您可以在 [pytorch/vision 仓库](https://github.com/pytorch/vision/blob/master/hubconf.py) 中查看完整脚本。

```python
  dependencies = ['torch']
  from torchvision.models.resnet import resnet18 as _resnet18

  # resnet18 是入口点的名称
  def resnet18(pretrained=False, **kwargs):
      """ # 此文档字符串会显示在 hub.help() 中
      Resnet18 模型
      pretrained (bool): kwargs，将预训练权重加载到模型中
      """
      # 调用模型，加载预训练权重
      model = _resnet18(pretrained=pretrained, **kwargs)
      return model
```

- `dependencies` 变量是 **加载** 模型所需的包名称**列表**。请注意，这可能与训练模型所需的依赖项略有不同。
- `args` 和 `kwargs` 会传递给实际的可调用函数。
- 函数的文档字符串用作帮助信息。它解释了模型的功能以及允许的位置/关键字参数。强烈建议在此处添加一些示例。
- 入口点函数可以返回一个模型（nn.module），或者返回辅助工具以使工作流程更顺畅，例如分词器。
- 以下划线为前缀的可调用对象被视为辅助函数，不会出现在 `torch.hub.list()` 中。
- 预训练权重可以存储在 GitHub 仓库本地，或者通过 `torch.hub.load_state_dict_from_url()` 加载。如果小于 2GB，建议将其附加到 [项目发布](https://help.github.com/en/articles/distributing-large-binaries) 并使用发布中的 URL。
  在上面的示例中，`torchvision.models.resnet.resnet18` 处理了 `pretrained` 参数，或者您可以将以下逻辑放在入口点定义中。

```python
  if pretrained:
      # 对于保存在本地 GitHub 仓库中的检查点，例如 <RELATIVE_PATH_TO_CHECKPOINT>=weights/save.pth
      dirname = os.path.dirname(__file__)
      checkpoint = os.path.join(dirname, <RELATIVE_PATH_TO_CHECKPOINT>)
      state_dict = torch.load(checkpoint)
      model.load_state_dict(state_dict)

      # 对于保存在其他地方的检查点
      checkpoint = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
      model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))
```

### 重要提示

- 发布的模型应至少位于一个分支/标签中。不能是随机的提交。

## 从 Hub 加载模型

PyTorch Hub 提供了便捷的 API，可以通过 `torch.hub.list()` 探索 hub 中所有可用的模型，通过 `torch.hub.help()` 显示文档字符串和示例，并使用 `torch.hub.load()` 加载预训练模型。


### 运行加载的模型：

请注意，`torch.hub.load()` 中的 `*args` 和 `**kwargs` 用于 **实例化** 模型。加载模型后，如何找出可以用该模型做什么？
建议的工作流程是：

- `dir(model)` 查看模型的所有可用方法。
- `help(model.foo)` 检查运行 `model.foo` 需要哪些参数。

为了帮助用户在不反复查阅文档的情况下进行探索，我们强烈建议仓库所有者使函数帮助信息清晰简洁。包含一个最小工作示例也很有帮助。

### 我下载的模型保存在哪里？

按以下顺序使用位置：

- 调用 `hub.set_dir(<PATH_TO_HUB_DIR>)`
- `$TORCH_HOME/hub`，如果设置了环境变量 `TORCH_HOME`。
- `$XDG_CACHE_HOME/torch/hub`，如果设置了环境变量 `XDG_CACHE_HOME`。
- `~/.cache/torch/hub`


### 缓存逻辑

默认情况下，我们加载文件后不会清理它们。如果文件已存在于 `~torch.hub.get_dir()` 返回的目录中，Hub 默认会使用缓存。

用户可以通过调用 `hub.load(..., force_reload=True)` 强制重新加载。这将删除现有的 GitHub 文件夹和下载的权重，并重新初始化一次新的下载。当更新发布到同一分支时，这很有用，用户可以跟上最新的发布。

### 已知限制：

Torch hub 的工作原理是像安装包一样导入它。Python 导入会引入一些副作用。例如，您可以在 Python 缓存 `sys.modules` 和 `sys.path_importer_cache` 中看到新条目，这是正常的 Python 行为。
这也意味着，如果不同仓库具有相同的子包名称（通常是 `model` 子包），则在从不同仓库导入不同模型时，可能会出现导入错误。解决此类导入错误的一种方法是从 `sys.modules` 字典中删除有问题的子包；更多详细信息可以在 [此 GitHub 问题](https://github.com/pytorch/hub/issues/243#issuecomment-942403391) 中找到。

这里值得提及一个已知的限制：用户**无法**在**同一个 Python 进程**中加载同一仓库的两个不同分支。这就像在 Python 中安装两个同名的包一样，这是不可取的。如果你真的尝试这样做，缓存机制可能会介入并带来意想不到的结果。当然，在独立的进程中分别加载它们是完全可行的。