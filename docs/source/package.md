```{eval-rst}
.. automodule:: torch.package
.. py:module:: torch.package.analyze

.. currentmodule:: torch.package
```

# torch.package
`torch.package` 增加了对创建包含工件和任意 PyTorch 代码的包的支持。这些包可以被保存、共享，用于在以后或不同机器上加载和执行模型，甚至可以使用 `torch::deploy` 部署到生产环境。

本文档包含教程、操作指南、解释说明和 API 参考，将帮助您了解更多关于 `torch.package` 的信息以及如何使用它。

```{warning}
此模块依赖于不安全的 `pickle` 模块。仅解包您信任的数据。

有可能构造恶意的 pickle 数据，**在反序列化期间执行任意代码**。
切勿解包可能来自不可信来源或可能被篡改的数据。

更多信息，请查阅 `pickle` 模块的[文档](https://docs.python.org/3/library/pickle.html)。
```

```{contents}
:local:
:depth: 2
```

## 教程
### 打包您的第一个模型
一个指导您打包和解包简单模型的教程可在 [Colab](https://colab.research.google.com/drive/1lFZkLyViGfXxB-m3jqlyTQuYToo3XLo-) 上找到。完成此练习后，您将熟悉创建和使用 Torch 包的基本 API。

## 如何...

### 查看包内有什么？

#### 将包视为 ZIP 归档文件

`torch.package` 的容器格式是 ZIP，因此任何适用于标准 ZIP 文件的工具都应能用于探索其内容。与 ZIP 文件交互的一些常见方式：

* `unzip my_package.pt` 会将 `torch.package` 归档文件解压到磁盘，您可以在其中自由检查其内容。

```
$ unzip my_package.pt && tree my_package
my_package
├── .data
│   ├── 94304870911616.storage
│   ├── 94304900784016.storage
│   ├── extern_modules
│   └── version
├── models
│   └── model_1.pkl
└── torchvision
    └── models
        ├── resnet.py
        └── utils.py
~ cd my_package && cat torchvision/models/resnet.py
...
```

* Python 的 `zipfile` 模块提供了读取和写入 ZIP 归档文件内容的标准方式。

```python
from zipfile import ZipFile
with ZipFile("my_package.pt") as myzip:
    file_bytes = myzip.read("torchvision/models/resnet.py")
    # 以某种方式编辑 file_bytes
    myzip.writestr("torchvision/models/resnet.py", new_file_bytes)
```

* vim 能够原生读取 ZIP 归档文件。您甚至可以编辑文件并使用 :`write` 将它们写回归档文件！

```vim
# 将此添加到您的 .vimrc 以将 `*.pt` 文件视为 zip 文件
au BufReadCmd *.pt call zip#Browse(expand("<amatch>"))

~ vi my_package.pt
```

#### 使用 `file_structure()` API
{class}`PackageImporter` 提供了一个 `file_structure()` 方法，该方法将返回一个可打印且可查询的 {class}`Directory` 对象。{class}`Directory` 对象是一个简单的目录结构，您可以用它来探索 `torch.package` 的当前内容。

{class}`Directory` 对象本身可以直接打印，并将打印出文件树表示。要过滤返回的内容，请使用 glob 风格的 `include` 和 `exclude` 过滤参数。

```python
with PackageExporter('my_package.pt') as pe:
    pe.save_pickle('models', 'model_1.pkl', mod)

importer = PackageImporter('my_package.pt')
# 可以使用 include/exclude 参数限制打印的项目
print(importer.file_structure(include=["**/utils.py", "**/*.pkl"], exclude="**/*.storage"))
print(importer.file_structure()) # 将打印出所有文件
```

输出：

```
# 使用 glob 模式过滤：
#    include=["**/utils.py", "**/*.pkl"], exclude="**/*.storage"
─── my_package.pt
    ├── models
    │   └── model_1.pkl
    └── torchvision
        └── models
            └── utils.py

# 所有文件
─── my_package.pt
    ├── .data
    │   ├── 94304870911616.storage
    │   ├── 94304900784016.storage
    │   ├── extern_modules
    │   └── version
    ├── models
    │   └── model_1.pkl
    └── torchvision
        └── models
            ├── resnet.py
            └── utils.py
```

您还可以使用 `has_file()` 方法查询 {class}`Directory` 对象。

```python
importer_file_structure = importer.file_structure()
found: bool = importer_file_structure.has_file("package_a/subpackage.py")
```

### 查看为什么给定的模块被作为依赖项包含？

假设有一个给定的模块 `foo`，您想知道为什么您的 {class}`PackageExporter` 将 `foo` 作为依赖项引入。

{meth}`PackageExporter.get_rdeps` 将返回所有直接依赖于 `foo` 的模块。

如果您想查看给定模块 `src` 如何依赖于 `foo`，{meth}`PackageExporter.all_paths` 方法将返回一个 DOT 格式的图，显示 `src` 和 `foo` 之间的所有依赖路径。

如果您只想查看 :class:`PackageExporter` 的整个依赖图，可以使用 {meth}`PackageExporter.dependency_graph_string`。

### 将任意资源包含在我的包中并在以后访问它们？
{class}`PackageExporter` 公开了三个方法：`save_pickle`、`save_text` 和 `save_binary`，允许您将 Python 对象、文本和二进制数据保存到包中。

```python
with torch.PackageExporter("package.pt") as exporter:
    # 将对象序列化并保存到归档中的 `my_resources/tensor.pkl`。
    exporter.save_pickle("my_resources", "tensor.pkl", torch.randn(4))
    exporter.save_text("config_stuff", "words.txt", "a sample string")
    exporter.save_binary("raw_data", "binary", my_bytes)

```
{class}`PackageImporter` 公开了名为 `load_pickle`、`load_text` 和 `load_binary` 的互补方法，允许您从包中加载 Python 对象、文本和二进制数据。

```python
importer = torch.PackageImporter("package.pt")
my_tensor = importer.load_pickle("my_resources", "tensor.pkl")
text = importer.load_text("config_stuff", "words.txt")
binary = importer.load_binary("raw_data", "binary")
```

### 如何自定义类的打包方式？
`torch.package` 允许自定义类的打包方式。此行为通过在类上定义 `__reduce_package__` 方法并定义相应的解包函数来实现。这类似于为 Python 的正常 pickle 过程定义 `__reduce__`。

步骤：

1. 在目标类上定义方法 `__reduce_package__(self, exporter: PackageExporter)`。此方法应完成将类实例保存到包内的工作，并应返回一个元组，包含相应的解包函数以及调用该解包函数所需的参数。当 `PackageExporter` 遇到目标类的实例时，会调用此方法。
2. 为该类定义一个解包函数。此解包函数应完成重建并返回该类实例的工作。函数签名的第一个参数应为 `PackageImporter` 实例，其余参数由用户定义。

```python
# foo.py [自定义类 Foo 打包方式的示例]
from torch.package import PackageExporter, PackageImporter
import time


class Foo:
    def __init__(self, my_string: str):
        super().__init__()
        self.my_string = my_string
        self.time_imported = 0
        self.time_exported = 0

    def __reduce_package__(self, exporter: PackageExporter):
        """
        当保存此对象的实例时，由 ``torch.package.PackageExporter`` 的 Pickler 的 ``persistent_id`` 调用。
        此方法应完成将此对象保存到 ``torch.package`` 归档文件内的工作。

        返回一个包含解包函数及其参数的元组，以便从 ``torch.package.PackageImporter`` 的 Pickler 的 ``persistent_load`` 函数加载该对象。
        """

        # 使用此模式确保与正常依赖项没有命名冲突，
        # 在此模块名下保存的任何内容都不应与包中的其他项冲突
        generated_module_name = f"foo-generated._{exporter.get_unique_id()}"
        exporter.save_text(
            generated_module_name,
            "foo.txt",
            self.my_string + ", with exporter modification!",
        )
        time_exported = time.clock_gettime(1)

        # 返回解包函数及其调用所需的参数
        return (unpackage_foo, (generated_module_name, time_exported,))


def unpackage_foo(
    importer: PackageImporter, generated_module_name: str, time_exported: float
) -> Foo:
    """
    当解封一个 Foo 对象时，由 ``torch.package.PackageImporter`` 的 Pickler 的 ``persistent_load`` 函数调用。
    执行从 ``torch.package`` 归档文件加载并返回 Foo 实例的工作。
    """
    time_imported = time.clock_gettime(1)
    foo = Foo(importer.load_text(generated_module_name, "foo.txt"))
    foo.time_imported = time_imported
    foo.time_exported = time_exported
    return foo

```


```python
# 保存类 Foo 实例的示例

import torch
from torch.package import PackageImporter, PackageExporter
import foo

foo_1 = foo.Foo("foo_1 initial string")
foo_2 = foo.Foo("foo_2 initial string")
with PackageExporter('foo_package.pt') as pe:
    # 正常保存，无需额外工作
    pe.save_pickle('foo_collection', 'foo1.pkl', foo_1)
    pe.save_pickle('foo_collection', 'foo2.pkl', foo_2)

pi = PackageImporter('foo_package.pt')
print(pi.file_structure())
imported_foo = pi.load_pickle('foo_collection', 'foo1.pkl')
print(f"foo_1 string: '{imported_foo.my_string}'")
print(f"foo_1 export time: {imported_foo.time_exported}")
print(f"foo_1 import time: {imported_foo.time_imported}")
```

```
# 运行上述脚本的输出
─── foo_package
    ├── foo-generated
    │   ├── _0
    │   │   └── foo.txt
    │   └── _1
    │       └── foo.txt
    ├── foo_collection
    │   ├── foo1.pkl
    │   └── foo2.pkl
    └── foo.py

foo_1 string: 'foo_1 initial string, with reduction modification!'
foo_1 export time: 9857706.650140837
foo_1 import time: 9857706.652698385
```

### 如何在源代码中测试是否在包内执行？

{class}`PackageImporter` 会为其初始化的每个模块添加属性 `__torch_package__`。您的代码可以检查此属性是否存在，以确定是否在打包的上下文中执行。

```python
# 在 foo/bar.py 中：

if "__torch_package__" in dir():  # 如果代码是从包中加载的，则为真
    def is_in_package():
        return True

    UserException = Exception
else:
    def is_in_package():
        return False

    UserException = UnpackageableException
```

现在，代码的行为将根据它是通过您的 Python 环境正常导入还是从 `torch.package` 导入而有所不同。

```python
from foo.bar import is_in_package

print(is_in_package())  # False

loaded_module = PackageImporter(my_package).import_module("foo.bar")
loaded_module.is_in_package()  # True
```

**警告**：通常，让代码根据是否打包而表现不同是一种不好的做法。这可能导致难以调试的问题，这些问题对您导入代码的方式很敏感。如果您的包打算被大量使用，请考虑重构您的代码，使其无论以何种方式加载都表现一致。


### 如何将代码修补到包中？
{class}`PackageExporter` 提供了 `save_source_string()` 方法，允许将任意 Python 源代码保存到您选择的模块。
```python
with PackageExporter(f) as exporter:
    # 保存当前 Python 环境中可用的 my_module.foo。
    exporter.save_module("my_module.foo")

    # 这将提供的字符串保存到包归档文件中的 my_module/foo.py。
    # 它将覆盖先前保存的 my_module.foo。
    exporter.save_source_string("my_module.foo", textwrap.dedent(
        """\
        def my_function():
            print('hello world')
        """
    ))
```

# 如果你想将 my_module.bar 视为一个包
# (例如保存到 `my_module/bar/__init__.py` 而不是 `my_module/bar.py`)
# 请传递 is_package=True,
exporter.save_source_string("my_module.bar",
                            "def foo(): print('hello')\n",
                            is_package=True)

importer = PackageImporter(f)
importer.import_module("my_module.foo").my_function()  # 打印 'hello world'
```

### 从打包代码中访问包内容？
{class}`PackageImporter` 实现了
[`importlib.resources`](https://docs.python.org/3/library/importlib.html#module-importlib.resources)
API，用于从包内部访问资源。

```python
with PackageExporter(f) as exporter:
    # 将文本保存到存档中的 my_resource/a.txt
    exporter.save_text("my_resource", "a.txt", "hello world!")
    # 将张量保存到 my_pickle/obj.pkl
    exporter.save_pickle("my_pickle", "obj.pkl", torch.ones(2, 2))

    # 模块内容见下文
    exporter.save_module("foo")
    exporter.save_module("bar")
```

`importlib.resources` API 允许从打包代码内部访问资源。

```python
# foo.py:
import importlib.resources
import my_resource

# 返回 "hello world!"
def get_my_resource():
    return importlib.resources.read_text(my_resource, "a.txt")
```

使用 `importlib.resources` 是从打包代码内部访问包内容的推荐方式，因为它符合
Python 标准。然而，也可以从打包代码内部访问父 :class:`PackageImporter` 实例本身。

```python
# bar.py:
import torch_package_importer # 这是导入此模块的 PackageImporter。

# 打印 "hello world!"，等同于 importlib.resources.read_text
def get_my_resource():
    return torch_package_importer.load_text("my_resource", "a.txt")

# 你也可以做一些 importlib.resources API 不支持的事情，比如
# 从包中加载一个已序列化的对象。
def get_my_pickle():
    return torch_package_importer.load_pickle("my_pickle", "obj.pkl")
```

### 区分打包代码和非打包代码？
要判断一个对象的代码是否来自 `torch.package`，请使用 `torch.package.is_from_package()` 函数。
注意：如果一个对象来自一个包，但其定义来自标记为 `extern` 的模块或来自 `stdlib`，
此检查将返回 `False`。

```python
importer = PackageImporter(f)
mod = importer.import_module('foo')
obj = importer.load_pickle('model', 'model.pkl')
txt = importer.load_text('text', 'my_test.txt')

assert is_from_package(mod)
assert is_from_package(obj)
assert not is_from_package(txt) # str 来自 stdlib，因此这将返回 False
```

### 重新导出一个已导入的对象？
要重新导出一个先前由 {class}`PackageImporter` 导入的对象，你必须让新的 {class}`PackageExporter`
知道原始的 {class}`PackageImporter`，以便它能找到对象依赖项的源代码。

```python
importer = PackageImporter(f)
obj = importer.load_pickle("model", "model.pkl")

# 在一个新包中重新导出 obj
with PackageExporter(f2, importer=(importer, sys_importer)) as exporter:
    exporter.save_pickle("model", "model.pkl", obj)
```

## 说明

### `torch.package` 格式概述
一个 `torch.package` 文件是一个 ZIP 存档，通常使用 `.pt` 扩展名。在 ZIP 存档内部，有两种类型的文件：

* 框架文件，放置在 `.data/` 目录中。
* 用户文件，即所有其他文件。

例如，这是一个来自 `torchvision` 的完全打包的 ResNet 模型的样子：

```
resnet
├── .data  # 所有框架特定的数据都存储在这里。
│   │      # 其命名是为了避免与用户序列化的代码冲突。
│   ├── 94286146172688.storage  # 张量数据
│   ├── 94286146172784.storage
│   ├── extern_modules  # 包含外部模块名称的文本文件（例如 'torch'）
│   ├── version         # 版本元数据
│   ├── ...
├── model  # 序列化的模型
│   └── model.pkl
└── torchvision  # 所有代码依赖项都以源文件形式捕获
    └── models
        ├── resnet.py
        └── utils.py
```

#### 框架文件
`.data/` 目录由 torch.package 拥有，其内容被视为私有实现细节。
`torch.package` 格式不保证 `.data/` 的内容，但所做的任何更改都将向后兼容
（即，新版本的 PyTorch 将始终能够加载旧的 `torch.packages`）。

目前，`.data/` 目录包含以下项目：

* `version`：序列化格式的版本号，以便 `torch.package` 导入基础设施知道如何加载此包。
* `extern_modules`：被视为 `extern` 的模块列表。`extern` 模块将使用加载环境的系统导入器导入。
* `*.storage`：序列化的张量数据。

```
.data
├── 94286146172688.storage
├── 94286146172784.storage
├── extern_modules
├── version
├── ...
```

#### 用户文件
存档中的所有其他文件都是由用户放置的。其布局与 Python
[常规包](https://docs.python.org/3/reference/import.html#regular-packages) 相同。要深入了解 Python 打包的工作原理，
请查阅 [这篇文章](https://www.python.org/doc/essays/packages/)（它稍微有些过时，因此请使用
[Python 参考文档](https://docs.python.org/3/library/importlib.html) 仔细核对实现细节）。

```
<包根目录>
├── model  # 序列化的模型
│   └── model.pkl
├── another_package
│   ├── __init__.py
│   ├── foo.txt         # 资源文件，参见 importlib.resources
│   └── ...
└── torchvision
    └── models
        ├── resnet.py   # torchvision.models.resnet
        └── utils.py    # torchvision.models.utils
```

### `torch.package` 如何查找代码的依赖项
#### 分析对象的依赖项
当你调用 `save_pickle(obj, ...)` 时，{class}`PackageExporter` 会正常地 pickle 该对象。然后，它使用 `pickletools` 标准库模块来解析 pickle 字节码。

在 pickle 中，对象会与一个 `GLOBAL` 操作码一起保存，该操作码描述了在哪里可以找到对象类型的实现，例如：

```
GLOBAL 'torchvision.models.resnet Resnet`
```

依赖解析器将收集所有 `GLOBAL` 操作码，并将它们标记为你 pickled 对象的依赖项。
有关 pickling 和 pickle 格式的更多信息，请查阅 [Python 文档](https://docs.python.org/3/library/pickle.html)。

#### 分析模块的依赖项
当一个 Python 模块被识别为依赖项时，`torch.package` 会遍历该模块的 Python AST 表示，并查找导入语句，完全支持标准形式：`from x import y`、`import z`、`from w import v as u` 等。当遇到这些导入语句之一时，`torch.package` 会将导入的模块注册为依赖项，然后这些依赖项本身也会以同样的 AST 遍历方式被解析。

**注意**：AST 解析对 `__import__(...)` 语法的支持有限，并且不支持 `importlib.import_module` 调用。通常，你不应期望 `torch.package` 能检测到动态导入。

### 依赖管理
`torch.package` 会自动查找你的代码和对象所依赖的 Python 模块。这个过程称为依赖解析。
对于依赖解析器找到的每个模块，你必须指定要采取的*操作*。

允许的操作有：

* `intern`：将此模块放入包中。
* `extern`：声明此模块为包的外部依赖项。
* `mock`：将此模块替换为桩模块。
* `deny`：依赖此模块将在包导出期间引发错误。

最后，还有一个重要的操作，严格来说不属于 `torch.package` 的一部分：

* 重构：移除或更改代码中的依赖项。

请注意，操作仅针对整个 Python 模块定义。无法“仅”打包模块中的一个函数或类而将其他部分排除在外。
这是有意设计的。Python 没有为模块中定义的对象提供清晰的边界。依赖组织唯一明确定义的单位是模块，因此 `torch.package` 也使用模块。

操作通过模式应用于模块。模式可以是模块名称（`"foo.bar"`）或通配符（如 `"foo.**"`）。你可以使用 {class}`PackageExporter` 上的方法将模式与操作关联起来，例如：

```python
my_exporter.intern("torchvision.**")
my_exporter.extern("numpy")
```

如果一个模块匹配某个模式，相应的操作将应用于它。对于给定的模块，将按照模式定义的顺序进行检查，并采取第一个匹配的操作。

#### `intern`
如果一个模块被 `intern`，它将被放入包中。

此操作用于你的模型代码，或任何你想要打包的相关代码。例如，如果你尝试打包来自 `torchvision` 的 ResNet，你将需要 `intern` 模块 torchvision.models.resnet。

在包导入时，当你的打包代码尝试导入一个被 `intern` 的模块时，PackageImporter 将在你的包内查找该模块。
如果找不到该模块，将引发错误。这确保了每个 {class}`PackageImporter` 与加载环境隔离——即使你的包和加载环境中都有 `my_interned_module`，{class}`PackageImporter` 也只会使用你包中的版本。

**注意**：只有 Python 源代码模块可以被 `intern`。其他类型的模块，如 C 扩展模块和字节码模块，如果你尝试 `intern` 它们，将会引发错误。这些类型的模块需要被 `mock` 或 `extern`。

#### `extern`
如果一个模块被 `extern`，它将不会被打包。相反，它将被添加到此包的外部依赖项列表中。你可以在 `package_exporter.extern_modules` 上找到此列表。

在包导入时，当打包代码尝试导入一个被 `extern` 的模块时，{class}`PackageImporter` 将使用默认的 Python 导入器来查找该模块，就像你执行了 `importlib.import_module("my_externed_module")` 一样。如果找不到该模块，将引发错误。

通过这种方式，你可以从包内部依赖第三方库（如 `numpy` 和 `scipy`），而无需将它们也打包进去。

**警告**：如果任何外部库以不向后兼容的方式更改，你的包可能无法加载。如果你需要包的长期可重现性，请尽量限制使用 `extern`。

#### `mock`
如果一个模块被 `mock`，它将不会被打包。相反，一个桩模块将被打包以替代它。桩模块将允许你从中检索对象（因此 `from my_mocked_module import foo` 不会出错），但任何使用该对象的尝试都将引发 `NotImplementedError`。

`mock` 应用于那些你“知道”在加载的包中不需要，但仍然希望在非打包内容中可用的代码。例如，初始化/配置代码，或仅用于调试/训练的代码。

**警告**：通常，`mock` 应作为最后的手段使用。它引入了打包代码和非打包代码之间的行为差异，这可能导致后续的混淆。更推荐的做法是重构你的代码以移除不需要的依赖项。

#### 重构
管理依赖项的最佳方式是根本没有依赖项！通常，可以通过重构代码来移除不必要的依赖项。以下是一些编写具有清晰依赖项的代码的指导原则（这些通常也是良好的实践！）：

**只包含你使用的部分**。不要在代码中留下未使用的导入。依赖解析器不够智能，无法判断它们确实未被使用，并且会尝试处理它们。

**限定导入范围**。例如，与其编写 `import foo` 并在后续使用 `foo.bar.baz`，不如直接编写 `from foo.bar import baz`。这样可以更精确地指定实际依赖项（`foo.bar`），并让依赖解析器知道你不需要整个 `foo` 模块。

**将包含无关功能的大型文件拆分为多个小文件**。如果你的 `utils` 模块包含一堆无关的功能，那么任何依赖 `utils` 的模块都需要引入大量无关的依赖项，即使你只需要其中一小部分功能。更好的做法是定义单一用途的模块，这些模块可以彼此独立地打包。

#### 模式
模式允许你使用便捷的语法来指定模块组。模式的语法和行为遵循 Bazel/Buck 的 [glob()](https://docs.bazel.build/versions/master/be/functions.html#glob)。

我们尝试与模式匹配的模块称为候选模块。候选模块由一系列由分隔符字符串分隔的段组成，例如 `foo.bar.baz`。

一个模式包含一个或多个段。段可以是：

* 字面字符串（例如 `foo`），表示精确匹配。
* 包含通配符的字符串（例如 `torch` 或 `foo*baz*`）。通配符匹配任何字符串，包括空字符串。
* 双通配符（`**`）。这匹配零个或多个完整的段。

示例：

* `torch.**`：匹配 `torch` 及其所有子模块，例如 `torch.nn` 和 `torch.nn.functional`。
* `torch.*`：匹配 `torch.nn` 或 `torch.functional`，但不匹配 `torch.nn.functional` 或 `torch`
* `torch*.**`：匹配 `torch`、`torchvision` 及其所有子模块

指定操作时，可以传递多个模式，例如：

```python
exporter.intern(["torchvision.models.**", "torchvision.utils.**"])
```

如果一个模块匹配其中任何一个模式，它就会匹配此操作。

你也可以指定要排除的模式，例如：

```python
exporter.mock("**", exclude=["torchvision.**"])
```

如果一个模块匹配任何排除模式，它将不会匹配此操作。在此示例中，我们模拟除 `torchvision` 及其子模块之外的所有模块。

当一个模块可能匹配多个操作时，将采用定义的第一个操作。

### `torch.package` 的注意事项
#### 避免模块中的全局状态
Python 使得在模块作用域内绑定对象和运行代码变得非常容易。这通常没问题——毕竟，函数和类就是以这种方式绑定到名称的。然而，当你打算在模块作用域内定义一个可变对象时，就会引入可变的全局状态，从而使情况变得更加复杂。

可变的全局状态非常有用——它可以减少样板代码，允许开放注册到表中等。但除非非常小心地使用，否则与 `torch.package` 一起使用时可能会导致复杂问题。

每个 {class}`PackageImporter` 为其内容创建一个独立的环境。这很好，因为这意味着我们可以加载多个包并确保它们彼此隔离，但当模块的编写方式假设了共享的可变全局状态时，这种行为可能会导致难以调试的错误。

#### 包和加载环境之间不共享类型
从 {class}`PackageImporter` 导入的任何类都将是特定于该导入器的类版本。例如：

```python
from foo import MyClass

my_class_instance = MyClass()

with PackageExporter(f) as exporter:
    exporter.save_module("foo")

importer = PackageImporter(f)
imported_MyClass = importer.import_module("foo").MyClass

assert isinstance(my_class_instance, MyClass)  # 正常
assert isinstance(my_class_instance, imported_MyClass)  # 错误！
```

在此示例中，`MyClass` 和 `imported_MyClass` **不是同一类型**。在这个特定示例中，`MyClass` 和 `imported_MyClass` 具有完全相同的实现，因此你可能认为将它们视为同一类是没问题的。但请考虑 `imported_MyClass` 来自一个具有完全不同 `MyClass` 实现的旧包的情况——在这种情况下，将它们视为同一类是不安全的。

在底层，每个导入器都有一个前缀，使其能够唯一标识类：

```python
print(MyClass.__name__)  # 输出 "foo.MyClass"
print(imported_MyClass.__name__)  # 输出 <torch_package_0>.foo.MyClass
```

这意味着，当其中一个参数来自包而另一个不是时，你不应期望 `isinstance` 检查能够正常工作。如果你需要此功能，请考虑以下选项：

* 使用鸭子类型（直接使用类，而不是显式检查它是否属于给定类型）。
* 将类型关系作为类契约的显式部分。例如，你可以添加属性标签 `self.handler = "handle_me_this_way"`，并让客户端代码检查 `handler` 的值，而不是直接检查类型。

### `torch.package` 如何保持包之间的隔离
每个 {class}`PackageImporter` 实例为其模块和对象创建一个独立、隔离的环境。包中的模块只能导入其他已打包的模块或标记为 `extern` 的模块。如果你使用多个 {class}`PackageImporter` 实例加载单个包，你将获得多个不交互的独立环境。

这是通过使用自定义导入器扩展 Python 的导入基础设施来实现的。{class}`PackageImporter` 提供了与 `importlib` 导入器相同的核心 API；即，它实现了 `import_module` 和 `__import__` 方法。

当你调用 {meth}`PackageImporter.import_module` 时，{class}`PackageImporter` 将构造并返回一个新模块，就像系统导入器所做的那样。然而，{class}`PackageImporter` 会修补返回的模块，使其使用 `self`（即该 {class}`PackageImporter` 实例）来满足未来的导入请求，通过查看包而不是搜索用户的 Python 环境来实现。

#### 名称重整
为避免混淆（“这个 `foo.bar` 对象是来自我的包，还是来自我的 Python 环境？”），{class}`PackageImporter` 会通过添加一个*重整前缀*来修改所有导入模块的 `__name__` 和 `__file__`。

对于 `__name__`，像 `torchvision.models.resnet18` 这样的名称会变为 `<torch_package_0>.torchvision.models.resnet18`。

对于 `__file__`，像 `torchvision/models/resnet18.py` 这样的名称会变为 `<torch_package_0>.torchvision/modules/resnet18.py`。

名称重整有助于避免不同包之间模块名称的无意冲突，并通过使堆栈跟踪和打印语句更清晰地显示它们是否引用打包代码来帮助调试。有关重整的面向开发者的详细信息，请查阅 `torch/package/` 中的 `mangling.md`。

## API 参考
```{eval-rst}
.. autoclass:: torch.package.PackagingError

.. autoclass:: torch.package.EmptyMatchError

.. autoclass:: torch.package.PackageExporter
  :members:

  .. automethod:: __init__

.. autoclass:: torch.package.PackageImporter
  :members:

  .. automethod:: __init__

.. autoclass:: torch.package.Directory
  :members:
```

<!-- 此模块需要文档记录。暂时添加在此处以供跟踪 -->
```{eval-rst}
.. py:module:: torch.package.analyze.find_first_use_of_broken_modules
.. py:module:: torch.package.analyze.is_from_package
.. py:module:: torch.package.analyze.trace_dependencies
.. py:module:: torch.package.file_structure_representation
.. py:module:: torch.package.find_file_dependencies
.. py:module:: torch.package.glob_group
.. py:module:: torch.package.importer
.. py:module:: torch.package.package_exporter
.. py:module:: torch.package.package_importer
```