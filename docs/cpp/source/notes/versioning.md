# 库版本控制

我们提供了版本号宏，用于识别正在使用的 LibTorch 版本。 使用示例：

``` cpp
#include <torch/torch.h>
#include <iostream>

int main() {
  std::cout << "PyTorch version from parts: "
    << TORCH_VERSION_MAJOR << "."
    << TORCH_VERSION_MINOR << "."
    << TORCH_VERSION_PATCH << std::endl;
  std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;
}
```

这将输出类似以下内容：

``` text
PyTorch version from parts: 1.8.0
PyTorch version: 1.8.0
```


> 📝 **注意**
> 这些宏仅在 PyTorch \>= 1.8.0 版本中可用。

