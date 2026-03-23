# 安装 PyTorch 的 C++ 发行版

我们提供了所有头文件、库以及 CMake 配置文件所需的二进制发行版，以便依赖 PyTorch。我们称此发行版为 *LibTorch*，您可以在 [我们的网站](https://pytorch.org/get-started/locally/) 上下载包含最新 LibTorch 发行版的 ZIP 压缩包。以下是一个依赖 LibTorch 并使用 PyTorch C++ API 附带的 `torch::Tensor` 类编写最小应用程序的小例子。

## 最小示例

第一步是通过上面的链接下载 LibTorch ZIP 压缩包。例如：

``` sh
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```

请注意，上面的链接是仅支持 CPU 的 libtorch。如果您想下载支持 GPU 的 libtorch，请在 <https://pytorch.org> 的链接选择器中找到正确的链接。

如果您是 Windows 开发者并且不想使用 CMake，可以跳转到 Visual Studio 扩展部分。

接下来，我们可以编写一个最小的 CMake 构建配置来开发一个依赖 LibTorch 的小型应用程序。使用 LibTorch 并不强制要求 CMake，但它是推荐且受支持的构建系统，并且未来会得到良好支持。一个最基本的 [CMakeLists.txt] 文件可能如下所示：

``` cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# 建议在 Windows 上使用以下代码块。
# 根据 https://github.com/pytorch/pytorch/issues/25457，
# 需要复制 DLL 以避免内存错误。
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
```

我们的示例实现将简单地创建一个新的 [torch::Tensor] 并打印它：

``` cpp
#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}
```

虽然您可以包含更细粒度的头文件以仅访问 PyTorch C++ API 的部分功能，但包含 [torch/torch.h] 是包含其大部分功能的最可靠方式。

最后一步是构建应用程序。为此，假设我们的示例目录结构如下：

``` sh
example-app/
  CMakeLists.txt
  example-app.cpp
```

我们现在可以在 `example-app/` 文件夹内运行以下命令来构建应用程序：

``` sh
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build . --config Release
```

其中 `/absolute/path/to/libtorch` 应该是解压后的 LibTorch 发行版的绝对（！）路径。如果 PyTorch 是通过 pip 安装的，可以使用 [torch.utils.cmake_prefix_path] 变量查询 [CMAKE_PREFIX_PATH]。在这种情况下，CMake 配置步骤将如下所示：

``` sh
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
```

如果一切顺利，输出将类似于：

``` sh
root@4b5a67132e81:/example-app# mkdir build
root@4b5a67132e81:/example-app# cd build
root@4b5a67132e81:/example-app/build# cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
-- The C compiler identification is GNU 5.4.0
-- The CXX compiler identification is GNU 5.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Looking for pthread_create
-- Looking for pthread_create - not found
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE
-- Configuring done
-- Generating done
-- Build files have been written to: /example-app/build
root@4b5a67132e81:/example-app/build# cmake --build . --config Release
Scanning dependencies of target example-app
[ 50%] Building CXX object CMakeFiles/example-app.dir/example-app.cpp.o
[100%] Linking CXX executable example-app
[100%] Built target example-app
```

现在执行在 `build` 文件夹中找到的 `example-app` 二进制文件应该会愉快地打印出张量（具体输出因随机性而异）：

``` sh
root@4b5a67132e81:/example-app/build# ./example-app
0.2063  0.6593  0.0866
0.0796  0.5841  0.1569
 Variable[CPUFloatType ]
```


> 💡 **提示**
> 在 Windows 上，调试版和发布版构建不兼容 ABI。如果您计划在调试模式下构建项目，请尝试使用 LibTorch 的调试版本。同时，请确保在上述 `cmake --build .` 行中指定正确的配置。
>
> ## 系统要求
>
> 为确保 LibTorch 的顺利安装和使用，请确保您的系统满足以下要求：
>
> 1\. **GLIBC 版本**：
>
> :   - 对于 cxx11 ABI 版本，需要 GLIBC 2.29 或更高版本
>
> 2\. **GCC 版本**：
>
> :   - 对于 cxx11 ABI 版本，需要 GCC 9 或更高版本
>
> ## Visual Studio 扩展
>
> [LibTorch Project Template](https://marketplace.visualstudio.com/items?itemName=YiZhang.LibTorch001) 可以帮助 Windows 开发者 为调试和发布版本设置所有 libtorch 项目配置和链接选项。 它易于使用，你可以查看 [演示视频](https://ossci-windows.s3.us-east-1.amazonaws.com/vsextension/demo.mp4)。 唯一的先决条件是从 <https://pytorch.org> 下载 libtorch。
>
> ## 支持
>
> 如果你在此安装和基本使用指南中遇到任何问题， 请使用我们的 [论坛](https://discuss.pytorch.org/) 或 [GitHub issues](https://github.com/pytorch/pytorch/issues) 联系我们。

