# libtorch (仅限 C++)

PyTorch 的核心不依赖于 Python。一个基于 CMake 的构建系统将 C++ 源代码编译成一个共享对象文件 libtorch.so。

## AMD ROCm 支持

如果您正在为 AMD ROCm 进行编译，请首先运行以下命令： :: cd \<pytorch_root\>

> \# 仅在为 ROCm 编译时运行此命令 python tools/amd_build/build_amd.py

关于 ROCm 支持的更多信息可以在顶层的 [README](https://github.com/pytorch/pytorch/blob/main/README.md) 中找到。

## 使用 Python 构建 libtorch

您可以使用位于 tools 包中的 Python 脚本/模块来构建 libtorch。 :: cd \<pytorch_root\>

> \# 创建一个新文件夹用于构建，以避免污染源代码目录 mkdir build_libtorch && cd build_libtorch
>
> \# 您可能需要在此处导出一些必需的环境变量。 \# 通常 setup.py 会设置良好的默认环境变量，但您需要手动完成。 python ../tools/build_libtorch.py

或者，您可以正常调用 setup.py，然后复制构建好的 C++ 库。此方法可能会对您当前活跃的 Python 环境产生副作用。 :: cd \<pytorch_root\> python setup.py build

> ls torch/lib/tmp_install \# 输出文件在此处生成 ls torch/lib/tmp_install/lib/libtorch.so \# 特别关注此文件

要生成 libtorch.a 而不是 libtorch.so，请设置环境变量 [BUILD_SHARED_LIBS=OFF]。

要使用 ninja 而不是 make，请设置 [CMAKE_GENERATOR=\"-GNinja\" CMAKE_INSTALL=\"ninja install\"]。

请注意，我们正在努力淘汰 tools/build_pytorch_libs.sh，转而采用统一的 cmake 构建。

## 使用 CMake 构建 libtorch

您可以直接使用 cmake 构建 C++ 的 libtorch.so。例如，要从主分支构建一个 Release 版本并将其安装到下面 CMAKE_INSTALL_PREFIX 指定的目录中，您可以使用： :: git clone -b main \--recurse-submodule <https://github.com/pytorch/pytorch.git> mkdir pytorch-build cd pytorch-build cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=\`which python3\` -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install ../pytorch cmake \--build . \--target install

例如，要使用发布分支 v1.6.0，请将 `main` 替换为 `v1.6.0`。如果您没有所需的依赖项（例如 Python3 的 PyYAML 包），将会出现错误。
