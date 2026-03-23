(torch_environment_variables)=
# Torch 环境变量

PyTorch 利用环境变量来调整影响其运行时行为的各种设置。
这些变量提供了对关键功能的控制，例如在遇到错误时显示 C++ 堆栈跟踪、同步 CUDA 内核的执行、
指定并行处理任务的线程数等等。

此外，PyTorch 利用了多个高性能库，例如 MKL 和 cuDNN，
这些库也使用环境变量来修改其功能。
这些设置的相互作用允许构建一个高度可定制的开发环境，可以针对效率、调试和计算资源管理进行优化。

请注意，虽然本文档涵盖了与 PyTorch 及其相关库相关的广泛环境变量，但并非详尽无遗。
如果您发现本文档中缺少任何内容、存在错误或有改进之处，请通过提交问题或创建拉取请求告知我们。

```{eval-rst}
.. toctree::
   :maxdepth: 1

   threading_environment_variables
   cuda_environment_variables
   mps_environment_variables
   debugging_environment_variables
   miscellaneous_environment_variables
   logging
   torch_nccl_environment_variables

```