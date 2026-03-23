
> ⚠️ **警告**
> 在某些版本的 cuDNN 和 CUDA 上，RNN 函数存在已知的非确定性问题。 您可以通过设置以下环境变量来强制实现确定性行为：
>
> 在 CUDA 10.1 上，设置环境变量 `CUDA_LAUNCH_BLOCKING=1`。 这可能会影响性能。
>
> 在 CUDA 10.2 或更高版本上，设置环境变量 （注意开头的冒号符号） `CUBLAS_WORKSPACE_CONFIG=:16:8` 或 `CUBLAS_WORKSPACE_CONFIG=:4096:2`。
>
> 更多信息请参阅 [cuDNN 8 Release Notes](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-880/release-notes/rel_8.html)。

