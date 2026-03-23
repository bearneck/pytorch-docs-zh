# torch.utils.module_tracker


此工具可用于追踪在 `torch.nn.Module` 层次结构中的当前位置。
它可以在其他追踪工具中使用，以便能够轻松地将测量到的量与用户友好的名称关联起来。这在当前的 FlopCounterMode 中尤其有用。
