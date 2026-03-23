# 调试环境变量


  * - 变量
    - 描述
  * - ``TORCH_SHOW_CPP_STACKTRACES``
    - 如果设置为 ``1``，当 PyTorch 检测到 C++ 错误时，会打印出堆栈跟踪信息。
  * - ``TORCH_CPP_LOG_LEVEL``
    - 设置 c10 日志工具的日志级别（同时支持 GLOG 和 c10 日志记录器）。有效值为 ``INFO``、``WARNING``、``ERROR`` 和 ``FATAL`` 或其对应的数字值 ``0``、``1``、``2`` 和 ``3``。
  * - ``TORCH_LOGS``
    -  关于此环境变量的更深入解释，请参阅 [/logging](/logging.md)。
