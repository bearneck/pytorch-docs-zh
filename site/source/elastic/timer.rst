过期计时器
==================

.. automodule:: torch.distributed.elastic.timer
.. currentmodule:: torch.distributed.elastic.timer

客户端方法
---------------
.. autofunction:: torch.distributed.elastic.timer.configure

.. autofunction:: torch.distributed.elastic.timer.expires

服务器/客户端实现
------------------------------
以下是 torchelastic 提供的计时器服务器和客户端对。

.. note:: 计时器服务器和客户端必须始终成对实现和使用，
          因为服务器和客户端之间存在消息传递协议。

以下是一对基于 ``multiprocess.Queue`` 实现的计时器服务器和客户端。

.. autoclass:: LocalTimerServer

.. autoclass:: LocalTimerClient

以下是另一对基于命名管道实现的计时器服务器和客户端。

.. autoclass:: FileTimerServer

.. autoclass:: FileTimerClient


编写自定义计时器服务器/客户端
--------------------------------------

要编写自己的计时器服务器和客户端，请分别继承
``torch.distributed.elastic.timer.TimerServer``（服务器）和
``torch.distributed.elastic.timer.TimerClient``（客户端）。
``TimerRequest`` 对象用于在服务器和客户端之间传递消息。

.. autoclass:: TimerRequest
   :members:

.. autoclass:: TimerServer
   :members:

.. autoclass:: TimerClient
   :members:


调试信息日志记录
-------------------

.. automodule:: torch.distributed.elastic.timer.debug_info_logging

.. autofunction:: torch.distributed.elastic.timer.debug_info_logging.log_debug_info_for_expired_timers