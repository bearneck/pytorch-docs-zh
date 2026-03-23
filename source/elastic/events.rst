.. _events-api:

事件
============================

.. automodule:: torch.distributed.elastic.events

API 方法
------------

.. autofunction:: torch.distributed.elastic.events.record

.. autofunction:: torch.distributed.elastic.events.construct_and_record_rdzv_event

.. autofunction:: torch.distributed.elastic.events.get_logging_handler

.. autofunction:: torch.distributed.elastic.events.record_rdzv_event

事件对象
-----------------

.. currentmodule:: torch.distributed.elastic.events.api

.. autoclass:: torch.distributed.elastic.events.api.Event

.. autoclass:: torch.distributed.elastic.events.api.EventSource

.. autoclass:: torch.distributed.elastic.events.api.EventMetadataValue