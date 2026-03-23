:orphan:

.. _remote-reference-protocol:

远程引用协议
=========================

本文档描述了远程引用协议的设计细节，并逐步介绍了不同场景下的消息流。在继续之前，请确保您熟悉 :ref:`distributed-rpc-framework`。

背景
^^^^^^^^^^

RRef 代表远程引用。它是对位于本地或远程工作进程上的对象的引用，并在底层透明地处理引用计数。从概念上讲，它可以被视为一个分布式共享指针。应用程序可以通过调用 :meth:`~torch.distributed.rpc.remote` 来创建 RRef。每个 RRef 由 :meth:`~torch.distributed.rpc.remote` 调用的被调用方工作进程（即所有者）拥有，并可以被多个用户使用。所有者存储真实数据并跟踪全局引用计数。每个 RRef 可以通过一个全局唯一的 ``RRefId`` 来标识，该 ID 在 :meth:`~torch.distributed.rpc.remote` 调用的调用方创建时分配。

在所有者工作进程上，只有一个 ``OwnerRRef`` 实例，它包含真实数据；而在用户工作进程上，可以根据需要创建任意多个 ``UserRRef``，且 ``UserRRef`` 不持有数据。所有者上的所有使用都将通过全局唯一的 ``RRefId`` 来检索唯一的 ``OwnerRRef`` 实例。当在 :meth:`~torch.distributed.rpc.rpc_sync`、:meth:`~torch.distributed.rpc.rpc_async` 或 :meth:`~torch.distributed.rpc.remote` 调用中作为参数或返回值使用时，会创建一个 ``UserRRef``，并且会通知所有者以更新引用计数。当全局范围内没有 ``UserRRef`` 实例，并且所有者上也没有对 ``OwnerRRef`` 的引用时，``OwnerRRef`` 及其数据将被删除。

假设
^^^^^^^^^^^

RRef 协议的设计基于以下假设。

- **瞬时网络故障**：RRef 设计通过重试消息来处理瞬时网络故障。它无法处理节点崩溃或永久性网络分区。当这些事件发生时，应用程序应停止所有工作进程，恢复到上一个检查点，并恢复训练。
- **非幂等用户定义函数**：我们假设提供给 :meth:`~torch.distributed.rpc.rpc_sync`、:meth:`~torch.distributed.rpc.rpc_async` 或 :meth:`~torch.distributed.rpc.remote` 的用户函数不是幂等的，因此不能重试。然而，内部的 RRef 控制消息是幂等的，并在消息失败时重试。
- **乱序消息传递**：我们不假设任意一对节点之间的消息传递顺序，因为发送方和接收方都使用多个线程。无法保证哪条消息会先被处理。

RRef 生命周期
^^^^^^^^^^^^^

该协议的目标是在适当的时间删除 ``OwnerRRef``。删除 ``OwnerRRef`` 的正确时机是当没有存活的 ``UserRRef`` 实例，并且用户代码也没有持有对 ``OwnerRRef`` 的引用时。棘手的部分是确定是否存在任何存活的 ``UserRRef`` 实例。

设计推理
----------------

用户可以通过三种情况获取 ``UserRRef``：

1) 从所有者接收一个 ``UserRRef``。
2) 从另一个用户接收一个 ``UserRRef``。
3) 创建一个由另一个工作进程拥有的新 ``UserRRef``。

情况 1 是最简单的，所有者将其 RRef 传递给用户，其中所有者调用 :meth:`~torch.distributed.rpc.rpc_sync`、:meth:`~torch.distributed.rpc.rpc_async` 或 :meth:`~torch.distributed.rpc.remote` 并使用其 RRef 作为参数。在这种情况下，将在用户上创建一个新的 ``UserRRef``。由于所有者是调用方，它可以轻松更新其本地 ``OwnerRRef`` 的引用计数。

唯一的要求是任何 ``UserRRef`` 在销毁时必须通知所有者。因此，我们需要第一个保证：

**G1. 当任何 UserRRef 被删除时，所有者将收到通知。**

由于消息可能会延迟到达或乱序，我们需要另一个保证来确保删除消息不会被过早处理。如果 A 向 B 发送一条涉及 RRef 的消息，我们称 A 上的 RRef（父 RRef）和 B 上的 RRef（子 RRef）。

**G2. 在子 RRef 被所有者确认之前，父 RRef 不会被删除。**

在情况 2 和 3 中，所有者可能仅部分或完全不知道 RRef 的分支图。例如，一个 RRef 可能在用户上创建，并且在所有者收到任何 RPC 调用之前，创建者用户可能已经与其他用户共享了该 RRef，而这些用户可能进一步共享该 RRef。一个不变式是，任何 RRef 的分支图始终是一棵树，因为分叉一个 RRef 总是在被调用方（除非被调用方是所有者）创建一个新的 ``UserRRef`` 实例，因此每个 RRef 都有一个单一的父节点。

所有者对树中任何 ``UserRRef`` 的看法有三个阶段：

.. code::

  1) 未知 -> 2) 已知 -> 3) 已删除。

所有者对整个树的看法不断变化。当所有者认为没有存活的 ``UserRRef`` 实例时，它会删除其 ``OwnerRRef`` 实例，即当 ``OwnerRRef`` 被删除时，所有 ``UserRRef`` 实例可能确实已被删除或处于未知状态。危险的情况是当一些分支未知而其他分支已删除时。

**G2** 显然保证了在所有者知道其所有子 ``UserRRef`` 实例之前，任何父 ``UserRRef`` 都不能被删除。然而，子 ``UserRRef`` 可能在所有者知道其父 ``UserRRef`` 之前被删除。

考虑以下示例，其中 ``OwnerRRef`` 分叉到 A，然后 A 分叉到 Y，Y 分叉到 Z：

.. code::

  OwnerRRef -> A -> Y -> Z

如果 Z 的所有消息（包括删除消息）都在 Y 的消息之前被所有者处理，那么所有者将在知道 Y 存在之前得知 Z 的删除。然而，这不会引起任何问题。因为，至少 Y 的一个祖先（A）仍然存活，并且它会阻止所有者删除 ``OwnerRRef``。更具体地说，如果所有者不知道 Y，那么由于 **G2**，A 不能被删除，而所有者知道 A，因为它是 A 的父节点。

如果 RRef 是在用户上创建的，情况会稍微复杂一些：


.. code::

  OwnerRRef
      ^
      |
      A -> Y -> Z


如果 Z 在 ``UserRRef`` 上调用 :meth:`~torch.distributed.rpc.RRef.to_here`，那么当 Z 被删除时，所有者至少知道 A，因为否则 :meth:`~torch.distributed.rpc.RRef.to_here` 不会完成。如果 Z 没有调用 :meth:`~torch.distributed.rpc.RRef.to_here`，那么所有者有可能在收到来自 A 和 Y 的任何消息之前，先收到来自 Z 的所有消息。在这种情况下，由于 ``OwnerRRef`` 的实际数据尚未创建，也就没有什么可删除的。这等同于 Z 根本不存在。因此，这仍然是可行的。

实现
--------------

**G1** 通过在 ``UserRRef`` 析构函数中发送删除消息来实现。为了提供 **G2**，每当父 ``UserRRef`` 被分叉时，它会被放入一个上下文中，并以新的 ``ForkId`` 为索引。父 ``UserRRef`` 只有在收到来自子节点的确认消息（ACK）时，才会从上下文中移除，而子节点只有在得到所有者确认后才会发送 ACK。


协议场景
^^^^^^^^^^^^^^^^^^

现在让我们讨论上述设计如何转化为以下四种场景中的协议。

用户将 RRef 作为返回值与所有者共享
------------------------------------------


.. code::

  import torch
  import torch.distributed.rpc as rpc

  # 在 worker A 上
  rref = rpc.remote('B', torch.add, args=(torch.ones(2), 1))
  # 假设 rref 的 RRefId 为 100，ForkId 为 1
  rref.to_here()


在这种情况下，``UserRRef`` 在用户 worker A 上创建，然后它随远程消息一起传递给所有者 worker B，随后 B 创建 ``OwnerRRef``。方法 :meth:`~torch.distributed.rpc.remote` 立即返回，这意味着 ``UserRRef`` 可以在所有者知道它之前被分叉/使用。

在所有者端，当收到 :meth:`~torch.distributed.rpc.remote` 调用时，它会创建 ``OwnerRRef``，并返回一个 ACK 来确认 ``{100, 1}``（``RRefId``, ``ForkId``）。只有在收到此 ACK 后，A 才能删除其 ``UserRRef``。这涉及 **G1** 和 **G2**。**G1** 是显而易见的。对于 **G2**，``OwnerRRef`` 是 ``UserRRef`` 的子节点，而 ``UserRRef`` 在收到来自所有者的 ACK 之前不会被删除。

.. image:: https://user-images.githubusercontent.com/16999635/69164772-98181300-0abe-11ea-93a7-9ad9f757cd94.png
    :alt: user_to_owner_ret.png
    :width: 500 px

上图展示了消息流，其中实线箭头包含用户函数，虚线箭头是内置消息。请注意，从 A 到 B 的前两条消息（:meth:`~torch.distributed.rpc.remote` 和 :meth:`~torch.distributed.rpc.RRef.to_here`）可能以任意顺序到达 B，但最终的删除消息只有在以下条件满足时才会发送：

- B 确认了 ``UserRRef {100, 1}``（G2），并且
- Python GC 同意删除本地的 ``UserRRef`` 实例。这发生在 RRef 不再在作用域内且符合垃圾回收条件时。



用户将 RRef 作为参数与所有者共享
--------------------------------------

.. code::

  import torch
  import torch.distributed.rpc as rpc

  # 在 worker A 和 worker B 上
  def func(rref):
    pass

  # 在 worker A 上
  rref = rpc.remote('B', torch.add, args=(torch.ones(2), 1))
  # 假设 rref 的 RRefId 为 100，ForkId 为 1
  rpc.rpc_async('B', func, args=(rref, ))


在这种情况下，A 上创建 ``UserRRef`` 后，A 在后续对 B 的 RPC 调用中将其作为参数使用。A 将保持 ``UserRRef {100, 1}`` 存活，直到收到来自 B 的确认（**G2**，不是 RPC 调用的返回值）。这是必要的，因为在所有先前消息被接收之前，A 不应发送删除消息，否则，由于我们不保证消息传递顺序，``OwnerRRef`` 可能在使用前被删除。这是通过创建 RRef 的子 ``ForkId``，将它们保存在一个映射中，直到收到所有者对子 ``ForkId`` 的确认来实现的。下图展示了消息流。

.. image:: https://user-images.githubusercontent.com/16999635/69164845-b67e0e80-0abe-11ea-93fa-d24674e75a2b.png
    :alt: user_to_owner_arg.png
    :width: 500 px


请注意，``UserRRef`` 可能在 func 完成甚至开始之前就在 B 上被删除。然而这是可行的，因为在 B 发送子 ``ForkId`` 的 ACK 时，它已经获取了 ``OwnerRRef`` 实例，这将防止其过早被删除。


所有者将 RRef 与用户共享
--------------------------

所有者到用户是最简单的情况，所有者可以在本地更新引用计数，并且不需要任何额外的控制消息来通知其他人。关于 **G2**，它与父节点立即收到来自所有者的 ACK 相同，因为父节点就是所有者。

.. code::

  import torch
  import torch.distributed.rpc as RRef, rpc

  # 在 worker B 和 worker C 上
  def func(rref):
    pass

  # 在 worker B 上，创建一个本地 RRef
  rref = RRef("data")
  # 假设 rref 的 RRefId 为 100
  dist.rpc_async('C', func, args=(rref, ))


.. image:: https://user-images.githubusercontent.com/16999635/69164921-c990de80-0abe-11ea-9250-d32ad00cf4ae.png
    :alt: owner_to_user.png
    :width: 500 px

上图展示了消息流。请注意，当 ``OwnerRRef`` 在 rpc_async 调用后退出作用域时，它不会被删除，因为内部有一个映射来保持其存活，如果存在任何已知的分叉，在本例中是 ``UserRRef {100, 1}``。（**G2**）


用户将 RRef 与用户共享
-------------------------

这是最复杂的情况，调用方用户（父级 ``UserRRef``）、被调用方用户（子级 ``UserRRef``）以及所有者都需要参与其中。

.. code::

  import torch
  import torch.distributed.rpc as rpc

  # 在 worker A 和 worker C 上
  def func(rref):
    pass

  # 在 worker A 上
  rref = rpc.remote('B', torch.add, args=(torch.ones(2), 1))
  # 假设该 rref 的 RRefId 为 100，ForkId 为 1
  rpc.rpc_async('C', func, args=(rref, ))

.. image:: https://user-images.githubusercontent.com/16999635/69164971-d6adcd80-0abe-11ea-971d-6b7af131f0fd.png
    :alt: user_to_user.png
    :width: 500 px

当 C 从 A 接收到子级 ``UserRRef`` 时，它会向所有者 B 发送一个 fork 请求。之后，当 B 确认了 C 上的 ``UserRRef`` 时，C 将并行执行两个操作：1) 向 A 发送子级确认（ACK），以及 2) 运行用户提供的函数。在此期间，父级（A）将保持其 ``UserRRef {100, 1}`` 存活，以实现 **G2** 目标。