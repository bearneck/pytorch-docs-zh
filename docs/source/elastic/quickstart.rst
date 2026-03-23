快速入门
===========

要启动一个**容错**作业，请在所有节点上运行以下命令。

.. code-block:: bash

    torchrun
       --nnodes=NUM_NODES
       --nproc-per-node=TRAINERS_PER_NODE
       --max-restarts=NUM_ALLOWED_FAILURES
       --rdzv-id=JOB_ID
       --rdzv-backend=c10d
       --rdzv-endpoint=HOST_NODE_ADDR
       YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)


要启动一个**弹性**作业，请在至少 ``MIN_SIZE`` 个节点和至多 ``MAX_SIZE`` 个节点上运行以下命令。

.. code-block:: bash

    torchrun
        --nnodes=MIN_SIZE:MAX_SIZE
        --nproc-per-node=TRAINERS_PER_NODE
        --max-restarts=NUM_ALLOWED_FAILURES_OR_MEMBERSHIP_CHANGES
        --rdzv-id=JOB_ID
        --rdzv-backend=c10d
        --rdzv-endpoint=HOST_NODE_ADDR
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

.. note::
   TorchElastic 将故障建模为成员关系变更。当一个节点发生故障时，这被视为一次"缩减"事件。当调度器替换了故障节点时，则是一次"扩容"事件。因此，对于容错作业和弹性作业，``--max-restarts`` 都用于控制在放弃之前允许的总重启次数，无论重启是由于故障还是扩缩容事件引起的。

``HOST_NODE_ADDR`` 的格式为 <主机>[:<端口>]（例如 node1.example.com:29400），它指定了应实例化并托管 C10d 集合后端的节点和端口。它可以是您训练集群中的任何节点，但理想情况下应选择一个具有高带宽的节点。

.. note::
   如果未指定端口号，``HOST_NODE_ADDR`` 默认为 29400。

.. note::
   可以传递 ``--standalone`` 选项来启动一个带有 sidecar 集合后端的单节点作业。当使用 ``--standalone`` 选项时，您无需传递 ``--rdzv-id``、``--rdzv-endpoint`` 和 ``--rdzv-backend``。

.. note::
   了解更多关于编写分布式训练脚本的信息，请参阅 `此处 <train_script.html>`_。

如果 ``torchrun`` 不能满足您的需求，您可以直接使用我们的 API 进行更强大的自定义。首先，请查看 `弹性代理 <agent.html>`_ API。