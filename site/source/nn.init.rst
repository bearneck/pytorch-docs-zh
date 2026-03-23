.. role:: hidden
    :class: hidden-section

.. _nn-init-doc:

torch.nn.init
=============

.. warning::
    本模块中的所有函数都旨在用于初始化神经网络参数，因此它们都在 :func:`torch.no_grad` 模式下运行，并且不会被 autograd 考虑在内。

.. currentmodule:: torch.nn.init
.. autofunction:: calculate_gain
.. autofunction:: uniform_
.. autofunction:: normal_
.. autofunction:: constant_
.. autofunction:: ones_
.. autofunction:: zeros_
.. autofunction:: eye_
.. autofunction:: dirac_
.. autofunction:: xavier_uniform_
.. autofunction:: xavier_normal_
.. autofunction:: kaiming_uniform_
.. autofunction:: kaiming_normal_
.. autofunction:: trunc_normal_
.. autofunction:: orthogonal_
.. autofunction:: sparse_