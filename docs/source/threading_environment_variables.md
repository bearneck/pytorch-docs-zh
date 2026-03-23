# 线程环境变量

```{list-table}


* - 变量
  - 描述
* - ``OMP_NUM_THREADS``
  - 设置用于 OpenMP 并行区域的最大线程数。
* - ``MKL_NUM_THREADS``
  - 设置用于 Intel MKL 库的最大线程数。注意，``MKL_NUM_THREADS`` 的优先级高于 ``OMP_NUM_THREADS``。
```
