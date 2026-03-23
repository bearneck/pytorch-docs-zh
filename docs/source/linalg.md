```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# torch.linalg

常用线性代数运算。

关于一些常见的数值边界情况，请参阅 {ref}`线性代数稳定性 <Linear Algebra Stability>`。

```{eval-rst}
.. automodule:: torch.linalg
.. currentmodule:: torch.linalg
```

## 矩阵属性

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    norm
    vector_norm
    matrix_norm
    diagonal
    det
    slogdet
    cond
    matrix_rank
```

## 分解

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    cholesky
    qr
    lu
    lu_factor
    eig
    eigvals
    eigh
    eigvalsh
    svd
    svdvals
```

(linalg solvers)=

## 求解器

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    solve
    solve_triangular
    lu_solve
    lstsq
```

(linalg inverses)=

## 逆矩阵

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    inv
    pinv
```

## 矩阵函数

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    matrix_exp
    matrix_power
```

## 矩阵乘积

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    cross
    matmul
    vecdot
    multi_dot
    householder_product
```

## 张量运算

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorinv
    tensorsolve
```

## 杂项

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    vander
```

## 实验性函数

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    cholesky_ex
    inv_ex
    solve_ex
    lu_factor_ex
    ldl_factor
    ldl_factor_ex
    ldl_solve
```