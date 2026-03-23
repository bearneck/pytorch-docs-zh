# 集合点 (Rendezvous)


torch.distributed.elastic.rendezvous

下方是一个描述集合点如何工作的状态图。

![image](etcd_rdzv_diagram.png)

## 注册表 (Registry)


RendezvousParameters


RendezvousHandlerRegistry


torch.distributed.elastic.rendezvous.registry

## 处理器 (Handler)


RendezvousHandler

## 数据类 (Dataclasses)


RendezvousInfo



> **AUTOCLASS**
> RendezvousStoreInfo
>
>
> build(rank, store)
>
>
> ## 异常 (Exceptions)
>
>
> RendezvousError
>
>
> RendezvousClosedError
>
>
> RendezvousTimeoutError
>
>
> RendezvousConnectionError
>
>
> RendezvousStateError
>
>
> RendezvousGracefulExitError
>
> ## 实现 (Implementations)
>
> ### 动态集合点 (Dynamic Rendezvous)
>
>
> create_handler
>
>
> DynamicRendezvousHandler()
>
>
> RendezvousBackend
>
>
> RendezvousTimeout
>
> #### C10d 后端 (C10d Backend)
>
>
> create_backend
>
>
> C10dRendezvousBackend
>
> #### Etcd 后端 (Etcd Backend)
>
>
> create_backend
>
>
> EtcdRendezvousBackend
>
> ### Etcd 集合点 (旧版) (Etcd Rendezvous (Legacy))


> ⚠️ **警告**
> `DynamicRendezvousHandler` 类已取代 `EtcdRendezvousHandler` 类，推荐大多数用户使用。`EtcdRendezvousHandler` 目前处于维护模式，未来将被弃用。
>
>
> EtcdRendezvousHandler
>
> ### Etcd 存储 (Etcd Store)
>
> `EtcdStore` 是当使用 etcd 作为集合点后端时，由 `next_rendezvous()` 返回的 C10d `Store` 实例类型。
>
>
> EtcdStore
>
> ### Etcd 服务器 (Etcd Server)
>
> `EtcdServer` 是一个便利类，可以让你轻松地在子进程中启动和停止一个 etcd 服务器。这对于测试或单节点（多工作进程）部署非常有用，因为手动在旁设置一个 etcd 服务器很麻烦。


> ⚠️ **警告**
> 对于生产环境和多节点部署，请考虑正确部署一个高可用的 etcd 服务器，因为这是分布式作业的单一故障点。
>
>
> EtcdServer

