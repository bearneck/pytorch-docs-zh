# 集合点 (Rendezvous) {#rendezvous-api}

 automodule
torch.distributed.elastic.rendezvous


下方是一个描述集合点如何工作的状态图。

![image](etcd_rdzv_diagram.png)

## 注册表 (Registry)

 {.autoclass members=""}
RendezvousParameters


 autoclass
RendezvousHandlerRegistry


 automodule
torch.distributed.elastic.rendezvous.registry


## 处理器 (Handler)

 currentmodule
torch.distributed.elastic.rendezvous


 {.autoclass members=""}
RendezvousHandler


## 数据类 (Dataclasses)

 autoclass
RendezvousInfo


 currentmodule
torch.distributed.elastic.rendezvous.api


 autoclass
RendezvousStoreInfo

 automethod
build(rank, store)


## 异常 (Exceptions)

 autoclass
RendezvousError


 autoclass
RendezvousClosedError


 autoclass
RendezvousTimeoutError


 autoclass
RendezvousConnectionError


 autoclass
RendezvousStateError


 autoclass
RendezvousGracefulExitError


## 实现 (Implementations)

### 动态集合点 (Dynamic Rendezvous)

 currentmodule
torch.distributed.elastic.rendezvous.dynamic_rendezvous


 autofunction
create_handler


 {.autoclass members="from_backend"}
DynamicRendezvousHandler()


 {.autoclass members=""}
RendezvousBackend


 {.autoclass members=""}
RendezvousTimeout


#### C10d 后端 (C10d Backend)

 currentmodule
torch.distributed.elastic.rendezvous.c10d_rendezvous_backend


 autofunction
create_backend


 {.autoclass members=""}
C10dRendezvousBackend


#### Etcd 后端 (Etcd Backend)

 currentmodule
torch.distributed.elastic.rendezvous.etcd_rendezvous_backend


 autofunction
create_backend


 {.autoclass members=""}
EtcdRendezvousBackend


### Etcd 集合点 (旧版) (Etcd Rendezvous (Legacy))

 warning
 title
Warning


`DynamicRendezvousHandler` 类已取代 `EtcdRendezvousHandler` 类，推荐大多数用户使用。`EtcdRendezvousHandler` 目前处于维护模式，未来将被弃用。


 currentmodule
torch.distributed.elastic.rendezvous.etcd_rendezvous


 autoclass
EtcdRendezvousHandler


### Etcd 存储 (Etcd Store)

`EtcdStore` 是当使用 etcd 作为集合点后端时，由 `next_rendezvous()` 返回的 C10d `Store` 实例类型。

 currentmodule
torch.distributed.elastic.rendezvous.etcd_store


 {.autoclass members=""}
EtcdStore


### Etcd 服务器 (Etcd Server)

`EtcdServer` 是一个便利类，可以让你轻松地在子进程中启动和停止一个 etcd 服务器。这对于测试或单节点（多工作进程）部署非常有用，因为手动在旁设置一个 etcd 服务器很麻烦。

 warning
 title
Warning


对于生产环境和多节点部署，请考虑正确部署一个高可用的 etcd 服务器，因为这是分布式作业的单一故障点。


 currentmodule
torch.distributed.elastic.rendezvous.etcd_server


 autoclass
EtcdServer

