# DeepEP v2（elastic）Intranode Dispatch 全链路：从 Python 到 NCCL 环境到 CPP/Kernel

> **目标**：把 DeepEP v2（`deep_ep::elastic` 命名空间）的 **intranode dispatch** 从
> Python 调用 → NCCL 环境/对称内存创建 → C++ API → JIT 内核启动 的完整流程梳理清楚，
> 供学习使用。每个关键结论都给出「文件 + 行号」作为证据。
>
> **范围**：聚焦 **单机（intranode / scaleup-NVLink）** 路径，即 `num_scaleout_ranks == 1`、
> `is_scaleup_nvlink == true`，内核走 `dispatch_impl`（而非多机的 `hybrid_dispatch_impl`）。
>
> **代码版本**：对齐当前 `DeepEP/` 子模块。核心内核 `dispatch_impl` 见
> `DeepEP/deep_ep/include/deep_ep/impls/dispatch.cuh`。

---

## 0. TL;DR（一眼看懂）

整个生命周期分为**两个阶段**：

1. **一次性：Buffer / NCCL 环境创建**（`ElasticBuffer.__init__`）
   - 复用 PyTorch 的 NCCL communicator（或新建）→ 拿到 `ncclComm_t`。
   - 创建 **NCCL device communicator**（`ncclDevCommCreate`，含 GIN contexts = QPs）。
   - 从 device comm 读出 **LSA 域大小**（`lsaSize/lsaRank`）→ 确定 NVLink 域。
   - 分配 **对称内存** 并注册 **NCCL window**（`ncclCommWindowRegister`）。
   - 取得所有 LSA peer 的映射指针（`ncclGetLsaDevicePointer`）→ 之后 `get_sym_ptr` 做地址翻译。

2. **每次调用：dispatch**（`ElasticBuffer.dispatch`）
   - Python 决定 SM / QP 数 → 调用 pybind 的 `runtime.dispatch`。
   - C++ 侧做 stream 控制、分配 handle 张量。
   - **JIT 生成并编译** `dispatch_impl<...>` 模板实例 → 启动内核（notify warps + dispatch warps，TMA + LSA 单边写）。
   - **默认 `do_cpu_sync=False`（本文聚焦）**：不做任何 CPU 同步，直接按**最坏情况**
     （`num_max_tokens_per_rank × num_ranks`）分配 `recv_x`（`buffer.hpp:1034-1040`）。整条链路 host 端零参与。
     （可选 `do_cpu_sync=True` 才会 CPU busy-wait 读回精确计数得到 `num_recv_tokens`，见 `buffer.hpp:986`；
     cached handle 则直接复用计数。）
   - 启动 **copy epilogue**（满 SM）：用 GPU 侧前缀和 `psum` 把对称 recv buffer 拷进/整理进最终 `recv_x`。
   - stream epilogue → 返回 `recv_x, recv_topk_idx, recv_topk_weights, handle, event`。

关键区别于 v1：**用 NCCL GIN/LSA device API 做单边通信 + JIT 模板实例化 + warp specialization + TMA + 编程式依赖启动（PDL）copy epilogue**。

---

## 1. 顶层调用图

```
Python 用户
  │  x, topk_idx, topk_weights
  ▼
ElasticBuffer.dispatch()                         deep_ep/buffers/elastic.py:708
  │  get_theoretical_num_sms / num_qps
  ▼
self.runtime.dispatch(...)  ── pybind ──▶ elastic::ElasticBuffer::dispatch()   csrc/elastic/buffer.hpp:670
  │
  ├─ stream_control_prologue()                    csrc/elastic/buffer.hpp:496
  ├─ 分配 psum / dst_buffer_slot_idx 等 handle 张量
  ├─ launch_dispatch(...)                         csrc/kernels/elastic/dispatch.hpp:214
  │     │  JIT 生成 "dispatch_impl<...15 args...>"
  │     ▼
  │   dispatch_impl<kIsScaleupNVLink=true,...>    deep_ep/include/deep_ep/impls/dispatch.cuh:31
  │     ├─ notify warps  : 统计 rank/expert 计数、跨 SM 归约、GIN put 计数、prefix sum
  │     └─ dispatch warps: TMA load token → 算 dst rank/slot → TMA store 到 peer LSA buffer
  │     └─ 末尾 grid barrier + cudaTriggerProgrammaticLaunchCompletion (PDL)
  │
  ├─ 定 num_recv_tokens：默认 do_cpu_sync=False → 最坏情况分配  csrc/elastic/buffer.hpp:1034
  │     （可选 do_cpu_sync=True 才 CPU busy-wait 读回精确计数        csrc/elastic/buffer.hpp:986）
  ├─ 分配 recv_x / recv_topk_idx ...
  ├─ launch_dispatch_copy_epilogue(...)           csrc/kernels/elastic/dispatch.hpp:368
  │     ▼
  │   dispatch_copy_epilogue_impl<...>            deep_ep/include/deep_ep/impls/dispatch_copy_epilogue.cuh
  │
  └─ stream_control_epilogue() → event            csrc/elastic/buffer.hpp:526
```

---

## 2. 阶段一：Buffer / NCCL 环境创建

### 2.1 Python 入口：`ElasticBuffer.__init__`

```123:143:DeepEP/deep_ep/buffers/elastic.py
    def __init__(self,
                 group: dist.ProcessGroup,
                 ...
                 explicitly_destroy: bool = False):
```

关键步骤（`elastic.py`）：

1. **拿 NCCL comm handle**（第 172 行）：
   ```172:172:DeepEP/deep_ep/buffers/elastic.py
        self.nccl_comm_handle = get_nccl_comm_handle(group)
   ```
2. **计算 buffer 字节数**（第 175-180 行）：若未显式给 `num_bytes`，用 MoE 参数
   （`num_max_tokens_per_rank / hidden / num_topk`）调 `_C.calculate_elastic_buffer_size`。
   注意 **v2 的 buffer 是按最坏情况预分配的**（这正是我们之前讨论 grpcoll 不能照搬的点）。
3. **决定 QP 数**（第 199-206 行）：hybrid 模式下 65 或 129；非 hybrid 17。
4. **构造 C++ runtime**（第 217-226 行）：
   ```217:226:DeepEP/deep_ep/buffers/elastic.py
        self.runtime = _C.ElasticBuffer(group.rank(), group.size(),
                                        self.nccl_comm_handle.get(), cpu_comm,
                                        num_bytes, num_cpu_bytes,
                                        ...)
   ```
5. **收尾 barrier**（第 236-239 行）：`dist.barrier()` 保证所有 rank 初始化可见。

### 2.2 NCCL communicator 的获取（复用 PyTorch 的）

```41:71:DeepEP/deep_ep/utils/comm.py
def get_nccl_comm_handle(group: dist.ProcessGroup) -> NCCLCommHandle:
    ...
    backend = group._get_backend(torch.device('cuda'))
    if hasattr(backend, '_comm_ptr') and int(os.getenv('EP_REUSE_NCCL_COMM', '1')):
        _storage[group] = NCCLCommHandle(backend._comm_ptr(), False)
        return _storage[group]
    ...
    dist.all_gather_object(nccl_unique_ids, _C.get_local_nccl_unique_id(), group)
    ...
    _storage[group] = NCCLCommHandle(
        _C.create_nccl_comm(root_unique_id, group.size(), group.rank()), True)
```

- **默认复用**（`EP_REUSE_NCCL_COMM=1`）PyTorch backend 的 `_comm_ptr()`，`managed=False`，不销毁它。
- 旧版 PyTorch 没有 `_comm_ptr` 时，回退到 `ncclGetUniqueId` + `all_gather_object` + `ncclCommInitRank`。

底层 C 调用（`nccl.cu`）：
```20:41:DeepEP/csrc/kernels/backend/nccl.cu
pybind11::bytearray get_local_unique_id() {
    ncclUniqueId unique_id;
    NCCL_CHECK(ncclGetUniqueId(&unique_id));
    ...
}
int64_t create_nccl_comm(...) {
    ...
    NCCL_CHECK(ncclCommInitRank(&comm, num_ranks, root_unique_id, rank_idx));
    ...
}
```

### 2.3 C++ Buffer 构造：`ElasticBuffer` ctor

```82:143:DeepEP/csrc/elastic/buffer.hpp
    ElasticBuffer(const int& rank_idx, const int& num_ranks,
                  const int64_t& nccl_comm, const symmetric::cpu_comm_t& cpu_comm,
                  ...):
        ... {
        ...
        // Create NCCL symmetric memory context
        this->nccl_context = std::make_shared<nccl::NCCLSymmetricMemoryContext>(
            nccl_comm, cpu_comm, num_ranks, rank_idx,
            num_sym_bytes, num_cpu_buffer_bytes,
            allow_hybrid_mode, sl_idx, num_allocated_qps);
        ...
        workspace = this->nccl_context->mapped_window_ptr;
        buffer = static_cast<uint8_t*>(workspace) + num_workspace_bytes;
        CUDA_RUNTIME_CHECK(cudaMemset(workspace, 0, num_workspace_bytes));
        // Allocate host workspaces (mapped, for CPU sync)
        CUDA_RUNTIME_CHECK(cudaMallocHost(&host_workspace, ..., cudaHostAllocMapped));
        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&mapped_host_workspace, host_workspace, 0));
    }
```

- **对称内存布局**：`[[[Workspace] GPU buffer] CPU buffer]`（第 20、111 行注释）。`workspace` 放在最前，
  用于跨 SM/跨 rank 的计数归约、barrier signals 等；`buffer` 是真正的 dispatch/combine 收发区。
- **host workspace** 是 pinned + mapped 内存，供 GPU 写、CPU 轮询读（CPU sync 用）。

### 2.4 NCCL 环境核心：`NCCLSymmetricMemoryContext`（**重点**）

这是「NCCL 环境创建」的核心，全部在 `nccl.cu`：

```62:143:DeepEP/csrc/kernels/backend/nccl.cu
NCCLSymmetricMemoryContext::NCCLSymmetricMemoryContext(...) : ... {
    // 1) 复用传入的 host communicator
    comm = reinterpret_cast<ncclComm_t>(nccl_comm);

    // 2) 查询 GIN 是否可用
    ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));
    EP_HOST_ASSERT((allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE ...);

    // 3) 创建 device communicator（关键！设置 GIN contexts = QP 数）
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    if (num_ranks > 1 and not gin_disabled) {
        reqs.ginContextCount = num_allocated_qps;
        reqs.ginExclusiveContexts = true;
        reqs.ginQueueDepth = 1024;
        reqs.ginTrafficClass = sl_idx;
        reqs.ginSignalCount = num_ranks + 2 * 2;
        reqs.ginConnectionType = allow_hybrid_mode ? NCCL_GIN_CONNECTION_RAIL: NCCL_GIN_CONNECTION_FULL;
    }
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));

    // 4) 从 device comm 读 LSA（NVLink）域信息
    num_nvl_ranks = dev_comm.lsaSize, nvl_rank_idx = dev_comm.lsaRank;
    num_rdma_ranks = num_ranks / num_nvl_ranks, rdma_rank_idx = rank_idx / num_nvl_ranks;

    // 5) 逻辑域划分（intranode: num_scaleout_ranks==1, num_scaleup_ranks==num_nvl_ranks）
    ...
    is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;

    // 6) 分配对称内存
    this->symmetric_memory = symmetric::alloc(...);

    // 7) 注册 NCCL window（collective，内部有 bootstrapBarrier）
    NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr, ..., &window, NCCL_WIN_DEFAULT));
    NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &mapped_window_ptr));

    // 8) 取所有 LSA peer 的设备指针
    nvl_window_ptrs.resize(num_nvl_ranks);
    for (int i = 0; i < num_nvl_ranks; ++ i)
        NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, i, &nvl_window_ptrs[i]));
}
```

几个学习要点：

- **`ncclDevCommCreate` = "把 host comm 变成能在 kernel 内发起通信的 device comm"**。GIN（GPU-Initiated
  Networking）context 数量就是 QP 数；`ginSignalCount` 为自定义 RDMA barrier 预留 signal。
- **LSA（Load/Store Accessible）= NVLink 可直接访存的域**。`dev_comm.lsaSize` 就是本 NVLink 域内 GPU 数；
  intranode 场景 `num_scaleup_ranks == num_nvl_ranks`，`is_scaleup_nvlink == true`。
- **对称内存 + window** 是单边通信的基础：`ncclCommWindowRegister` 把本地 buffer 注册成对所有 peer 对称可寻址；
  `ncclGetLsaDevicePointer` 拿到每个 NVLink peer 的映射指针。
- **地址翻译** `get_sym_ptr`：把「本地 window 内某指针」翻译成「目标 rank 上同偏移的指针」：
  ```145:148:DeepEP/csrc/kernels/backend/nccl.cu
  void* NCCLSymmetricMemoryContext::get_sym_ptr(void* ptr, const int& dst_rank_idx) const {
      const auto offset = static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(mapped_window_ptr);
      return static_cast<uint8_t*>(nvl_window_ptrs[dst_rank_idx]) + offset;
  }
  ```
  内核里 `gin.get_sym_ptr<team_t>(...)` 就是干这件事——这正是 intranode dispatch 把 token 直接 TMA-store
  到「目标 rank 的 recv buffer 槽位」的地址来源。

### 2.5 pybind 绑定

```1307:1341:DeepEP/csrc/elastic/buffer.hpp
static void register_apis(pybind11::module_& m) {
    pybind11::class_<ElasticBuffer>(m, "ElasticBuffer")
        .def(pybind11::init<...>())
        ...
        .def("dispatch", &ElasticBuffer::dispatch)
        .def("combine", &ElasticBuffer::combine);
    m.def("calculate_elastic_buffer_size", &ElasticBuffer::calculate_buffer_size);
    ...
    m.def("get_local_nccl_unique_id", &nccl::get_local_unique_id);
    m.def("create_nccl_comm", &nccl::create_nccl_comm);
    ...
}
```

模块名 `_C`，由 `python_api.cpp` 里 `deep_ep::elastic::register_apis(m)` 注册（第 38 行）。

---

## 3. 阶段二：一次 dispatch 调用

### 3.1 Python `ElasticBuffer.dispatch`

```708:728:DeepEP/deep_ep/buffers/elastic.py
    def dispatch(self,
                 x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 topk_idx: Optional[torch.Tensor] = None,
                 topk_weights: Optional[torch.Tensor] = None,
                 ...):
```

- **自动决定 SM / QP 数**（第 776-779 行）：`get_theoretical_num_sms`（带宽建模，见 `elastic.py:582`）、
  `get_theoretical_num_qps`。
- **cached handle 复用**（第 786-802 行）：若传入 `handle`，跳过 layout 重算，且不做 CPU sync。
- **调 C++**（第 820-836 行）：`self.runtime.dispatch(...)`。
- **打包返回**（第 837-855 行）：构造 `EPHandle`，返回 `(recv_x, recv_topk_idx, recv_topk_weights, handle, EventOverlap)`。

### 3.2 C++ `ElasticBuffer::dispatch`

```670:690:DeepEP/csrc/elastic/buffer.hpp
    dispatch(const torch::Tensor& x,
             const std::optional<torch::Tensor>& sf,
             const torch::Tensor& topk_idx,
             ...) const {
```

主要步骤（`buffer.hpp`）：

1. **检查输入**（第 711-757 行）：`x` 连续、`num_tokens <= num_max_tokens_per_rank` 等。
2. **stream 控制 prologue**（第 761 行）：切到 comm stream、等 `previous_event`。见 `buffer.hpp:496`。
3. **分配 handle 张量**（第 766-789 行）：`psum_num_recv_tokens_per_expert`、
   `psum_num_recv_tokens_per_scaleup_rank`。
4. **intranode 分支**（第 817-848 行，`num_scaleout_ranks == 1`）：分配
   `dst_buffer_slot_idx = [num_tokens, num_topk]`（deterministic 时先跑一个 prologue kernel 预排槽位）。
   注意此路径 `num_channels = 1`（第 793 行；channel 拆分只用于 hybrid 多机）。
5. **清 host workspace 计数**（第 936-944 行）：给这一轮 CPU sync 用。
6. **启动 dispatch 内核**（第 948-970 行）：`launch_dispatch(...)`（见 §3.3）。
7. **拿接收计数**（第 972-1040 行）：
   - **默认 `do_cpu_sync=False`（本文聚焦）**：不做 CPU 同步，按最坏情况
     `num_max_tokens_per_rank * num_ranks` 分配（第 1034-1040 行）；
   - `cached_mode`：直接用 handle 里缓存的计数（也不做 CPU 同步）；
   - `do_cpu_sync=True`（可选）：**CPU 忙等**读 `host_workspace` 里每个 scaleup rank / expert 的计数
     （第 986-1033 行），得到精确 `num_recv_tokens`。
8. **分配 recv 张量**（第 1042-1079 行）：`recv_x`、`recv_topk_idx/weights`、`recv_src_metadata`。
9. **copy epilogue**（第 1093-1111 行）：`launch_dispatch_copy_epilogue(...)`（满 SM，把对称 recv buffer
   拷进最终 `recv_x`；对应内核里 PDL 触发的后续 kernel）。
10. **stream epilogue**（第 1114-1127 行）→ 返回。

### 3.3 Launcher：`launch_dispatch`（JIT 实例化内核）

```214:307:DeepEP/csrc/kernels/elastic/dispatch.hpp
static void launch_dispatch(void* x, void* sf, ...) {
    ...
    const int num_notify_warps = cached_mode ? 0 : kNumNotifyWarps;  // kNumNotifyWarps = 4
    ...
    if (num_scaleout_ranks == 1) {
        // 单机：按 shared memory 预算决定 dispatch warp 数（总 warp 数 <= 512）
        num_dispatch_warps = std::min<int>(std::min<int>(
            (num_smem_bytes - num_notify_smem_bytes) / token_layout.get_num_bytes<true>(), 32 - num_notify_warps),
            math::ceil_div(512, num_sms));
        num_threads = (num_notify_warps + num_dispatch_warps) * 32;
    }
    ...
    const auto code = DispatchRuntime::generate(args);           // 生成模板实例化源码字符串
    const auto runtime = jit::compiler->build("dispatch", code); // NVRTC 编译
    DispatchRuntime::launch(runtime, args, stream);              // 启动
}
```

**JIT 的关键**：v2 不用 v1 那种 `SWITCH_RANKS` 宏静态实例化，而是把模板参数拼成字符串在运行期编译：

```126:163:DeepEP/csrc/kernels/elastic/dispatch.hpp
    static std::string generate_impl(const Args& args) {
        ...
        if (args.num_scaleout_ranks == 1) {
            header_name = "dispatch";
            func_name = fmt::format("dispatch_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                args.is_scaleup_nvlink, args.do_cpu_sync, args.reuse_slot_indices,
                args.launch_args.grid_dim.first,
                args.num_notify_warps, args.num_dispatch_warps,
                args.num_scaleup_ranks, ...);
        } else {
            header_name = "hybrid_dispatch";  // 多机走这条
            ...
        }
        return fmt::format(R"(#include <deep_ep/impls/{}.cuh> ... &{} ...)", header_name, func_name);
    }
```

- **intranode → `dispatch_impl`**（`is_scaleup_nvlink=true`）；多机 → `hybrid_dispatch_impl`。
- **cluster 维度**（第 303 行）：`LaunchArgs(..., 2 - (num_sms % 2), true)` → 用 **thread block cluster**
  （偶数 SM 时 cluster=2），配合内核「每个 warp 当作一个 channel」与计算 kernel 的 overlap。

### 3.4 内核：`dispatch_impl`（warp specialization + TMA + LSA 单边写）

内核签名与角色划分（详见前文分析，这里给锚点）：

```31:78:DeepEP/deep_ep/include/deep_ep/impls/dispatch.cuh
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_impl(...) {
    ...
    // 每个 warp 视作一个 channel
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<...>(sm_idx, warp_idx - kNumNotifyWarps, warp_idx < kNumNotifyWarps);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);
    // 起始 grid barrier
    comm::gpu_barrier<...>(...);
    // 角色划分
    if (warp_idx < kNumNotifyWarps) { /* notify warps */ }
    else { /* dispatch warps */ }
```

- **notify warps**（`warp_idx < kNumNotifyWarps`，第 78-252 行）：在 shared memory 上原子累加每个
  token 的目标 rank / expert 计数 → 跨 SM 全网格归约（`ptx::red_add` 到 workspace）→ SM0 汇总后用
  `gin.put_value` 把计数写到各 peer → 做 prefix sum。
- **dispatch warps**（第 253-388 行）：**token 并行**网格步进
  `token_start = dispatch_warp_idx * kNumSMs + sm_idx`（第 272 行），每个 token：
  1. `ptx::tma_load_1d` 把 hidden 从 global 载入 shared（第 283 行）；
  2. 载 topk、算目标 rank/slot（去重 + 原子分槽，第 308-345 行）；
  3. `ptx::tma_store_1d` 写到 **目标 rank 的 recv buffer 槽位**
     （`gin.get_sym_ptr<team_t>(recv_buffer.get_token_buffer(slot)...)`，第 367-372 行）——这就是 §2.4 的对称地址。
- **收尾**（第 391-397 行）：`gpu_barrier` 保证数据到达 + `cudaTriggerProgrammaticLaunchCompletion()`（PDL）
  触发 copy epilogue kernel 提前启动，与本 kernel 收尾 overlap。

### 3.5 Copy epilogue

`launch_dispatch_copy_epilogue`（`dispatch.hpp:368`）用**满 SM** 启动
`dispatch_copy_epilogue_impl`（`impls/dispatch_copy_epilogue.cuh`），借助 **GPU 侧前缀和 `psum`**
（`psum_num_recv_tokens_per_scaleup_rank` / `psum_num_recv_tokens_per_expert`）把对称 recv buffer 里按
rank/slot 排布的 token 拷贝/整理进用户可见的 `recv_x`（并回填 `recv_src_metadata` 供 combine 反向路由）。

> **为何需要这个独立 kernel（与 `do_cpu_sync` 无关）**：dispatch warps 是 LSA/TMA **单边写**，
> 只能写目标 rank 的**对称** recv buffer；而返给用户的 `recv_x` 是普通 tensor（非对称），远端无法直接写入。
> 因此数据必然先落在对称 buffer，再由本 kernel 用 `psum` gather/compact 进 `recv_x`。
> 拆成独立 kernel 还因为它用**满 SM**、且可被 dispatch kernel 末尾的 PDL 提前触发以 overlap。
> 这与 v1 不同：v1 接收端在 dispatch kernel 内直接消费 ring buffer 写 `recv_x`，无需 epilogue。

---

## 4. 读代码索引（按调用顺序）

| 步骤 | 位置 |
|---|---|
| Python: Buffer 构造 | `DeepEP/deep_ep/buffers/elastic.py:123` |
| Python: 取/建 NCCL comm | `DeepEP/deep_ep/utils/comm.py:41` |
| Python: dispatch | `DeepEP/deep_ep/buffers/elastic.py:708` |
| pybind: 模块注册 | `DeepEP/csrc/python_api.cpp:38` |
| pybind: elastic APIs | `DeepEP/csrc/elastic/buffer.hpp:1307` |
| C++: Buffer ctor | `DeepEP/csrc/elastic/buffer.hpp:82` |
| C++: NCCL 环境/对称内存 | `DeepEP/csrc/kernels/backend/nccl.cu:62` |
| C++: `create_nccl_comm` / unique id | `DeepEP/csrc/kernels/backend/nccl.cu:20` |
| C++: `get_sym_ptr` 地址翻译 | `DeepEP/csrc/kernels/backend/nccl.cu:145` |
| C++: dispatch 主逻辑 | `DeepEP/csrc/elastic/buffer.hpp:670` |
| C++: CPU sync 读计数 | `DeepEP/csrc/elastic/buffer.hpp:986` |
| Launcher: launch_dispatch (JIT) | `DeepEP/csrc/kernels/elastic/dispatch.hpp:214` |
| Launcher: JIT 模板实例化字符串 | `DeepEP/csrc/kernels/elastic/dispatch.hpp:126` |
| Kernel: dispatch_impl | `DeepEP/deep_ep/include/deep_ep/impls/dispatch.cuh:31` |
| Kernel: copy epilogue launcher | `DeepEP/csrc/kernels/elastic/dispatch.hpp:368` |

---

## 5. Intranode 关键约定小结

- **判定 intranode**：`num_scaleout_ranks == 1`（单节点，`allow_hybrid_mode` 下 `num_rdma_ranks == 1`），
  `is_scaleup_nvlink == true`，`num_channels == 1`，内核走 `dispatch_impl`（非 hybrid）。
- **通信方式**：GPU 内核内 **LSA 单边写**（NVLink 对称内存 + TMA store），元数据/计数用 GIN put + workspace 归约。
- **buffer 定容**：v2 按 `num_max_tokens_per_rank × num_ranks` 最坏情况预分配 recv 区（MoE 有天然上界）；
  这与分布式注意力（无廉价上界）不同，也是 magi grpcoll 保留 v1 环形 buffer 的原因
  （见 `docs/agent/grpcoll_gin_ring_buffer_design.md §1.1`）。
- **同步/计数**：**默认 `do_cpu_sync=False`**——不做 CPU 同步，`recv_x` 按最坏情况分配，
  compaction 全靠 copy epilogue 里的 GPU 前缀和 `psum`（host 端零参与）。可选 `do_cpu_sync=True` 才 CPU
  busy-wait 读回精确接收数；cached handle 则直接复用计数、同样跳过 CPU 同步。
- **overlap**：PDL（`cudaTriggerProgrammaticLaunchCompletion`）+ thread block cluster，让 copy epilogue 与
  计算 kernel 更好地重叠。
```

