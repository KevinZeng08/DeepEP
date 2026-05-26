# DeepEP V2 核心架构分析

> 基于 DeepEP main 分支代码，结合 NCCL-EP 论文和社区分析文章的深度调研。

---

## 阶段一：架构总览与 V1→V2 变化对比

### 1.1 V2 整体定位

DeepEP V2 是面向 MoE (Mixture of Experts) 模型的高性能 Expert Parallel 通信库，核心功能是实现 **dispatch**（将 token 路由到目标 expert 所在的 rank）和 **combine**（将 expert 计算结果反向路由回原始 rank 并加权归约）。

**代码佐证** — `deep_ep/__init__.py` 明确声明版本：

```python
# deep_ep/__init__.py
__version__ = '2.0.0'
```

导出的核心 API 从 V1 的 `Buffer` + `Config` 扩展为 V2 的 `ElasticBuffer` + `EPHandle`：

```python
from .buffers.legacy import Buffer          # V1 遗留
from .buffers.elastic import ElasticBuffer, EPHandle  # V2 核心
```

### 1.2 V1 vs V2：接口统一

**V1 的问题：接口碎片化。** V1 维护了三套独立的 kernel 路径：

| V1 路径 | 场景 | 通信方式 | 代码位置 |
|---------|------|---------|---------|
| Intranode Normal | 节点内训练/prefill | NVLink | `csrc/kernels/legacy/intranode.cu` |
| Internode Normal | 跨节点训练/prefill | RDMA+NVLink 转发 | `csrc/kernels/legacy/internode.cu` |
| Low-latency | 推理 decode | 纯 RDMA | `csrc/kernels/legacy/internode_ll.cu` |

**代码佐证** — V1 `Buffer` 构造函数需要用户分别指定 NVLink 和 RDMA buffer 大小：

```python
# deep_ep/buffers/legacy.py
class Buffer:
    def __init__(self,
                 group, num_nvl_bytes=0, num_rdma_bytes=0,
                 low_latency_mode=False, num_qps_per_rank=24, ...):
```

**V2 的解决方案：统一 `ElasticBuffer`。** 所有 EP 操作——高吞吐和低延迟——统一到单一接口：

```python
# deep_ep/buffers/elastic.py
class ElasticBuffer:
    """
    The elastic communication buffer, which supports:
        - high-throughput expert-parallel all-to-all (dispatch and combine, using NVLink and/or RDMA)
        - Engram (remote KV cache fetch, using RDMA)
        - pipeline-parallel send/recv (PP, using NVLink)
        - all-gather reduce-scatter (AGRS, using NVLink)
    """
    def __init__(self, group, num_bytes=None,
                 num_max_tokens_per_rank=0, hidden=0, num_topk=0,
                 use_fp8_dispatch=False, deterministic=False,
                 allow_hybrid_mode=True, ...):
```

用户只需提供 MoE 语义参数（`num_max_tokens_per_rank`、`hidden`、`num_topk`），buffer 大小由框架自动计算。

### 1.3 V1 vs V2：后端切换 NVSHMEM → NCCL Gin

这是 V2 最深层的架构变化。

**V1 依赖 NVSHMEM** — 需要独立安装 NVSHMEM，通过 `nvshmem_malloc`/`nvshmem_put`/`nvshmem_get` 等 API 实现 GPU-initiated RDMA：

```bash
# V1 安装方式
NVSHMEM_DIR=/path/to/installed/nvshmem python setup.py install
```

V1 通过 IPC handle 交换 + `nvshmem` 对称内存实现跨 GPU 通信（见 `deep_ep/buffers/legacy.py` 中的 `get_local_ipc_handle` 调用链）。

**V2 切换到 NCCL Gin 后端** — 利用 NCCL 2.30.4+ 内置的 GIN (GPU-Initiated Network) 设备 API，无需独立安装 NVSHMEM：

```bash
# V2 安装方式 — 只需 NCCL
pip install "nvidia-nccl-cu13>=2.30.4" --no-deps
```

**代码佐证** — NCCL 后端初始化（`csrc/kernels/backend/nccl.cu`）：

```cpp
// 创建 NCCL 设备通信器，配置 GIN 资源
ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
if (num_ranks > 1 and get_env("EP_DISABLE_GIN", 0) == 0) {
    reqs.ginContextCount = num_allocated_qps;
    reqs.ginExclusiveContexts = true;
    reqs.ginQueueDepth = 1024;
    reqs.ginTrafficClass = sl_idx;
    reqs.ginSignalCount = num_ranks + 2 * 2;
    reqs.ginConnectionType = allow_hybrid_mode ?
        NCCL_GIN_CONNECTION_RAIL : NCCL_GIN_CONNECTION_FULL;
}
NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));
```

对称内存通过 NCCL Window API 分配：

```cpp
// 分配对称内存并注册窗口
NCCL_CHECK(ncclMemAlloc(&raw_window_ptr, size));
NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr, size, &window, NCCL_WIN_DEFAULT));
NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &mapped_window_ptr));
```

**切换的核心优势：**

| 维度 | V1 (NVSHMEM) | V2 (NCCL Gin) |
|------|-------------|---------------|
| 安装依赖 | 独立 NVSHMEM 库 | NCCL 2.30.4+ 自带 |
| 内存管理 | nvshmem_malloc | ncclMemAlloc + ncclCommWindowRegister |
| 拓扑感知 | 需要手动配置 | 复用 NCCL 拓扑检测 |
| NVLink 访问 | IPC handle 交换 | LSA (Local Symmetric Access) 指针 |
| RDMA 操作 | nvshmem_put/get | gin.put/get/signal/wait |
| QP 管理 | 手动 | NCCL 自动管理，支持独占/共享 |
| 生态兼容 | 独立生态 | 与 NCCL 集合操作共存 |

### 1.4 V1 vs V2：SM 占用大幅降低

V1 默认使用 20 个 SM（`Buffer.num_sms = 20`），且需要人工调优。

V2 通过**解析式 SM 计算**替代了 auto-tuning，核心方法是 `get_theoretical_num_sms`。其建模思路：

1. 基于 top-k 路由概率，计算**期望的跨 rank/scaleout 流量比例**
2. 分别估算 SM 的 HBM read/write 带宽需求
3. 与网络带宽（NVLink/RDMA）对齐，找到**带宽瓶颈**
4. 反推所需的最少 SM 数量

**代码佐证** — 带宽建模核心逻辑（`deep_ep/buffers/elastic.py`）：

```python
def get_theoretical_num_sms(self, num_experts, num_topk, ...):
    # 计算 top-k 路由到不同 rank 的期望数量
    def get_expected_topk(num_groups):
        return num_groups * (1 - math.comb(num_experts - num_experts // num_groups, num_topk)
                             / math.comb(num_experts, num_topk))

    num_expected_topk = get_expected_topk(self.num_ranks)

    # 估算各类流量
    sm_read += 1 / num_expected_topk    # 读 token
    sm_write += ...                      # 写 send buffer / NVLink
    nvlink_traffic += ...                # NVLink 流量
    rdma_traffic += ...                  # RDMA 流量

    # 找带宽瓶颈
    if rdma_traffic / rdma_gbs > nvlink_traffic / nvlink_gbs:
        bounded_traffic, bounded_gbs = rdma_traffic, rdma_gbs
    else:
        bounded_traffic, bounded_gbs = nvlink_traffic, nvlink_gbs

    # 反推 SM 数
    num_sms = max(
        bounded_gbs / bounded_traffic * sm_read / sm_read_gbs,
        bounded_gbs / bounded_traffic * sm_write / sm_write_gbs,
    )
    num_sms = align(max(4, math.ceil(num_sms * 1.25)), 2)  # 上浮 25%，对齐到偶数
```

**实际效果**：V3 配置（EP8x2）下，V2 仅需 **6 SM** 即可达到 V1 的 24 SM 性能，同时峰值性能提升 1.3x。

### 1.5 V2 的拓扑抽象：ScaleUp 与 ScaleOut

V2 引入了清晰的两层拓扑抽象：

- **ScaleUp 域**：NVLink 互联的 GPU 组（通常是单节点内的 8 GPU），使用 **LSA 对称指针**直接 load/store
- **ScaleOut 域**：跨节点的 RDMA 互联，使用 **GIN put/get** 操作

**代码佐证** — 域大小计算（`csrc/kernels/backend/nccl.cu`）：

```cpp
// 物理域 = NCCL 自动探测的 NVLink 拓扑
num_nvl_ranks = dev_comm.lsaSize;
num_rdma_ranks = num_ranks / num_nvl_ranks;

// 逻辑域 = 根据 hybrid_mode 决定
if (allow_hybrid_mode) {
    num_scaleout_ranks = num_rdma_ranks;   // 跨节点
    num_scaleup_ranks = num_nvl_ranks;     // 节点内
} else {
    num_scaleout_ranks = 1;                // 所有 rank 视为一个域
    num_scaleup_ranks = num_ranks;
}
```

当 `allow_hybrid_mode=true`（默认）时，V2 使用**层次化通信**：ScaleUp 域内先用 NVLink 聚合，再通过 RDMA 跨 ScaleOut 域传输。当 `allow_hybrid_mode=false` 时，所有 rank 视为纯 ScaleUp 域（适用于单节点或通过 NVLink 互联的多节点如 MNNVL）。

### 1.6 V2 的内存布局

V2 的 buffer 采用统一的**对称内存窗口**，workspace 在前、data buffer 在后：

```
[Window Memory]
├── WorkspaceLayout (固定大小，约 100+ KB)
│   ├── NVLink barrier 信号 (16 bytes)
│   ├── Notify reduction workspace
│   ├── ScaleUp rank/expert 发送/接收计数 (int64_t)
│   ├── ScaleOut channel metadata
│   ├── PP prev/next rank 计数
│   └── AGRS 信号
└── Data Buffer (用户指定大小)
    ├── Dispatch: recv_buffer + send_buffer
    ├── Combine: recv_buffer + send_buffer
    ├── Engram: storage + recv
    ├── PP: send + recv rings
    └── AGRS: session slots
```

**代码佐证** — buffer 分配（`csrc/elastic/buffer.hpp`）：

```cpp
// Workspace + Buffer 连续分配在同一对称内存窗口中
workspace = this->nccl_context->mapped_window_ptr;
buffer = static_cast<uint8_t*>(workspace) + layout::WorkspaceLayout::get_num_bytes();
```

WorkspaceLayout 的最大支持规模（`deep_ep/include/deep_ep/common/layout.cuh`）：

```cpp
static constexpr int kNumMaxRanks = 1024;
static constexpr int kNumMaxExperts = 2048;     // 支持 EP2048
static constexpr int kNumMaxExpertsPerRank = 256;
```

### 1.7 V2 的 JIT 编译框架

V1 的 kernel 是预编译的（`.cu` 文件直接在 setup.py build 时编译）。V2 转向**全 JIT**——kernel 在运行时根据模板参数生成 CUDA 代码、编译并缓存。

**设计动机**：dispatch/combine kernel 有大量编译期模板参数（`num_ranks`、`num_experts`、`num_topk`、`hidden`、`num_sms` 等），预编译会产生组合爆炸。JIT 可以为每种具体配置生成最优代码。

**代码佐证** — 以 dispatch prologue 为例（`csrc/kernels/elastic/dispatch.hpp`）：

```cpp
class DispatchPrologueRuntime final : public jit::LaunchRuntime<DispatchPrologueRuntime> {
    static std::string generate_impl(const Args& args) {
        // 运行时生成 CUDA 代码，模板参数硬编码为常量
        return fmt::format(R"(
#include <deep_ep/impls/dispatch_deterministic_prologue.cuh>
using namespace deep_ep::elastic;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &dispatch_deterministic_prologue_impl<{}, {}, {}, {}, {}, {}>);
}}
)", args.launch_args.grid_dim.first,
   args.num_warps, args.num_ranks,
   args.num_max_tokens_per_rank,
   args.num_experts, args.num_topk);
    }
};
```

编译结果缓存在 `$HOME/.deep_ep`（可通过 `EP_JIT_CACHE_DIR` 配置），避免重复编译。

---

## 阶段二：NCCL Gin 后端与通信原语

### 2.1 NCCL Gin 三层通信抽象

V2 的设备侧通信建立在 NCCL 提供的三层设备 API 之上，在 `handle.cuh` 中封装为 `NCCLGin` 结构体：

| 层级 | NCCL Team Tag | 通信方式 | 适用场景 |
|------|--------------|---------|---------|
| **LSA** (Local Symmetric Access) | `ncclTeamTagLsa` | NVLink load/store 对称指针 | 节点内 ScaleUp |
| **World** | `ncclTeamTagWorld` | 全局通信（NVLink 可达用 LSA，否则走 GIN RDMA） | 非 hybrid 模式 |
| **Rail** | `ncclTeamTagRail` | 同一 rail 内的跨节点 RDMA | hybrid 模式 ScaleOut |

**代码佐证** — `NCCLGin` 构造函数初始化三类 team（`deep_ep/include/deep_ep/common/handle.cuh`）：

```cpp
struct NCCLGin {
    ncclGin gin;
    ncclTeam team_world, team_lsa, team_rail;
    uint64_t lsa_base_ptr;

    __device__ __forceinline__
    NCCLGin(const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,
            const int& qp_idx, const ncclGinResourceSharingMode& resource_sharing_mode):
        gin(ncclGin(nccl_dev_comm, qp_idx, resource_sharing_mode)),
        team_world(ncclTeamWorld(nccl_dev_comm)),
        team_lsa(ncclTeamLsa(nccl_dev_comm)),
        team_rail(ncclTeamRail(nccl_dev_comm)) {
        lsa_base_ptr = reinterpret_cast<uint64_t>(
            ncclGetLsaPointer(nccl_window, 0, team_lsa.rank));
    }
```

### 2.2 NVLink vs RDMA 路径的自动路由

`NCCLGin` 的每个通信操作都会根据目标 rank 是否 NVLink 可达，自动选择 LSA 对称指针（零拷贝 load/store）或 GIN RDMA 操作。

**代码佐证** — `put_value` 自动路由（`handle.cuh`）：

```cpp
template <typename team_t, typename dtype_t>
__device__ __forceinline__
void put_value(dtype_t* sym_ptr, const dtype_t& value, const int& dst_rank_idx, ...) const {
    const auto dst_ptr = get_sym_ptr<team_t>(sym_ptr, dst_rank_idx);
    if (dst_ptr != nullptr) {
        // NVLink 可达 → 直接 store
        ptx::st_relaxed_sys(dst_ptr, value);
    } else {
        // NVLink 不可达 → GIN RDMA put
        gin.putValue(TEAM_WORLD_RAIL(), dst_rank_idx,
                     nccl_window, offset, value, ...);
    }
}
```

同样的模式应用于 `red_add_rel`（原子归约）：NVLink 可达时使用 `ptx::red_add_rel_sys`（inline PTX 原子加），否则通过 `gin.signal` 发送 RDMA 信号。

```cpp
template <typename team_t, typename dtype_t>
void red_add_rel(dtype_t* sym_ptr, const dtype_t& value, const int& dst_rank_idx, ...) const {
    const auto dst_ptr = get_sym_ptr<team_t>(sym_ptr, dst_rank_idx);
    if (dst_ptr != nullptr) {
        ptx::red_add_rel_sys(dst_ptr, value);           // NVLink 原子加
    } else {
        gin.signal(TEAM_WORLD_RAIL(), dst_rank_idx,
            ncclGin_VASignalAdd(nccl_window, offset, value), ...); // RDMA signal
    }
}
```

### 2.3 GIN 操作原语

V2 通过 `NCCLGin` 封装了以下 GIN 设备侧操作：

| 操作 | 方法 | 语义 | 应用场景 |
|------|------|------|---------|
| **put** | `gin.put<team_t>(recv_ptr, send_ptr, bytes, dst_rank)` | 单边 RDMA 写 | dispatch 跨节点发送 token |
| **get** | `gin.get<team_t>(src_ptr, dst_ptr, bytes, src_rank)` | 单边 RDMA 读 | Engram 远程 KV cache 获取 |
| **signal** | `gin.signal<team_t>(dst_rank, remote_action)` | 原子信号/通知 | barrier 同步、计数通知 |
| **put_value** | `gin.put_value<team_t>(ptr, value, dst_rank)` | 单值写入 | 通知 rank/expert 计数 |
| **flush_async** | `gin.flush_async<team_t>(src_rank, request)` | 异步刷新 | 确保 RDMA 完成 |
| **wait** | `gin.wait(request)` | 等待操作完成 | Engram flush 等待 |

**代码佐证** — `put` 操作的完整实现（`handle.cuh`）：

```cpp
template <typename team_t, typename remote_action_t = ncclGin_None>
void put(void* recv_sym_ptr, void* send_sym_ptr, const int& num_bytes,
         const int& dst_rank_idx, ...) const {
    gin.put(TEAM_WORLD_RAIL(), dst_rank_idx,
            nccl_window, offset(recv_sym_ptr),  // 目标偏移
            nccl_window, offset(send_sym_ptr),  // 源偏移
            num_bytes, remote_action, ...);
}
```

所有偏移量都通过 `lsa_base_ptr` 计算 — 这是 NCCL Window 的基址，确保所有 rank 使用一致的对称地址空间。

### 2.4 QP 资源管理与映射策略

V2 的 QP（Queue Pair）分配在初始化时由 NCCL 完成，kernel 内部通过 `get_qp_mode` 将 SM/warp 映射到 QP：

**代码佐证** — QP 分配策略（`deep_ep/include/deep_ep/common/comm.cuh`）：

```cpp
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps = false>
__device__ __forceinline__
std::pair<int, ncclGinResourceSharingMode> get_qp_mode(
    const int& sm_idx, const int& channel_in_sm_idx, const bool& is_notify_warp) {
    // Notify warp 独占 QP 0
    if (is_notify_warp) return {0, kSharingCTA};

    constexpr int kQPStartIdx = static_cast<int>(kWithNotifyWarps);
    constexpr int kNumAvailableQPs = kNumQPs - kQPStartIdx;

    if constexpr (kNumSMs <= kNumAvailableQPs) {
        // SM 数 ≤ 可用 QP 数 → 每个 SM 独占一组 QP（CTA 级共享）
        return {kQPStartIdx + sm_idx + (channel_idx % num_qps_in_sm) * kNumSMs,
                kSharingCTA};
    } else {
        // SM 数 > QP 数 → 所有 SM 共享 QP（GPU 级共享）
        return {kQPStartIdx + (global_channel_idx % kNumAvailableQPs),
                kSharingGrid};
    }
}
```

Python 层的 QP 数量自动计算（`elastic.py`）：

```python
# Hybrid 模式需要更多 QP（每个 channel + notify 独立 QP）
if self.allow_hybrid_mode:
    num_allocated_qps = 65 if check_fast_rdma_atomic_support() else 129
else:
    num_allocated_qps = 17
```

### 2.5 GPU Barrier 实现

V2 实现了两种 GPU 侧 barrier，根据拓扑自动选择：

**NVLink Barrier**（ScaleUp 域内）— 基于 LSA 原子加 + 信号轮询，使用交替 +1/-1 避免 ABA 问题：

```cpp
// 每个 rank 向所有 peer 的信号位做原子加
if (thread_idx < kNumRanks) {
    const auto dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(
        workspace.get_nvl_barrier_signal_ptr(phase), thread_idx);
    ptx::red_add_rel_sys(dst_ptr, sign ? -1 : 1);
}
// 等待信号达到目标值 (kNumRanks 或 0)
timeout_while<kNumTimeoutCycles>(thread_idx == 0, [=](...) {
    return ptx::ld_acquire_sys<int>(
        workspace.get_nvl_barrier_signal_ptr(phase)) == target;
});
```

**GIN Barrier**（ScaleOut 跨节点）— 基于 GIN signal + GDAKI 信号表轮询：

```cpp
// 使用 GIN signal 通知所有 peer
for (int i = thread_idx; i < kNumRanks; i += kNumThreads)
    gin.signal(team, i, ncclGin_SignalInc{signal_idx});
// 轮询 GDAKI 信号表等待所有 peer 到达
timeout_while<kNumTimeoutCycles>([=](...) {
    return ptx::ld_acquire_sys<uint64_t>(signal_ptr) >= target;
});
```

**Hybrid Barrier** — SM 0 做 ScaleUp barrier，其余 SM 做 ScaleOut barrier，**并行执行**：

```cpp
if (sm_idx == 0) {
    scaleup_barrier_wo_local_sync<...>(...);   // NVLink
} else {
    scaleout_barrier_wo_local_sync<...>(...);  // GIN RDMA
}
```

### 2.6 关键 PTX 指令的使用

V2 大量使用 SM90 (Hopper) PTX 指令，封装在 `ptx.cuh` 中：

| PTX 指令 | 封装函数 | 用途 |
|----------|---------|------|
| `cp.async.bulk.shared::cluster.global` | `tma_load_1d` | TMA 从 HBM 异步加载到 SMEM |
| `cp.async.bulk.global.shared::cta` | `tma_store_1d` | TMA 从 SMEM 异步写入 HBM/NVLink |
| `cp.async.bulk.commit_group` | `tma_store_commit` | 提交 TMA store group |
| `cp.async.bulk.wait_group` | `tma_store_wait` | 等待 TMA store 完成 |
| `mbarrier.init` | `mbarrier_init_with_fence` | 初始化 mbarrier |
| `mbarrier.arrive.expect_tx` | `mbarrier_arrive_and_set_tx` | 设置 mbarrier 期望字节数 |
| `mbarrier.try_wait.parity` | `mbarrier_wait_and_flip_phase` | 等待 mbarrier 并翻转 phase |
| `elect.sync` | `elect_one_sync` | Warp 内选举单线程执行 |
| `cp.async.ca` | `cp_async_ca` | 传统 async copy（用于 SF） |
| `red.add.release.sys` | `red_add_rel_sys` | 系统级 release 语义原子加 |
| `ld.acquire.sys` | `ld_acquire_sys` | 系统级 acquire 语义加载 |

**代码佐证** — TMA load 使用 L2 cache hint 控制缓存策略（`ptx.cuh`）：

```cpp
__device__ void tma_load_1d(const void* dst_ptr, const void* src_ptr,
                            mbarrier* ptr, const int& num_bytes,
                            const TMACacheHint& hint = TMACacheHint::kEvictFirst) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        ".L2::cache_hint [%0], [%1], %2, [%3], %4;\n" :: ...);
}
```

Cache hint 设计选择：
- **TMA load 默认 `kEvictFirst`**：加载的数据即将被 TMA store 到远端，本地不再需要
- **TMA store 默认 `kEvictNormal`**：写入远端后对方会立即使用

---

## 阶段三：Dispatch 内核深入分析

Dispatch 是 MoE 通信的"正向路由"：将每个 rank 上的 token 按照 top-k gate 决策发送到目标 expert 所在的 rank。V2 的 dispatch 由三个阶段构成：**Prologue → Main Dispatch → Copy Epilogue**。

### 3.1 整体流水线架构

```
┌─────────────────────┐
│  Prologue            │  确定性路由统计 + slot 分配
│  (独立 kernel)       │  → 输出 dst_buffer_slot_idx
└─────────┬───────────┘
          │ cudaLaunchKernelEx (PDL)
┌─────────▼───────────┐
│  Main Dispatch       │  数据搬运：HBM → SMEM → NVLink/RDMA
│  Notify Warps        │  → 统计 rank/expert 计数 + 通知 peer
│  Dispatch Warps      │  → TMA load token + TMA store 到 peer buffer
└─────────┬───────────┘
          │ cudaTriggerProgrammaticLaunchCompletion (PDL)
┌─────────▼───────────┐
│  Copy Epilogue       │  从 recv buffer 重排到输出 tensor
│  (独立 kernel)       │  → 按 expert 排列 + 写 metadata
└─────────────────────┘
```

三个 kernel 通过 **CUDA Programmatic Dependent Launch (PDL)** 链接，main dispatch 完成后自动触发 epilogue，无需 CPU 干预。

**代码佐证** — main dispatch 结尾触发 epilogue（`dispatch.cuh`）：

```cpp
// Barrier to ensure data arrival
comm::gpu_barrier<kIsScaleupNVLink, ...>(gin, ...);

// Trigger the copy epilogue kernel (PDL)
cudaTriggerProgrammaticLaunchCompletion();
```

Epilogue 开头等待 dispatch 完成（`dispatch_copy_epilogue.cuh`）：

```cpp
// Will block until the main dispatch kernel has finished
cudaGridDependencySynchronize();
```

### 3.2 Prologue：确定性路由统计

当 `deterministic=True` 时，V2 使用单独的 prologue kernel 预计算 slot 分配，确保多次运行结果一致。

**核心算法**（`dispatch_deterministic_prologue.cuh`）：

1. **Per-warp 局部统计**：每个 warp 独立扫描其负责的 token 区间，对每个 token 的 top-k expert 做 rank 级去重统计

```cpp
// 每个 warp 扫描 token，统计每个 rank 的发送量
for (int i = start_token_idx; i < end_token_idx; i += kNumTokensPerGroup) {
    const auto expert_idx = __ldg(topk_idx + i * kNumTopk + lane_idx);
    const auto rank_idx = expert_idx / kNumExpertsPerRank;
    // Warp 内去重：同一 token 发往同一 rank 只计数一次
    const auto deduped_rank_idx = is_unique(rank_idx) ? rank_idx : -1;
    if ((rank_idx_mask >> lane_idx) == 1 and deduped_rank_idx >= 0)
        rank_count_warp_sum[deduped_rank_idx] += __popc(rank_idx_mask);
}
```

2. **Block 级聚合**：每个 SM 汇总所有 warp 的计数

```cpp
// Block 内各 warp 汇总
for (int rank_idx = thread_idx; rank_idx < kNumScaleupRanks; rank_idx += kNumThreads) {
    int rank_count_block_sum = 0;
    for (int i = 0; i < kNumWarps; i++)
        rank_count_block_sum += get_other_rank_count_warp_sum(i)[rank_idx];
    rank_count_buffer[sm_idx * kNumScaleupRanks + rank_idx] = rank_count_block_sum;
}
cooperative_groups::this_grid().sync();  // Grid 级同步
```

3. **全局前缀和 + slot 分配**：计算全局的每 rank 前缀和，然后第二遍扫描为每个 token 分配确定性 slot

```cpp
// 第二遍扫描：分配确定性 slot
const auto stored_dst_slot_idx = deduped_rank_idx >= 0 ?
    rank_count_warp_psum[deduped_rank_idx] + count_ones_before(rank_idx_mask, lane_idx) : -1;
dst_buffer_slot_idx[i * kNumTopk + lane_idx] =
    scaleup_rank_idx * kNumMaxTokensPerRank + stored_dst_slot_idx;
```

**关键技巧** — `ptx::match` + `ptx::deduplicate` 实现 warp 内高效去重：

```cpp
// match: 返回 warp 中所有值相同的 lane 的 bitmask
// deduplicate: 每组相同值只保留一个代表 lane
const auto deduped_rank_idx = is_unique(rank_idx) ? rank_idx : -1;
```

### 3.3 Main Dispatch：双角色 Warp 架构（Direct 模式）

Direct 模式（`num_scaleout_ranks == 1`）中，kernel 使用 **Notify Warps + Dispatch Warps** 两种角色：

```
┌──────────────────────────────────────────────┐
│ dispatch_impl kernel                          │
│                                               │
│  Notify Warps (warp 0..N-1):                 │
│    ├─ 统计 token → expert/rank 计数           │
│    ├─ SM 间归约 (notify_reduction_workspace)  │
│    ├─ SM0 汇总 + 通知 peer（put_value/put）   │
│    ├─ 等待 peer 回复计数                       │
│    └─ 计算前缀和写入 psum tensors              │
│                                               │
│  Dispatch Warps (warp N..M-1):               │
│    ├─ TMA load token 从 HBM → SMEM           │
│    ├─ 加载 topk_idx, topk_weights             │
│    ├─ 去重分配 slot (atomicAdd)               │
│    ├─ TMA store 到 peer recv buffer          │
│    │   ├─ NVLink: gin.get_sym_ptr → 直接写    │
│    │   └─ RDMA: TMA → send buffer → gin.put  │
│    └─ 循环处理所有 token                       │
└──────────────────────────────────────────────┘
```

**代码佐证** — Dispatch warps 的数据搬运（`dispatch.cuh`）：

```cpp
// 每个 token 的处理流程：
for (int token_idx = token_start; token_idx < num_tokens; token_idx += token_stride) {
    // 1. TMA load token 到 shared memory
    ptx::tma_load_1d(tma_buffer.get_hidden_ptr(),
        math::advance_ptr(x, token_idx * kNumHiddenBytes),
        mbarrier_ptr, kNumHiddenBytes);

    // 2. 加载 top-k metadata
    tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + ...);

    // 3. 等待 TMA load 完成
    ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
    ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);

    // 4a. NVLink: TMA store 到 peer 对称内存
    const auto dst_ptr = gin.get_sym_ptr<team_t>(
        recv_buffer.get_token_buffer(slot_idx).get_base_ptr(), dst_rank_idx);
    ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), num_bytes);

    // 4b. RDMA: 先写 send buffer，再 gin.put
    if (dst_ptr == nullptr) {
        ptx::tma_store_1d(send_buffer_ptr, tma_buffer.get_base_ptr(), num_bytes);
        ptx::tma_store_wait<1>();  // 等待 send buffer 写入完成
        gin.put<team_t>(recv_buffer.get_token_buffer(slot_idx).get_base_ptr(),
                        send_buffer_ptr, num_bytes, dst_rank_idx);
    }
}
```

### 3.4 Main Dispatch：三角色 Warp 架构（Hybrid 模式）

Hybrid 模式（`num_scaleout_ranks > 1`）中，kernel 使用**三种 warp 角色**：

```
┌──────────────────────────────────────────────────┐
│ hybrid_dispatch_impl kernel                       │
│                                                   │
│  Notify Warps:                                   │
│    ├─ 统计 rank/expert 计数                       │
│    ├─ 归约 → GIN put 通知 ScaleOut peers          │
│    ├─ 等待 ScaleOut 回复 → 归约到 ScaleUp peers    │
│    └─ 等待 ScaleUp 回复 → 计算前缀和              │
│                                                   │
│  ScaleOut Send Warps (channels):                 │
│    ├─ TMA load token → SMEM                       │
│    ├─ 判断 token 目标是否本 ScaleOut 域            │
│    │   ├─ 本域: 写 ScaleUp buffer (NVLink)        │
│    │   └─ 跨域: 写 ScaleOut send buffer (RDMA)    │
│    └─ 定期 signal → 通知 ForwardWarp              │
│                                                   │
│  Forward Warps:                                  │
│    ├─ 接收 ScaleOut recv buffer                   │
│    ├─ 轮询 channel tail 等待数据到达              │
│    └─ TMA store 到 ScaleUp peer (NVLink)          │
└──────────────────────────────────────────────────┘
```

**层次化通信的核心思路**：
1. **ScaleOut Send Warps** 负责跨节点方向的 RDMA 发送——将 token 从本节点写入远程节点的 recv buffer
2. **Forward Warps** 负责节点内方向的 NVLink 转发——将远程到达的 token 从 recv buffer 分发到目标 ScaleUp rank

这种**分离发送与转发**的设计使得 RDMA 和 NVLink 流量可以**流水线化**，不必等待 RDMA 全部完成再做 NVLink 分发。

**代码佐证** — ScaleOut 层次化通知（`hybrid_dispatch.cuh`）：

```cpp
// Notify warps: ScaleOut → ScaleUp 两级通知
// Step 1: GIN RDMA put 通知所有 ScaleOut peers
gin.put<ncclTeamTagRail>(
    workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_rank_idx),
    workspace_layout.get_scaleout_rank_count_ptr<true>(dst_scaleout_rank_idx),
    kNumScaleupRanks * sizeof(int), dst_scaleout_rank_idx);

// Step 2: 等待所有 ScaleOut peers 回复，汇总后通过 NVLink 通知 ScaleUp peers
const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {
    return workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_peer_idx, i);
});
gin.put_value<ncclTeamTagLsa>(
    workspace_layout.get_scaleup_rank_count_ptr<false>() + scaleup_rank_idx,
    counter, i);
```

### 3.5 Copy Epilogue：从 Buffer 重排到输出 Tensor

Epilogue 将 recv buffer 中按 rank 排列的 token 重排到输出 tensor 中。支持两种模式：

- **Non-expand 模式**（`do_expand=false`）：输出 tensor 按 token 顺序排列，每个 token 位置对应其收到的 slot
- **Expand 模式**（`do_expand=true`）：输出 tensor 按 expert 排列，每个 expert 的 token 连续存放，通过 `psum_num_recv_tokens_per_expert` 分配位置

**代码佐证** — Epilogue 核心逻辑（`dispatch_copy_epilogue.cuh`）：

```cpp
for (int i = global_warp_idx; i < num_recv_tokens; ...) {
    // 1. TMA load 从 recv buffer 读取 token（含数据+元数据）
    ptx::tma_load_1d(tma_buffer.get_base_ptr(), buffer_token.get_base_ptr(),
                     mbarrier_ptr, tma_buffer.get_num_bytes<false>());

    // 2. 验证 expert 索引范围
    const auto in_range = expert_start_idx <= dst_expert_idx and dst_expert_idx < expert_end_idx;
    dst_expert_idx = in_range ? dst_expert_idx - expert_start_idx : -1;

    // 3. 分配输出位置
    if (not kDoExpand) {
        dst_tensor_idx = i;  // Non-expand: 直接按序
    } else if (dst_expert_idx >= 0) {
        dst_tensor_idx = atomicAdd(psum_num_recv_tokens_per_expert + dst_expert_idx, 1);
    }

    // 4. TMA store 到输出 tensor
    ptx::tma_store_1d(
        math::advance_ptr(recv_x, dst_tensor_idx * kNumHiddenBytes),
        tma_buffer.get_hidden_ptr(), kNumHiddenBytes);

    // 5. 写源 metadata（用于 combine 反向路由）
    recv_src_metadata[i * kMetadataStride + 0] = *tma_buffer.get_src_token_global_idx_ptr();
    recv_src_metadata[i * kMetadataStride + 1] = current_rank_idx * kNumTopk + master_src_topk_idx;
}
```

### 3.6 Token Buffer 布局

每个 token 在 buffer 中的布局由 `TokenLayout` 定义（`layout.cuh`），所有字段 32 字节对齐（满足 TMA `LDG.256` 要求）：

```
┌──────────────────────────┐
│ Hidden Data              │  kNumHiddenBytes (BF16/FP8)，32B 对齐
├──────────────────────────┤
│ Scale Factors (SF)       │  kNumSFPacks * 4 bytes (仅 FP8)，32B 对齐
├──────────────────────────┤
│ Metadata                 │  32B 对齐
│  ├─ topk_idx[num_topk]   │  int32 × num_topk
│  ├─ topk_weights[num_topk]│  float × num_topk
│  ├─ src_token_global_idx │  int32（来源 rank*max_tokens+token_idx）
│  └─ linked_list_idx[]    │  int32 × num_topk（hybrid 模式）
├──────────────────────────┤
│ mbarrier (仅 SMEM)       │  8 bytes，32B 对齐
└──────────────────────────┘
```

**代码佐证** — `TokenLayout` 的 metadata 大小计算：

```cpp
// Metadata = topk_idx + topk_weights + (src_idx + linked_list)
num_metadata_bytes = num_topk * (sizeof(int) + sizeof(float)) +
                     (with_metadata ? (1 + num_topk) * sizeof(int) : 0);
```

整个 dispatch 的 buffer 内存布局：

```
[Data Buffer]
├── recv_buffer: [num_ranks × num_max_tokens_per_rank] × TokenLayout
└── send_buffer: [1 × num_max_tokens_per_rank] × TokenLayout
```

---

## 阶段四：Combine 内核深入分析

Combine 是 Dispatch 的逆操作：将 expert 计算后的结果从目标 rank 路由回原始 rank，并按 `topk_weights` 加权归约。V2 的 combine 同样分为 **Main Combine → Reduce Epilogue** 两阶段（通过 PDL 链接）。

### 4.1 Combine 与 Dispatch 的对称性

Dispatch 和 Combine 的关系是 **正反向对称**：

| 维度 | Dispatch | Combine |
|------|----------|---------|
| 数据方向 | 原始 rank → expert rank | expert rank → 原始 rank |
| 路由信息 | gate 决定的 topk_idx | dispatch 生成的 `recv_src_metadata` |
| 数据内容 | token embedding (BF16/FP8) | expert 输出 (BF16) |
| 归约操作 | 无（纯路由） | 按 topk_weights 加权求和 |
| Buffer 复用 | 写入 peer 的 recv buffer | 读取 dispatch 写入的 metadata，写入 peer 的 combine recv buffer |

**代码佐证** — 训练中 dispatch 的反向就是 combine（`README.md` 示例）：

```python
def dispatch_backward(grad_recv_x, grad_recv_topk_weights, handle):
    """The backward pass of MoE dispatch is actually a combine."""
    combined_grad_x, combined_grad_topk_weights, event = _buffer.combine(
        grad_recv_x, handle=handle, topk_weights=grad_recv_topk_weights, ...)
```

### 4.2 Combine 主内核（Direct 模式）

Direct 模式的 combine（`combine.cuh`）**没有 notify/dispatch 角色分离**，每个 warp 独立处理 token 子集：

```
┌──────────────────────────────────────────────┐
│ combine_impl kernel                           │
│  All Warps 均匀分配 token:                    │
│   1. Barrier 等待 expert 计算完成              │
│   2. 读取 src_metadata 获取源 rank/token 信息  │
│   3. 根据 expand/reduce 模式选择策略:          │
│      a. 无归约 → TMA load + TMA store 直传    │
│      b. 本地归约 → 多 slot 读取 + BF16 加法    │
│      c. 展开发送 → 逐 top-k 独立发送           │
│   4. NVLink → TMA store 到 peer 对称内存       │
│      RDMA → TMA store 到 send buffer → gin.put│
│   5. Barrier 等待所有数据到达                  │
└──────────────────────────────────────────────┘
```

**代码佐证** — Combine 的三种处理路径（`combine.cuh`）：

**路径 A：无归约（最快路径）** — 当 token 只有一个有效 top-k slot 时，直接 TMA 搬运：

```cpp
auto no_local_reduce = not kUseExpandedLayout or
    (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);
if (no_local_reduce) {
    if (ptx::elect_one_sync()) {
        ptx::tma_store_wait();
        ptx::tma_load_1d(tma_buffer, load_ptr, mbarrier_ptr, kNumHiddenBytes);
        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
        ptx::tma_store_1d(master_token_buffer, tma_buffer, kNumHiddenBytes);
        ptx::tma_store_commit();
    }
}
```

**路径 B：本地归约** — 当 `allow_multiple_reduction=true` 且有多个 top-k slot 指向同一 token 时，先在 SMEM 做 BF16 加法归约：

```cpp
else if constexpr (kAllowMultipleReduction) {
    // 排序有效 top-k 到前面
    compute_topk_slots(topk_slot_idx, reduce_valid_mask, ...);
    // 多 slot 归约到 shared memory
    combine_reduce<kHiddenVec, kUnrollFactor, ...>(
        lane_idx, topk_slot_idx, tma_buffer_ptr,
        get_src_buffer_ptr_func, wait_buffer_func);
    // 归约结果 TMA store 到 peer
    ptx::tma_store_1d(master_token_buffer, tma_buffer, kNumHiddenBytes);
}
```

**路径 C：展开发送** — 当 `allow_multiple_reduction=false` 时，不做本地归约，将每个 top-k 的结果分别发送到原始 rank，由 Reduce Epilogue 统一归约：

```cpp
else {
    // 逐 top-k 独立发送
    for (int k = 0; k < kNumTopk; ++ k) {
        // TMA load → TMA store (NVLink) 或 TMA load → send buffer → gin.put (RDMA)
    }
}
```

### 4.3 归约核心：`combine_reduce` 函数

`combine_reduce`（`combine_utils.cuh`）是 combine 的性能核心，实现多 top-k slot 的向量化归约：

**关键优化 — BF16 bypass**：当 top-k ≤ 2 且无 bias 时，直接使用 `nv_bfloat162` 加法（无需 FP32 累加），避免类型转换开销：

```cpp
const bool enable_hadd_bypass =
    (bias_0 == nullptr and bias_1 == nullptr) and
    (kNumValidTopk <= 2 or topk_slot_idx[2] < 0);

if (enable_hadd_bypass) {
    // 直接 BF16 加法：读 slot_0 + slot_1，BF162 累加
    vec_t values_0[kUnrollFactor], values_1[kUnrollFactor];
    // ... load with predicate ...
    // BF16x2 向量加法
    const auto bf162_view_0 = reinterpret_cast<nv_bfloat162*>(values_0);
    const auto bf162_view_1 = reinterpret_cast<nv_bfloat162*>(values_1);
    for (int j = ...; ...)
        bf162_view_0[j] += bf162_view_1[j];
    dst_buffer_ptr[...] = values_0[j];
}
```

当 top-k > 2 或有 bias 时，使用 FP32 累加再转换回 BF16：

```cpp
else {
    float2 reduced[...] = {};
    for (int k = 0; k < kNumValidTopk; ++ k) {
        vec_t values[kUnrollFactor];
        // load with predicate
        ptx::accumulate(reduced[j], bf162_view[j]);  // BF16 → FP32 累加
    }
    // FP32 → BF16 转换后写入
}
```

**向量化策略** — SM100 (Blackwell) 上使用 `longlong4_t`（32 bytes），SM90 (Hopper) 上使用 `int4`（16 bytes）：

```cpp
template <int kHiddenBytes>
struct CombineVecTraits {
#if __CUDA_ARCH__ >= 1000
    // SM100+: 如果 hidden 对齐 32 字节，使用 LDG.256
    static constexpr bool kUseLonglong4 = (kHiddenBytes % sizeof(longlong4_t) == 0);
    using vec_t = std::conditional_t<kUseLonglong4, longlong4_t, int4>;
#else
    using vec_t = int4;  // SM90: LDG.128
#endif
};
```

**带谓词的 load** — 当 slot 无效时（`topk_slot_idx < 0`），使用 PTX 谓词 load 自动返回 0：

```cpp
values[j] = ptx::ldg_with_gez_pred(
    src_base_ptr + offset, topk_slot_idx[k]);
// PTX 实现:
// setp.ge.s32 p, slot_idx, 0;
// @p ld.global.nc.v4.s32 {ret}, [ptr], cache_hint;
```

### 4.4 Reduce Epilogue

当 `allow_multiple_reduction=false` 时（即 combine 不做本地归约），需要一个独立的 Reduce Epilogue kernel 来完成最终归约。

**代码佐证** — `combine_reduce_epilogue.cuh` 核心逻辑：

```cpp
// 等待 combine 主内核完成（PDL）
cudaGridDependencySynchronize();

for (int token_idx = global_warp_idx; token_idx < num_combined_tokens; ...) {
    // 1. 从 topk_idx 确定每个 top-k 来自哪个 rank
    stored_dst_rank_idx = stored_dst_expert_idx / kNumExpertsPerRank;

    // 2. 去重 + 排序有效 slot
    compute_topk_slots(topk_slot_idx, reduce_valid_mask, ...);

    // 3. 多路归约到 SMEM
    combine_reduce<kHiddenVec, kUnrollFactor, kNumTokensInLayout>(
        lane_idx, topk_slot_idx, tma_buffer_ptr,
        /* 从各 rank 的 buffer 读取 */ [=](const int& slot_idx) {
            return comm_buffer.get_rank_buffer(slot_idx)
                              .get_token_buffer(token_idx).get_base_ptr();
        },
        wait_buffer_func,
        bias_0, bias_1);

    // 4. TMA store 归约结果到输出 tensor
    ptx::tma_store_1d(output_buffer.get_token_buffer(token_idx).get_base_ptr(),
                      tma_buffer.get_base_ptr(), kNumHiddenBytes);

    // 5. 写 topk_weights
    combined_topk_weights[token_idx * kNumTopk + lane_idx] = value;
}
```

### 4.5 Hybrid Combine（ScaleOut 模式）

Hybrid 模式的 combine 使用 **ScaleUp Warps + Forward Warps** 两种角色，与 dispatch 的 hybrid 模式对称：

```
┌──────────────────────────────────────────────────┐
│ hybrid_combine_impl kernel                        │
│                                                   │
│  ScaleUp Warps (channels):                       │
│    ├─ 遍历 channel 的 linked list                 │
│    ├─ 对每个 token：                              │
│    │   ├─ 读 src_metadata 确定源 rank             │
│    │   ├─ NVLink 可达: TMA load 本地 → store 远端 │
│    │   └─ RDMA 不可达: 写 ScaleOut send buffer    │
│    └─ 定期 st.release.sys 更新 tail 通知 Forward  │
│                                                   │
│  Forward Warps:                                  │
│    ├─ 接收 ScaleOut send buffer 的数据            │
│    ├─ 轮询 ScaleOut channel tail 等待到达         │
│    ├─ 做本地归约或直传                            │
│    └─ RDMA put 到原始 rank                        │
└──────────────────────────────────────────────────┘
```

**关键优化 — 寄存器调配**：Hybrid combine 中 ScaleUp warps 是带宽受限（等待 NVLink tail 信号），Forward warps 是计算密集（需要做归约）。V2 通过 PTX `warpgroup_reg_dealloc`/`warpgroup_reg_alloc` 在同一 kernel 内为不同角色分配不同的寄存器数量：

```cpp
if (warp_idx < kNumScaleupWarps) {
    if constexpr (kAdjustRegisters)
        ptx::warpgroup_reg_dealloc<kNumRegistersForScaleupWarps>();  // 40 regs
    // ... ScaleUp warp logic (轻量，主要是 TMA)
} else {
    if constexpr (kAdjustRegisters)
        ptx::warpgroup_reg_alloc<kNumRegistersForForwardWarps>();    // 216 regs
    // ... Forward warp logic (重量，需要做归约)
}
```

**ScaleUp Warps 的 tail 通知机制**：ScaleUp warps 通过 `st.release.sys` 定期更新 NVLink 对称内存中的 tail 指针，Forward warps 通过 `ld.acquire.sys` 轮询，实现**无 barrier 的流水线通信**：

```cpp
// ScaleUp warp: 定期更新 tail（每 kNumScaleupUpdateInterval=3 个 token）
const auto update_tails = [&](const bool& finish = false) {
    if (finish or update_counter == kNumScaleupUpdateInterval) {
        ptx::tma_store_wait();
        __syncwarp();
        // 向所有 ScaleUp peer 写入当前发送进度
        ptx::st_release_sys(
            gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j),
            stored_num_tokens_sent[i]);
    }
};
```

### 4.6 `allow_multiple_reduction` 的设计权衡

这个配置控制了 combine 的精度与通信量之间的取舍：

| 配置 | 精度 | 通信量 | 适用场景 |
|------|------|--------|---------|
| `allow_multiple_reduction=True`（默认） | 略低（多次 BF16 加法） | 少（先本地归约再发送） | 训练（可容忍微小精度损失） |
| `allow_multiple_reduction=False` | 最高（仅一次 FP32 归约） | 多（每个 top-k 独立发送） | 推理（需要精确结果） |

**代码佐证** — combine buffer 布局根据此选项变化：

```cpp
// allow_multiple_reduction=True: recv buffer 按 rank 分区
const auto recv_buffer = BufferLayout<false>(
    token_layout, kNumTokensInLayout, kNumMaxTokensPerRank, buffer);
// kNumTokensInLayout = min(kNumRanks, kNumTopk) — 按 rank 去重

// allow_multiple_reduction=False: recv buffer 按 top-k 分区
// kNumTokensInLayout = kNumTopk — 每个 top-k 独立一个 buffer slot
```

### 4.7 Combine 的 Buffer 布局

```
[Data Buffer — Combine]
├── recv_buffer: [kNumTokensInLayout × kNumMaxTokensPerRank] × TokenLayout(hidden, 0, topk, false)
│   └── kNumTokensInLayout = min(num_ranks, num_topk) 或 num_topk
└── send_buffer: [num_ranks × kNumMaxTokensPerRank × (expand ? topk : 1)] × TokenLayout
```

注意与 dispatch 的 `TokenLayout` 的区别：
- Combine 的 token 不携带 SF（scale factors），因为 combine 操作始终在 BF16 下
- Combine 的 token 不携带 `src_token_global_idx`（combine 是反向操作，不需要再记录来源）
- 但携带 `topk_weights`（用于反向时传回梯度）

---

## 总结：V2 核心设计理念

| 设计理念 | 体现 |
|---------|------|
| **统一接口** | `ElasticBuffer` 统一 HT/LL 模式，统一 NVLink/RDMA 路径 |
| **解析式调优** | `get_theoretical_num_sms` 替代 auto-tuning，SM 占用降至 4-6 |
| **层次化通信** | ScaleUp (NVLink) + ScaleOut (RDMA) 分离，Hybrid 三角色 warp |
| **JIT 编译** | 运行时生成最优 kernel，避免模板参数组合爆炸 |
| **TMA 驱动** | 全链路 TMA (cp.async.bulk) 数据搬运，mbarrier 同步 |
| **GPU-initiated** | 所有通信由 GPU 侧 GIN API 发起，无 CPU 干预 |
| **PDL 流水线** | Prologue → Dispatch → Epilogue 通过 PDL 自动链接 |
| **正反向对称** | Dispatch/Combine 共享 buffer 结构和通信原语 |
| **精度可控** | `allow_multiple_reduction` 在精度和通信量间权衡 |
| **NCCL 生态** | 复用 NCCL comm、拓扑检测、QP 管理，降低集成成本 |
