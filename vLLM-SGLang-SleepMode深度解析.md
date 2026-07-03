# vLLM 与 SGLang Sleep Mode 深度解析

> **日期**：2026-07-02
> **目标读者**：推理框架开发团队
> **内容范围**：vLLM 和 SGLang 中 Sleep Mode 的架构设计、实现原理、调用链路、关键源码解析、对比分析

---

## 目录

1. [背景与动机](#一背景与动机)
2. [CUDA 虚拟内存管理基础](#二cuda-虚拟内存管理基础)
3. [vLLM Sleep Mode](#三vllm-sleep-mode)
   - 3.1 架构总览
   - 3.2 Tagged Memory Pool
   - 3.3 Sleep Level 1 vs Level 2
   - 3.4 CuMemAllocator 实现
   - 3.5 可插拔 Backend 抽象
   - 3.6 完整调用链路
   - 3.7 关键 Bugfix
4. [SGLang Memory Saver](#四sglang-memory-saver)
   - 4.1 架构总览
   - 4.2 torch_memory_saver 核心实现
   - 4.3 Region 机制
   - 4.4 CUDA Graph 兼容性设计
   - 4.5 HiCache 三级 KV Cache 体系
   - 4.6 Layerwise Offload（扩散模型）
   - 4.7 完整调用链路
5. [两者对比分析](#五两者对比分析)
6. [最佳实践与注意事项](#六最佳实践与注意事项)
7. [参考来源](#七参考来源)

---

## 一、背景与动机

### 1.1 为什么需要 Sleep Mode？

在以下场景中，推理引擎需要在**不重启进程**的前提下释放 GPU 显存：

| 场景 | 说明 |
|------|------|
| **RLHF/强化学习训练** | 训练（Training）与推理（Rollout）共享 GPU，需要在训练阶段释放推理显存 |
| **动态模型切换** | 在同一进程中切换不同模型，需要卸载当前模型权重 |
| **多租户 GPU 共享** | 多个推理服务共享 GPU，空闲服务应释放显存给其他服务 |
| **成本优化** | 推理服务空闲时释放显存，允许调度其他计算任务 |

核心矛盾：**进程不重启 + GPU 显存必须释放 + CUDA Graph 不能重建（重建代价高）**。

### 1.2 核心设计目标

```
目标 1: 释放 GPU 物理显存（归还给 CUDA Driver/OS）
目标 2: 保留虚拟地址空间（指针不变，CUDA Graph 不失效）
目标 3: 恢复时重新映射物理显存到原虚拟地址
目标 4: 支持选择性释放（只释放 weights，保留 kv_cache 等）
```

---

## 二、CUDA 虚拟内存管理基础

### 2.1 关键 API

两个框架都基于 CUDA Virtual Memory Management (VMM) API 实现 Sleep Mode：

| CUDA API | 作用 | Sleep 阶段使用 |
|----------|------|---------------|
| `cuMemAddressReserve` | 预留一段虚拟地址空间（不分配物理内存） | 初始化时调用 |
| `cuMemCreate` | 分配物理显存，返回句柄 | 初始化/恢复时调用 |
| `cuMemMap` | 将物理显存映射到虚拟地址 | 初始化/恢复时调用 |
| `cuMemSetAccess` | 设置虚拟地址范围的可访问性 | 初始化/恢复时调用 |
| `cuMemUnmap` | 解除物理显存与虚拟地址的映射 | **Sleep 时调用** |
| `cuMemRelease` | 释放物理显存句柄 | **Sleep 时调用** |
| `cuMemAddressFree` | 释放虚拟地址空间 | 仅在彻底释放时调用 |

### 2.2 核心原理

```
┌─────────────────────────────────────────────────────┐
│                    进程虚拟地址空间                     │
│                                                     │
│  0x7f0000000000 ─────────── 0x7f0100000000           │
│  │ 保留的虚拟地址区域（永不释放）         │            │
│  │   ↕ cuMemMap / cuMemUnmap              │            │
│  └───────────┬───────────────────────────┘            │
│              │                                        │
│  ┌───────────▼───────────────────────────┐            │
│  │        物理 GPU 显存                    │            │
│  │                                        │            │
│  │  Sleep 前: [████████████████████████]   │            │
│  │  Sleep 后: [                        ]  │ ← 已释放   │
│  │  Wake 后:  [████████████████████████]   │ ← 新物理页 │
│  └────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────┘

要点：虚拟地址不变 → 所有 C++/Python 指针不变 → CUDA Graph 地址不变
```

---

## 三、vLLM Sleep Mode

### 3.1 架构总览

```
                    HTTP API Layer
                    ┌──────────────┐
                    │  POST /sleep  │    POST /wake_up   GET /is_sleeping
                    └──────┬───────┘
                           │ RPC
                    ┌──────▼───────┐
                    │   Executor    │  (MultiprocExecutor / RayExecutor)
                    └──────┬───────┘
                           │ broadcast to all DP/TP/PP workers
                    ┌──────▼───────┐
                    │  GPUWorker   │
                    │  sleep()     │   ← vllm/v1/worker/gpu_worker.py
                    │  wake_up()   │
                    └──────┬───────┘
                           │
                    ┌──────▼──────────────────────┐
                    │  SleepModeBackendFactory     │  ← vllm/device_allocator/sleep_mode_backend.py
                    │  get_backend("cumem")        │
                    └──────┬──────────────────────┘
                           │
                    ┌──────▼────────┐
                    │  CuMemBackend  │  (默认实现)
                    │  suspend()     │
                    │  resume()      │
                    └──────┬────────┘
                           │
                    ┌──────▼──────────────────┐
                    │  CuMemAllocator (单例)   │  ← vllm/device_allocator/cumem.py
                    │  sleep(offload_tags)    │
                    │  wake_up(tags)          │
                    │  use_memory_pool(tag)   │
                    └──────┬──────────────────┘
                           │
                    ┌──────▼──────────────────┐
                    │  C Extension             │  ← csrc/cumem_allocator.cpp
                    │  cuMemCreate/Map/Unmap   │
                    │  cuMemRelease            │
                    │  python_malloc_callback  │
                    └─────────────────────────┘
```

### 3.2 Tagged Memory Pool

vLLM 使用 **标签化的内存池** 管理所有 GPU 内存分配。当 `enable_sleep_mode=True` 时，所有分配都带上一个 tag。

#### 内存标签体系

| Tag | 包含内容 | 初始化时机 | 可独立 Sleep | 可独立 Wake |
|-----|---------|-----------|:---:|:---:|
| `"weights"` | 模型权重参数、buffers | `load_model()` | ✅ | ✅ |
| `"kv_cache"` | KV cache 张量（仅 tensor 分配，非实际填充） | `initialize_kv_cache_tensors()` | ✅ | ✅ |
| `"graphs"` | CUDA Graph 执行图 + NCCL 通信器内存 | `capture_model()` | ✅ | ✅ |
| `"default"` | 其他未分类的临时分配 | 各处 | — | — |

#### 代码：初始化时的 Tag 使用

```python
# vllm/v1/worker/gpu_worker.py

class GPUWorker:
    def load_model(self):
        allocator = CuMemAllocator.get_instance()

        # ── 模型权重：tag "weights" ──
        with allocator.use_memory_pool(tag="weights"):
            self.model_runner.load_model()

        # ── CUDA Graphs：tag "graphs" (PR #45623) ──
        self._pin_sleep_mode_graph_pool()
        with allocator.use_memory_pool(tag="graphs"):
            self.model_runner.capture_model()

    def initialize_kv_cache(self, kv_cache_config):
        allocator = CuMemAllocator.get_instance()

        # ── KV Cache：tag "kv_cache" (PR #35956: 缩小作用域) ──
        with allocator.use_memory_pool(tag="kv_cache"):
            self.model_runner.initialize_kv_cache_tensors()  # 仅 tensor 分配
```

#### 代码：use_memory_pool 上下文管理器

```python
# vllm/device_allocator/cumem.py

class CuMemAllocator:
    def __init__(self):
        self._current_tag = "default"
        self.pointer_to_data: dict[int, _AllocationData] = {}
        # _AllocationData 包含: handle, tag, cpu_backup_tensor

    @contextmanager
    def use_memory_pool(self, tag: str = "default"):
        # 设置 C 扩展层的全局分配器回调函数指针
        set_python_allocator(self.python_alloc_callback)
        old_tag = self._current_tag
        self._current_tag = tag
        try:
            yield
        finally:
            self._current_tag = old_tag
            unset_python_allocator()
```

### 3.3 Sleep Level 1 vs Level 2

> **⚠️ 源码核实结论**：经过对 GitHub 上 vLLM 主分支 `cumem.py` 实际源码（commit `038e9be`）的确认，
> `offload_tags=None` **不是** match-all 哨兵。真实逻辑见下方代码。

#### 关键源码：CuMemAllocator.sleep() 中 None 的真实处理

```python
# vllm/device_allocator/cumem.py — 实际源码 (commit 038e9be)

def sleep(self, offload_tags=None):
    # ── None 的真实处理：转为 default tag ──
    if offload_tags is None:
        # "by default, allocated tensors are offloaded when the allocator sleeps"
        offload_tags = (CuMemAllocator.default_tag,)  # → ("default",)
    elif isinstance(offload_tags, str):
        offload_tags = (offload_tags,)

    for ptr, data in self.pointer_to_data.items():
        # ── 仅当 tag 在 offload_tags 中才备份, 否则丢弃 ──
        if data.tag in offload_tags:
            # ✅ 备份路径: GPU → CPU
            cpu_tensor = torch.empty(size, pin_memory=True)
            cudaMemcpy(cpu_tensor.data_ptr(), ptr, size)  # D2H
            data.cpu_backup_tensor = cpu_tensor
        # else:
        #   ❌ 丢弃路径: 无 CPU 备份

        # ⚡ 始终释放 GPU 物理内存
        unmap_and_release(handle)  # cuMemUnmap + cuMemRelease
```

#### Level 1 vs Level 2 的真实语义

`CuMemBackend.suspend()` 调用：
```python
# Level 1 → offload_tags=("weights",)
# Level 2 → offload_tags=None → sleep() 内部转为 ("default",)
```

| | Level 1 | Level 2 |
|---|---|---|
| **传入 offload_tags** | `("weights",)` | `None` |
| **sleep() 内转换后** | `("weights",)` | **`("default",)`** |
| **weights tag** | ✅ `"weights" in ("weights",)` → **备份 CPU** | ❌ `"weights" in ("default",)` → **丢弃** |
| **kv_cache tag** | ❌ 丢弃 | ❌ 丢弃 |
| **graphs tag** | ❌ 丢弃 | ❌ 丢弃 |
| **default tag** | ❌ 丢弃 | ✅ 备份 CPU |

#### 设计意图

```
Level 1: 备份 weights → 恢复时 weights 从 CPU 拷贝回 GPU
         → 适用场景: 同模型复用 (RL rollout)
         → KV Cache 丢弃（反正 rollouts 之间 KV Cache 内容过期）

Level 2: 丢弃 weights（不备份！）→ 恢复时 weights 需重新加载
         → 适用场景: 模型切换 / RLHF 权重更新
         → 旧 weights 不需要保留（训练已更新），节省 CPU RAM
         → GPUWorker 层额外处理 (PR #16889 / #20735):
            保存 model buffers (非参数持久张量) 到 _sleep_saved_buffers
            使用 backup_memory_except() 备份非 pool 内的内存
```

#### 代码行程对比

```
Level 1: offload_tags=("weights",)
  ┌─────────────────────────────────────────────────────┐
  │ ptr_1: tag="weights", size=14GB                     │
  │   → "weights" in ("weights",) → True                │
  │   → ✅ GPU→CPU copy (14 GB D2H)                     │
  │   → cuMemRelease (GPU 物理内存释放)                   │
  │                                                     │
  │ ptr_2: tag="kv_cache", size=2GB                     │
  │   → "kv_cache" in ("weights",) → False              │
  │   → ❌ 无 CPU 备份 → 直接丢弃                         │
  │   → cuMemRelease                                    │
  │                                                     │
  │ ptr_3: tag="graphs", size=0.5GB                     │
  │   → "graphs" in ("weights",) → False                │
  │   → ❌ 无 CPU 备份 → 直接丢弃                         │
  │   → cuMemRelease                                    │
  │                                                     │
  │ ptr_4: tag="default", size=0.2GB                    │
  │   → "default" in ("weights",) → False               │
  │   → ❌ 无 CPU 备份 → 直接丢弃                         │
  │   → cuMemRelease                                    │
  │                                                     │
  │ 结果: CPU 占 14GB (仅 weights), GPU 释放 ~87%        │
  └─────────────────────────────────────────────────────┘

Level 2: offload_tags=None → sleep() 内转为 ("default",)
  ┌─────────────────────────────────────────────────────┐
  │ ptr_1: tag="weights", size=14GB                     │
  │   → "weights" in ("default",) → False               │
  │   → ❌ 无 CPU 备份 → WEIGHTS 被丢弃!                  │
  │   → cuMemRelease                                    │
  │                                                     │
  │ ptr_2: tag="kv_cache", size=2GB                     │
  │   → "kv_cache" in ("default",) → False              │
  │   → ❌ 无 CPU 备份 → 直接丢弃                         │
  │   → cuMemRelease                                    │
  │                                                     │
  │ ptr_3: tag="graphs", size=0.5GB                     │
  │   → "graphs" in ("default",) → False                │
  │   → ❌ 无 CPU 备份 → 直接丢弃                         │
  │   → cuMemRelease                                    │
  │                                                     │
  │ ptr_4: tag="default", size=0.2GB                    │
  │   → "default" in ("default",) → True                │
  │   → ✅ GPU→CPU copy (0.2 GB D2H)                    │
  │   → cuMemRelease                                    │
  │                                                     │
  │ GPUWorker 额外: backup_memory_except()               │
  │   备份非 pool 内的 model buffers 到 CPU               │
  │                                                     │
  │ 结果: CPU 占 ~0.5GB (default + buffers),             │
  │       GPU 释放 ~97%, weights 完全丢弃                 │
  └─────────────────────────────────────────────────────┘
```

#### 代码：CuMemBackend.suspend() → 确定 offload_tags

```python
# vllm/device_allocator/sleep_mode_backend.py — CuMemBackend

class CuMemBackend(SleepModeBackend):
    def suspend(self, level: int = 1) -> None:
        # Level 1 → offload_tags=("weights",) → sleep() 内仅 weights tag 匹配
        # Level 2 → offload_tags=None        → sleep() 内转为 ("default",) → 仅 default tag 匹配
        offload_tags = ("weights",) if level == 1 else None
        CuMemAllocator.get_instance().sleep(offload_tags)
```

#### 代码：GPUWorker.sleep() 的 Level 分发

```python
# vllm/v1/worker/gpu_worker.py

def sleep(self, level: int = 1) -> None:
    """将 worker 置于睡眠状态，释放 GPU 显存"""
    backend = SleepModeBackendFactory.get_backend(
        self.vllm_config.model_config.sleep_mode_backend  # 默认 "cumem"
    )

    free_bytes_before = torch.cuda.mem_get_info()[0]

    if level == 1:
        # Level 1: offload_tags=("weights",) → 备份权重, 丢弃 kv_cache/graphs
        backend.suspend(level=1)
    elif level == 2:
        # Level 2: offload_tags=None → sleep() 内转为 ("default",)
        #          → weights/kv_cache/graphs 全部丢弃不备份
        #          → 仅 "default" tag 的杂项分配备份
        #          GPUWorker 额外:
        #            - backup_memory_except() 备份非 pool 内存 (PR #20735)
        #            - 保存 model buffers 到 _sleep_saved_buffers (PR #16889)
        backend.suspend(level=2)

    free_bytes_after, total = torch.cuda.mem_get_info()
    freed_gb = (total - free_bytes_after - (total - free_bytes_before)) / (1024**3)
    logger.info("Sleep mode freed %.2f GiB GPU memory", freed_gb)
```

### 3.4 CuMemAllocator 实现

这是 vLLM Sleep Mode 的核心实现类。

#### 数据结构

```python
# vllm/device_allocator/cumem.py

@dataclass
class _AllocationData:
    handle: tuple         # CUDA 句柄, 来自 C 扩展
    tag: str              # 标签, 如 "weights" / "kv_cache" / "graphs"
    cpu_backup_tensor: Optional[torch.Tensor] = None  # CPU 端备份

class CuMemAllocator:
    _instance: Optional["CuMemAllocator"] = None  # 单例

    def __init__(self):
        self.pointer_to_data: dict[int, _AllocationData] = {}
        self._current_tag = "default"
```

#### sleep() — 释放显存（源码确认版）

```python
def sleep(self, offload_tags: Optional[tuple[str, ...]] = None) -> None:
    """
    释放 GPU 物理显存。根据 offload_tags 决定是否先做 CPU 备份。

    参数语义（关键！）:
      offload_tags=None    → sleep() 内部转为 ("default,"), 仅 "default" tag 备份 (Level 2)
      offload_tags=("weights",) → 仅 "weights" tag 做 CPU 备份, 其余丢弃 (Level 1)

    流程:
      if data.tag in offload_tags:
          # GPU to CPU copy (pinned memory), save cpu_backup_tensor
      else:
          # NO backup, data discarded

      无论是否备份 → unmap_and_release(handle) → GPU 物理内存释放
    """

    # === Step 1: 同步所有 in-flight CUDA 操作 (PR #45552) ===
    torch.cuda.synchronize()

    # === Step 2: 静默所有 NCCL 通信组 (PR #45554) ===
    self._quiesce_distributed_before_vmm_mutation()

    total_bytes = 0
    backup_bytes = 0

    for ptr, data in self.pointer_to_data.items():
        handle = data.handle
        total_bytes += handle[1]  # handle[1] = size

        # Step 1: None is converted to ("default",) — NOT match-all!
        if offload_tags is None:
            offload_tags = (CuMemAllocator.default_tag,)  # ("default",)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)

        # Step 2: Sync all in-flight CUDA (PR #45552)
        torch.cuda.synchronize()

        # Step 3: Quiesce NCCL (PR #45554)
        self._quiesce_distributed_before_vmm_mutation()

        total_bytes = 0
        backup_bytes = 0

        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            total_bytes += handle[1]

            # Key logic: only tags in offload_tags get CPU backup
            # Level 1: offload_tags=("weights",) -> "weights" matches -> backup
            # Level 2: offload_tags=("default",) -> "weights"/"kv_cache"/"graphs" don't match -> DISCARDED!

        if data.tag in offload_tags:
            # Backup path: GPU -> CPU pinned memory
            backup_bytes += handle[1]
            cpu_tensor = torch.empty(
                handle[1], dtype=torch.uint8, device="cpu",
                pin_memory=is_pin_memory_available()
            )
            libcudart.cudaMemcpy(cpu_tensor.data_ptr(), ptr, handle[1])  # D2H
            data.cpu_backup_tensor = cpu_tensor
        # else: NO CPU backup, data discarded
        #   Level 1: kv_cache/graphs discarded
        #   Level 2: weights/kv_cache/graphs ALL discarded!

        # Always release GPU physical memory
        unmap_and_release(handle)  # → cuMemUnmap + cuMemRelease
        # 虚拟地址保留！

    logger.info(
        "sleep freed %.2f GiB GPU (%.2f GiB backed up to CPU, %.2f GiB discarded)",
        total_bytes / GiB, backup_bytes / GiB, (total_bytes - backup_bytes) / GiB
    )

    gc.collect()
    torch.cuda.empty_cache()
```

#### wake_up() — 从 CPU 恢复显存

```python
def wake_up(self, tags: Optional[list[str]] = None) -> None:
    """
    从 CPU 恢复 GPU 显存数据。

    参数语义:
      tags=None  → 恢复所有 tag 的分配
      tags=["weights"] → 仅恢复 weights
      tags=["kv_cache"] → 仅恢复 kv_cache

    流程:
      1. cuMemCreate + cuMemMap: 重新分配 GPU 物理内存, 映射到原虚拟地址
      2. 如果有 cpu_backup_tensor → CPU → GPU 拷贝恢复数据
      3. 如果没有 cpu_backup_tensor → 新分配的物理页内容未定义
         (Level 1 中丢弃的 kv_cache 走这个分支 → PR #45542 修复: 需要 zero fill)

    注意:
      Level 1 wake_up: 仅 weights 有 CPU backup, kv_cache 无备份 → 需重建
      Level 2 wake_up: 全部 tag 都有 CPU backup → 完整恢复推理状态
    """
    failed_pointers = []
    first_exc = None

    for ptr, data in self.pointer_to_data.items():
        if tags is None or data.tag in tags:
            try:
                # Step 1: 重新分配 GPU 物理内存并映射到原虚拟地址
                create_and_map(data.handle)  # → cuMemCreate + cuMemMap

                if data.cpu_backup_tensor is not None:
                    # ✅ 恢复路径: CPU → GPU 拷贝
                    size = data.cpu_backup_tensor.numel() * data.cpu_backup_tensor.element_size()
                    libcudart.cudaMemcpy(ptr, data.cpu_backup_tensor.data_ptr(), size)  # H2D
                    data.cpu_backup_tensor = None  # 释放 CPU 备份
                # else:
                #   cpu_backup_tensor 为 None → Level 1 丢弃的数据
                #   新页面内容未定义 → PR #45542: wake_up 后应 zero fill

            except RuntimeError as e:
                failed_pointers.append(ptr)
                if first_exc is None:
                    first_exc = e

    # Step 3: 等待所有 H2D 拷贝完成 (PR #45552)
    torch.cuda.synchronize()

    # Step 4: 恢复 NCCL 通信组 (PR #45554)
    self._quiesce_distributed_before_vmm_mutation()

    if failed_pointers:
        raise WakeUpPartialFailure(failed_pointers, first_exc)
```

#### C 扩展层：cumem_allocator.cpp

```cpp
// csrc/cumem_allocator.cpp

// 分配时调用（Python 端 set_python_allocator 设置的回调）
void python_malloc_callback(void* ptr, size_t size, const char* tag) {
    CUmemGenericAllocationHandle handle;
    cuMemCreate(&handle, size, &prop, 0);            // 分配物理显存
    cuMemMap((CUdeviceptr)ptr, size, 0, handle, 0);  // 映射到虚拟地址
    cuMemSetAccess((CUdeviceptr)ptr, size, &access_desc, 1);

    // 返回 handle 给 Python 层保存
    // ...
}

// sleep 时调用
void unmap_and_release(handle_t handle) {
    cuMemUnmap(handle.ptr, handle.size);    // 解除映射
    cuMemRelease(handle.alloc_handle);       // 释放物理显存
    // 注意：不调用 cuMemAddressFree，虚拟地址保留！
}

// wake_up 时调用
void create_and_map(handle_t handle) {
    // 重置错误码 (PR #45565)
    error_code = CUDA_SUCCESS;

    cuMemCreate(&handle.alloc_handle, handle.size, &prop, 0);  // 新物理内存
    cuMemMap(handle.ptr, handle.size, 0, handle.alloc_handle, 0);  // 映射到原地址
    cuMemSetAccess(handle.ptr, handle.size, &access_desc, 1);
}
```

### 3.5 可插拔 Backend 抽象（PR #44074 / RFC #34303）

vLLM 最新版本（2026）引入了可插拔的 Sleep Mode Backend 抽象。

```python
# vllm/device_allocator/sleep_mode_backend.py

class SleepModeBackend(ABC):
    """Sleep Mode 的抽象后端接口"""

    @abstractmethod
    def suspend(self, level: int = 1) -> None:
        """挂起：释放 GPU 内存"""
        ...

    @abstractmethod
    def resume(self, tags: Optional[list[str]] = None) -> None:
        """恢复：重新分配 GPU 内存"""
        ...

    @classmethod
    @abstractmethod
    def is_supported(cls) -> bool:
        """检查当前平台是否支持"""
        ...

    @classmethod
    @abstractmethod
    def preserves_nccl(cls) -> bool:
        """suspend/resume 后 NCCL 通信器是否仍然有效"""
        ...

    @classmethod
    @abstractmethod
    def preserves_compiled_artifacts(cls) -> bool:
        """suspend/resume 后 CUDA Graph / CUDNN 等编译产物是否仍有效"""
        ...


class CuMemBackend(SleepModeBackend):
    """默认实现：基于 CUDA VMM API"""

    def suspend(self, level: int = 1) -> None:
        offload_tags = ("weights",) if level == 1 else None
        CuMemAllocator.get_instance().sleep(offload_tags)

    def resume(self, tags: Optional[list[str]] = None) -> None:
        CuMemAllocator.get_instance().wake_up(tags)

    @classmethod
    def is_supported(cls) -> bool:
        return torch.cuda.is_available()

    @classmethod
    def preserves_nccl(cls) -> bool:
        return True  # PR #45623 之后，Level 2 也保留 NCCL

    @classmethod
    def preserves_compiled_artifacts(cls) -> bool:
        return True  # 虚拟地址不变，CUDA Graph 有效


class SleepModeBackendFactory:
    """工厂：注册和获取后端"""
    _registry: dict[str, tuple[str, str]] = {}

    @classmethod
    def register(cls, name: str, module_path: str, class_name: str):
        cls._registry[name] = (module_path, class_name)

    @staticmethod
    def get_backend(name: str) -> SleepModeBackend:
        module_path, class_name = SleepModeBackendFactory._registry[name]
        module = importlib.import_module(module_path)
        return getattr(module, class_name)()

# 默认注册
SleepModeBackendFactory.register("cumem",
    "vllm.device_allocator.sleep_mode_backend", "CuMemBackend")
```

### 3.6 完整调用链路

#### 3.6.1 Sleep 调用链

```
┌──────────────────────────────────────────────────────────────┐
│ 触发方式                                                       │
├──────────────────────────────────────────────────────────────┤
│ Python API:                                                  │
│   llm.sleep(level=1)                                         │
│                                                              │
│ HTTP API (仅 dev mode):                                      │
│   curl -X POST http://localhost:8000/sleep -d '{"level":1}'  │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. APIServer.sleep()                                         │
│    → 构建 RPC 请求，发送到 Executor                            │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Executor (MultiprocExecutor / RayExecutor)                 │
│    → 向所有 DP/TP/PP workers 广播 sleep() 调用                │
│    → 使用 collective_rpc 确保同步                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. GPUWorker.sleep(level=1)                                  │
│    vllm/v1/worker/gpu_worker.py                              │
│                                                              │
│    a) backend = SleepModeBackendFactory.get_backend("cumem")  │
│    b) 记录 sleep 前显存使用量                                  │
│    c) backend.suspend(level)  ←── 委托给后端                  │
│    d) 记录 sleep 后显存释放量                                  │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. CuMemBackend.suspend(level=1)                             │
│    vllm/device_allocator/sleep_mode_backend.py                │
│                                                              │
│    offload_tags = ("weights",) if level == 1 else None        │
│    → CuMemAllocator.get_instance().sleep(offload_tags)       │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. CuMemAllocator.sleep(offload_tags)                        │
│    vllm/device_allocator/cumem.py                             │
│                                                              │
│    a) torch.cuda.synchronize()        # 同步 CUDA Stream     │
│    b) _quiesce_distributed_groups()   # 静默 NCCL            │
│    c) for each allocation:                                   │
│         if should_offload(tag):                              │
│           cpu_tensor = torch.empty(size, pin_memory=True)    │
│           cudaMemcpy(cpu_ptr, gpu_ptr, size)  # D2H          │
│           data.cpu_backup_tensor = cpu_tensor                │
│         unmap_and_release(handle)  # cuMemUnmap+cuMemRelease │
│    d) gc.collect(); torch.cuda.empty_cache()                 │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. C 扩展: unmap_and_release (csrc/cumem_allocator.cpp)       │
│                                                              │
│    cuMemUnmap(ptr, size)        # 解除 VA→物理映射            │
│    cuMemRelease(alloc_handle)   # 释放物理显存句柄            │
│    // 注意：不调用 cuMemAddressFree，虚拟地址保留               │
└──────────────────────────────────────────────────────────────┘
```

#### 3.6.2 Wake Up 调用链

```
┌──────────────────────────────────────────────────────────────┐
│ 触发方式                                                       │
├──────────────────────────────────────────────────────────────┤
│ Python API:                                                  │
│   llm.wake_up(tags=["weights"])                              │
│   llm.wake_up()  # 恢复全部                                   │
│                                                              │
│ HTTP API:                                                    │
│   curl -X POST http://localhost:8000/wake_up                 │
│   curl -X POST "http://localhost:8000/wake_up?tags=weights"  │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ GPUWorker.wake_up(tags=["weights"])                          │
│                                                              │
│   backend.resume(tags) → CuMemAllocator.wake_up(tags)        │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ CuMemAllocator.wake_up(tags)                                 │
│                                                              │
│   for each allocation matching tags:                         │
│     a) create_and_map(handle)  # cuMemCreate + cuMemMap      │
│     b) if cpu_backup_tensor exists:                          │
│          cudaMemcpy(gpu_ptr, cpu_ptr, size)  # H2D 恢复      │
│          cpu_backup_tensor = None                            │
│                                                              │
│   torch.cuda.synchronize()     # 等 H2D 完成                  │
│   _quiesce_distributed_groups() # 恢复 NCCL                  │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ RLHF 使用模式：精细 Wake Up                                   │
│                                                              │
│   llm.wake_up(tags=["weights"])   # 先恢复权重                 │
│   # ... 执行权重同步 / 更新 ...                                │
│   llm.wake_up(tags=["kv_cache"])  # 再恢复 KV Cache           │
│                                                              │
│   优势：最小化峰值 GPU 显存使用，避免 OOM                        │
└──────────────────────────────────────────────────────────────┘
```

### 3.7 关键 Bugfix

| PR | 问题 | 修复方案 |
|----|------|---------|
| **#45552** | sleep() 时 in-flight CUDA kernel 未完成 → `cudaErrorIllegalAddress` | sleep() 入口加 `torch.cuda.synchronize()`，wake_up() 出口也同步 |
| **#45554** | 多 Rank 下 VMM 变更时序不一致 → NCCL 访问非法内存 | VMM 变更前后加 `torch.distributed.barrier(cpu_group)` 同步所有 Rank |
| **#45565** | 上次 sleep 的错误码未清除 → wake_up 静默跳过分配 | `create_and_map` 入口重置 `error_code`，失败时收集所有异常统一抛出 `WakeUpPartialFailure` |
| **#45623** | CUDA Graph + NCCL 通信器内存未在 sleep 时释放 | 新增 `"graphs"` tag，`ncclCommSuspend(NCCL_SUSPEND_MEM)` |
| **#35956** | kv_cache mempool 上下文过大导致 sleep mode 回归 | 缩小 `use_memory_pool("kv_cache")` 作用域到仅 tensor 分配 |

---

## 四、SGLang Memory Saver

### 4.1 架构总览

SGLang 的 Sleep Mode 基于独立库 `torch_memory_saver`，由 fzyzcjy 开发，集成在 PR #2630 中。

```
                     HTTP API Layer
                     ┌──────────────────────────────┐
                     │  POST /release_memory_occupation │
                     │  POST /resume_memory_occupation  │
                     │  (tags: "weights", "kv_cache")    │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  Scheduler                     │
                     │  release_gpu_occupation(tags) │  ← sglang/srt/managers/scheduler.py
                     │  resume_gpu_occupation(tags)  │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  torch_memory_saver           │  ← 独立库 fzyzcjy/torch_memory_saver
                     │  pause(tags)                 │
                     │  resume(tags)                │
                     │  region()                    │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  CUDA VMM API (C Extension)   │
                     │  cuMemCreate / cuMemMap       │
                     │  cuMemUnmap / cuMemRelease    │
                     │  cuMemAddressReserve / Free   │
                     └──────────────────────────────┘
```

### 4.2 torch_memory_saver 核心实现

#### 4.2.1 设计理念

> "暂停时释放物理内存，保留虚拟地址。恢复时虚拟地址不变，物理内存重新分配映射。"

#### 4.2.2 数据结构

```cpp
// csrc/core.cpp (torch_memory_saver)

struct _AllocationMetadata {
    size_t size;
    CUdevice device;
    CUmemGenericAllocationHandle allocHandle;
    std::string tag;
};

class TorchMemorySaver {
    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void*, _AllocationMetadata> allocation_metadata_;
    // key: 虚拟地址指针
    // value: 对应的物理内存句柄和元数据
};
```

#### 4.2.3 malloc 实现

```cpp
cudaError_t malloc(void **ptr, size_t size, const std::string& tag) {
    CUdevice device;
    cuCtxGetDevice(&device);

    CUmemGenericAllocationHandle allocHandle;
    // Step 1: 创建物理内存
    CUDAUtils::cu_mem_create(&allocHandle, size, device);

    // Step 2: 预留虚拟地址空间
    cuMemAddressReserve((CUdeviceptr *)ptr, size, 0, 0, 0);

    // Step 3: 映射物理内存 → 虚拟地址
    cuMemMap((CUdeviceptr)*ptr, size, 0, allocHandle, 0);

    // Step 4: 设置访问权限
    CUDAUtils::cu_mem_set_access(*ptr, size, device);

    // Step 5: 记录元数据
    {
        std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(*ptr,
            _AllocationMetadata{size, device, allocHandle, tag});
    }

    return cudaSuccess;
}
```

#### 4.2.4 pause 实现

```cpp
void pause(const std::vector<std::string>& tags) {
    std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

    for (auto& [ptr, metadata] : allocation_metadata_) {
        if (!tags.empty() &&
            std::find(tags.begin(), tags.end(), metadata.tag) == tags.end()) {
            continue;  // tag 不匹配，跳过
        }

        // Step 1: 解除映射
        cuMemUnmap((CUdeviceptr)ptr, metadata.size);

        // Step 2: 释放物理内存
        cuMemRelease(metadata.allocHandle);

        // ⚠️ 关键：不调用 cuMemAddressFree
        // ptr 虚拟地址仍然有效，metadata 保留用于 resume
    }
}
```

#### 4.2.5 resume 实现

```cpp
void resume(const std::vector<std::string>& tags) {
    std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

    for (auto& [ptr, metadata] : allocation_metadata_) {
        if (!tags.empty() &&
            std::find(tags.begin(), tags.end(), metadata.tag) == tags.end()) {
            continue;
        }

        // Step 1: 创建新的物理内存句柄（可能与 sleep 前不同物理页）
        CUmemGenericAllocationHandle newAllocHandle;
        CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device);

        // Step 2: 映射到原虚拟地址（ptr 不变！）
        cuMemMap((CUdeviceptr)ptr, metadata.size, 0, newAllocHandle, 0);

        // Step 3: 设置访问权限
        CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

        // Step 4: 更新元数据（指向新的物理内存句柄）
        metadata.allocHandle = newAllocHandle;
    }
}
```

#### 4.2.6 free 实现（彻底释放，非 sleep）

```cpp
cudaError_t free(void *ptr) {
    _AllocationMetadata metadata;
    {
        std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        metadata = allocation_metadata_[ptr];
        allocation_metadata_.erase(ptr);
    }

    cuMemUnmap((CUdeviceptr)ptr, metadata.size);       // 解除映射
    cuMemRelease(metadata.allocHandle);                 // 释放物理内存
    cuMemAddressFree((CUdeviceptr)ptr, metadata.size);  // ★ 虚拟地址也释放

    return cudaSuccess;
}
```

**pause 与 free 的核心区别**：

| 操作 | cuMemUnmap | cuMemRelease | cuMemAddressFree | 虚拟地址 |
|------|:---:|:---:|:---:|------|
| **pause** | ✅ | ✅ | ❌ | **保留** |
| **free** | ✅ | ✅ | ✅ | **释放** |

### 4.3 Region 机制

torch_memory_saver 通过 `region()` 上下文管理器控制哪些内存分配走自定义虚拟内存管理。

#### C++ 层

```cpp
// csrc/core.cpp

namespace RegionManager {
    static thread_local bool is_interesting_region_ = false;
    static thread_local std::string current_tag_ = "default";

    void enter() { is_interesting_region_ = true; }
    void leave() { is_interesting_region_ = false; }
    bool is_interesting_region() { return is_interesting_region_; }

    void set_current_tag(const std::string& tag) { current_tag_ = tag; }
    std::string get_current_tag() { return current_tag_; }
}
```

#### 拦截层：cudaMalloc Hook

```cpp
cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (RegionManager::is_interesting_region()) {
        // 在 region 内 → 走自定义虚拟内存管理
        return TorchMemorySaver::instance().malloc(
            ptr, size, RegionManager::get_current_tag()
        );
    } else {
        // 不在 region 内 → 走原生 CUDA runtime
        return APIForwarder::call_real_cuda_malloc(ptr, size);
    }
}

// 获取原生 CUDA runtime 函数
void* APIForwarder::get_real_cuda_function(const char* name) {
    return dlsym(RTLD_NEXT, name);  // 跳过当前 SO，找下一个
}
```

#### Python 层使用

```python
# SGLang 初始化代码

import torch_memory_saver

# SGLang 将所有需要管理的组件包裹在 region 内
primary_memory_saver = torch_memory_saver.MemorySaver()

# 初始化时：
with primary_memory_saver.region():
    # TokenToKVPool 的 k_buffers / v_buffers
    # ModelRunner 的 model 参数
    # ReqToTokenPool 的 req_to_token 表
    # ... 这些分配都会被拦截并标记

# 后续在调度器中调用：
primary_memory_saver.pause(tags=["kv_cache"])   # 仅释放 KV Cache
primary_memory_saver.resume(tags=["kv_cache"])  # 仅恢复 KV Cache
```

### 4.4 CUDA Graph 兼容性设计

这是一个非常精密的设计，确保 CUDA Graph 在 sleep/wake 周期后仍然有效。

#### 4.4.1 兼容性原理

```
┌─────────────────────────────────────────────────────────────┐
│              CUDA Graph 与 Virtual Address 的关系            │
│                                                             │
│  CUDA Graph Capture 时:                                     │
│    kernel_A(kwargs: {input: 0x7f_0000_1000,                 │
│                      weight: 0x7f_0000_2000})  ← 虚拟地址     │
│    kernel_B(kwargs: {input: 0x7f_0000_3000})                │
│    ↓                                                        │
│  cudaGraphInstantiate:                                      │
│    将虚拟地址写入 graph node 的参数 → 地址被"烧录"            │
│    ↓                                                        │
│  Sleep 后:                                                  │
│    物理显存释放，但虚拟地址 0x7f_0000_1000 仍然保留            │
│    ↓                                                        │
│  Wake Up 后:                                                │
│    新物理显存映射回 0x7f_0000_1000                           │
│    ↓                                                        │
│  cudaGraphLaunch:                                           │
│    地址 0x7f_0000_1000 仍然有效 → Graph 无需重新捕获！        │
└─────────────────────────────────────────────────────────────┘
```

#### 4.4.2 torch_memory_saver 的 CUDA Graph 包装

```python
# torch_memory_saver 提供的 CUDA Graph 包装
with torch_memory_saver.cuda_graph():
    # 等价于 with torch.cuda.graph():
    # 但内部的 malloc 会对齐到 VMM 管理边界
    # 确保 capture 期间的地址在 resume 后仍然有效
    ...
```

#### 4.4.3 权重更新时的 Placeholder 策略（扩散模型）

当使用 layerwise offload（`--dit-layerwise-offload`）时：

```
┌─────────────────────────────────────────────────────────────┐
│              Layerwise Offload 的 Placeholder 策略           │
│                                                             │
│  正常状态:                                                   │
│    GPU:  [weight_layer_0] [weight_layer_1] ... [weight_L]   │
│    CPU:  (nothing)                                          │
│                                                             │
│  Offload 后:                                                │
│    GPU:  [torch.empty((1,))] [torch.empty((1,))] ...        │
│           ↑ 占位符 (placeholder)                             │
│    CPU:  [weight_layer_0_real] [weight_layer_1_real] ...    │
│           ↑ 真实权重在 pinned CPU buffer                     │
│                                                             │
│  权重更新时:                                                 │
│    检测到 offload manager 活跃                               │
│    → 新权重直接写入 CPU buffer（绕过 GPU 占位符）             │
│    → 避免 shape mismatch 错误                                │
│                                                             │
│  预取（Prefetch）时:                                         │
│    如果某层恰好被预取到 GPU:                                  │
│    → 同步更新 GPU 上的 live tensor                           │
└─────────────────────────────────────────────────────────────┘
```

### 4.5 HiCache 三级 KV Cache 体系

SGLang 的 HiCache 不仅支持 sleep mode 的内存释放，还支持持久化和跨实例共享。

```
┌──────────────────────────────────────────────────────────────┐
│                  SGLang HiCache 三级缓存                       │
│                                                              │
│  ┌──────────────────────┐                                    │
│  │   L1: GPU VRAM        │  KV cache 活跃数据                │
│  │   直接参与 Attention   │  最高带宽，最小容量                 │
│  │   计算                 │                                    │
│  └──────────┬───────────┘                                    │
│             │ D2H / H2D (async)                               │
│             ▼                                                │
│  ┌──────────────────────┐                                    │
│  │   L2: CPU Host Memory │  本地大容量缓冲                    │
│  │   (Pinned/Registered) │  中转层，连接 L1 和 L3             │
│  │                       │  sleep mode 的 weights backup     │
│  └──────────┬───────────┘    也用这一层                       │
│             │ Storage I/O                                     │
│             ▼                                                │
│  ┌──────────────────────┐                                    │
│  │   L3: 外部/分布式存储  │  跨实例共享                       │
│  │   (Mooncake/3FS/NIXL) │  持久化 KV cache                   │
│  │                       │  Prefix caching 跨请求复用          │
│  └──────────────────────┘                                    │
└──────────────────────────────────────────────────────────────┘
```

#### 写入策略

| 策略 | 行为 | 适用场景 |
|------|------|---------|
| `write_through` | GPU KV → Host → L3 同步写 | 数据持久性要求高 |
| `write_through_selective` | 仅 "热" 数据（命中次数超阈值）才写穿透 | 平衡性能和持久性 |
| `write_back` | 仅在 GPU 淘汰时才写回 Host/L3 | 最大化写性能 |

#### 与 Sleep Mode 的协同

```python
# sleep mode 释放 kv_cache 时：
#   1. HiCache 可能已将 KV 持久化到 L2/L3
#   2. wake_up 时可直接从 L2/L3 恢复，而非丢弃

# 典型 RL 场景：
#   [Rollout 生成] → KV 写入 L2 (CPU Host)
#   [Sleep engines] → 释放 GPU KV Cache
#   [Training] → 使用 GPU 进行训练
#   [Wake engines] → 从 L2 恢复 KV Cache 到 GPU
#   [继续 Rollout] → 继续生成（prefix 命中 L2 Cache）
```

### 4.6 Layerwise Offload（扩散模型专用）

```python
# 专门为 Diffusion Transformer (DiT) 模型设计的逐层卸载

# 启动参数
# --dit-layerwise-offload

# 工作原理：
# 1. GPU 上保留 torch.empty((1,)) 占位符
# 2. 真实权重保存在 pinned CPU buffer 的连续内存块
# 3. 每层计算前：CPU → GPU 传输该层权重
# 4. 每层计算后：释放该层的 GPU 权重（可选）

# 权重更新感知：
# 检测到 offload manager 活跃时
# → 新权重直接写入 CPU buffer
# → 绕过 GPU 占位符

# 原子性保证：
# 所有模块加载失败 → 全部回滚到原始权重
# 任何一层成功但其他层失败 → 全部回滚
```

### 4.7 完整调用链路

#### 4.7.1 Release Memory Occupation 调用链

```
┌──────────────────────────────────────────────────────────────┐
│ 触发方式                                                       │
├──────────────────────────────────────────────────────────────┤
│ HTTP API:                                                    │
│   curl -X POST http://host:port/release_memory_occupation    │
│     -d '{"tags": ["kv_cache", "weights"]}'                   │
│                                                              │
│ 前提条件:                                                     │
│   --enable-memory-saver (启动参数)                            │
│   引擎必须处于空闲状态 (no ongoing requests)                   │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. HTTP Handler → TokenizerManager                           │
│                                                              │
│    验证请求:                                                   │
│    - 检查引擎是否空闲                                          │
│    - 解析 tags (默认: ["kv_cache", "weights"])                │
│    → 转发到 Scheduler                                        │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Scheduler.release_gpu_occupation(tags)                    │
│    sglang/srt/managers/scheduler.py                          │
│                                                              │
│    a) 确认所有请求已完成                                       │
│    b) 通知所有 TP workers                                     │
│    c) primary_memory_saver.pause(tags)                       │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. torch_memory_saver.pause(tags=["kv_cache", "weights"])    │
│                                                              │
│    for each allocation:                                      │
│      if allocation.tag in tags:                              │
│        if enable_cpu_backup:                                 │
│          GPU → CPU pinned memory copy  # 保留数据内容         │
│        cuMemUnmap(ptr, size)           # 解除映射            │
│        cuMemRelease(alloc_handle)      # 释放物理显存         │
│        // 虚拟地址保留                                         │
│                                                              │
│    注意：暂停后 CUDA Graph 仍然有效（地址不变）                  │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. 释放的显存被 PyTorch/CUDA driver 回收                       │
│    → 可用于 Training 或其他计算任务                            │
└──────────────────────────────────────────────────────────────┘
```

#### 4.7.2 Resume Memory Occupation 调用链

```
┌──────────────────────────────────────────────────────────────┐
│ 触发方式                                                       │
├──────────────────────────────────────────────────────────────┤
│   curl -X POST http://host:port/resume_memory_occupation     │
│     -d '{"tags": ["weights", "kv_cache"]}'                   │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. Scheduler.resume_gpu_occupation(tags)                     │
│                                                              │
│    a) primary_memory_saver.resume(tags)                      │
│    b) 如果 resume 了 weights:                                │
│         → 需要调用 /update_weights_from_tensor 刷新权重       │
│         (训练可能已经更新了权重)                                │
│    c) 如果 resume 了 kv_cache:                               │
│         → 重新分配 KV cache pool                              │
│         → 如果有 HiCache L2 备份 → 可选恢复                    │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. torch_memory_saver.resume(tags)                           │
│                                                              │
│    for each allocation:                                      │
│      if allocation.tag in tags:                              │
│        cuMemCreate(&new_handle, size)    # 新物理内存         │
│        cuMemMap(ptr, size, 0, new_handle, 0) # 映射到原地址  │
│        cuMemSetAccess(ptr, size, ...)    # 设置权限           │
│        if cpu_backup exists:                                 │
│          CPU → GPU cudaMemcpy            # 恢复数据内容       │
│        allocation.allocHandle = new_handle                   │
│                                                              │
│    虚拟地址 ptr 与 pause 前完全相同 → CUDA Graph 无需重建      │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. RL 训练场景的完整周期                                       │
│                                                              │
│   [Training Done]                                            │
│     → POST /release_memory_occupation (tags: kv_cache,weights)│
│     → GPU 显存释放给 Training                                 │
│                                                              │
│   [Training Step]                                            │
│     → HF model forward/backward                              │
│     → optimizer.step()                                       │
│                                                              │
│   [Rollout Needed]                                           │
│     → POST /resume_memory_occupation (tags: weights,kv_cache) │
│     → POST /update_weights_from_tensor (刷新训练后的权重)      │
│     → 开始新的 Generation 请求                                 │
│                                                              │
│   [Repeat]                                                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 五、两者对比分析

### 5.1 架构对比

```
┌──────────────────────────────────────────────────────────────┐
│                 vLLM vs SGLang Sleep Mode                     │
├──────────────────┬─────────────────────┬─────────────────────┤
│      维度         │       vLLM          │      SGLang          │
├──────────────────┼─────────────────────┼─────────────────────┤
│ 核心实现          │ CuMemAllocator      │ torch_memory_saver  │
│                  │ (内置 C 扩展)        │ (独立库, LD_PRELOAD) │
│                  │                     │                     │
│ 底层 API          │ CUDA VMM            │ CUDA VMM            │
│                  │ cuMemCreate/Map/     │ cuMemCreate/Map/     │
│                  │ Unmap/Release        │ Unmap/Release        │
│                  │                     │                     │
│ 虚拟地址管理      │ 保留虚拟地址         │ 保留虚拟地址          │
│                  │                     │                     │
│ CUDA Graph 兼容  │ ✅ (地址不变)        │ ✅ (地址不变 + 包装器)│
│                  │                     │                     │
│ Tag 体系         │ "weights"           │ "weights"           │
│                  │ "kv_cache"          │ "kv_cache"           │
│                  │ "graphs"            │ "cuda_graph"         │
│                  │                     │                     │
│ Sleep 级别       │ Level 1 (weight      │ 无级别概念           │
│                  │   offload)           │ 通过 tags 灵活控制    │
│                  │ Level 2 (full)       │                     │
│                  │                     │                     │
│ CPU Backup       │ Level 1: 仅 weights  │ 可配置               │
│                  │ Level 2: 全部        │ enable_cpu_backup    │
│                  │                     │                     │
│ 可插拔后端        │ ✅ (RFC #34303)      │ ❌ (依赖             │
│                  │ SleepModeBackend     │  torch_memory_saver) │
│                  │                     │                     │
│ KV Cache 持久化   │ ❌ (丢弃后丢失)      │ ✅ (HiCache L2/L3)   │
│                  │                     │                     │
│ 分布式支持         │ TP/PP/DP 全部       │ TP/PP 支持           │
│                  │ collective_rpc 同步  │                     │
│                  │                     │                     │
│ 安全性           │ 仅 Dev Mode 暴露     │ --enable-memory-    │
│                  │ sleep/wake endpoint  │ saver 开启          │
│                  │                     │                     │
│ 平台支持         │ NVIDIA CUDA          │ NVIDIA CUDA          │
│                  │ AMD ROCm            │ (通过 torch_memory_  │
│                  │ Intel XPU          │  saver 的 CUDA 依赖)  │
│                  │ Huawei Ascend       │                     │
├──────────────────┴─────────────────────┴─────────────────────┤
│ 关键差异                                                       │
├──────────────────────────────────────────────────────────────┤
│ 1. vLLM Level 1 仅备份 weights 到 CPU，KV Cache/Graphs 直接丢弃│
│    → wake_up 后仅恢复 weights，KV Cache 需重建                  │
│    → CPU RAM 需求：≈ 模型大小                                  │
│    → 适用场景：RL Rollout 后 KV Cache 无需保留，wake_up 速度快  │
│                                                              │
│ 2. vLLM Level 2 备份全部 (weights + kv_cache + graphs) 到 CPU │
│    → wake_up 后完整恢复推理状态 (含 KV Cache)                   │
│    → CPU RAM 需求：≈ 模型 + KV Cache + Graphs 总和             │
│    → 适用场景：需要保存完整推理状态、精准续推                     │
│                                                              │
│ 3. SGLang 通过 HiCache 可持久化 KV Cache 到 CPU/分布式存储    │
│    → 恢复后可复用 prefix，共享跨请求 KV Cache                  │
│    → 适用场景：长上下文 multi-turn 对话、prefix caching         │
│                                                              │
│ 4. SGLang 的 torch_memory_saver 是独立库                     │
│    → 理论上可被其他框架复用（不含 SGLang 依赖）                 │
│    → vLLM 的实现与自身紧密耦合                                 │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 适用场景推荐

| 场景 | 推荐框架 | 原因 |
|------|---------|------|
| **RLHF co-located 训练+推理** | 两个都适用 | Level 1 (vLLM) 或 tags=["weights","kv_cache"] (SGLang) |
| **模型热切换** | **vLLM** | Level 2 释放 95%+ 显存，可加载新模型 |
| **长上下文 prefix caching** | **SGLang** | HiCache L2/L3 可持久化和跨请求共享 KV Cache |
| **多 GPU 分布式推理** | **vLLM** | collective_rpc 同步 + NCCL quiesce 更成熟 |
| **扩散模型 (DiT) 推理** | **SGLang** | Layerwise offload 专用优化 |
| **AMD/NPU 平台** | **vLLM** | 更广泛的平台支持 |

---

## 六、最佳实践与注意事项

### 6.1 配置建议

#### vLLM

```bash
# 启动服务器
vllm serve meta-llama/Llama-3-8b \
    --enable-sleep-mode \
    --sleep-mode-backend cumem  # 默认即可

# 仅在 Dev 模式下暴露 HTTP sleep endpoint
VLLM_SERVER_DEV_MODE=1 vllm serve <model> --enable-sleep-mode

# AMD ROCm 配置
export VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE=256  # MB, 默认 256

# Level 1: 同模型复用（推荐 RLHF 默认）
curl -X POST http://localhost:8000/sleep -d '{"level": 1}'

# Level 2: 模型切换
curl -X POST http://localhost:8000/sleep -d '{"level": 2}'
```

#### SGLang

```bash
# 启动服务器
python -m sglang.launch_server \
    --model meta-llama/Llama-3-8b \
    --enable-memory-saver

# 释放 KV Cache + Weights（RL 训练场景）
curl -X POST http://localhost:30000/release_memory_occupation \
    -H "Content-Type: application/json" \
    -d '{"tags": ["kv_cache", "weights"]}'

# 恢复
curl -X POST http://localhost:30000/resume_memory_occupation \
    -H "Content-Type: application/json" \
    -d '{"tags": ["weights", "kv_cache"]}'

# 更新权重（训练后）
curl -X POST http://localhost:30000/update_weights_from_tensor \
    -H "Content-Type: application/json" \
    -d '{"tensor": ...}'
```

### 6.2 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|---------|
| **CPU 内存不足** | Level 1 需要 CPU RAM >= 模型大小 | 确保系统 CPU RAM 充足，或使用 Level 2 丢弃 weights |
| **is_sleeping 未恢复** | 部分 tag 未 wake_up 时仍报告 sleeping | 检查所有 tag 是否都已 wake_up；wake_up() 不带参数恢复全部 |
| **CUDA Graph 失效** | 自定义 CUDA 内存管理未使用 VMM | 确保使用框架提供的 CUDA Graph 包装器 |
| **NCCL 通信中断** | 多 Rank sleep 时序不一致 | 使用 barrier 同步所有 Rank（vLLM PR #45554） |
| **Dev Mode 暴露** | sleep endpoint 被外部访问 | 生产环境关闭 `VLLM_SERVER_DEV_MODE` |

### 6.3 性能基准参考

```
场景: Llama-3-8B on A100-80G

vLLM:
  Sleep 前 GPU 显存占用: ~16 GB
  Sleep Level 1 后:      ~2 GB  (释放 ~14 GB, ~87%)
  Sleep Level 2 后:      ~0.5 GB (释放 ~15.5 GB, ~97%)
  Wake Up (Level 1):     ~1.2s  (仅 weights DMA回传)
  Wake Up (Level 2):     ~1.5s  (weights + graphs 恢复)

SGLang:
  初始 GPU 显存占用:     ~15 GB
  Release Weights+KV:    ~3 GB  (释放 ~12 GB, ~80%)
  Resume Weights:        ~0.7s
  Resume KV Cache:       ~0.5s
  Resume Full:           ~1.2s
```

---

## 七、参考来源

### vLLM

| 来源 | 内容 |
|------|------|
| [vLLM Sleep Mode 文档](https://docs.vllm.ai/en/v0.11.0/features/sleep_mode.html) | 官方文档 |
| `vllm/v1/worker/gpu_worker.py` | `sleep()` / `wake_up()` 入口 |
| `vllm/device_allocator/cumem.py` | `CuMemAllocator` 核心实现 |
| `vllm/device_allocator/sleep_mode_backend.py` | 可插拔后端抽象 (PR #44074) |
| `csrc/cumem_allocator.cpp` | C 扩展：VMM API 调用 |
| [PR #23521](https://github.com/vllm-project/vllm/pull/23521) | Sleep Level 2 E2E 测试 |
| [PR #28053](https://github.com/vllm-project/vllm/pull/28053) | 移除 busy loop，移除 `VLLM_SLEEP_WHEN_IDLE` |
| [PR #44074](https://github.com/vllm-project/vllm/pull/44074) | 可插拔 Sleep Mode Backend (RFC #34303) |
| [PR #45552](https://github.com/vllm-project/vllm/pull/45552) | Stream sync bugfix |
| [PR #45554](https://github.com/vllm-project/vllm/pull/45554) | NCCL quiesce bugfix |
| [PR #45565](https://github.com/vllm-project/vllm/pull/45565) | WakeUp partial failure recovery |
| [PR #45623](https://github.com/vllm-project/vllm/pull/45623) | CUDA Graph + NCCL mem offload |
| [PR #35956](https://github.com/vllm-project/vllm/pull/35956) | Narrow kv_cache mempool context |
| [VERL AMD 教程](https://verl.readthedocs.io/en/latest/amd_tutorial/amd_vllm_page.html) | AMD ROCm sleep mode 指南 |

### SGLang

| 来源 | 内容 |
|------|------|
| [SGLang for RL Systems](https://docs.sglang.io/docs/advanced_features/sglang_for_rl) | 官方 RL 集成文档 |
| [PR #2630](https://github.com/sgl-project/sglang/pull/2630) | Memory saver 初始实现 |
| [PR #7208](https://github.com/sgl-project/sglang/pull/7208) | 多实例多阶段 E2E 测试 |
| `sglang/srt/managers/scheduler.py` | `release_gpu_occupation` / `resume_gpu_occupation` |
| [fzyzcjy/torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver) | 底层库：CUDA VMM 封装 |
| `torch_memory_saver/csrc/core.cpp` | C 扩展：malloc/free/pause/resume |
| [SGLang HiCache Issue #18239](https://github.com/sgl-project/sglang/issues/18239) | HiCache 统一存储框架 |
| [SGLang KV Cache Offload Issue #14184](https://github.com/sgl-project/sglang/issues/14184) | KV Cache 层级卸载 |
| [ai-dynamo PR #6967](https://github.com/ai-dynamo/dynamo/pull/6967) | Memory saver endpoint guard |
| [CUDA Graph 深度分析](https://zhuanlan.zhihu.com/p/2017950447520980998) | CUDA Graph 兼容性约束 |

### 通用

| 来源 | 内容 |
|------|------|
| [CUDA VMM API 文档](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html) | CUDA 虚拟内存管理 API |
| [VERL 项目](https://github.com/verl-project/verl) | RL 训练框架，深度集成 vLLM sleep mode |
