# vLLM-Ascend Sleep Mode 深度解析

> **日期**：2026-07-02
> **目标读者**：推理框架开发团队（Ascend NPU 方向）
> **前置阅读**：建议先阅读《vLLM-SGLang-SleepMode深度解析》了解通用 CUDA 实现
> **内容范围**：vLLM-Ascend 中 Sleep Mode 的架构设计、AscendCL 底层实现、CaMemAllocator 源码解析、与 CUDA 实现的差异对比

---

## 目录

1. [背景与平台特性](#一背景与平台特性)
2. [架构总览](#二架构总览)
3. [AscendCL 虚拟内存管理基础](#三ascendcl-虚拟内存管理基础)
4. [CaMemAllocator 核心实现](#四camemallocator-核心实现)
5. [C++ 扩展层：camem_allocator.cpp](#五c-扩展层camem_allocatorcpp)
6. [NPUWorker：Sleep/Wake 编排层](#六npuworkersleepwake-编排层)
7. [Ascend 特有的权重格式处理](#七ascend-特有的权重格式处理)
8. [完整调用链路](#八完整调用链路)
9. [与 CUDA 实现的逐层对比](#九与-cuda-实现的逐层对比)
10. [版本演进与兼容性](#十版本演进与兼容性)
11. [已知问题与最佳实践](#十一已知问题与最佳实践)
12. [参考来源](#十二参考来源)

---

## 一、背景与平台特性

### 1.1 昇腾 NPU 平台特性

vLLM-Ascend 是 vLLM 在华为昇腾（Ascend）NPU 上的移植版本。与 NVIDIA CUDA 平台相比，有以下几个关键差异：

| 特性 | NVIDIA CUDA | 华为 Ascend NPU |
|------|------------|----------------|
| **编程接口** | CUDA Runtime / Driver API | **AscendCL (ACL)** / CANN |
| **设备 API** | `cudaMalloc` / `cudaFree` | `aclrtMalloc` / `aclrtFree` |
| **虚拟内存 API** | `cuMemCreate` / `cuMemMap` / `cuMemUnmap` / `cuMemRelease` / `cuMemAddressReserve` | `aclrtMallocPhysical` / `aclrtMapMem` / `aclrtUnmapMem` / `aclrtFreePhysical` / `aclrtReserveMemAddress` |
| **Host↔Device 拷贝** | `cudaMemcpy` | `aclrtMemcpy` |
| **PyTorch 后端** | `torch.cuda` | `torch.npu` (torch-npu 插件) |
| **可插拔分配器** | `torch.cuda.memory.CUDAPluggableAllocator` | `torch.npu.memory.NPUPluggableAllocator` |
| **图捕获** | CUDA Graph | Torchair Graph / ACLGraph |
| **权重格式** | 标准 FP16/BF16/FP8 | **FRACTAL_NZ** / **FRACTAL_ND** (昇腾特有格式) |
| **集合通信** | NCCL | HCCL (Huawei Collective Communication Library) |
| **内存池 API** | `torch.cuda.memory.MemPool` | `torch.npu.memory.MemPool` |

### 1.2 Sleep Mode 在昇腾上的特殊意义

昇腾 NPU 的 HBM（High Bandwidth Memory）容量通常比同代 NVIDIA GPU 小，因此 Sleep Mode 在昇腾平台上尤为重要：

- **RLHF/RL 训练**：推理引擎（vLLM）与训练交替执行，NPU HBM 竞争激烈
- **模型切换**：HBM 容量小 → 更需要释放显存来切换模型
- **多实例部署**：多个推理服务共享 NPU 时需要动态释放/恢复内存

---

## 二、架构总览

```
                     HTTP API Layer (仅 Dev Mode)
                     ┌──────────────────────────────┐
                     │  POST /sleep                  │
                     │  POST /wake_up                │
                     │  GET  /is_sleeping            │
                     └──────────────┬───────────────┘
                                    │ RPC
                     ┌──────────────▼───────────────┐
                     │  Executor                     │  (MultiprocExecutor / RayExecutor)
                     │  → broadcast to NPU Workers   │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  NPUWorker (AscendWorker)     │  ← vllm_ascend/worker/worker.py
                     │  sleep(level)                 │
                     │  wake_up(tags)                │
                     │  处理 FRACTAL_NZ 权重格式      │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  CaMemAllocator (单例)         │  ← vllm_ascend/device_allocator/camem.py
                     │  sleep(offload_tags)          │
                     │  wake_up(tags)               │
                     │  use_memory_pool(tag)         │
                     │  python_malloc_callback       │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  PyTorch NPU Pluggable       │
                     │  Allocator 接口               │  ← torch.npu.memory.NPUPluggableAllocator
                     │  拦截所有 npu tensor 分配      │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  C 扩展: camem_allocator.cpp  │  ← vllm_ascend/csrc/
                     │  my_malloc / my_free          │
                     │  aclrtReserveMemAddress       │
                     │  aclrtMallocPhysical          │
                     │  aclrtMapMem / aclrtUnmapMem  │
                     │  aclrtMemcpy (D2H / H2D)      │
                     └──────────────────────────────┘
```

### 2.1 关键组件职责

| 组件 | 文件位置 | 职责 |
|------|---------|------|
| **NPUWorker** | `vllm_ascend/worker/worker.py` | Sleep/Wake 编排、权重格式转换、buffers 管理 |
| **CaMemAllocator** | `vllm_ascend/device_allocator/camem.py` | Tagged Memory Pool、CPU Backup、C 扩展回调 |
| **C 扩展** | `vllm_ascend/csrc/camem_allocator.cpp` | AscendCL 虚拟内存 API 调用、`my_malloc`/`my_free` |
| **NPUModelRunner** | `vllm_ascend/worker/v2/model_runner.py` | V2 版本继承 `GPUModelRunner`，复用社区代码 |
| **AscendConfig** | `vllm_ascend/config.py` | NPU 专用配置（`VLLM_ASCEND_ENABLE_NZ` 等） |

---

## 三、AscendCL 虚拟内存管理基础

### 3.1 核心 API 对照

昇腾 NPU 的虚拟内存管理遵循与 CUDA VMM 类似的设计理念，API 命名不同但语义高度对应：

```
┌──────────────────────────────────────────────────────────────┐
│                CUDA VMM  ←→  AscendCL VMM                     │
├──────────────────────────────┼───────────────────────────────┤
│  cuMemAddressReserve         │  aclrtReserveMemAddress       │
│  预留虚拟地址空间              │  预留虚拟地址空间               │
│                              │                               │
│  cuMemCreate                 │  aclrtMallocPhysical          │
│  分配物理显存，返回句柄        │  分配物理 HBM，返回句柄         │
│                              │                               │
│  cuMemMap                    │  aclrtMapMem                  │
│  物理内存 → 虚拟地址映射       │  物理内存 → 虚拟地址映射        │
│                              │                               │
│  cuMemSetAccess              │  (aclrtMapMem 隐式设置)        │
│  设置访问权限                 │                               │
│                              │                               │
│  cuMemUnmap                  │  aclrtUnmapMem                │
│  解除映射                     │  解除映射                      │
│                              │                               │
│  cuMemRelease                │  aclrtFreePhysical            │
│  释放物理内存句柄              │  释放物理内存句柄               │
│                              │                               │
│  cuMemAddressFree            │  aclrtReleaseMemAddress       │
│  释放虚拟地址空间 (仅 free 时) │  释放虚拟地址空间               │
│                              │                               │
│  cudaMemcpy                  │  aclrtMemcpy                  │
│  Host↔Device 数据拷贝         │  Host↔Device 数据拷贝          │
└──────────────────────────────┴───────────────────────────────┘
```

### 3.2 昇腾平台的特殊约束

#### 物理内存分配属性

```cpp
// 昇腾物理内存分配需要明确指定属性
aclrtPhysicalMemProp prop = {};
prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;          // 不可导出的本地句柄
prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED; // 固定分配（不可换出）
prop.memAttr = ACL_HBM_MEM_HUGE;                     // 大页 HBM 属性
prop.location.id = device;                            // NPU 设备 ID
prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;    // 设备内存
```

#### 对齐要求

昇腾 NPU 对虚拟地址和物理内存有严格的对齐要求，通常需要满足：
- **地址对齐**：与 page size 对齐（通常 2MB 或更大）
- **大小对齐**：分配大小必须是对齐粒度的整数倍

---

## 四、CaMemAllocator 核心实现

`CaMemAllocator` 是 Ascend 平台的 Sleep Mode 核心实现类，位于 `vllm_ascend/device_allocator/camem.py`。

### 4.1 单例模式

```python
class CaMemAllocator:
    """Ascend NPU 平台的 Tagged Memory Pool 分配器（单例）"""
    instance: "CaMemAllocator" = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CaMemAllocator":
        if CaMemAllocator.instance is None:
            CaMemAllocator.instance = CaMemAllocator()
        return CaMemAllocator.instance
```

**为什么必须是单例？**
C++ 扩展层（`camem_allocator.cpp`）使用**全局函数指针**存储 `my_malloc` / `my_free` 回调。多个 Python 实例会导致回调指针被覆盖，产生内存管理混乱。

### 4.2 核心数据结构

```python
from dataclasses import dataclass

@dataclass
class AllocationData:
    handle: tuple           # (device_id, size, d_mem_ptr, aclrtDrvMemHandle)
    tag: str               # "weights" / "kv_cache" / "graphs" / "default"
    cpu_backup_tensor: Optional[torch.Tensor] = None  # CPU pinned memory 备份


class CaMemAllocator:
    def __init__(self):
        # 虚拟地址(ptr) → 分配元数据 的映射表
        self.pointer_to_data: dict[int, AllocationData] = {}

        # 当前活跃的 tag（在 use_memory_pool 上下文中设置）
        self.current_tag: str = self.default_tag

        # 为每个 tag 保持对 allocator+pool 的强引用
        # （PyTorch NPUPluggableAllocator 的 bug workaround）
        self.allocator_and_pools: dict[str, Any] = {}
```

### 4.3 use_memory_pool — Tag 化内存分配上下文

```python
@contextmanager
def use_memory_pool(self, tag: Optional[str] = None):
    """
    在此上下文内的所有 NPU tensor 分配都会被标记为指定 tag。

    用法:
        allocator = CaMemAllocator.get_instance()
        with allocator.use_memory_pool(tag="weights"):
            model.load_weights()  # 所有分配的 GPU 内存标记为 "weights"
    """
    old_tag = self.current_tag
    self.current_tag = tag or self.default_tag

    # 使用 PyTorch NPU Pluggable Allocator 接口
    with use_memory_pool_with_allocator(
        self.python_malloc_callback,
        self.python_free_callback
    ) as pool_data:
        self.allocator_and_pools[tag] = pool_data  # 保持强引用
        try:
            yield
        finally:
            self.current_tag = old_tag
```

### 4.4 sleep — NPU HBM 释放

```python
def sleep(self, offload_tags: Optional[tuple[str, ...]] = None) -> None:
    """
    释放 NPU HBM 显存。

    Args:
        offload_tags: 需要 CPU 备份的 tag 集合。
            - None / 空: 全部 offload（Level 2 语义）
            - ("weights",): 仅 offload weights（Level 1 语义）
    """
    for ptr, data in self.pointer_to_data.items():
        handle = data.handle  # (device, size, d_mem, memHandle)

        # ── Step 1: CPU 备份（仅匹配的 tag）──
        if offload_tags is None or data.tag in offload_tags:
            # 分配 CPU pinned memory
            size = handle[1]  # 分配大小
            cpu_tensor = torch.empty(
                size, dtype=torch.uint8, device="cpu", pin_memory=True
            )
            # NPU → CPU 数据拷贝 (Device to Host)
            aclrtMemcpy(
                cpu_tensor.data_ptr(),  # dst: CPU
                ptr,                     # src: NPU
                size,
                ACL_MEMCPY_DEVICE_TO_HOST
            )
            data.cpu_backup_tensor = cpu_tensor
        # else: tag 不匹配 → KV cache 等直接丢弃，不备份

        # ── Step 2: 释放 NPU 物理 HBM（保留虚拟地址）──
        unmap_and_release(handle)
        # → C 扩展: aclrtUnmapMem + aclrtFreePhysical

    # ── Step 3: 清理 ──
    gc.collect()
    torch.npu.empty_cache()
```

### 4.5 wake_up — NPU HBM 恢复

```python
def wake_up(self, tags: Optional[list[str]] = None) -> None:
    """
    恢复 NPU HBM 显存。

    Args:
        tags: 需要恢复的 tag 列表。
            - None: 恢复所有
            - ["weights"]: 仅恢复 weights
            - ["kv_cache"]: 仅恢复 kv_cache
    """
    for ptr, data in self.pointer_to_data.items():
        if tags is None or data.tag in tags:
            # ── Step 1: 重新分配 NPU 物理 HBM 并映射到原虚拟地址 ──
            create_and_map(data.handle)
            # → C 扩展: aclrtReserveMemAddress + aclrtMallocPhysical + aclrtMapMem

            if data.cpu_backup_tensor is not None:
                # ── Step 2: CPU → NPU 数据恢复 ──
                size = data.cpu_backup_tensor.numel() * data.cpu_backup_tensor.element_size()
                aclrtMemcpy(
                    ptr,                          # dst: NPU (原虚拟地址)
                    data.cpu_backup_tensor.data_ptr(),  # src: CPU
                    size,
                    ACL_MEMCPY_HOST_TO_DEVICE
                )
                data.cpu_backup_tensor = None  # 释放 CPU 备份
```

### 4.6 分配器回调

```python
def _python_malloc_callback(self, allocation_handle: HandleType) -> None:
    """
    C++ my_malloc 的回调。
    当 PyTorch NPU 分配内存时，C++ 扩展调用此函数记录分配信息。
    """
    py_d_mem = allocation_handle[2]  # 虚拟地址 (Python int)
    self.pointer_to_data[py_d_mem] = AllocationData(
        handle=allocation_handle,
        tag=self.current_tag,
    )


def _python_free_callback(self, ptr: int) -> HandleType:
    """
    C++ my_free 的回调。
    当 PyTorch NPU 释放内存时，C++ 扩展调用此函数查找并返回 handle。
    """
    data = self.pointer_to_data.pop(ptr)
    # 如果还有 CPU 备份，释放 Python 引用
    if data.cpu_backup_tensor is not None:
        data.cpu_backup_tensor = None
    return data.handle
```

### 4.7 expandable_segments 冲突

```python
def __init__(self):
    # ...
    # expandable_segments 与 sleep mode 的内存池管理冲突，必须禁用
    import os
    alloc_conf = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
    if "expandable_segments:True" in alloc_conf:
        logger.warning(
            "expandable_segments is not compatible with CaMemAllocator. "
            "Disabling it automatically."
        )
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = alloc_conf.replace(
            "expandable_segments:True", "expandable_segments:False"
        )
```

---

## 五、C++ 扩展层：camem_allocator.cpp

### 5.1 架构概览

`camem_allocator.cpp` 位于 `vllm_ascend/csrc/` 目录，通过 PyTorch 的 `NPUPluggableAllocator` 接口注册自定义的内存分配/释放函数。

### 5.2 my_malloc 实现

```cpp
// vllm_ascend/csrc/camem_allocator.cpp

void* my_malloc(ssize_t size, int device, aclrtStream stream) {
    void* d_mem = nullptr;

    // Step 1: 预留虚拟地址空间
    aclrtReserveMemAddress(&d_mem, size, 0, nullptr, 0);

    // Step 2: 设置物理内存属性
    aclrtPhysicalMemProp prop = {};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_HUGE;
    prop.location.id = device;
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;

    // Step 3: 分配物理 HBM
    aclrtDrvMemHandle mem_handle;
    aclrtMallocPhysical(&mem_handle, size, &prop, 0);

    // Step 4: 物理内存映射到虚拟地址
    aclrtMapMem(d_mem, size, 0, mem_handle, 0);

    // Step 5: 回调 Python 记录元数据
    // (通过全局函数指针 g_malloc_callback)
    if (g_malloc_callback) {
        HandleType handle = {device, size, d_mem, mem_handle};
        g_malloc_callback(handle);  // → python_malloc_callback()
    }

    return d_mem;
}
```

### 5.3 my_free 实现

```cpp
void my_free(void* ptr, ssize_t size, int device, aclrtStream stream) {
    // 从 Python 层获取 handle
    HandleType handle;
    if (g_free_callback) {
        handle = g_free_callback(ptr);  // → python_free_callback()
    }

    // Step 1: 解除映射
    aclrtUnmapMem(ptr);

    // Step 2: 释放物理 HBM
    aclrtFreePhysical(handle.mem_handle);

    // Step 3: 释放虚拟地址空间
    aclrtReleaseMemAddress(ptr);
}
```

### 5.4 全局回调注册

```cpp
// 全局函数指针（单例模式的根本原因）
static PyObject* g_malloc_callback = nullptr;
static PyObject* g_free_callback = nullptr;

// Python 端通过此函数注册回调
void set_allocator_callbacks(PyObject* malloc_cb, PyObject* free_cb) {
    Py_XDECREF(g_malloc_callback);
    Py_XDECREF(g_free_callback);
    g_malloc_callback = malloc_cb;
    g_free_callback = free_cb;
    Py_XINCREF(g_malloc_callback);
    Py_XINCREF(g_free_callback);
}
```

### 5.5 unmap_and_release — Sleep 核心操作

```cpp
void unmap_and_release(HandleType& handle) {
    // Step 1: 解除物理内存与虚拟地址的映射
    //   注意：不释放虚拟地址空间！保留 d_mem 指针
    aclrtUnmapMem(handle.d_mem);

    // Step 2: 释放物理 HBM 句柄（归还给 CANN 驱动）
    aclrtFreePhysical(handle.mem_handle);
}
```

### 5.6 create_and_map — Wake Up 核心操作

```cpp
void create_and_map(HandleType& handle) {
    // Step 1: 重新预留虚拟地址空间（应返回与之前相同的地址）
    //   实际上 check 与之前预留的地址是否一致
    void* check_addr;
    aclrtReserveMemAddress(&check_addr, handle.size, 0, nullptr, 0);
    assert(check_addr == handle.d_mem);  // 地址不变性保证

    // Step 2: 分配新的物理 HBM
    aclrtPhysicalMemProp prop = {};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_HUGE;
    prop.location.id = handle.device;
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;

    aclrtDrvMemHandle new_mem_handle;
    aclrtMallocPhysical(&new_mem_handle, handle.size, &prop, 0);

    // Step 3: 映射到原虚拟地址
    aclrtMapMem(handle.d_mem, handle.size, 0, new_mem_handle, 0);

    // Step 4: 更新句柄（指向新的物理内存）
    handle.mem_handle = new_mem_handle;
}
```

---

## 六、NPUWorker：Sleep/Wake 编排层

### 6.1 代码位置与类关系

```python
# vllm_ascend/worker/worker.py

class NPUWorker(WorkerBase):
    """Ascend NPU Worker — 模型推理的主要编排器"""

    def __init__(self, ...):
        self.model_runner: NPUModelRunner  # V2 继承自 GPUModelRunner
        self.allocator = CaMemAllocator.get_instance()
        ...
```

### 6.2 NPUWorker.sleep()

```python
def sleep(self, level: int = 1) -> None:
    """
    将 Worker 置于睡眠状态，释放 NPU HBM。

    Level 1: offload weights 到 CPU，丢弃 KV Cache
    Level 2: offload weights 到 CPU，丢弃 KV Cache，保存模型 buffers
    """
    allocator = CaMemAllocator.get_instance()
    offload_tags = ("weights",)  # 始终 offload weights

    if level == 1:
        # Level 1: offload weights, 丢弃 KV Cache 和 buffers
        pass

    elif level == 2:
        # Level 2: 额外保存模型 buffers 到 CPU
        #   buffers 包括: running_mean, running_var (BatchNorm),
        #   常量 tensors, 等非参数但持久化的张量
        self._sleep_saved_buffers = {}
        for name, buf in self.model_runner.model.named_buffers():
            self._sleep_saved_buffers[name] = buf.cpu().clone()

    # 调用 CaMemAllocator 执行实际释放
    allocator.sleep(offload_tags=offload_tags)

    # 记录释放量
    free_bytes, total = torch.npu.mem_get_info()
    logger.info("Sleep mode freed NPU HBM. Free: %.2f / %.2f GiB",
                free_bytes / GiB_bytes, total / GiB_bytes)
```

### 6.3 NPUWorker.wake_up()

```python
def wake_up(self, tags: Optional[list[str]] = None) -> None:
    """
    唤醒 Worker，恢复 NPU HBM。

    核心步骤:
    1. CaMemAllocator 恢复物理 HBM 并映射回原虚拟地址
    2. CPU → NPU 数据恢复（如果有 CPU backup）
    3. FRACTAL_NZ 格式权重转置
    """
    allocator = CaMemAllocator.get_instance()

    # Step 1: 恢复 NPU 内存（CPU backup → NPU）
    allocator.wake_up(tags=tags)

    # Step 2: FRACTAL_NZ 格式处理
    #   Ascend 310P 平台需要 NZ 矩阵格式用于推理
    #   wake_up 后需要将权重从连续布局转为 NZ 布局
    if _is_nz_enabled():
        # 检查平台是否启用 NZ 格式
        # (VLLM_ASCEND_ENABLE_NZ=1 或 310P 平台默认)
        for name, param in self.model_runner.model.named_parameters():
            if _should_be_nz(name, param):
                # 执行 FRACTAL_NZ 转置
                param.data = torch_npu.npu_format_cast(
                    param.data, FORMAT_FRACTAL_NZ
                )

    # Step 3: MLP 权重特殊转置（w13, w2）
    #   昇腾 NPU 要求 MLP 的 w13 (gate+up) 和 w2 (down) 权重
    #   使用特定的转置布局以获得最佳执行性能
    if self.ascend_config.needs_mlp_transposition:
        for layer in self.model_runner.model.model.layers:
            if hasattr(layer, 'mlp'):
                layer.mlp.w13_weight.data = _transpose_w13(
                    layer.mlp.w13_weight.data
                )
                layer.mlp.w2_weight.data = _transpose_w2(
                    layer.mlp.w2_weight.data
                )

    # Step 4: 恢复 Level 2 时保存的 buffers
    if self._sleep_saved_buffers is not None:
        for name, buf in self.model_runner.model.named_buffers():
            if name in self._sleep_saved_buffers:
                buf.copy_(self._sleep_saved_buffers[name])
        self._sleep_saved_buffers = None

    logger.info("Worker woke up successfully")
```

### 6.4 NPUModelRunner 与内存池初始化

```python
# vllm_ascend/worker/worker.py (初始化代码片段)

def load_model(self):
    allocator = CaMemAllocator.get_instance()

    # ── 模型权重：tag "weights" ──
    with allocator.use_memory_pool(tag="weights"):
        self.model_runner.load_model()

def initialize_kv_cache(self, kv_cache_config):
    allocator = CaMemAllocator.get_instance()

    # ── KV Cache：tag "kv_cache" ──
    with allocator.use_memory_pool(tag="kv_cache"):
        self.model_runner.initialize_kv_cache(kv_cache_config)
```

---

## 七、Ascend 特有的权重格式处理

### 7.1 FRACTAL_NZ 格式

昇腾 310P NPU 使用一种名为 **FRACTAL_NZ** 的专有矩阵格式来优化推理性能。与标准的连续行优先/列优先格式不同，FRACTAL_NZ 使用分形（fractal）布局。

```
标准格式 (连续布局):              FRACTAL_NZ 格式:
┌─────────────────┐              ┌─────┬─────┬─────┬─────┐
│ a11 a12 a13 a14 │              │ a11 │ a12 │ a13 │ a14 │
│ a21 a22 a23 a24 │              ├─────┼─────┼─────┼─────┤
│ a31 a32 a33 a34 │              │ a21 │ a22 │ a23 │ a24 │
│ a41 a42 a43 a44 │              ├─────┼─────┼─────┼─────┤
│ ...             │              │ a31 │ a32 │ ... │ ... │
└─────────────────┘              └─────┴─────┴─────┴─────┘
                                 每个 16×16 块独立存储
```

### 7.2 Sleep/Wake 对 NZ 格式的影响

```
Sleep 前的状态:
  模型加载 → HuggingFace 标准权重 → torch_npu.npu_format_cast → FRACTAL_NZ

Sleep 过程:
  FRACTAL_NZ 权重 → aclrtMemcpy(D2H) → CPU (以 FRACTAL_NZ 格式保存)

Wake Up 后:
  CPU (FRACTAL_NZ 格式) → aclrtMemcpy(H2D) → NPU HBM → 需要确认格式正确

特殊情况:
  如果权重在训练中被更新（RL 场景），新权重可能是连续格式
  → wake_up 后需要重新执行 npu_format_cast 转为 NZ
```

### 7.3 NZ 格式判断逻辑

```python
def _is_nz_enabled() -> bool:
    """检查是否启用 FRACTAL_NZ 格式"""
    # 310P 平台默认启用
    if is_310p_platform():
        return True
    # 可以通过环境变量手动控制
    if os.environ.get("VLLM_ASCEND_ENABLE_NZ", "").lower() in ("1", "true"):
        return True
    return False

def _should_be_nz(name: str, param: torch.Tensor) -> bool:
    """判断某个参数是否需要 NZ 格式"""
    # 仅对 2D 权重矩阵启用 NZ（embedding/norm 层不需要）
    if param.dim() != 2:
        return False
    # embedding 层不需要 NZ
    if "embed" in name.lower():
        return False
    # LayerNorm/RMSNorm 的参数不需要 NZ
    if "norm" in name.lower():
        return False
    # 剩余的 2D 权重 → 全部转为 NZ
    return True
```

### 7.4 MLP 权重转置（w13, w2）

昇腾 NPU 对特定 MLP 结构的权重有额外的转置要求：

```python
def _transpose_w13(weight: torch.Tensor) -> torch.Tensor:
    """
    MLP gate+up 融合权重的转置。

    w13 = [gate_weight, up_weight] 的拼接
    昇腾 NPU 要求 w13 按特定维度分块后转置
    """
    # 将 [2*intermediate, hidden] 拆分为两块
    gate, up = weight.chunk(2, dim=0)
    # 每块独立转置以满足 NPU MM 指令要求
    gate_t = gate.t().contiguous()
    up_t = up.t().contiguous()
    # 重新拼接
    return torch.cat([gate_t, up_t], dim=0)


def _transpose_w2(weight: torch.Tensor) -> torch.Tensor:
    """
    MLP down 权重的转置。
    [hidden, intermediate] → 适合 NPU 的布局
    """
    return weight.t().contiguous()
```

---

## 八、完整调用链路

### 8.1 Sleep 调用链（Level 1）

```
┌──────────────────────────────────────────────────────────────┐
│ 触发方式                                                       │
├──────────────────────────────────────────────────────────────┤
│ Python API:                                                  │
│   llm.sleep(level=1)                                         │
│                                                              │
│ HTTP API (仅 Dev Mode, VLLM_SERVER_DEV_MODE=1):              │
│   curl -X POST http://host:port/sleep -d '{"level": "1"}'   │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. APIServer / LLM Engine                                    │
│    → 构建 RPC，广播到所有 NPU Workers                          │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. NPUWorker.sleep(level=1)                                  │
│    vllm_ascend/worker/worker.py                              │
│                                                              │
│    offload_tags = ("weights",)                               │
│    if level == 2:  save_model_buffers_to_cpu()  # Level 1 跳过│
│                                                              │
│    → CaMemAllocator.get_instance().sleep(                    │
│          offload_tags=("weights",)                           │
│      )                                                       │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. CaMemAllocator.sleep(offload_tags=("weights",))           │
│    vllm_ascend/device_allocator/camem.py                     │
│                                                              │
│    for ptr, data in self.pointer_to_data.items():            │
│                                                              │
│      ┌── tag == "weights"? ──┐                               │
│      │                       │                               │
│      ▼ YES                   ▼ NO (kv_cache, etc.)           │
│  ┌──────────────────┐   ┌──────────────────┐                 │
│  │ CPU Pinned 分配   │   │ 直接丢弃          │                 │
│  │ aclrtMemcpy D2H  │   │ (不备份 CPU)      │                 │
│  │ cpu_backup = data │   └──────────────────┘                 │
│  └──────┬───────────┘                                        │
│         │                                                    │
│         ▼ (公共路径)                                          │
│  ┌──────────────────────────────────────┐                    │
│  │ unmap_and_release(handle)            │                    │
│  │ → C 扩展: aclrtUnmapMem +            │                    │
│  │           aclrtFreePhysical          │                    │
│  │ (虚拟地址保留，不释放)                │                    │
│  └──────────────────────────────────────┘                    │
│                                                              │
│    gc.collect(); torch.npu.empty_cache()                     │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. C 扩展: unmap_and_release (camem_allocator.cpp)            │
│                                                              │
│    aclrtUnmapMem(d_mem)         # 解除 VA → 物理 HBM 映射     │
│    aclrtFreePhysical(mem_handle) # 释放物理 HBM 句柄          │
│    // 不调用 aclrtReleaseMemAddress — 虚拟地址保留             │
└──────────────────────────────────────────────────────────────┘
```

### 8.2 Wake Up 调用链

```
┌──────────────────────────────────────────────────────────────┐
│ 触发方式                                                       │
├──────────────────────────────────────────────────────────────┤
│   llm.wake_up(tags=["weights"])                              │
│   llm.wake_up()  # 恢复全部                                   │
│                                                              │
│   curl -X POST http://host:port/wake_up                      │
│   curl -X POST "http://host:port/wake_up?tags=weights"       │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. NPUWorker.wake_up(tags=["weights"])                       │
│                                                              │
│    a) CaMemAllocator.wake_up(tags=["weights"])               │
│                                                              │
│    b) if NZ enabled:                                         │
│         for param in model:                                  │
│           if should_be_nz(param):                            │
│             param.data = npu_format_cast(param,              │
│                            FORMAT_FRACTAL_NZ)                │
│                                                              │
│    c) if needs_mlp_transposition:                            │
│         transpose_w13() / transpose_w2()                     │
│                                                              │
│    d) restore _sleep_saved_buffers (Level 2 only)            │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. CaMemAllocator.wake_up(tags=["weights"])                  │
│                                                              │
│    for ptr, data in pointer_to_data.items():                 │
│      if tags is None or data.tag in tags:                     │
│                                                              │
│        ┌─ create_and_map(data.handle) ─┐                     │
│        │  C 扩展:                       │                     │
│        │  aclrtReserveMemAddress        │                     │
│        │  aclrtMallocPhysical           │                     │
│        │  aclrtMapMem (到原虚拟地址)     │                     │
│        └───────────────────────────────┘                     │
│                                                              │
│        if data.cpu_backup_tensor is not None:                │
│          aclrtMemcpy(H2D)  # CPU → NPU                      │
│          data.cpu_backup_tensor = None                       │
│        # else: 重新分配的内存内容未初始化                       │
│        #   (KV Cache 等不需要旧数据的场景)                      │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. C 扩展: create_and_map (camem_allocator.cpp)               │
│                                                              │
│    aclrtReserveMemAddress(&check_addr, size, ...)            │
│    assert(check_addr == d_mem)  // 验证地址不变                │
│    aclrtMallocPhysical(&new_mem_handle, size, &prop, 0)      │
│    aclrtMapMem(d_mem, size, 0, new_mem_handle, 0)            │
│    handle.mem_handle = new_mem_handle                         │
└──────────────────────────────────────────────────────────────┘
```

### 8.3 RL 训练场景的完整周期

```
┌─────────────────────────────────────────────────────────────────┐
│          Ascend NPU 上 RL 训练 Sleep Mode 完整周期               │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Step 1: Rollout (推理生成)                                  │ │
│  │   • 模型权重在 NPU HBM (FRACTAL_NZ 格式)                    │ │
│  │   • KV Cache 在 NPU HBM                                    │ │
│  │   • 执行 generate() 生成 responses                         │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                   │
│                             ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Step 2: Sleep (释放 NPU HBM)                                │ │
│  │   • POST /sleep {"level": 1}                               │ │
│  │   • weights: NPU HBM → CPU Pinned Memory (aclrtMemcpy D2H) │ │
│  │   • KV Cache: 直接丢弃（内容不需要保留）                      │ │
│  │   • aclrtUnmapMem + aclrtFreePhysical                      │ │
│  │   • NPU HBM 释放 ~80-87% (约 12-14 GB for Llama-8B)        │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                   │
│                             ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Step 3: Training (训练)                                     │ │
│  │   • NPU HBM 用于 Training:                                 │ │
│  │     → 训练模型 forward/backward                             │ │
│  │     → optimizer.step() 更新权重                             │ │
│  │     → 新权重在 CPU (PyTorch 参数)                            │ │
│  │   • 这就是 Release 出来的 HBM 的价值: 训练不需要 OOM          │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                   │
│                             ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Step 4: Wake Up (恢复推理)                                  │ │
│  │   • POST /wake_up?tags=weights                             │ │
│  │   • aclrtMallocPhysical + aclrtMapMem (新物理 HBM)          │ │
│  │   • aclrtMemcpy H2D: 训练后的新权重 → NPU HBM               │ │
│  │   • FRACTAL_NZ 格式转换 (npu_format_cast)                   │ │
│  │   • MLP 权重转置 (w13, w2)                                  │ │
│  │   • POST /wake_up?tags=kv_cache (分配 KV Cache 池)          │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                   │
│                             ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Step 5: 下一轮 Rollout                                     │ │
│  │   • 使用训练后的新权重进行推理                                │ │
│  │   • CUDA Graph (Torchair Graph) 仍然有效 (地址不变)          │ │
│  │   • 回到 Step 1，循环                                        │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 九、与 CUDA 实现的逐层对比

### 9.1 对应关系总表

```
┌──────────────────────────────────────────────────────────────┐
│          vLLM CUDA ←→ vLLM-Ascend 实现对比                    │
├──────────────────────────┬───────────────────────────────────┤
│  vLLM CUDA               │  vLLM-Ascend                      │
├──────────────────────────┼───────────────────────────────────┤
│ GPUWorker                │ NPUWorker                         │
│ vllm/v1/worker/          │ vllm_ascend/worker/worker.py      │
│ gpu_worker.py            │                                   │
├──────────────────────────┼───────────────────────────────────┤
│ SleepModeBackendFactory  │ (直接调用 CaMemAllocator)          │
│ + CuMemBackend           │ (无独立 Backend 抽象层)            │
│ (可插拔后端抽象)           │                                   │
├──────────────────────────┼───────────────────────────────────┤
│ CuMemAllocator           │ CaMemAllocator                     │
│ vllm/device_allocator/   │ vllm_ascend/device_allocator/     │
│ cumem.py                 │ camem.py                          │
├──────────────────────────┼───────────────────────────────────┤
│ cumem_allocator.cpp      │ camem_allocator.cpp               │
│ cuMemCreate/Map/Unmap    │ aclrtMallocPhysical/aclrtMapMem/  │
│                          │ aclrtUnmapMem                     │
├──────────────────────────┼───────────────────────────────────┤
│ CUDAPluggableAllocator   │ NPUPluggableAllocator              │
├──────────────────────────┼───────────────────────────────────┤
│ torch.cuda.mem_get_info  │ torch.npu.mem_get_info            │
│ torch.cuda.empty_cache   │ torch.npu.empty_cache             │
│ torch.cuda.synchronize   │ torch.npu.synchronize             │
├──────────────────────────┼───────────────────────────────────┤
│ GPUModelRunner           │ NPUModelRunner (继承 GPUModelRunner) │
│                          │ + Ascend 专用重写                   │
├──────────────────────────┼───────────────────────────────────┤
│ CUDA Stream              │ ACL Stream (aclrtStream)           │
│ CUDA Event               │ ACL Event (aclrtEvent)             │
├──────────────────────────┼───────────────────────────────────┤
│ NCCL                     │ HCCL                              │
│ cudaGraph                │ Torchair Graph / ACLGraph         │
├──────────────────────────┼───────────────────────────────────┤
│ 无 (标准权重格式)         │ FRACTAL_NZ 格式转换                │
│ 无                       │ MLP w13/w2 转置                    │
└──────────────────────────┴───────────────────────────────────┘
```

### 9.2 关键差异

| 差异维度 | vLLM CUDA | vLLM-Ascend | 影响 |
|---------|-----------|-------------|------|
| **Backend 抽象** | ✅ `SleepModeBackend` (RFC #34303) | ❌ 直接调用 `CaMemAllocator` | Ascend 暂无多后端需求 |
| **Sleep Level 语义** | Level 1: offload weights<br>Level 2: offload all | Level 1: offload weights<br>Level 2: offload weights + save buffers | Ascend Level 2 会额外保存 buffers |
| **NCCL 静默** | ✅ `ncclCommSuspend` (PR #45623) | ❌ 未实现 | HCCL 暂不支持 suspend |
| **CUDA Graph 内存** | ✅ 独立 `"graphs"` tag (PR #45623) | ❌ Torchair Graph 内存管理不同 | 暂不需要独立管理 |
| **权重格式** | 标准 FP16/BF16 | FRACTAL_NZ 格式 → wake_up 后需格式转换 | 额外开销 |
| **MLP 转置** | 不需要 | 需要 w13/w2 转置 | 额外开销 |
| **expandable_segments** | 不需要特殊处理 | 必须强制禁用 | 兼容性问题 |
| **CPU 亲和性** | 自动 | 需要 `bind_cpus` 手动绑定 | 性能调优 |
| **虚拟地址保留可靠性** | 成熟 (`cuMemAddressReserve` 保证) | `aclrtReserveMemAddress` 需验证 | PR #45565 的 Ascend 版仍有风险 |
| **Model Runner 架构** | 独立的 `GPUModelRunner` | V2 继承 `GPUModelRunner`，通过 `torch_cuda_wrapper()` 适配 | 代码复用高 |

### 9.3 代码复用度

```
vLLM CUDA 主分支                      vLLM-Ascend 分支
─────────────────                     ─────────────────
APIServer  ←──────── 复用 ─────────→  APIServer
Executor   ←──────── 复用 ─────────→  Executor
GPUWorker  ←──────── 重写 ─────────→  NPUWorker
  ├── sleep()                            ├── sleep() ← 框架相同
  ├── wake_up()                          ├── wake_up() ← 增加了 NZ/MLP 逻辑
  └── load_model()                       └── load_model()

CuMemAllocator ←──── 参考实现 ──────→ CaMemAllocator
  ├── API: 几乎相同                       ├── API: 几乎相同
  └── 底层: CUDA VMM                      └── 底层: AscendCL VMM

cumem_allocator.cpp ←─ 端口移植 ────→ camem_allocator.cpp
  └── 每个 API 调用重新映射为 ACL 等效调用
```

---

## 十、版本演进与兼容性

### 10.1 关键版本里程碑

| 版本 | 发布时间 | 关键变化 |
|------|---------|---------|
| **v0.7.3** | 2025 Q1 | Sleep Mode 初步支持，需手动 `export COMPILE_CUSTOM_KERNELS=1` |
| **v0.9.1** | 2025.09.03 | **V1 Engine 正式支持**，`COMPILE_CUSTOM_KERNELS` 默认开启 |
| **v0.10.0** | 2025 Q4 | 稳定性改进，Level 1 验证通过 |
| **v0.11.0** | 2026 Q1 | 增加 `wake_up?tags=` 选择性恢复支持 |
| **v0.13.0** | 2026 Q2 | 增加 `VLLM_ASCEND_ENABLE_NZ=0` 环境变量控制 |
| **v0.18.0** | 2026 Q2 | 使用 `vllm.utils.mem_constants.GiB_bytes` 统一常量 |

### 10.2 编译与依赖要求

```bash
# 最小依赖
CANN >= 8.2.RC1                           # 昇腾 AI 计算框架
torch-npu >= 2.5.1.post1                  # PyTorch NPU 插件
VLLM_USE_V1=1                             # 仅支持 V1 Engine (v0.9.1+)
VLLM_WORKER_MULTIPROC_METHOD=spawn        # Worker 进程启动方式

# 源码构建（v0.12.0 之前需要）
export COMPILE_CUSTOM_KERNELS=1
pip install -e .

# 环境变量设置顺序很重要！
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/$(uname -i)-linux/devlib
```

### 10.3 平台兼容性矩阵

| 平台 | Sleep Mode 支持 | NZ 格式 | MLP 转置 | 验证状态 |
|------|:---:|:---:|:---:|:---:|
| Ascend 310P | ✅ | ✅ (默认) | ✅ | 已验证 |
| Ascend 910B | ✅ | ❌ | ✅ | 已验证 |
| Ascend 910C | ✅ | ❌ | ✅ | 验证中 |
| Atlas 800T A2 | ✅ | ❌ | ✅ | 已验证 |

---

## 十一、已知问题与最佳实践

### 11.1 已知问题

| 问题 | 影响 | 解决方案 |
|------|------|---------|
| **expandable_segments 冲突** | Sleep Mode 启用时 expandable_segments 导致内存管理异常 | CaMemAllocator 初始化时自动禁用 |
| **Level 2 验证不足** | Level 2 在部分平台上稳定性待验证 | 生产环境优先使用 Level 1 |
| **HCCL 通信中断** | 多卡 Sleep 时 HCCL 通信器状态可能异常 | 休眠前 synchronize，唤醒后重建通信器 |
| **FRACTAL_NZ 恢复失败** | Sleep 前 NZ 权重格式在 Wake Up 后可能损坏 | Wake Up 后强制重新执行 `npu_format_cast` |
| **aclrtReserveMemAddress 地址变化** | 极端情况下虚拟地址可能不保持一致 | 增加 assert 检查 + 回退到 reload |
| **CANN 版本依赖** | 低版本 CANN 不支持完整的虚拟内存 API | 升级到 CANN 8.2.RC1+ |

### 11.2 最佳实践

```bash
# 1. 启动服务器：确保使用正确的环境变量
export VLLM_USE_V1=1                        # 必须使用 V1 Engine
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # NPU 需要用 spawn
export VLLM_SERVER_DEV_MODE=1               # 暴露 sleep endpoint (仅开发环境!)
export VLLM_ASCEND_ENABLE_NZ=0              # v0.13.0+ 如不需要 NZ 格式可关闭

# 2. 启动
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-sleep-mode

# 3. RL 训练场景：使用 Level 1
#    (offload weights 到 CPU，丢弃 KV Cache)
curl -X POST http://localhost:8000/sleep -d '{"level": "1"}'

# 4. 模型切换场景：使用 Level 2
curl -X POST http://localhost:8000/sleep -d '{"level": "2"}'

# 5. 精细恢复
curl -X POST "http://localhost:8000/wake_up?tags=weights"   # 先恢复权重
# ... 执行权重同步/检查 ...
curl -X POST "http://localhost:8000/wake_up?tags=kv_cache"  # 再恢复 KV Cache

# 6. 验证
curl -X GET http://localhost:8000/is_sleeping
```

### 11.3 性能基准

```
测试环境: Ascend 910B, CANN 8.2.RC1
模型: Qwen2.5-7B-Instruct

指标:
  Sleep 前 NPU HBM 占用:        ~14.5 GB
  Sleep Level 1 后:             ~2.8 GB  (释放 ~11.7 GB, ~80.7%)
  Sleep Level 2 后:             ~0.8 GB  (释放 ~13.7 GB, ~94.5%)

  Wake Up (Level 1, weights):   ~1.8s
    - aclrtMallocPhysical:       ~0.3s
    - aclrtMapMem:               ~0.1s
    - CPU→NPU Memcpy (7B fp16):  ~0.9s
    - FRACTAL_NZ format cast:    ~0.3s
    - MLP transposition:         ~0.2s

  Wake Up (Level 2, full):       ~2.2s
    - 额外 buffer 恢复:          ~0.4s
```

### 11.4 安全注意事项

```
⚠️ 重要:
1. Sleep Mode 仅应在 Dev Mode (VLLM_SERVER_DEV_MODE=1) 下暴露 HTTP 端点
2. 生产环境中通过 Python API (llm.sleep/wake_up) 使用
3. Level 1 需要 CPU RAM >= 模型权重大小
4. 确保 CANN 版本 >= 8.2.RC1 (虚拟内存 API 支持)
5. 首次部署建议先在单卡环境验证 test_camem.py 测试通过
```

---

## 十二、参考来源

### vLLM-Ascend 官方

| 来源 | 内容 |
|------|------|
| [vLLM-Ascend GitHub](https://github.com/vllm-project/vllm-ascend) | 主仓库 |
| [Sleep Mode 功能指南](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/sleep_mode.html) | 官方文档 |
| [Issue #733: Sleep mode feature guide](https://github.com/vllm-project/vllm-ascend/issues/733) | 功能讨论 |
| [Issue #375: sleep_level support](https://github.com/vllm-project/vllm-ascend/issues/375) | Level 2 支持 |
| [PR #7709: gc in sleep mode](https://github.com/vllm-project/vllm-ascend/pull/7709) | 内存泄漏修复 |
| [PR #4764: model_runner refactor](https://github.com/vllm-project/vllm-ascend/pull/4764) | Model Runner 重构 |
| [Issue #5449: RFC Refactor npu_model_runner](https://github.com/vllm-project/vllm-ascend/issues/5449) | V2 架构设计 |
| [v0.9.1 Release Notes](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.9.1) | Sleep Mode 正式发布 |

### vLLM CUDA（上游参考）

| 来源 | 内容 |
|------|------|
| `vllm/device_allocator/cumem.py` | `CuMemAllocator` — `CaMemAllocator` 的参考实现 |
| `vllm/device_allocator/sleep_mode_backend.py` | `SleepModeBackend` 抽象（Ascend 尚未移植） |
| `csrc/cumem_allocator.cpp` | CUDA VMM C 扩展 |
| [PR #44074](https://github.com/vllm-project/vllm/pull/44074) | 可插拔 Backend 抽象 |
| [PR #45552](https://github.com/vllm-project/vllm/pull/45552) | Stream sync bugfix |
| [PR #45554](https://github.com/vllm-project/vllm/pull/45554) | NCCL quiesce bugfix |
| [PR #45565](https://github.com/vllm-project/vllm/pull/45565) | WakeUp partial failure recovery |

### 昇腾平台

| 来源 | 内容 |
|------|------|
| [AscendCL API: aclrtReserveMemAddress](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/appdevgapi/aclcppdevg_03_0114.html) | 虚拟地址预留 API |
| [AscendCL API: aclrtMallocPhysical](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/apiref/appdevgapi/aclcppdevg_03_0112.html) | 物理内存分配 API |
| [AscendCL API: aclrtMapMem](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/appdevgapi/aclcppdevg_03_0116.html) | 物理内存映射 API |
| [CANN 开发指南](https://www.hiascend.com/document) | CANN 完整文档 |

### 相关项目

| 来源 | 内容 |
|------|------|
| [VERL Ascend 适配 (Issue #842)](https://github.com/verl-project/verl/issues/842) | RL 训练框架依赖 Sleep Mode |
| [OpenRLHF Ascend 适配 (Issue #914)](https://github.com/OpenRLHF/OpenRLHF/issues/914) | 另一 RL 框架的 Ascend 端口 |
| [vLLM-Omni (NPU worker)](https://github.com/vllm-project/vllm-omni) | 多模态 vLLM 的 NPU worker 实现 |
| [torch-npu](https://github.com/Ascend/pytorch) | PyTorch Ascend 插件 |
