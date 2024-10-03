# AMD GPU HAL Driver (`amdgpu`)

**NOTE**: the code is the authoritative documentation source. This document is an overview of the implementation and should be treated as informational only. See the linked files for details.

## Quick Start

Configure CMake with the following options:
```sh
-DIREE_BUILD_COMPILER=ON
-DIREE_TARGET_BACKEND_ROCM=ON
-DIREE_HAL_DRIVER_AMDGPU=ON
-DIREE_HIP_TEST_TARGET_CHIP=gfx1100
```

**TODO(benvanik):** make a cmake variable for controlling target architectures.
Today you *must* add (or replace locally) your architecture to the device [BUILD.bazel](/runtime/src/iree/hal/drivers/amdgpu/device/BUILD.bazel) and run bazel-to-cmake (`python build_tools/bazel_to_cmake/bazel_to_cmake.py`).

Use `amdgpu` to specify devices at runtime:
```sh
# Single logical device with all available physical devices:
iree-run-module --device=amdgpu
# Device ordinal 0 (danger, this may change across reboots):
iree-run-module --device=amdgpu:0
# Device with a stable UUID for a device:
iree-run-module --device=amdgpu://GPU-0e12865a3bf5b7ab
# Single logical device with the two devices given by their UUIDs:
iree-run-module --device=amdgpu://GPU-0e12865a3bf5b7ab,GPU-89e8bdf59a10cf6d
# Single logical device with physical devices with ordinals 2 and 3:
ROCR_VISIBLE_DEVICES=2,3 iree-run-module --device=amdgpu
# Two logical devices with two physical devices each:
iree-run-module --device=amdgpu://0,1 --device=amdgpu://2,3
```

Use `amdgpu` to specify the AMDGPU target when compiling programs:
```sh
iree-compile --iree-hal-target-device=amdgpu ...
```

## Build Notes

### HSA/ROCR Dependency

We maintain a fork of the HSA headers required for compilation as [third_party/hsa-runtime-headers/](https://github.com/iree-org/hsa-runtime-headers). This fork may also contain tweaks not yet upstreamed required to use the headers in our build.

We require that at runtime a dynamic library with the name `libhsa-runtime64.so` exists on the path. This can be overridden programmatically when constructing the driver, via the `--amdgpu_libhsa_search_path=` flag if using the command line tools, via the `IREE_HAL_AMDGPU_LIBHSA_PATH` environment variable, or by just adding a directory containing the file to `PATH`.

It's recommended that developers check out a copy of the [ROCR-Runtime](https://github.com/ROCm/ROCR-Runtime) and build it locally in whatever configuration they are using (debug/release/ASAN/etc). This allows for easier debugging and profiling (as symbols are present) and may be required to get recent features not available in platform installs. Eventually IREE will ship its own copy of the library (directly or indirectly) as part of the install packages such that only a relatively recent AMDGPU driver is required.

See [HSA/ROCR Library](#hsarocr-library) for more information on our usage.

### Device Library Compilation

**Required CMake Options**: `-DIREE_BUILD_COMPILER=ON -DIREE_TARGET_BACKEND_ROCM=ON`

**Top-level Build Target**: `iree_hal_drivers_amdgpu_device_binaries`

Currently IREE's CMake configuration must have the compiler enabled in order to build the runtime including the AMDGPU HAL implementation. This will be made better in the future (allowing for just building what we need instead of the full MLIR stack, using an existing ROCM install, etc). See [Device Library](#device-library) for more information.

There may currently be issues with CMake caching around changes to the device library headers. Unfortunately CMake does not invalidate caches on header changes - I think. Sometimes touching a .c in the device library is needed if only changing headers there.

The device library should be compiled automatically when building the AMDGPU HAL driver and gets embedded inside the runtime binary so that no additional files are required at runtime.

## Engineering Notes

### Physical Device Grouping

Much of the HAL implementation requires that all physical devices participating as part of a single logical device are "compatible." This allows attributes like queue size limits, allocation granularity, and supported ISAs to be assumed uniform. If at some point we want to allow multiple devices with differing attributes we could do so as it makes sense. As an example supporting devices with different ISAs is a small extension to select and load the appropriate per-device device library binary instead of loading it once and reusing it. With queue affinity passed on HAL calls we can also support things like unified memory import when the devices the imported buffer is requested to be used on all support unified memory even if other physical devices in the group do not.

### Host and Device Memory

Though HSA allows pointers to be accessed from the host and any device there is a non-trivial performance impact of actually accessing across devices. Tightly tuned access code that is able to be pipelined by the host or device is sometimes OK but any data dependencies across the bus can total microseconds of delay. For example if the device needs to access a data structure and must first query the pointer of the data structure from another data structure that is stored on the host the single access can take 1-10 microseconds if uncached. Our device library code runs infrequently (between tens to thousands of dispatches) and is almost guaranteed to have a cold cache which leads to expensive stalls on cache fills for the dependent data. The reverse applies to the host: device library maintained data structures such as queue entries are expensive to access from the host and always result in cache misses.

Host HAL implementation code that retains pointers to device memory should annotate any pointer that _may_ be in device memory with `IREE_AMDGPU_DEVICE_PTR`. This is a hint to the reader that indirecting through the pointer or accessing the data referenced by it may incur significant overhead. The primary pattern that requires careful attention is indirection through remote memory (host doing `some_device_ptr->other_device_ptr->value`) and in such cases we try to shadow indirected values (host doing `host_shadow_ptr->other_device_ptr->value`) or avoid dependent calculations (embedding data in structs such that the remote side need only do pointer math). A rarer pattern but one that can have even higher performance impact is RPCs where for example the device side originates a request the host side must respond to: in these cases the information that was already in device registers in order to produce the request must make it all the way to the host - if that requires ~10 cache misses walking data structures it can easily add 10's of microseconds of latency. In such cases we try to encode all of the required information in AQL packets such that single cache line fills are sufficient to avoid any dependent indirection in hot paths.

#### Memory Visibility

Allowing access to memory located on one device to another device is expensive enough to want to avoid however once access is allowed there's no measurable performance penalty for having the access shared. Performance implications arise when the memory is accessed from remote devices as there's additional bus traffic involved in each cache line fill. Our usage of cross-device memory in the HAL implementation is limited mostly to AQL queues and our custom queue data structures as those are used for RPC. Particular cases where we replicate data to ensure local access even though the memory may be easy to make accessible are hot data structures like command buffers and kernel descriptors: in such cases we want to ensure that issuing commands and dispatches are guaranteed to be local (if not cached) to the device performing the work.

#### Queue Placement

**WARNING: Experimental**

ROCR recently grew the ability to allocate queue ringbuffers in device memory instead of the default host memory via the `HSA_ALLOCATE_QUEUE_DEV_MEM` environment variable. Work is ongoing to make it part of the API so we can adjust it per-queue and store the queue structure (read/write packet IDs, etc) in device memory as well: https://github.com/ROCm/ROCR-Runtime/issues/269

Given that 99% of all of our AQL queue submissions are from the device command buffer execution and they are going to the same device we need the execution queues to be in device memory. Today we go across the system bus for every single packet insertion and that adds significant latency.

For queues submissions that happen from the host there may still be value in keeping them in host memory assuming that the firmware and hardware are designed to properly prefetch. Early testing shows that it's not particularly well optimized and that any microseconds we save on the host end up getting directly transferred to the device.

#### Ringbuffers

* See: [iree/hal/drivers/amdgpu/util/vmem.h](/runtime/src/iree/hal/amdgpu/util/vmem.h)

For ringbuffers with fixed-sized entries we follow the same design as HSA queues by having read/write indices be absolute increasing and then masked to the ringbuffer storage.

For ringbuffers with dynamically-sized entries such as tracing or kernargs we use virtual memory mapping as described in https://lo.calho.st/posts/black-magic-buffer/ to map the backing buffer memory to contiguous adjacent ranges. This allows access to any data within the ringbuffer as a contiguous virtual range even if it is wrapped in the physical backing storage. Since setting up one of these ringbuffers requires additional host work and they can't be suballocated or embedded in other data structures we prefer fixed-size entries particularly for small ringbuffers.

### Host HAL

See: [iree/hal/drivers/amdgpu/logical_device.h](/runtime/src/iree/hal/amdgpu/logical_device.h)
See: [iree/hal/drivers/amdgpu/physical_device.h](/runtime/src/iree/hal/amdgpu/physical_device.h)
See: [iree/hal/drivers/amdgpu/queue.h](/runtime/src/iree/hal/amdgpu/queue.h)

```
                   +--------------+     +-----------+
+-------------+    |+--------------+    |+-----------+    +---------------+
| logical dev |--->+| physical dev |--->+|   queue   |-+->| control queue |
+-------------+     +--------------+     +-----------+ |  +---------------+
                                                       |  +-----------------+
                                                       |  |+-----------------+
                                                       +->+| execution queue |
                                                           +-----------------+
```

Each logical HAL device (`iree_hal_device_t` implemented as `iree_hal_amdgpu_logical_device_t`) is one or more physical AMDGPU devices (`iree_hal_amdgpu_physical_device_t`) corresponding to an HSA GPU agent (`hsa_agent_t`). Each physical device may have one or more HAL queues (`iree_hal_amdgpu_queue_t`) that obey HAL semantics (namely that work scheduled on the same queue runs with some locality). Each of the HAL queues is backed by one high-priority HSA queue used for control operations (scheduling, command buffer issues, etc) and one or more HSA queues used for execution (transfers and dispatches). The topology is flexible and some sharing is allowed where documented. Upon creation the logical device instantiates the device library (`iree_hal_amdgpu_device_library_t`) on each physical device and allocates device and queue-specific data structures like the scheduler, tracing buffers, and pools.

Work scheduled against the logical HAL device is mapped to an `iree_hal_amdgpu_queue_t` by queue affinity. If the topology defines 2 physical devices each with 2 queues a total of 4 addressable HAL queue affinities are available. A queue affinity of `0b0001` would indicate that only the first queue on the first physical device should be used while `0b1111` (or `IREE_HAL_QUEUE_AFFINITY_ANY`) indicates that any queue can be used. Note that HAL resources must have been created with queue affinities including any that it will eventually be scheduled for use on and the implementation uses that to control memory visibility across devices.

#### Topology and System Instances

* See: [iree/hal/drivers/amdgpu/system.h](/runtime/src/iree/hal/amdgpu/system.h)
* See: [iree/hal/drivers/amdgpu/util/topology.h](/runtime/src/iree/hal/amdgpu/util/topology.h)

Programmatic and automatic topology detection and construction is provided by `iree_hal_amdgpu_topology_t`. Handling of environment variables (like `ROCR_VISIBLE_DEVICES`) and HAL device path parsing are handled within as well as validation of whether the selected devices are compatible for use individually or grouped together as a single logical device. Topologies are frozen once they are used to create a device and commonly queried during initialization.

Each logical device owns its own `iree_hal_amdgpu_system_t` holding the chosen topology, loaded HSA runtime, and the loaded (per-device) `iree_hal_amdgpu_device_library_t` such that two different logical devices may be configured independently and without lifetime conflicts. This is critical in hosted environments such as python where the concept of a global startup or shutdown does not exist and the garbage collector may retain logical devices for indeterminate amounts of time. The system instance is used by various subsystems to get device-agnostic information.

#### HSA/ROCR Library

See: [iree/hal/drivers/amdgpu/util/libhsa.h](/runtime/src/iree/hal/amdgpu/util/libhsa.h)
See: [iree/hal/drivers/amdgpu/util/libhsa_tables.h](/runtime/src/iree/hal/amdgpu/util/libhsa_tables.h)

The IREE runtime as commonly packaged and deployed cannot rely on dynamic libraries that are not present on the deployment target. We exclusively link dynamically against any such optional library (HIP, CUDA, Vulkan, etc) and do the same with HSA. Upon initialization of the AMDGPU HAL driver we try to discover and load the library and fail if it is not found.

Symbols defined in [libhsa_tables.h](/runtime/src/iree/hal/amdgpu/util/libhsa_tables.h) will be loaded from the library. `iree_` prefixed versions of all of the functions are defined (e.g. `hsa_amd_agents_allow_access` -> `iree_hsa_amd_agents_allow_access`) that take an `iree_hal_amdgpu_libhsa_t` pointer and call site information in order to provide the indirect call to the symbol, add tracing information, and convert HSA errors into `iree_status_t` instances. Extensions loaded via HSA's extension mechanism are not currently wrapped.

For the most part dynamic linking has little cost to us as the functions we call within the ROCR HSA implementation are usually themselves rather chatty (lots of indirection, etc). In the few cases that matter - namely HSA signal and queue operations - we inline those into our implementation so that they can be inlined into call sites.

An incomplete option is available for evaluation where ROCR can be linked statically to avoid the indirect calls. It so far has shown no measurable difference in performance thanks to the design of HSA (all hot paths are in user mode and do not require driver involvement) but is there in case it is required.

#### Blocked Allocation

* See: [iree/hal/drivers/amdgpu/util/block_pool.h](/runtime/src/iree/hal/amdgpu/util/block_pool.h)

Allocations serviced by HSA (or the kernel) can be extremely expensive by involving syscalls, page table updates, full cache flushes, and sometimes full device synchronization. The total number of allocations also has scaling limits due to code on the allocation path performing linear scans or queries across existing allocations. To control for this we have our own suballocator that is backed by fixed-size device memory blocks. Each block is sized at the recommended allocation granularity of the device and accessible to all devices that may need to access it. By using fixed-size blocks that we infrequently allocate or free (usually only on trims) we avoid introducing a significant amount of fragmentation beyond what the application running on top of the HAL introduces by its own usage. Many subsystems within the HAL share the block pools and we bucket between small and large allocations to avoid internal fragmentation within the blocks.

#### Buffer and Semaphore Pooling

* See: [iree/hal/drivers/amdgpu/buffer_pool.h](/runtime/src/iree/hal/amdgpu/buffer_pool.h)
* See: [iree/hal/drivers/amdgpu/semaphore_pool.h](/runtime/src/iree/hal/amdgpu/semaphore_pool.h)

Internal buffers and semaphores (those created and used exclusively within the AMDGPU HAL) have device-side metadata structures backing the host-side HAL `iree_hal_buffer_t` and `iree_hal_semaphore_t`. Since buffers and semaphores may be used across any physical device within a logical device group both pool types are shared across all physical devices. External HAL buffers and semaphores (those created by another HAL implementation or wrapping host resources) are not able to be accessed directly by the device and are not pooled.

#### Command Buffer Recording

* See: [iree/hal/drivers/amdgpu/command_buffer.h](/runtime/src/iree/hal/amdgpu/command_buffer.h)
* See: [iree/hal/drivers/amdgpu/device/command_buffer.h](/runtime/src/iree/hal/amdgpu/device/command_buffer.h)

A queue affinity is specified when command buffers are created to indicate which HAL queues a command buffer may be scheduled on for execution. To reduce latency and avoid additional memory consumption the implementation uses an encoder utility to track state and broadcast recorded commands to all physical devices that map to the requested queue affinity simultaneously. In cases where there is only a single physical device or a command buffer is only declared as executable on one physical device this adds no additional overhead. In cases where there are multiple physical devices the cost per device only increases by a few writes to the device.

See [Command Buffer Execution](#command-buffer-execution) for details on how command buffers are represented in memory. Essentially command buffers are programs composed of basic blocks with control instructions linking them together. Each block contains one or more commands.

When command buffers are created a transient `iree_hal_amdgpu_command_encoder_t` is allocated from the shared host pool to track the recording state and perform basic command encoding. Commands are appended to active blocks as recorded with `iree_hal_amdgpu_command_encoder_append_cmd`; this acquires space from the block to store the command and constructs the standard `iree_hal_amdgpu_device_cmd_header_t` header used by all commands. If the block capacity is exceeded the encoder will split the block in two, join them with a branch, and resume recording. There is no copy of the command stored in host memory: commands are directly written into device memory for each physical device required. Device-specific pointers are avoided where possible (and required to enable replay) and otherwise are fixed up as needed per-device (e.g. using `iree_hal_amdgpu_executable_lookup_kernel_args_for_device` to bake per-device kernel arguments into the device-specific copy of the command). Some commands require payloads larger than the fixed command size and the encoder supplies a dedicated data bump pointer allocator that can be used to broadcast the data (`iree_hal_amdgpu_command_encoder_emplace_data` as used by `iree_hal_command_buffer_update_buffer` to hold the source buffer contents).

By broadcasting while recording there's no per-command bookkeeping required and no per-command finalization. The overhead of recording a single command is the same as for 10000. HAL validation is only performed once, only the exact blocks required are allocated on each device, and upon completion the encoder state is returned to the system pool to be reused by other subsystems. Nearly all validation and lookups are performed during the recording such that the in-memory representation only needs buffer binding table resolution while issuing commands and all other information required to construct the final AQL packet is available in the cache/prefetch-friendly contiguously-allocated device local command entry.

There is provisional support for reducing memory consumption via recording flags like `DATA_ON_LEAD_PHYSICAL_DEVICE` and `COMPACT_ON_FINALIZE`. With large numbers (thousands) of persistent command buffers or command buffers that embed a lot of data (don't!) these may be useful for reducing memory consumption. For most programs today this is not a concern. Block pools used to service the device allocations can also be tuned to reduce unutilized block space.

Since all metadata required to setup command buffer execution is contained in device memory the submission action on the host merely acquires a queue entry, specifies the pointer to the device local copy of the metadata where execution will occur, and kicks the scheduler. There's no additional host information required for the scheduler to begin execution and as such enabling device->device submission is trivial (acquire queue entry, copy fields, enqueue the scheduler directly to target AQL queue). Replay and hostless scheduling is possible by encoding binding table slots or buffer references in each command such that any buffer a device has can be provided to any other device (or itself).

#### Host Service Worker

* See: [iree/hal/drivers/amdgpu/host_worker.h](/runtime/src/iree/hal/amdgpu/host_worker.h)
* See: [iree/hal/drivers/amdgpu/device/host.h](/runtime/src/iree/hal/amdgpu/device/host.h)

Each physical device is assigned a worker thread pinned to the nearest CPU that manages the host service HSA queue ala an RPC server. The worker processes the queue as per the AQL specification and handles incoming HSA agent dispatch packets as calls to host services.

Each host worker executes synchronously and runs independently from all other workers as to not indirectly synchronize multiple devices that may be collaborating and requiring host services. The supported host operations are classified as either calls ("grow memory pool and return new pointer") or unidirectional posts ("notify the host that a semaphore was signaled by the device").

Most requests are intended to be handled quickly (~microseconds) and be relatively infrequent. The most common operation is releasing retained resource sets and signaling external semaphores (if any) after a queue entry completes. The service also routes requests for allocations that require dedicated device memory or that are used to grow or trim a device managed pool to the appropriate subsystem.

#### Trace Worker

* See: [iree/hal/drivers/amdgpu/trace_worker.h](/runtime/src/iree/hal/amdgpu/trace_worker.h)
* See: [iree/hal/drivers/amdgpu/device/tracing.h](/runtime/src/iree/hal/amdgpu/device/tracing.h)

The host is responsible for periodically flushing the contents of the tracing ringbuffer and exporting those events to the active tracing tool. The worker runs independently per-queue to avoid blocking any other queue while flushing trace events at the cost of additional host work. The expectation is that on a host with multiple AMDGPU devices there are more than enough cores to support the workers. For information on using tracing on the device see [Device Tracing](#tracing).

Correlating timing between multiple devices, the host, and the external tracing source is non-trivial. Or at least it's eluded me - it's currently broken/hacked. Help would be very welcome! We capture the timestamps in agent domain and then need to translate them to the common host domain compatible with trace tooling. Because we process the ringbuffer in batches we can amortize the overhead involved in periodic calibration.

### Device Library

* See: [build_tools/bazel/iree_amdgpu_binary.bzl](/build_tools/bazel/iree_amdgpu_binary.bzl)
* See: [build_tools/cmake/iree_amdgpu_binary.cmake](/build_tools/cmake/iree_amdgpu_binary.cmake)
* See: [iree/hal/drivers/amdgpu/device/support/common.h](/runtime/src/iree/hal/drivers/amdgpu/device/support/common.h)

Our device library containing the scheduler and all builtin kernels used by it is written in bare-metal C23. We do not use a libc-alike library or the AMD device libraries. Building device library binaries requires only a clang/llvm-link/lld with the AMDGPU targets enabled.

Because we are developing bare-metal style for a GPU nearly everything above the base language level is off the table - no TLS (what would that even mean?), no globals, no library or system calls, and no C++ things (e.g. global initializers). Since even things like atomics differ external header-only libraries are unlikely to be usable unless very tightly scoped and we'd fork them for internal use if needed.

The device library code living under [iree/hal/drivers/amdgpu/device/](/runtime/src/iree/hal/amdgpu/device/) depends wholely on itself. The host HAL implementation in the parent [iree/hal/drivers/amdgpu/](/runtime/src/iree/hal/amdgpu/) pulls in headers from the device library in order to share data structures and enums but does not try to compile the code. Code specific to the device library is guarded by `IREE_AMDGPU_TARGET_DEVICE` and code specific to the host compilation environment is guarded by `IREE_AMDGPU_TARGET_HOST`. Since crossing the streams can quickly spiral into madness effort is spent to avoid intermixing as much as practical while trying to minimize duplication of things that may potentially get out of sync.

#### Versioning

The device library is only ever shipped with the runtime it is built for. The API between host and device is not stable and considered an implementation detail of the HAL. If any of the API does leak out into deployments that may have different versions - such as compiled kernels that may rely on custom ABI or details of our execution environment - they will need to be versioned appropriately.

#### Architecture-Specific Binaries

* See: [iree/hal/drivers/amdgpu/device/BUILD.bazel](/runtime/src/iree/hal/drivers/amdgpu/device/BUILD.bazel)

Unfortunately all AMD device binaries are architecture dependent and not backward or forward compatible. This results in us needing to include precompiled binaries for every architecture that the runtime may be used with and there are quite a few. In development builds it's best to set the single architecture or two a developer is using to keep compile times down but release builds need to produce binaries for all officially supported architectures. In the future if AMD gains a forward-compatible representation we'll jump on that: nothing about our device library relies on architecture-specific features as it's mostly straight-line scalar C code running in a single work item shuffling bits around. Given growing SPIR-V support in LLVM that may be one route assuming that SPIR-V binaries can be loaded (and JITed) by the HSA code object mechanism.

**TODO(benvanik)**: document cmake flags for controlling which binaries are built.

#### Scoped Atomics

* See: [iree/hal/drivers/amdgpu/device/support/common.h](/runtime/src/iree/hal/amdgpu/device/support/common.h)

Atomics performed on the device need to indicate the scope at which they are synchronized within the system. Normal C11 atomics do not include a scope and are assumed to operate at system level meaning that an atomic update of a value that is only ever produced and consumed on a single device must be made visible to the entire system (host and all other devices). To avoid this potential performance issue atomic operations that are used on devices generally include a scope that indicates the visibility and that scope must be as wide as required and should be as narrow as possible. The `iree_amdgpu_scoped_atomic_*` functions mirror the C11 atomics but also take a scope, e.g. `iree_amdgpu_scoped_atomic_fetch_add(..., iree_amdgpu_memory_scope_system);` Where we know that a particular atomic operation _may_ cross device boundaries we err on the side of `iree_amdgpu_memory_scope_system` and when we are positive it _will not_ we use `iree_amdgpu_memory_scope_device`. The finer scopes such as `work_item` and `sub_group` are rarely used in our device library as we rarely use per-work-item atomics as part of a single dispatch.

#### HSA Queues and Signals

* See: [iree/hal/drivers/amdgpu/device/support/queue.h](/runtime/src/iree/hal/amdgpu/device/support/queue.h)
* See: [iree/hal/drivers/amdgpu/device/support/signal.h](/runtime/src/iree/hal/amdgpu/device/support/signal.h)

The HSA specification defines the `hsa_queue_t` structure and the interface for `hsa_signal_t`. Usage of these structures via `hsa_*` functions normally requires the AMD device library to be linked into the device library binary and is a subset of the operations we perform as (effectively) an HSA driver implementation. Thankfully the AMDGPU-specific definitions of the queue and signal (`amd_queue_t` and `amd_signal_t`) are specified and something we can (for our purposes) directly poke. We rely on ROCR to allocate queues and configure the driver and device but manipulate the queues directly (read/write packet IDs, the queue ringbuffer, etc). Signals are simpler and for the most part on architectures we support (everything in the last ~5+ years) they are simply atomic operations that are easy to implement directly. For device-only signals not backed by platform events we don't need to use ROCR and can instead treat any memory as a signal, enabling us to slab allocate thousands of signals cheaply and pool signals entirely on device (critical when timing dispatches that each require a unique signal).

Since the AMD device support library is not shipped as part of LLVM and the implementations of the functions are not required we redefine the `amd_queue_t` and `amd_signal_t` structures in our own headers. This avoids the need to fetch/link the library (which is geared for OpenCL and HIP and includes a significant amount of baggage) and avoid the requirement for implicit kernel arguments.

#### Buffers

* See: [iree/hal/drivers/amdgpu/device/buffer.h](/runtime/src/iree/hal/amdgpu/device/buffer.h)

Supporting replayable command buffers requires that buffers are referenced indirectly as the buffers used may change from submission to submission. The HAL is structured to allow this by supporting binding tables passed alongside command buffers when submitting work for execution but also allows for any buffer referenced statically to be indirected. The primary source of indirect buffers is queue-ordered allocation (`iree_hal_device_queue_alloca`) where the address of an allocation is not guaranteed to be available (let alone committed) until the asynchronous operation completes. To enable recording command buffers that reference such buffers or submitting command buffers for execution with binding tables containing the results of asynchronous allocations we use fat pointers throughout and try to perform the indirection just-in-time.

`iree_hal_amdgpu_device_buffer_ref_t` is used to reference buffers of all types with an enum indicating whether the pointer is available immediately (a statically allocated buffer), available with delay (a queue-ordered allocation), or sourced from a binding table slot at submission-time. Each buffer allocated in queue-order is assigned a `iree_hal_amdgpu_device_allocation_handle_t` in device memory that is returned immediately when requesting the allocation and that has a pointer which is updated when the allocation is committed.

Currently the implementation is performing the indirection on allocation handles to fetch the final address when either queue submissions (copy, fill, etc) or dispatches within a command buffer are issued. This avoids any additional overhead within the kernels as they just see a device pointer. It does mean that each binding is resolved per dispatch issued instead of per binding in the binding table as is the minimum required and that may be improved in the future at the cost of slightly higher latency when scheduling a command buffer.

#### Semaphores

* See: [iree/hal/drivers/amdgpu/semaphore.h](/runtime/src/iree/hal/amdgpu/semaphore.h)
* See: [iree/hal/drivers/amdgpu/device/semaphore.h](/runtime/src/iree/hal/amdgpu/device/semaphore.h)

The device library supports two types of semaphores: internal and external. Internal semaphores are those created by the AMDGPU HAL and used exclusively with it (on any device). Internal semaphores are strongly preferred and should be used in nearly all cases except when interacting with external APIs (interop with Vulkan, host code, etc).

Internal semaphores are backed by `iree_hal_amdgpu_device_semaphore_t` in device memory which maintains a list of waiters to be woken as certain semaphore values are reached. This allows the device to sequence work both on itself and across other devices or the host by waking the target without needing to involve the host. No platform events are involved and in the steady state of a device->device pipeline no host interrupts are required. To support the host waiting on internal semaphores signaled by a device we also include an HSA signal per semaphore that mirrors the payload value of the semaphore thereby allowing HSA multi-wait operations to efficiently block (or spin).

External semaphores are treated as opaque from the perspective of the device library and always route through the host when signaled. Waiting on external semaphores must happen on the host as the device cannot interact with the platform or vtabled HAL object.

#### Kernels

* See: [iree/hal/drivers/amdgpu/executable.h](/runtime/src/iree/hal/amdgpu/device/executable.h)
* See: [iree/hal/drivers/amdgpu/device/kernels.h](/runtime/src/iree/hal/amdgpu/device/kernels.h)
* See: [iree/hal/drivers/amdgpu/device/kernel_tables.h](/runtime/src/iree/hal/amdgpu/device/kernel_tables.h)

AQL dispatch packets require metadata (segment sizes, workgroup sizes, etc) and we retain these per-kernel in a device-local `iree_hal_amdgpu_device_kernel_args_t` structure. In addition to the attributes required by AQL we include ones used by the device library to issue the dispatches such as constant and binding count in order to decode command buffers and tracing identifiers used to tag trace events with a host-visible source location. HAL executables allocate kernel arguments on each device and populate them as defined in the metadata embedded in the executable binary. Built-in device library entry points are specified in a table baked into the binary.

Because every dispatch packet enqueued requires the metadata we need to store the information in device memory. In addition to the per-device copies the host also retains a copy in order to record and validate command buffers without hitting device memory. Note that all copies of the metadata between the host and devices a kernel is loaded on must be identical.

##### Implicit Arguments

OpenCL and HIP both tag on a non-trivial amount of implicit arguments to each kernel launch. These support device library features such as host calls (printf, etc), device-side enqueuing (in OpenCL), and programming model support (OpenCL grid offsets). We need none of those and as such as omit them from both our device library functions and compiler-produced kernel functions. This can save considerable kernarg ringbuffer space - over 1000 dispatches in a command buffer at 256 bytes per dispatch can easily cause stalls without over-allocating the kernarg ring. This is one cause of host stalls observed when dispatching even moderate amount of work via HIP and OpenCL: the host spins waiting for more ringbuffer space. If we ever find ourselves needing these implicit arguments (for dispatching HIP or OpenCL kernels unmodified) we can enable conditional support for those dispatches that need them, however some features such as host support will not be practical to support in most cases.

#### Command Buffer Execution

* See: [iree/hal/drivers/amdgpu/command_buffer.h](/runtime/src/iree/hal/amdgpu/command_buffer.h)
* See: [iree/hal/drivers/amdgpu/device/command_buffer.h](/runtime/src/iree/hal/amdgpu/device/command_buffer.h)

Command buffers are recorded into an in-memory format optimized for issuing repeatedly by the device scheduler. Information that may change across submissions such as buffers referenced in the per-submission binding table or offsets in HSA queues or kernarg ringbuffers are left as symbolic during recording and populated by the device scheduler on-demand. The in-memory representation is a program with one or more basic blocks (`iree_hal_amdgpu_device_command_block_t`) containing one or more commands (`iree_hal_amdgpu_device_cmd_t`) terminating in a control command (e.g. `iree_hal_amdgpu_device_cmd_branch_t`). When a queue submission is ready to issue (all waits satisfied) the device-side scheduler uses the static command buffer information to populate an `iree_hal_amdgpu_device_execution_state_t` based on the submission pointing at the entry block and referencing reserved ranges of various ringbuffers and pools. A device-side block issue kernel is launched to then convert each command in the block from the in-memory representation to one or more AQL packets in the target execution queue. The commands then execute as the device processes the packets and when the terminating control command is reached (branch, return, etc) the next block or originating scheduler is enqueued to continue program execution or complete the submission.

Each submission maintains its own execution state allowing for the same command buffer recording to be issued on the same device simultaneously. The immutable in-memory representation of the command buffer encodes offsets/deltas for kernargs, completion signals, and tracing events that are overlaid on to the scheduled resources when the command buffer is issued. Since some of those offsets/deltas are submission-dependent or queue-dependent sometimes alternates are included in the metadata to allow the scheduler to pick the appropriate set (e.g. including per-command profiling query IDs for when operating at different tracing levels).

##### Indirect Dispatches

An indirect dispatch is one that sources its workgroup count from a buffer immediately prior to executing. This allows for prior dispatches within the same command buffer to produce the `uint32_t workgroup_count[3]` based on data-dependent information. In user programs with dynamically sized buffers (originating from dynamically shaped tensors in ML programs) this can often allow for the same command buffer to be replayed even if shapes vary. AQL does not currently have a way to perform this as part of dispatch packets, unfortunately, and we have to emulate it.

Emulation requires an ancillary dispatch that reads the `workgroup_count[3]` from the source buffer to patches the subsequent actual dispatch packet. This adds additional latency to the dispatch that would be avoided if the AQL packet natively supported a way of specifying a buffer; note the packet processor may incur memory fetch costs but those are orders of magnitude less costly than the emulation.

In the common case today indirect dispatches are frequently fixed at the time a command buffer is issued. To avoid the emulation overhead such dispatches are recorded with the `IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_STATIC` flag and the workgroup count is fetched as part of producing the AQL dispatch packet. Any indirect dispatches not known to be static during a submission instead use `IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_DYNAMIC` and get the additional dispatch to patch the packet.

It's a strong request for this to be improved in future packet processor versions - programs will only get more dynamic.

##### Large Command Buffer Optimization

Outside of specific usage patterns (e.g. layer-level tensor parallelism) most command buffers are on the order of 500 to 5000 commands. Our queue scheduling behavior allows that to be a single host->device or device->device operation regardless of count and reduce cross-device traffic considerably. Once issuing command buffer execution on the target device we are scoped to that local device and ordered in the target execution queue such that no synchronization (internal or otherwise) is required. As part of recording the command buffer all per-command information is encoded such that each command can be issued independently. This includes dynamic inputs like buffer bindings and kernarg ringbuffer offsets as well as things only known at issue time like the AQL packet range in the target execution queue.

The `iree_hal_amdgpu_device_cmd_block_issue` kernel is responsible for taking the immutable in-memory command buffer recording and translating each command into zero or more AQL packets. The scheduler reserves ranges of packets, signals, and kernargs before dispatching the kernel with one work item per command. Because the entire range of AQL packets is reserved the device is able to translate commands in any order as work items progress and immediately start processing. This relies on the AQL packet processing rules that say that a hardware doorbell only need indicate packets will have their type updated from `INVALID` to something valid _at some point_ and not that it needs to be prior to notifying the device.

An additional optimization not performed today would be sorting the commands within a block such that commands with the same type are contiguous. Alternatively an indirection table of a flattened contiguous space could be stored as part of the recorded command buffer. This would allow us to use a larger workgroup size knowing that each work item would be producing the same type of packet and therefore running (roughly) the same instructions.

##### Small Command Buffer Optimization

If performance is a concern as much work as possible must be recorded into the same command buffer as possible. Though the scheduling model allows us to avoid a tremendous amount of host->device traffic by requiring only a single host->device enqueue of the command buffer regardless of command count that only has impact at a scale above ~5 commands. Under that the additional dispatch latency of the scheduler and command buffer issue and retire logic outweighs the benefits of batching.

For cases where command buffers are tiny (1 or 2 commands) it's preferred that applications instead issue queue operations if possible (such as `iree_hal_device_queue_fill` and `iree_hal_device_queue_copy`) as this avoids the extra command buffer logic. The host-side HAL queue _could_ identify such command buffers and process them directly on the host but today we do not (it would require significant logic duplication).

In cases where queue operations are not available (such as dispatches) and the device needs to schedule a small command buffer we switch to issuing commands serially on the execution queue instead of dispatching the `iree_hal_amdgpu_device_cmd_block_issue` kernel to do so in parallel on the control queue. This trades off slower serial packet enqueuing against the extra dispatch latency of the issue.

This particular optimization can be useful even outside of issuing small command buffers as when a command buffer contains control flow commands (branch, etc) we may end up with some blocks that contain very few commands such as a loop body that contains only two dispatches. For conditional execution ("dispatch kernel A or kernel B based on this buffer value") it's preferred that indirect dispatch is used as a form of predication: not enqueuing a packet at issue time is significantly faster than jumping around between blocks within a command buffer.

#### Scheduler

* See: [iree/hal/drivers/amdgpu/queue.h](/runtime/src/iree/hal/amdgpu/queue.h)
* See: [iree/hal/drivers/amdgpu/device/scheduler.h](/runtime/src/iree/hal/amdgpu/device/scheduler.h)

```
                   +--------------+     +-----------+
+-------------+    |+--------------+    |+-----------+    +---------------+
| logical dev |--->+| physical dev |--->+| scheduler |-+->| control queue |
+-------------+     +--------------+     +-----------+ |  +---------------+
                                                       |  +-----------------+
                                                       |  |+-----------------+
                                                       +->+| execution queue |
                                                           +-----------------+
```

Each physical device may have one or more schedulers with an associated control queue and one or more execution queues. Each scheduler maintains its own execution resources such as kernarg storage and pending queue entry lists allowing each to run independently. The exact topology is configurable and may vary from one extreme (1:1:1:1 on small devices) to the other (2:8:8:16 on larger devices).

The control queue is exclusively for scheduler operations and runs with elevated priority. Scheduler operations are generally very short (microseconds) and what either issue new work for the device to process or retire completed work in order to unblock dependent work. Since dependent work may be on the host or another device it's important that the latency is as low in order to reduce bubbles. By using a single queue for scheduler-related work (such as command buffer execution management) we ensure only a single scheduler operation is running at a time and we need no internal synchronization. Note that operations on the control queue may still need to synchronize with the host or other devices via structured mechanisms (mailboxes/HSA queues/etc).

Schedulers obey the dependency requirements of submitted work as represented by semaphore wait and signal lists attached to each queue operation. The execution queues may be shared across multiple schedulers or exclusive to a particular scheduler: each command buffer execution can be thought of as a fiber with the hardware cooperatively scheduling the commands in each. A single scheduler may decide to run independent operations on separate execution queues to gain concurrency or dependent operations on the same queue to reduce overheads (as no synchronization is required within a single queue).

Today the scheduler is relatively basic: each time a scheduling operation is to be performed the `iree_hal_amdgpu_device_queue_scheduler_tick` kernel is dispatched on the control queue and it handles any scheduler operations that can be performed. By having the tick poll and process any work available at the time it is actually executed (vs enqueued) we avoid additional latency from dispatching per queue operation and waiting for each to clear. A single tick may accept incoming queue entries from the host or other another scheduler, retire completed entries, check whether they or any existing entry is ready to execute (all waiters satisfied), and issue all ready entries (if resources allow). In common cases of N chained queue entries - even if across multiple devices - this results in 2 + N scheduler ticks (initial issue, each retire->issue, and final retire).

#### Tracing

* See: [iree/hal/drivers/amdgpu/trace_worker.h](/runtime/src/iree/hal/amdgpu/trace_worker.h)
* See: [iree/hal/drivers/amdgpu/device/tracing.h](/runtime/src/iree/hal/amdgpu/device/tracing.h)

An AMDGPU-specific tracing mechanism very similar (but not identical to) the main IREE tracing mechanism is provided to instrument device library code and record performance information that can be relayed to host tracing tools.

Each device library kernel has through some means access to an `iree_hal_amdgpu_device_trace_buffer_t` and can use `IREE_AMDGPU_TRACE_BUFFER_SCOPE(trace_buffer);` to make the trace buffer active for the current function scope. The host-side tracing uses TLS to accomplish this instead. Once a trace buffer scope is activated within a function the `IREE_AMDGPU_TRACE_*` macros can be used to define instrumented zones, attach payloads like integers or strings, log messages, and plot values.

Which tracing features are available is determined by the `IREE_HAL_AMDGPU_TRACING_FEATURES` bitfield and it's possible to significantly reduce code size by disabling ones not required in a deployment. Since the tracing feature pulls in a non-trivial amount of code and embeds a lot of additional strings it is recommended to disable tracing entirely unless needed and otherwise only enable the features required. For example, a release deployment would have tracing disabled on both host and device while a tracing deployment may have tracing enabled but debug messages, instrumentation, and allocation tracking disabled to allow users to still get execution timings but not add the additional overhead of the other features. The `IREE_HAL_AMDGPU_HAS_TRACING_FEATURE` macro is used to guard any code relying on a particular feature.

During development `IREE_AMDGPU_DBG` is useful as a `printf` as it accepts a small but practical subset of `printf` format specifiers and routes the message into the host tracing tool. These are only enabled in release (`NDEBUG`) builds. Note that there's significant overhead involved in formatting strings on the device and even though it's only present in debug builds having any such logging checked in can severely degrade the usability of debug builds for any other purpose - to that end, `IREE_AMDGPU_DBG` should be treated like a `printf` and never checked in unless on cold paths or critical to daily development.

Tracing is implemented by a ringbuffer shared with the host that is populated by the device and flushed occasionally. Timestamps are captured in the device ("agent" in HSA) domain and require later translation into system times that correlate with other tools or timing sources. See [Host Trace Buffer](#trace-buffer) for more information on host handling.

## Missing/Emulated Features

Some features are not yet implemented and some are implemented via emulation. After the initial milestone lands they will be developed further.

### Queue Read/Write

The `iree_hal_device_queue_read` and `iree_hal_device_queue_write` APIs are implemented using emulation. This is extremely inefficient and could be done better. Ideally there would be some accelerated DMA utilities we could use but today it seems like the best we can do is more intelligently scheduling the operations and reducing overheads. Future work will have the device scheduler publish requested read or write operations to the host and the host can use platform APIs to perform the operations zero-copy when possible. e.g. scheduler request file ranges (0-1024) and (2048-4096) be read to device buffers A and B and the host issues read requests with io_uring that have the kernel asynchronously and concurrently populate the device buffers without intermediates. If intermediates are required we will maintain them as ringbuffers that the scheduler manages to still allow concurrency and out of order staging but without the need for host/device synchronization.

### External API Interop

Placeholders are present for allowing the use of external buffers and semaphores that have been imported using supported APIs (or by a future query mechanism). Additional host service handling is likely required for these.

## HSA/AQL/ROCR Wishlist

### Ergonomic/Quality-of-Life Features

See https://github.com/iree-org/iree/issues/19636 for requested improvements to the ROCR.

### Windows Support

Currently the AMDGPU HAL driver compiles and runs on Windows however unfortunately there's no HSA implementation on Windows to run with. Once ROCR can be compiled and run on Windows IREE can take advantage of it immediately.

### Forward-Compatible ISA

See [Architecture-Specific Binaries](#architecture-specific-binaries) for the current woes.

### Indirect Dispatch Packet

See [Indirect Dispatches](#indirect-dispatches) for the current suboptimal behavior. An `hsa_kernel_dispatch_packet_t`-like packet or packet flag that changed `uint32_t grid_size[3]` to `uint32_t* grid_size_ptr` and had the packet processor fetch the grid size prior to issuing the dispatch for execution would reduce per-dispatch latency by ~10us.

### AQL DMA Packet Sanity

We do not currently use SDMA as it's very unergonomic and (seemingly) difficult to fit into an async pipeline. Ideally there would be AQL packets that could be scheduled that would perform copies and fills such that dispatches, copies, and fills could be interleaved and execute in device order using HSA signals. It seems like the custom packet format, custom trap mechanism for firing events (maybe incompatible with device->device signaling?), and the entire undocumented nature of it makes it hard to build anything around besides basic synchronous memcpys. We want device to device copies to be efficient and need SDMA to do it but it's currently too difficult to decipher.

### Agent Dispatches on Device

It's a hack that we run our device scheduler on execution units. Besides the case of large command buffer issuing where we parallelize all other code is single threaded scalar code performing reads and writes to device memory. A mechanism whereby we could compile our device library in such a way that we could use the `IREE_HSA_PACKET_TYPE_AGENT_DISPATCH` packet and run the functions out-of-band with execution would reduce scheduling latency and avoid under utilizing execution resources. This requires an LLVM/Clang-accessible ISA that our C code could target.

### Device-side Signals/Triggers

In a situation where an external event generator (another non-AMDGPU device, host code from any origin, etc) is the trigger for device execution there's a non-trivial latency between when the event is generated and when the first instruction of the reacting handler can be executed on device. This usually involves a host thread to wait or poll on the event, that thread to wait to get scheduled on a CPU (lots of variance), time on the CPU to perform the application logic and enqueue the dispatch packet to run the handler code on the device, and the time for the packet processor to pick up the new work and begin executing it. In high-frequency/low-latency multi-device or host/device sequences this can create bubbles from 20-200us.

A mechanism whereby the device could have "signals" (in the POSIX sense, not HSA) registered with dispatch parameters assigned could help eliminate this latency a much lower cost. Polling a table of registered signals and enqueuing the dispatch packet on the local device would (for reasonable numbers of signals e.g. ~1-4) reduce latencies to local packet processing rates.
