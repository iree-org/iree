---
hide:
  - tags
tags:
  - GPU
  - CUDA
---

# CUDA HAL Driver

This document lists technical details regarding the CUDA implemenation of
IREE's Hardware Abstraction Layer, called a CUDA HAL driver.

IREE provides a [Hardware Abstraction Layer (HAL)][iree-hal] as a common
interface to different compute accelerators. IREE HAL's design draws inspiration
from modern GPU architecture and APIs; so implementing a HAL driver using CUDA
is mostly straightforward; though there are places we need emulation given no
direct mapping concepts or mechanisms.

## Overall design choices

### CUDA driver vs runtime API

IREE HAL's design draws inspiration from modern GPU APIs--it provides explicit
control of low-level GPU objects. The compiler is expected to plan the object
lifetime and schedule workload and synchronization in an optimized way; IREE
HAL implementation and the underlying GPU driver stack is expected to be a thin
layer without much smarts and magic.

Therefore when implementing the IREE HAL using CUDA, we use the
[driver API][cuda-driver-api] instead of the [runtime API][cuda-runtime-api].
At runtime the HAL CUDA driver will load the `libcuda.so`/`nvcuda.dll` library
dynamically and query a subset of the CUDA driver API used in HAL via
[the `cuGetProcAddress()` API][cu-get-proc-address].

## GPU Objects

### Driver

There is no direct CUDA construct that map to the IREE HAL `iree_hal_driver_t`
abstraction. We use it to hold the dynamic symbols loaded for all devices,
and device enumeration and creation.

### Device

`iree_hal_cuda_device_t` implements [`iree_hal_device_t`][hal-device] to provide
the interface to CUDA GPU device by wrapping a [`CUdevice`][cu-device].
For each device, right now we create two `CUstream`s--one for issuing commands
for memory allocation and kernel lauches as instructed by the program; the other
for issue host callback functions after dispatched command buffers completes.
See [synchronization](#synchronization) section regarding the details.

#### Async allocation

The CUDA HAL drivers supports async allocation
(`iree_hal_device_queue_alloca()` and `iree_hal_device_queue_dealloca()`)
via [CUDA stream ordered memory allocation][cuda-stream-ordered-alloc].

The `async_allocations` in the `iree_hal_cuda_device_params_t` struct allows
to enable this feature.

### Command buffer

[`iree_hal_command_buffer_t`][hal-command-buffer] is a recording of commands to
issue to the GPU; when the command buffer is submitted to the device it's then
actually executed on the GPU asynchronously.

Two implementations of `iree_hal_command_buffer_t` exist in the CUDA HAL
driver--[one][cuda-graph-command-buffer] backed by [`CUgraph`][cu-graph] and
[the other][cuda-stream-command-buffer] backed by `CUstream`.

`CUgraph` conceptually matches `iree_hal_command_buffer_t` better given it's
a recording of commands to issue to the GPU. Also using the `CUgraph` API allows
to easily encode fine grain dependencies between dispatch without having to
create multiple streams. Therefore, the `CUgraph`-backed implementation is a
more natural one.
Though note that `CUgraph` API is meant to be used for recording once and
replying multiple times and there may be a performance penalty to using
`CUgraph` API for one-shot command buffer.

The `CUstream`-backed implementation just issues commands directly to a
`CUstream` when recording. Commands issued to `CUstream` can be immediately
sent to the GPU for execution; there is no recording and replaying separation.
In order to match the recording semantics of `iree_hal_command_buffer_t`, to
use the `CUstream`-backed command buffer, we need to first record the command
buffer into an in-memory
[`iree_hal_deferred_command_buffer_t`][hal-deferred-command-buffer], and then
when applying the command buffer, we create a new `CUstream`-backed
implementation.

The `command_buffer_mode` in the `iree_hal_cuda_device_params_t` struct allows
to select which implementation to use.

### Allocator

The allocator will forward allocation requests to `cuMemHostAlloc()` for host
local memory, `cuMemAlloc()` for device local and host invisible memory, and
`cuMemAllocManaged()` for device local and host visible memory.

### Buffer

CUDA buffers are represented either as a host pointer or a device pointer of
type `CUdeviceptr`.

### Executable

[`iree_hal_executable_t`][hal-executable] maps naturally to `CUmodule`.

The compiler generates a FlatBuffer containing a PTX image as well as a
list of entry point functions and their associated metadata (names,
workgroup size, dynamic shared memory size, etc.). At runtime, the CUDA HAL
driver loads the PTX image and creates `CUfunction`s out of it for various
entry points.

## Synchronization

### Event

[`iree_hal_event_t`][hal-event] right now is not used in the compiler so it's
not yet implemented in the CUDA HAL driver.

### Semaphore

The IREE HAL uses semaphores to synchronize work between host CPU threads and
device GPU streams. It's a unified primitive that covers all directions--host
to host, host to device, device to host, and device to device, and allows
flexible signal and wait ordering--signal before wait, or wait before signal.
There is no limit on the number of waits of the same value too.

The core state of a HAL semaphore consists of a monotonically increasing 64-bit
integer value, which forms a timeline--signaling the semaphore to a larger
value advances the timeline and unblocks work waiting on some earlier values.
The semantics closely mirrors
[Vulkan timeline semaphore][vulkan-timeline-semaphore].

In CUDA, there is no direct equivalent primitives providing all the capabilities
needed by the HAL semaphore abstraction:

* [Stream memory operations][cu-mem-ops] provides `cuStreamWriteValue64()` and
  `cuStreamWaitValue64()`, which can implment HAL semaphore 64-bit integer value
  signal and wait. Though these operations require device pointers and cannot
  accepts pointers to managed memory buffers, meaning no support for the host.
  Additionally, per the spec, "synchronization ordering established through
  these APIs is not visible to CUDA. CUDA tasks that are (even indirectly)
  ordered by these APIs should also have that order expressed with
  CUDA-visible dependencies such as events." So it's not suitable for
  integration with other CUDA components.
* For [external resource interoperability][cu-external-resource], we have APIs
  like `cuSignalExternalSemaphoresAsync()` and `cuWaitExternalSemaphoresAsync()`,
  which can directly map to Vulkan timeline semaphores. Though these APIs are
  meant to handle exernal resources--there is no way to create
  `CUexternalSemaphore` objects directly other than `cuImportExternalSemaphore()`.

Therefore, to implement the support, we need to leverage multiple native CPU or
CUDA primitives under the hood.

#### `CUevent` capabilities

The main synchronization mechanism is [CUDA event--`CUevent`][cu-event].
As a functionality and integration baseline, we use `CUevent` to implement the
IREE HAL semaphore abstraction.

`CUevent` natively supports the following capabilities:

* State: binary; either unsignaled or signaled. There can exist multiple
  waits (e.g., via `cuEventSynchronize()` or `cuGraphAddEventWaitNode()`) for
  the same `CUevent` signal (e.g., via `cuEventRecord()` or
  `cuGraphAddEventRecordNode()`).
* Ordering: must be signal before wait. Waiting before signal would mean
  waiting an empty set of work, or previously recorded work.
* Direction: device to device, device to host.

We need to fill the remaining capability gaps. Before going into details,
the overall approach would be to:

* State: we need a 64-bit integer value timeline. Given the binary state of
  a `CUevent`, each `CUevent` would just be a "timepoint" on the timeline.
* Ordering: we need to defer releasing the workload to the GPU until the
  semaphore waits are reached on the host, or we can have some device
  `CUevent` to wait on.
* Direction: host to host and host to device is missing; we can support that
  with host synchronization mechanisms.

#### Signal to wait analysis

Concretely, for a given HAL semaphore, looking at the four directions:

##### CPU signal

A CPU thread signals the semaphore timeline to a new value.

If there are CPU waits, it is purely on the CPU side. We just need to use common
CPU notification mechanisms. In IREE we have `iree_event_t` wrapping various
low-level OS primitives for it. So we can just use that to represent a wait
timepoint. We need to keep track of all CPU wait timepoints in the timeline.
After a new signaled value, go through the timeline and notify all those waiting
on earlier values.

If there are GPU waits, given that there are no way we can signal a `CUevent` on
CPU, one way to handle this is to cache and defer the submission batches by
ourselves until CPU signals past the desired value. To support this, we would
need to implement a deferred/pending actions queue.

##### GPU signal

GPU signals can only be through a `CUevent` object, which has a binary state.
We need to advance the timeline too. One way is to use `cuLaunchHostFunc()`
to advance from the CPU side with `iree_hal_semaphore_list_signal()`.
This additionally would mean we can reuse the logic form CPU signaling to
unblock CPU waits.

After advancing the timeline from the CPU side with `cuLaunchHostFunc()`,
we can release more workload from the deferred/pending actions queue to the GPU.
Though, per the [documentation][cu-launch-host-func] of `cuLaunchHostFunc()`,
"the host function must not make any CUDA API calls." So we cannot do that
directly inside `cuLaunchHostFunc()`; we need to notify another separate
thread to call CUDA APIs to push more work to the GPU. So the deferred/pending
action queue should have an associcated thread.

For GPU waits, we can also leverage the same logic--using CPU signaling to
unblock deferred GPU queue actions. Though this is performant, given that
the CPU is involved for GPU internal synchronization. We want to use `CUevent`
instead:

* We keep track of all GPU signals in the timeline. Once we see a GPU wait
  request, try to scan the timeline to find a GPU signal that advances the
  timeline past the desired value, and use that for waiting instead. (This
  actually applies to CPU waits too, and it's an optimization over pure
  CPU side `iree_event_t` polling.)
* We may not see GPU signal before seeing GPU wait requests, then we can also
  keep track of all GPU waits in the timeline. Later once see either a CPU
  signal or GPU signal advancing past the waited value, we can handle them
  accordingly--submitting immediately or associating the `CUevent`.
  This would also guarantee the requirement of `CUevent`--recording should
  happen before waiting.
* We can use the same `CUevent` to unblock multiple GPU waits. That's allowed,
  though it would mean we need to be careful regarding `CUevent` lifetime
  management. Here we can use reference counting to see how many timepoints
  are using it and automatically return to a pool once done.

Another problem is that per the `cuLaunchHostFunc()` doc, "the function will
be called after currently enqueued work and will block work added after it."
We don't want the blocking behavior involving host. So we can use a dedicated
`CUstream` for launching the host function, waiting on the `CUevent` from the
original stream too. We can also handle resource deallocation together there.

#### Data structures

To summarize, we need the following data structures to implement HAL semaphore:

* `iree_event_t`: CPU notification mechanism wrapping low-level OS primitives.
  Used by host wait timepoints.
* `iree_event_pool_t`: a pool for CPU `iree_event_t` objects to recycle.
* `iree_hal_cuda_event_t`: GPU notification mechanism wrapping a `CUevent` and
  a reference count. Used by device signal and wait timepoints. Associates with
  a `iree_hal_cuda_event_pool_t` pool--returns to the pool directly on once
  reference count goes to 0.
* `iree_hal_cuda_event_pool_t`: a pool for GPU `iree_hal_cuda_event_t` objects
  to recycle.
* `iree_hal_cuda_timepoint_t`: an object that wraps a CPU `iree_event_t` or
  GPU `iree_hal_cuda_event_t` to represent wait/signal of a timepoint on a
  timeline.
* `iree_hal_cuda_timepoint_pool_t`: a pool for `iree_hal_cuda_timepoint_t`
  objects to recycle. This pool builds upon the CPU and GPU event pool--it
  acquires CPU/GPU event objects there.
* `iree_hal_cuda_timeline_semaphore_t`: contains a list of CPU wait and GPU
  wait/signal timepoints.
* `iree_hal_cuda_queue_action_t`: a pending queue action (kernel launch or
  stream-ordered allocation).
* `iree_hal_cuda_pending_queue_actions_t`: a data structure to manage pending
  queue actions. It provides APIs to enqueue actions, and advance the queue on
  demand--queue actions are released to the GPU when all their wait semaphores
  are signaled past the desired value, or we can have a `CUevent` object already
  recorded to some `CUstream` to wait on.


[cuda-driver-api]: https://docs.nvidia.com/cuda/cuda-driver-api/index.html
[cuda-runtime-api]: https://docs.nvidia.com/cuda/cuda-runtime-api/index.html
[vulkan-timeline-semaphore]: https://www.khronos.org/blog/vulkan-timeline-semaphores
[iree-hal]: https://github.com/openxla/iree/tree/main/runtime/src/iree/hal
[hal-allocator]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/allocator.h
[hal-buffer]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/buffer.h
[hal-command-buffer]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/command_buffer.h
[hal-descriptor-set-layout]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/pipeline_layout.h
[hal-pipeline-layout]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/pipeline_layout.h
[hal-device]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/device.h
[hal-driver]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/driver.h
[hal-executable]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/executable.h
[hal-executable-cache]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/executable_cache.h
[hal-semaphore]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/semaphore.h
[hal-event]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/event.h
[hal-deferred-command-buffer]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/utils/deferred_command_buffer.h
[cu-device]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html
[cu-get-proc-address]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DRIVER__ENTRY__POINT.html#group__CUDA__DRIVER__ENTRY__POINT_1gcae5adad00590572ab35b2508c2d6e0d
[cu-mem-ops]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html
[cu-external-resource]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html
[cu-event]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html
[cu-graph]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH
[cu-launch-host-func]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gab95a78143bae7f21eebb978f91e7f3f
[cuda-stream-command-buffer]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/drivers/cuda/stream_command_buffer.c
[cuda-graph-command-buffer]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/drivers/cuda/graph_command_buffer.c
[cuda-stream-ordered-alloc]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC
