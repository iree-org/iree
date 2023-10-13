# IREE CUDA HAL Driver

This document lists technical details regarding the CUDA implemenation of
IREE's [Hardware Abstraction Layer (HAL)][iree-hal], called a CUDA HAL driver.

Note that there is an existing CUDA HAL driver under the
[`iree/hal/drivers/cuda/`][iree-cuda] directory; what this directory holds is
a rewrite for it. Once this rewrite is mature enough, it will replace the
existing one. For the rewrite rationale, goals, and plans, please see
[Issue #13245][iree-cuda-rewrite].

## Synchronization

### HAL Semaphore

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
to perform the advance from the CPU side. This additionally would mean we can
reuse the logic form CPU signaling to unblock CPU waits.

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
* `iree_hal_cuda2_event_t`: GPU notification mechanism wrapping a `CUevent` and
  a reference count. Used by device signal and wait timepoints. Associates with
  a `iree_hal_cuda2_event_pool_t` pool--returns to the pool directly on once
  reference count goes to 0.
* `iree_hal_cuda2_event_pool_t`: a pool for GPU `iree_hal_cuda2_event_t` objects
  to recycle.
* `iree_hal_cuda2_timepoint_t`: an object that wraps a CPU `iree_event_t` or
  GPU `iree_hal_cuda2_event_t` to represent wait/signal of a timepoint on a
  timeline.
* `iree_hal_cuda2_timepoint_pool_t`: a pool for `iree_hal_cuda2_timepoint_t`
  objects to recycle. This pool builds upon the CPU and GPU event pool--it
  acquires CPU/GPU event objects there.
* `iree_hal_cuda_timeline_semaphore_t`: contains a list of CPU wait and GPU
  wait/signal timepoints.
* `iree_hal_cuda2_queue_action_t`: a pending queue action (kernel launch or
  stream-ordered allocation).
* `iree_hal_cuda2_pending_queue_actions_t`: a data structure to manage pending
  queue actions. It provides APIs to enqueue actions, and advance the queue on
  demand--queue actions are released to the GPU when all their wait semaphores
  are signaled past the desired value, or we can have a `CUevent` object already
  recorded to some `CUstream` to wait on.


[iree-hal]: https://github.com/openxla/iree/tree/main/runtime/src/iree/hal
[iree-cuda]: https://github.com/openxla/iree/tree/main/runtime/src/iree/hal/drivers/cuda
[iree-cuda-rewite]: https://github.com/openxla/iree/issues/13245
[vulkan-timeline-semaphore]: https://www.khronos.org/blog/vulkan-timeline-semaphores
[cu-mem-ops]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html
[cu-external-resource]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html
[cu-event]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html
