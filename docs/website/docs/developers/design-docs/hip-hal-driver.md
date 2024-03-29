---
hide:
  - tags
tags:
  - GPU
  - HIP
---

# HIP HAL driver

This document lists technical details regarding the HIP implemenation of
IREE's Hardware Abstraction Layer, called a HIP HAL driver.

IREE provides a [Hardware Abstraction Layer (HAL)][iree-hal] as a common
interface to different compute accelerators. IREE HAL's design draws inspiration
from modern GPU architecture and APIs; so implementing a HAL driver using HIP
is mostly straightforward; though there are places we need emulation given no
direct mapping concepts or mechanisms. HIP HAL driver draws inspiration from the
CUDA HAL driver and the code structure is based off of that implementation.

## Overall design choices

### HIP driver vs runtime API

IREE HAL's design draws inspiration from modern GPU APIs--it provides explicit
control of low-level GPU objects. The compiler is expected to plan the object
lifetime and schedule workload and synchronization in an optimized way; IREE
HAL implementation and the underlying GPU driver stack is expected to be a thin
layer without much smarts and magic.

Unlike CUDA, HIP doesn't provide two separate API's with the same functionality
in the name of driver and runtime. Instead it extends the HIP API with Modules
and Ctx control API's that the CUDA driver API's exclusively offer.At runtime
the HIP HAL driver will load the `libamdhip64.so`/`amdhip64.dll` library dynamically.

## GPU Objects

### Driver

There is no direct HIP construct that map to the IREE HAL `iree_hal_driver_t`
abstraction. We use it to hold the dynamic symbols loaded for all devices,
and device enumeration and creation.

### Device

`iree_hal_hip_device_t` implements [`iree_hal_device_t`][hal-device] to provide
the interface to HIP GPU device by wrapping a `hipDevice_t`.
For each device, right now we create two `hipStream_t`s--one for issuing commands
for memory allocation and kernel lauches as instructed by the program; the other
for issue host callback functions after dispatched command buffers completes.
See synchronization section regarding the details.

#### Async allocation

The HIP HAL drivers supports async allocation
(`iree_hal_device_queue_alloca()` and `iree_hal_device_queue_dealloca()`)
via HIP stream ordered memory allocation.

The `async_allocations` in the `iree_hal_hip_device_params_t` struct allows
to enable this feature.

### Command buffer

[`iree_hal_command_buffer_t`][hal-command-buffer] is a recording of commands to
issue to the GPU; when the command buffer is submitted to the device it's then
actually executed on the GPU asynchronously.

Two implementations of `iree_hal_command_buffer_t` exist in the HIP HAL
driver--one backed by `hipGraph_t` and the other backed by `hipStream_t`.

`hipGraph_t` conceptually matches `iree_hal_command_buffer_t` better given it's
a recording of commands to issue to the GPU. Also using the `hipGraph_t` API allows
to easily encode fine grain dependencies between dispatch without having to
create multiple streams. Therefore, the `hipGraph_t`-backed implementation is a
more natural one.
Though note that `hipGraph_t` API is meant to be used for recording once and
replaying multiple times and there may be a performance penalty to using
`hipGraph_t` API for one-shot command buffer.

The `hipStream_t`-backed implementation just issues commands directly to a
`hipStream_t` when recording. Commands issued to `hipStream_t` can be immediately
sent to the GPU for execution; there is no recording and replaying separation.
In order to match the recording semantics of `iree_hal_command_buffer_t`, to
use the `hipStream_t`-backed command buffer, we need to first record the command
buffer into an in-memory
[`iree_hal_deferred_command_buffer_t`][hal-deferred-command-buffer], and then
when applying the command buffer, we create a new `hipStream_t`-backed
implementation.

The `command_buffer_mode` in the `iree_hal_hips_device_params_t` struct allows
to select which implementation to use.

### Allocator

The allocator will forward allocation requests to `hipHostMalloc()` for host
local memory, `hipMalloc()` for device local and host invisible memory, and
`hipMallocManaged()` for device local and host visible memory.

### Buffer

HIP buffers are represented either as a host pointer or a device pointer of
type `hipDeviceptr_t`.

### Executable

[`iree_hal_executable_t`][hal-executable] maps naturally to `hipModule_t`.

The compiler generates a FlatBuffer containing a HSACO image as well as a
list of entry point functions and their associated metadata (names,
workgroup size, dynamic shared memory size, etc.). At runtime, the HIP HAL
driver loads the HSACO image and creates `hipFunction_t`s out of it for various
entry points.

## Synchronization

### Event

[`iree_hal_event_t`][hal-event] right now is not used in the compiler so it's
not yet implemented in the HIP HAL driver.

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

In HIP, there is no direct equivalent primitives providing all the capabilities
needed by the HAL semaphore abstraction. Therefore, to implement the support,
we need to leverage multiple native CPU or HIP primitives under the hood.

#### `hipEvent_t` capabilities

The main synchronization mechanism is HIP event--`hipEvent_t`.
As a functionality and integration baseline, we use `hipEvent_t` to implement the
IREE HAL semaphore abstraction.

`hipEvent_t` natively supports the following capabilities:

* State: binary; either unsignaled or signaled. There can exist multiple
  waits (e.g., via `hipEventSynchronize()` or `hipGraphAddEventWaitNode()`) for
  the same `hipEvent_t` signal (e.g., via `hipEventRecord()` or
  `hipGraphAddEventRecordNode()`).
* Ordering: must be signal before wait. Waiting before signal would mean
  waiting an empty set of work, or previously recorded work.
* Direction: device to device, device to host.

We need to fill the remaining capability gaps. Before going into details,
the overall approach would be to:

* State: we need a 64-bit integer value timeline. Given the binary state of
  a `hipEvent_t`, each `hipEvent_t` would just be a "timepoint" on the timeline.
* Ordering: we need to defer releasing the workload to the GPU until the
  semaphore waits are reached on the host, or we can have some device
  `hipEvent_t` to wait on.
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

If there are GPU waits, given that there are no way we can signal a `hipEvent_t` on
CPU, one way to handle this is to cache and defer the submission batches by
ourselves until CPU signals past the desired value. To support this, we would
need to implement a deferred/pending actions queue.

##### GPU signal

GPU signals can only be through a `hipEvent_t` object, which has a binary state.
We need to advance the timeline too. One way is to use `hipLaunchHostFunc()`
to advance from the CPU side with `iree_hal_semaphore_list_signal()`.
This additionally would mean we can reuse the logic form CPU signaling to
unblock CPU waits.

After advancing the timeline from the CPU side with `hipLaunchHostFunc()`,
we can release more workload from the deferred/pending actions queue to the GPU.
Though, per the documentation of `hipLaunchHostFunc()`, "the host function must
not make any HIP API calls." So we cannot do that directly inside `hipLaunchHostFunc()`;
we need to notify another separate thread to call HIP APIs to push more work to the GPU.
So the deferred/pending action queue should have an associcated thread.

For GPU waits, we can also leverage the same logic--using CPU signaling to
unblock deferred GPU queue actions. Though this is performant, given that
the CPU is involved for GPU internal synchronization. We want to use `hipEvent_t`
instead:

* We keep track of all GPU signals in the timeline. Once we see a GPU wait
  request, try to scan the timeline to find a GPU signal that advances the
  timeline past the desired value, and use that for waiting instead. (This
  actually applies to CPU waits too, and it's an optimization over pure
  CPU side `iree_event_t` polling.)
* We may not see GPU signal before seeing GPU wait requests, then we can also
  keep track of all GPU waits in the timeline. Later once see either a CPU
  signal or GPU signal advancing past the waited value, we can handle them
  accordingly--submitting immediately or associating the `hipEvent_t`.
  This would also guarantee the requirement of `hipEvent_t`--recording should
  happen before waiting.
* We can use the same `hipEvent_t` to unblock multiple GPU waits. That's allowed,
  though it would mean we need to be careful regarding `hipEvent_t` lifetime
  management. Here we can use reference counting to see how many timepoints
  are using it and automatically return to a pool once done.

Another problem is that per the `hipLaunchHostFunc()` doc, "the function will
be called after currently enqueued work and will block work added after it."
We don't want the blocking behavior involving host. So we can use a dedicated
`hipStream_t` for launching the host function, waiting on the `hipEvent_t` from the
original stream too. We can also handle resource deallocation together there.

#### Data structures

To summarize, we need the following data structures to implement HAL semaphore:

* `iree_event_t`: CPU notification mechanism wrapping low-level OS primitives.
  Used by host wait timepoints.
* `iree_event_pool_t`: a pool for CPU `iree_event_t` objects to recycle.
* `iree_hal_hip_event_t`: GPU notification mechanism wrapping a `hipEvent_t` and
  a reference count. Used by device signal and wait timepoints. Associates with
  a `iree_hal_hip_event_pool_t` pool--returns to the pool directly on once
  reference count goes to 0.
* `iree_hal_hip_event_pool_t`: a pool for GPU `iree_hal_hip_event_t` objects
  to recycle.
* `iree_hal_hip_timepoint_t`: an object that wraps a CPU `iree_event_t` or
  GPU `iree_hal_hip_event_t` to represent wait/signal of a timepoint on a
  timeline.
* `iree_hal_hip_timepoint_pool_t`: a pool for `iree_hal_hip_timepoint_t`
  objects to recycle. This pool builds upon the CPU and GPU event pool--it
  acquires CPU/GPU event objects there.
* `iree_hal_hip_timeline_semaphore_t`: contains a list of CPU wait and GPU
  wait/signal timepoints.
* `iree_hal_hip_queue_action_t`: a pending queue action (kernel launch or
  stream-ordered allocation).
* `iree_hal_hip_pending_queue_actions_t`: a data structure to manage pending
  queue actions. It provides APIs to enqueue actions, and advance the queue on
  demand--queue actions are released to the GPU when all their wait semaphores
  are signaled past the desired value, or we can have a `hipEvent_t` object already
  recorded to some `hipStream_t` to wait on.

[vulkan-timeline-semaphore]: https://www.khronos.org/blog/vulkan-timeline-semaphores
[iree-hal]: https://github.com/openxla/iree/tree/main/runtime/src/iree/hal
[hal-command-buffer]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/command_buffer.h
[hal-device]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/device.h
[hal-executable]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/executable.h
[hal-event]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/event.h
[hal-deferred-command-buffer]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/utils/deferred_command_buffer.h
