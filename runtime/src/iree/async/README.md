# iree/async/ -- Proactor-Based Async I/O System

## Overview

`iree/async/` is a completion-based (proactor pattern) async I/O layer for IREE.
It provides a unified submission/completion interface over platform-specific
async mechanisms: io_uring on Linux, a POSIX backend using worker threads
with pluggable event notification (poll, epoll, kqueue) on all POSIX platforms,
and IOCP on Windows.

The async system depends only on `iree/base/` and serves as the foundation for
higher-level layers -- HAL drivers, networking, task executors, and the VM
runtime all build on these primitives.

**Design principles:**

- **Completion-based**: Operations are submitted and callbacks fire when
  complete. Natural alignment with io_uring's submission/completion queue model.
- **Caller-driven**: The proactor makes progress only when `poll()` is called.
  No hidden threads, no surprises about which thread callbacks run on. A utility
  wrapper (`util/proactor_thread.h`) provides optional dedicated-thread
  operation.
- **Zero-copy capable**: Registered memory regions, scatter-gather I/O, fixed
  file descriptors, dmabuf registration. The abstraction preserves every
  optimization the kernel offers.
- **Vtable-dispatched**: Proactors and semaphores are polymorphic. Custom
  implementations for testing, embedding, or bridging other systems.
- **No silent failures**: Every operation carries status. Errors propagate
  across async flows with rich annotations and stack traces.
- **Pervasive tracing**: Tracy zones on all significant paths. Fiber tracking
  for async context. Latency measurement built in.

## Design Rationale

ML inference at scale moves large tensors between GPUs, across networks, and
through storage -- with latency budgets measured in microseconds. The data that
matters (model weights, activations, KV caches) lives in GPU VRAM. Moving it
between machines for distributed inference or to NVMe for checkpointing should
not require the CPU to touch every byte. Modern hardware can do this directly:
NICs read from GPU VRAM (GPUDirect RDMA), NVMe controllers write to GPU memory
(GPU Direct Storage), and GPUs access each other's memory across PCIe or other
specialized busses. The software layer's job is to orchestrate these transfers,
not to participate in them.

**Why not select()/poll()/epoll + read()/write()?** Readiness-based I/O
(reactor pattern) tells you "this fd is ready," then you make a separate syscall
to actually do the I/O. Every data movement costs at least two syscalls (wait +
I/O), a copy between kernel and user buffers, and CPU scheduling at every
transition. For a pipeline like "wait for GPU completion -> send result over
network -> signal receiving GPU," a reactor requires the CPU to be in the loop
at every arrow. A completion-based model (io_uring) can execute linked
operations entirely in kernel space -- one submission syscall, zero userspace
transitions between steps.

**Why not vendor libraries (NCCL, RCCL, rocshmem, etc)?** They optimize
one communication pattern (GPU collectives) for one vendor's hardware. They
don't compose with file I/O, timers, signals, or arbitrary network protocols.
You cannot express "wait for GPU work, then send a result over TCP, then write
to NVMe" in NCCL. Layering multiple runtime systems means multiple threading
models, multiple synchronization primitives, and multiple memory management
systems -- each adding latency and variability at every boundary where they
meet. A unified proactor that handles all I/O through one submission/completion
interface eliminates these boundaries entirely.

**Why this design is the correct modern approach.** io_uring, dmabuf, sync_file
fds, GPUDirect RDMA, devmem TCP, GPU Direct Storage -- these are different
kernel subsystems and hardware teams all converging on the same model: register
memory once, submit operations that reference it by handle, let hardware and
kernel execute the pipeline. The proactor pattern is the natural software
expression of what the hardware already does. Approaches that don't follow this
model reintroduce CPU mediation at boundaries where the hardware doesn't need
it.

A unified proactor also means control-plane traffic (RPCs, heartbeats, health
checks) shares the same event loop as data-plane transfers. No separate progress
engine, no extra threads, no cross-system synchronization to coordinate them.
Cancellation and timeouts propagate through the entire dependency graph -- an
RPC deadline can cancel a pending GPU wait and its dependent network send in one
operation, which is not expressible when GPU sync, networking, and timers live
in separate runtime systems.

The same code runs everywhere: io_uring on Linux for production throughput,
kqueue on macOS for development, a threaded fallback for testing and embedded
targets. Capabilities are discovered at runtime and callers adapt gracefully.

This is not datacenter-only: the high-performance io_uring proactor can run on
a Raspberry Pi and the epoll-based proactor can run in a seccomp sandbox on a
cloud VM. Having the same substrate available on all platforms allows for common
connectivity between platforms (Mac <-> Linux <-> Windows <-> whatever).

---

## Architecture

```
+----------------------------------------------------------------+
|                         Applications                           |
+-------------+-------------+-------------+-------------+--------+
|  iree/hal/  |  iree/net/  | iree/task/  |  iree/vm/   |   ..   |
|  (drivers)  |  (network)  | (executors) |  (runtime)  |        |
+-------------+-------------+-------------+-------------+--------+
|                          iree/async/                           |
|                                                                |
|  Proactor       Semaphore       Frontier        Operations     |
|  (vtable)       (vtable)        Tracker         (subtypes)     |
|                                                                |
|  Socket  File  Event  Notification  Region/Span  Relay  Slab   |
|  (ref)   (ref) (ref)  (proactor)    (ref/value)  (reg)  (reg)  |
+----------------------------------------------------------------+
|                          iree/base/                            |
|  (allocators, status, atomics, threading, tracing)             |
+----------------------------------------------------------------+
```

### Dependency Rules

- `iree/async/` depends ONLY on `iree/base/`
- `iree/net/` depends on `iree/async/` (never on `iree/hal/`)
- `iree/hal/` depends on `iree/async/` (implements
  `iree_async_semaphore_vtable_t`) and `iree/net/` (for communications)
- No circular dependencies

### How Layers Connect

HAL drivers bridge to the async world by implementing `iree_async_semaphore_t`.
When GPU work completes, the HAL signals an async semaphore. Network and file
I/O gate on these semaphores through the proactor.

The design point is *proactive scheduling*: the entire pipeline is submitted
before any work completes, so the kernel can chain steps without returning to
userspace between them.

```
Scheduling time (before GPU work finishes):

  Net layer submits a linked sequence:
    steps[0] = SEMAPHORE_WAIT (GPU completion semaphore, target value)
    steps[1] = SEND (GPU output buffer, registered memory)
  → Proactor accepts the pipeline as one submission

Runtime (data path — no host involvement on io_uring):

  GPU finishes
    → HAL driver signals semaphore to target value
      → Kernel satisfies SEMAPHORE_WAIT (io_uring: futex CQE)
        → Kernel chains directly to SEND (linked SQE)
          → NIC transmits from registered buffer

  Proactor poll() eventually sees both CQEs and fires completion
  callbacks for bookkeeping (state tracking, resource release).
  The proactor is not in the data path.
```

On POSIX backends without kernel-chained operations, the proactor
emulates the pipeline: each step's completion triggers submission of
the next step. This adds one poll round-trip per step, but that is
exactly what a reactive approach would do anyway — wait for completion,
then submit the next thing. The proactive scheduling API costs nothing
extra on POSIX while enabling zero-round-trip execution on io_uring.

### Zero-Copy Data Paths

With registered memory (dmabuf, fixed buffers) and kernel-sequenced operations,
the CPU is not in the data path -- hardware and kernel handle the transfers
directly when the platform supports it:

| Path | Mechanism | Data Flow |
|------|-----------|-----------|
| GPU -> NIC | GPUDirect RDMA / dmabuf peer mapping | Send GPU output to peer machine |
| NIC -> GPU | devmem TCP (kernel 6.10+) | Receive directly into VRAM |
| GPU -> NVMe | GPU Direct Storage | Checkpoint from VRAM to disk |
| NVMe -> GPU | GPU Direct Storage | Load weights / restore checkpoints into VRAM |
| GPU -> GPU (same host) | P2P dmabuf export/import | Multi-GPU data sharing across PCIe/NVLink |
| GPU -> GPU (cross host) | GPUDirect RDMA (send from source VRAM, recv into dest VRAM) | Distributed inference handoff between machines |

Buffer registration (`register_buffer()`, `register_dmabuf()`) pins memory and
pre-computes backend handles so that I/O operations reference memory by handle
rather than re-mapping on every operation. Once registered, a GPU buffer can be
used in I/O operations without per-operation page pinning or address translation
-- the cost is paid once at registration time.

### Device Fence Bridging

The proactor bridges between kernel device fences (sync_file fds from GPU
drivers) and the async semaphore system. This connects GPU synchronization
to I/O pipelines so the entire flow can be scheduled ahead of time.

**GPU → Network** (send results when GPU finishes):

```
Schedule:
  1. Import GPU fence:
     import_fence(sync_file_fd, semaphore, value)
     Registers a poll on the sync_file fd; when it signals, the
     proactor advances the semaphore.
  2. Submit linked pipeline:
     steps[0] = SEMAPHORE_WAIT (same semaphore, same value)
     steps[1] = SEND (registered GPU buffer → peer machine)

Execute (kernel-chained on io_uring):
  GPU command buffer completes
    → sync_file fd signals → proactor advances semaphore
      → SEMAPHORE_WAIT satisfied → chains to SEND (linked SQE)
        → NIC reads from GPU VRAM (GPUDirect RDMA if available)
```

**Network → GPU** (receive data for GPU processing):

```
Schedule:
  1. Export fence:
     export_fence(semaphore, value, &sync_file_fd)
     Creates a sync_file fd that signals when the semaphore reaches
     the target value.
  2. Submit GPU command buffer waiting on the exported sync_file fd.
  3. Submit linked pipeline:
     steps[0] = RECV (into registered buffer)
     steps[1] = SIGNAL_SEMAPHORE (advance to target value)

Execute (kernel-chained on io_uring):
  NIC delivers data into registered buffer
    → RECV completes → chains to SIGNAL_SEMAPHORE (linked SQE)
      → Semaphore reaches target value → sync_file fd signals
        → GPU begins processing (its fence dependency is satisfied)
```

In a multi-machine distributed inference pipeline, machine A schedules
[SEMAPHORE_WAIT → SEND] gated on its GPU, machine B schedules
[RECV → SIGNAL_SEMAPHORE] feeding its GPU. After the initial submissions,
the host CPUs on both machines are idle during data movement — the kernel
and hardware handle every transition.

### NUMA-Aware Scaling

Each proactor thread pins to a CPU complex with buffer pools allocated on the
local NUMA node. A 4-socket server with 8 GPUs runs 4 proactor threads, each
handling 2 GPUs and their attached NICs. Data-path memory traffic stays within
the NUMA domain; only control-plane coordination crosses socket boundaries.

---

## Platform Backends

### io_uring (Linux 5.1+)

The primary production backend. Maps naturally to the proactor model —
operations become SQEs, completions arrive as CQEs, and linked sequences
(`IOSQE_IO_LINK`) execute entirely in kernel space. Uses direct syscalls
(no liburing dependency). Created via `iree_async_proactor_create_io_uring()`
from `platform/io_uring/api.h`.

Key features exploited beyond basic I/O:
- Fixed files and registered buffers (avoid per-op fd lookup / page pinning)
- Buffer rings for kernel-selected multishot recv buffers
- Linked SQEs for zero-round-trip operation sequences
- Zero-copy send (`SEND_ZC`), MSG_RING cross-proactor messaging
- Futex ops (6.7+) for kernel-side semaphore waits in LINK chains
- Sync_file fd polling for device fence import

See [`platform/io_uring/`](platform/io_uring/) for implementation details.

### POSIX Backend (All POSIX Platforms)

Broad-coverage backend using a pluggable event notification mechanism
for fd monitoring. Emulates features that io_uring provides natively
(linked operations, multishot, etc.) with per-step poll round-trips —
functionally equivalent, same API, higher per-step latency. Created via
`iree_async_proactor_create_posix()` (platform-default event backend) or
`iree_async_proactor_create_posix_with_backend()` (explicit selection).

| Event Backend | Platform | Scaling | Selection Constant |
|---------------|----------|---------|---------------------|
| poll() | All POSIX | O(n) fds | `IREE_ASYNC_POSIX_EVENT_BACKEND_POLL` |
| epoll | Linux | O(1) ready | `IREE_ASYNC_POSIX_EVENT_BACKEND_EPOLL` |
| kqueue | macOS/BSD | O(1) ready | `IREE_ASYNC_POSIX_EVENT_BACKEND_KQUEUE` |

The default selects epoll on Linux, kqueue on macOS/BSD, poll() elsewhere.

See [`platform/posix/`](platform/posix/) for implementation details.

### Windows IOCP

IOCP is planned but not yet implemented. It's closer in behavior to io_uring
than the polling-based POSIX backend.

### Platform Selection

`iree_async_proactor_create_platform()` (from `proactor_platform.h`) selects the
best available backend:
- **Linux**: io_uring (kernel 5.1+), falls back to POSIX
- **macOS/BSD**: POSIX with kqueue
- **Other POSIX**: POSIX with poll()

---

## Capability Matrix

Backends report capabilities via `iree_async_proactor_query_capabilities()`.
Callers use these to select optimal code paths or skip features gracefully.

| Capability | POSIX | io_uring | Notes |
|---|---|---|---|
| `MULTISHOT` | Emulated (poll re-arm) | 5.19+ native | Persistent accept/recv |
| `FIXED_FILES` | Emulated | 5.1+ native | Registered fd table |
| `REGISTERED_BUFFERS` | Emulated | 5.1+ native | Pre-pinned DMA buffers |
| `LINKED_OPERATIONS` | Emulated (callback chain) | 5.3+ native | Kernel-side sequences |
| `ZERO_COPY_SEND` | Not supported (copy) | 6.0+ native | MSG_ZEROCOPY / SEND_ZC |
| `DMABUF` | Not supported | 5.19+ | GPU memory registration |
| `DEVICE_FENCE` | Poll-based (fd import) | Poll-based (fd import) | sync_file bridging |
| `ABSOLUTE_TIMEOUT` | Emulated (relative) | 5.4+ native | Drift-free timers |
| `FUTEX_OPERATIONS` | Not supported | 6.7+ | Kernel-side futex in LINK chains |
| `PROACTOR_MESSAGING` | MPSC queue + wake | 5.18+ MSG_RING | Cross-proactor messages |

"Emulated" means the API is functional but uses a software fallback with
per-operation overhead rather than a kernel-optimized path.

---

## Core Concepts

Each concept has a dedicated header file. This section explains how the pieces
fit together; see the referenced headers for authoritative type definitions and
function signatures.

### Proactor (`proactor.h`)

The central abstraction. Manages async operation submission and completion
dispatch. Vtable-dispatched for backend polymorphism.

The proactor's event loop is three methods:

- **`submit(operations)`**: Hand operations to the proactor for async execution.
  Thread-safe (batched internally via MPSC queue on POSIX, or direct SQE fill
  on io_uring). Maps to a single `io_uring_submit()` or `kevent()` call
  internally where possible.
- **`poll(timeout)`**: Block until completions arrive (or timeout). Invoke
  callbacks for all completed operations. Returns the count of callbacks fired.
  Single-thread ownership: the first thread to call `poll()` becomes the poll
  owner for the proactor's lifetime.
- **`wake()`**: Thread-safe, idempotent. Interrupts a blocked `poll()` from
  another thread. Async-signal-safe on POSIX.

Additional methods: `cancel()` for in-flight operations,
`query_capabilities()` for feature detection, resource creation (sockets, files,
events, notifications), relay registration, buffer/slab registration, device
fence import/export, cross-proactor messaging, and signal subscription.

### Operations (`operation.h`, `operations/*.h`)

All operations inherit from `iree_async_operation_t`. Caller-owned storage
(intrusive -- no proactor allocation on submit). The proactor invokes the
callback when the operation completes.

Operation subtypes live in `operations/`:
- `operations/scheduling.h` -- nop, timer, event_wait, sequence
- `operations/semaphore.h` -- semaphore wait, semaphore signal
- `operations/net.h` -- accept, connect, recv, recv_pool, send, sendto,
  recvfrom, close
- `operations/file.h` -- open, read, write, close
- `operations/futex.h` -- futex wait, futex wake (io_uring 6.7+ only)
- `operations/message.h` -- cross-proactor message (io_uring MSG_RING)

Each subtype extends the base with type-specific parameters (inputs) and
results (outputs filled by the proactor on completion).

Behavioral flags on operations:
- `IREE_ASYNC_OPERATION_FLAG_MULTISHOT`: Persistent delivery (accept, recv).
  Callbacks fire with `IREE_ASYNC_COMPLETION_FLAG_MORE` until final.
- `IREE_ASYNC_OPERATION_FLAG_LINKED`: Kernel-side chaining to next operation
  in submission batch (requires `LINKED_OPERATIONS` capability).

### Semaphores (`semaphore.h`)

Cross-layer synchronization primitive with timeline semantics (monotonically
increasing uint64 values). This is the bridge between GPU work and I/O:

- HAL drivers implement the semaphore vtable (signal when GPU work completes)
- The async layer waits on semaphores before submitting dependent I/O
- A software semaphore implementation exists for pure-CPU coordination

Key vtable methods: `query()`, `signal(value, frontier)`, `fail(status)`,
`acquire_timepoint()`, `cancel_timepoint()`, `query_frontier()`,
`export_primitive()`.

The `signal` method takes an optional `iree_async_frontier_t*` for causal
context propagation. Pass NULL for local-only signals. Pass non-NULL when the
signal's ordering must propagate to remote machines.

Failure is sticky: once a semaphore is failed, all current and future waiters
receive the failure status.

### Resources (`socket.h`, `file.h`, `event.h`)

Proactor-managed handles wrapping platform primitives. All are ref-counted.

- **Sockets**: TCP (v4/v6), UDP (v4/v6), Unix stream, Unix dgram. Created with
  immutable options (REUSE_ADDR, REUSE_PORT, NO_DELAY, KEEPALIVE, ZERO_COPY,
  etc.), then configured with bind/listen (synchronous). Async operations:
  accept, connect, recv, recv_pool, send, sendto, recvfrom, close. Imported
  sockets (pre-existing fds) supported via `import_socket`.
- **Files**: Positioned I/O (pread/pwrite semantics). Imported via
  `import_file` from existing fds. Async operations: open, read, write, close.
- **Events**: Lightweight signaling primitive for cross-thread wakeup. `set()`
  from any thread, wait via `event_wait` operations. Platform-native mechanisms:
  eventfd on Linux, pipes on macOS/BSD.

Sticky failure: once a socket encounters an error, it enters a permanently
failed state. Subsequent operations complete immediately with the recorded
failure.

### Notifications (`notification.h`)

Level-triggered signaling for waking worker threads from I/O completions.

Unlike events (edge-triggered, one signal per wait), notifications use epoch
counting: multiple signals coalesce, and waiters observe any signal that
occurred after their wait was submitted.

Key operations:
- `iree_async_notification_signal(notification, wake_count)` -- Thread-safe,
  async-signal-safe.
- `iree_async_notification_wait(notification, timeout)` -- Blocking wait for
  worker threads outside the proactor's poll loop.
- `NOTIFICATION_WAIT` / `NOTIFICATION_SIGNAL` operation types -- Async variants
  that integrate with the proactor's event loop and support LINK chains.

Platform mapping: futex on Linux (when available), condvar fallback on macOS.
On io_uring 6.7+, notification operations use kernel-side FUTEX_WAIT/WAKE for
relay LINK chains without userspace round-trips.

### Relays (`relay.h`)

Declarative source-to-sink event dataflow. A relay connects an event source to
an event sink: "when X happens, trigger Y."

Source types:
- `PRIMITIVE`: fd/HANDLE becomes readable (POLL_ADD)
- `NOTIFICATION`: notification epoch advances

Sink types:
- `SIGNAL_PRIMITIVE`: write to an fd (eventfd write)
- `SIGNAL_NOTIFICATION`: signal a notification

Relay flags: `PERSISTENT` (re-arm after each trigger), `OWN_SOURCE_PRIMITIVE`
(close source fd on unregister), `ERROR_SENSITIVE` (invoke error callback on
re-arm failure).

On io_uring, certain source/sink combinations execute entirely in kernel space
via LINK chains (e.g., POLL_ADD -> WRITE, POLL_ADD -> FUTEX_WAKE).

### Memory (`region.h`, `span.h`, `slab.h`, `types.h`)

Registered memory for zero-copy I/O.

- **Region**: Ref-counted registered memory block with backend-specific handles
  (io_uring buffer IDs, provided buffer ring group IDs). Created by the
  proactor during `register_buffer()` / `register_dmabuf()` / `register_slab()`.
- **Span**: Value-type subrange of a region `{region, offset, length}`. Used in
  all I/O operations. Non-owning, but the proactor retains the span's region
  during in-flight operations.
- **Slab**: Contiguous memory block divided into fixed-size slots for indexed
  buffer allocation. Register with `register_slab()` for zero-copy I/O.
  Singleton constraint: only one READ-access slab per proactor (mirrors
  io_uring's single fixed buffer table).
- **Buffer registration state** (`types.h`): Header-only types embeddable in
  HAL buffers. Tracks which proactors a buffer is registered with. The `:types`
  Bazel target has no link dependency -- suitable for embedding in HAL code
  without pulling in the full async library.

### Frontiers (`frontier.h`, `frontier_tracker.h`)

Vector clocks for causal ordering across machines.

A frontier is a set of `(axis, epoch)` pairs. Each axis identifies a causal
source (a GPU queue, a collective, a host thread). Each epoch is a monotonic
timeline value. A frontier says "I depend on all of these axes having reached
at least these epochs."

The frontier tracker maps axes to semaphores and dispatches waiters when
frontiers are satisfied.

Frontiers are for remote coordination. Local operations (same machine) use
semaphores directly.

### Buffer Pool (`buffer_pool.h`)

Pre-registered slab of fixed-size buffers with O(1) acquire/release. Used for
pool-based multishot receives where the kernel (io_uring provided buffer ring)
or the proactor selects the receive buffer.

### Event Pool (`event_pool.h`)

High-performance pooling for async events. Separate Bazel target (`:event_pool`)
for consumers that need bulk event allocation.

### Affinity (`affinity.h`)

NUMA-aware locality domain. Groups CPU cores, memory controllers, and PCIe
devices. Used at pool and proactor creation time to ensure NUMA-local
allocation.

### Signal Handling (`proactor.h`, `util/signal.h`)

Process-wide signal subscription through the proactor. On Linux, uses signalfd
for efficient kernel-managed delivery. On other POSIX platforms, uses a
self-pipe pattern.

Only one proactor per process may own signal subscriptions. Signals are
dispatched as callbacks from within `poll()`.

Supported signals: `INTERRUPT` (SIGINT), `TERMINATE` (SIGTERM), `HANGUP`
(SIGHUP), `QUIT` (SIGQUIT), `USER1` (SIGUSR1), `USER2` (SIGUSR2).

Startup utilities:
- `iree_async_signal_block_default()`: Block handled signals before creating
  threads.
- `iree_async_signal_ignore_broken_pipe()`: Ignore SIGPIPE globally (required
  for network servers).

### Cross-Proactor Messaging (`proactor.h`, `operations/message.h`)

Send messages between proactors running on different threads.

Two interfaces:
- `iree_async_proactor_send_message()`: Fire-and-forget, thread-safe, minimal
  overhead. Sets a message callback via
  `iree_async_proactor_set_message_callback()`.
- `MESSAGE` operation type: Supports LINK chains and completion callbacks.
  Requires `PROACTOR_MESSAGING` capability for kernel-mediated delivery.

On io_uring 5.18+, messages are delivered via MSG_RING (kernel posts CQE
directly to target ring). On POSIX, messages use a pre-allocated lock-free pool
with wake.

---

## Memory Ownership

### Operation Ownership

The ownership rule: **caller owns the operation before submit and after the
final callback. The proactor owns it in between.**

```
Single-shot:
  1. Caller allocates/acquires operation
  2. Caller fills parameters
  3. Caller submits to proactor          -> proactor owns
  4. Poll invokes callback               -> caller owns again (can reuse/release)

Multishot (ACCEPT, RECV with MULTISHOT flag):
  1-3. Same as single-shot
  4. Poll invokes callback with IREE_ASYNC_COMPLETION_FLAG_MORE  -> proactor still owns
  5. ... more callbacks with MORE flag ...
  6. Final callback without MORE flag    -> caller owns again

Cancellation:
  1. Caller calls cancel(operation)
  2. Proactor eventually invokes callback with IREE_STATUS_CANCELLED, no MORE flag
  3. Caller owns operation again
```

### Multishot Termination

Multishot operations persist until one of these conditions:

1. **Resource close**: Closing the underlying socket/file generates a final
   callback (error status, no MORE flag).
2. **Error**: Network disconnect, peer close, etc. generates a final callback.
3. **Explicit cancellation**: `cancel()` generates a final callback with
   `IREE_STATUS_CANCELLED`.

**Critical**: `release()` does NOT terminate multishot operations. The operation
holds a reference to the resource, keeping it alive.

**Correct cleanup pattern**:
```c
// 1. Close the socket (async operation).
iree_async_socket_close_operation_t close_op = {0};
close_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE;
close_op.socket = listener;
close_op.base.completion_fn = on_close;
iree_async_proactor_submit_one(proactor, &close_op.base);

// 2. Wait for:
//    - The close completion (close_op callback fires)
//    - The multishot final callback (no MORE flag)
// Both will fire; order depends on kernel behavior.

// 3. Release the socket AFTER both callbacks complete.
iree_async_socket_release(listener);
```

### Span Region Lifetime

Spans are non-owning, but operations that embed spans may outlive the caller's
scope. The proactor guarantees region safety:

- **At submit time**: the proactor retains each span's region.
- **After the final callback**: the proactor releases each span's region. For
  multishot operations, the release happens only on the final invocation.
- **NULL regions**: no retain/release. The caller manages raw memory lifetime.

### Resource Lifetime Rules

- **Proactor** outlives all resources created from it (sockets, files, events,
  notifications, relays, pools, registrations).
- **Sockets/files/events** must not be destroyed while operations referencing
  them are in flight. Cancel or wait for completion first.
- **Buffer pools** must have all leases returned before `_free()`.
- **Semaphores** are retained by their timepoints. Safe to release the caller's
  reference while timepoints are pending.

---

## Thread Safety Model

| Type | Thread Safety |
|------|---------------|
| Proactor `submit()` | Thread-safe (batched internally) |
| Proactor `poll()` | Single-thread ownership (one poller per proactor) |
| Proactor `wake()` | Thread-safe, idempotent, async-signal-safe |
| Proactor `cancel()` | Thread-safe |
| Proactor `send_message()` | Thread-safe |
| Semaphore `signal()` | Thread-safe |
| Semaphore `query()` | Thread-safe (atomic load) |
| Semaphore `acquire_timepoint()` | Thread-safe |
| Event `set()` | Thread-safe |
| Notification `signal()` | Thread-safe, async-signal-safe |
| Notification `wait()` | Thread-safe (blocking, for worker threads) |
| Buffer pool acquire/release | **NOT** thread-safe (proactor-thread only) |
| Frontier tracker `advance()` | Thread-safe |
| Frontier tracker `wait()` | Thread-safe |
| Registration state | **NOT** thread-safe (serialize setup, then read-only) |
| Signal subscribe/unsubscribe | Must be serialized with poll() |
| Event source register/unregister | Must be serialized with poll() |
| Relay register/unregister | Must be serialized with poll() |

**The golden rule**: Callbacks fire from `poll()`, on the thread that calls
`poll()`. If you use `iree_async_proactor_thread_t`, that's the proactor's
dedicated thread.

---

## Building and Testing

```bash
# All async tests (unit tests + CTS across all backends):
iree-bazel-test --config=asan //runtime/src/iree/async/...

# CTS tests for a specific backend:
iree-bazel-test --config=asan //runtime/src/iree/async/platform/io_uring/cts:core_tests
iree-bazel-test --config=asan //runtime/src/iree/async/platform/posix/cts:socket_tests

# CTS benchmarks (no ASAN for meaningful numbers):
iree-bazel-test --compilation_mode=opt \
    //runtime/src/iree/async/platform/io_uring/cts:core_benchmarks_test
```

---

## Conformance Test Suite (CTS)

The CTS validates all proactor backends against a shared set of test suites
and benchmarks. Tests are written once and run against every registered backend
configuration (e.g. 5 io_uring configurations with different capability masks,
plus per-platform and per-feature POSIX configurations). Tag-based filtering
ensures tests only run against backends that support the features they exercise.

Test suites cover core operations, sockets, events, synchronization primitives,
buffer pools, and futex operations. Benchmarks measure dispatch scalability,
sequence overhead, relay fan-out, socket throughput, and event pool performance.

CTS test and benchmark suites are libraries in `cts/`. Runnable test binaries
are assembled at the backend level via link-time composition — see the
`platform/{io_uring,posix}/cts/` directories.

See [`cts/`](cts/) for the CTS architecture, backend configurations, tag
filtering, and test categories.

---

## API Quick Reference

### Create a Proactor

```c
#include "iree/async/proactor_platform.h"

iree_async_proactor_t* proactor = NULL;
iree_async_proactor_options_t options = iree_async_proactor_options_default();
IREE_RETURN_IF_ERROR(iree_async_proactor_create_platform(
    options, iree_allocator_system(), &proactor));

// Or create a specific backend:
#include "iree/async/platform/io_uring/api.h"
IREE_RETURN_IF_ERROR(iree_async_proactor_create_io_uring(
    options, iree_allocator_system(), &proactor));

#include "iree/async/platform/posix/proactor.h"
IREE_RETURN_IF_ERROR(iree_async_proactor_create_posix(
    options, iree_allocator_system(), &proactor));

// Cleanup (must have quiesced):
iree_async_proactor_release(proactor);
```

### Submit and Poll

```c
#include "iree/async/api.h"

// Set up a timer operation.
iree_async_timer_operation_t timer = {0};
iree_async_operation_initialize(&timer.base, IREE_ASYNC_OPERATION_TYPE_TIMER,
                                IREE_ASYNC_OPERATION_FLAG_NONE,
                                on_timer_callback, user_data);
timer.deadline_ns = iree_time_now() + iree_make_duration_ms(100);

// Submit.
IREE_RETURN_IF_ERROR(iree_async_proactor_submit_one(proactor, &timer.base));

// Poll until completion.
while (!done) {
  iree_status_t status = iree_async_proactor_poll(
      proactor, iree_make_timeout_ms(1000), NULL);
  if (iree_status_is_deadline_exceeded(status)) {
    iree_status_ignore(status);
    continue;
  }
  IREE_RETURN_IF_ERROR(status);
}
```

### Linked Pipeline (Proactive Scheduling)

```c
#include "iree/async/api.h"

// Schedule: "when the GPU semaphore is signaled, send the result."
// The entire pipeline is submitted before the GPU finishes. On io_uring,
// both steps become linked SQEs that execute in kernel space with no
// userspace round-trips between them.

// Step 1: Wait for GPU completion semaphore.
iree_async_semaphore_wait_operation_t wait = {0};
iree_async_operation_initialize(&wait.base,
    IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
    IREE_ASYNC_OPERATION_FLAG_NONE, NULL, NULL);
wait.semaphores = &gpu_semaphore;
wait.values = &gpu_target_value;
wait.count = 1;
wait.mode = IREE_ASYNC_WAIT_MODE_ALL;

// Step 2: Send GPU output buffer to peer machine.
iree_async_socket_send_operation_t send = {0};
iree_async_operation_initialize(&send.base,
    IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND,
    IREE_ASYNC_OPERATION_FLAG_NONE, NULL, NULL);
send.socket = peer_socket;
send.buffers = iree_async_span_list_make(&gpu_output_span, 1);

// Combine into a sequence with no step_fn (eligible for linked SQEs).
iree_async_operation_t* steps[] = {&wait.base, &send.base};
iree_async_sequence_operation_t seq = {0};
iree_async_operation_initialize(&seq.base,
    IREE_ASYNC_OPERATION_TYPE_SEQUENCE,
    IREE_ASYNC_OPERATION_FLAG_NONE,
    on_pipeline_complete, user_data);
seq.steps = steps;
seq.step_count = 2;
seq.step_fn = NULL;

IREE_RETURN_IF_ERROR(iree_async_proactor_submit_one(proactor, &seq.base));

// The proactor handles the rest:
//   GPU completes → semaphore signal → wait satisfied → send fires
// on_pipeline_complete fires for cleanup, not for data-path decisions.
```

### Dedicated Poll Thread

```c
#include "iree/async/util/proactor_thread.h"

iree_async_proactor_thread_t* thread = NULL;
iree_async_proactor_thread_options_t thread_options =
    iree_async_proactor_thread_options_default();
thread_options.debug_name = iree_make_cstring_view("io-main");
IREE_RETURN_IF_ERROR(iree_async_proactor_thread_create(
    proactor, thread_options, iree_allocator_system(), &thread));

// ... submit operations from any thread ...
// Completions fire on the proactor thread automatically.

// Shutdown:
iree_async_proactor_thread_request_stop(thread);
IREE_RETURN_IF_ERROR(iree_async_proactor_thread_join(
    thread, IREE_DURATION_INFINITE));
iree_async_proactor_thread_release(thread);
```

### Check Capabilities

```c
iree_async_proactor_capabilities_t caps =
    iree_async_proactor_query_capabilities(proactor);
if (iree_any_bit_set(caps, IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND)) {
  // Use zero-copy send path.
}
if (iree_any_bit_set(caps, IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS)) {
  // Use kernel-chained sequences.
}
```
