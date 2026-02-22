# Causal Dependency Tracking with Vector Clock Frontiers

This document describes the design of IREE's causal dependency tracking system
for heterogeneous, multi-device, and multi-host scheduling. It is written for
developers familiar with timeline semaphores (Vulkan) or fences (D3D12) and
binary events (CUDA/HIP) who want to understand what frontiers add and why.

<!-- markdownlint-disable MD013 -->

<div class="visualizer-link" markdown>
[Open interactive visualizer :material-open-in-new:](visualizer/){ .md-button target="_blank" }
</div>

<iframe id="visualizer-iframe" src="visualizer/?inline&scenario=streaming-translation"
        style="width: 100%; height: 800px; border: 1px solid #e0e0e0; border-radius: 8px;"
        loading="lazy"
        title="Vector Clock & Frontier Explorer"></iframe>

<!-- markdownlint-enable MD013 -->

<script>
// Sync the visualizer iframe theme with the mkdocs Material color scheme.
(function() {
  var iframe = document.getElementById('visualizer-iframe');
  if (!iframe) return;
  function isDark() {
    return document.body.getAttribute('data-md-color-scheme') === 'slate';
  }
  // Set initial src with dark parameter if the page is already in dark mode.
  if (isDark()) {
    iframe.src = iframe.src + '&dark';
    iframe.style.borderColor = '#333';
  }
  // On theme toggle, notify the iframe and update the border.
  new MutationObserver(function() {
    var dark = isDark();
    iframe.style.borderColor = dark ? '#333' : '#e0e0e0';
    if (iframe.contentWindow) {
      iframe.contentWindow.postMessage({type: 'theme', dark: dark}, '*');
    }
  }).observe(document.body, {
    attributes: true,
    attributeFilter: ['data-md-color-scheme']
  });
})();
</script>

The interactive simulator above visualizes these concepts with step-through
execution, DAG rendering, and frontier propagation. Use the scenario selector
to explore different dependency patterns, and the width/depth/GPU sliders to
see how scheduling scales. Open the
[standalone visualizer](visualizer/) for the full experience
with semaphore state tables, operation frontiers, and event logs.

---

## The Problem

Modern ML workloads span CPUs, GPUs, NPUs, NICs, and NVMe devices — often
across multiple machines. Each device has its own execution queues, and work
on one device frequently depends on work completing on another. The scheduling
system needs to answer three questions efficiently:

1. **Ordering**: When can a new operation safely begin, given its dependencies?
2. **Memory safety**: When can a buffer be reused after the operations that
   reference it have completed?
3. **Pipeline scheduling**: Can a remote device queue dependent work *before*
   the prerequisite work has completed, knowing the hardware FIFO will enforce
   correct ordering once the prerequisite arrives?

Timeline semaphores (monotonically increasing uint64 values with >= wait
semantics) answer question 1 for a *single* dependency chain. But they don't
carry information about *what else* must have happened for a given timeline
value to be reached. A wait on semaphore S >= 5 tells you S reached 5, but
not which queues on which devices contributed to reaching that value.

Frontiers answer all three questions by attaching a compact causal history to
every semaphore signal, making the transitive dependency structure available
to any scheduler that receives the signal — without round-trips.

---

## Frontiers

A **frontier** is a sparse vector clock: a set of (axis, epoch) pairs where
each axis identifies a timeline participant (a queue, collective channel, or
host thread) and each epoch is the latest known position on that timeline.

```text
Frontier = { (axis_A, 7), (axis_B, 3), (axis_C, 12) }
```

This frontier asserts: "everything that happened on axis A up to epoch 7,
on axis B up to epoch 3, and on axis C up to epoch 12 is in my causal past."

### Operations

**Merge** combines two frontiers by taking the component-wise maximum:

```text
merge({A:5, B:3}, {A:2, B:7, C:4}) = {A:5, B:7, C:4}
```

Merge is the "join" of two causal histories — the result knows everything
either input knew. It is associative, commutative, and idempotent.

**Dominance** checks whether one frontier's knowledge subsumes another's:

```text
dominates({A:5, B:7, C:4}, {A:3, B:7}) = true   (A:5>=3, B:7>=7)
dominates({A:5, B:7},      {A:3, C:4}) = false  (C missing from left)
```

If frontier F dominates frontier G, then everything G depends on has already
happened according to F's knowledge. This is the core predicate for scheduling
decisions: "are this operation's prerequisites satisfied?"

**Insert-or-raise** adds or updates an (axis, epoch) entry:

```text
insert_or_raise({A:5, B:3}, C, 4) = {A:5, B:3, C:4}
insert_or_raise({A:5, B:3}, A, 8) = {A:8, B:3}
```

### Capacity and Overflow

Frontiers have a fixed maximum number of entries. When a new axis would exceed
capacity, the entry with the smallest epoch (the least recently relevant axis)
is evicted and the frontier is marked as **tainted**. Tainted frontiers remain
correct for dominance checking — they are conservative, never optimistic —
but certain scheduling optimizations that depend on complete causal knowledge
are disabled.

In practice, most operations interact with a small number of axes. The capacity
is sized to accommodate the common case (a few queues plus a collective
channel or two) without overflow. Collective channels (described below) are
the primary mechanism for keeping frontier size manageable at scale.

---

## Axes

An **axis** is a globally unique, never-reused 64-bit identifier for a
timeline participant. The "never-reused" property prevents ABA hazards in
frontier comparisons: if axis A appears in a frontier, it refers to exactly
one specific timeline, even if that timeline's queue has been destroyed.

Axes encode enough structure for locality-aware scheduling:

- **Machine index**: Which machine in a cluster (for routing decisions)
- **Domain**: What kind of timeline (queue, collective, host thread, etc.)
- **Ordinal**: Unique within (machine, domain), monotonically increasing

### Queue Axes

Each hardware queue (CUDA stream, HIP stream, Vulkan queue, CPU task executor)
gets a unique axis at creation time. Operations submitted to that queue
increment the epoch. The (axis, epoch) pair uniquely identifies a causal
position within that queue's timeline.

### Collective Channel Axes

Multi-device collective operations (all-reduce, all-gather, broadcast) are
assigned a *single shared axis* rather than separate per-device axes. When
8 GPUs participate in an all-reduce, the result is one frontier entry
`(collective_axis, sequence_number)` instead of eight entries for eight
individual queues.

This compression is essential for scaling: without it, an 8-GPU tensor-
parallel model would consume most of the frontier capacity on every operation,
leaving little room for cross-model dependencies or other system axes (NVMe,
NIC, CPU). With collective channels, tensor parallelism's synchronization
cost in the frontier is constant regardless of device count.

---

## Semaphores

A **semaphore** is a timeline value (monotonically increasing uint64) paired
with a frontier. When an operation signals a semaphore to a new value, it
attaches the operation's causal frontier. When another operation waits on that
semaphore, it imports the frontier — gaining transitive knowledge of
everything the signaler depended on.

### Signal

When queue Q signals semaphore S to value V:

1. Q's current frontier (including Q's own axis at its current epoch) is
   computed.
2. S's stored frontier is updated to Q's frontier.
3. S's timeline value is advanced to V.

The frontier attached to S at value V is a compact summary of all the work
that happened before the signal. Any consumer that waits on S >= V will
inherit this summary.

### Wait

When queue Q waits on semaphore S >= V:

1. If S's current value is already >= V, the wait resolves immediately (no
   device operation needed).
2. Q imports S's frontier (the one attached at the time the value was reached),
   merging it into Q's own frontier via component-wise max.
3. Q now has transitive knowledge of everything the signaler knew.

### Transitivity

This is the crucial property. Consider three queues:

```text
Q_A signals semaphore S_1 to value 1
  S_1 frontier: {A: 5}

Q_B waits on S_1 >= 1, then signals semaphore S_2 to value 1
  S_2 frontier: {A: 5, B: 3}  (merged A's frontier with B's own axis)

Q_C waits on S_2 >= 1
  Q_C's frontier after import: {A: 5, B: 3, C: ...}
```

Q_C now knows about Q_A's work, even though Q_C never directly interacted
with Q_A or semaphore S_1. The knowledge propagated transitively through
the frontier merge at Q_B's signal of S_2.

This transitivity is what enables remote scheduling without round-trips:
when a remote machine receives a wait frontier, it can locally determine
whether the prerequisite work across all contributing queues (including
queues on other machines it has never communicated with directly) has been
completed or can be pipelined.

### Scheduling vs. Causality

The >= semantics of timeline semaphores create an important distinction.
Consider:

```text
Q signals S to values 1, 2, 3, 4, 5 (five sequential operations)
Later, Q_B submits a wait on S >= 2
```

If S is already at 5 when Q_B's wait is processed, the wait resolves
immediately — S >= 2 is trivially satisfied. But Q_B's frontier inherits the
frontier from the signal that set S to 2, **not** from the signal that set
S to 5. Q_B depends on "S reached at least 2." The work that advanced S
from 2 to 5 is not in Q_B's causal past — it happened to run, but Q_B
didn't depend on it.

This decoupling keeps frontiers tight: they track actual dependencies, not
coincidental execution order. Try
[scenario 6](visualizer/?scenario=late-waiter) in the
visualizer to see this property in action.

---

## Scopes

Not all semaphores need the same level of hardware support. A semaphore that
is only waited on by the same queue that signals it needs no device primitive
at all — ordering is implicit in the queue's FIFO. A semaphore shared across
machines needs cross-process signaling support.

Semaphore scopes are immutable after creation:

- **Queue-local**: No device primitive allocated. Ordering comes from the
  queue's FIFO and frontier tracking. Zero synchronization overhead.
- **Device-local**: A device-side primitive (HSA signal, Vulkan timeline
  semaphore, etc.) enables cross-queue signaling within one device.
- **Device-group**: P2P-capable primitive (XGMI, NVLink) enables direct
  GPU-to-GPU signaling without host involvement.
- **Process-local**: Host-visible primitive for devices without P2P capability.
- **System-wide**: Exportable handle (sync_fd, Win32 handle) for cross-process
  or cross-driver synchronization.

Each scope level adds latency but also visibility. The scheduler selects the
narrowest scope that satisfies the semaphore's actual usage pattern, ensuring
that the common case (queue-local sequential work) pays nothing.

---

## Taint

When a semaphore is advanced by an external system (a CUDA event import, a
sync_fd from another process, work submitted outside the scheduler's control),
the scheduler does not have full causal knowledge of what contributed to that
advancement. The affected timeline values are marked **tainted**.

Taint is tracked per-timepoint using a watermark: all timeline values up to
the watermark are untainted (the scheduler has full knowledge), and values
beyond the watermark are tainted (external origin, incomplete knowledge).

Tainted values remain correct — the scheduler will not make unsound decisions.
But it will conservatively issue device waits rather than eliding them via
frontier dominance, because it cannot prove the prerequisites are satisfied
from frontier information alone.

When the scheduler subsequently signals past the tainted range (through its
own HAL-tracked operations), the watermark advances and performance
automatically recovers. Taint is not permanent — it precisely tracks the
window of external uncertainty.

---

## Buffer Reuse

When a buffer is deallocated, the deallocating queue records its current
frontier as the buffer's **death frontier**. This captures "everything that
happened before this buffer was freed" — which, by the structure of the
compiled program, includes all operations that read or wrote the buffer.

When another queue wants to reuse the buffer, it checks:

```text
dominates(requesting_queue_frontier, buffer_death_frontier)
```

If true, the requesting queue has already observed the completion of all work
that used the buffer. Reuse is safe with zero synchronization.

If dominance fails, the allocator can scan the signaling queue's recent
timeline ring for minimal waits that would close the gap, issue those device
waits, and re-test dominance. This is the "try before you fence" pattern:
most reuse checks succeed on the dominance fast path, and the fallback
requires only the minimum necessary device waits.

This eliminates per-use buffer tracking entirely. A weight tensor read by
hundreds of operations has one death frontier recorded at deallocation, not
hundreds of per-operation events. The cost is O(1) per buffer lifecycle, not
O(operations) per buffer lifetime.

---

## Submission Pipeline

### Submit-Before-Wait

Timeline semaphores with >= semantics enable **submit-before-wait**: a
consumer can submit work that waits on a timeline value that hasn't been
signaled yet. The hardware queue holds the work until the signal arrives.

Combined with frontiers, this enables speculative pipeline construction.
A client can submit an entire multi-stage pipeline at once:

```text
Stage 1 on GPU_A: signal(S, 1)
Stage 2 on GPU_B: wait(S, 1), signal(S, 2)
Stage 3 on GPU_B: wait(S, 2), signal(S, 3)
```

All three stages are submitted before any work begins. GPU_B receives
stages 2 and 3 and queues them on its hardware FIFO. When GPU_A completes
stage 1 and signals S to 1, GPU_B's hardware automatically unblocks stage 2.
When stage 2 completes and signals S to 2, stage 3 unblocks. No round-trips
to the client are needed.

The frontier system makes this safe for remote schedulers: when GPU_B
receives stage 3's wait on S >= 2, the frontier at S=2 will contain GPU_A's
axis (transitively through stage 2). GPU_B's scheduler can verify that all
transitive dependencies will be resolved in order.

### Wait-Before-Signal (Same Queue)

On backends that support it natively (Vulkan timeline semaphores, D3D12
fences, HSA barrier packets), a queue can submit a wait for a value that
a later submission on the *same queue* will signal. The hardware handles
this without deadlock because timeline waits are non-blocking barriers.

On backends without native support (CUDA/HIP binary events), the scheduler
holds the waiting operation as PENDING until the signaling operation completes
and the frontier indicates safety, then issues the wait — or elides it
entirely if queue FIFO already guarantees ordering.

### Wait Elision

When a queue's local frontier already dominates an operation's wait frontier,
every dependency is already satisfied. The scheduler can skip the device wait
entirely. This is the zero-cost fast path for sequential, single-queue
workloads: every wait is elided because the queue's own epoch already
implies all prerequisites.

---

## What This Provides

The frontier system is the common substrate shared by CPU, GPU, NPU, NIC,
and NVMe scheduling. It provides:

- **Transitive causal knowledge without round-trips**: A remote scheduler
  receiving a frontier knows the complete dependency history without
  contacting upstream devices.

- **Wait elision via dominance checking**: If the local queue's frontier
  already subsumes the wait frontier, the device wait is skipped entirely.

- **Allocation-free buffer reuse**: Death frontiers enable safe reuse checking
  with O(1) metadata per buffer, not per-operation tracking.

- **Compositional scheduling**: Independent workloads have disjoint frontier
  axes and create zero causal interference, even when sharing hardware. The
  scheduler can interleave them freely.

- **Pipeline submission**: Entire multi-stage, multi-device pipelines can be
  submitted atomically. Hardware FIFOs and semaphore ordering ensure correct
  execution without client-side step-by-step orchestration.

- **Scope-optimized signaling**: Queue-local operations pay zero
  synchronization cost. Cross-device operations use the narrowest scope
  that provides sufficient visibility.

- **Graceful external interop**: Taint tracking handles imported events
  conservatively without permanently degrading performance.

See [Frontiers vs. Binary Events](comparisons.md) for a
detailed comparison with binary event systems, and
[Multi-Model Scheduling Scenarios](scenarios.md) for concrete
scheduling scenarios across different hardware configurations.
