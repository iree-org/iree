# Frontiers vs. Binary Events

This document compares three synchronization approaches — binary events,
timeline semaphores, and timeline semaphores with causal frontiers — across
the scheduling scenarios that arise in heterogeneous, multi-device, and
multi-model systems.

The comparison is not "binary events are bad." For single-device, single-model
workloads, binary events work well and are the industry standard. The argument
is that binary events are **insufficient by design** for the scheduling
problems that arise when multiple models share heterogeneous hardware, and
that compensating for their limitations is **prohibitively expensive** in
both engineering effort and runtime overhead.

See [Causal Dependency Tracking with Vector Clock Frontiers](index.md)
for a full description of the frontier system.

---

## The Three Levels

### Binary Events (CUDA events, HIP events)

A binary event has two states: unsignaled and signaled. One operation records
the event; another operation waits on it. The event carries no information
beyond "has this specific thing happened."

```text
cuEventRecord(event, stream_A);    // marks event on stream A
cuStreamWaitEvent(stream_B, event); // stream B blocks until event fires
```

**Information content**: 1 bit (signaled or not).

### Timeline Semaphores (Vulkan, D3D12 fences, HSA signals)

A timeline semaphore is a monotonically increasing uint64. Operations signal
it to a value; other operations wait for it to reach or exceed a target value.

```text
vkQueueSubmit(..., signal semaphore S to value 5);
vkQueueSubmit(..., wait for S >= 5);
```

**Information content**: 64 bits (current position on one timeline).

### Frontiers (Vector Clocks over Timeline Semaphores)

A frontier is a set of (axis, epoch) pairs attached to each semaphore signal.
When an operation waits on a semaphore, it imports the frontier, gaining
transitive knowledge of all contributing queues.

**Information content**: ~k × 128 bits, where k is the number of axes in the
frontier (typically 1–6 in practice).

---

## Scenario Comparisons

### 1. Sequential single-queue work

**Setup**: Three operations on one GPU, each depending on the previous.

**Binary events**: Work well. Record event after each operation, wait on it
before the next. Or skip the events entirely — in-stream FIFO provides
ordering for free.

**Frontiers**: Also trivial. Wait elision kicks in immediately — the queue's
own epoch already implies all prior work. Zero device primitives needed.

**Verdict**: Both approaches are equivalent. Frontiers add no overhead
(queue-local scope skips device primitive allocation) and no benefit.

### 2. Cross-queue dependency (same device)

**Setup**: GPU has two queues. Queue A produces data, Queue B consumes it.

**Binary events**: Record event on queue A, wait on queue B. One event per
dependency edge. Works.

**Frontiers**: Semaphore carries A's frontier to B. If B later signals another
semaphore, that semaphore's frontier includes A's axis transitively. Works,
and propagates context.

**Verdict**: Binary events work. Frontiers add transitive propagation that
pays off downstream (see scenario 4).

### 3. Fork-join (fan-out, fan-in)

**Setup**: One operation fans out to N parallel branches, which join back.

**Binary events**: N events for the fan-out (one per branch), N waits at
the join point. The join operation must enumerate all N events. If a
downstream operation depends on the join result, it waits on one event
from the join — but that event carries no information about what the join
depended on.

**Frontiers**: Each branch signals its own semaphore. The join operation
waits on all of them and its frontier becomes the component-wise max of all
branches. A single semaphore signal from the join carries the complete
merged history. Downstream consumers need one wait, not N.

**Verdict**: Binary events scale linearly in fan-in width at the join point
and lose transitive information. Frontiers capture the merge in one vector.

### 4. Transitive cross-device dependencies

**Setup**: GPU_A produces data. GPU_B transforms it. GPU_C consumes the
result. GPU_C depends on GPU_A's work transitively through GPU_B.

**Binary events**: GPU_A records event_1 on stream_A. GPU_B waits on
event_1, does work, records event_2 on stream_B. GPU_C waits on event_2.
GPU_C knows GPU_B is done, but has no information about GPU_A — the event
carries no transitive context. If GPU_C also needs a direct dependency on
GPU_A (for buffer reuse, scheduling decisions, or additional work), it
must be given event_1 explicitly. The application must propagate the
dependency graph manually.

**Frontiers**: GPU_B's signal to S_2 carries frontier {A: epoch_a, B: epoch_b}.
GPU_C waits on S_2 and imports both axes. GPU_C now knows about GPU_A's
completion without any direct interaction. The transitive dependency is
captured in the data structure, not the application logic.

**Verdict**: Binary events require the application to manually propagate
dependency edges through the entire graph. Frontiers propagate transitivity
automatically through the merge algebra.

### 5. Remote pipeline scheduling (submit-before-wait)

**Setup**: A client submits work to GPU_0 and GPU_1, where GPU_1's work
depends on GPU_0. The client then pipelines additional work on GPU_1 that
depends on the first GPU_1 task. GPU_1 may be on a different machine.

**Binary events**: The client creates event_A (GPU_0 completion) and
event_B (first GPU_1 task completion). The second GPU_1 task waits on
event_B. This works for a linear chain. But GPU_1's scheduler, upon
receiving the second task's wait on event_B, knows nothing about what
event_B depends on. It cannot determine whether the dependency chain is
safe to pipeline on its FIFO hardware queue without querying the client
or examining the full event graph.

When dependencies fan in from multiple sources — GPU_0 and GPU_2 both
feed GPU_1, and GPU_1's second task depends on the result of the join —
event_B tells GPU_1's scheduler nothing about GPU_0 or GPU_2. GPU_1
cannot make local scheduling decisions; it must either block until the
event fires or have been given the complete dependency topology in advance.

**Frontiers**: GPU_1's scheduler receives the second task with a wait on
semaphore S >= 2. The frontier at S=2 contains {GPU_0_axis: epoch_x,
GPU_2_axis: epoch_y, GPU_1_axis: epoch_z}. GPU_1's scheduler can locally
verify that all transitive dependencies are satisfiable by its hardware
FIFO and pipeline the work immediately. No round-trip to the client. No
global topology knowledge required.

**Verdict**: Binary events cannot support local pipeline scheduling
decisions at remote devices without transmitting the full dependency graph.
Frontiers carry the dependency summary inline.

### 6. Buffer reuse safety

**Setup**: A buffer is allocated, used by many operations, then deallocated.
Later, a different queue wants to reuse the memory.

**Binary events**: The allocator must track which events correspond to
operations that read or wrote the buffer. On reuse, it waits on all of
them. For a weight tensor read by 100 decoder layers, that is 100 events
to track and query. Each event requires a host query (cuEventQuery) or a
device wait (cuStreamWaitEvent). The tracking is per-use, and the number
of events scales with the number of operations that touch the buffer.

The alternative is to use a single "last use" event, but determining the
last use requires application-level analysis that the scheduling layer does
not have for arbitrary workloads.

**Frontiers**: The deallocating queue records its frontier as the buffer's
death frontier. This single frontier captures "everything that happened
before the dealloc" — which by construction includes all operations that
used the buffer (the dealloc post-dominates all uses in the compiled IR).
The reuse check is one dominance comparison, O(k) where k is the frontier
size, regardless of how many operations touched the buffer.

**Verdict**: Binary events require O(operations) tracking per buffer
lifetime. Frontiers require O(1) tracking per buffer lifecycle.

### 7. Independent workload isolation

**Setup**: Two unrelated models share the same GPU. They should have zero
scheduling interference.

**Binary events**: If the two models use separate CUDA contexts or separate
streams with no shared events, they are isolated. But if they share a
memory pool (which they should, for utilization), the allocator's event
tracking can accidentally create dependencies between them — model A's
event appears in the allocator's tracking for a buffer that model B wants
to reuse.

**Frontiers**: Independent workloads have disjoint frontier axes. Their
frontiers share no components. Dominance checks between them produce
immediate results: a buffer freed by model A with death frontier
{A_axis: 5} is immediately reusable by model B because model B's wait
frontier has no A_axis component, and a frontier with no overlapping axes
trivially dominates the relevant subset (which is empty). Hardware
scheduling can interleave their work freely; the causal structure captures
zero interference.

**Verdict**: Binary events can create accidental cross-model coupling through
shared infrastructure (allocators, event pools). Frontiers maintain structural
isolation through disjoint axis namespaces.

### 8. Speculative execution tracking

**Setup**: A draft model generates speculative tokens. A verifier model
checks them. The draft runs ahead without waiting for verification.

**Binary events**: The draft's event and the verifier's event are separate.
Nothing in the event system distinguishes "speculative" from "verified"
work. The application must maintain its own metadata about what is
speculative. If speculation fails and work must be cancelled, the
application must track which events correspond to speculative work.

**Frontiers**: The draft model's frontier has no verify-axis component.
The verifier's frontier includes both draft and verify axes. The presence
or absence of the verify axis in a frontier structurally encodes whether
work is speculative or verified. Cancellation is precise: invalidate
work whose frontier lacks the verify axis beyond the rejection point.

**Verdict**: Binary events encode no structural information about execution
status. Frontiers make speculation/verification status visible in the
synchronization substrate.

---

## Workarounds in the Wild

### vLLM's IPC Shared Memory Pool

vLLM added inter-process shared memory for multi-model serving, allowing
multiple vLLM instances to share GPU memory. The implementation uses POSIX
shared memory segments with manual reference counting and IPC semaphores
for synchronization.

The approach addresses memory utilization but not scheduling: each vLLM
instance still runs its own scheduler with its own event tracking, and
cross-model dependencies (one model's output feeding another) require
serialization through the host. The IPC overhead for each cross-model
buffer transfer is on the order of microseconds (shared memory + semaphore
round-trip), and the lack of transitive dependency information means the
receiving model cannot pipeline further work until the transfer is
confirmed complete.

### CUDA Graphs

CUDA graphs pre-record a sequence of operations (kernel launches, memory
copies, event records/waits) into a reusable graph object. The graph
captures the dependency structure using binary events internally, and the
runtime can optimize event placement and eliminate launch overhead.

Limitations for multi-model scheduling:

- **Total topology knowledge at creation time**: The graph must contain the
  complete operation DAG at graph creation. For dynamic workloads (MoE
  expert selection, speculative decoding with variable acceptance rates,
  variable-length sequences), the graph must be rebuilt or the dynamism
  must be handled outside the graph.

- **Single-device scope**: CUDA graphs operate within a single CUDA context.
  Cross-device dependencies require graph-external synchronization, which
  reintroduces the binary event limitations for the cross-device portion.

- **No incremental composition**: You cannot take two independently-created
  graphs and compose them with a dependency edge added later. Graphs must
  be constructed with full knowledge of the final topology. For multi-model
  workloads where each model's graph is independent and cross-model
  dependencies arise dynamically from request routing, this requires
  rebuilding the combined graph for each new dependency pattern.

- **Binary events internally**: Within the graph, event-based synchronization
  has the same transitive-knowledge limitations. The graph optimizer can
  elide redundant events, but this optimization requires the complete graph
  at optimization time — it cannot be done incrementally as new work arrives.

### MPS / MIG (Multi-Process Service / Multi-Instance GPU)

NVIDIA's MPS allows multiple processes to share a GPU with reduced context
switching overhead. MIG partitions a GPU into isolated instances. Both
address the "multiple models on one GPU" problem at the process/hardware
isolation level.

Neither addresses scheduling across the partition boundary: each process
or MIG instance has its own event space, and cross-boundary dependencies
still require host-mediated synchronization. The frontier system, by
contrast, operates within a single process with shared queue and semaphore
namespaces, making cross-model scheduling a normal semaphore operation
rather than an IPC boundary crossing.

---

## Cost Analysis

### Binary events: where the costs compound

For a system running N models with M queues across D devices:

- **Event creation/destruction**: Each cross-queue dependency requires an
  event. In a multi-model system, event lifecycles must be managed across
  models. Event pools help, but pool sizing is global and model-dependent.

- **Dependency propagation**: Each consumer must be given the explicit set
  of events it depends on. For transitive dependencies (consumer depends
  on producer via intermediary), the intermediary must forward events or
  the system must maintain a global dependency graph.

- **Buffer reuse tracking**: O(operations) events per buffer. With hundreds
  of operations per inference step and thousands of buffers in the pool,
  the event tracking metadata grows rapidly.

- **Scheduling queries**: Determining "is this operation ready?" requires
  querying events — either host-side (cuEventQuery per event, microsecond
  per query) or by recording a device wait (cuStreamWaitEvent, which
  consumes device scheduler resources even if the event is already
  signaled).

- **No information at rest**: An unsignaled event tells the scheduler
  nothing about progress or dependencies. The scheduler must either
  poll or maintain its own progress tracking infrastructure.

### Frontiers: where the costs are paid

- **Frontier storage**: ~200 bytes per frontier. Every semaphore stores one
  frontier (the latest signal's context). Every buffer stores one death
  frontier. This is constant-size, not proportional to workload.

- **Merge cost**: O(k) per merge, where k is the frontier entry count
  (typically 1–6). This is tens of nanoseconds — comparable to a single
  atomic load.

- **Dominance check cost**: O(k) per check. This replaces per-event queries
  with a single vector comparison.

- **Signal cost**: One frontier copy per signal (in addition to the timeline
  value update). The frontier is written once at signal time and read at
  wait time — no per-query cost.

- **Overflow cost**: When a frontier overflows its capacity, it evicts an
  entry and marks itself tainted. The taint disables some optimizations
  (wait elision) for the affected values but does not affect correctness.
  In practice, collective channels keep frontiers compact.

The overall tradeoff: frontiers add a constant per-signal cost (~200 bytes
written, ~50ns for the merge) and eliminate per-event-query, per-buffer-use,
and per-dependency-edge costs that scale with workload size.

---

## Summary

| Property | Binary Events | Timeline Semaphores | Frontiers |
|----------|---------------|---------------------|-----------|
| Information per signal | 1 bit | 64 bits | ~k × 128 bits |
| Transitive propagation | Manual | None | Automatic |
| Buffer reuse tracking | O(operations) per buffer | O(operations) per buffer | O(1) per buffer |
| Remote pipeline scheduling | Requires full topology | Requires full topology | Local decision from frontier |
| Independent workload isolation | Accidental coupling via pools | Better, but no structural guarantee | Structural isolation via disjoint axes |
| Wait elision | Not possible (event or no event) | Possible for single-axis chains | Possible across all axes |
| Collective compression | N/A | N/A | N devices → 1 frontier entry |
| Dynamic composition | Requires graph rebuild | Works | Works |
| Speculation tracking | Application metadata | Application metadata | Structural (axis presence/absence) |

Binary events are the right tool for single-device, single-model, static-
topology workloads — the domain they were designed for. Timeline semaphores
extend the model to support pipelining and >= semantics. Frontiers extend it
further to support the transitive, compositional, multi-device scheduling
that heterogeneous multi-model systems require.
