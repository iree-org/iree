# IREE Task System: Process-Based Cooperative Scheduler

## Overview

The IREE task system is a cooperative, multi-worker scheduler built around a
single abstraction: the **process**. A process is any unit of work that can be
drained incrementally by one or more workers. Workers call `drain()` on
processes, do bounded work, and return — freeing them to handle other work
(I/O completions, higher-priority processes, newly-arrived submissions) before
coming back for more. There are no task types, no DAGs, no coordinator thread,
and no per-operation allocation at execution time.

The drain/return model is the key design choice. Every scheduling decision
happens at the boundary between drain() calls — a natural yield point that
costs one atomic load (~1ns) to check for higher-priority work. This gives the
system fine-grained interleaving of compute and I/O without preemption,
context switches, or kernel involvement.

### Design goals

- **Dispatch-to-dispatch latency < 10μs.** The dominant cost should be the
  kernel itself, not the scheduling machinery around it. No per-command task
  allocation, no DAG construction, no coordinator round-trip.

- **Zero-overhead pipelining.** Multiple command buffers in flight
  simultaneously, with I/O completions (semaphore signals, buffer readiness)
  processed between drain() calls. The system pipeline stays full without
  explicit overlap management.

- **Unified execution model.** Compute (tiled dispatch), I/O (buffer
  transfer), host callbacks, VM invocations, and queue management all use the
  same process interface. One scheduling policy, one worker loop, one wake
  mechanism.

- **Direct wake, zero hop.** When a GPU completion, semaphore signal, or I/O
  event makes a process runnable, the signaling thread directly activates it
  (atomic decrement + list push). No coordinator lock, no ready-queue
  round-trip, no batched wake — the process is scannable by workers on the
  next loop iteration.

### What this replaces

The previous task system had six task types (NOP, CALL, BARRIER, FENCE,
DISPATCH, DISPATCH_SHARD), a coordinator mutex, per-command task allocation
during recording, shard fan-out at dispatch issue, and a centralized
classify-and-route loop. The process model replaces all of these with one
type, one worker loop, and one activation mechanism.

## Process Model

### The process type

A process has:

- A **drain function**: called by workers to do bounded work. Multiple workers
  may call concurrently. Returns quickly (one region pass for compute, one
  operation for queue management).

- A **suspend count**: atomic counter. Process is runnable when count is zero.
  External events (semaphore signals, upstream process completion) decrement
  the count. The thread that drives it to zero activates the process.

- A **wake budget**: how many workers the executor should try to wake for this
  process. Updated dynamically by the drain function (e.g., a block processor
  updates the budget at each region transition based on available tiles). This
  is a wake fan-out hint, not an admission limit on active drainers.

- A **dependent list**: processes to activate when this process completes.
  Completion decrements each dependent's suspend count. This replaces BARRIER
  tasks — dependency fan-out is an atomic decrement per dependent, inline in
  the completing worker.

- A **cancellation flag**: atomic. When set, drain() returns completed with an
  error. Wake events on cancelled processes are no-ops. Cancellation
  propagates to dependents (they see the error in their scope/group).

- An **error status**: first error wins via CAS. Checked on drain() entry.
  When a process errors, dependents are still activated (their suspend count
  decrements normally) but they see the error and bail.

### Process lifecycle

```
                suspend_count > 0
 Created ──────────────────────────► Suspended
    │                                    │
    │ suspend_count == 0                 │ last decrement drives to 0
    │                                    │
    ▼                                    ▼
 Activated ◄─────────────────────── Activated
    │                                    │
    │ pushed to run list                 │ pushed to run list
    │                                    │
    ▼                                    ▼
 Runnable ◄──────────────────────── Runnable
    │                                    │
    │ worker calls drain()               │ worker calls drain()
    │                                    │
    ▼                                    ▼
 Draining ──── drain() returns ────► Runnable  (more work available)
    │
    │ drain() returns completed
    │
    ▼
 Completing ──── resolve dependents ──► (dependents activated)
    │
    │ cleanup
    │
    ▼
 Freed
```

A **sleeping** process is one that returned `did_work=false` from drain().
The executor marks it as deprioritized in the run list — workers skip it
during normal scan. When an external event wakes it (semaphore signal changes
its input state), the process is reprioritized. This avoids polling sleeping
processes on every scan revolution.

### Cancellation and teardown

Cancellation sets the process's cancel flag. The next drain() call sees it and
returns completed with `IREE_STATUS_CANCELLED`. If the process is sleeping,
cancellation wakes it (to trigger the drain-and-exit path).

Cancellation propagates through dependents: when a cancelled process
"completes" (with error), its dependents activate normally but inherit the
error. Their drain() calls bail immediately, cascading the cancellation
through the dependency chain.

For device teardown: a HAL device has a small number of well-defined processes
(2-3 per queue). Teardown cancels these processes and waits for them to drain
to completion. Child processes (block processors issued by the queue) are
cancelled transitively. The teardown is bounded — it waits for at most one
drain() call per process per worker, which is sub-millisecond.

## Worker Model

### Worker states

Each worker thread has three states:

| State | CPU | Wake latency | Description |
|-------|-----|-------------|-------------|
| **Draining** | Full | N/A | Executing a drain() call |
| **Spinning** | ~Full | < 100ns | Scanning for work, no drain() in progress |
| **Parked** | Zero | 1-5μs | Sleeping on notification (futex) |

State transitions:

- **Draining → Spinning**: drain() returns, worker scans for next process.
- **Spinning → Draining**: scan finds a runnable process, calls drain().
- **Spinning → Parked**: no work found after spin timeout.
- **Parked → Spinning**: notification posted (new process activated, budget
  increase).

The spin timeout is adaptive: when aggregate worker budget is high (lots of
compute work available), spinning workers wait longer before parking. When
budget is low (few processes, low parallelism), they park quickly to avoid
wasting CPU.

### Worker loop

```
while (!exit_requested) {
    // 1. Check for immediate work (budget-1, FIFO).
    process = try_pop(executor.immediate_list);
    if (process) {
        process.drain(worker_index, worker_state, &result);
        process_complete(process);
        continue;
    }

    // 2. Scan compute processes (budget-N, array-based).
    process = scan_compute(executor.compute_slots, &my_position);
    if (process && is_runnable(process)) {
        process.drain(worker_index, worker_state, &result);
        if (result.completed) process_complete(process);
        continue;
    }

    // 3. No work — transition toward parked.
    if (should_park()) {
        park_on_notification();
    }
}
```

**Why two lists?** The immediate list handles short-lived, single-worker
operations (queue issue, host callback, retire/signal). The compute list
handles long-lived, multi-worker operations (block processor, streaming
queue). Separating them means the compute scan — the hot path, 99% of all
cycles — never touches immediate-work process metadata. A failed `try_pop`
on an empty immediate list costs one atomic load (~1ns).

### Per-worker process state

Workers maintain per-process state across drain() calls (e.g., the block
processor's `worker_state_t` with cached block_sequence). This state is
indexed by process slot and owned entirely by the worker — no sharing, no
atomics. The executor allocates this state per-worker when a process is
activated, sized by `process.worker_state_size`.

## Scheduling

### Wake budget and distribution

Each process declares a wake budget — how many concurrent drainers it benefits
from waking. The executor maintains an aggregate budget (sum across all
runnable processes) and uses it for worker management:

- **Aggregate budget > active workers**: wake parked workers to meet demand.
- **Aggregate budget < active workers**: excess workers park after finding no
  productive work.
- **Budget changes**: processes update their budget dynamically (e.g., block
  processor transitions from a 20-dispatch region to a 1-dispatch region).
  Workers see the change on their next scan.

Budget is a hint, not a hard limit. Any worker can drain any process.
A process with budget=4 will work correctly with 1 worker or 16 — the
budget only affects how many workers the executor *tries* to keep active.

### Scan policy

The compute list is an array of process slots. Workers scan by index, starting
at different positions (worker_id modulo slot count) to spread coverage.
The scan is read-only — workers load slot pointers and check runnable flags.
No contention between scanners.

The scan order determines scheduling policy. Options:

- **Round-robin** (default): scan all slots in index order. Simple, fair,
  predictable. Each process gets visited proportional to the number of
  active workers.

- **Priority-ordered**: maintain an indirection table that orders slots by
  priority. Workers scan through the indirection table instead of raw slots.
  The indirection table is updated when processes are activated/deactivated,
  and is immutable between updates (read-only for scanners). High-priority
  processes get visited first.

- **Budget-weighted**: workers visit processes proportional to their worker
  budget. A process with budget=8 gets visited ~4x as often as one with
  budget=2. Implemented via the indirection table (repeat high-budget entries).

The scan function is a small, hot piece of code in the worker loop — easy to
swap, easy to benchmark, easy to tune per-deployment. The process type and
drain interface are agnostic to scan policy.

## Wake and Notification

### Direct wake

When an external event makes a process runnable, the signaling thread
activates it directly:

1. Decrement the process's suspend count (fetch_sub, acq_rel).
2. If the old value was 1 (transition to zero): push to the appropriate run
   list (immediate or compute) and wake workers.
3. If the old value was > 1: another dependency is still pending. Do nothing.

This is **zero-hop activation** — no coordinator, no ready-queue, no lock.
The signaling thread (proactor, semaphore callback, completing worker) does
the activation inline. The latency from "event occurs" to "process is
scannable" is one atomic operation + one slist push.

### Wake trees

Waking many workers from a single event (e.g., 0 → 32 workers when a large
command buffer becomes ready after all workers were idle) is expensive if done
serially: one futex wake per worker, ~5μs each, ~160μs total for 32 workers.

Wake trees distribute the wake cost:

1. The activating thread sets `desired_wake = wake_budget` and wakes one
   parked worker.
2. Each waking worker claims a share of `desired_wake` (fetch_sub) and wakes
   that many additional workers before starting to drain.
3. The tree fills in ~log2(N) rounds. With fanout=2 and 32 workers: 5 rounds,
   ~25μs total.

Workers interleave waking with draining: a worker that wakes two others
immediately starts scanning for work. It doesn't wait for the tree to
complete. Useful work begins on the first wake, not the last.

## Process Types

### Compute process: block processor

The primary workload. A block processor process wraps a recorded command
buffer (block ISA) and executes it cooperatively:

- **drain()**: claims tiles from the current region via epoch-tagged CAS,
  executes kernels, participates in completer election. Returns after one
  region pass.
- **wake_budget**: set from the current region's tile count. Updated at each
  region transition.
- **worker_state**: cached block_sequence for detecting block transitions.
- **Activation**: pushed to compute list when the queue process issues it.
- **Completion**: signals the submission's signal semaphores, frees the
  processor context, advances the frontier.

This is where 99% of cycles are spent. The drain() hot path is: atomic load
(block_sequence check) → atomic load (active_region_index) → atomic load
(region_epoch) → CAS loop on tile_index → kernel call. No allocation, no
indirection beyond the drain function pointer.

### Immediate process: queue management

A HAL queue is a persistent process that manages submissions:

- **Internal state**: a list of submissions in three states (waiting, ready,
  issued). Semaphore timepoints move submissions from waiting → ready.
  drain() moves submissions from ready → issued.
- **drain()**: pops a ready submission, allocates a block processor context,
  wraps it as a compute process, pushes it to the compute list. Returns after
  issuing one submission.
- **wake_budget**: 1. Queue management is sequential.
- **Sleeping**: when no submissions are ready, the queue process sleeps.
  Woken by: semaphore signal satisfying a wait, or new submission arriving.
- **Lifetime**: created at device startup, lives for the device's lifetime.
  Teardown cancels the queue process and all its issued child processes.

The queue process replaces the current `wait_cmd → issue_cmd → retire_cmd`
CALL task chain. Three task allocations per submission → zero (queue process
is pre-allocated, block processor context is arena-allocated).

### Queue-inline operations: alloca, dealloca, host calls

Not all queue operations warrant a separate process. Transient allocations
(`queue_alloca` is typically a sub-allocation from a pool, `queue_dealloca`
releases back to it) and host callbacks are small, synchronous operations
that execute inline in the queue process's drain().

Each pending queue operation (command buffer submission, alloca, dealloca,
host call) has its own wait count — one per unsatisfied semaphore
dependency. Semaphore timepoint callbacks decrement the wait count directly.
When the last wait resolves (count driven to zero), the callback pushes the
operation onto the queue's **ready list** (lock-free slist) and wakes the
queue process. No scanning — the ready list contains only operations that
are immediately executable.

The queue process's drain() pops from the ready list and handles each
operation by type:

- **Command buffer submission**: allocate block processor context, create
  compute process, push to compute list.
- **Alloca**: sub-allocate from device pool (~10ns), signal semaphores.
- **Dealloca**: release back to pool (~10ns), signal semaphores.
- **Host callback**: call user function, signal semaphores.

This is the same direct-wake pattern used everywhere else: the signaling
thread (semaphore callback, proactor, completing worker) does the readiness
transition inline. The queue process never scans — it wakes, pops ready
work, handles it, and sleeps again if the ready list is empty. Execution
order is determined entirely by semaphore resolution timing, not submission
order.

### Coroutine process: VM execution

A VM invocation context that runs model execution graphs. The VM executes
bytecode until it hits an async wait (e.g., waiting for HAL operations to
complete), then the process sleeps. When the waited-on semaphore signals, the
process wakes and resumes execution.

- **drain()**: resumes VM execution from the current bytecode position. Runs
  until the VM yields (async wait) or completes.
- **Sleeping**: when the VM hits an async wait, drain() returns with
  `sleeping=true`. The process is deprioritized until the semaphore
  signal wakes it.
- **Activation**: the semaphore timepoint callback decrements suspend count,
  promoting the process back to runnable. The latency from "semaphore signal"
  to "VM resumes" is one scan revolution — sub-microsecond if a worker is
  spinning, 1-5μs if all workers were parked.

This turns the proactor's "offload work from the CQ thread" problem into a
scheduling problem: the proactor signals a semaphore, the semaphore activates
a VM process, a worker picks it up on its next scan. No dedicated offload
thread, no thread-hop latency, no oversubscription. The CQ thread stays
focused on completions; the worker pool handles everything else.

### Streaming process: direct queue operations

An alternative to recording full command buffers: individual dispatches,
fills, and copies submitted directly to a queue. The queue accumulates
operations in a ringbuffer, and workers drain them directly.

- **drain()**: pops operations from the ringbuffer, executes them.
- **wake_budget**: depends on operation mix. Single fills/copies → budget=1.
  Multi-tile dispatches → budget=N.
- **Useful when**: many small independent operations where command buffer
  recording overhead isn't worthwhile.

This is a future extension — the block processor handles the general case
(recorded command buffers with barriers and multi-dispatch regions). The
streaming process handles the simple case (independent operations, no
barriers).

## Concurrency Design

### Cache line discipline

Process metadata is laid out for minimal false sharing:

```
Cache line 0 (immutable after creation):
    drain function pointer, worker_state_size, scope pointer,
    dependent list, dependent count, ...

Cache line 1 (activation/completion — written by signaling threads):
    suspend_count, cancelled flag, state

Cache line 2 (scheduling — written by drain function):
    wake_budget, sleeping flag, last_did_work

Cache line 3+ (process-specific mutable state):
    Block processor: active_region_index, region_epoch, remaining_tiles, ...
```

Workers scanning the compute list touch cache line 0 (read function pointer)
and cache line 1 (read state/runnable). These are read-shared in L1 across
workers — no coherence traffic during scan.

Workers draining touch cache line 3+ (process-specific state). For the block
processor, tile_indices are individually cache-line-padded so workers claiming
tiles from different dispatches have zero contention.

### Compute slot array

The compute list is a fixed-size array of pointers (`process_t*`). The array
size is bounded (e.g., 64 slots — matching the maximum number of concurrent
command buffer executions). Workers scan by loading pointers, which are
8 bytes each — 8 slots fit in one cache line. Scanning 64 slots touches
8 cache lines, all read-shared.

Activation (writing a slot) uses CAS: `NULL → process_ptr`. Deactivation
uses CAS: `process_ptr → NULL`. Scanners never write to the array — zero
contention between scanners and rare contention between scanner and
activator (one CAS per process lifetime).

### Immediate list

A lock-free MPSC slist (multiple producers — any thread can push;
single consumer — one worker pops per item). Push is `slist_push` (CAS on
head). Pop is `slist_pop` (CAS on head). Typical size: 0-2 items. The
list is empty 99.9% of the time; the failed pop costs one atomic load.

### Suspension and wake ordering

A process's suspend count is the synchronization mechanism for all
dependencies. The ordering guarantee: all writes performed by the
signaling thread before the `fetch_sub` are visible to the draining
worker after a successful scan (the scan loads the process pointer with
acquire, which pairs with the release in the activating push).

The wake sequence for a process with two dependencies:

```
Thread A (upstream process completes):
    [writes to shared state]
    fetch_sub(&process.suspend_count, 1, release)  → old=2, new=1
    // Old was 2, not driving to 0 — do nothing.

Thread B (semaphore signals):
    [writes to shared state]
    fetch_sub(&process.suspend_count, 1, acq_rel)  → old=1, new=0
    // Drove to 0 — this thread activates the process.
    // The acquire on this fetch_sub sees Thread A's release.
    slist_push(&run_list, process)  // release semantics on push
    wake_workers()

Worker C (scans and finds process):
    process = load(&slot, acquire)  // sees the push's release
    process.drain(...)              // sees all state from A and B
```

## Performance Characteristics

### Latency

| Operation | Latency | Mechanism |
|-----------|---------|-----------|
| Drain boundary (check for other work) | ~1ns | Atomic load of immediate list head |
| Process activation (suspend→runnable) | ~10ns | fetch_sub + slist_push |
| Worker wake (parked→spinning) | 1-5μs | futex/notification |
| Wake tree (0→N workers) | ~5μs × log2(N) | Distributed wake |
| Scan revolution (64 compute slots) | ~40ns | 8 cache line reads |
| Command buffer issue (queue→compute) | ~200ns | Arena alloc + context init |

### Throughput

Workers spend 99%+ of their time in drain() calls (kernel execution). The
scheduling overhead per drain() call is:
- One `try_pop` on immediate list (failed): ~1ns
- One slot load from compute array: ~5ns
- One runnable check (state load): ~5ns
- Drain function pointer call: ~2ns
- Total: ~13ns per drain() call

For a drain() call that executes a single tile of a 100μs kernel, the
scheduling overhead is 0.01%. For a 1μs kernel, it's 1.3%.

### Comparison with previous system

| Metric | Previous (task DAG) | Process model |
|--------|-------------------|---------------|
| Per-command allocation at recording | 1 task per command | 0 (builder writes .text) |
| Per-dispatch allocation at issue | 1 dispatch + N shards | 0 (processor context from arena) |
| Coordinator lock per submission | 1 acquire/release | 0 (no coordinator) |
| Dispatch-to-dispatch latency | ~300μs | Target: < 10μs |
| Semaphore signal → dependent start | 3+ hops (coordinator) | 1 hop (direct wake) |
| Worker wake mechanism | Serial (N futex calls) | Tree (log2(N) rounds) |

## Example Flows

### Simple command buffer execution

```
1. User records command buffer (builder writes .text).
2. User calls queue_execute(wait=[], cmd_buf, signal=[S@100]).
3. Queue process drain():
   a. No waits — submission goes directly to ready state.
   b. Allocate block processor context from arena.
   c. Wrap as compute process (suspend_count=0, budget=8).
   d. Push to compute list.
   e. Wake workers (wake tree, desired_wake=8).
4. Workers scan compute list, find block processor process.
5. Workers call drain() concurrently:
   a. Epoch-tagged CAS claims tiles from current region.
   b. Execute kernels.
   c. Completer advances region, updates wake_budget.
   d. Repeat until RETURN reached.
6. Last worker's drain() returns completed=true.
7. process_complete():
   a. Signal semaphore S to value 100.
   b. Free arena (processor context + submission state).
   c. Remove from compute list.
```

### Pipelined execution with semaphore dependency

```
1. User submits: CB1 (signal S@1) then CB2 (wait S@1, signal S@2).

2. Queue process drain() — first call:
   a. CB1 has no waits → issue as compute process P1.
   b. CB2 waits on S@1 → register timepoint on S.
      Submission stays in waiting state.

3. Workers drain P1. Meanwhile:
   a. I/O completion arrives on proactor thread.
   b. Proactor signals a different semaphore → activates unrelated process.
   c. A worker handles the activation between drain() calls (sub-μs).

4. P1 completes → process_complete() signals S to value 1.

5. Semaphore S dispatches timepoints:
   a. CB2's timepoint fires → decrements CB2 submission wait count.
   b. Wait count reaches 0 → submission moves to ready.
   c. Wake queue process.

6. Queue process drain() — second call:
   a. CB2 is ready → issue as compute process P2.
   b. Push P2 to compute list, wake workers.

7. Workers drain P2. Latency from "P1 done" to "first P2 tile":
   ~200ns (signal + timepoint + queue drain + first scan).
```

### VM coroutine with async HAL operations

```
1. VM invocation starts as a coroutine process (on immediate list).
2. Worker drains: VM executes bytecode, submits HAL operations.
3. VM hits async wait (waiting for HAL results) → process sleeps.
   Worker moves on to other work.
4. HAL operations execute (block processor processes on compute list).
5. Last HAL operation completes → signals semaphore.
6. Semaphore timepoint decrements VM process's suspend count.
7. Suspend count hits 0 → VM process activated (pushed to immediate list).
8. Next worker scan: pops VM process, calls drain().
9. VM resumes execution from where it left off.
   Latency from "HAL done" to "VM resumes": < 5μs.
```
