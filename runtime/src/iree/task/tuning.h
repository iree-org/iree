// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_TUNING_H_
#define IREE_TASK_TUNING_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Maximum number of groups a topology can contain. Each group maps to one
// worker thread in an executor. Must be a positive multiple of 64. Override at
// compile time for smaller deployments (e.g., embedded with 2 cores → 64).
// Affinity sets use one 64-bit word per 64 groups, so the cost of a larger
// limit is proportional: 64→1 word, 128→2, 256→4.
#ifndef IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT
#define IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT (256)
#endif  // IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT

// Maximum number of workers that an executor can manage. Derived from the
// topology max group count (one worker per group).
#define IREE_TASK_EXECUTOR_MAX_WORKER_COUNT IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT

// Number of 64-bit words in an affinity set. Derived from the topology limit.
#define IREE_TASK_AFFINITY_SET_WORD_COUNT \
  ((IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT + 63) / 64)

// Number of compute slots for wake_budget > 1 processes. Workers scan these
// slots round-robin looking for cooperative drain work. Processes beyond this
// count overflow to a slist and are promoted into slots as they are released.
// wake_budget == 1 processes use the immediate list and are not counted here.
#define IREE_TASK_EXECUTOR_MAX_COMPUTE_SLOTS (16)

// Number of additional workers each waking worker wakes before starting to
// drain. Controls the branching factor of the wake cascade:
//   FANOUT=1: serial chain — each worker wakes one more (O(N) rounds).
//   FANOUT=2: binary tree — each worker wakes two (O(log N) rounds).
//   FANOUT=K: K-ary tree — fewer rounds but more notification posts per worker.
//
// Higher values reduce rounds but delay each relaying worker's drain start by
// the cost of FANOUT notification posts. On x86_64 (where futex wake is ~5us
// and CAS contention on desired_wake scales poorly), FANOUT=1 measured 20-40%
// faster across all worker counts. On ARM (Apple M2 Ultra), FANOUT=4 was
// faster at high worker counts (16+) due to lower per-notification cost.
// FANOUT=1 is a conservative default that avoids CAS contention.
#define IREE_TASK_WAKE_FANOUT (1)

// Time warm-retained workers spin near a compute process before sleeping on the
// process-local retention epoch. Workers share a process-local deadline so
// late-arriving retainers do not each get a fresh private spin window. This
// keeps short inter-region gaps out of the kernel while bounding wasted CPU on
// long tails. Define to 0 to skip the warm spin window and sleep immediately.
#ifndef IREE_TASK_WARM_WAIT_SPIN_NS
#define IREE_TASK_WARM_WAIT_SPIN_NS (10 * 1000)
#endif  // IREE_TASK_WARM_WAIT_SPIN_NS

// Time explicitly retained warm workers spin when the process has reported
// near-future work that benefits from a wider active worker set. This is
// separate from the generic tail spin above: only drain functions with concrete
// lookahead evidence request this wider handoff window.
#ifndef IREE_TASK_WARM_WAIT_LOOKAHEAD_SPIN_NS
#define IREE_TASK_WARM_WAIT_LOOKAHEAD_SPIN_NS (10 * 1000)
#endif  // IREE_TASK_WARM_WAIT_LOOKAHEAD_SPIN_NS

// Maximum number of remaining active drainers where a no-work warm retainer is
// allowed to enter the shared spin window. Larger tails mean some workers are
// still executing already-claimed work and extra retainers would only herd-spin
// before parking. Keeping the final few drainers warm lets them bridge short
// region-transition gaps without waking the whole worker pool.
#ifndef IREE_TASK_WARM_WAIT_TAIL_SPIN_DRAINER_LIMIT
#define IREE_TASK_WARM_WAIT_TAIL_SPIN_DRAINER_LIMIT (8)
#endif  // IREE_TASK_WARM_WAIT_TAIL_SPIN_DRAINER_LIMIT

// Time warm-retained workers spin after the last tile in a region completes
// and before the next region becomes claimable. The completer publishes this
// shared deadline before running region-transition bookkeeping so workers
// already spinning near the process can survive short handoff gaps. Sleeping
// retainers remain parked until the retention epoch advances.
#ifndef IREE_TASK_WARM_WAIT_TRANSITION_SPIN_NS
#define IREE_TASK_WARM_WAIT_TRANSITION_SPIN_NS (3 * 1000)
#endif  // IREE_TASK_WARM_WAIT_TRANSITION_SPIN_NS

// Maximum time a warm-retained worker sleeps before polling for executor exit.
// Normal work publication wakes sleepers immediately by advancing the retention
// epoch; the timeout only bounds shutdown latency and recovers from missed
// platform wakes.
#ifndef IREE_TASK_WARM_WAIT_SLEEP_NS
#define IREE_TASK_WARM_WAIT_SLEEP_NS (1000 * 1000)
#endif  // IREE_TASK_WARM_WAIT_SLEEP_NS

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_TUNING_H_
