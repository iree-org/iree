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

// Maximum number of workers that an executor can manage.
// A 64 worker hard limit is based on us using uint64_t as a bitmask to select
// workers. It's easy to go smaller (just use fewer bits) if it's known that
// only <64 will ever be used (such as for devices with 2 cores).
#define IREE_TASK_EXECUTOR_MAX_WORKER_COUNT (64)

// Maximum number of concurrent budget>1 (compute) processes that the executor
// can schedule simultaneously. Workers scan these slots round-robin looking
// for cooperative drain work. Budget-1 processes use the immediate list
// instead and are not counted against this limit.
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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_TUNING_H_
