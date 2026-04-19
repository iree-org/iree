// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_STATISTICS_INTERNAL_H_
#define IREE_HAL_STATISTICS_INTERNAL_H_

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/statistics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// HAL Statistics Compile-Time Gating
//===----------------------------------------------------------------------===//

// HAL statistics producers must keep disabled accounting out of hot paths.
// Backend code should use these macros instead of local preprocessor branches
// so the compile-time switch is centralized here.
#if IREE_HAL_STATISTICS_ENABLE

// Declares a statistics-only field in a runtime struct.
#define IREE_HAL_STATISTICS_FIELD(...) __VA_ARGS__

// Emits statistics-only statements.
#define IREE_HAL_STATISTICS(expr) expr

// Emits statistics-only statements guarded by a runtime enable check.
#define IREE_HAL_STATISTICS_IF_ENABLED(enabled_expr, expr) \
  do {                                                     \
    if (iree_hal_statistics_is_enabled((enabled_expr))) {  \
      expr                                                 \
    }                                                      \
  } while (0)

// Returns true when runtime statistics accounting is enabled for a producer.
//
// The producer owns the meaning of |enabled_bits|. Common callsites should keep
// it in already-hot queue/device state and avoid additional loads, locks, or
// scans before this check.
static inline bool iree_hal_statistics_is_enabled(uint32_t enabled_bits) {
  return enabled_bits != 0;
}

#else

#define IREE_HAL_STATISTICS_FIELD(...)
#define IREE_HAL_STATISTICS(expr)
#define IREE_HAL_STATISTICS_IF_ENABLED(enabled_expr, expr) \
  do {                                                     \
  } while (0)

// Returns true when runtime statistics accounting is enabled for a producer.
//
// The producer owns the meaning of |enabled_bits|. Common callsites should keep
// it in already-hot queue/device state and avoid additional loads, locks, or
// scans before this check.
static inline bool iree_hal_statistics_is_enabled(uint32_t enabled_bits) {
  (void)enabled_bits;
  return false;
}

#endif  // IREE_HAL_STATISTICS_ENABLE

//===----------------------------------------------------------------------===//
// Aggregate Helpers
//===----------------------------------------------------------------------===//

// Resets aggregate operation counts to zero.
static inline void iree_hal_statistics_operation_counts_reset(
    iree_hal_statistics_operation_counts_t* counts) {
  memset(counts, 0, sizeof(*counts));
}

// Adds |delta| into |target|.
static inline void iree_hal_statistics_operation_counts_merge(
    iree_hal_statistics_operation_counts_t* target,
    const iree_hal_statistics_operation_counts_t* delta) {
  target->submitted_count += delta->submitted_count;
  target->completed_count += delta->completed_count;
  target->failed_count += delta->failed_count;
  target->cancelled_count += delta->cancelled_count;
}

// Resets a timing distribution to the no-sample state.
static inline void iree_hal_statistics_timing_ns_reset(
    iree_hal_statistics_timing_ns_t* timing) {
  memset(timing, 0, sizeof(*timing));
}

// Records one duration sample in |timing|.
//
// Durations are independent samples. Their sum may double-count overlapping
// work; callers must use a separate span/latency sample when they want elapsed
// wall time across concurrent children.
static inline void iree_hal_statistics_timing_ns_record(
    iree_hal_statistics_timing_ns_t* timing, uint64_t duration_ns) {
  if (timing->sample_count == 0) {
    timing->minimum_duration_ns = duration_ns;
    timing->maximum_duration_ns = duration_ns;
  } else if (duration_ns < timing->minimum_duration_ns) {
    timing->minimum_duration_ns = duration_ns;
  } else if (duration_ns > timing->maximum_duration_ns) {
    timing->maximum_duration_ns = duration_ns;
  }
  timing->sample_count += 1;
  timing->total_duration_ns += duration_ns;
  timing->last_duration_ns = duration_ns;
}

// Adds |delta| into |target|.
//
// The merged last-duration value is taken from |delta| when it contains
// samples, as |delta| is expected to be newer at producer merge boundaries.
static inline void iree_hal_statistics_timing_ns_merge(
    iree_hal_statistics_timing_ns_t* target,
    const iree_hal_statistics_timing_ns_t* delta) {
  if (delta->sample_count == 0) return;
  if (target->sample_count == 0) {
    *target = *delta;
    return;
  }
  target->sample_count += delta->sample_count;
  target->total_duration_ns += delta->total_duration_ns;
  if (delta->minimum_duration_ns < target->minimum_duration_ns) {
    target->minimum_duration_ns = delta->minimum_duration_ns;
  }
  if (delta->maximum_duration_ns > target->maximum_duration_ns) {
    target->maximum_duration_ns = delta->maximum_duration_ns;
  }
  target->last_duration_ns = delta->last_duration_ns;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_STATISTICS_INTERNAL_H_
