// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_COUNTER_H_
#define IREE_TOOLING_PROFILE_COUNTER_H_

#include "iree/tooling/profile/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_profile_counter_set_t {
  // Immutable counter set metadata record copied from the profile bundle.
  iree_hal_profile_counter_set_record_t record;
  // Borrowed counter set name from the mapped profile bundle.
  iree_string_view_t name;
} iree_profile_counter_set_t;

typedef struct iree_profile_counter_t {
  // Immutable counter metadata record copied from the profile bundle.
  iree_hal_profile_counter_record_t record;
  // Borrowed hardware block name from the mapped profile bundle.
  iree_string_view_t block_name;
  // Borrowed counter name from the mapped profile bundle.
  iree_string_view_t name;
  // Borrowed counter description from the mapped profile bundle.
  iree_string_view_t description;
} iree_profile_counter_t;

typedef struct iree_profile_counter_aggregate_t {
  // Session-local physical device ordinal for this aggregate row.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal for this aggregate row.
  uint32_t queue_ordinal;
  // Producer-defined stream identifier for this aggregate row.
  uint64_t stream_id;
  // Producer-local counter set identifier for this aggregate row.
  uint64_t counter_set_id;
  // Counter ordinal within |counter_set_id| for this aggregate row.
  uint32_t counter_ordinal;
  // Producer-local executable identifier for this aggregate row.
  uint64_t executable_id;
  // Export ordinal for this aggregate row.
  uint32_t export_ordinal;
  // Process-local command-buffer identifier for this aggregate row.
  uint64_t command_buffer_id;
  // Number of matched counter samples contributing to this aggregate row.
  uint64_t sample_count;
  // Number of raw uint64_t values contributing to this aggregate row.
  uint64_t raw_value_count;
  // First matched producer-local counter sample identifier.
  uint64_t first_sample_id;
  // Last matched producer-local counter sample identifier.
  uint64_t last_sample_id;
  // Minimum per-sample counter value sum.
  double minimum_value;
  // Maximum per-sample counter value sum.
  double maximum_value;
  // Sum of per-sample counter value sums.
  double total_value;
  // Running mean of per-sample counter value sums.
  double mean_value;
  // Running sum of squares of differences from |mean_value|.
  double m2_value;
} iree_profile_counter_aggregate_t;

typedef struct iree_profile_counter_context_t {
  // Host allocator used for dynamic counter metadata and aggregate rows.
  iree_allocator_t host_allocator;
  // Dynamic array of counter set metadata entries.
  iree_profile_counter_set_t* counter_sets;
  // Number of valid entries in |counter_sets|.
  iree_host_size_t counter_set_count;
  // Capacity of |counter_sets| in entries.
  iree_host_size_t counter_set_capacity;
  // Dynamic array of counter metadata entries.
  iree_profile_counter_t* counters;
  // Number of valid entries in |counters|.
  iree_host_size_t counter_count;
  // Capacity of |counters| in entries.
  iree_host_size_t counter_capacity;
  // Dynamic array of aggregate counter rows.
  iree_profile_counter_aggregate_t* aggregates;
  // Number of valid entries in |aggregates|.
  iree_host_size_t aggregate_count;
  // Capacity of |aggregates| in entries.
  iree_host_size_t aggregate_capacity;
  // Total counter sample records parsed before filtering.
  uint64_t total_sample_count;
  // Counter sample records matched by --id and --filter.
  uint64_t matched_sample_count;
  // Matched counter sample records from truncated chunks.
  uint64_t truncated_sample_count;
} iree_profile_counter_context_t;

// Reads a profile bundle from |path| and writes a counter report to |file|.
iree_status_t iree_profile_counter_file(iree_string_view_t path,
                                        iree_string_view_t format,
                                        iree_string_view_t filter,
                                        int64_t id_filter, bool emit_samples,
                                        FILE* file,
                                        iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_COUNTER_H_
