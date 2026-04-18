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

typedef struct iree_profile_model_t iree_profile_model_t;

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

typedef struct iree_profile_counter_sample_row_t {
  // Counter sample record valid only for the callback duration.
  const iree_hal_profile_counter_sample_record_t* sample;
  // Counter set metadata associated with |sample|.
  const iree_profile_counter_set_t* counter_set;
  // Counter metadata for the sampled value represented by this row.
  const iree_profile_counter_t* counter;
  // Resolved executable/export key valid only for the callback duration.
  iree_string_view_t key;
  // Raw sample values valid only for the callback duration.
  iree_const_byte_span_t sample_values;
  // Sum of the raw values for |counter|.
  double value_sum;
  // True when the source chunk was marked truncated by the producer.
  bool is_truncated;
} iree_profile_counter_sample_row_t;

typedef iree_status_t (*iree_profile_counter_sample_callback_fn_t)(
    void* user_data, const iree_profile_counter_sample_row_t* row);

typedef struct iree_profile_counter_sample_callback_t {
  // Optional callback invoked for each matched counter sample row.
  iree_profile_counter_sample_callback_fn_t fn;
  // Opaque user data passed to |fn|.
  void* user_data;
} iree_profile_counter_sample_callback_t;

// Initializes |out_context| for counter metadata and sample aggregation.
void iree_profile_counter_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_counter_context_t* out_context);

// Releases dynamic arrays owned by |context|.
void iree_profile_counter_context_deinitialize(
    iree_profile_counter_context_t* context);

// Processes counter-set and counter metadata records into |context|.
iree_status_t iree_profile_counter_process_metadata_record(
    iree_profile_counter_context_t* context,
    const iree_hal_profile_file_record_t* record);

// Processes counter sample records from one profile file record into |context|.
//
// Counter-set and counter metadata must be processed into |context| before
// sample records are processed.
// Non-counter-sample records are ignored. |filter| matches counter names,
// counter-set names, and resolved executable/export keys. |id_filter| restricts
// matched sample/dispatch/submission/command-buffer ids when non-negative.
// When |sample_callback.fn| is non-NULL, each matched counter sample value row
// is delivered to the callback after the aggregate state has been updated.
iree_status_t iree_profile_counter_process_sample_records(
    iree_profile_counter_context_t* counter_context,
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter, iree_profile_counter_sample_callback_t sample_callback);

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
