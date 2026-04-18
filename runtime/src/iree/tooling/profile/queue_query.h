// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_QUEUE_QUERY_H_
#define IREE_TOOLING_PROFILE_QUEUE_QUERY_H_

#include "iree/tooling/profile/model.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Host-timestamped queue event retained by the active queue query.
typedef struct iree_profile_queue_event_row_t {
  // Immutable queue event record copied from the profile bundle.
  iree_hal_profile_queue_event_t record;
} iree_profile_queue_event_row_t;

// Device-timestamped queue event retained by the active queue query.
typedef struct iree_profile_queue_device_event_row_t {
  // Immutable queue device event record copied from the profile bundle.
  iree_hal_profile_queue_device_event_t record;
} iree_profile_queue_device_event_row_t;

// Host-timestamped execution span retained by the active queue query.
typedef struct iree_profile_host_execution_event_row_t {
  // Immutable host execution event record copied from the profile bundle.
  iree_hal_profile_host_execution_event_t record;
} iree_profile_host_execution_event_row_t;

// Filtered queue-operation query state built for queue and explain reports.
typedef struct iree_profile_queue_event_query_t {
  // Host allocator used for dynamic query rows.
  iree_allocator_t host_allocator;
  // Dynamic array of query-selected queue operation event rows.
  iree_profile_queue_event_row_t* queue_events;
  // Number of valid entries in |queue_events|.
  iree_host_size_t queue_event_count;
  // Capacity of |queue_events| in entries.
  iree_host_size_t queue_event_capacity;
  // Dynamic array of query-selected device-timestamped queue event rows.
  iree_profile_queue_device_event_row_t* queue_device_events;
  // Number of valid entries in |queue_device_events|.
  iree_host_size_t queue_device_event_count;
  // Capacity of |queue_device_events| in entries.
  iree_host_size_t queue_device_event_capacity;
  // Dynamic array of query-selected host execution span rows.
  iree_profile_host_execution_event_row_t* host_execution_events;
  // Number of valid entries in |host_execution_events|.
  iree_host_size_t host_execution_event_count;
  // Capacity of |host_execution_events| in entries.
  iree_host_size_t host_execution_event_capacity;
  // Total queue operation records parsed before filtering.
  uint64_t total_queue_event_count;
  // Queue operation records matched by the active filter.
  uint64_t matched_queue_event_count;
  // Device-timestamped queue operation records parsed before filtering.
  uint64_t total_queue_device_event_count;
  // Device-timestamped queue operation records matched by the active filter.
  uint64_t matched_queue_device_event_count;
  // Host execution span records parsed before filtering.
  uint64_t total_host_execution_event_count;
  // Host execution span records matched by the active filter.
  uint64_t matched_host_execution_event_count;
} iree_profile_queue_event_query_t;

// Initializes |out_query| for filtered queue operation rows.
void iree_profile_queue_event_query_initialize(
    iree_allocator_t host_allocator,
    iree_profile_queue_event_query_t* out_query);

// Releases dynamic rows owned by |query|.
void iree_profile_queue_event_query_deinitialize(
    iree_profile_queue_event_query_t* query);

// Processes one queue-operation chunk into |query| if it is a supported queue
// event family. Non-queue chunks are ignored.
iree_status_t iree_profile_queue_event_query_process_record(
    iree_profile_queue_event_query_t* query, const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_QUEUE_QUERY_H_
