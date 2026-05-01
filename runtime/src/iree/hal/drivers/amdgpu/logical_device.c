// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/logical_device.h"

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/amdgpu/abi/signal.h"
#include "iree/hal/drivers/amdgpu/allocator.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/aql_program_builder.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/executable_cache.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile_events.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/profile_counters.h"
#include "iree/hal/drivers/amdgpu/profile_device_metrics.h"
#include "iree/hal/drivers/amdgpu/profile_traces.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/epoch_signal_table.h"
#include "iree/hal/drivers/amdgpu/util/kfd.h"
#include "iree/hal/drivers/amdgpu/util/notification_ring.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
#include "iree/hal/utils/file_registry.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static iree_hal_amdgpu_queue_affinity_domain_t
iree_hal_amdgpu_logical_device_queue_affinity_domain(
    const iree_hal_amdgpu_logical_device_t* logical_device) {
  return (iree_hal_amdgpu_queue_affinity_domain_t){
      .supported_affinity = logical_device->queue_affinity_mask,
      .physical_device_count = logical_device->physical_device_count,
      .queue_count_per_physical_device =
          logical_device->system->topology.gpu_agent_queue_count,
  };
}

// Returns the queue for a flattened logical queue ordinal.
static iree_status_t iree_hal_amdgpu_logical_device_queue_from_ordinal(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_host_size_t queue_ordinal,
    iree_hal_amdgpu_virtual_queue_t** out_queue) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(out_queue);
  *out_queue = NULL;

  iree_hal_amdgpu_queue_affinity_resolved_t resolved;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_resolve_ordinal(
      iree_hal_amdgpu_logical_device_queue_affinity_domain(logical_device),
      queue_ordinal, &resolved));

  iree_hal_amdgpu_physical_device_t* physical_device =
      logical_device->physical_devices[resolved.physical_device_ordinal];
  if (IREE_UNLIKELY(resolved.physical_queue_ordinal >=
                    physical_device->host_queue_count)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "queue affinity ordinal %" PRIhsz
                            " maps to invalid host queue ordinal "
                            "%" PRIhsz " on physical device %" PRIhsz,
                            queue_ordinal, resolved.physical_queue_ordinal,
                            resolved.physical_device_ordinal);
  }

  *out_queue =
      &physical_device->host_queues[resolved.physical_queue_ordinal].base;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_logical_device_options_t
//===----------------------------------------------------------------------===//

// Power-of-two size for the shared host small block pool in bytes.
// Used for small host-side transients/wrappers of device-side resources.
#define IREE_HAL_AMDGPU_LOGICAL_DEVICE_DEFAULT_SMALL_HOST_BLOCK_SIZE (8 * 1024)

// Minimum size of a small host block (some structures require at least this
// much memory).
#define IREE_HAL_AMDGPU_LOGICAL_DEVICE_MIN_SMALL_HOST_BLOCK_SIZE (4 * 1024)

// Power-of-two size for the shared host large block pool in bytes.
// Used for resource tracking and other larger host-side transients.
#define IREE_HAL_AMDGPU_LOGICAL_DEVICE_DEFAULT_LARGE_HOST_BLOCK_SIZE (64 * 1024)

// Minimum size of a large host block (some structures require at least this
// much memory).
#define IREE_HAL_AMDGPU_LOGICAL_DEVICE_MIN_LARGE_HOST_BLOCK_SIZE (64 * 1024)

IREE_API_EXPORT void iree_hal_amdgpu_logical_device_options_initialize(
    iree_hal_amdgpu_logical_device_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));

  // TODO(benvanik): set defaults based on compiler configuration. Flags should
  // not be used as multiple devices may be configured within the process or the
  // hosting application may be authored in python/etc that does not use a flags
  // mechanism accessible here.

  out_options->host_block_pools.small.block_size =
      IREE_HAL_AMDGPU_LOGICAL_DEVICE_DEFAULT_SMALL_HOST_BLOCK_SIZE;
  out_options->host_block_pools.large.block_size =
      IREE_HAL_AMDGPU_LOGICAL_DEVICE_DEFAULT_LARGE_HOST_BLOCK_SIZE;
  out_options->host_block_pools.command_buffer.usable_block_size =
      IREE_HAL_AMDGPU_AQL_PROGRAM_DEFAULT_BLOCK_SIZE;

  out_options->device_block_pools.small.block_size =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCK_SIZE_DEFAULT;
  out_options->device_block_pools.small.initial_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCK_INITIAL_CAPACITY_DEFAULT;
  out_options->device_block_pools.large.block_size =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCK_SIZE_DEFAULT;
  out_options->device_block_pools.large.initial_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCK_INITIAL_CAPACITY_DEFAULT;

  out_options->default_pool.range_length =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_RANGE_LENGTH_DEFAULT;
  out_options->default_pool.alignment =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_ALIGNMENT_DEFAULT;
  out_options->default_pool.frontier_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_FRONTIER_CAPACITY_DEFAULT;

  out_options->queue_placement = IREE_HAL_AMDGPU_QUEUE_PLACEMENT_ANY;
  out_options->host_queues.aql_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_HOST_QUEUE_AQL_CAPACITY;
  out_options->host_queues.notification_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_HOST_QUEUE_NOTIFICATION_CAPACITY;
  out_options->host_queues.kernarg_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_HOST_QUEUE_KERNARG_CAPACITY;
  out_options->host_queues.upload_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_HOST_QUEUE_UPLOAD_CAPACITY;

  out_options->preallocate_pools = 1;
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_logical_device_options_parse(
    iree_hal_amdgpu_logical_device_options_t* options,
    iree_string_pair_list_t params) {
  IREE_ASSERT_ARGUMENT(options);
  if (!params.count) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_string_pair_t* first_param = &params.pairs[0];
  iree_status_t status = iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "AMDGPU logical device options do not support key/value parameter '%.*s'",
      (int)first_param->key.size, first_param->key.data);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amdgpu_logical_device_options_verify_supported_features(
    const iree_hal_amdgpu_logical_device_options_t* options) {
  IREE_ASSERT_ARGUMENT(options);
  switch (options->queue_placement) {
    case IREE_HAL_AMDGPU_QUEUE_PLACEMENT_ANY:
    case IREE_HAL_AMDGPU_QUEUE_PLACEMENT_HOST:
      break;
    case IREE_HAL_AMDGPU_QUEUE_PLACEMENT_DEVICE:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AMDGPU device queue placement is not implemented; use "
          "queue_placement=any or queue_placement=host");
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid AMDGPU queue placement value %u",
                              (uint32_t)options->queue_placement);
  }
  if (options->exclusive_execution) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU exclusive_execution is not implemented");
  }
  if (options->wait_active_for_ns < 0) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait_active_for_ns must be non-negative (got %" PRId64 ")",
        options->wait_active_for_ns);
  }
  if (options->wait_active_for_ns != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU wait_active_for_ns is not implemented; "
                            "use 0");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_logical_device_options_verify(
    const iree_hal_amdgpu_logical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_logical_device_options_verify_supported_features(
              options));

  if (options->host_block_pools.small.block_size <
          IREE_HAL_AMDGPU_LOGICAL_DEVICE_MIN_SMALL_HOST_BLOCK_SIZE ||
      !iree_host_size_is_power_of_two(
          options->host_block_pools.small.block_size)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "small host block pool size invalid, expected a "
                "power-of-two greater than %d and got %" PRIhsz,
                IREE_HAL_AMDGPU_LOGICAL_DEVICE_MIN_SMALL_HOST_BLOCK_SIZE,
                options->host_block_pools.small.block_size));
  }
  if (options->host_block_pools.large.block_size <
          IREE_HAL_AMDGPU_LOGICAL_DEVICE_MIN_LARGE_HOST_BLOCK_SIZE ||
      !iree_host_size_is_power_of_two(
          options->host_block_pools.large.block_size)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "large host block pool size invalid, expected a "
                "power-of-two greater than %d and got %" PRIhsz,
                IREE_HAL_AMDGPU_LOGICAL_DEVICE_MIN_LARGE_HOST_BLOCK_SIZE,
                options->host_block_pools.large.block_size));
  }
  if (options->host_block_pools.command_buffer.usable_block_size <
          IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE ||
      options->host_block_pools.command_buffer.usable_block_size > UINT32_MAX ||
      !iree_host_size_is_power_of_two(
          options->host_block_pools.command_buffer.usable_block_size)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "command-buffer host block pool usable size invalid, expected "
                "a power-of-two between %u and %u and got %" PRIhsz,
                IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE, UINT32_MAX,
                options->host_block_pools.command_buffer.usable_block_size));
  }

  if (topology->gpu_agent_queue_count > UINT8_MAX) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "gpu_agent_queue_count=%" PRIhsz
                             " exceeds the queue-axis encoding limit (%u)",
                             topology->gpu_agent_queue_count, UINT8_MAX));
  }
  iree_host_size_t total_queue_count = 0;
  if (!iree_host_size_checked_mul(topology->gpu_agent_count,
                                  topology->gpu_agent_queue_count,
                                  &total_queue_count) ||
      total_queue_count > IREE_HAL_MAX_QUEUES) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "topology queue space does not fit in iree_hal_queue_affinity_t "
            "(gpu_agent_count=%" PRIhsz ", gpu_agent_queue_count=%" PRIhsz
            ", max_total_queues=%" PRIhsz ")",
            topology->gpu_agent_count, topology->gpu_agent_queue_count,
            (iree_host_size_t)IREE_HAL_MAX_QUEUES));
  }
  if (!iree_host_size_is_power_of_two(options->host_queues.aql_capacity) ||
      !iree_host_size_is_power_of_two(
          options->host_queues.notification_capacity) ||
      !iree_host_size_is_power_of_two(options->host_queues.kernarg_capacity) ||
      (options->host_queues.upload_capacity != 0 &&
       !iree_host_size_is_power_of_two(options->host_queues.upload_capacity))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "host queue AQL, notification, kernarg, and "
                             "upload capacities must all be powers of two, "
                             "with zero allowed for disabled upload capacity "
                             "(got aql=%u, notification=%u, kernarg_blocks=%u, "
                             "upload_bytes=%u)",
                             options->host_queues.aql_capacity,
                             options->host_queues.notification_capacity,
                             options->host_queues.kernarg_capacity,
                             options->host_queues.upload_capacity));
  }
  if (options->host_queues.kernarg_capacity / 2u <
      options->host_queues.aql_capacity) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "host queue kernarg capacity must be at least 2x the AQL queue "
                "capacity (got kernarg_blocks=%u, aql_packets=%u)",
                options->host_queues.kernarg_capacity,
                options->host_queues.aql_capacity));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_logical_device_t
//===----------------------------------------------------------------------===//

static const iree_hal_device_vtable_t iree_hal_amdgpu_logical_device_vtable;

static iree_hal_amdgpu_logical_device_t* iree_hal_amdgpu_logical_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_logical_device_vtable);
  return (iree_hal_amdgpu_logical_device_t*)base_value;
}

static bool iree_hal_amdgpu_logical_device_profiling_needs_hsa_timestamps(
    iree_hal_device_profiling_data_families_t data_families) {
  return iree_any_bit_set(data_families,
                          IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |
                              IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                              IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES |
                              IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES);
}

static bool iree_hal_amdgpu_logical_device_profiling_needs_clock_correlations(
    iree_hal_device_profiling_data_families_t data_families) {
  return iree_hal_amdgpu_logical_device_profiling_needs_hsa_timestamps(
             data_families) ||
         iree_any_bit_set(data_families,
                          IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES);
}

static iree_hal_device_profiling_data_families_t
iree_hal_amdgpu_logical_device_lightweight_statistics_data_families(void) {
  return IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
         IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |
         IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
}

static iree_hal_device_profiling_options_t
iree_hal_amdgpu_logical_device_resolve_profiling_options(
    const iree_hal_device_profiling_options_t* options) {
  iree_hal_device_profiling_options_t resolved_options = *options;
  if (resolved_options.data_families == IREE_HAL_DEVICE_PROFILING_DATA_NONE &&
      iree_hal_device_profiling_options_requests_lightweight_statistics(
          options)) {
    resolved_options.data_families =
        iree_hal_amdgpu_logical_device_lightweight_statistics_data_families();
  }
  resolved_options.flags &=
      ~IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS;
  return resolved_options;
}

// Power-of-two capacity for logical-device memory lifecycle event buffering.
#define IREE_HAL_AMDGPU_LOGICAL_DEVICE_PROFILE_MEMORY_EVENT_CAPACITY (64 * 1024)

// Power-of-two capacity for logical-device queue operation event buffering.
#define IREE_HAL_AMDGPU_LOGICAL_DEVICE_PROFILE_QUEUE_EVENT_CAPACITY (64 * 1024)

static iree_hal_profile_chunk_metadata_t
iree_hal_amdgpu_logical_device_profile_session_metadata(
    iree_hal_amdgpu_logical_device_t* logical_device, uint64_t session_id) {
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_SESSION;
  metadata.name = logical_device->identifier;
  metadata.session_id = session_id;
  return metadata;
}

static uint64_t iree_hal_amdgpu_logical_device_profile_queue_stream_id(
    uint32_t physical_device_ordinal, uint32_t queue_ordinal) {
  return ((uint64_t)physical_device_ordinal << 32) | (uint64_t)queue_ordinal;
}

static bool iree_hal_amdgpu_logical_device_profile_memory_events_requested(
    const iree_hal_amdgpu_logical_device_t* logical_device) {
  return iree_hal_device_profiling_options_requests_data(
             &logical_device->profiling.options,
             IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS) &&
         logical_device->profiling.options.sink &&
         iree_hal_amdgpu_profile_event_streams_has_memory_storage(
             &logical_device->profiling.event_streams);
}

bool iree_hal_amdgpu_logical_device_should_record_profile_memory_events(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  return iree_hal_amdgpu_logical_device_profile_memory_events_requested(
      logical_device);
}

static void iree_hal_amdgpu_logical_device_reset_profile_options(
    iree_hal_amdgpu_logical_device_t* logical_device) {
  iree_hal_device_profiling_options_storage_free(
      logical_device->profiling.options_storage,
      logical_device->host_allocator);
  logical_device->profiling.options_storage = NULL;
  logical_device->profiling.options = (iree_hal_device_profiling_options_t){0};
}

bool iree_hal_amdgpu_logical_device_should_profile_dispatch(
    iree_hal_amdgpu_logical_device_t* logical_device, uint64_t executable_id,
    uint32_t export_ordinal, uint64_t command_buffer_id, uint32_t command_index,
    uint32_t physical_device_ordinal, uint32_t queue_ordinal) {
  if (!iree_any_bit_set(logical_device->profiling.options.data_families,
                        IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                            IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES |
                            IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES)) {
    return false;
  }

  const iree_hal_profile_capture_filter_t* filter =
      &logical_device->profiling.options.capture_filter;
  if (!iree_hal_profile_capture_filter_matches_location(
          filter, command_buffer_id, command_index, physical_device_ordinal,
          queue_ordinal)) {
    return false;
  }
  if (iree_any_bit_set(
          filter->flags,
          IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_EXPORT_PATTERN)) {
    return iree_hal_amdgpu_profile_metadata_export_matches(
        &logical_device->profile_metadata, executable_id, export_ordinal,
        filter->executable_export_pattern);
  }
  return true;
}

uint64_t iree_hal_amdgpu_logical_device_allocate_profile_memory_allocation_id(
    iree_hal_device_t* base_device, uint64_t* out_session_id) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  *out_session_id = 0;
  if (!iree_hal_amdgpu_logical_device_profile_memory_events_requested(
          logical_device)) {
    return 0;
  }

  return iree_hal_amdgpu_profile_event_streams_allocate_memory_allocation_id(
      &logical_device->profiling.event_streams,
      logical_device->profiling.session_id, out_session_id);
}

bool iree_hal_amdgpu_logical_device_record_profile_memory_event_for_session(
    iree_hal_device_t* base_device, uint64_t session_id,
    const iree_hal_profile_memory_event_t* event) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  if (!iree_hal_amdgpu_logical_device_profile_memory_events_requested(
          logical_device)) {
    return false;
  }

  return iree_hal_amdgpu_profile_event_streams_record_memory_event(
      &logical_device->profiling.event_streams,
      logical_device->profiling.session_id, session_id, event);
}

bool iree_hal_amdgpu_logical_device_record_profile_memory_event(
    iree_hal_device_t* base_device,
    const iree_hal_profile_memory_event_t* event) {
  return iree_hal_amdgpu_logical_device_record_profile_memory_event_for_session(
      base_device, /*session_id=*/0, event);
}

static bool iree_hal_amdgpu_logical_device_profile_queue_events_requested(
    const iree_hal_amdgpu_logical_device_t* logical_device) {
  return iree_hal_device_profiling_options_requests_data(
             &logical_device->profiling.options,
             IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS) &&
         logical_device->profiling.options.sink &&
         iree_hal_amdgpu_profile_event_streams_has_queue_storage(
             &logical_device->profiling.event_streams);
}

void iree_hal_amdgpu_logical_device_record_profile_queue_event(
    iree_hal_device_t* base_device,
    const iree_hal_profile_queue_event_t* event) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  if (!iree_hal_amdgpu_logical_device_profile_queue_events_requested(
          logical_device)) {
    return;
  }

  iree_hal_amdgpu_profile_event_streams_record_queue_event(
      &logical_device->profiling.event_streams, event);
}

static iree_status_t
iree_hal_amdgpu_logical_device_sample_profile_clock_correlation(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_amdgpu_physical_device_t* physical_device,
    iree_hal_profile_clock_correlation_record_t* out_record) {
  if (IREE_UNLIKELY(physical_device->device_ordinal > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "profile clock correlation physical device ordinal out of range: "
        "%" PRIhsz,
        physical_device->device_ordinal);
  }

  iree_hal_amdgpu_device_clock_counters_t counters = {0};
  const iree_time_t host_time_begin_ns = iree_time_now();
  iree_status_t status = iree_hal_amdgpu_device_clock_source_sample(
      &logical_device->system->device_clock_source, physical_device->driver_uid,
      &counters);
  const iree_time_t host_time_end_ns = iree_time_now();

  if (iree_status_is_ok(status)) {
    *out_record = iree_hal_profile_clock_correlation_record_default();
    out_record->flags =
        IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
        IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP |
        IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_SYSTEM_TIMESTAMP |
        IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET;
    out_record->physical_device_ordinal =
        (uint32_t)physical_device->device_ordinal;
    out_record->sample_id =
        logical_device->profiling.next_clock_correlation_sample_id++;
    out_record->device_tick = counters.device_clock_counter;
    out_record->host_cpu_timestamp_ns = counters.host_cpu_timestamp_ns;
    out_record->host_system_timestamp = counters.host_system_timestamp;
    out_record->host_system_frequency_hz = counters.host_system_frequency_hz;
    out_record->host_time_begin_ns = host_time_begin_ns;
    out_record->host_time_end_ns = host_time_end_ns;
  } else {
    status = iree_status_annotate_f(
        status,
        "sampling profile clock correlation for physical_device_ordinal=%zu "
        "driver_uid=%" PRIu32,
        physical_device->device_ordinal, physical_device->driver_uid);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_logical_device_write_profile_devices(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_profile_sink_t* sink, uint64_t session_id) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t record_count = logical_device->physical_device_count;
  if (record_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "logical device has no physical devices (initialization incomplete)");
  }

  iree_host_size_t records_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              0, &records_size,
              IREE_STRUCT_FIELD(record_count, iree_hal_profile_device_record_t,
                                NULL)));
  iree_hal_profile_device_record_t* records = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(logical_device->host_allocator, records_size,
                                (void**)&records));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < record_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[i];
    if (IREE_UNLIKELY(physical_device->device_ordinal > UINT32_MAX ||
                      physical_device->host_queue_count > UINT32_MAX)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "profile device metadata ordinals out of range: device=%" PRIhsz
          ", queue_count=%" PRIhsz,
          physical_device->device_ordinal, physical_device->host_queue_count);
      break;
    }

    records[i] = iree_hal_profile_device_record_default();
    records[i].physical_device_ordinal =
        (uint32_t)physical_device->device_ordinal;
    records[i].queue_count = (uint32_t)physical_device->host_queue_count;
    if (physical_device->has_physical_device_uuid) {
      records[i].flags |= IREE_HAL_PROFILE_DEVICE_FLAG_PHYSICAL_DEVICE_UUID;
      memcpy(records[i].physical_device_uuid,
             physical_device->physical_device_uuid,
             sizeof(records[i].physical_device_uuid));
    }
  }

  if (iree_status_is_ok(status)) {
    iree_hal_profile_chunk_metadata_t metadata =
        iree_hal_profile_chunk_metadata_default();
    metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES;
    metadata.name = logical_device->identifier;
    metadata.session_id = session_id;
    iree_const_byte_span_t iovec =
        iree_make_const_byte_span(records, records_size);
    status = iree_hal_profile_sink_write(sink, &metadata, 1, &iovec);
  }

  iree_allocator_free(logical_device->host_allocator, records);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_logical_device_write_profile_queues(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_profile_sink_t* sink, uint64_t session_id) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t record_count = 0;
  for (iree_host_size_t i = 0; i < logical_device->physical_device_count; ++i) {
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[i];
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            record_count, physical_device->host_queue_count, &record_count))) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "profile queue metadata count overflow");
    }
  }
  if (record_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "logical device has no host queues (initialization incomplete)");
  }

  iree_host_size_t records_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              0, &records_size,
              IREE_STRUCT_FIELD(record_count, iree_hal_profile_queue_record_t,
                                NULL)));
  iree_hal_profile_queue_record_t* records = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(logical_device->host_allocator, records_size,
                                (void**)&records));

  iree_status_t status = iree_ok_status();
  iree_host_size_t record_ordinal = 0;
  for (iree_host_size_t i = 0;
       i < logical_device->physical_device_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[i];
    if (IREE_UNLIKELY(physical_device->device_ordinal > UINT32_MAX)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "profile queue metadata physical device "
                                "ordinal out of range: %" PRIhsz,
                                physical_device->device_ordinal);
      break;
    }
    const uint32_t physical_device_ordinal =
        (uint32_t)physical_device->device_ordinal;
    for (iree_host_size_t j = 0;
         j < physical_device->host_queue_count && iree_status_is_ok(status);
         ++j) {
      if (IREE_UNLIKELY(j > UINT32_MAX)) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "profile queue metadata queue ordinal out of range: %" PRIhsz, j);
        break;
      }
      const uint32_t queue_ordinal = (uint32_t)j;
      records[record_ordinal] = iree_hal_profile_queue_record_default();
      records[record_ordinal].physical_device_ordinal = physical_device_ordinal;
      records[record_ordinal].queue_ordinal = queue_ordinal;
      records[record_ordinal].stream_id =
          iree_hal_amdgpu_logical_device_profile_queue_stream_id(
              physical_device_ordinal, queue_ordinal);
      ++record_ordinal;
    }
  }

  if (iree_status_is_ok(status)) {
    iree_hal_profile_chunk_metadata_t metadata =
        iree_hal_profile_chunk_metadata_default();
    metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES;
    metadata.name = logical_device->identifier;
    metadata.session_id = session_id;
    iree_const_byte_span_t iovec =
        iree_make_const_byte_span(records, records_size);
    status = iree_hal_profile_sink_write(sink, &metadata, 1, &iovec);
  }

  iree_allocator_free(logical_device->host_allocator, records);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t
iree_hal_amdgpu_logical_device_write_profile_clock_correlations(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_profile_sink_t* sink, uint64_t session_id) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t record_count = logical_device->physical_device_count;
  if (record_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "logical device has no physical devices (initialization incomplete)");
  }

  iree_host_size_t records_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              0, &records_size,
              IREE_STRUCT_FIELD(record_count,
                                iree_hal_profile_clock_correlation_record_t,
                                NULL)));
  iree_hal_profile_clock_correlation_record_t* records = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(logical_device->host_allocator, records_size,
                                (void**)&records));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < record_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amdgpu_logical_device_sample_profile_clock_correlation(
        logical_device, logical_device->physical_devices[i], &records[i]);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_profile_chunk_metadata_t metadata =
        iree_hal_profile_chunk_metadata_default();
    metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS;
    metadata.name = logical_device->identifier;
    metadata.session_id = session_id;
    iree_const_byte_span_t iovec =
        iree_make_const_byte_span(records, records_size);
    status = iree_hal_profile_sink_write(sink, &metadata, 1, &iovec);
  }

  iree_allocator_free(logical_device->host_allocator, records);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static bool iree_hal_amdgpu_logical_device_profile_needs_executable_artifacts(
    iree_hal_device_profiling_data_families_t data_families) {
  return iree_any_bit_set(data_families,
                          IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
                              IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES);
}

static iree_status_t iree_hal_amdgpu_logical_device_write_profile_metadata(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_hal_device_profiling_data_families_t data_families) {
  const bool emit_executable_artifacts =
      iree_hal_amdgpu_logical_device_profile_needs_executable_artifacts(
          data_families);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_write_profile_devices(
      logical_device, sink, session_id));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_write_profile_queues(
      logical_device, sink, session_id));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_metadata_write(
      &logical_device->profile_metadata, sink, session_id,
      logical_device->identifier, emit_executable_artifacts,
      &logical_device->profiling.metadata_cursor));
  if (iree_hal_amdgpu_logical_device_profiling_needs_clock_correlations(
          data_families)) {
    return iree_hal_amdgpu_logical_device_write_profile_clock_correlations(
        logical_device, sink, session_id);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_logical_device_write_profile_events(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_profile_sink_t* sink, uint64_t session_id) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_amdgpu_profile_event_streams_write_queue(
      &logical_device->profiling.event_streams, sink, session_id,
      logical_device->host_allocator);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_event_streams_write_memory(
        &logical_device->profiling.event_streams, sink, session_id,
        logical_device->host_allocator);
  }
  for (iree_host_size_t i = 0;
       i < logical_device->physical_device_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[i];
    for (iree_host_size_t j = 0;
         j < physical_device->host_queue_count && iree_status_is_ok(status);
         ++j) {
      status = iree_hal_amdgpu_host_queue_write_profile_events(
          &physical_device->host_queues[j], sink, session_id);
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_hal_amdgpu_host_queue_profile_flags_t
iree_hal_amdgpu_logical_device_queue_profile_flags(
    const iree_hal_device_profiling_options_t* options) {
  iree_hal_amdgpu_host_queue_profile_flags_t flags =
      IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_FLAG_NONE;
  if (iree_hal_device_profiling_options_requests_data(
          options, IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS)) {
    flags |= IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_FLAG_QUEUE_EVENTS;
  }
  if (iree_hal_device_profiling_options_requests_data(
          options, IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS)) {
    flags |= IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_FLAG_QUEUE_DEVICE_EVENTS;
  }
  if (iree_any_bit_set(options->data_families,
                       IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                           IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES |
                           IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES)) {
    flags |= IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_FLAG_DISPATCHES;
  }
  return flags;
}

static void iree_hal_amdgpu_logical_device_set_queue_profiling_enabled(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_amdgpu_host_queue_profile_flags_t flags) {
  for (iree_host_size_t i = 0; i < logical_device->physical_device_count; ++i) {
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[i];
    for (iree_host_size_t j = 0; j < physical_device->host_queue_count; ++j) {
      iree_hal_amdgpu_host_queue_set_profile_flags(
          &physical_device->host_queues[j], flags);
    }
  }
}

static iree_status_t iree_hal_amdgpu_logical_device_set_hsa_profiling_enabled(
    iree_hal_amdgpu_logical_device_t* logical_device, bool enabled) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, enabled ? 1 : 0);

  iree_status_t status = iree_ok_status();
  iree_host_size_t changed_count = 0;
  for (iree_host_size_t i = 0;
       i < logical_device->physical_device_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amdgpu_physical_device_set_hsa_profiling_enabled(
        logical_device->physical_devices[i], enabled);
    if (iree_status_is_ok(status)) {
      ++changed_count;
    }
  }

  if (!iree_status_is_ok(status) && enabled) {
    for (iree_host_size_t i = 0; i < changed_count; ++i) {
      status = iree_status_join(
          status, iree_hal_amdgpu_physical_device_set_hsa_profiling_enabled(
                      logical_device->physical_devices[i], false));
    }
  } else if (!enabled) {
    for (iree_host_size_t i = changed_count;
         i < logical_device->physical_device_count; ++i) {
      status = iree_status_join(
          status, iree_hal_amdgpu_physical_device_set_hsa_profiling_enabled(
                      logical_device->physical_devices[i], false));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Returns true when |queue_ordinal| is the physical device's counter range
// sampling queue.
//
// Queue affinity ANY resolves to queue 0 for ordinary submissions, so using
// the final queue gives the sampler the best chance to run independently while
// the default queue is saturated. When only one queue exists we fall back to
// that queue and sampling is necessarily ordered behind user work.
static bool iree_hal_amdgpu_logical_device_is_profile_counter_range_queue(
    const iree_hal_amdgpu_physical_device_t* physical_device,
    iree_host_size_t queue_ordinal) {
  return queue_ordinal + 1 == physical_device->host_queue_count;
}

static iree_hal_amdgpu_host_queue_t*
iree_hal_amdgpu_logical_device_select_profile_counter_range_queue(
    iree_hal_amdgpu_physical_device_t* physical_device) {
  if (physical_device->host_queue_count == 0) return NULL;
  return &physical_device->host_queues[physical_device->host_queue_count - 1];
}

static iree_status_t
iree_hal_amdgpu_logical_device_set_counter_profiling_enabled(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_amdgpu_profile_counter_session_t* counter_session, bool enabled) {
  if (!iree_hal_amdgpu_profile_counter_session_is_active(counter_session)) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, enabled ? 1 : 0);

  iree_status_t status = iree_ok_status();
  const bool capture_dispatch_samples =
      iree_hal_amdgpu_profile_counter_session_captures_dispatch_samples(
          counter_session);
  const bool capture_queue_ranges =
      iree_hal_amdgpu_profile_counter_session_captures_queue_ranges(
          counter_session);
  for (iree_host_size_t i = 0;
       i < logical_device->physical_device_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[i];
    for (iree_host_size_t j = 0;
         j < physical_device->host_queue_count && iree_status_is_ok(status);
         ++j) {
      iree_hal_amdgpu_host_queue_t* queue = &physical_device->host_queues[j];
      if (enabled) {
        iree_hal_amdgpu_profile_counter_enable_flags_t flags =
            IREE_HAL_AMDGPU_PROFILE_COUNTER_ENABLE_FLAG_NONE;
        if (capture_dispatch_samples) {
          flags |= IREE_HAL_AMDGPU_PROFILE_COUNTER_ENABLE_FLAG_DISPATCH_SAMPLES;
        }
        if (capture_queue_ranges &&
            iree_hal_amdgpu_logical_device_is_profile_counter_range_queue(
                physical_device, j)) {
          flags |= IREE_HAL_AMDGPU_PROFILE_COUNTER_ENABLE_FLAG_QUEUE_RANGES;
        }
        status = iree_hal_amdgpu_host_queue_enable_profile_counters(
            queue, counter_session, flags);
      } else {
        iree_hal_amdgpu_host_queue_disable_profile_counters(queue);
      }
    }
  }

  if (!iree_status_is_ok(status) && enabled) {
    for (iree_host_size_t i = 0; i < logical_device->physical_device_count;
         ++i) {
      iree_hal_amdgpu_physical_device_t* physical_device =
          logical_device->physical_devices[i];
      for (iree_host_size_t j = 0; j < physical_device->host_queue_count; ++j) {
        iree_hal_amdgpu_host_queue_disable_profile_counters(
            &physical_device->host_queues[j]);
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t
iree_hal_amdgpu_logical_device_start_profile_counter_ranges(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_amdgpu_profile_counter_session_t* counter_session) {
  if (!iree_hal_amdgpu_profile_counter_session_captures_queue_ranges(
          counter_session)) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  iree_host_size_t started_device_count = 0;
  for (iree_host_size_t i = 0;
       i < logical_device->physical_device_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[i];
    if (IREE_UNLIKELY(physical_device->host_queue_count == 0)) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "logical device physical device has no host "
                                "queues (initialization incomplete)");
    } else {
      iree_hal_amdgpu_host_queue_t* queue =
          iree_hal_amdgpu_logical_device_select_profile_counter_range_queue(
              physical_device);
      status = iree_hal_amdgpu_host_queue_start_profile_counter_ranges(queue);
      if (iree_status_is_ok(status)) {
        ++started_device_count;
      }
    }
  }

  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < started_device_count; ++i) {
      iree_hal_amdgpu_physical_device_t* physical_device =
          logical_device->physical_devices[i];
      iree_hal_amdgpu_host_queue_t* queue =
          iree_hal_amdgpu_logical_device_select_profile_counter_range_queue(
              physical_device);
      status = iree_status_join(
          status, iree_hal_amdgpu_host_queue_flush_profile_counter_ranges(
                      queue, /*sink=*/NULL, /*session_id=*/0,
                      IREE_HAL_AMDGPU_PROFILE_COUNTER_RANGE_FLUSH_FLAG_NONE));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t
iree_hal_amdgpu_logical_device_flush_profile_counter_ranges(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_amdgpu_profile_counter_session_t* counter_session,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_hal_amdgpu_profile_counter_range_flush_flags_t flags) {
  if (!iree_hal_amdgpu_profile_counter_session_captures_queue_ranges(
          counter_session)) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < logical_device->physical_device_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[i];
    if (IREE_UNLIKELY(physical_device->host_queue_count == 0)) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "logical device physical device has no host "
                                "queues (initialization incomplete)");
    } else {
      iree_hal_amdgpu_host_queue_t* queue =
          iree_hal_amdgpu_logical_device_select_profile_counter_range_queue(
              physical_device);
      status = iree_hal_amdgpu_host_queue_flush_profile_counter_ranges(
          queue, sink, session_id, flags);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_logical_device_set_trace_profiling_enabled(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_amdgpu_profile_trace_session_t* trace_session, bool enabled) {
  if (!iree_hal_amdgpu_profile_trace_session_is_active(trace_session)) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, enabled ? 1 : 0);

  iree_status_t status = iree_ok_status();
  iree_host_size_t changed_queue_count = 0;
  for (iree_host_size_t i = 0;
       i < logical_device->physical_device_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[i];
    for (iree_host_size_t j = 0;
         j < physical_device->host_queue_count && iree_status_is_ok(status);
         ++j) {
      iree_hal_amdgpu_host_queue_t* queue = &physical_device->host_queues[j];
      if (enabled) {
        status = iree_hal_amdgpu_host_queue_enable_profile_traces(
            queue, trace_session);
        if (iree_status_is_ok(status)) {
          ++changed_queue_count;
        }
      } else {
        iree_hal_amdgpu_host_queue_disable_profile_traces(queue);
      }
    }
  }

  if (!iree_status_is_ok(status) && enabled) {
    for (iree_host_size_t i = 0, seen_queue_count = 0;
         i < logical_device->physical_device_count &&
         seen_queue_count < changed_queue_count;
         ++i) {
      iree_hal_amdgpu_physical_device_t* physical_device =
          logical_device->physical_devices[i];
      for (iree_host_size_t j = 0; j < physical_device->host_queue_count &&
                                   seen_queue_count < changed_queue_count;
           ++j, ++seen_queue_count) {
        iree_hal_amdgpu_host_queue_disable_profile_traces(
            &physical_device->host_queues[j]);
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Selects one host queue from |queue_affinity| after intersecting with this
// logical device's supported queues. The current policy is deterministic
// first-set-bit selection, which is enough to honor explicit HIP stream
// affinities and keeps the CTS path stable. A multi-bit affinity therefore acts
// as "any of these queues"; queue_flush handles multi-bit masks by iterating
// all selected queues instead.
static iree_status_t iree_hal_amdgpu_logical_device_select_host_queue(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_amdgpu_virtual_queue_t** out_queue) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(out_queue);
  *out_queue = NULL;

  iree_hal_amdgpu_queue_affinity_resolved_t resolved;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_resolve(
      iree_hal_amdgpu_logical_device_queue_affinity_domain(logical_device),
      queue_affinity, &resolved));
  return iree_hal_amdgpu_logical_device_queue_from_ordinal(
      logical_device, resolved.queue_ordinal, out_queue);
}

// Selects the physical device backing |queue_affinity| for pool creation.
//
// Queue pools are scoped to one physical memory domain, but |queue_affinity|
// still has the usual "any queue in this mask" meaning. This helper therefore
// collapses multi-bit masks with the same deterministic first-set-bit policy as
// host queue submission. In practice IREE_HAL_QUEUE_AFFINITY_ANY usually
// selects queue 0 after intersecting with this device's supported queue mask.
static iree_status_t
iree_hal_amdgpu_logical_device_select_queue_pool_physical_device(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_amdgpu_physical_device_t** out_physical_device) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(out_physical_device);
  *out_physical_device = NULL;

  iree_hal_amdgpu_queue_affinity_resolved_t resolved;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_resolve(
      iree_hal_amdgpu_logical_device_queue_affinity_domain(logical_device),
      queue_affinity, &resolved));
  *out_physical_device =
      logical_device->physical_devices[resolved.physical_device_ordinal];
  return iree_ok_status();
}

// Normalizes command-buffer queue affinity to queues on one physical device and
// returns the physical device ordinal whose executable kernel objects may be
// baked into the recorded command stream.
static iree_status_t
iree_hal_amdgpu_logical_device_normalize_command_buffer_affinity(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_affinity_t* out_queue_affinity,
    iree_host_size_t* out_device_ordinal) {
  *out_queue_affinity = 0;
  *out_device_ordinal = 0;

  return iree_hal_amdgpu_queue_affinity_normalize_for_physical_device(
      iree_hal_amdgpu_logical_device_queue_affinity_domain(logical_device),
      queue_affinity, out_queue_affinity, out_device_ordinal);
}

static bool iree_hal_amdgpu_logical_device_query_pool_epoch(
    void* user_data, iree_async_axis_t axis, uint64_t epoch) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      (iree_hal_amdgpu_logical_device_t*)user_data;
  hsa_signal_t epoch_signal = {0};
  if (!iree_hal_amdgpu_epoch_signal_table_lookup(
          logical_device->host_queue_epoch_table, axis, &epoch_signal)) {
    return false;
  }
  iree_amd_signal_t* signal =
      (iree_amd_signal_t*)(uintptr_t)epoch_signal.handle;
  const iree_hsa_signal_value_t current_value = iree_atomic_load(
      (iree_atomic_int64_t*)&signal->value, iree_memory_order_acquire);
  if (IREE_UNLIKELY(current_value < 0 ||
                    current_value > IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE)) {
    return false;
  }
  const uint64_t current_epoch =
      (uint64_t)IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE - (uint64_t)current_value;
  return current_epoch >= epoch;
}

static void iree_hal_amdgpu_logical_device_deassign_frontier(
    iree_hal_amdgpu_logical_device_t* logical_device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  for (iree_host_size_t i = 0; i < logical_device->physical_device_count; ++i) {
    iree_hal_amdgpu_physical_device_deassign_frontier(
        logical_device->physical_devices[i]);
  }

  iree_async_frontier_tracker_release(logical_device->frontier_tracker);
  logical_device->frontier_tracker = NULL;
  logical_device->axis = 0;
  memset(&logical_device->topology_info, 0,
         sizeof(logical_device->topology_info));

  if (logical_device->host_queue_epoch_table) {
    iree_allocator_free(logical_device->host_allocator,
                        logical_device->host_queue_epoch_table);
    logical_device->host_queue_epoch_table = NULL;
  }
  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_logical_device_error_handler(void* user_data,
                                                         iree_status_t status) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      (iree_hal_amdgpu_logical_device_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Display the error in trace tooling.
  IREE_TRACE({
    char buffer[1024];
    iree_host_size_t buffer_length = 0;
    if (iree_status_format(status, sizeof(buffer), buffer, &buffer_length)) {
      IREE_TRACE_MESSAGE_DYNAMIC(ERROR, buffer, buffer_length);
    }
  });

  // Set the device sticky error status (if it is not already set).
  intptr_t current_value = 0;
  if (!iree_atomic_compare_exchange_strong(
          &logical_device->failure_status, &current_value, (intptr_t)status,
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    // Previous status was not OK; the sticky slot owns only the first failure.
    iree_status_free(status);
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_logical_device_translate_physical_options(
    const iree_hal_amdgpu_logical_device_options_t* options,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_physical_device_options_t* out_options) {
  iree_hal_amdgpu_physical_device_options_initialize(out_options);
  out_options->device_block_pools.small.block_size =
      options->device_block_pools.small.block_size;
  out_options->device_block_pools.small.initial_capacity =
      options->device_block_pools.small.initial_capacity;
  out_options->device_block_pools.large.block_size =
      options->device_block_pools.large.block_size;
  out_options->device_block_pools.large.initial_capacity =
      options->device_block_pools.large.initial_capacity;
  out_options->default_pool.range_length = options->default_pool.range_length;
  out_options->default_pool.alignment = options->default_pool.alignment;
  out_options->default_pool.frontier_capacity =
      options->default_pool.frontier_capacity;
  out_options->host_block_pool_initial_capacity =
      options->preallocate_pools ? 16 : 0;
  out_options->host_queue_count = topology->gpu_agent_queue_count;
  out_options->host_queue_aql_capacity = options->host_queues.aql_capacity;
  out_options->host_queue_notification_capacity =
      options->host_queues.notification_capacity;
  out_options->host_queue_kernarg_capacity =
      options->host_queues.kernarg_capacity;
  out_options->host_queue_upload_capacity =
      options->host_queues.upload_capacity;
  out_options->force_wait_barrier_defer = options->force_wait_barrier_defer;
}

static iree_status_t iree_hal_amdgpu_logical_device_verify_physical_options(
    const iree_hal_amdgpu_physical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology) {
  for (iree_host_size_t i = 0; i < topology->gpu_agent_count; ++i) {
    hsa_agent_t gpu_agent = topology->gpu_agents[i];
    hsa_agent_t cpu_agent = topology->cpu_agents[topology->gpu_cpu_map[i]];
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_physical_device_options_verify(options, libhsa,
                                                       cpu_agent, gpu_agent),
        "verifying GPU agent %" PRIhsz " meets required options", i);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_logical_device_allocate_storage(
    iree_string_view_t identifier, const iree_hal_amdgpu_topology_t* topology,
    iree_host_size_t physical_device_size, iree_allocator_t host_allocator,
    iree_hal_amdgpu_logical_device_t** out_logical_device) {
  *out_logical_device = NULL;

  iree_hal_amdgpu_logical_device_t* logical_device = NULL;
  iree_host_size_t physical_device_data_offset = 0;
  iree_host_size_t identifier_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(*logical_device), &total_size,
      IREE_STRUCT_FIELD(topology->gpu_agent_count,
                        iree_hal_amdgpu_physical_device_t*, NULL),
      IREE_STRUCT_ARRAY_FIELD_ALIGNED(
          topology->gpu_agent_count, physical_device_size, uint8_t,
          iree_max_align_t, &physical_device_data_offset),
      IREE_STRUCT_FIELD(identifier.size, char, &identifier_offset)));

  const iree_hal_amdgpu_queue_affinity_domain_t queue_affinity_domain = {
      .supported_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .physical_device_count = topology->gpu_agent_count,
      .queue_count_per_physical_device = topology->gpu_agent_queue_count,
  };
  iree_hal_queue_affinity_t logical_queue_affinity_mask = 0;
  for (iree_host_size_t i = 0; i < topology->gpu_agent_count; ++i) {
    iree_hal_queue_affinity_t physical_device_affinity = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_for_physical_device(
        queue_affinity_domain, i, &physical_device_affinity));
    iree_hal_queue_affinity_or_into(logical_queue_affinity_mask,
                                    physical_device_affinity);
  }

  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size,
                                             (void**)&logical_device));
  memset(logical_device, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_amdgpu_logical_device_vtable,
                               &logical_device->resource);
  iree_string_view_append_to_buffer(identifier, &logical_device->identifier,
                                    (char*)logical_device + identifier_offset);
  logical_device->host_allocator = host_allocator;
  logical_device->failure_status = IREE_ATOMIC_VAR_INIT(0);
  iree_atomic_store(&logical_device->epoch, 0, iree_memory_order_relaxed);
  logical_device->next_profile_session_id = 1;
  iree_hal_amdgpu_profile_metadata_initialize(
      host_allocator, &logical_device->profile_metadata);
  iree_hal_amdgpu_profile_event_streams_initialize(
      &logical_device->profiling.event_streams);

  // Setup physical device table first so failure cleanup has a valid table.
  logical_device->physical_device_count = topology->gpu_agent_count;
  logical_device->queue_affinity_mask = logical_queue_affinity_mask;
  uint8_t* physical_device_base =
      (uint8_t*)logical_device + physical_device_data_offset;
  for (iree_host_size_t i = 0; i < logical_device->physical_device_count; ++i) {
    logical_device->physical_devices[i] =
        (iree_hal_amdgpu_physical_device_t*)physical_device_base;
    physical_device_base += physical_device_size;
  }

  *out_logical_device = logical_device;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_logical_device_initialize_host_resources(
    iree_hal_amdgpu_logical_device_t* logical_device,
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_async_proactor_pool_t* proactor_pool,
    iree_allocator_t host_allocator) {
  logical_device->proactor_pool = proactor_pool;
  iree_async_proactor_pool_retain(logical_device->proactor_pool);

  iree_arena_block_pool_initialize(options->host_block_pools.small.block_size,
                                   host_allocator,
                                   &logical_device->host_block_pools.small);
  iree_arena_block_pool_initialize(options->host_block_pools.large.block_size,
                                   host_allocator,
                                   &logical_device->host_block_pools.large);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_block_pool_initialize(
      options->host_block_pools.command_buffer.usable_block_size,
      host_allocator, &logical_device->host_block_pools.command_buffer));
  return iree_async_proactor_pool_get(logical_device->proactor_pool, 0,
                                      &logical_device->proactor);
}

static iree_status_t
iree_hal_amdgpu_logical_device_initialize_system_and_allocator(
    iree_hal_amdgpu_logical_device_t* logical_device,
    const iree_hal_amdgpu_logical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    iree_allocator_t host_allocator) {
  iree_hal_amdgpu_system_options_t system_options = {
      .exclusive_execution = options->exclusive_execution,
  };
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_system_allocate(libhsa, topology, system_options,
                                      host_allocator, &logical_device->system));
  return iree_hal_amdgpu_allocator_create(
      logical_device, &logical_device->system->libhsa,
      &logical_device->system->topology, host_allocator,
      &logical_device->device_allocator);
}

static iree_status_t iree_hal_amdgpu_logical_device_initialize_physical_devices(
    iree_hal_amdgpu_logical_device_t* logical_device,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_physical_device_options_t* options,
    iree_allocator_t host_allocator) {
  for (iree_host_size_t device_ordinal = 0;
       device_ordinal < logical_device->physical_device_count;
       ++device_ordinal) {
    const iree_host_size_t host_ordinal = topology->gpu_cpu_map[device_ordinal];
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_physical_device_initialize(
        (iree_hal_device_t*)logical_device, logical_device->system, options,
        logical_device->proactor, host_ordinal,
        &logical_device->system->host_memory_pools[host_ordinal],
        device_ordinal, host_allocator,
        logical_device->physical_devices[device_ordinal]));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_logical_device_warmup_host_pools(
    iree_hal_amdgpu_logical_device_t* logical_device) {
  IREE_RETURN_IF_ERROR(iree_arena_block_pool_preallocate(
      &logical_device->host_block_pools.small, 16));
  IREE_RETURN_IF_ERROR(iree_arena_block_pool_preallocate(
      &logical_device->host_block_pools.large, 16));
  return iree_arena_block_pool_preallocate(
      &logical_device->host_block_pools.command_buffer, 16);
}

iree_status_t iree_hal_amdgpu_logical_device_create(
    iree_string_view_t identifier,
    const iree_hal_amdgpu_logical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_device = NULL;

  // Verify the topology is valid for a logical device.
  // This may have already been performed by the caller but doing it here
  // ensures all code paths must verify prior to creating a device.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_topology_verify(topology, libhsa),
      "verifying topology");

  // Verify the parameters prior to creating resources.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_amdgpu_logical_device_options_verify(options, libhsa, topology),
      "verifying logical device options");

  iree_hal_amdgpu_physical_device_options_t physical_device_options = {0};
  iree_hal_amdgpu_logical_device_translate_physical_options(
      options, topology, &physical_device_options);

  // Verify all GPU agents meet the required physical device options. Each
  // embedded physical device has the same layout because all physical devices
  // in one logical device share the same host-queue options.
  const iree_host_size_t physical_device_size =
      iree_hal_amdgpu_physical_device_calculate_size(&physical_device_options);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_amdgpu_logical_device_verify_physical_options(
          &physical_device_options, libhsa, topology),
      "verifying physical device options");

  // Allocate the logical device and all nested physical device data structures.
  iree_hal_amdgpu_logical_device_t* logical_device = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_logical_device_allocate_storage(
              identifier, topology, physical_device_size, host_allocator,
              &logical_device));
  iree_status_t status =
      iree_hal_amdgpu_logical_device_initialize_host_resources(
          logical_device, options, create_params->proactor_pool,
          host_allocator);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_logical_device_initialize_system_and_allocator(
        logical_device, options, libhsa, topology, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_logical_device_initialize_physical_devices(
        logical_device, topology, &physical_device_options, host_allocator);
  }

  // If requested then warmup pools that we expect to grow on the first usage of
  // the backend. The first use may need more than the warmup provides here but
  // that's ok - users can warmup if they want.
  if (iree_status_is_ok(status) && options->preallocate_pools) {
    status = iree_hal_amdgpu_logical_device_warmup_host_pools(logical_device);
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)logical_device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)logical_device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_logical_device_destroy(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_profile_counter_session_t* counter_session =
      logical_device->profiling.counter_session;
  iree_hal_amdgpu_profile_trace_session_t* trace_session =
      logical_device->profiling.trace_session;
  if (trace_session) {
    for (iree_host_size_t i = 0; i < logical_device->physical_device_count;
         ++i) {
      iree_hal_amdgpu_physical_device_t* physical_device =
          logical_device->physical_devices[i];
      for (iree_host_size_t j = 0; j < physical_device->host_queue_count; ++j) {
        iree_hal_amdgpu_host_queue_disable_profile_traces(
            &physical_device->host_queues[j]);
      }
    }
    logical_device->profiling.trace_session = NULL;
    iree_hal_amdgpu_profile_trace_session_free(trace_session);
  }
  if (counter_session) {
    for (iree_host_size_t i = 0; i < logical_device->physical_device_count;
         ++i) {
      iree_hal_amdgpu_physical_device_t* physical_device =
          logical_device->physical_devices[i];
      for (iree_host_size_t j = 0; j < physical_device->host_queue_count; ++j) {
        iree_hal_amdgpu_host_queue_disable_profile_counters(
            &physical_device->host_queues[j]);
      }
    }
    logical_device->profiling.counter_session = NULL;
    iree_hal_amdgpu_profile_counter_session_free(counter_session);
  }
  iree_hal_amdgpu_logical_device_reset_profile_options(logical_device);
  logical_device->profiling.session_id = 0;
  iree_hal_amdgpu_profile_event_streams_deinitialize(
      &logical_device->profiling.event_streams, logical_device->host_allocator);

  iree_hal_amdgpu_logical_device_deassign_frontier(logical_device);

  // Devices may hold allocations and need to be cleaned up first.
  for (iree_host_size_t i = 0; i < logical_device->physical_device_count; ++i) {
    iree_hal_amdgpu_physical_device_deinitialize(
        logical_device->physical_devices[i]);
  }

  iree_hal_allocator_release(logical_device->device_allocator);
  iree_hal_channel_provider_release(logical_device->channel_provider);

  // This may unload HSA; must come after all resources are released.
  iree_hal_amdgpu_system_free(logical_device->system);

  iree_hal_amdgpu_profile_metadata_deinitialize(
      &logical_device->profile_metadata);

  // Note that these may be used by other child data types and must be freed
  // last.
  iree_arena_block_pool_deinitialize(&logical_device->host_block_pools.small);
  iree_arena_block_pool_deinitialize(&logical_device->host_block_pools.large);
  iree_arena_block_pool_deinitialize(
      &logical_device->host_block_pools.command_buffer);

  iree_async_proactor_pool_release(logical_device->proactor_pool);

  iree_allocator_free(host_allocator, logical_device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_amdgpu_logical_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  return logical_device->identifier;
}

static iree_allocator_t iree_hal_amdgpu_logical_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  return logical_device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_amdgpu_logical_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  return logical_device->device_allocator;
}

static void iree_hal_amdgpu_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(logical_device->device_allocator);
  logical_device->device_allocator = new_allocator;
}

static void iree_hal_amdgpu_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(logical_device->channel_provider);
  logical_device->channel_provider = new_provider;
}

static iree_status_t iree_hal_amdgpu_logical_device_trim(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // Release pooled resources from each physical device. These may return items
  // back to the parent logical device pools.
  for (iree_host_size_t i = 0; i < logical_device->physical_device_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_physical_device_trim(
        logical_device->physical_devices[i]));
  }

  // Trim the allocator pools, if any.
  IREE_RETURN_IF_ERROR(
      iree_hal_allocator_trim(logical_device->device_allocator));

  // Trim host pools.
  iree_arena_block_pool_trim(&logical_device->host_block_pools.small);
  iree_arena_block_pool_trim(&logical_device->host_block_pools.large);
  iree_arena_block_pool_trim(&logical_device->host_block_pools.command_buffer);

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_logical_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    // NOTE: this is a fuzzy match and can allow a program to work with multiple
    // device implementations.
    *out_value =
        iree_string_view_match_pattern(logical_device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  iree_hal_amdgpu_system_t* system = logical_device->system;

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    bool is_supported = false;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_format_supported(
        &system->libhsa, system->topology.gpu_agents[0], key, &is_supported,
        /*out_isa=*/NULL));
    *out_value = is_supported ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.device"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      *out_value = system->topology.gpu_agent_count *
                   system->topology.gpu_agent_queue_count;
      return iree_ok_status();
    }
  } else if (iree_string_view_equal(category, IREE_SV("hal.dispatch"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      uint32_t compute_unit_count = 0;
      IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
          IREE_LIBHSA(&system->libhsa), system->topology.gpu_agents[0],
          (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
          &compute_unit_count));
      *out_value = compute_unit_count;
      return iree_ok_status();
    }
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_amdgpu_logical_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  memset(out_capabilities, 0, sizeof(*out_capabilities));

  if (logical_device->physical_device_count == 0) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "logical device has no physical devices (initialization incomplete)");
  }

  // A multi-GPU logical device is a composite HAL device. Generic HAL topology
  // has only one node for it, so do not expose a physical-device-0 identity as
  // though it represented the entire composite. Exact internal physical device
  // identity is reported through AMDGPU profile/device metadata and queue
  // affinity records.
  const bool is_composite_device = logical_device->physical_device_count > 1;
  iree_hal_amdgpu_physical_device_t* physical_device =
      logical_device->physical_devices[0];

  memset(out_capabilities->physical_device_uuid, 0,
         sizeof(out_capabilities->physical_device_uuid));
  if (!is_composite_device && physical_device->has_physical_device_uuid) {
    memcpy(out_capabilities->physical_device_uuid,
           physical_device->physical_device_uuid,
           sizeof(out_capabilities->physical_device_uuid));
    out_capabilities->has_physical_device_uuid = true;
  }

  // Report a NUMA affinity only when the composite has a single nearest host
  // node that fits the generic HAL uint8_t representation. Mixed-NUMA
  // composites intentionally leave the default 0 because generic topology
  // cannot express one logical device spanning multiple CPU NUMA nodes.
  uint32_t host_numa_node = physical_device->host_numa_node;
  bool has_representative_numa_node = host_numa_node <= UINT8_MAX;
  for (iree_host_size_t i = 1; i < logical_device->physical_device_count &&
                               has_representative_numa_node;
       ++i) {
    has_representative_numa_node =
        logical_device->physical_devices[i]->host_numa_node == host_numa_node;
  }
  if (has_representative_numa_node) {
    out_capabilities->numa_node = (uint8_t)host_numa_node;
  }

  // External handle types (DMA-BUF support from system info).
  if (logical_device->system->info.dmabuf_supported) {
    out_capabilities->buffer_export_types |=
        IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF;
    out_capabilities->buffer_import_types |=
        IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF;
  }

  // Memory-system capability flags are the intersection across the physical
  // devices in this logical device. SVM/pageable-memory facts are distinct
  // from peer-pool addressability; refine_topology_edge owns the latter.
  iree_hal_device_capability_bits_t memory_system_flags =
      iree_hal_amdgpu_select_memory_system_device_capability_flags(
          &physical_device->memory_system);
  for (iree_host_size_t i = 1; i < logical_device->physical_device_count; ++i) {
    memory_system_flags &=
        iree_hal_amdgpu_select_memory_system_device_capability_flags(
            &logical_device->physical_devices[i]->memory_system);
  }
  out_capabilities->flags |= memory_system_flags;

  // AMDGPU semaphores are native async timeline semaphores (not binary
  // emulation).
  out_capabilities->flags |= IREE_HAL_DEVICE_CAPABILITY_TIMELINE_SEMAPHORES;

  // Fine-grained memory provides host coherency without explicit flushes.
  // Coarse-grained memory requires fences, but the driver manages that
  // transparently.
  out_capabilities->flags |= IREE_HAL_DEVICE_CAPABILITY_HOST_COHERENT;

  // All AMDGPU devices support device-scope atomics. System-scope atomics are
  // supported on fine-grained memory when callers explicitly opt into
  // host-visible placement.
  out_capabilities->flags |= IREE_HAL_DEVICE_CAPABILITY_ATOMIC_SCOPE_DEVICE;
  out_capabilities->flags |= IREE_HAL_DEVICE_CAPABILITY_ATOMIC_SCOPE_SYSTEM;

  // All AMD GPUs support peer-to-peer DMA (through XGMI or PCIe). The actual
  // access mode for a specific GPU pair is determined by
  // refine_topology_edge — here we declare the capability in principle.
  out_capabilities->flags |= IREE_HAL_DEVICE_CAPABILITY_P2P_COPY;

  // Driver handle (HSA agent handle for same-driver refinement). Composite
  // devices intentionally leave this unset: a single HSA agent handle would
  // make generic topology alias detection treat a composite as one GPU.
  if (!is_composite_device) {
    out_capabilities->driver_device_handle =
        (uintptr_t)physical_device->device_agent.handle;
  }

  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_amdgpu_logical_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  return &logical_device->topology_info;
}

// Maximum number of HSA memory-pool link hops we will stack-allocate.
#define IREE_HAL_AMDGPU_MAX_TOPOLOGY_LINK_HOPS 16

typedef struct iree_hal_amdgpu_topology_edge_aggregate_t {
  // Physical capability facts produced by cross-pair aggregation.
  struct {
    // Positive capabilities conservatively intersected across every pair.
    iree_hal_topology_capability_t guaranteed;
    // Requirement bits unioned across pairs because any pair can constrain use.
    iree_hal_topology_capability_t required;
  } physical_capabilities;
  // Worst non-coherent read mode across all physical pairs.
  iree_hal_topology_interop_mode_t noncoherent_read_mode;
  // Worst non-coherent write mode across all physical pairs.
  iree_hal_topology_interop_mode_t noncoherent_write_mode;
  // Worst coherent read mode across all physical pairs.
  iree_hal_topology_interop_mode_t coherent_read_mode;
  // Worst coherent write mode across all physical pairs.
  iree_hal_topology_interop_mode_t coherent_write_mode;
  // Worst link class across all physical pairs.
  iree_hal_topology_link_class_t link_class;
  // Worst copy-cost class across all physical pairs.
  uint8_t copy_cost;
  // Worst latency class across all physical pairs.
  uint8_t latency_class;
  // Worst normalized NUMA distance across all physical pairs.
  uint8_t numa_distance;
} iree_hal_amdgpu_topology_edge_aggregate_t;

static iree_status_t iree_hal_amdgpu_query_physical_topology_edge(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_physical_device_t* source_physical_device,
    const iree_hal_amdgpu_physical_device_t* destination_physical_device,
    iree_hal_amdgpu_physical_topology_edge_t* out_physical_edge) {
  hsa_agent_t source_agent = source_physical_device->device_agent;
  hsa_agent_t destination_agent = destination_physical_device->device_agent;

  // Find both memory pool types on the destination agent. Not all devices
  // expose both pool types; missing pools are treated as NEVER_ALLOWED for that
  // pool kind, but an agent with no global pool at all is not a usable topology
  // node.
  hsa_amd_memory_pool_t dst_coarse_pool = {0};
  bool has_coarse_pool = iree_hal_amdgpu_try_find_coarse_global_memory_pool(
      libhsa, destination_agent, &dst_coarse_pool);
  hsa_amd_memory_pool_t dst_fine_pool = {0};
  bool has_fine_pool = iree_hal_amdgpu_try_find_fine_global_memory_pool(
      libhsa, destination_agent, &dst_fine_pool);
  if (!has_coarse_pool && !has_fine_pool) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "destination agent has neither coarse nor fine global memory pool");
  }

  iree_hal_amdgpu_physical_topology_edge_selection_t selection = {
      .memory_access =
          {
              .coarse = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED,
              .fine = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED,
          },
  };
  if (has_coarse_pool) {
    IREE_RETURN_IF_ERROR(iree_hsa_amd_agent_memory_pool_get_info(
        IREE_LIBHSA(libhsa), source_agent, dst_coarse_pool,
        HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
        &selection.memory_access.coarse));
  }
  if (has_fine_pool) {
    IREE_RETURN_IF_ERROR(iree_hsa_amd_agent_memory_pool_get_info(
        IREE_LIBHSA(libhsa), source_agent, dst_fine_pool,
        HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &selection.memory_access.fine));
  }

  // Query link hop count and topology. The link topology describes the
  // interconnect between agents and is the same regardless of pool granularity;
  // use whichever pool is present, preferring coarse-grained memory.
  hsa_amd_memory_pool_t link_query_pool =
      has_coarse_pool ? dst_coarse_pool : dst_fine_pool;
  uint32_t hop_count = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_agent_memory_pool_get_info(
      IREE_LIBHSA(libhsa), source_agent, link_query_pool,
      HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS, &hop_count));
  if (hop_count > IREE_HAL_AMDGPU_MAX_TOPOLOGY_LINK_HOPS) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "HSA reports %" PRIu32 " link hops between GPU agents (max %" PRIhsz
        ")",
        hop_count, (iree_host_size_t)IREE_HAL_AMDGPU_MAX_TOPOLOGY_LINK_HOPS);
  }

  hsa_amd_memory_pool_link_info_t
      link_hops[IREE_HAL_AMDGPU_MAX_TOPOLOGY_LINK_HOPS];
  memset(link_hops, 0, sizeof(link_hops[0]) * hop_count);
  if (hop_count > 0) {
    // The LINK_INFO query writes exactly hop_count entries into the caller's
    // buffer with no separate size parameter.
    IREE_RETURN_IF_ERROR(iree_hsa_amd_agent_memory_pool_get_info(
        IREE_LIBHSA(libhsa), source_agent, link_query_pool,
        HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO, link_hops));
  }

  selection.link.hops = link_hops;
  selection.link.count = hop_count;
  return iree_hal_amdgpu_select_physical_topology_edge(&selection,
                                                       out_physical_edge);
}

static void iree_hal_amdgpu_topology_edge_aggregate_initialize(
    iree_hal_topology_edge_t edge,
    iree_hal_amdgpu_topology_edge_aggregate_t* out_aggregate) {
  // Start physical facts at their best value so the aggregate can both upgrade
  // an imprecise base edge and then monotonically worsen with each pair.
  // Per-pair DISALLOWED_BY_DEFAULT access remains copy-only until an allocation
  // policy proves that direct access was explicitly granted.
  out_aggregate->physical_capabilities.guaranteed =
      IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY |
      IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT |
      IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_DEVICE |
      IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_SYSTEM;
  out_aggregate->physical_capabilities.required =
      IREE_HAL_TOPOLOGY_CAPABILITY_NONE;
  out_aggregate->noncoherent_read_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE;
  out_aggregate->noncoherent_write_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE;
  out_aggregate->coherent_read_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE;
  out_aggregate->coherent_write_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE;
  out_aggregate->link_class = IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE;
  out_aggregate->copy_cost = 0;
  out_aggregate->latency_class = 0;
  out_aggregate->numa_distance = iree_hal_topology_edge_numa_distance(edge.lo);
}

static void iree_hal_amdgpu_topology_edge_aggregate_include(
    const iree_hal_amdgpu_physical_topology_edge_t* physical_edge,
    iree_hal_amdgpu_topology_edge_aggregate_t* aggregate) {
  aggregate->physical_capabilities.guaranteed &=
      physical_edge->capabilities.guaranteed;
  aggregate->physical_capabilities.required |=
      physical_edge->capabilities.required;

  aggregate->noncoherent_read_mode = iree_max(
      aggregate->noncoherent_read_mode, physical_edge->modes.noncoherent_read);
  aggregate->noncoherent_write_mode =
      iree_max(aggregate->noncoherent_write_mode,
               physical_edge->modes.noncoherent_write);
  aggregate->coherent_read_mode = iree_max(aggregate->coherent_read_mode,
                                           physical_edge->modes.coherent_read);
  aggregate->coherent_write_mode = iree_max(
      aggregate->coherent_write_mode, physical_edge->modes.coherent_write);

  if (physical_edge->link.link_class > aggregate->link_class) {
    aggregate->link_class = physical_edge->link.link_class;
  }
  if (physical_edge->link.copy_cost > aggregate->copy_cost) {
    aggregate->copy_cost = physical_edge->link.copy_cost;
  }
  if (physical_edge->link.latency_class > aggregate->latency_class) {
    aggregate->latency_class = physical_edge->link.latency_class;
  }
  if (physical_edge->link.numa_distance > aggregate->numa_distance) {
    aggregate->numa_distance = physical_edge->link.numa_distance;
  }
}

static void iree_hal_amdgpu_topology_edge_apply_aggregate(
    const iree_hal_amdgpu_topology_edge_aggregate_t* aggregate,
    iree_hal_topology_edge_t* edge) {
  edge->lo = iree_hal_topology_edge_set_buffer_read_mode_noncoherent(
      edge->lo, aggregate->noncoherent_read_mode);
  edge->lo = iree_hal_topology_edge_set_buffer_write_mode_noncoherent(
      edge->lo, aggregate->noncoherent_write_mode);
  edge->lo = iree_hal_topology_edge_set_buffer_read_mode_coherent(
      edge->lo, aggregate->coherent_read_mode);
  edge->lo = iree_hal_topology_edge_set_buffer_write_mode_coherent(
      edge->lo, aggregate->coherent_write_mode);

  edge->lo =
      iree_hal_topology_edge_set_link_class(edge->lo, aggregate->link_class);
  edge->lo =
      iree_hal_topology_edge_set_copy_cost(edge->lo, aggregate->copy_cost);
  edge->lo = iree_hal_topology_edge_set_latency_class(edge->lo,
                                                      aggregate->latency_class);
  edge->lo = iree_hal_topology_edge_set_numa_distance(edge->lo,
                                                      aggregate->numa_distance);

  iree_hal_topology_capability_t capabilities =
      iree_hal_topology_edge_capability_flags(edge->lo);
  const iree_hal_topology_capability_t physical_guaranteed_capability_mask =
      IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY |
      IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT |
      IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_DEVICE |
      IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_SYSTEM;
  const iree_hal_topology_capability_t physical_required_capability_mask =
      IREE_HAL_TOPOLOGY_CAPABILITY_PEER_ACCESS_REQUIRES_GRANT;
  capabilities &= ~(physical_guaranteed_capability_mask |
                    physical_required_capability_mask);
  capabilities |= aggregate->physical_capabilities.guaranteed &
                  physical_guaranteed_capability_mask;
  capabilities |= aggregate->physical_capabilities.required &
                  physical_required_capability_mask;
  edge->lo =
      iree_hal_topology_edge_set_capability_flags(edge->lo, capabilities);
}

static iree_status_t iree_hal_amdgpu_logical_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  iree_hal_amdgpu_logical_device_t* src_logical =
      iree_hal_amdgpu_logical_device_cast(src_device);
  iree_hal_amdgpu_logical_device_t* dst_logical =
      iree_hal_amdgpu_logical_device_cast(dst_device);
  const iree_hal_amdgpu_libhsa_t* libhsa = &src_logical->system->libhsa;
  if (src_logical->physical_device_count == 0 ||
      dst_logical->physical_device_count == 0) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "cannot refine AMDGPU topology edge with an empty physical device set");
  }

  iree_hal_amdgpu_topology_edge_aggregate_t aggregate;
  iree_hal_amdgpu_topology_edge_aggregate_initialize(*edge, &aggregate);

  // A composite logical device has one generic HAL topology node but several
  // physical HSA agents. The generic edge must be valid for any source/dest
  // physical pair because the scheduler cannot encode a subset-specific edge.
  for (iree_host_size_t source_index = 0;
       source_index < src_logical->physical_device_count; ++source_index) {
    const iree_hal_amdgpu_physical_device_t* source_physical_device =
        src_logical->physical_devices[source_index];
    for (iree_host_size_t destination_index = 0;
         destination_index < dst_logical->physical_device_count;
         ++destination_index) {
      const iree_hal_amdgpu_physical_device_t* destination_physical_device =
          dst_logical->physical_devices[destination_index];
      iree_hal_amdgpu_physical_topology_edge_t physical_edge;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_query_physical_topology_edge(
          libhsa, source_physical_device, destination_physical_device,
          &physical_edge));
      iree_hal_amdgpu_topology_edge_aggregate_include(&physical_edge,
                                                      &aggregate);
    }
  }

  iree_hal_amdgpu_topology_edge_apply_aggregate(&aggregate, edge);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_logical_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  if (!topology_info) {
    iree_hal_amdgpu_logical_device_deassign_frontier(logical_device);
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amdgpu_system_t* system = logical_device->system;

  const uint8_t device_count = (uint8_t)system->topology.gpu_agent_count;
  const uint8_t queue_stride = (uint8_t)system->topology.gpu_agent_queue_count;
  const iree_host_size_t table_size =
      iree_hal_amdgpu_epoch_signal_table_size(device_count, queue_stride);
  iree_status_t status =
      iree_allocator_malloc(logical_device->host_allocator, table_size,
                            (void**)&logical_device->host_queue_epoch_table);
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_epoch_signal_table_initialize(
        logical_device->host_queue_epoch_table,
        iree_async_axis_session(topology_info->frontier.base_axis),
        iree_async_axis_machine(topology_info->frontier.base_axis),
        device_count, queue_stride);
  }

  for (iree_host_size_t device_ordinal = 0;
       device_ordinal < logical_device->physical_device_count &&
       iree_status_is_ok(status);
       ++device_ordinal) {
    const iree_host_size_t host_ordinal =
        system->topology.gpu_cpu_map[device_ordinal];
    status = iree_hal_amdgpu_physical_device_assign_frontier(
        base_device, system, logical_device->proactor,
        topology_info->frontier.tracker, topology_info->frontier.base_axis,
        logical_device->host_queue_epoch_table,
        &system->host_memory_pools[host_ordinal],
        logical_device->host_allocator,
        logical_device->physical_devices[device_ordinal]);
  }

  if (iree_status_is_ok(status)) {
    logical_device->topology_info = *topology_info;
    logical_device->frontier_tracker = topology_info->frontier.tracker;
    logical_device->axis = topology_info->frontier.base_axis;
    iree_async_frontier_tracker_retain(logical_device->frontier_tracker);
  } else {
    iree_hal_amdgpu_logical_device_deassign_frontier(logical_device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_logical_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU collective channels not yet implemented");
}

static iree_status_t iree_hal_amdgpu_logical_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_queue_affinity_t effective_queue_affinity = 0;
  iree_host_size_t device_ordinal = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_logical_device_normalize_command_buffer_affinity(
          logical_device, queue_affinity, &effective_queue_affinity,
          &device_ordinal));
  const iree_hal_amdgpu_physical_device_t* physical_device =
      logical_device->physical_devices[device_ordinal];
  return iree_hal_amdgpu_aql_command_buffer_create(
      iree_hal_device_allocator(base_device), mode, command_categories,
      effective_queue_affinity, binding_capacity, device_ordinal,
      physical_device->prepublished_kernarg_storage,
      &logical_device->profile_metadata,
      &logical_device->host_block_pools.command_buffer,
      &logical_device->host_block_pools.small, logical_device->host_allocator,
      out_command_buffer);
}

static iree_status_t iree_hal_amdgpu_logical_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU events not yet implemented");
}

static iree_status_t iree_hal_amdgpu_logical_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  return iree_hal_amdgpu_executable_cache_create(
      base_device, &logical_device->system->libhsa,
      &logical_device->system->topology, &logical_device->profile_metadata,
      identifier, iree_hal_device_host_allocator(base_device),
      out_executable_cache);
}

static iree_status_t iree_hal_amdgpu_logical_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_normalize(
      logical_device->queue_affinity_mask, queue_affinity, &queue_affinity));

  return iree_hal_file_from_handle(
      iree_hal_device_allocator(base_device), queue_affinity, access, handle,
      logical_device->proactor, iree_hal_device_host_allocator(base_device),
      out_file);
}

static iree_status_t iree_hal_amdgpu_logical_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  return iree_hal_amdgpu_semaphore_create(
      logical_device, logical_device->proactor, queue_affinity, initial_value,
      flags, logical_device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_amdgpu_logical_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  if (iree_hal_amdgpu_semaphore_isa(semaphore)) {
    return IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL;
  }
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_amdgpu_logical_device_query_queue_pool_backend(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_physical_device_t* physical_device = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_logical_device_select_queue_pool_physical_device(
          logical_device, queue_affinity, &physical_device));
  out_backend->slab_provider = physical_device->default_slab_provider;
  out_backend->notification = physical_device->default_pool_notification;
  out_backend->epoch_query = (iree_hal_pool_epoch_query_t){
      .fn = iree_hal_amdgpu_logical_device_query_pool_epoch,
      .user_data = logical_device,
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_host_queue(
      logical_device, queue_affinity, &queue));
  return queue->vtable->alloca(queue, wait_semaphore_list,
                               signal_semaphore_list, pool, params,
                               allocation_size, flags, out_buffer);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_host_queue(
      logical_device, queue_affinity, &queue));
  return queue->vtable->dealloca(queue, wait_semaphore_list,
                                 signal_semaphore_list, buffer, flags);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  // Match the HAL contract documented on iree_hal_command_buffer_fill_buffer
  // (1/2/4-byte patterns only) so queue_fill and command_buffer_fill accept
  // the same inputs across all backends. The device kernel itself supports an
  // 8-byte pattern path via iree_hal_amdgpu_device_buffer_fill_x8, but we
  // deliberately do not expose that here — callers writing 8-byte fills would
  // then be portable only to amdgpu.
  if (IREE_UNLIKELY(pattern_length != 1 && pattern_length != 2 &&
                    pattern_length != 4)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fill patterns must be 1, 2, or 4 bytes (got %" PRIhsz ")",
        pattern_length);
  }
  if (IREE_UNLIKELY(!pattern)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fill pattern pointer is required");
  }
  uint64_t pattern_bits = 0;
  memcpy(&pattern_bits, pattern, pattern_length);

  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_host_queue(
      logical_device, queue_affinity, &queue));
  return queue->vtable->fill(queue, wait_semaphore_list, signal_semaphore_list,
                             target_buffer, target_offset, length, pattern_bits,
                             pattern_length, flags);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_host_queue(
      logical_device, queue_affinity, &queue));
  return queue->vtable->update(
      queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
      source_offset, target_buffer, target_offset, length, flags);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_host_queue(
      logical_device, queue_affinity, &queue));
  return queue->vtable->copy(queue, wait_semaphore_list, signal_semaphore_list,
                             source_buffer, source_offset, target_buffer,
                             target_offset, length, flags);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_host_queue(
      logical_device, queue_affinity, &queue));
  return queue->vtable->read(queue, wait_semaphore_list, signal_semaphore_list,
                             source_file, source_offset, target_buffer,
                             target_offset, length, flags);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_host_queue(
      logical_device, queue_affinity, &queue));
  return queue->vtable->write(queue, wait_semaphore_list, signal_semaphore_list,
                              source_buffer, source_offset, target_file,
                              target_offset, length, flags);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_host_queue(
      logical_device, queue_affinity, &queue));
  return queue->vtable->host_call(queue, wait_semaphore_list,
                                  signal_semaphore_list, call, args, flags);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_dispatch(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_host_queue(
      logical_device, queue_affinity, &queue));
  return queue->vtable->dispatch(
      queue, wait_semaphore_list, signal_semaphore_list, executable,
      export_ordinal, config, constants, bindings, flags);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_host_queue(
      logical_device, queue_affinity, &queue));
  return queue->vtable->execute(queue, wait_semaphore_list,
                                signal_semaphore_list, command_buffer,
                                binding_table, flags);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_normalize(
      logical_device->queue_affinity_mask, queue_affinity, &queue_affinity));

  IREE_HAL_FOR_QUEUE_AFFINITY(queue_affinity) {
    iree_hal_amdgpu_virtual_queue_t* queue = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_queue_from_ordinal(
        logical_device, queue_ordinal, &queue));
    IREE_RETURN_IF_ERROR(queue->vtable->flush(queue));
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_logical_device_verify_queue_device_profiling_supported(
    iree_hal_amdgpu_logical_device_t* logical_device) {
  for (iree_host_size_t i = 0; i < logical_device->physical_device_count; ++i) {
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[i];
    if (iree_hal_amdgpu_vendor_packet_capabilities_support_timestamp_range(
            physical_device->vendor_packet_capabilities)) {
      continue;
    }
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU queue operation profiling requires PM4 timestamp range "
        "support on physical device %" PRIhsz,
        physical_device->device_ordinal);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_logical_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_device_profiling_options_t resolved_options =
      iree_hal_amdgpu_logical_device_resolve_profiling_options(options);

  if (iree_hal_device_profiling_options_requests_data(
          &resolved_options,
          IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU profiling does not produce host execution events");
  }
  if (resolved_options.data_families == IREE_HAL_DEVICE_PROFILING_DATA_NONE) {
    return iree_ok_status();
  }
  if (!logical_device->frontier_tracker) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU profiling requires an assigned device topology");
  }
  if (logical_device->profiling.options.data_families !=
      IREE_HAL_DEVICE_PROFILING_DATA_NONE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot nest AMDGPU profile captures");
  }
  if (iree_hal_device_profiling_options_requests_data(
          &resolved_options,
          IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_logical_device_verify_queue_device_profiling_supported(
            logical_device));
  }

  bool sink_session_begun = false;
  bool hsa_profiling_enabled = false;
  bool counter_profiling_enabled = false;
  bool counter_ranges_started = false;
  bool trace_profiling_enabled = false;
  iree_hal_device_profiling_options_t session_options = {0};
  iree_hal_device_profiling_options_storage_t* options_storage = NULL;
  iree_hal_amdgpu_profile_counter_session_t* counter_session = NULL;
  iree_hal_amdgpu_profile_trace_session_t* trace_session = NULL;
  iree_hal_amdgpu_profile_device_metrics_session_t* device_metrics_session =
      NULL;
  iree_status_t status = iree_hal_device_profiling_options_clone(
      &resolved_options, logical_device->host_allocator, &session_options,
      &options_storage);
  iree_hal_profile_sink_t* sink = session_options.sink;
  uint64_t session_id = 0;
  iree_hal_profile_chunk_metadata_t metadata = {0};
  if (iree_status_is_ok(status)) {
    session_id = logical_device->next_profile_session_id++;
    metadata = iree_hal_amdgpu_logical_device_profile_session_metadata(
        logical_device, session_id);
    logical_device->profiling.next_clock_correlation_sample_id = 1;
    memset(&logical_device->profiling.metadata_cursor, 0,
           sizeof(logical_device->profiling.metadata_cursor));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_counter_session_allocate(
        logical_device, &session_options, logical_device->host_allocator,
        &counter_session);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_trace_session_allocate(
        logical_device, &session_options, logical_device->host_allocator,
        &trace_session);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_device_metrics_session_allocate(
        logical_device, &session_options, logical_device->host_allocator,
        &device_metrics_session);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_sink_begin_session(sink, &metadata);
    sink_session_begun = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_device_profiling_options_requests_data(
          &session_options, IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS)) {
    status = iree_hal_amdgpu_profile_event_streams_ensure_queue_storage(
        &logical_device->profiling.event_streams,
        IREE_HAL_AMDGPU_LOGICAL_DEVICE_PROFILE_QUEUE_EVENT_CAPACITY,
        logical_device->host_allocator);
    if (iree_status_is_ok(status)) {
      iree_hal_amdgpu_profile_event_streams_clear_queue(
          &logical_device->profiling.event_streams);
    }
  }
  if (iree_status_is_ok(status) &&
      iree_hal_device_profiling_options_requests_data(
          &session_options, IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS)) {
    status = iree_hal_amdgpu_profile_event_streams_ensure_memory_storage(
        &logical_device->profiling.event_streams,
        IREE_HAL_AMDGPU_LOGICAL_DEVICE_PROFILE_MEMORY_EVENT_CAPACITY,
        logical_device->host_allocator);
    if (iree_status_is_ok(status)) {
      iree_hal_amdgpu_profile_event_streams_clear_memory(
          &logical_device->profiling.event_streams);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_logical_device_write_profile_metadata(
        logical_device, sink, session_id, session_options.data_families);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_counter_session_write_metadata(
        counter_session, sink, session_id, logical_device->identifier);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_device_metrics_session_write_metadata(
        device_metrics_session, sink, session_id, logical_device->identifier);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_logical_device_profiling_needs_hsa_timestamps(
          session_options.data_families)) {
    status = iree_hal_amdgpu_logical_device_set_hsa_profiling_enabled(
        logical_device, true);
    hsa_profiling_enabled = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_logical_device_set_counter_profiling_enabled(
        logical_device, counter_session, true);
    counter_profiling_enabled = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_logical_device_start_profile_counter_ranges(
        logical_device, counter_session);
    counter_ranges_started = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_logical_device_set_trace_profiling_enabled(
        logical_device, trace_session, true);
    trace_profiling_enabled = iree_status_is_ok(status);
  }

  if (iree_status_is_ok(status)) {
    logical_device->profiling.options = session_options;
    logical_device->profiling.options_storage = options_storage;
    logical_device->profiling.session_id = session_id;
    logical_device->profiling.counter_session = counter_session;
    logical_device->profiling.trace_session = trace_session;
    logical_device->profiling.device_metrics_session = device_metrics_session;
    iree_hal_amdgpu_logical_device_set_queue_profiling_enabled(
        logical_device,
        iree_hal_amdgpu_logical_device_queue_profile_flags(&session_options));
  } else {
    if (trace_profiling_enabled) {
      status = iree_status_join(
          status, iree_hal_amdgpu_logical_device_set_trace_profiling_enabled(
                      logical_device, trace_session, false));
    }
    if (counter_ranges_started) {
      status = iree_status_join(
          status,
          iree_hal_amdgpu_logical_device_flush_profile_counter_ranges(
              logical_device, counter_session, /*sink=*/NULL, /*session_id=*/0,
              IREE_HAL_AMDGPU_PROFILE_COUNTER_RANGE_FLUSH_FLAG_NONE));
    }
    if (counter_profiling_enabled) {
      status = iree_status_join(
          status, iree_hal_amdgpu_logical_device_set_counter_profiling_enabled(
                      logical_device, counter_session, false));
    }
    if (hsa_profiling_enabled) {
      status = iree_status_join(
          status, iree_hal_amdgpu_logical_device_set_hsa_profiling_enabled(
                      logical_device, false));
    }
    if (sink_session_begun) {
      status = iree_status_join(
          status, iree_hal_profile_sink_end_session(sink, &metadata,
                                                    iree_status_code(status)));
    }
    logical_device->profiling.next_clock_correlation_sample_id = 0;
    memset(&logical_device->profiling.metadata_cursor, 0,
           sizeof(logical_device->profiling.metadata_cursor));
    iree_hal_device_profiling_options_storage_free(
        options_storage, logical_device->host_allocator);
    iree_hal_amdgpu_profile_counter_session_free(counter_session);
    iree_hal_amdgpu_profile_trace_session_free(trace_session);
    iree_hal_amdgpu_profile_device_metrics_session_free(device_metrics_session);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_logical_device_profiling_flush(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  const iree_hal_device_profiling_options_t* options =
      &logical_device->profiling.options;
  if (options->data_families == IREE_HAL_DEVICE_PROFILING_DATA_NONE) {
    return iree_ok_status();
  }
  iree_hal_profile_sink_t* sink = options->sink;
  const bool emit_executable_artifacts =
      iree_hal_amdgpu_logical_device_profile_needs_executable_artifacts(
          options->data_families);
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_logical_device_flush_profile_counter_ranges(
          logical_device, logical_device->profiling.counter_session, sink,
          logical_device->profiling.session_id,
          IREE_HAL_AMDGPU_PROFILE_COUNTER_RANGE_FLUSH_FLAG_RESTART));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_metadata_write(
      &logical_device->profile_metadata, sink,
      logical_device->profiling.session_id, logical_device->identifier,
      emit_executable_artifacts, &logical_device->profiling.metadata_cursor));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_write_profile_events(
      logical_device, sink, logical_device->profiling.session_id));
  if (iree_hal_amdgpu_logical_device_profiling_needs_clock_correlations(
          options->data_families)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_logical_device_write_profile_clock_correlations(
            logical_device, sink, logical_device->profiling.session_id));
  }
  return iree_hal_amdgpu_profile_device_metrics_session_sample_and_write(
      logical_device->profiling.device_metrics_session, sink,
      logical_device->profiling.session_id, logical_device->identifier);
}

static iree_status_t iree_hal_amdgpu_logical_device_profiling_end(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  iree_status_t status = iree_ok_status();
  const iree_hal_device_profiling_data_families_t data_families =
      logical_device->profiling.options.data_families;
  if (data_families == IREE_HAL_DEVICE_PROFILING_DATA_NONE) {
    return iree_ok_status();
  }

  iree_hal_profile_sink_t* sink = logical_device->profiling.options.sink;
  iree_hal_amdgpu_profile_counter_session_t* counter_session =
      logical_device->profiling.counter_session;
  iree_hal_amdgpu_profile_trace_session_t* trace_session =
      logical_device->profiling.trace_session;
  iree_hal_amdgpu_profile_device_metrics_session_t* device_metrics_session =
      logical_device->profiling.device_metrics_session;
  const uint64_t session_id = logical_device->profiling.session_id;
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_amdgpu_logical_device_profile_session_metadata(logical_device,
                                                              session_id);
  const bool emit_executable_artifacts =
      iree_hal_amdgpu_logical_device_profile_needs_executable_artifacts(
          data_families);

  status = iree_hal_amdgpu_logical_device_flush_profile_counter_ranges(
      logical_device, counter_session, sink, session_id,
      IREE_HAL_AMDGPU_PROFILE_COUNTER_RANGE_FLUSH_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_metadata_write(
        &logical_device->profile_metadata, sink, session_id,
        logical_device->identifier, emit_executable_artifacts,
        &logical_device->profiling.metadata_cursor);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_logical_device_write_profile_events(
        logical_device, sink, session_id);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_logical_device_profiling_needs_clock_correlations(
          data_families)) {
    status = iree_hal_amdgpu_logical_device_write_profile_clock_correlations(
        logical_device, sink, session_id);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_device_metrics_session_sample_and_write(
        device_metrics_session, sink, session_id, logical_device->identifier);
  }
  status = iree_status_join(
      status, iree_hal_amdgpu_logical_device_set_trace_profiling_enabled(
                  logical_device, trace_session, false));
  status = iree_status_join(
      status, iree_hal_amdgpu_logical_device_set_counter_profiling_enabled(
                  logical_device, counter_session, false));
  if (iree_hal_amdgpu_logical_device_profiling_needs_hsa_timestamps(
          data_families)) {
    status = iree_status_join(
        status, iree_hal_amdgpu_logical_device_set_hsa_profiling_enabled(
                    logical_device, false));
  }
  status =
      iree_status_join(status, iree_hal_profile_sink_end_session(
                                   sink, &metadata, iree_status_code(status)));

  iree_hal_amdgpu_logical_device_reset_profile_options(logical_device);
  logical_device->profiling.session_id = 0;
  logical_device->profiling.next_clock_correlation_sample_id = 0;
  memset(&logical_device->profiling.metadata_cursor, 0,
         sizeof(logical_device->profiling.metadata_cursor));
  logical_device->profiling.counter_session = NULL;
  logical_device->profiling.trace_session = NULL;
  logical_device->profiling.device_metrics_session = NULL;
  iree_hal_amdgpu_logical_device_set_queue_profiling_enabled(
      logical_device, IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_FLAG_NONE);
  iree_hal_amdgpu_profile_counter_session_free(counter_session);
  iree_hal_amdgpu_profile_trace_session_free(trace_session);
  iree_hal_amdgpu_profile_device_metrics_session_free(device_metrics_session);
  return status;
}

static iree_status_t iree_hal_amdgpu_logical_device_external_capture_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_external_capture_options_t* options) {
  (void)base_device;
  (void)options;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU external capture not implemented");
}

static iree_status_t iree_hal_amdgpu_logical_device_external_capture_end(
    iree_hal_device_t* base_device) {
  (void)base_device;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU external capture not implemented");
}

static const iree_hal_device_vtable_t iree_hal_amdgpu_logical_device_vtable = {
    .destroy = iree_hal_amdgpu_logical_device_destroy,
    .id = iree_hal_amdgpu_logical_device_id,
    .host_allocator = iree_hal_amdgpu_logical_device_host_allocator,
    .device_allocator = iree_hal_amdgpu_logical_device_allocator,
    .replace_device_allocator = iree_hal_amdgpu_replace_device_allocator,
    .replace_channel_provider = iree_hal_amdgpu_replace_channel_provider,
    .trim = iree_hal_amdgpu_logical_device_trim,
    .query_i64 = iree_hal_amdgpu_logical_device_query_i64,
    .query_capabilities = iree_hal_amdgpu_logical_device_query_capabilities,
    .topology_info = iree_hal_amdgpu_logical_device_topology_info,
    .refine_topology_edge = iree_hal_amdgpu_logical_device_refine_topology_edge,
    .assign_topology_info = iree_hal_amdgpu_logical_device_assign_topology_info,
    .create_channel = iree_hal_amdgpu_logical_device_create_channel,
    .create_command_buffer =
        iree_hal_amdgpu_logical_device_create_command_buffer,
    .create_event = iree_hal_amdgpu_logical_device_create_event,
    .create_executable_cache =
        iree_hal_amdgpu_logical_device_create_executable_cache,
    .import_file = iree_hal_amdgpu_logical_device_import_file,
    .create_semaphore = iree_hal_amdgpu_logical_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_amdgpu_logical_device_query_semaphore_compatibility,
    .query_queue_pool_backend =
        iree_hal_amdgpu_logical_device_query_queue_pool_backend,
    .queue_alloca = iree_hal_amdgpu_logical_device_queue_alloca,
    .queue_dealloca = iree_hal_amdgpu_logical_device_queue_dealloca,
    .queue_fill = iree_hal_amdgpu_logical_device_queue_fill,
    .queue_update = iree_hal_amdgpu_logical_device_queue_update,
    .queue_copy = iree_hal_amdgpu_logical_device_queue_copy,
    .queue_read = iree_hal_amdgpu_logical_device_queue_read,
    .queue_write = iree_hal_amdgpu_logical_device_queue_write,
    .queue_host_call = iree_hal_amdgpu_logical_device_queue_host_call,
    .queue_dispatch = iree_hal_amdgpu_logical_device_queue_dispatch,
    .queue_execute = iree_hal_amdgpu_logical_device_queue_execute,
    .queue_flush = iree_hal_amdgpu_logical_device_queue_flush,
    .profiling_begin = iree_hal_amdgpu_logical_device_profiling_begin,
    .profiling_flush = iree_hal_amdgpu_logical_device_profiling_flush,
    .profiling_end = iree_hal_amdgpu_logical_device_profiling_end,
    .external_capture_begin =
        iree_hal_amdgpu_logical_device_external_capture_begin,
    .external_capture_end = iree_hal_amdgpu_logical_device_external_capture_end,
};
