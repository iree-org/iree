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
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/epoch_signal_table.h"
#include "iree/hal/drivers/amdgpu/util/notification_ring.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
#include "iree/hal/utils/file_registry.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Returns the queue for a flattened logical queue ordinal.
static iree_status_t iree_hal_amdgpu_logical_device_queue_from_ordinal(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_host_size_t queue_ordinal,
    iree_hal_amdgpu_virtual_queue_t** out_queue) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(out_queue);
  *out_queue = NULL;

  const iree_host_size_t per_device_queue_count =
      logical_device->system->topology.gpu_agent_queue_count;
  const iree_host_size_t physical_device_ordinal =
      queue_ordinal / per_device_queue_count;
  const iree_host_size_t host_queue_ordinal =
      queue_ordinal % per_device_queue_count;

  if (IREE_UNLIKELY(physical_device_ordinal >=
                    logical_device->physical_device_count)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "queue affinity ordinal %" PRIhsz
                            " maps to invalid physical device ordinal %" PRIhsz,
                            queue_ordinal, physical_device_ordinal);
  }

  iree_hal_amdgpu_physical_device_t* physical_device =
      logical_device->physical_devices[physical_device_ordinal];
  if (IREE_UNLIKELY(host_queue_ordinal >= physical_device->host_queue_count)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "queue affinity ordinal %" PRIhsz
                            " maps to invalid host queue ordinal "
                            "%" PRIhsz " on physical device %" PRIhsz,
                            queue_ordinal, host_queue_ordinal,
                            physical_device_ordinal);
  }

  *out_queue = &physical_device->host_queues[host_queue_ordinal].base;
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

  out_options->preallocate_pools = 1;
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_logical_device_options_parse(
    iree_hal_amdgpu_logical_device_options_t* options,
    iree_string_pair_list_t params) {
  IREE_ASSERT_ARGUMENT(options);
  if (!params.count) return iree_ok_status();  // no-op
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): parameters.

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_logical_device_options_verify(
    const iree_hal_amdgpu_logical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): verify that the parameters are within expected ranges and
  // any requested features are supported.

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
      !iree_host_size_is_power_of_two(options->host_queues.kernarg_capacity)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "host queue AQL, notification, and kernarg capacities must all "
                "be powers of two (got aql=%u, notification=%u, "
                "kernarg_blocks=%u)",
                options->host_queues.aql_capacity,
                options->host_queues.notification_capacity,
                options->host_queues.kernarg_capacity));
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
    iree_hal_device_profiling_mode_t mode) {
  return iree_any_bit_set(
      mode, IREE_HAL_DEVICE_PROFILING_MODE_QUEUE_OPERATIONS |
                IREE_HAL_DEVICE_PROFILING_MODE_DISPATCH_COUNTERS |
                IREE_HAL_DEVICE_PROFILING_MODE_EXECUTABLE_COUNTERS);
}

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

static iree_status_t iree_hal_amdgpu_logical_device_write_profile_metadata(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_profile_sink_t* sink, uint64_t session_id) {
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_write_profile_devices(
      logical_device, sink, session_id));
  return iree_hal_amdgpu_logical_device_write_profile_queues(logical_device,
                                                             sink, session_id);
}

static iree_status_t iree_hal_amdgpu_logical_device_write_profile_events(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_profile_sink_t* sink, uint64_t session_id) {
  if (!sink) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
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

  iree_hal_queue_affinity_and_into(queue_affinity,
                                   logical_device->queue_affinity_mask);
  if (iree_hal_queue_affinity_is_empty(queue_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid queue affinity bits specified");
  }

  const iree_host_size_t queue_ordinal =
      iree_hal_queue_affinity_find_first_set(queue_affinity);
  return iree_hal_amdgpu_logical_device_queue_from_ordinal(
      logical_device, queue_ordinal, out_queue);
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

  iree_hal_queue_affinity_and_into(queue_affinity,
                                   logical_device->queue_affinity_mask);
  if (iree_hal_queue_affinity_is_empty(queue_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid queue affinity bits specified");
  }

  const iree_host_size_t queue_ordinal =
      iree_hal_queue_affinity_find_first_set(queue_affinity);
  const iree_host_size_t per_device_queue_count =
      logical_device->system->topology.gpu_agent_queue_count;
  const iree_host_size_t physical_device_ordinal =
      queue_ordinal / per_device_queue_count;
  *out_physical_device =
      logical_device->physical_devices[physical_device_ordinal];
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

  const bool is_any_affinity = iree_hal_queue_affinity_is_any(queue_affinity);
  iree_hal_queue_affinity_t effective_affinity =
      is_any_affinity ? logical_device->queue_affinity_mask : queue_affinity;
  iree_hal_queue_affinity_and_into(effective_affinity,
                                   logical_device->queue_affinity_mask);
  if (iree_hal_queue_affinity_is_empty(effective_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid queue affinity bits specified");
  }

  const iree_host_size_t queue_ordinal =
      iree_hal_queue_affinity_find_first_set(effective_affinity);
  const iree_host_size_t per_device_queue_count =
      logical_device->system->topology.gpu_agent_queue_count;
  const iree_host_size_t device_ordinal =
      queue_ordinal / per_device_queue_count;
  iree_hal_queue_affinity_t device_queue_affinity = 0;
  for (iree_host_size_t queue_index = 0; queue_index < per_device_queue_count;
       ++queue_index) {
    const iree_host_size_t device_queue_ordinal =
        device_ordinal * per_device_queue_count + queue_index;
    iree_hal_queue_affinity_or_into(device_queue_affinity,
                                    ((iree_hal_queue_affinity_t)1)
                                        << device_queue_ordinal);
  }
  iree_hal_queue_affinity_and_into(device_queue_affinity,
                                   logical_device->queue_affinity_mask);

  if (!is_any_affinity &&
      iree_any_bit_set(effective_affinity, ~device_queue_affinity)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU command buffers must target one physical device; queue "
        "affinity 0x%" PRIx64 " spans multiple physical devices",
        queue_affinity);
  }

  iree_hal_queue_affinity_t selected_affinity = device_queue_affinity;
  if (!is_any_affinity) {
    iree_hal_queue_affinity_and_into(selected_affinity, effective_affinity);
  }
  *out_queue_affinity = selected_affinity;
  *out_device_ordinal = device_ordinal;
  return iree_ok_status();
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
    // Previous status was not OK; drop our new status.
    IREE_IGNORE_ERROR(status);
  }

  IREE_TRACE_ZONE_END(z0);
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

  // Copy options relevant during construction.
  //
  // TODO(benvanik): maybe expose these on the public API? feels like too much
  // churn for too little benefit - option parsing is still possible, though.
  iree_hal_amdgpu_physical_device_options_t physical_device_options = {0};
  iree_hal_amdgpu_physical_device_options_initialize(&physical_device_options);
  physical_device_options.device_block_pools.small.block_size =
      options->device_block_pools.small.block_size;
  physical_device_options.device_block_pools.small.initial_capacity =
      options->device_block_pools.small.initial_capacity;
  physical_device_options.device_block_pools.large.block_size =
      options->device_block_pools.large.block_size;
  physical_device_options.device_block_pools.large.initial_capacity =
      options->device_block_pools.large.initial_capacity;
  physical_device_options.default_pool.range_length =
      options->default_pool.range_length;
  physical_device_options.default_pool.alignment =
      options->default_pool.alignment;
  physical_device_options.default_pool.frontier_capacity =
      options->default_pool.frontier_capacity;
  physical_device_options.host_block_pool_initial_capacity =
      options->preallocate_pools ? 16 : 0;
  physical_device_options.host_queue_count = topology->gpu_agent_queue_count;
  physical_device_options.host_queue_aql_capacity =
      options->host_queues.aql_capacity;
  physical_device_options.host_queue_notification_capacity =
      options->host_queues.notification_capacity;
  physical_device_options.host_queue_kernarg_capacity =
      options->host_queues.kernarg_capacity;
  physical_device_options.force_wait_barrier_defer =
      options->force_wait_barrier_defer;

  // Verify all GPU agents meet the required physical device options. Each
  // embedded physical device has the same layout because all physical devices
  // in one logical device share the same host-queue options.
  const iree_host_size_t physical_device_size =
      iree_hal_amdgpu_physical_device_calculate_size(&physical_device_options);
  for (iree_host_size_t i = 0; i < topology->gpu_agent_count; ++i) {
    hsa_agent_t gpu_agent = topology->gpu_agents[i];
    hsa_agent_t cpu_agent = topology->cpu_agents[topology->gpu_cpu_map[i]];
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_amdgpu_physical_device_options_verify(
            &physical_device_options, libhsa, cpu_agent, gpu_agent),
        "verifying GPU agent %" PRIhsz " meets required options", i);
  }

  // Allocate the logical device and all nested physical device data structures.
  iree_hal_amdgpu_logical_device_t* logical_device = NULL;
  iree_host_size_t physical_device_data_offset = 0;
  iree_host_size_t identifier_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(*logical_device), &total_size,
              IREE_STRUCT_FIELD(topology->gpu_agent_count,
                                iree_hal_amdgpu_physical_device_t*, NULL),
              IREE_STRUCT_ARRAY_FIELD_ALIGNED(
                  topology->gpu_agent_count, physical_device_size, uint8_t,
                  iree_max_align_t, &physical_device_data_offset),
              IREE_STRUCT_FIELD(identifier.size, char, &identifier_offset)));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                (void**)&logical_device));
  memset(logical_device, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_amdgpu_logical_device_vtable,
                               &logical_device->resource);
  iree_string_view_append_to_buffer(identifier, &logical_device->identifier,
                                    (char*)logical_device + identifier_offset);
  logical_device->host_allocator = host_allocator;
  logical_device->failure_status = IREE_ATOMIC_VAR_INIT(0);

  // Retain the proactor pool. A proactor is acquired after block pools are
  // initialized so failure cleanup can always deinitialize the pools.
  logical_device->proactor_pool = create_params->proactor_pool;
  iree_async_proactor_pool_retain(logical_device->proactor_pool);
  iree_atomic_store(&logical_device->epoch, 0, iree_memory_order_relaxed);
  logical_device->next_profile_session_id = 1;

  // Setup physical device table.
  // We need to initialize this first so that any failure cleanup has a valid
  // table.
  logical_device->physical_device_count = topology->gpu_agent_count;
  uint8_t* physical_device_base =
      (uint8_t*)logical_device + physical_device_data_offset;
  for (iree_host_size_t i = 0, queue_index = 0;
       i < logical_device->physical_device_count; ++i) {
    logical_device->physical_devices[i] =
        (iree_hal_amdgpu_physical_device_t*)physical_device_base;
    physical_device_base += physical_device_size;
    for (iree_host_size_t j = 0; j < topology->gpu_agent_queue_count;
         ++j, ++queue_index) {
      iree_hal_queue_affinity_or_into(logical_device->queue_affinity_mask,
                                      1ull << queue_index);
    }
  }

  // Block pools used by subsequent data structures.
  iree_arena_block_pool_initialize(options->host_block_pools.small.block_size,
                                   host_allocator,
                                   &logical_device->host_block_pools.small);
  iree_arena_block_pool_initialize(options->host_block_pools.large.block_size,
                                   host_allocator,
                                   &logical_device->host_block_pools.large);
  iree_status_t status = iree_hal_amdgpu_aql_program_block_pool_initialize(
      options->host_block_pools.command_buffer.usable_block_size,
      host_allocator, &logical_device->host_block_pools.command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_get(logical_device->proactor_pool, 0,
                                          &logical_device->proactor);
  }

  // Instantiate system container for agents used by the logical device. Loads
  // fixed per-agent resources like the device library.
  iree_hal_amdgpu_system_options_t system_options = {
      .trace_execution = options->trace_execution,
      .exclusive_execution = options->exclusive_execution,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_system_allocate(libhsa, topology, system_options,
                                             host_allocator,
                                             &logical_device->system);
  }
  iree_hal_amdgpu_system_t* system = logical_device->system;

  // Create the device allocator backed by HSA memory pools.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_allocator_create(
        logical_device, &system->libhsa, &system->topology, host_allocator,
        &logical_device->device_allocator);
  }

  // Initialize physical devices for each GPU agent in the topology.
  // Their order matches the original but each may represent more than one
  // logical queue affinity bit.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t device_ordinal = 0;
         device_ordinal < logical_device->physical_device_count;
         ++device_ordinal) {
      const iree_host_size_t host_ordinal =
          topology->gpu_cpu_map[device_ordinal];
      status = iree_hal_amdgpu_physical_device_initialize(
          (iree_hal_device_t*)logical_device, system, &physical_device_options,
          logical_device->proactor, host_ordinal,
          &system->host_memory_pools[host_ordinal], device_ordinal,
          host_allocator, logical_device->physical_devices[device_ordinal]);
      if (!iree_status_is_ok(status)) break;
    }
  }

  // If requested then warmup pools that we expect to grow on the first usage of
  // the backend. The first use may need more than the warmup provides here but
  // that's ok - users can warmup if they want.
  if (options->preallocate_pools) {
    if (iree_status_is_ok(status)) {
      status = iree_arena_block_pool_preallocate(
          &logical_device->host_block_pools.small, 16);
    }
    if (iree_status_is_ok(status)) {
      status = iree_arena_block_pool_preallocate(
          &logical_device->host_block_pools.large, 16);
    }
    if (iree_status_is_ok(status)) {
      status = iree_arena_block_pool_preallocate(
          &logical_device->host_block_pools.command_buffer, 16);
    }
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

  iree_hal_profile_sink_release(logical_device->profiling.sink);
  logical_device->profiling.sink = NULL;
  logical_device->profiling.mode = IREE_HAL_DEVICE_PROFILING_MODE_NONE;
  logical_device->profiling.session_id = 0;

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

  // For single-GPU logical devices, query the first physical device.
  // TODO(multi-gpu): for multi-GPU logical devices, aggregate capabilities from
  // all physical devices (take intersection of supported features, lowest
  // common denominator for limits, etc.).
  if (logical_device->physical_device_count == 0) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "logical device has no physical devices (initialization incomplete)");
  }

  iree_hal_amdgpu_physical_device_t* physical_device =
      logical_device->physical_devices[0];
  hsa_agent_t gpu_agent = physical_device->device_agent;
  const iree_hal_amdgpu_libhsa_t* libhsa = &logical_device->system->libhsa;

  memset(out_capabilities->physical_device_uuid, 0,
         sizeof(out_capabilities->physical_device_uuid));
  if (physical_device->has_physical_device_uuid) {
    memcpy(out_capabilities->physical_device_uuid,
           physical_device->physical_device_uuid,
           sizeof(out_capabilities->physical_device_uuid));
    out_capabilities->has_physical_device_uuid = true;
  }

  // Query NUMA node from HSA.
  uint32_t numa_node;
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), gpu_agent, HSA_AGENT_INFO_NODE, &numa_node));
  out_capabilities->numa_node = (uint8_t)numa_node;

  // External handle types (DMA-BUF support from system info).
  if (logical_device->system->info.dmabuf_supported) {
    out_capabilities->buffer_export_types |=
        IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF;
    out_capabilities->buffer_import_types |=
        IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF;
  }

  // Capability flags.
  if (logical_device->system->info.svm_accessible_by_default) {
    out_capabilities->flags |= IREE_HAL_DEVICE_CAPABILITY_UNIFIED_MEMORY;
  }

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

  // Peer addressability depends on whether SVM is enabled (large BAR / XGMI
  // provides load/store access to peer memory without explicit grants).
  if (logical_device->system->info.svm_accessible_by_default) {
    out_capabilities->flags |= IREE_HAL_DEVICE_CAPABILITY_PEER_ADDRESSABLE;
    // SVM implies peer coherency on fine-grained memory.
    out_capabilities->flags |= IREE_HAL_DEVICE_CAPABILITY_PEER_COHERENT;
  }

  // Driver handle (HSA agent handle for same-driver refinement).
  out_capabilities->driver_device_handle = (uintptr_t)gpu_agent.handle;

  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_amdgpu_logical_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  return &logical_device->topology_info;
}

// Maps an HSA link type to a HAL topology link class.
// For multi-hop links, the caller should take the worst (highest) class.
static iree_hal_topology_link_class_t iree_hal_amdgpu_link_type_to_link_class(
    hsa_amd_link_info_type_t link_type) {
  switch (link_type) {
    case HSA_AMD_LINK_INFO_TYPE_XGMI:
      return IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF;
    case HSA_AMD_LINK_INFO_TYPE_PCIE:
      return IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT;
    case HSA_AMD_LINK_INFO_TYPE_QPI:
    case HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT:
      // Cross-socket interconnects — treat as cross-root PCIe.
      return IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_CROSS_ROOT;
    case HSA_AMD_LINK_INFO_TYPE_INFINBAND:
      return IREE_HAL_TOPOLOGY_LINK_CLASS_FABRIC;
    default:
      return IREE_HAL_TOPOLOGY_LINK_CLASS_OTHER;
  }
}

static iree_status_t iree_hal_amdgpu_logical_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  // Both devices are AMDGPU logical devices (same-driver guarantee from
  // the device group builder).
  iree_hal_amdgpu_logical_device_t* src_logical =
      iree_hal_amdgpu_logical_device_cast(src_device);
  iree_hal_amdgpu_logical_device_t* dst_logical =
      iree_hal_amdgpu_logical_device_cast(dst_device);

  // Extract GPU agents from the first physical device of each.
  hsa_agent_t src_agent = src_logical->physical_devices[0]->device_agent;
  hsa_agent_t dst_agent = dst_logical->physical_devices[0]->device_agent;
  const iree_hal_amdgpu_libhsa_t* libhsa = &src_logical->system->libhsa;

  // Find both memory pool types on the dst agent. Not all devices expose both
  // pool types — APUs may only have fine-grained, some virtualized configs may
  // lack one. A missing pool is treated as NEVER_ALLOWED (no access via that
  // pool type) rather than a fatal error.
  hsa_amd_memory_pool_t dst_coarse_pool = {0};
  bool has_coarse_pool = iree_hal_amdgpu_try_find_coarse_global_memory_pool(
      libhsa, dst_agent, &dst_coarse_pool);
  hsa_amd_memory_pool_t dst_fine_pool = {0};
  bool has_fine_pool = iree_hal_amdgpu_try_find_fine_global_memory_pool(
      libhsa, dst_agent, &dst_fine_pool);
  if (!has_coarse_pool && !has_fine_pool) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "dst agent has neither coarse nor fine global memory pool");
  }

  // Query whether the src agent can access each dst pool type.
  // Missing pools default to NEVER_ALLOWED.
  hsa_amd_memory_pool_access_t coarse_access =
      HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  if (has_coarse_pool) {
    IREE_RETURN_IF_ERROR(iree_hsa_amd_agent_memory_pool_get_info(
        IREE_LIBHSA(libhsa), src_agent, dst_coarse_pool,
        HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &coarse_access));
  }
  hsa_amd_memory_pool_access_t fine_access =
      HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  if (has_fine_pool) {
    IREE_RETURN_IF_ERROR(iree_hsa_amd_agent_memory_pool_get_info(
        IREE_LIBHSA(libhsa), src_agent, dst_fine_pool,
        HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &fine_access));
  }

  // Query link hop count and topology. The link topology describes the physical
  // interconnect between agents and is the same regardless of pool granularity.
  // Use whichever pool is present (prefer coarse, fall back to fine).
  hsa_amd_memory_pool_t link_query_pool =
      has_coarse_pool ? dst_coarse_pool : dst_fine_pool;
  uint32_t hop_count = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_agent_memory_pool_get_info(
      IREE_LIBHSA(libhsa), src_agent, link_query_pool,
      HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS, &hop_count));

  // Determine link class and characteristics from link topology.
  // Use the bottleneck link (worst class, lowest bandwidth) for multi-hop
  // paths.
  iree_hal_topology_link_class_t link_class =
      IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT;
  uint32_t min_bandwidth_mbps = UINT32_MAX;
  uint32_t max_latency_ns = 0;
  bool all_coherent = true;
  bool all_atomic_64bit = true;
  uint32_t link_numa_distance = 0;

  // The LINK_INFO query writes exactly hop_count entries into the caller's
  // buffer with no separate size parameter — we must allocate for the full
  // count returned by NUM_LINK_HOPS.
  hsa_amd_memory_pool_link_info_t link_info[16];
  if (hop_count > IREE_ARRAYSIZE(link_info)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HSA reports %" PRIu32
                            " link hops between GPU agents (max %" PRIhsz ")",
                            hop_count, IREE_ARRAYSIZE(link_info));
  }

  if (hop_count > 0) {
    memset(link_info, 0, sizeof(link_info[0]) * hop_count);
    IREE_RETURN_IF_ERROR(iree_hsa_amd_agent_memory_pool_get_info(
        IREE_LIBHSA(libhsa), src_agent, link_query_pool,
        HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO, link_info));

    for (uint32_t i = 0; i < hop_count; ++i) {
      // Link class: take the worst (highest enum value = slower link type).
      iree_hal_topology_link_class_t hop_class =
          iree_hal_amdgpu_link_type_to_link_class(link_info[i].link_type);
      if (hop_class > link_class) link_class = hop_class;

      // Bandwidth: take the minimum across hops (bottleneck).
      if (link_info[i].min_bandwidth < min_bandwidth_mbps) {
        min_bandwidth_mbps = link_info[i].min_bandwidth;
      }
      // Latency: take the maximum across hops (accumulates).
      if (link_info[i].max_latency > max_latency_ns) {
        max_latency_ns = link_info[i].max_latency;
      }

      if (!link_info[i].coherent_support) all_coherent = false;
      if (!link_info[i].atomic_support_64bit) all_atomic_64bit = false;

      // NUMA distance: take the last hop's distance (it's relative to the
      // querying agent and represents the accumulated distance).
      link_numa_distance = link_info[i].numa_distance;
    }
  }

  // Available for future bandwidth-proportional copy_cost scaling (e.g.
  // interpolating within a link class based on actual link bandwidth) and
  // latency-proportional latency_class assignment.
  (void)min_bandwidth_mbps;
  (void)max_latency_ns;

  // Refine link class.
  edge->lo = iree_hal_topology_edge_set_link_class(edge->lo, link_class);

  // Refine copy cost based on link type and bandwidth.
  // Scale: 0-15 where 0=same-die, 3=XGMI, 7=PCIe-P2P, 13=host-staged.
  uint32_t copy_cost;
  uint32_t latency_class;
  switch (link_class) {
    case IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF:
      // XGMI: ~200-400 GB/s between GPUs.
      copy_cost = 3;
      latency_class = 3;
      break;
    case IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT:
      // PCIe: ~16-32 GB/s depending on gen.
      copy_cost = 7;
      latency_class = 7;
      break;
    case IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_CROSS_ROOT:
      // Cross-socket PCIe or HyperTransport/QPI traversal.
      copy_cost = 9;
      latency_class = 9;
      break;
    default:
      // Unknown or fabric links.
      copy_cost = 11;
      latency_class = 10;
      break;
  }
  edge->lo = iree_hal_topology_edge_set_copy_cost(edge->lo, copy_cost);
  edge->lo = iree_hal_topology_edge_set_latency_class(edge->lo, latency_class);

  // Refine NUMA distance from link info if available.
  if (link_numa_distance > 0) {
    // HSA reports NUMA distance as SLIT-like values (10=same, 20=1hop, etc.).
    // Normalize to 0-15 scale matching from_capabilities.
    uint32_t scaled =
        link_numa_distance > 10 ? (link_numa_distance - 10) / 2 : 0;
    if (scaled > 15) scaled = 15;
    edge->lo = iree_hal_topology_edge_set_numa_distance(edge->lo, scaled);
  }

  // Refine capability flags based on access mode and link properties.
  // P2P_COPY is available if either pool (coarse or fine) is accessible.
  bool coarse_accessible =
      coarse_access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  bool fine_accessible =
      fine_access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  iree_hal_topology_capability_t caps =
      iree_hal_topology_edge_capability_flags(edge->lo);
  if (coarse_accessible || fine_accessible) {
    caps |= IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY;
    if (all_coherent) {
      caps |= IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT;
    }
    if (all_atomic_64bit) {
      caps |= IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_SYSTEM;
    }
  } else {
    // Neither pool is accessible — no P2P at all. No memory access path means
    // no cross-device copies, coherency, or atomics.
    caps &= ~IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY;
    caps &= ~IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT;
    caps &= ~IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_DEVICE;
    caps &= ~IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_SYSTEM;
  }

  // Non-coherent buffer modes are determined by coarse pool access.
  if (!coarse_accessible) {
    // Coarse pool inaccessible — non-coherent transfers need host staging.
    edge->lo = iree_hal_topology_edge_set_buffer_read_mode_noncoherent(
        edge->lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
    edge->lo = iree_hal_topology_edge_set_buffer_write_mode_noncoherent(
        edge->lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  } else if (coarse_access == HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT) {
    // SVM/large BAR: load/store accessible without explicit grants.
    edge->lo = iree_hal_topology_edge_set_buffer_read_mode_noncoherent(
        edge->lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
    edge->lo = iree_hal_topology_edge_set_buffer_write_mode_noncoherent(
        edge->lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  }
  // DISALLOWED_BY_DEFAULT: DMA engine can access after explicit grant via
  // hsa_amd_agents_allow_access, but shader load/store requires SVM.
  // Leave non-coherent buffer modes at whatever from_capabilities set.

  // Coherent buffer modes are determined by fine pool access.
  if (!fine_accessible) {
    // Fine pool inaccessible — coherent transfers need host staging.
    edge->lo = iree_hal_topology_edge_set_buffer_read_mode_coherent(
        edge->lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
    edge->lo = iree_hal_topology_edge_set_buffer_write_mode_coherent(
        edge->lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  } else if (fine_access == HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT) {
    // Fine-grained memory is coherent and accessible — native load/store.
    edge->lo = iree_hal_topology_edge_set_buffer_read_mode_coherent(
        edge->lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
    edge->lo = iree_hal_topology_edge_set_buffer_write_mode_coherent(
        edge->lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  }
  // DISALLOWED_BY_DEFAULT: fine pool requires explicit access grants.
  // Leave coherent buffer modes at whatever from_capabilities set.

  // The link_class reflects the physical interconnect, not per-pool access
  // modes. Only downgrade to HOST_STAGED when BOTH pools are inaccessible
  // (no P2P path at all). When only one pool is inaccessible, the link is
  // still usable for the other pool type — the link_class from link_info
  // (set above) accurately describes the physical connection.
  if (!coarse_accessible && !fine_accessible) {
    edge->lo = iree_hal_topology_edge_set_link_class(
        edge->lo, IREE_HAL_TOPOLOGY_LINK_CLASS_HOST_STAGED);
    edge->lo = iree_hal_topology_edge_set_copy_cost(edge->lo, 13);
    edge->lo = iree_hal_topology_edge_set_latency_class(edge->lo, 11);
  }

  edge->lo = iree_hal_topology_edge_set_capability_flags(edge->lo, caps);

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
  return iree_hal_amdgpu_aql_command_buffer_create(
      iree_hal_device_allocator(base_device), mode, command_categories,
      effective_queue_affinity, binding_capacity, device_ordinal,
      &logical_device->host_block_pools.command_buffer,
      logical_device->host_allocator, out_command_buffer);
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
      &logical_device->system->libhsa, &logical_device->system->topology,
      identifier, iree_hal_device_host_allocator(base_device),
      out_executable_cache);
}

static iree_status_t iree_hal_amdgpu_logical_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // Mask the user-provided queue affinity to only those we have.
  iree_hal_queue_affinity_and_into(queue_affinity,
                                   logical_device->queue_affinity_mask);
  if (iree_hal_queue_affinity_is_empty(queue_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid queue affinity bits specified");
  }

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
  iree_hal_queue_affinity_and_into(queue_affinity,
                                   logical_device->queue_affinity_mask);
  if (iree_hal_queue_affinity_is_empty(queue_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid queue affinity bits specified");
  }

  IREE_HAL_FOR_QUEUE_AFFINITY(queue_affinity) {
    iree_hal_amdgpu_virtual_queue_t* queue = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_queue_from_ordinal(
        logical_device, queue_ordinal, &queue));
    IREE_RETURN_IF_ERROR(queue->vtable->flush(queue));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_logical_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  const iree_hal_device_profiling_mode_t supported_modes =
      IREE_HAL_DEVICE_PROFILING_MODE_QUEUE_OPERATIONS |
      IREE_HAL_DEVICE_PROFILING_MODE_DISPATCH_COUNTERS |
      IREE_HAL_DEVICE_PROFILING_MODE_EXECUTABLE_COUNTERS;
  if (iree_any_bit_set(options->mode, ~supported_modes)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported AMDGPU profiling mode bits 0x%" PRIx64,
                            options->mode & ~supported_modes);
  }
  if (options->mode == IREE_HAL_DEVICE_PROFILING_MODE_NONE) {
    return iree_ok_status();
  }
  if (!logical_device->frontier_tracker) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU profiling requires an assigned device topology");
  }
  if (logical_device->profiling.mode != IREE_HAL_DEVICE_PROFILING_MODE_NONE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot nest AMDGPU profile captures");
  }

  iree_hal_profile_sink_t* sink = options->sink;
  iree_hal_profile_sink_retain(sink);

  const uint64_t session_id = logical_device->next_profile_session_id++;
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_amdgpu_logical_device_profile_session_metadata(logical_device,
                                                              session_id);

  bool sink_session_begun = false;
  iree_status_t status = iree_ok_status();
  if (sink) {
    status = iree_hal_profile_sink_begin_session(sink, &metadata);
    sink_session_begun = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status) && sink) {
    status = iree_hal_amdgpu_logical_device_write_profile_metadata(
        logical_device, sink, session_id);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_logical_device_profiling_needs_hsa_timestamps(
          options->mode)) {
    status = iree_hal_amdgpu_logical_device_set_hsa_profiling_enabled(
        logical_device, true);
  }

  if (iree_status_is_ok(status)) {
    logical_device->profiling.mode = options->mode;
    logical_device->profiling.session_id = session_id;
    logical_device->profiling.sink = sink;
  } else {
    if (sink_session_begun) {
      status = iree_status_join(
          status, iree_hal_profile_sink_end_session(sink, &metadata,
                                                    iree_status_code(status)));
    }
    iree_hal_profile_sink_release(sink);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_logical_device_profiling_flush(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  if (logical_device->profiling.mode == IREE_HAL_DEVICE_PROFILING_MODE_NONE) {
    return iree_ok_status();
  }
  return iree_hal_amdgpu_logical_device_write_profile_events(
      logical_device, logical_device->profiling.sink,
      logical_device->profiling.session_id);
}

static iree_status_t iree_hal_amdgpu_logical_device_profiling_end(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  iree_status_t status = iree_ok_status();
  const iree_hal_device_profiling_mode_t mode = logical_device->profiling.mode;
  if (mode == IREE_HAL_DEVICE_PROFILING_MODE_NONE) {
    return iree_ok_status();
  }

  iree_hal_profile_sink_t* sink = logical_device->profiling.sink;
  const uint64_t session_id = logical_device->profiling.session_id;
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_amdgpu_logical_device_profile_session_metadata(logical_device,
                                                              session_id);

  if (sink) {
    status = iree_hal_amdgpu_logical_device_write_profile_events(
        logical_device, sink, session_id);
  }
  if (iree_hal_amdgpu_logical_device_profiling_needs_hsa_timestamps(mode)) {
    status = iree_status_join(
        status, iree_hal_amdgpu_logical_device_set_hsa_profiling_enabled(
                    logical_device, false));
  }
  if (sink) {
    status = iree_status_join(
        status, iree_hal_profile_sink_end_session(sink, &metadata,
                                                  iree_status_code(status)));
  }

  logical_device->profiling.mode = IREE_HAL_DEVICE_PROFILING_MODE_NONE;
  logical_device->profiling.session_id = 0;
  logical_device->profiling.sink = NULL;
  iree_hal_profile_sink_release(sink);
  return status;
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
};
