// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/logical_device.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/base/target_platform.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#endif  // defined(IREE_PLATFORM_WINDOWS)

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/vulkan/allocator.h"
#include "iree/hal/drivers/vulkan/builtins.h"
#include "iree/hal/drivers/vulkan/command_buffer.h"
#include "iree/hal/drivers/vulkan/device_plan.h"
#include "iree/hal/drivers/vulkan/executable.h"
#include "iree/hal/drivers/vulkan/executable_cache.h"
#include "iree/hal/drivers/vulkan/physical_device.h"
#include "iree/hal/drivers/vulkan/queue.h"
#include "iree/hal/drivers/vulkan/semaphore.h"
#include "iree/hal/drivers/vulkan/syms.h"
#include "iree/hal/local/profile.h"
#include "iree/hal/utils/file_registry.h"

//===----------------------------------------------------------------------===//
// Physical-device selection
//===----------------------------------------------------------------------===//

typedef enum iree_hal_vulkan_physical_device_selector_mode_e {
  IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_DEFAULT = 0,
  IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_ID = 1,
  IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_PATH = 2,
} iree_hal_vulkan_physical_device_selector_mode_t;

typedef struct iree_hal_vulkan_physical_device_selector_t {
  // Selection mode used when walking visible physical devices.
  iree_hal_vulkan_physical_device_selector_mode_t mode;

  // HAL device id to match when mode is ID.
  iree_hal_device_id_t device_id;

  // HAL device path to match when mode is PATH.
  iree_string_view_t device_path;
} iree_hal_vulkan_physical_device_selector_t;

static iree_status_t iree_hal_vulkan_device_options_parse(
    iree_hal_vulkan_device_options_t* options, iree_string_pair_list_t params) {
  IREE_ASSERT_ARGUMENT(options);
  for (iree_host_size_t i = 0; i < params.count; ++i) {
    const iree_string_pair_t* param = &params.pairs[i];
    if (iree_string_view_equal(param->key, IREE_SV("dispatch_abi")) ||
        iree_string_view_equal(param->key, IREE_SV("vulkan_dispatch_abi"))) {
      IREE_RETURN_IF_ERROR(iree_hal_vulkan_dispatch_abis_parse(
          param->value, &options->dispatch_abis));
    } else if (iree_string_view_equal(param->key,
                                      IREE_SV("cached_bda_replay_instances")) ||
               iree_string_view_equal(
                   param->key, IREE_SV("vulkan_cached_bda_replay_instances"))) {
      if (!iree_string_view_atoi_uint32(
              param->value, &options->max_cached_bda_replay_instances)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "invalid Vulkan cached BDA replay instance count '%.*s'",
            (int)param->value.size, param->value.data);
      }
    } else if (iree_string_view_equal(
                   param->key,
                   IREE_SV("cached_bda_replay_publication_bytes")) ||
               iree_string_view_equal(
                   param->key,
                   IREE_SV("vulkan_cached_bda_replay_publication_bytes"))) {
      if (!iree_string_view_atoi_uint64(
              param->value,
              &options->max_cached_bda_replay_publication_bytes)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "invalid Vulkan cached BDA replay publication byte limit '%.*s'",
            (int)param->value.size, param->value.data);
      }
    } else if (iree_string_view_equal(
                   param->key,
                   IREE_SV("retained_cached_bda_replay_instances")) ||
               iree_string_view_equal(
                   param->key,
                   IREE_SV("vulkan_retained_cached_bda_replay_instances"))) {
      if (!iree_string_view_atoi_uint32(
              param->value, &options->retained_cached_bda_replay_instances)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "invalid Vulkan retained cached BDA replay instance count '%.*s'",
            (int)param->value.size, param->value.data);
      }
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown Vulkan logical device option '%.*s'",
                              (int)param->key.size, param->key.data);
    }
  }
  if (options->max_cached_bda_replay_instances != 0 &&
      options->retained_cached_bda_replay_instances >
          options->max_cached_bda_replay_instances) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan retained cached BDA replay instances (%u) must be <= maximum "
        "cached instances (%u)",
        options->retained_cached_bda_replay_instances,
        options->max_cached_bda_replay_instances);
  }
  return iree_hal_vulkan_dispatch_abis_verify(options->dispatch_abis);
}

static bool iree_hal_vulkan_selector_matches(
    const iree_hal_vulkan_physical_device_selector_t* selector,
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_status_t* out_status) {
  *out_status = iree_ok_status();
  switch (selector->mode) {
    case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_DEFAULT:
      return iree_hal_vulkan_physical_device_supports_baseline(snapshot);
    case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_ID:
      return selector->device_id ==
             (iree_hal_device_id_t)(snapshot->ordinal + 1);
    case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_PATH: {
      char path_storage[64] = {0};
      iree_string_builder_t builder;
      iree_string_builder_initialize_with_storage(
          path_storage, sizeof(path_storage), &builder);
      iree_string_view_t candidate_path = iree_string_view_empty();
      *out_status = iree_hal_vulkan_append_device_path(snapshot, &builder,
                                                       &candidate_path);
      const bool matches =
          iree_status_is_ok(*out_status) &&
          iree_string_view_equal(candidate_path, selector->device_path);
      iree_string_builder_deinitialize(&builder);
      return matches;
    }
    default:
      *out_status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                     "invalid Vulkan selector mode %u",
                                     (uint32_t)selector->mode);
      return false;
  }
}

static iree_status_t iree_hal_vulkan_select_physical_device(
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    const iree_hal_vulkan_driver_options_t* driver_options,
    const iree_hal_vulkan_physical_device_selector_t* selector,
    iree_allocator_t host_allocator, iree_hal_vulkan_instance_t* out_instance,
    iree_hal_vulkan_physical_device_snapshot_t* out_snapshot) {
  IREE_ASSERT_ARGUMENT(libvulkan);
  IREE_ASSERT_ARGUMENT(driver_options);
  IREE_ASSERT_ARGUMENT(selector);
  IREE_ASSERT_ARGUMENT(out_instance);
  IREE_ASSERT_ARGUMENT(out_snapshot);
  memset(out_instance, 0, sizeof(*out_instance));
  memset(out_snapshot, 0, sizeof(*out_snapshot));
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_vulkan_instance_initialize(
      libvulkan, driver_options, host_allocator, out_instance);

  uint32_t physical_device_count = 0;
  VkPhysicalDevice* physical_devices = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_enumerate_physical_device_handles(
        out_instance, host_allocator, &physical_device_count,
        &physical_devices);
  }

  if (iree_status_is_ok(status) && !physical_device_count) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "Vulkan driver has no physical devices");
  }

  if (iree_status_is_ok(status) &&
      selector->mode == IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_ID) {
    const uint32_t ordinal = (uint32_t)(selector->device_id - 1);
    if (selector->device_id == IREE_HAL_DEVICE_ID_DEFAULT ||
        ordinal >= physical_device_count) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan device id %" PRIu64 " out of range; driver has %u devices",
          (uint64_t)selector->device_id, physical_device_count);
    }
  }

  bool selected = false;
  for (uint32_t i = 0;
       i < physical_device_count && iree_status_is_ok(status) && !selected;
       ++i) {
    iree_hal_vulkan_physical_device_snapshot_t snapshot;
    status = iree_hal_vulkan_physical_device_snapshot_initialize(
        out_instance, physical_devices[i], i, host_allocator, &snapshot);
    if (!iree_status_is_ok(status)) break;

    iree_status_t match_status = iree_ok_status();
    const bool matches =
        iree_hal_vulkan_selector_matches(selector, &snapshot, &match_status);
    if (!iree_status_is_ok(match_status)) {
      status = match_status;
    } else if (matches) {
      if (iree_hal_vulkan_physical_device_supports_baseline(&snapshot)) {
        *out_snapshot = snapshot;
        memset(&snapshot, 0, sizeof(snapshot));
        selected = true;
      } else {
        status = iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "Vulkan physical device %u does not satisfy the Vulkan 1.3 "
            "baseline",
            i);
      }
    }
    iree_hal_vulkan_physical_device_snapshot_deinitialize(host_allocator,
                                                          &snapshot);
  }

  if (iree_status_is_ok(status) && !selected) {
    switch (selector->mode) {
      case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_DEFAULT:
        status = iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "no Vulkan physical device satisfies the Vulkan 1.3 baseline");
        break;
      case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_ID:
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "Vulkan device id %" PRIu64 " not found",
                                  (uint64_t)selector->device_id);
        break;
      case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_PATH:
        status = iree_make_status(
            IREE_STATUS_NOT_FOUND, "Vulkan device path '%.*s' not found",
            (int)selector->device_path.size, selector->device_path.data);
        break;
      default:
        break;
    }
  }

  iree_allocator_free(host_allocator, physical_devices);
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_physical_device_snapshot_deinitialize(host_allocator,
                                                          out_snapshot);
    iree_hal_vulkan_instance_deinitialize(out_instance);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Resolved queue handles
//===----------------------------------------------------------------------===//

#define IREE_HAL_VULKAN_COMMAND_BUFFER_BLOCK_SIZE (64 * 1024)

typedef struct iree_hal_vulkan_resolved_queue_t {
  // Queue identity and queue-family capability facts selected by the plan.
  iree_hal_vulkan_queue_selection_t selection;

  // Vulkan queue handle borrowed from the logical device.
  VkQueue handle;
} iree_hal_vulkan_resolved_queue_t;

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_logical_device_t
//===----------------------------------------------------------------------===//

struct iree_hal_vulkan_logical_device_t {
  // HAL resource header.
  iree_hal_resource_t resource;

  // Host allocator used for logical-device-owned host allocations.
  iree_allocator_t host_allocator;

  // Stable device identifier stored inline after this struct.
  iree_string_view_t identifier;

  // Retained Vulkan loader keeping resolved entry-point code live.
  iree_hal_vulkan_libvulkan_t libvulkan;

  // Proactor pool retained from create_params for async host waits.
  iree_async_proactor_pool_t* proactor_pool;

  // Proactor borrowed from the pool for device-local async operations.
  iree_async_proactor_t* proactor;

  // Driver-owned Vulkan instance and instance dispatch table.
  iree_hal_vulkan_instance_t instance;

  // True when destruction should call vkDestroyInstance.
  bool owns_instance;

  // Immutable physical-device inventory used for capability decisions.
  iree_hal_vulkan_physical_device_snapshot_t physical_device;

  // Vulkan logical device handle.
  VkDevice logical_device;

  // True when destruction should call vkDestroyDevice.
  bool owns_logical_device;

  // Device-level Vulkan dispatch table.
  iree_hal_vulkan_device_syms_t syms;

  // HAL feature bits enabled on the logical device.
  iree_hal_vulkan_features_t enabled_features;

  // Executable dispatch ABI bits enabled on the logical device.
  iree_hal_vulkan_dispatch_abis_t enabled_dispatch_abis;

  // Host block pool for command-buffer resource sets and future command blocks.
  iree_arena_block_pool_t command_buffer_block_pool;

  // Recognized Vulkan device extension bits enabled on the logical device.
  iree_hal_vulkan_device_extensions_t enabled_extensions;

  // Device-owned built-in pipelines used by queue command polyfills.
  iree_hal_vulkan_builtins_t builtins;

  // Selected compute-capable queue.
  iree_hal_vulkan_resolved_queue_t compute_queue;

  // Selected transfer-capable queue.
  iree_hal_vulkan_resolved_queue_t transfer_queue;

  // Internal queue used for sparse memory binding operations.
  iree_hal_vulkan_resolved_queue_t sparse_binding_queue;

  // Host synchronization objects for borrowed VkQueue handles.
  struct {
    // Serializes host access to the selected compute queue handle.
    iree_slim_mutex_t compute;

    // Serializes host access to the selected transfer queue handle.
    iree_slim_mutex_t transfer;

    // Serializes host access to a distinct sparse-binding queue handle.
    iree_slim_mutex_t sparse_binding;
  } queue_handle_mutexes;

  // Queue lanes initialized from distinct selected queues, including hidden
  // internal lanes such as sparse binding.
  iree_hal_vulkan_queue_t queue_lanes[IREE_HAL_VULKAN_MAX_QUEUE_LANES];

  // Count of initialized entries in queue_lanes.
  iree_host_size_t queue_lane_count;

  // Queue lane used for compute-capable submissions.
  iree_hal_vulkan_queue_t* compute_queue_lane;

  // Queue lane used for transfer-capable submissions.
  iree_hal_vulkan_queue_t* transfer_queue_lane;

  // Internal queue lane used for sparse binding submissions.
  iree_hal_vulkan_queue_t* sparse_binding_queue_lane;

  // Count of distinct HAL queues exposed through queue affinity.
  iree_host_size_t queue_count;

  // Mask of valid queue affinity bits for this logical device.
  iree_hal_queue_affinity_t queue_affinity_mask;

  // Maximum cached native BDA replay instances retained per queue lane.
  uint32_t max_cached_bda_replay_instances;

  // Maximum BDA publication bytes retained by cached replay instances per lane.
  uint64_t max_cached_bda_replay_publication_bytes;

  // Idle cached native BDA replay instances retained per lane after trim.
  uint32_t retained_cached_bda_replay_instances;

  // Logical allocator.
  iree_hal_allocator_t* device_allocator;

  // Active HAL-native profile recorder, when profiling is enabled.
  iree_hal_local_profile_recorder_t* profile_recorder;

  // Host time domain selected for calibrated profiling samples.
  VkTimeDomainEXT profile_host_time_domain;

  // Next process-local profiling session id.
  uint64_t next_profile_session_id;

  // Next clock-correlation sample id for the active profiling session.
  uint64_t next_profile_clock_correlation_sample_id;

  // Next profile submission id shared across queue lanes.
  iree_atomic_int64_t next_profile_submission_id;

  // Clock alignment state shared with queue lanes while profiling is active.
  iree_hal_vulkan_profile_clock_alignment_t profile_clock_alignment;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  // Shared frontier tracker retained after topology assignment.
  iree_async_frontier_tracker_t* frontier_tracker;

  // Topology-assigned axis registered with the frontier tracker.
  iree_async_axis_t axis;

  // Topology information if this device is part of a multi-device topology.
  iree_hal_device_topology_info_t topology_info;

  // + trailing identifier string storage.
};

static const iree_hal_device_vtable_t iree_hal_vulkan_logical_device_vtable;

static iree_hal_vulkan_logical_device_t* iree_hal_vulkan_logical_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_logical_device_vtable);
  return (iree_hal_vulkan_logical_device_t*)base_value;
}

static iree_status_t iree_hal_vulkan_unimplemented(
    iree_string_view_t operation) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "Vulkan %.*s is not implemented in the Vulkan HAL",
                          (int)operation.size, operation.data);
}

// Power-of-two capacity for logical-device memory lifecycle event buffering.
#define IREE_HAL_VULKAN_LOGICAL_DEVICE_PROFILE_MEMORY_EVENT_CAPACITY (64 * 1024)

// Power-of-two capacity for logical-device queue operation event buffering.
#define IREE_HAL_VULKAN_LOGICAL_DEVICE_PROFILE_QUEUE_EVENT_CAPACITY (64 * 1024)

// Power-of-two capacity for logical-device dispatch event buffering.
#define IREE_HAL_VULKAN_LOGICAL_DEVICE_PROFILE_DISPATCH_EVENT_CAPACITY \
  (64 * 1024)

// Power-of-two capacity for logical-device queue device event buffering.
#define IREE_HAL_VULKAN_LOGICAL_DEVICE_PROFILE_QUEUE_DEVICE_EVENT_CAPACITY \
  (64 * 1024)

static bool iree_hal_vulkan_logical_device_query_pool_epoch(
    void* user_data, iree_async_axis_t axis, uint64_t epoch) {
  iree_hal_vulkan_logical_device_t* device =
      (iree_hal_vulkan_logical_device_t*)user_data;
  return device->frontier_tracker && iree_async_frontier_tracker_query_epoch(
                                         device->frontier_tracker, axis, epoch);
}

static uint32_t iree_hal_vulkan_logical_device_profile_count(
    iree_host_size_t value) {
  return value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
}

static iree_hal_device_profiling_data_families_t
iree_hal_vulkan_logical_device_lightweight_statistics_data_families(void) {
  return IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
         IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
         IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS;
}

static iree_hal_device_profiling_options_t
iree_hal_vulkan_logical_device_resolve_profiling_options(
    const iree_hal_device_profiling_options_t* options) {
  iree_hal_device_profiling_options_t resolved_options = *options;
  if (resolved_options.data_families == IREE_HAL_DEVICE_PROFILING_DATA_NONE &&
      iree_hal_device_profiling_options_requests_lightweight_statistics(
          options)) {
    resolved_options.data_families =
        iree_hal_vulkan_logical_device_lightweight_statistics_data_families();
  }
  if (iree_hal_device_profiling_options_requests_data(
          &resolved_options,
          IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
              IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |
              IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS)) {
    resolved_options.data_families |=
        IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
  }
  resolved_options.flags &=
      ~IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS;
  return resolved_options;
}

static iree_status_t
iree_hal_vulkan_logical_device_select_profile_host_time_domain(
    const iree_hal_vulkan_logical_device_t* device,
    VkTimeDomainEXT* out_time_domain) {
  *out_time_domain = VK_TIME_DOMAIN_DEVICE_EXT;
  const iree_hal_vulkan_time_domain_flags_t domains =
      device->physical_device.calibrated_timestamp_time_domains;
  if (!iree_all_bits_set(domains, IREE_HAL_VULKAN_TIME_DOMAIN_DEVICE)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Vulkan device queue profiling requires calibrated device timestamps");
  }
  if (iree_all_bits_set(domains, IREE_HAL_VULKAN_TIME_DOMAIN_CLOCK_MONOTONIC)) {
    *out_time_domain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT;
    return iree_ok_status();
  }
  if (iree_all_bits_set(domains,
                        IREE_HAL_VULKAN_TIME_DOMAIN_CLOCK_MONOTONIC_RAW)) {
    *out_time_domain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT;
    return iree_ok_status();
  }
#if defined(IREE_PLATFORM_WINDOWS)
  if (iree_all_bits_set(
          domains, IREE_HAL_VULKAN_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER)) {
    *out_time_domain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT;
    return iree_ok_status();
  }
#endif  // defined(IREE_PLATFORM_WINDOWS)
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "Vulkan device queue profiling requires a calibrated host timestamp "
      "domain compatible with IREE host time");
}

static iree_status_t
iree_hal_vulkan_logical_device_validate_queue_device_profiling(
    const iree_hal_vulkan_logical_device_t* device,
    VkTimeDomainEXT* out_time_domain) {
  if (!iree_all_bits_set(
          device->enabled_extensions,
          IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_CALIBRATED_TIMESTAMPS)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "Vulkan device queue profiling requires "
                            "VK_EXT_calibrated_timestamps");
  }
#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC
  if (!device->syms.vkGetCalibratedTimestampsEXT) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan calibrated timestamp extension is enabled but "
        "vkGetCalibratedTimestampsEXT is not loaded");
  }
#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_logical_device_select_profile_host_time_domain(
          device, out_time_domain));
  for (iree_host_size_t i = 0; i < device->queue_lane_count; ++i) {
    const iree_hal_vulkan_queue_t* queue = &device->queue_lanes[i];
    if (queue->timestamp_valid_bits != 64) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "Vulkan device queue profiling requires 64 valid timestamp bits on "
          "queue family %u, but it reports %u",
          queue->queue_family_index, queue->timestamp_valid_bits);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_host_time_domain_frequency(
    VkTimeDomainEXT time_domain, uint64_t* out_frequency_hz) {
  *out_frequency_hz = 0;
  switch (time_domain) {
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unsupported Vulkan calibrated host time domain %u",
          (uint32_t)time_domain);
    case VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT:
    case VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT:
      *out_frequency_hz = 1000000000ull;
      return iree_ok_status();
#if defined(IREE_PLATFORM_WINDOWS)
    case VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT: {
      LARGE_INTEGER frequency;
      if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "QueryPerformanceCounter frequency is unavailable");
      }
      *out_frequency_hz = (uint64_t)frequency.QuadPart;
      return iree_ok_status();
    }
#endif  // defined(IREE_PLATFORM_WINDOWS)
  }
}

static uint64_t iree_hal_vulkan_host_time_domain_timestamp_ns(
    uint64_t timestamp, uint64_t frequency_hz) {
  if (frequency_hz == 1000000000ull) return timestamp;
  const uint64_t seconds = timestamp / frequency_hz;
  const uint64_t remainder = timestamp % frequency_hz;
  return seconds * 1000000000ull + (remainder * 1000000000ull) / frequency_hz;
}

static void iree_hal_vulkan_profile_clock_alignment_reset(
    iree_hal_vulkan_profile_clock_alignment_t* clock_alignment) {
  iree_slim_mutex_lock(&clock_alignment->mutex);
  clock_alignment->minimum_clock_tick = UINT64_MAX;
  clock_alignment->maximum_clock_tick = 0;
  clock_alignment->minimum_event_tick = UINT64_MAX;
  clock_alignment->maximum_event_tick = 0;
  clock_alignment->has_clock_ticks = false;
  clock_alignment->has_event_ticks = false;
  clock_alignment->has_invalid_alignment = false;
  iree_slim_mutex_unlock(&clock_alignment->mutex);
}

static bool iree_hal_vulkan_profile_clock_alignment_record_clock_tick(
    iree_hal_vulkan_profile_clock_alignment_t* clock_alignment,
    uint64_t calibrated_device_tick) {
  iree_slim_mutex_lock(&clock_alignment->mutex);
  if (clock_alignment->has_clock_ticks) {
    clock_alignment->minimum_clock_tick =
        iree_min(clock_alignment->minimum_clock_tick, calibrated_device_tick);
    clock_alignment->maximum_clock_tick =
        iree_max(clock_alignment->maximum_clock_tick, calibrated_device_tick);
  } else {
    clock_alignment->minimum_clock_tick = calibrated_device_tick;
    clock_alignment->maximum_clock_tick = calibrated_device_tick;
    clock_alignment->has_clock_ticks = true;
  }
  if (clock_alignment->has_event_ticks &&
      (clock_alignment->minimum_event_tick <
           clock_alignment->minimum_clock_tick ||
       clock_alignment->maximum_event_tick >
           clock_alignment->maximum_clock_tick)) {
    clock_alignment->has_invalid_alignment = true;
  }
  const bool has_invalid_alignment = clock_alignment->has_invalid_alignment;
  iree_slim_mutex_unlock(&clock_alignment->mutex);
  return has_invalid_alignment;
}

static iree_status_t iree_hal_vulkan_logical_device_write_clock_correlation(
    iree_hal_vulkan_logical_device_t* device) {
  if (!iree_hal_local_profile_recorder_is_enabled(
          device->profile_recorder,
          IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |
              IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS)) {
    return iree_ok_status();
  }

  uint64_t host_frequency_hz = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_host_time_domain_frequency(
      device->profile_host_time_domain, &host_frequency_hz));

  VkCalibratedTimestampInfoEXT timestamp_infos[2] = {
      {
          .sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT,
          .timeDomain = VK_TIME_DOMAIN_DEVICE_EXT,
      },
      {
          .sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT,
          .timeDomain = device->profile_host_time_domain,
      },
  };
  uint64_t timestamps[2] = {0, 0};
  uint64_t max_deviation = 0;
  const iree_time_t host_time_begin_ns = iree_time_now();
  IREE_RETURN_IF_ERROR(iree_vkGetCalibratedTimestampsEXT(
      IREE_VULKAN_DEVICE(&device->syms), device->logical_device,
      IREE_ARRAYSIZE(timestamp_infos), timestamp_infos, timestamps,
      &max_deviation));
  const iree_time_t host_time_end_ns = iree_time_now();
  (void)max_deviation;
  const bool has_invalid_alignment =
      iree_hal_vulkan_profile_clock_alignment_record_clock_tick(
          &device->profile_clock_alignment, timestamps[0]);

  iree_hal_profile_clock_correlation_record_t record =
      iree_hal_profile_clock_correlation_record_default();
  record.flags = IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
                 IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP |
                 IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_SYSTEM_TIMESTAMP |
                 IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET;
  if (has_invalid_alignment) {
    record.flags |=
        IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK_UNALIGNED;
  }
  record.physical_device_ordinal =
      device->topology_info.topology ? device->topology_info.topology_index : 0;
  record.sample_id = ++device->next_profile_clock_correlation_sample_id;
  record.device_tick = timestamps[0];
  record.host_cpu_timestamp_ns = iree_hal_vulkan_host_time_domain_timestamp_ns(
      timestamps[1], host_frequency_hz);
  record.host_system_timestamp = timestamps[1];
  record.host_system_frequency_hz = host_frequency_hz;
  record.host_time_begin_ns = host_time_begin_ns;
  record.host_time_end_ns = host_time_end_ns;
  return iree_hal_local_profile_recorder_write_clock_correlations(
      device->profile_recorder, 1, &record);
}

static iree_hal_local_profile_queue_scope_t
iree_hal_vulkan_logical_device_profile_queue_scope(
    const iree_hal_vulkan_logical_device_t* device, uint32_t queue_ordinal) {
  const uint32_t physical_device_ordinal =
      device->topology_info.topology ? device->topology_info.topology_index : 0;
  return (iree_hal_local_profile_queue_scope_t){
      .physical_device_ordinal = physical_device_ordinal,
      .queue_ordinal = queue_ordinal,
      .stream_id = ((uint64_t)physical_device_ordinal << 32) | queue_ordinal,
  };
}

static void iree_hal_vulkan_logical_device_clear_topology_info(
    iree_hal_vulkan_logical_device_t* device) {
  if (device->frontier_tracker) {
    for (iree_host_size_t i = 0; i < device->queue_lane_count; ++i) {
      iree_hal_vulkan_queue_retire_frontier(&device->queue_lanes[i]);
    }
    iree_async_frontier_tracker_release(device->frontier_tracker);
    device->frontier_tracker = NULL;
    device->axis = 0;
  }
  memset(&device->topology_info, 0, sizeof(device->topology_info));
}

static void iree_hal_vulkan_logical_device_destroy(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  iree_allocator_t host_allocator = device->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_ASSERT(!device->profile_recorder,
              "profiling sessions must be ended before device destruction");

  iree_hal_vulkan_logical_device_clear_topology_info(device);
  iree_hal_channel_provider_release(device->channel_provider);
  for (iree_host_size_t i = 0; i < device->queue_lane_count; ++i) {
    iree_hal_vulkan_queue_deinitialize(&device->queue_lanes[i]);
  }
  iree_hal_allocator_release(device->device_allocator);
  iree_hal_vulkan_builtins_deinitialize(&device->builtins);
  iree_async_proactor_pool_release(device->proactor_pool);
  if (device->logical_device && device->owns_logical_device) {
    iree_vkDestroyDevice(IREE_VULKAN_DEVICE(&device->syms),
                         device->logical_device, /*pAllocator=*/NULL);
  }
  iree_hal_vulkan_physical_device_snapshot_deinitialize(
      host_allocator, &device->physical_device);
  if (device->owns_instance) {
    iree_hal_vulkan_instance_deinitialize(&device->instance);
  }
  iree_hal_vulkan_libvulkan_deinitialize(&device->libvulkan);
  iree_arena_block_pool_deinitialize(&device->command_buffer_block_pool);
  iree_slim_mutex_deinitialize(&device->profile_clock_alignment.mutex);
  iree_slim_mutex_deinitialize(&device->queue_handle_mutexes.sparse_binding);
  iree_slim_mutex_deinitialize(&device->queue_handle_mutexes.transfer);
  iree_slim_mutex_deinitialize(&device->queue_handle_mutexes.compute);
  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_vulkan_logical_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_vulkan_logical_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_vulkan_logical_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_vulkan_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_vulkan_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_vulkan_logical_device_trim(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  for (iree_host_size_t i = 0; i < device->queue_lane_count; ++i) {
    iree_hal_vulkan_queue_trim(&device->queue_lanes[i]);
  }
  iree_arena_block_pool_trim(&device->command_buffer_block_pool);
  return iree_hal_allocator_trim(device->device_allocator);
}

static bool iree_hal_vulkan_logical_device_query_queue_i64(
    iree_hal_vulkan_logical_device_t* device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  int64_t total_value = 0;
  bool has_value = false;
  for (iree_host_size_t i = 0; i < device->queue_lane_count; ++i) {
    int64_t queue_value = 0;
    if (iree_hal_vulkan_queue_query_i64(&device->queue_lanes[i], category, key,
                                        &queue_value)) {
      has_value = true;
      if (queue_value > INT64_MAX - total_value) {
        total_value = INT64_MAX;
      } else {
        total_value += queue_value;
      }
    }
  }
  *out_value = total_value;
  return has_value;
}

static iree_status_t iree_hal_vulkan_logical_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }
  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value =
        iree_hal_vulkan_executable_format_supported(
            device->enabled_features, device->enabled_dispatch_abis, key)
            ? 1
            : 0;
    return iree_ok_status();
  }
  if (iree_string_view_equal(category, IREE_SV("hal.device"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      *out_value = (int64_t)device->queue_count;
      return iree_ok_status();
    }
  } else if (iree_string_view_equal(category, IREE_SV("hal.dispatch"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      *out_value = 1;
      return iree_ok_status();
    }
  } else if (iree_string_view_equal(category, IREE_SV("vulkan.device"))) {
    if (iree_string_view_equal(key, IREE_SV("api_version"))) {
      *out_value = device->physical_device.properties2.properties.apiVersion;
      return iree_ok_status();
    } else if (iree_string_view_equal(key, IREE_SV("subgroup_size"))) {
      *out_value = device->physical_device.subgroup_properties.subgroupSize;
      return iree_ok_status();
    }
  } else if (iree_hal_vulkan_logical_device_query_queue_i64(device, category,
                                                            key, out_value)) {
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_vulkan_logical_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  memset(out_capabilities, 0, sizeof(*out_capabilities));

  memcpy(out_capabilities->physical_device_uuid,
         device->physical_device.id_properties.deviceUUID,
         sizeof(out_capabilities->physical_device_uuid));
  out_capabilities->has_physical_device_uuid = true;
  out_capabilities->driver_device_handle =
      (uintptr_t)device->physical_device.handle;
  out_capabilities->flags |= IREE_HAL_DEVICE_CAPABILITY_TIMELINE_SEMAPHORES |
                             IREE_HAL_DEVICE_CAPABILITY_ATOMIC_SCOPE_DEVICE;
  if (iree_all_bits_set(
          device->enabled_extensions,
          IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY_FD)) {
    out_capabilities->buffer_export_types |=
        IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_FD;
    out_capabilities->buffer_import_types |=
        IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_FD;
  }
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_vulkan_logical_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_vulkan_logical_device_refine_topology_edge(
    iree_hal_device_t* source_device, iree_hal_device_t* target_device,
    iree_hal_topology_edge_t* edge) {
  (void)source_device;
  (void)target_device;
  (void)edge;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_logical_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  if (!topology_info) {
    iree_hal_vulkan_logical_device_clear_topology_info(device);
    return iree_ok_status();
  }
  iree_async_frontier_tracker_t* frontier_tracker =
      topology_info->frontier.tracker;
  iree_async_axis_t base_axis = topology_info->frontier.base_axis;
  const uint8_t session_epoch = iree_async_axis_session(base_axis);
  const uint8_t machine_index = iree_async_axis_machine(base_axis);
  const uint8_t device_index = iree_async_axis_device_index(base_axis);

  iree_status_t status = iree_ok_status();
  iree_host_size_t assigned_queue_count = 0;
  for (iree_host_size_t i = 0;
       i < device->queue_lane_count && iree_status_is_ok(status); ++i) {
    iree_async_axis_t queue_axis = iree_async_axis_make_queue(
        session_epoch, machine_index, device_index, (uint8_t)i);
    status = iree_hal_vulkan_queue_assign_frontier(
        &device->queue_lanes[i], frontier_tracker, queue_axis);
    if (iree_status_is_ok(status)) assigned_queue_count = i + 1;
  }
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < assigned_queue_count; ++i) {
      iree_hal_vulkan_queue_retire_frontier(&device->queue_lanes[i]);
    }
    return status;
  }

  device->topology_info = *topology_info;
  device->frontier_tracker = frontier_tracker;
  device->axis = base_axis;
  iree_async_frontier_tracker_retain(device->frontier_tracker);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_logical_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  (void)base_device;
  (void)queue_affinity;
  (void)params;
  *out_channel = NULL;
  return iree_hal_vulkan_unimplemented(IREE_SV("collective channels"));
}

static iree_status_t iree_hal_vulkan_logical_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_affinity_normalize(
      device->queue_affinity_mask, queue_affinity, &queue_affinity));
  return iree_hal_vulkan_command_buffer_create(
      device->device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, &device->command_buffer_block_pool,
      device->host_allocator, out_command_buffer);
}

static iree_status_t iree_hal_vulkan_logical_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  (void)base_device;
  (void)queue_affinity;
  (void)flags;
  *out_event = NULL;
  return iree_hal_vulkan_unimplemented(IREE_SV("events"));
}

static iree_status_t iree_hal_vulkan_logical_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  return iree_hal_vulkan_executable_cache_create(
      &device->syms, device->logical_device, &device->physical_device,
      device->enabled_features, device->enabled_extensions, identifier,
      device->enabled_dispatch_abis, device->host_allocator,
      out_executable_cache);
}

static iree_status_t iree_hal_vulkan_logical_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;
  if (flags != IREE_HAL_EXTERNAL_FILE_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan external file flags: 0x%" PRIx32, flags);
  }
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_affinity_normalize(
      device->queue_affinity_mask, queue_affinity, &queue_affinity));

  iree_hal_file_t* file = NULL;
  iree_status_t status = iree_hal_file_from_handle(
      device->device_allocator, queue_affinity, access, handle,
      device->proactor, device->host_allocator, &file);
  if (iree_status_is_ok(status) &&
      iree_io_file_handle_type(handle) == IREE_IO_FILE_HANDLE_TYPE_FD &&
      !iree_hal_file_storage_buffer(file) &&
      !iree_hal_file_async_handle(file)) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan fd file transfers require a proactor-backed async file handle");
  }
  if (iree_status_is_ok(status)) {
    *out_file = file;
  } else {
    iree_hal_file_release(file);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_logical_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_affinity_normalize(
      device->queue_affinity_mask, queue_affinity, &queue_affinity));
  return iree_hal_vulkan_semaphore_create(
      device, &device->syms, device->logical_device, device->proactor,
      queue_affinity, initial_value, flags, device->host_allocator,
      out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_vulkan_logical_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  (void)base_device;
  if (iree_hal_vulkan_semaphore_isa(semaphore)) {
    return IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL;
  }
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_vulkan_logical_device_query_queue_pool_backend(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_query_queue_pool_backend(
      device->device_allocator, queue_affinity, out_backend));
  out_backend->epoch_query = (iree_hal_pool_epoch_query_t){
      .fn = iree_hal_vulkan_logical_device_query_pool_epoch,
      .user_data = device,
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_logical_device_select_queue(
    iree_hal_vulkan_logical_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_vulkan_queue_t** out_queue) {
  IREE_ASSERT_ARGUMENT(out_queue);
  *out_queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_affinity_normalize(
      device->queue_affinity_mask, queue_affinity, &queue_affinity));
  if (iree_any_bit_set(queue_affinity,
                       device->compute_queue.selection.affinity)) {
    *out_queue = device->compute_queue_lane;
    return iree_ok_status();
  }
  if (iree_any_bit_set(queue_affinity,
                       device->transfer_queue.selection.affinity)) {
    *out_queue = device->transfer_queue_lane;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "no Vulkan queue lane matches affinity 0x%016" PRIx64,
                          queue_affinity);
}

static iree_status_t iree_hal_vulkan_logical_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  *out_buffer = NULL;
  const iree_device_size_t byte_length = allocation_size;

  iree_hal_vulkan_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_logical_device_select_queue(
      device, queue_affinity, &queue));
  const iree_hal_queue_affinity_t allocation_queue_affinity =
      queue->queue_affinity;

  iree_hal_vulkan_queue_alloca_plan_t allocation_plan;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_select_queue_alloca_plan(
      device->device_allocator, pool, &params, &allocation_size,
      &allocation_plan));
  params.queue_affinity = allocation_queue_affinity;
  if (allocation_plan.strategy ==
          IREE_HAL_VULKAN_QUEUE_ALLOCA_STRATEGY_SPARSE &&
      !iree_all_bits_set(queue->queue_flags, VK_QUEUE_SPARSE_BINDING_BIT)) {
    if (!device->sparse_binding_queue_lane) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan sparse queue_alloca requires a sparse-binding queue");
    }
    queue = device->sparse_binding_queue_lane;
  }
  return iree_hal_vulkan_queue_submit_alloca(
      queue, wait_semaphore_list, signal_semaphore_list, allocation_plan,
      params, allocation_size, byte_length, flags, out_buffer);
}

static iree_status_t iree_hal_vulkan_logical_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  iree_hal_vulkan_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_logical_device_select_queue(
      device, queue_affinity, &queue));
  return iree_hal_vulkan_queue_submit_dealloca(
      queue, wait_semaphore_list, signal_semaphore_list, buffer, flags);
}

static iree_status_t iree_hal_vulkan_logical_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  iree_hal_vulkan_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_logical_device_select_queue(
      device, queue_affinity, &queue));
  return iree_hal_vulkan_queue_submit_fill(
      queue, wait_semaphore_list, signal_semaphore_list, target_buffer,
      target_offset, length, pattern, pattern_length, flags);
}

static iree_status_t iree_hal_vulkan_logical_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  iree_hal_vulkan_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_logical_device_select_queue(
      device, queue_affinity, &queue));
  return iree_hal_vulkan_queue_submit_update(
      queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
      source_offset, target_buffer, target_offset, length, flags);
}

static iree_status_t iree_hal_vulkan_logical_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  iree_hal_vulkan_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_logical_device_select_queue(
      device, queue_affinity, &queue));
  return iree_hal_vulkan_queue_submit_copy(
      queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
      source_offset, target_buffer, target_offset, length, flags);
}

static iree_status_t iree_hal_vulkan_logical_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  iree_hal_vulkan_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_logical_device_select_queue(
      device, queue_affinity, &queue));
  return iree_hal_vulkan_queue_submit_read(
      queue, wait_semaphore_list, signal_semaphore_list, source_file,
      source_offset, target_buffer, target_offset, length, flags);
}

static iree_status_t iree_hal_vulkan_logical_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  iree_hal_vulkan_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_logical_device_select_queue(
      device, queue_affinity, &queue));
  return iree_hal_vulkan_queue_submit_write(
      queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
      source_offset, target_file, target_offset, length, flags);
}

static iree_status_t iree_hal_vulkan_logical_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  iree_hal_vulkan_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_logical_device_select_queue(
      device, queue_affinity, &queue));
  return iree_hal_vulkan_queue_submit_host_call(
      queue, wait_semaphore_list, signal_semaphore_list, call, args, flags);
}

static iree_status_t iree_hal_vulkan_logical_device_queue_dispatch(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  if (!iree_hal_vulkan_executable_isa(executable)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue_dispatch executable is not a Vulkan "
                            "executable");
  }
  if (bindings.count != 0 && !bindings.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue_dispatch binding storage is NULL");
  }
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    if (!bindings.values[i].buffer) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "queue_dispatch binding %" PRIhsz
          " is indirect; direct queue dispatch has no binding table",
          i);
    }
  }

  iree_hal_vulkan_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_logical_device_select_queue(
      device, queue_affinity, &queue));

  return iree_hal_vulkan_queue_submit_dispatch(
      queue, wait_semaphore_list, signal_semaphore_list, executable,
      export_ordinal, config, constants, bindings, flags);
}

static iree_status_t iree_hal_vulkan_logical_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  if (iree_any_bit_set(
          flags, ~(iree_hal_execute_flags_t)
                     IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan queue execute flags: 0x%" PRIx64, flags);
  }
  if (command_buffer) {
    if (!iree_hal_vulkan_command_buffer_isa(command_buffer)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "queue_execute command buffer is not a Vulkan "
                              "command buffer");
    }
  }
  const bool is_empty_command_buffer =
      !command_buffer ||
      iree_hal_vulkan_command_buffer_is_empty(command_buffer);
  if (is_empty_command_buffer && binding_table.count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "barrier-only queue_execute must not provide a binding table "
        "(count=%" PRIhsz ")",
        binding_table.count);
  }
  iree_hal_vulkan_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_logical_device_select_queue(
      device, queue_affinity, &queue));
  if (is_empty_command_buffer) {
    return iree_hal_vulkan_queue_submit_barrier(queue, wait_semaphore_list,
                                                signal_semaphore_list);
  }
  return iree_hal_vulkan_queue_submit_execute(
      queue, wait_semaphore_list, signal_semaphore_list, command_buffer,
      binding_table, flags, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE);
}

static iree_status_t iree_hal_vulkan_logical_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_affinity_normalize(
      device->queue_affinity_mask, queue_affinity, &queue_affinity));
  for (iree_host_size_t i = 0; i < device->queue_lane_count; ++i) {
    if (iree_any_bit_set(queue_affinity,
                         device->queue_lanes[i].queue_affinity)) {
      iree_hal_vulkan_queue_drain_completions(&device->queue_lanes[i]);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_logical_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  if (device->profile_recorder) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot nest Vulkan profile captures");
  }
  const iree_hal_device_profiling_options_t resolved_options =
      iree_hal_vulkan_logical_device_resolve_profiling_options(options);
  const iree_hal_device_profiling_data_families_t supported_data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
      IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS;
  const iree_hal_device_profiling_data_families_t unsupported_data_families =
      resolved_options.data_families & ~supported_data_families;
  if (unsupported_data_families != IREE_HAL_DEVICE_PROFILING_DATA_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported Vulkan profiling data families 0x%" PRIx64,
        unsupported_data_families);
  }
  VkTimeDomainEXT profile_host_time_domain = VK_TIME_DOMAIN_DEVICE_EXT;
  const bool device_queue_events_enabled =
      iree_hal_device_profiling_options_requests_data(
          &resolved_options,
          IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS);
  const bool dispatch_events_enabled =
      iree_hal_device_profiling_options_requests_data(
          &resolved_options, IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS);
  if (device_queue_events_enabled || dispatch_events_enabled) {
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_logical_device_validate_queue_device_profiling(
            device, &profile_host_time_domain));
    for (iree_host_size_t i = 0; i < device->queue_lane_count; ++i) {
      IREE_RETURN_IF_ERROR(
          iree_hal_vulkan_queue_prepare_profile_timestamp_queries(
              &device->queue_lanes[i]));
    }
  }
  iree_hal_vulkan_profile_clock_alignment_reset(
      &device->profile_clock_alignment);

  const uint32_t physical_device_ordinal =
      device->topology_info.topology ? device->topology_info.topology_index : 0;
  iree_hal_profile_device_record_t device_record =
      iree_hal_profile_device_record_default();
  device_record.physical_device_ordinal = physical_device_ordinal;
  device_record.queue_count =
      iree_hal_vulkan_logical_device_profile_count(device->queue_lane_count);
  device_record.flags |= IREE_HAL_PROFILE_DEVICE_FLAG_PHYSICAL_DEVICE_UUID;
  memcpy(device_record.physical_device_uuid,
         device->physical_device.id_properties.deviceUUID,
         sizeof(device_record.physical_device_uuid));

  iree_hal_profile_queue_record_t* queue_records = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      device->host_allocator, device->queue_lane_count, sizeof(*queue_records),
      (void**)&queue_records));
  for (iree_host_size_t i = 0; i < device->queue_lane_count; ++i) {
    const uint32_t queue_ordinal =
        iree_hal_vulkan_logical_device_profile_count(i);
    const iree_hal_local_profile_queue_scope_t scope =
        iree_hal_vulkan_logical_device_profile_queue_scope(device,
                                                           queue_ordinal);
    queue_records[i] = iree_hal_profile_queue_record_default();
    queue_records[i].physical_device_ordinal = scope.physical_device_ordinal;
    queue_records[i].queue_ordinal = scope.queue_ordinal;
    queue_records[i].stream_id = scope.stream_id;
  }

  iree_hal_local_profile_recorder_options_t recorder_options = {
      .name = device->identifier,
      .session_id = ++device->next_profile_session_id,
      .device_record_count = 1,
      .device_records = &device_record,
      .queue_record_count = device->queue_lane_count,
      .queue_records = queue_records,
      .dispatch_event_capacity =
          IREE_HAL_VULKAN_LOGICAL_DEVICE_PROFILE_DISPATCH_EVENT_CAPACITY,
      .queue_event_capacity =
          IREE_HAL_VULKAN_LOGICAL_DEVICE_PROFILE_QUEUE_EVENT_CAPACITY,
      .producer_data_families =
          (dispatch_events_enabled
               ? IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS
               : IREE_HAL_DEVICE_PROFILING_DATA_NONE) |
          (device_queue_events_enabled
               ? IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS
               : IREE_HAL_DEVICE_PROFILING_DATA_NONE),
      .queue_device_event_capacity =
          IREE_HAL_VULKAN_LOGICAL_DEVICE_PROFILE_QUEUE_DEVICE_EVENT_CAPACITY,
      .memory_event_capacity =
          IREE_HAL_VULKAN_LOGICAL_DEVICE_PROFILE_MEMORY_EVENT_CAPACITY,
  };
  iree_hal_local_profile_recorder_t* recorder = NULL;
  iree_status_t status = iree_hal_local_profile_recorder_create(
      &recorder_options, &resolved_options, device->host_allocator, &recorder);
  iree_allocator_free(device->host_allocator, queue_records);
  if (!iree_status_is_ok(status) || !recorder) return status;

  device->profile_recorder = recorder;
  device->profile_host_time_domain = profile_host_time_domain;
  device->next_profile_clock_correlation_sample_id = 0;
  status = iree_hal_vulkan_logical_device_write_clock_correlation(device);
  if (!iree_status_is_ok(status)) {
    device->profile_recorder = NULL;
    status =
        iree_status_join(status, iree_hal_local_profile_recorder_end(recorder));
    iree_hal_local_profile_recorder_destroy(recorder);
    return status;
  }

  iree_atomic_store(&device->next_profile_submission_id, 1,
                    iree_memory_order_relaxed);
  for (iree_host_size_t i = 0; i < device->queue_lane_count; ++i) {
    iree_hal_vulkan_queue_set_profile_recorder(
        &device->queue_lanes[i], recorder,
        iree_hal_vulkan_logical_device_profile_queue_scope(
            device, iree_hal_vulkan_logical_device_profile_count(i)),
        &device->next_profile_submission_id, &device->profile_clock_alignment);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_logical_device_profiling_flush(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  for (iree_host_size_t i = 0; i < device->queue_lane_count; ++i) {
    iree_hal_vulkan_queue_drain_completions(&device->queue_lanes[i]);
  }
  iree_status_t status =
      iree_hal_vulkan_logical_device_write_clock_correlation(device);
  if (iree_status_is_ok(status)) {
    status = iree_hal_local_profile_recorder_flush(device->profile_recorder);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_logical_device_profiling_end(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  iree_hal_local_profile_recorder_t* recorder = device->profile_recorder;
  if (!recorder) return iree_ok_status();

  for (iree_host_size_t i = 0; i < device->queue_lane_count; ++i) {
    iree_hal_vulkan_queue_drain_completions(&device->queue_lanes[i]);
  }
  iree_status_t status =
      iree_hal_vulkan_logical_device_write_clock_correlation(device);

  const iree_hal_local_profile_queue_scope_t empty_scope =
      iree_hal_local_profile_queue_scope_default();
  for (iree_host_size_t i = 0; i < device->queue_lane_count; ++i) {
    iree_hal_vulkan_queue_set_profile_recorder(
        &device->queue_lanes[i], /*profile_recorder=*/NULL, empty_scope,
        /*submission_counter=*/NULL, /*clock_alignment=*/NULL);
  }
  device->profile_recorder = NULL;
  device->profile_host_time_domain = VK_TIME_DOMAIN_DEVICE_EXT;
  device->next_profile_clock_correlation_sample_id = 0;
  iree_atomic_store(&device->next_profile_submission_id, 0,
                    iree_memory_order_relaxed);

  status =
      iree_status_join(status, iree_hal_local_profile_recorder_end(recorder));
  iree_hal_local_profile_recorder_destroy(recorder);
  return status;
}

static iree_status_t iree_hal_vulkan_logical_device_external_capture_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_external_capture_options_t* options) {
  (void)base_device;
  (void)options;
  return iree_hal_vulkan_unimplemented(IREE_SV("external capture"));
}

static iree_status_t iree_hal_vulkan_logical_device_external_capture_end(
    iree_hal_device_t* base_device) {
  (void)base_device;
  return iree_hal_vulkan_unimplemented(IREE_SV("external capture"));
}

static iree_status_t iree_hal_vulkan_logical_device_allocate(
    iree_string_view_t identifier, const iree_hal_vulkan_libvulkan_t* libvulkan,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_logical_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(libvulkan);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;

  iree_host_size_t total_size = sizeof(iree_hal_vulkan_logical_device_t);
  if (!iree_host_size_checked_add(total_size, identifier.size, &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan logical device allocation overflow");
  }

  iree_hal_vulkan_logical_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  memset(device, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_vulkan_logical_device_vtable,
                               &device->resource);
  device->host_allocator = host_allocator;
  iree_arena_block_pool_initialize(IREE_HAL_VULKAN_COMMAND_BUFFER_BLOCK_SIZE,
                                   host_allocator,
                                   &device->command_buffer_block_pool);
  iree_string_view_append_to_buffer(identifier, &device->identifier,
                                    (char*)device + sizeof(*device));

  iree_status_t status =
      iree_hal_vulkan_libvulkan_copy(libvulkan, &device->libvulkan);
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_initialize(&device->queue_handle_mutexes.compute);
    iree_slim_mutex_initialize(&device->queue_handle_mutexes.transfer);
    iree_slim_mutex_initialize(&device->queue_handle_mutexes.sparse_binding);
    iree_slim_mutex_initialize(&device->profile_clock_alignment.mutex);
    iree_hal_vulkan_profile_clock_alignment_reset(
        &device->profile_clock_alignment);
    *out_device = device;
  } else {
    iree_arena_block_pool_deinitialize(&device->command_buffer_block_pool);
    iree_allocator_free(host_allocator, device);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_logical_device_initialize_proactor(
    iree_hal_vulkan_logical_device_t* device,
    const iree_hal_device_create_params_t* create_params) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(create_params);

  device->proactor_pool = create_params->proactor_pool;
  iree_async_proactor_pool_retain(device->proactor_pool);
  return iree_async_proactor_pool_get(device->proactor_pool, 0,
                                      &device->proactor);
}

static iree_status_t iree_hal_vulkan_logical_device_initialize_allocator(
    iree_hal_vulkan_logical_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  return iree_hal_vulkan_allocator_create(
      (iree_hal_device_t*)device, &device->syms, device->logical_device,
      &device->physical_device, device->enabled_features,
      device->enabled_extensions, device->queue_affinity_mask,
      device->sparse_binding_queue_lane, device->proactor,
      device->host_allocator, &device->device_allocator);
}

static void iree_hal_vulkan_logical_device_resolve_queue_assignment(
    iree_hal_vulkan_logical_device_t* device,
    const iree_hal_vulkan_queue_assignment_t* queue_assignment) {
  device->compute_queue.selection = queue_assignment->compute;
  VkDeviceQueueInfo2 queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
      .queueFamilyIndex = device->compute_queue.selection.family_index,
      .queueIndex = device->compute_queue.selection.queue_index,
  };
  iree_vkGetDeviceQueue2(IREE_VULKAN_DEVICE(&device->syms),
                         device->logical_device, &queue_info,
                         &device->compute_queue.handle);

  device->transfer_queue.selection = queue_assignment->transfer;
  queue_info.queueFamilyIndex = device->transfer_queue.selection.family_index;
  queue_info.queueIndex = device->transfer_queue.selection.queue_index;
  iree_vkGetDeviceQueue2(IREE_VULKAN_DEVICE(&device->syms),
                         device->logical_device, &queue_info,
                         &device->transfer_queue.handle);

  device->sparse_binding_queue.selection = queue_assignment->sparse_binding;
  if (iree_hal_vulkan_queue_assignment_has_sparse_binding(queue_assignment)) {
    if (iree_hal_vulkan_queue_selection_is_same(
            &device->sparse_binding_queue.selection,
            &device->compute_queue.selection)) {
      device->sparse_binding_queue.handle = device->compute_queue.handle;
    } else if (iree_hal_vulkan_queue_selection_is_same(
                   &device->sparse_binding_queue.selection,
                   &device->transfer_queue.selection)) {
      device->sparse_binding_queue.handle = device->transfer_queue.handle;
    } else {
      queue_info.queueFamilyIndex =
          device->sparse_binding_queue.selection.family_index;
      queue_info.queueIndex =
          device->sparse_binding_queue.selection.queue_index;
      iree_vkGetDeviceQueue2(IREE_VULKAN_DEVICE(&device->syms),
                             device->logical_device, &queue_info,
                             &device->sparse_binding_queue.handle);
    }
  }

  device->queue_count = queue_assignment->queue_count;
  device->queue_affinity_mask = (1ull << device->queue_count) - 1;
}

static iree_slim_mutex_t* iree_hal_vulkan_logical_device_queue_handle_mutex(
    iree_hal_vulkan_logical_device_t* device,
    const iree_hal_vulkan_resolved_queue_t* queue) {
  if (iree_hal_vulkan_queue_selection_is_same(
          &queue->selection, &device->compute_queue.selection)) {
    return &device->queue_handle_mutexes.compute;
  }
  if (iree_hal_vulkan_queue_selection_is_same(
          &queue->selection, &device->transfer_queue.selection)) {
    return &device->queue_handle_mutexes.transfer;
  }
  return &device->queue_handle_mutexes.sparse_binding;
}

static iree_status_t iree_hal_vulkan_logical_device_initialize_queue_lane(
    iree_hal_vulkan_logical_device_t* device,
    const iree_hal_vulkan_resolved_queue_t* queue,
    iree_hal_vulkan_queue_role_t role, iree_hal_vulkan_queue_t** out_queue) {
  IREE_ASSERT_ARGUMENT(out_queue);
  *out_queue = NULL;
  if (device->queue_lane_count >= IREE_HAL_VULKAN_MAX_QUEUE_LANES) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Vulkan logical device queue lane storage is full");
  }

  iree_hal_vulkan_queue_t* queue_lane =
      &device->queue_lanes[device->queue_lane_count];
  iree_hal_vulkan_queue_params_t params = {
      .device = device,
      .syms = &device->syms,
      .logical_device = device->logical_device,
      .builtins = &device->builtins,
      .enabled_dispatch_abis = device->enabled_dispatch_abis,
      .queue = queue->handle,
      .queue_flags = queue->selection.flags,
      .timestamp_valid_bits = queue->selection.timestamp_valid_bits,
      .queue_handle_mutex =
          iree_hal_vulkan_logical_device_queue_handle_mutex(device, queue),
      .proactor = device->proactor,
      .queue_family_index = queue->selection.family_index,
      .queue_index = queue->selection.queue_index,
      .queue_affinity = queue->selection.affinity,
      .role = role,
      .host_allocator = device->host_allocator,
      .max_cached_bda_replay_instances =
          device->max_cached_bda_replay_instances,
      .max_cached_bda_replay_publication_bytes =
          device->max_cached_bda_replay_publication_bytes,
      .retained_cached_bda_replay_instances =
          device->retained_cached_bda_replay_instances,
  };
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_initialize(&params, queue_lane));
  device->queue_lane_count = device->queue_lane_count + 1;
  *out_queue = queue_lane;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_logical_device_initialize_queues(
    iree_hal_vulkan_logical_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  device->queue_lane_count = 0;
  device->compute_queue_lane = NULL;
  device->transfer_queue_lane = NULL;
  device->sparse_binding_queue_lane = NULL;

  IREE_RETURN_IF_ERROR(iree_hal_vulkan_logical_device_initialize_queue_lane(
      device, &device->compute_queue, IREE_HAL_VULKAN_QUEUE_ROLE_COMPUTE,
      &device->compute_queue_lane));
  iree_status_t status = iree_ok_status();
  if (device->transfer_queue.selection.affinity ==
      device->compute_queue.selection.affinity) {
    device->transfer_queue_lane = device->compute_queue_lane;
  } else {
    status = iree_hal_vulkan_logical_device_initialize_queue_lane(
        device, &device->transfer_queue, IREE_HAL_VULKAN_QUEUE_ROLE_TRANSFER,
        &device->transfer_queue_lane);
  }
  if (iree_status_is_ok(status) && device->sparse_binding_queue.handle) {
    if (iree_hal_vulkan_queue_selection_is_same(
            &device->sparse_binding_queue.selection,
            &device->compute_queue.selection)) {
      device->sparse_binding_queue_lane = device->compute_queue_lane;
    } else if (iree_hal_vulkan_queue_selection_is_same(
                   &device->sparse_binding_queue.selection,
                   &device->transfer_queue.selection)) {
      device->sparse_binding_queue_lane = device->transfer_queue_lane;
    } else {
      status = iree_hal_vulkan_logical_device_initialize_queue_lane(
          device, &device->sparse_binding_queue,
          IREE_HAL_VULKAN_QUEUE_ROLE_SPARSE_BINDING,
          &device->sparse_binding_queue_lane);
    }
  }
  return status;
}

static iree_status_t iree_hal_vulkan_logical_device_initialize_queue_staging(
    iree_hal_vulkan_logical_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(device->device_allocator);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < device->queue_lane_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_vulkan_queue_initialize_staging(&device->queue_lanes[i],
                                                      device->device_allocator);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_logical_device_initialize_from_plan(
    iree_hal_vulkan_logical_device_t* device,
    const iree_hal_vulkan_device_plan_t* device_plan,
    const iree_hal_vulkan_device_options_t* device_options,
    const iree_hal_device_create_params_t* create_params) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(device_plan);
  IREE_ASSERT_ARGUMENT(device_options);
  IREE_ASSERT_ARGUMENT(create_params);

  iree_status_t status =
      iree_hal_vulkan_logical_device_initialize_proactor(device, create_params);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_libvulkan_load_device_syms(
        &device->instance.syms, device->logical_device, &device->syms);
  }
  if (iree_status_is_ok(status)) {
    device->enabled_features = device_plan->enabled_features;
    device->enabled_extensions = device_plan->enabled_extensions;
    device->enabled_dispatch_abis = device_plan->enabled_dispatch_abis;
    device->max_cached_bda_replay_instances =
        device_options->max_cached_bda_replay_instances;
    device->max_cached_bda_replay_publication_bytes =
        device_options->max_cached_bda_replay_publication_bytes;
    device->retained_cached_bda_replay_instances =
        device_options->retained_cached_bda_replay_instances;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_builtins_initialize(
        &device->syms, device->logical_device, &device->physical_device,
        &device->builtins);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_logical_device_resolve_queue_assignment(
        device, &device_plan->queue_assignment);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_logical_device_initialize_queues(device);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_logical_device_initialize_allocator(device);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_logical_device_initialize_queue_staging(device);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_logical_device_create_from_selection(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* driver_options,
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    const iree_hal_vulkan_device_options_t* device_options,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_vulkan_instance_t* instance,
    iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_device_plan_t device_plan;
  iree_status_t status = iree_hal_vulkan_device_plan_initialize_for_create(
      snapshot, device_options, driver_options->requested_features,
      &device_plan);

  iree_hal_vulkan_logical_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_logical_device_allocate(identifier, libvulkan,
                                                     host_allocator, &device);
  }
  if (iree_status_is_ok(status)) {
    device->instance = *instance;
    device->owns_instance = true;
    memset(instance, 0, sizeof(*instance));
    device->physical_device = *snapshot;
    memset(snapshot, 0, sizeof(*snapshot));
  }
  if (iree_status_is_ok(status)) {
    VkDeviceCreateInfo device_create_info;
    iree_hal_vulkan_device_plan_make_create_info(&device_plan,
                                                 &device_create_info);
    status =
        iree_vkCreateDevice(IREE_VULKAN_INSTANCE(&device->instance.syms),
                            device->physical_device.handle, &device_create_info,
                            /*pAllocator=*/NULL, &device->logical_device);
  }
  if (iree_status_is_ok(status)) {
    device->owns_logical_device = true;
    status = iree_hal_vulkan_logical_device_initialize_from_plan(
        device, &device_plan, device_options, create_params);
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else if (device) {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_logical_device_create_with_selector(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* driver_options,
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    const iree_hal_vulkan_physical_device_selector_t* selector,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(driver_options);
  IREE_ASSERT_ARGUMENT(libvulkan);
  IREE_ASSERT_ARGUMENT(selector);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!create_params->proactor_pool) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan logical device creation requires a proactor pool");
  }

  iree_hal_vulkan_device_options_t device_options =
      driver_options->device_options;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_device_options_parse(&device_options,
                                               (iree_string_pair_list_t){
                                                   .count = param_count,
                                                   .pairs = params,
                                               }));

  iree_hal_vulkan_instance_t instance;
  iree_hal_vulkan_physical_device_snapshot_t snapshot;
  iree_status_t status = iree_hal_vulkan_select_physical_device(
      libvulkan, driver_options, selector, host_allocator, &instance,
      &snapshot);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_logical_device_create_from_selection(
        identifier, driver_options, libvulkan, &device_options, create_params,
        host_allocator, &instance, &snapshot, out_device);
  }
  iree_hal_vulkan_physical_device_snapshot_deinitialize(host_allocator,
                                                        &snapshot);
  iree_hal_vulkan_instance_deinitialize(&instance);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_logical_device_create_by_id(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* driver_options,
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    iree_hal_device_id_t device_id, iree_host_size_t param_count,
    const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  const iree_hal_vulkan_physical_device_selector_t selector = {
      .mode = device_id == IREE_HAL_DEVICE_ID_DEFAULT
                  ? IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_DEFAULT
                  : IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_ID,
      .device_id = device_id,
  };
  return iree_hal_vulkan_logical_device_create_with_selector(
      identifier, driver_options, libvulkan, &selector, param_count, params,
      create_params, host_allocator, out_device);
}

iree_status_t iree_hal_vulkan_logical_device_create_by_path(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* driver_options,
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  const iree_hal_vulkan_physical_device_selector_t selector = {
      .mode = iree_string_view_is_empty(device_path)
                  ? IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_DEFAULT
                  : IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_PATH,
      .device_path = device_path,
  };
  return iree_hal_vulkan_logical_device_create_with_selector(
      identifier, driver_options, libvulkan, &selector, param_count, params,
      create_params, host_allocator, out_device);
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_wrap_device(
    iree_string_view_t identifier,
    const iree_hal_vulkan_device_options_t* options,
    const iree_hal_device_create_params_t* create_params,
    const iree_hal_vulkan_syms_t* instance_syms, VkInstance instance,
    VkPhysicalDevice physical_device, VkDevice logical_device,
    const iree_hal_vulkan_external_device_params_t* external_device_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(instance_syms);
  IREE_ASSERT_ARGUMENT(external_device_params);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (options->max_cached_bda_replay_instances != 0 &&
      options->retained_cached_bda_replay_instances >
          options->max_cached_bda_replay_instances) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan retained cached BDA replay instances (%u) must be <= maximum "
        "cached instances (%u)",
        options->retained_cached_bda_replay_instances,
        options->max_cached_bda_replay_instances);
  }
  if (!instance || !physical_device || !logical_device) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "external Vulkan wrapping requires non-null instance, physical_device, "
        "and logical_device handles");
  }
  if (!create_params->proactor_pool) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan logical device creation requires a proactor pool");
  }

  iree_hal_vulkan_instance_t wrapped_instance = {
      .handle = instance,
  };
  iree_status_t status = iree_hal_vulkan_libvulkan_load_instance_syms(
      &instance_syms->libvulkan, instance, &wrapped_instance.syms);

  iree_hal_vulkan_physical_device_snapshot_t snapshot;
  memset(&snapshot, 0, sizeof(snapshot));
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_physical_device_snapshot_initialize(
        &wrapped_instance, physical_device, /*ordinal=*/0, host_allocator,
        &snapshot);
  }
  iree_hal_vulkan_device_plan_t device_plan;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_device_plan_initialize_for_wrap(
        &snapshot, options, external_device_params, &device_plan);
  }

  iree_hal_vulkan_logical_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_logical_device_allocate(
        identifier, &instance_syms->libvulkan, host_allocator, &device);
  }
  if (iree_status_is_ok(status)) {
    device->instance = wrapped_instance;
    memset(&wrapped_instance, 0, sizeof(wrapped_instance));
    device->physical_device = snapshot;
    memset(&snapshot, 0, sizeof(snapshot));
    device->logical_device = logical_device;
    status = iree_hal_vulkan_logical_device_initialize_from_plan(
        device, &device_plan, options, create_params);
  }
  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else if (device) {
    iree_hal_device_release((iree_hal_device_t*)device);
  }

  iree_hal_vulkan_physical_device_snapshot_deinitialize(host_allocator,
                                                        &snapshot);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_device_vtable_t iree_hal_vulkan_logical_device_vtable = {
    .destroy = iree_hal_vulkan_logical_device_destroy,
    .id = iree_hal_vulkan_logical_device_id,
    .host_allocator = iree_hal_vulkan_logical_device_host_allocator,
    .device_allocator = iree_hal_vulkan_logical_device_allocator,
    .replace_device_allocator = iree_hal_vulkan_replace_device_allocator,
    .replace_channel_provider = iree_hal_vulkan_replace_channel_provider,
    .trim = iree_hal_vulkan_logical_device_trim,
    .query_i64 = iree_hal_vulkan_logical_device_query_i64,
    .query_capabilities = iree_hal_vulkan_logical_device_query_capabilities,
    .topology_info = iree_hal_vulkan_logical_device_topology_info,
    .refine_topology_edge = iree_hal_vulkan_logical_device_refine_topology_edge,
    .assign_topology_info = iree_hal_vulkan_logical_device_assign_topology_info,
    .create_channel = iree_hal_vulkan_logical_device_create_channel,
    .create_command_buffer =
        iree_hal_vulkan_logical_device_create_command_buffer,
    .create_event = iree_hal_vulkan_logical_device_create_event,
    .create_executable_cache =
        iree_hal_vulkan_logical_device_create_executable_cache,
    .import_file = iree_hal_vulkan_logical_device_import_file,
    .create_semaphore = iree_hal_vulkan_logical_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_vulkan_logical_device_query_semaphore_compatibility,
    .query_queue_pool_backend =
        iree_hal_vulkan_logical_device_query_queue_pool_backend,
    .queue_alloca = iree_hal_vulkan_logical_device_queue_alloca,
    .queue_dealloca = iree_hal_vulkan_logical_device_queue_dealloca,
    .queue_fill = iree_hal_vulkan_logical_device_queue_fill,
    .queue_update = iree_hal_vulkan_logical_device_queue_update,
    .queue_copy = iree_hal_vulkan_logical_device_queue_copy,
    .queue_read = iree_hal_vulkan_logical_device_queue_read,
    .queue_write = iree_hal_vulkan_logical_device_queue_write,
    .queue_host_call = iree_hal_vulkan_logical_device_queue_host_call,
    .queue_dispatch = iree_hal_vulkan_logical_device_queue_dispatch,
    .queue_execute = iree_hal_vulkan_logical_device_queue_execute,
    .queue_flush = iree_hal_vulkan_logical_device_queue_flush,
    .profiling_begin = iree_hal_vulkan_logical_device_profiling_begin,
    .profiling_flush = iree_hal_vulkan_logical_device_profiling_flush,
    .profiling_end = iree_hal_vulkan_logical_device_profiling_end,
    .external_capture_begin =
        iree_hal_vulkan_logical_device_external_capture_begin,
    .external_capture_end = iree_hal_vulkan_logical_device_external_capture_end,
};
