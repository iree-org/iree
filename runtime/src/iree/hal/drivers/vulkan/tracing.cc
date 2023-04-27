// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/tracing.h"

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

#include "iree/base/api.h"
#include "iree/base/target_platform.h"

// Total number of queries the per-queue query pool will contain. This
// translates to the maximum number of outstanding queries before collection is
// required.
#define IREE_HAL_VULKAN_TRACING_DEFAULT_QUERY_CAPACITY (32 * 1024)

// Total number of queries that can be read back from the API in a single
// collection.
#define IREE_HAL_VULKAN_TRACING_READBACK_QUERY_CAPACITY (8 * 1024)

// Number of times we will query the max_deviation from calibrated timestamps.
// The more we do the better confidence we have in a lower-bound.
#define IREE_HAL_VULKAN_TRACING_MAX_DEVIATION_PROBE_COUNT 32

typedef struct iree_hal_vulkan_timestamp_query_t {
  uint64_t timestamp;
  uint64_t availability;  // non-zero if available
} iree_hal_vulkan_timestamp_query_t;

struct iree_hal_vulkan_tracing_context_t {
  // Device and queue the context represents.
  iree::hal::vulkan::VkDeviceHandle* logical_device;
  VkQueue queue;
  iree_allocator_t host_allocator;

  // Maintenance queue that supports dispatch commands and can be used to reset
  // queries.
  VkQueue maintenance_dispatch_queue;
  // Command pool that serves command buffers compatible with the
  // |maintenance_dispatch_queue|.
  iree::hal::vulkan::VkCommandPoolHandle* maintenance_command_pool;

  // A unique GPU zone ID allocated from Tracy.
  // There is a global limit of 255 GPU zones (ID 255 is special).
  uint8_t id;

  // Defines how the timestamps are interpreted (device-specific, posix, QPC).
  // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkTimeDomainEXT.html
  VkTimeDomainEXT time_domain;

  // Maximum expected deviation between CPU and GPU timestamps based on an
  // average computed at startup. Calibration events that exceed this value are
  // discarded.
  uint64_t max_expected_deviation;

  // Vulkan-reported CPU timestamp of the last calibration.
  // Used to detect when drift occurs and we need to notify tracy.
  uint64_t previous_cpu_time;

  // Pool of query instances that we treat as a backing store for a ringbuffer.
  VkQueryPool query_pool;

  // Indices into |query_pool| defining a ringbuffer.
  uint32_t query_head;
  uint32_t query_tail;
  uint32_t query_capacity;

  // Readback storage; large enough to get a decent chunk of queries back from
  // the API in one shot.
  //
  // Data is stored as [[timestamp, availability], ...].
  // Availability will be non-zero if the timestamp is valid. Since we put all
  // timestamps in order once we reach an unavailable timestamp we can bail
  // and leave that for future collections.
  iree_hal_vulkan_timestamp_query_t
      readback_buffer[IREE_HAL_VULKAN_TRACING_READBACK_QUERY_CAPACITY];
};

// Allocates and begins a command buffer and returns its handle.
// Returns VK_NULL_HANDLE if allocation fails.
static VkCommandBuffer iree_hal_vulkan_tracing_begin_command_buffer(
    iree_hal_vulkan_tracing_context_t* context) {
  const auto& syms = context->logical_device->syms();

  VkCommandBufferAllocateInfo command_buffer_info = {};
  command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_buffer_info.commandPool = *context->maintenance_command_pool;
  command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_buffer_info.commandBufferCount = 1;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  IREE_IGNORE_ERROR(context->maintenance_command_pool->Allocate(
      &command_buffer_info, &command_buffer));
  if (!command_buffer) return VK_NULL_HANDLE;

  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  syms->vkBeginCommandBuffer(command_buffer, &begin_info);

  return command_buffer;
}

// Ends and submits |command_buffer| and waits for it to complete.
static void iree_hal_vulkan_tracing_submit_command_buffer(
    iree_hal_vulkan_tracing_context_t* context,
    VkCommandBuffer command_buffer) {
  const auto& syms = context->logical_device->syms();

  syms->vkEndCommandBuffer(command_buffer);

  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;
  syms->vkQueueSubmit(context->maintenance_dispatch_queue, 1, &submit_info,
                      VK_NULL_HANDLE);
  syms->vkQueueWaitIdle(context->maintenance_dispatch_queue);

  context->maintenance_command_pool->Free(command_buffer);
}

// Synchronously resets a range of querys in a query pool.
// This may submit commands to the queue.
static void iree_hal_vulkan_tracing_reset_query_pool(
    iree_hal_vulkan_tracing_context_t* context, uint32_t query_index,
    uint32_t query_count) {
  const auto& syms = context->logical_device->syms();

  // Fast-path for when host-side vkResetQueryPool is available.
  // This is core in Vulkan 1.2.
  if (context->logical_device->enabled_extensions().host_query_reset) {
    PFN_vkResetQueryPool vkResetQueryPool_fn = syms->vkResetQueryPool
                                                   ? syms->vkResetQueryPool
                                                   : syms->vkResetQueryPoolEXT;
    if (vkResetQueryPool_fn != NULL) {
      vkResetQueryPool_fn(*context->logical_device, context->query_pool,
                          query_index, query_count);
      return;
    }
  }

  // Slow-path submitting a command buffer to reset the query pool. It's obvious
  // why vkResetQueryPool was added :)
  VkCommandBuffer command_buffer =
      iree_hal_vulkan_tracing_begin_command_buffer(context);
  if (command_buffer != VK_NULL_HANDLE) {
    syms->vkCmdResetQueryPool(command_buffer, context->query_pool, query_index,
                              query_count);
    iree_hal_vulkan_tracing_submit_command_buffer(context, command_buffer);
  }
}

// Attempts to get a timestamp from both the CPU and GPU that are correlated
// with each other. Only valid when calibration is supported.
static void iree_hal_vulkan_tracing_query_calibration_timestamps(
    iree_hal_vulkan_tracing_context_t* context, uint64_t* out_cpu_time,
    uint64_t* out_gpu_time) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_cpu_time = 0;
  *out_gpu_time = 0;

  VkCalibratedTimestampInfoEXT timestamp_infos[2];
  timestamp_infos[0].sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT;
  timestamp_infos[0].pNext = NULL;
  timestamp_infos[0].timeDomain = VK_TIME_DOMAIN_DEVICE_EXT;
  timestamp_infos[1].sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT;
  timestamp_infos[1].pNext = NULL;
  timestamp_infos[1].timeDomain = context->time_domain;
  uint64_t timestamps[2] = {0, 0};
  uint64_t max_deviation = 0;
  do {
    context->logical_device->syms()->vkGetCalibratedTimestampsEXT(
        *context->logical_device, IREE_ARRAYSIZE(timestamps), timestamp_infos,
        timestamps, &max_deviation);
  } while (max_deviation > context->max_expected_deviation);

  *out_gpu_time = timestamps[0];
  *out_cpu_time = timestamps[1];
  switch (context->time_domain) {
#if defined(IREE_PLATFORM_WINDOWS)
    case VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT:
      *out_cpu_time *= (uint64_t)(1000000000.0 / iree_tracing_frequency());
      break;
#else
    case VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT:
    case VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT:
      // TODO(benvanik): posix calibrated timestamps - ignored for now.
      break;
#endif  // IREE_PLATFORM_WINDOWS
    default:
      break;
  }

  IREE_TRACE_ZONE_END(z0);
}

// Populates |out_cpu_time| and |out_gpu_time| with calibrated timestamps.
// Depending on whether VK_EXT_calibrated_timestamps is available this may be
// a guess done by ourselves (with lots of slop) or done by the driver (with
// less slop).
static void iree_hal_vulkan_tracing_perform_initial_calibration(
    iree_hal_vulkan_tracing_context_t* context, uint64_t* out_cpu_time,
    uint64_t* out_gpu_time) {
  const auto& syms = context->logical_device->syms();
  *out_cpu_time = 0;
  *out_gpu_time = 0;

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0,
                              context->time_domain == VK_TIME_DOMAIN_DEVICE_EXT
                                  ? "VK_TIME_DOMAIN_DEVICE_EXT"
                                  : "VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT");

  // Attempt to get a timestamp from both the device and the host at roughly the
  // same time. There's a gap between when we get control returned to use after
  // submitting and waiting for idle and that will be the slop we have in the
  // timings in the tracy UI.
  if (context->time_domain == VK_TIME_DOMAIN_DEVICE_EXT) {
    // Submit a device timestamp.
    VkCommandBuffer command_buffer =
        iree_hal_vulkan_tracing_begin_command_buffer(context);
    if (command_buffer != VK_NULL_HANDLE) {
      syms->vkCmdWriteTimestamp(command_buffer,
                                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                context->query_pool, 0);
      iree_hal_vulkan_tracing_submit_command_buffer(context, command_buffer);
    }

    // Query the timestamp from the host and the device.
    *out_cpu_time = iree_tracing_time();
    syms->vkGetQueryPoolResults(
        *context->logical_device, context->query_pool, 0, 1,
        sizeof(*out_gpu_time), out_gpu_time, sizeof(*out_gpu_time),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    // Reset the query used.
    iree_hal_vulkan_tracing_reset_query_pool(context, 0, 1);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // From the spec:
  // The maximum deviation may vary between calls to
  // vkGetCalibratedTimestampsEXT even for the same set of time domains due to
  // implementation and platform specific reasons. It is the application’s
  // responsibility to assess whether the returned maximum deviation makes the
  // timestamp values suitable for any particular purpose and can choose to
  // re-issue the timestamp calibration call pursuing a lower devation value.
  // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetCalibratedTimestampsEXT.html
  //
  // We perform a small number of queries here and find the minimum deviation
  // across all of them to get an average lower bound on the maximum deviation
  // from any particular query. We then use that as our baseline (plus some
  // slop) to see if calibration events in the future are reasonable.
  VkCalibratedTimestampInfoEXT timestamp_infos[2];
  timestamp_infos[0].sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT;
  timestamp_infos[0].pNext = NULL;
  timestamp_infos[0].timeDomain = VK_TIME_DOMAIN_DEVICE_EXT;
  timestamp_infos[1].sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT;
  timestamp_infos[1].pNext = NULL;
  timestamp_infos[1].timeDomain = context->time_domain;
  uint64_t max_deviations[IREE_HAL_VULKAN_TRACING_MAX_DEVIATION_PROBE_COUNT];
  IREE_TRACE_ZONE_BEGIN_NAMED(z1, "vkGetCalibratedTimestampsEXT");
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(max_deviations); ++i) {
    uint64_t timestamps[2] = {0, 0};
    syms->vkGetCalibratedTimestampsEXT(
        *context->logical_device, IREE_ARRAYSIZE(timestamps), timestamp_infos,
        timestamps, &max_deviations[i]);
  }
  IREE_TRACE_ZONE_END(z1);
  uint64_t min_deviation = max_deviations[0];
  for (iree_host_size_t i = 1; i < IREE_ARRAYSIZE(max_deviations); ++i) {
    min_deviation = iree_min(min_deviation, max_deviations[i]);
  }
  context->max_expected_deviation = min_deviation * 3 / 2;

  iree_hal_vulkan_tracing_query_calibration_timestamps(
      context, &context->previous_cpu_time, out_gpu_time);
  *out_cpu_time = iree_tracing_time();

  IREE_TRACE_ZONE_END(z0);
}

// Performs a periodic calibration (if supported) and sends the data to tracy.
// Over time the host and device clocks may drift (especially with power events)
// and by frequently performing this we ensure that the samples we are sending
// to tracy are able to be correlated.
void iree_hal_vulkan_tracing_perform_calibration(
    iree_hal_vulkan_tracing_context_t* context) {
  if (context->time_domain == VK_TIME_DOMAIN_DEVICE_EXT) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  uint64_t cpu_time = 0;
  uint64_t gpu_time = 0;
  iree_hal_vulkan_tracing_query_calibration_timestamps(context, &cpu_time,
                                                       &gpu_time);

  uint64_t tracy_time = iree_tracing_time();
  if (cpu_time > context->previous_cpu_time) {
    uint64_t cpu_delta = cpu_time - context->previous_cpu_time;
    context->previous_cpu_time = cpu_time;
    iree_tracing_gpu_context_calibrate(context->id, cpu_delta, tracy_time,
                                       gpu_time);
  }

  IREE_TRACE_ZONE_END(z0);
}

// Prepares the VkQueryPool backing storage for our query ringbuffer.
static void iree_hal_vulkan_tracing_prepare_query_pool(
    iree_hal_vulkan_tracing_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Create a query pool with the largest query capacity it can provide.
  VkQueryPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  pool_info.queryCount = IREE_HAL_VULKAN_TRACING_DEFAULT_QUERY_CAPACITY;
  pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  IREE_TRACE_ZONE_APPEND_VALUE(z0, pool_info.queryCount);
  while (context->logical_device->syms()->vkCreateQueryPool(
             *context->logical_device, &pool_info,
             context->logical_device->allocator(),
             &context->query_pool) != VK_SUCCESS) {
    pool_info.queryCount /= 2;
    IREE_TRACE_ZONE_APPEND_VALUE(z0, pool_info.queryCount);
  }
  context->query_capacity = pool_info.queryCount;

  // Perform initial reset of the query pool. All queries must be reset upon
  // creation before first use.
  iree_hal_vulkan_tracing_reset_query_pool(context, 0, context->query_capacity);

  IREE_TRACE_ZONE_END(z0);
}

// Prepares the Tracy-related GPU context that events are fed into. Each context
// will appear as a unique plot in the tracy UI with the given |queue_name|.
static void iree_hal_vulkan_tracing_prepare_gpu_context(
    iree_hal_vulkan_tracing_context_t* context,
    VkPhysicalDevice physical_device, iree_string_view_t queue_name) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // The number of nanoseconds required for a timestamp query to be incremented
  // by 1.
  VkPhysicalDeviceProperties device_properties;
  context->logical_device->syms()->vkGetPhysicalDeviceProperties(
      physical_device, &device_properties);
  float timestamp_period = device_properties.limits.timestampPeriod;

  // Perform initial calibration for tracy to be able to correlate timestamps
  // between CPU and GPU.
  uint64_t cpu_time = 0;
  uint64_t gpu_time = 0;
  iree_hal_vulkan_tracing_perform_initial_calibration(context, &cpu_time,
                                                      &gpu_time);

  // Allocate the GPU context and pass initial calibration data.
  // We may need to periodically refresh the calibration depending on the device
  // timestamp mode.
  bool is_calibrated = context->time_domain == VK_TIME_DOMAIN_DEVICE_EXT;
  context->id = iree_tracing_gpu_context_allocate(
      IREE_TRACING_GPU_CONTEXT_TYPE_VULKAN, queue_name.data, queue_name.size,
      is_calibrated, cpu_time, gpu_time, timestamp_period);

  IREE_TRACE_ZONE_END(z0);
}

// Returns the best possible platform-supported time domain, falling back to
// VK_TIME_DOMAIN_DEVICE_EXT. By default it is one that is only usable for
// device-relative calculations and that we need to perform our own hacky
// calibration on.
static VkTimeDomainEXT iree_hal_vulkan_tracing_query_time_domain(
    VkPhysicalDevice physical_device,
    iree::hal::vulkan::VkDeviceHandle* logical_device) {
  if (!logical_device->enabled_extensions().calibrated_timestamps) {
    // Calibrated timestamps extension is not available; we'll only have the
    // device domain.
    return VK_TIME_DOMAIN_DEVICE_EXT;
  }

  uint32_t time_domain_count = 0;
  if (logical_device->syms()->vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(
          physical_device, &time_domain_count, NULL) != VK_SUCCESS) {
    return VK_TIME_DOMAIN_DEVICE_EXT;
  }
  VkTimeDomainEXT* time_domains = (VkTimeDomainEXT*)iree_alloca(
      time_domain_count * sizeof(VkTimeDomainEXT));
  if (logical_device->syms()->vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(
          physical_device, &time_domain_count, time_domains) != VK_SUCCESS) {
    return VK_TIME_DOMAIN_DEVICE_EXT;
  }

  for (uint32_t i = 0; i < time_domain_count; i++) {
    switch (time_domains[i]) {
#if defined(IREE_PLATFORM_WINDOWS)
      case VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT:
        return time_domains[i];
#else
      case VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT:
      case VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT:
        // TODO(benvanik): support posix clock domains with some kind of math.
        // return time_domains[i];  -- ignored
#endif  // IREE_PLATFORM_WINDOWS
      default:
        continue;
    }
  }
  return VK_TIME_DOMAIN_DEVICE_EXT;
}

iree_status_t iree_hal_vulkan_tracing_context_allocate(
    VkPhysicalDevice physical_device,
    iree::hal::vulkan::VkDeviceHandle* logical_device, VkQueue queue,
    iree_string_view_t queue_name, VkQueue maintenance_dispatch_queue,
    iree::hal::vulkan::VkCommandPoolHandle* maintenance_command_pool,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_tracing_context_t** out_context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;

  iree_hal_vulkan_tracing_context_t* context = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*context), (void**)&context);
  if (iree_status_is_ok(status)) {
    context->logical_device = logical_device;
    context->queue = queue;
    context->host_allocator = host_allocator;
    context->time_domain = iree_hal_vulkan_tracing_query_time_domain(
        physical_device, logical_device);
    context->maintenance_dispatch_queue = maintenance_dispatch_queue;
    context->maintenance_command_pool = maintenance_command_pool;

    // Prepare the query pool and perform the initial calibration.
    iree_hal_vulkan_tracing_prepare_query_pool(context);

    // Prepare the Tracy GPU context.
    iree_hal_vulkan_tracing_prepare_gpu_context(context, physical_device,
                                                queue_name);
  }

  if (iree_status_is_ok(status)) {
    *out_context = context;
  } else {
    iree_hal_vulkan_tracing_context_free(context);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_vulkan_tracing_context_free(
    iree_hal_vulkan_tracing_context_t* context) {
  if (!context) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (context->query_pool != VK_NULL_HANDLE) {
    // Always perform a collection on shutdown.
    iree_hal_vulkan_tracing_context_collect(context, VK_NULL_HANDLE);

    auto* logical_device = context->logical_device;
    logical_device->syms()->vkDestroyQueryPool(
        *logical_device, context->query_pool, logical_device->allocator());
  }

  iree_allocator_t host_allocator = context->host_allocator;
  iree_allocator_free(host_allocator, context);

  IREE_TRACE_ZONE_END(z0);
}

uint32_t iree_hal_vulkan_tracing_context_acquire_query_id(
    iree_hal_vulkan_tracing_context_t* context) {
  uint32_t id = context->query_head;
  context->query_head = (context->query_head + 1) % context->query_capacity;
  assert(context->query_head != context->query_tail);
  return id;
}

void iree_hal_vulkan_tracing_context_collect(
    iree_hal_vulkan_tracing_context_t* context,
    VkCommandBuffer command_buffer) {
  if (!context) return;
  if (context->query_tail == context->query_head) {
    // No outstanding queries.
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  const auto& syms = context->logical_device->syms();

  while (context->query_tail != context->query_head) {
    // Compute the contiguous range of queries ready to be read.
    // If the ringbuffer wraps around we'll handle that in the next loop.
    uint32_t try_query_count =
        context->query_head < context->query_tail
            ? context->query_capacity - context->query_tail
            : context->query_head - context->query_tail;
    try_query_count = iree_min(try_query_count,
                               IREE_HAL_VULKAN_TRACING_READBACK_QUERY_CAPACITY);

    // Read back all of the queries. Note that we also are reading back the
    // availability such that we can handle partial readiness of the outstanding
    // range of queries.
    uint32_t query_base = context->query_tail;
    if (syms->vkGetQueryPoolResults(
            *context->logical_device, context->query_pool, query_base,
            try_query_count, sizeof(context->readback_buffer),
            context->readback_buffer, sizeof(iree_hal_vulkan_timestamp_query_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) !=
        VK_SUCCESS) {
      break;
    }

    // Scan and feed the times to tracy, stopping when we hit the first
    // unavailable query.
    uint32_t read_query_count = 0;
    for (uint32_t i = 0; i < try_query_count; ++i) {
      if (context->readback_buffer[i].availability == 0) break;
      read_query_count = i + 1;
      iree_tracing_gpu_zone_notify(context->id, (uint16_t)(query_base + i),
                                   context->readback_buffer[i].timestamp);
    }

    // Reset the range of queries read back.
    if (command_buffer != VK_NULL_HANDLE) {
      syms->vkCmdResetQueryPool(command_buffer, context->query_pool, query_base,
                                read_query_count);
    } else {
      iree_hal_vulkan_tracing_reset_query_pool(context, query_base,
                                               read_query_count);
    }

    context->query_tail += read_query_count;
    if (context->query_tail >= context->query_capacity) {
      context->query_tail = 0;
    }
  }

  // Run calibration - we could do this less frequently in cases where collect
  // is called every submission, however it's relatively cheap compared to all
  // this other tracing overhead.
  iree_hal_vulkan_tracing_perform_calibration(context);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_vulkan_tracing_zone_begin_impl(
    iree_hal_vulkan_tracing_context_t* context, VkCommandBuffer command_buffer,
    const iree_tracing_location_t* src_loc) {
  if (!context) return;

  uint32_t query_id = iree_hal_vulkan_tracing_context_acquire_query_id(context);
  context->logical_device->syms()->vkCmdWriteTimestamp(
      command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, context->query_pool,
      query_id);

  iree_tracing_gpu_zone_begin(context->id, (uint16_t)query_id, src_loc);
}

void iree_hal_vulkan_tracing_zone_begin_external_impl(
    iree_hal_vulkan_tracing_context_t* context, VkCommandBuffer command_buffer,
    const char* file_name, size_t file_name_length, uint32_t line,
    const char* function_name, size_t function_name_length, const char* name,
    size_t name_length) {
  if (!context) return;

  uint32_t query_id = iree_hal_vulkan_tracing_context_acquire_query_id(context);
  context->logical_device->syms()->vkCmdWriteTimestamp(
      command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, context->query_pool,
      query_id);

  iree_tracing_gpu_zone_begin_external(
      context->id, (uint16_t)query_id, file_name, file_name_length, line,
      function_name, function_name_length, name, name_length);
}

void iree_hal_vulkan_tracing_zone_end_impl(
    iree_hal_vulkan_tracing_context_t* context,
    VkCommandBuffer command_buffer) {
  if (!context) return;

  uint32_t query_id = iree_hal_vulkan_tracing_context_acquire_query_id(context);
  context->logical_device->syms()->vkCmdWriteTimestamp(
      command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, context->query_pool,
      query_id);

  iree_tracing_gpu_zone_end(context->id, (uint16_t)query_id);
}

#else

iree_status_t iree_hal_vulkan_tracing_context_allocate(
    VkPhysicalDevice physical_device,
    iree::hal::vulkan::VkDeviceHandle* logical_device, VkQueue queue,
    iree_string_view_t queue_name, VkQueue maintenance_dispatch_queue,
    iree::hal::vulkan::VkCommandPoolHandle* maintenance_command_pool,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_tracing_context_t** out_context) {
  *out_context = NULL;
  return iree_ok_status();
}

void iree_hal_vulkan_tracing_context_free(
    iree_hal_vulkan_tracing_context_t* context) {}

void iree_hal_vulkan_tracing_context_collect(
    iree_hal_vulkan_tracing_context_t* context,
    VkCommandBuffer command_buffer) {}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
