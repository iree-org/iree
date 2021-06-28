// Copyrieht 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/vulkan/vendor_specific_profiler.h"

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

#include <chrono>
#include <thread>

#include "iree/base/internal/flags.h"
#include "iree/base/internal/threading.h"
#include "iree/base/time.h"
#include "iree/hal/vulkan/mali_profiler.h"

IREE_FLAG(int32_t, vulkan_counter_sampling_interval, 100,
          "Time interval in microsecond for GPU counter sampling");

typedef enum iree_vulkan_vendor_e {
  IREE_VULKAN_VENDOR_ARM = 0,
} iree_vulkan_vendor_t;

struct iree_hal_vulkan_vendor_specific_profiler_context_t {
  iree_allocator_t host_allocator;

  // The thread for continuously sampling GPU counters.
  iree_thread_t* sampling_thread;
  // Whether the sampling thread should keep sampling.
  // Used by the main thread to singal to the sampling thread to continue
  // polling.
  iree_atomic_int32_t should_keep_sampling;
  // Whether the sampling thread has already stopped.
  // Used by the sampling thread to singal to the main thread that cleanup
  // work can be performed.
  iree_notification_t sampling_has_stopped;

  iree_vulkan_vendor_t vendor;
  union {
    iree_hal_vulkan_mali_profiler_context_t* arm;
  } profiler;
};

iree_status_t iree_hal_vulkan_vendor_specific_profiler_context_allocate(
    VkPhysicalDevice physical_device,
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_vendor_specific_profiler_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(out_context);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_context = NULL;

  // Query the physical device name to determine which vendor it is.
  VkPhysicalDeviceProperties physical_device_properties;
  logical_device->syms()->vkGetPhysicalDeviceProperties(
      physical_device, &physical_device_properties);

  // We only support Mali GPUs for now. Return okay status for others as vendor
  // specific profiling is an add-on feature; it's fine to not support.
  iree_string_view_t device_name =
      iree_make_string_view(physical_device_properties.deviceName,
                            strlen(physical_device_properties.deviceName));
  if (!iree_string_view_starts_with(device_name,
                                    iree_string_view_literal("Mali"))) {
    return iree_ok_status();
  }

  iree_hal_vulkan_vendor_specific_profiler_context_t* context = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*context),
                                             (void**)&context));

  context->host_allocator = host_allocator;
  context->sampling_thread = NULL;
  iree_atomic_store_int32(&context->should_keep_sampling, 0,
                          iree_memory_order_relaxed);

  context->vendor = IREE_VULKAN_VENDOR_ARM;
  iree_hal_vulkan_mali_profiler_context_allocate(host_allocator,
                                                 &context->profiler.arm);
  *out_context = context;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_vulkan_vendor_specific_profiler_context_free(
    iree_hal_vulkan_vendor_specific_profiler_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!context) return;

  iree_allocator_t host_allocator = context->host_allocator;
  switch (context->vendor) {
    case IREE_VULKAN_VENDOR_ARM:
      iree_hal_vulkan_mali_profiler_context_free(host_allocator,
                                                 context->profiler.arm);
      break;
  }

  iree_allocator_free(host_allocator, context);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_vulkan_vendor_specific_profiler_sample(
    iree_hal_vulkan_vendor_specific_profiler_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT(context->sampling_thread);
  IREE_TRACE_ZONE_BEGIN(z0);

  switch (context->vendor) {
    case IREE_VULKAN_VENDOR_ARM:
      iree_hal_vulkan_mali_profiler_sample(
          context->profiler.arm,
          +[](const char* counter_name, int64_t sample_value) {
            IREE_TRACE_PLOT_VALUE_I64(counter_name, sample_value);
          });
      break;
  }

  IREE_TRACE_ZONE_END(z0);
}

// Wraps around the sampling function in a loop with stop singal check and time
// interval wait.
static int iree_hal_vulkan_vendor_specific_profiler_sample_loop(void* args) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_duration_t interval = FLAG_vulkan_counter_sampling_interval * 1000;
  iree_time_t end_time = iree_relative_timeout_to_deadline_ns(interval);

  auto* context =
      reinterpret_cast<iree_hal_vulkan_vendor_specific_profiler_context_t*>(
          args);
  while (iree_atomic_load_int32(&context->should_keep_sampling,
                                iree_memory_order_relaxed) != 0) {
    iree_hal_vulkan_vendor_specific_profiler_sample(context);
    iree_time_t now = iree_time_now();
    if (now < end_time) {
      // Oh.. There should exist an API for this in IREE's threading.h..
      std::this_thread::sleep_for(std::chrono::nanoseconds(end_time - now));
    }
    end_time += interval;
  }
  // Notify the main thread that the sampling thread is exiting.
  iree_notification_post(&context->sampling_has_stopped, IREE_ALL_WAITERS);

  IREE_TRACE_ZONE_END(z0);
  return 0;
}

iree_status_t iree_hal_vulkan_vendor_specific_profiler_start(
    iree_hal_vulkan_vendor_specific_profiler_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (context->sampling_thread) return iree_ok_status();  // Already started.

  // Start the vendor specific profiler.
  switch (context->vendor) {
    case IREE_VULKAN_VENDOR_ARM:
      iree_hal_vulkan_mali_profiler_configure_counter(
          context->profiler.arm,
          +[](const char* counter_name, uint8_t counter_type) {
            IREE_TRACE_SET_PLOT_TYPE(counter_name, counter_type);
          });
      iree_hal_vulkan_mali_profiler_start(context->profiler.arm);
      break;
  }

  iree_atomic_store_int32(&context->should_keep_sampling, 1,
                          iree_memory_order_relaxed);
  iree_notification_initialize(&context->sampling_has_stopped);

  // Create and start the thread for continuously sampling GPU counters.
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));
  params.name = iree_string_view_literal("iree-gpu-sample");
  // Give it the highest priority so that the thread can be kept active as much
  // as possible for sampling. It's not a hard guarantee; but it might help to
  // avoid missing counter numbers because of inactive thread.
  params.priority_class = IREE_THREAD_PRIORITY_CLASS_HIGHEST;
  IREE_RETURN_IF_ERROR(
      iree_thread_create(iree_hal_vulkan_vendor_specific_profiler_sample_loop,
                         context, params, context->host_allocator,
                         &context->sampling_thread),
      "failed to create GPU counter sampling thread");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_vulkan_vendor_specific_profiler_stop(
    iree_hal_vulkan_vendor_specific_profiler_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!context->sampling_thread) return;

  // Send stop signal to the sampling thread.
  iree_atomic_store_int32(&context->should_keep_sampling, 0,
                          iree_memory_order_relaxed);

  // Wait for the thread to really exit. This makes sure that it's safe to call
  // other APIs like _context_free. Otherwise, the sampling thread might still
  // need to access shared data.
  iree_wait_token_t wait_token =
      iree_notification_prepare_wait(&context->sampling_has_stopped);
  iree_notification_commit_wait(&context->sampling_has_stopped, wait_token);
  iree_notification_deinitialize(&context->sampling_has_stopped);

  iree_thread_release(context->sampling_thread);
  context->sampling_thread = NULL;

  // Stop the vendor specific profiler.
  switch (context->vendor) {
    case IREE_VULKAN_VENDOR_ARM:
      iree_hal_vulkan_mali_profiler_stop(context->profiler.arm);
      break;
  }

  IREE_TRACE_ZONE_END(z0);
}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
