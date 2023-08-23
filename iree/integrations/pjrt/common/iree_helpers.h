// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/api.h"

namespace iree::pjrt {

// Anonymous namespace containing helpers and wrappers for IREE API
// functions which can perform verbose logging when enabled. These all
// match an IREE api but will have the |iree_| prefix elided, so they are
// used as IreeApi::hal_allocator_allocate_buffer(...), which should be a
// drop-in for iree_hal_allocator_allocate_buffer(...).
namespace IreeApi {
namespace {

// Controls whether logging is printed to stderr. We may want to make this
// more configurable in the future.
const bool LOGGING_ENABLED = false;

IREE_PRINTF_ATTRIBUTE(2, 3)
void LogInvoke(const char* func, const char* fmt, ...) {
  if (LOGGING_ENABLED) {
    fprintf(stderr, ":: IREE INVOKE (%s): ", func);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fflush(stderr);
  }
}
iree_status_t HandleStatus(const char* func, iree_status_t status) {
  if (LOGGING_ENABLED) {
    if (!iree_status_is_ok(status)) {
      fprintf(stderr, " (");
      iree_status_fprint(stderr, status);
      fprintf(stderr, ")\n");
    } else {
      fprintf(stderr, " (OK)\n");
    }
  }
  return status;
}
std::string SemaphoreListToString(const iree_hal_semaphore_list_t sl) {
  std::string result;
  char fmtBuffer[64];
  for (iree_host_size_t i = 0; i < sl.count; ++i) {
    snprintf(fmtBuffer, sizeof(fmtBuffer), "%p:%" PRIu64, sl.semaphores[i],
             sl.payload_values[i]);
    if (i > 0) {
      result.append(", ");
    }
    result.append(fmtBuffer);
  }
  return result;
}
std::string FenceToString(iree_hal_fence_t* fence) {
  return SemaphoreListToString(iree_hal_fence_semaphore_list(fence));
}

iree_status_t hal_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  auto status = iree_hal_allocator_allocate_buffer(allocator, params,
                                                   allocation_size, out_buffer);
  if (LOGGING_ENABLED) {
    LogInvoke(__func__, "allocator=%p, size=%zu, buffer=%p", allocator,
              (size_t)allocation_size, *out_buffer);
  }
  return HandleStatus(__func__, status);
}

iree_status_t hal_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_params_t params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  if (LOGGING_ENABLED) {
    LogInvoke(__func__, "external_buffer=%p", external_buffer);
  }
  return HandleStatus(__func__, iree_hal_allocator_import_buffer(
                                    allocator, params, external_buffer,
                                    release_callback, out_buffer));
}

iree_status_t hal_device_queue_alloca(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  if (LOGGING_ENABLED) {
    LogInvoke(__func__, "device=%p, size=%zd, wait={%s}, signal={%s}", device,
              (size_t)allocation_size,
              SemaphoreListToString(wait_semaphore_list).c_str(),
              SemaphoreListToString(signal_semaphore_list).c_str());
  }
  return HandleStatus(__func__, iree_hal_device_queue_alloca(
                                    device, queue_affinity, wait_semaphore_list,
                                    signal_semaphore_list, pool, params,
                                    allocation_size, out_buffer));
}

iree_status_t hal_device_queue_dealloca(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer) {
  if (LOGGING_ENABLED) {
    LogInvoke(__func__, "device=%p, buffer=%p, wait={%s}, signal={%s}", device,
              buffer, SemaphoreListToString(wait_semaphore_list).c_str(),
              SemaphoreListToString(signal_semaphore_list).c_str());
  }
  return HandleStatus(__func__, iree_hal_device_queue_dealloca(
                                    device, queue_affinity, wait_semaphore_list,
                                    signal_semaphore_list, buffer));
}

iree_status_t hal_device_queue_barrier(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  if (LOGGING_ENABLED) {
    LogInvoke(__func__, "device=%p, wait={%s}, signal={%s}", device,
              SemaphoreListToString(wait_semaphore_list).c_str(),
              SemaphoreListToString(signal_semaphore_list).c_str());
  }
  return HandleStatus(__func__, iree_hal_device_queue_barrier(
                                    device, queue_affinity, wait_semaphore_list,
                                    signal_semaphore_list));
}

iree_status_t hal_device_queue_execute(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers) {
  if (LOGGING_ENABLED) {
    LogInvoke(__func__, "device=%p, wait={%s}, signal={%s}", device,
              SemaphoreListToString(wait_semaphore_list).c_str(),
              SemaphoreListToString(signal_semaphore_list).c_str());
  }
  return HandleStatus(__func__, iree_hal_device_queue_execute(
                                    device, queue_affinity, wait_semaphore_list,
                                    signal_semaphore_list, command_buffer_count,
                                    command_buffers));
}

iree_status_t hal_fence_create(iree_host_size_t capacity,
                               iree_allocator_t host_allocator,
                               iree_hal_fence_t** out_fence) {
  auto status = iree_hal_fence_create(capacity, host_allocator, out_fence);
  if (LOGGING_ENABLED) {
    LogInvoke(__func__, "capacity=%zu, fence=%p", (size_t)capacity, *out_fence);
  }
  return HandleStatus(__func__, status);
}

iree_status_t hal_fence_create_at(iree_hal_semaphore_t* semaphore,
                                  uint64_t value,
                                  iree_allocator_t host_allocator,
                                  iree_hal_fence_t** out_fence) {
  auto status =
      iree_hal_fence_create_at(semaphore, value, host_allocator, out_fence);
  if (LOGGING_ENABLED) {
    LogInvoke(__func__, "semaphore=%p, value=%" PRIu64 ", fence=%p", semaphore,
              value, *out_fence);
  }
  return HandleStatus(__func__, status);
}

iree_status_t hal_fence_extend(iree_hal_fence_t* into_fence,
                               iree_hal_fence_t* from_fence) {
  if (LOGGING_ENABLED) {
    LogInvoke(__func__, "into_fence=%p, from_fence=%p", into_fence, from_fence);
  }
  return HandleStatus(__func__, iree_hal_fence_extend(into_fence, from_fence));
}

iree_status_t hal_fence_insert(iree_hal_fence_t* fence,
                               iree_hal_semaphore_t* semaphore,
                               uint64_t value) {
  if (LOGGING_ENABLED) {
    LogInvoke(__func__, "fence=%p, semaphore=%p, value=%" PRIu64, fence,
              semaphore, value);
  }
  return HandleStatus(__func__, iree_hal_fence_insert(fence, semaphore, value));
}

}  // namespace
}  // namespace IreeApi

}  // namespace iree::pjrt
