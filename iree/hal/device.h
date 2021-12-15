// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DEVICE_H_
#define IREE_HAL_DEVICE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/buffer.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/descriptor_set.h"
#include "iree/hal/descriptor_set_layout.h"
#include "iree/hal/event.h"
#include "iree/hal/executable_cache.h"
#include "iree/hal/executable_layout.h"
#include "iree/hal/resource.h"
#include "iree/hal/semaphore.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// An opaque driver-specific handle to identify different devices.
typedef uintptr_t iree_hal_device_id_t;

#define IREE_HAL_DEVICE_ID_INVALID 0ull

// Describes features supported by a device.
// These flags indicate the availability of features that may be enabled at the
// request of the calling application. Note that certain features may disable
// runtime optimizations or require compilation flags to ensure the required
// metadata is present in executables.
enum iree_hal_device_feature_bits_t {
  IREE_HAL_DEVICE_FEATURE_NONE = 0u,

  // Device supports executable debugging.
  // When present executables *may* be compiled with
  // IREE_HAL_EXECUTABLE_CACHING_MODE_ENABLE_DEBUGGING and will have usable
  // debugging related methods. Note that if the input executables do not have
  // embedded debugging information they still may not be able to perform
  // disassembly or fine-grained breakpoint insertion.
  IREE_HAL_DEVICE_FEATURE_SUPPORTS_DEBUGGING = 1u << 0,

  // Device supports executable coverage information.
  // When present executables *may* be compiled with
  // IREE_HAL_EXECUTABLE_CACHING_MODE_ENABLE_COVERAGE and will produce
  // coverage buffers during dispatch. Note that input executables must have
  // partial embedded debug information to allow mapping back to source offsets.
  IREE_HAL_DEVICE_FEATURE_SUPPORTS_COVERAGE = 1u << 1,

  // Device supports executable and command queue profiling.
  // When present executables *may* be compiled with
  // IREE_HAL_EXECUTABLE_CACHING_MODE_ENABLE_PROFILING and will produce
  // profiling buffers during dispatch. Note that input executables must have
  // partial embedded debug information to allow mapping back to source offsets.
  IREE_HAL_DEVICE_FEATURE_SUPPORTS_PROFILING = 1u << 2,
};
typedef uint32_t iree_hal_device_feature_t;

// Describes an enumerated HAL device.
typedef struct iree_hal_device_info_t {
  // Opaque handle used by drivers. Not valid across driver instances.
  iree_hal_device_id_t device_id;
  // Name of the device as returned by the API.
  iree_string_view_t name;
} iree_hal_device_info_t;

// A list of semaphores and their corresponding payloads.
// When signaling each semaphore will be set to the new payload value provided.
// When waiting each semaphore must reach or exceed the payload value.
typedef struct iree_hal_semaphore_list_t {
  iree_host_size_t count;
  iree_hal_semaphore_t** semaphores;
  uint64_t* payload_values;
} iree_hal_semaphore_list_t;

// A single batch of command buffers submitted to a device queue.
// All of the wait semaphores must reach or exceed the given payload value prior
// to the batch beginning execution. Each command buffer begins execution in the
// order it is present in the list, though note that the command buffers
// execute concurrently and require internal synchronization via events if there
// are any dependencies between them. Only after all command buffers have
// completed will the signal semaphores be updated to the provided payload
// values.
//
// Matches Vulkan's VkSubmitInfo:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkSubmitInfo.html
// Note that as the HAL only models timeline semaphores we take the payload
// values directly in this struct; see:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkTimelineSemaphoreSubmitInfo.html
typedef struct iree_hal_submission_batch_t {
  // Semaphores to wait on prior to executing any command buffer.
  iree_hal_semaphore_list_t wait_semaphores;

  // Command buffers to execute, in order.
  iree_host_size_t command_buffer_count;
  iree_hal_command_buffer_t** command_buffers;

  // Semaphores to signal once all command buffers have completed execution.
  iree_hal_semaphore_list_t signal_semaphores;
} iree_hal_submission_batch_t;

// Defines how a multi-wait operation treats the results of multiple semaphores.
typedef enum iree_hal_wait_mode_e {
  // Waits for all semaphores to reach or exceed their specified values.
  IREE_HAL_WAIT_MODE_ALL = 0,
  // Waits for one or more semaphores to reach or exceed their specified values.
  IREE_HAL_WAIT_MODE_ANY = 1,
} iree_hal_wait_mode_t;

//===----------------------------------------------------------------------===//
// iree_hal_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_device_t iree_hal_device_t;

// Retains the given |device| for the caller.
IREE_API_EXPORT void iree_hal_device_retain(iree_hal_device_t* device);

// Releases the given |device| from the caller.
IREE_API_EXPORT void iree_hal_device_release(iree_hal_device_t* device);

// Returns the device identifier.
// This identifier may vary based on the runtime device type; for example, a
// Vulkan device may return `vulkan-v1.1` or `vulkan-v1.2-spec1`.
IREE_API_EXPORT iree_string_view_t
iree_hal_device_id(iree_hal_device_t* device);

// Returns the host allocator used for objects.
IREE_API_EXPORT iree_allocator_t
iree_hal_device_host_allocator(iree_hal_device_t* device);

// Returns a reference to the allocator of the device that can be used for
// allocating buffers.
IREE_API_EXPORT iree_hal_allocator_t* iree_hal_device_allocator(
    iree_hal_device_t* device);

// Trims pools and caches used by the HAL to the minimum required for live
// allocations. This can be used on low-memory conditions or when
// suspending/parking instances.
IREE_API_EXPORT
iree_status_t iree_hal_device_trim(iree_hal_device_t* device);

// Queries a configuration value as an int32_t.
// The |category| and |key| will be provided to the device driver to interpret
// in a device-specific way and if recognized the value will be converted to an
// int32_t and returned in |out_value|. Fails if the value represented by the
// key is not convertable (overflows a 32-bit integer, not a number, etc).
//
// This is roughly equivalent to the `sysconf` linux syscall
// (https://man7.org/linux/man-pages/man3/sysconf.3.html) in that the exact
// set of categories and keys available and their interpretation is
// target-dependent.
//
// Well-known queries (category :: key):
//   hal.device.id :: some-pattern-*
//   hal.device.feature :: some-pattern-*
//   hal.device.architecture :: some-pattern-*
//   hal.executable.format :: some-pattern-*
//
// Returned values must remain the same for the lifetime of the device as
// callers may cache them to avoid redundant calls.
IREE_API_EXPORT iree_status_t iree_hal_device_query_i32(
    iree_hal_device_t* device, iree_string_view_t category,
    iree_string_view_t key, int32_t* out_value);

// Synchronously executes one or more transfer operations against a queue.
// All buffers must be compatible with |device| and ranges must not overlap
// (same as with memcpy).
//
// This is a blocking operation and may incur significant overheads as
// internally it issues a command buffer with the transfer operations and waits
// for it to complete. Users should do that themselves so that the work can be
// issued concurrently and batched effectively. This is only useful as a
// fallback for implementations that require it or tools where things like I/O
// are transferred without worrying about performance. When submitting other
// work it's preferable to use iree_hal_create_transfer_command_buffer and a
// normal queue submission that allows for more fine-grained sequencing and
// amortizes the submission cost by batching other work.
//
// The transfer will begin after the optional |wait_semaphore| reaches
// |wait_value|. Behavior is undefined if no semaphore is provided and there are
// in-flight operations concurrently using the buffer ranges.
// Returns only after all transfers have completed and been flushed.
IREE_API_EXPORT iree_status_t iree_hal_device_transfer_and_wait(
    iree_hal_device_t* device, iree_hal_semaphore_t* wait_semaphore,
    uint64_t wait_value, iree_host_size_t transfer_count,
    const iree_hal_transfer_command_t* transfer_commands,
    iree_timeout_t timeout);

// Submits one or more batches of work to a device queue.
//
// The queue is selected based on the flags set in |command_categories| and the
// |queue_affinity|. As the number of available queues can vary the
// |queue_affinity| is used to hash into the available queues for the required
// categories. For example if 2 queues support transfer commands and the
// affinity is 5 the resulting queue could be index hash(5)=1. The affinity can
// thus be treated as just a way to indicate whether two submissions must be
// placed on to the same queue. Note that the exact hashing function is
// implementation dependent.
//
// The submission behavior matches Vulkan's vkQueueSubmit, with each batch
// executing its command buffers in the order they are defined but allowing the
// command buffers to complete out-of-order. See:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueueSubmit.html
IREE_API_EXPORT iree_status_t iree_hal_device_queue_submit(
    iree_hal_device_t* device, iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches);

// Submits batches of work and waits until |wait_semaphore| reaches or exceeds
// |wait_value|.
//
// This is equivalent to following iree_hal_device_queue_submit with a
// iree_hal_semaphore_wait on |wait_timeout|/|wait_value| but
// may help to reduce overhead by preventing thread wakeups, kernel calls, and
// internal tracking.
//
// See iree_hal_device_queue_submit for more information about the queuing
// behavior and iree_hal_semaphore_wait for the waiting  behavior.
IREE_API_EXPORT iree_status_t iree_hal_device_submit_and_wait(
    iree_hal_device_t* device, iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches,
    iree_hal_semaphore_t* wait_semaphore, uint64_t wait_value,
    iree_timeout_t timeout);

// Blocks the caller until the semaphores reach or exceed the specified payload
// values or the |timeout| elapses. All semaphores in |semaphore_list| must be
// created from this device (or be imported into it).
//
// |wait_mode| can be used to decide when the wait will proceed; whether *all*
// semaphores in |semaphore_list| must be signaled or whether *any* (one or
// more) can be signaled before an early return.
//
// Returns success if the wait is successful and semaphores have been signaled
// satisfying the |wait_mode|.
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED if the |timeout| elapses without the
// |wait_mode| being satisfied. Note that even on success only a subset of the
// semaphores may have been signaled and each can be queried to see which ones.
//
// Returns IREE_STATUS_ABORTED if one or more semaphores has failed. Callers can
// use iree_hal_semaphore_query on the semaphores to find the ones that have
// failed and get the status.
IREE_API_EXPORT iree_status_t iree_hal_device_wait_semaphores(
    iree_hal_device_t* device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list, iree_timeout_t timeout);

// Blocks the caller until all outstanding requests on all queues have been
// completed or the |timeout| elapses. This is equivalent to having waited
// on all semaphores outstanding at the time of the call, meaning that if new
// work is submitted by another thread it may not be waited on prior to this
// call returning.
//
// Returns success if the device reaches an idle point during the call.
//
// Returns DEADLINE_EXCEEDED if the |timeout| elapses without the device having
// become idle.
IREE_API_EXPORT iree_status_t
iree_hal_device_wait_idle(iree_hal_device_t* device, iree_timeout_t timeout);

//===----------------------------------------------------------------------===//
// iree_hal_device_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_device_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_device_t* device);

  iree_string_view_t(IREE_API_PTR* id)(iree_hal_device_t* device);

  iree_allocator_t(IREE_API_PTR* host_allocator)(iree_hal_device_t* device);
  iree_hal_allocator_t*(IREE_API_PTR* device_allocator)(
      iree_hal_device_t* device);

  iree_status_t(IREE_API_PTR* trim)(iree_hal_device_t* device);

  iree_status_t(IREE_API_PTR* query_i32)(iree_hal_device_t* device,
                                         iree_string_view_t category,
                                         iree_string_view_t key,
                                         int32_t* out_value);

  iree_status_t(IREE_API_PTR* create_command_buffer)(
      iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
      iree_hal_command_category_t command_categories,
      iree_hal_queue_affinity_t queue_affinity,
      iree_hal_command_buffer_t** out_command_buffer);

  iree_status_t(IREE_API_PTR* create_descriptor_set)(
      iree_hal_device_t* device, iree_hal_descriptor_set_layout_t* set_layout,
      iree_host_size_t binding_count,
      const iree_hal_descriptor_set_binding_t* bindings,
      iree_hal_descriptor_set_t** out_descriptor_set);

  iree_status_t(IREE_API_PTR* create_descriptor_set_layout)(
      iree_hal_device_t* device,
      iree_hal_descriptor_set_layout_usage_type_t usage_type,
      iree_host_size_t binding_count,
      const iree_hal_descriptor_set_layout_binding_t* bindings,
      iree_hal_descriptor_set_layout_t** out_descriptor_set_layout);

  iree_status_t(IREE_API_PTR* create_event)(iree_hal_device_t* device,
                                            iree_hal_event_t** out_event);

  iree_status_t(IREE_API_PTR* create_executable_cache)(
      iree_hal_device_t* device, iree_string_view_t identifier,
      iree_hal_executable_cache_t** out_executable_cache);

  iree_status_t(IREE_API_PTR* create_executable_layout)(
      iree_hal_device_t* device, iree_host_size_t push_constants,
      iree_host_size_t set_layout_count,
      iree_hal_descriptor_set_layout_t** set_layouts,
      iree_hal_executable_layout_t** out_executable_layout);

  iree_status_t(IREE_API_PTR* create_semaphore)(
      iree_hal_device_t* device, uint64_t initial_value,
      iree_hal_semaphore_t** out_semaphore);

  iree_status_t(IREE_API_PTR* queue_submit)(
      iree_hal_device_t* device, iree_hal_command_category_t command_categories,
      iree_hal_queue_affinity_t queue_affinity, iree_host_size_t batch_count,
      const iree_hal_submission_batch_t* batches);

  iree_status_t(IREE_API_PTR* submit_and_wait)(
      iree_hal_device_t* device, iree_hal_command_category_t command_categories,
      iree_hal_queue_affinity_t queue_affinity, iree_host_size_t batch_count,
      const iree_hal_submission_batch_t* batches,
      iree_hal_semaphore_t* wait_semaphore, uint64_t wait_value,
      iree_timeout_t timeout);

  iree_status_t(IREE_API_PTR* wait_semaphores)(
      iree_hal_device_t* device, iree_hal_wait_mode_t wait_mode,
      const iree_hal_semaphore_list_t* semaphore_list, iree_timeout_t timeout);

  iree_status_t(IREE_API_PTR* wait_idle)(iree_hal_device_t* device,
                                         iree_timeout_t timeout);
} iree_hal_device_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_device_vtable_t);

IREE_API_EXPORT void iree_hal_device_destroy(iree_hal_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DEVICE_H_
