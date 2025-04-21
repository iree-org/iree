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
#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/channel.h"
#include "iree/hal/channel_provider.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/event.h"
#include "iree/hal/executable_cache.h"
#include "iree/hal/fence.h"
#include "iree/hal/file.h"
#include "iree/hal/queue.h"
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

#define IREE_HAL_DEVICE_ID_DEFAULT 0ull

// Describes features supported by a device.
// These flags indicate the availability of features that may be enabled at the
// request of the calling application. Note that certain features may disable
// runtime optimizations or require compilation flags to ensure the required
// metadata is present in executables.
typedef uint64_t iree_hal_device_feature_t;
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

// Describes an enumerated HAL device.
typedef struct iree_hal_device_info_t {
  // Opaque handle used by drivers. Not valid across driver instances.
  iree_hal_device_id_t device_id;
  // Stable driver-specific path used to reference the device.
  iree_string_view_t path;
  // Human-readable name of the device as returned by the API.
  iree_string_view_t name;
} iree_hal_device_info_t;

// Defines what information is captured during profiling.
// Not all implementations will support all modes.
typedef uint64_t iree_hal_device_profiling_mode_t;
enum iree_hal_device_profiling_mode_bits_t {
  IREE_HAL_DEVICE_PROFILING_MODE_NONE = 0u,

  // Capture queue operations such as command buffer submissions and the
  // transfer/dispatch commands within them. This gives a high-level overview
  // of HAL API usage with minimal overhead.
  IREE_HAL_DEVICE_PROFILING_MODE_QUEUE_OPERATIONS = 1u << 0,

  // Capture aggregated dispatch performance counters across all commands within
  // the profiled range.
  IREE_HAL_DEVICE_PROFILING_MODE_DISPATCH_COUNTERS = 1u << 1,

  // Capture detailed executable performance counters correlated to source
  // locations. This can have a significant performance impact and should only
  // be used when investigating the performance of an individual dispatch.
  IREE_HAL_DEVICE_PROFILING_MODE_EXECUTABLE_COUNTERS = 1u << 2,
};

// Controls profiling options.
typedef struct iree_hal_device_profiling_options_t {
  // Defines what kind of profiling information is captured.
  iree_hal_device_profiling_mode_t mode;

  // A file system path where profile data will be written if supported by the
  // profiling implementation. Depending on the tool this may be a template
  // path/prefix for a unique per capture name or a full path that will be
  // overwritten each capture.
  const char* file_path;
} iree_hal_device_profiling_options_t;

// Bitfield specifying flags controlling an async allocation operation.
typedef uint64_t iree_hal_alloca_flags_t;
enum iree_hal_alloca_flag_bits_t {
  IREE_HAL_ALLOCA_FLAG_NONE = 0,

  // Buffer lifetime is indeterminate indicating that the compiler or
  // application allocating the buffer is unable to determine when it is safe to
  // deallocate the buffer. Explicit deallocation requests are ignored and the
  // buffer deallocation will happen synchronously when the last remaining
  // reference to the buffer is released.
  IREE_HAL_ALLOCA_FLAG_INDETERMINATE_LIFETIME = 1ull << 0,
};

// Bitfield specifying flags controlling an async deallocation operation.
typedef uint64_t iree_hal_dealloca_flags_t;
enum iree_hal_dealloca_flag_bits_t {
  IREE_HAL_DEALLOCA_FLAG_NONE = 0,

  // The provided device and queue affinity will be overridden with the origin
  // of the allocation as defined by the placement, if available.
  // If the buffer has no origin device (imported heap buffers and other rare
  // cases) or was not allocated asynchronously the provided device and queue
  // affinity will be used to insert a queue barrier. Callers must ensure the
  // provided device is compatible with the fences provided.
  IREE_HAL_DEALLOCA_FLAG_PREFER_ORIGIN = 1ull << 0,
};

// Bitfield specifying flags controlling a file read operation.
typedef uint64_t iree_hal_read_flags_t;
enum iree_hal_read_flag_bits_t {
  IREE_HAL_READ_FLAG_NONE = 0,
};

// Bitfield specifying flags controlling a file write operation.
typedef uint64_t iree_hal_write_flags_t;
enum iree_hal_write_flag_bits_t {
  IREE_HAL_WRITE_FLAG_NONE = 0,
};

// Bitfield specifying flags controlling an execution operation.
typedef uint64_t iree_hal_execute_flags_t;
enum iree_hal_execute_flag_bits_t {
  IREE_HAL_EXECUTE_FLAG_NONE = 0,
};

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

// Replaces the default device memory allocator.
// The |new_allocator| will be retained for the lifetime of the device or until
// the allocator is replaced again. The common usage pattern is to shim the
// default allocator with a wrapper:
//   // Retain the existing allocator in the new wrapper.
//   wrap_allocator(iree_hal_device_allocator(device), &new_allocator);
//   // Update the device to use the wrapper for allocations.
//   iree_hal_device_replace_allocator(device, new_allocator);
//
// WARNING: this is not thread-safe and must only be performed when the device
// is idle and all buffers that may have been allocated from the existing
// allocator have been released. In general the only safe time to call this is
// immediately after device creation and before any buffers have been allocated.
// Beware: there are no internal checks for this condition!
//
// TODO(benvanik): remove this method and instead allow allocators to be
// composed without the safety caveats. This may take the form of unbound
// allocators that the device can inject the base allocator into. Another
// approach would be to replace the singular allocator with queue-specific pools
// and make the user register those pools explicitly with the implementation
// they desire.
IREE_API_EXPORT void iree_hal_device_replace_allocator(
    iree_hal_device_t* device, iree_hal_allocator_t* new_allocator);

// Replaces the current collective channel provider.
// The |new_provider| will be retained for the lifetime of the device or until
// the provider is replaced again.
//
// WARNING: this is not thread-safe and must only be performed when the device
// is idle and all channels that may have been created from the existing
// provider have been released. In general the only safe time to call this is
// immediately after device creation and before any channels have been created.
// Beware: there are no internal checks for this condition!
IREE_API_EXPORT void iree_hal_device_replace_channel_provider(
    iree_hal_device_t* device, iree_hal_channel_provider_t* new_provider);

// Trims pools and caches used by the HAL to the minimum required for live
// allocations. This can be used on low-memory conditions or when
// suspending/parking instances.
IREE_API_EXPORT
iree_status_t iree_hal_device_trim(iree_hal_device_t* device);

// Queries a configuration value as an int64_t.
// The |category| and |key| will be provided to the device driver to interpret
// in a device-specific way and if recognized the value will be converted to an
// int64_t and returned in |out_value|. Fails if the value represented by the
// key is not convertable.
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
IREE_API_EXPORT iree_status_t iree_hal_device_query_i64(
    iree_hal_device_t* device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value);

// Queries in what ways the given |semaphore| may be used with |device|.
IREE_API_EXPORT iree_hal_semaphore_compatibility_t
iree_hal_device_query_semaphore_compatibility(iree_hal_device_t* device,
                                              iree_hal_semaphore_t* semaphore);

// Reserves and returns a device-local queue-ordered transient buffer.
// The allocation will not be committed until the entire |wait_semaphore_list|
// has been reached. Once the storage is available for use the
// |signal_semaphore_list| will be signaled. The contents of the buffer are
// undefined until signaled even if all waits have been resolved and callers
// must always wait for the signal.
//
// For optimal performance and minimal memory consumption the returned buffer
// should be deallocated using iree_hal_device_queue_dealloca as soon as
// possible. It's still safe to synchronously release the buffer but the
// lifetime will then be controlled by all potential retainers.
//
// Usage:
//   iree_hal_device_queue_alloca(wait(0), signal(1), &buffer);
//   iree_hal_device_queue_execute(wait(1), signal(2), commands...);
//   iree_hal_device_queue_dealloca(wait(2), signal(3), buffer);
IREE_API_EXPORT iree_status_t iree_hal_device_queue_alloca(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer);

// Deallocates a queue-ordered transient buffer.
// The deallocation will not be made until the entire |wait_semaphore_list| has
// been reached. Once the storage is available for reuse the
// |signal_semaphore_list| will be signaled. After all waits have been resolved
// the contents of the buffer are immediately undefined even if the signal has
// not yet occurred. If the buffer was not allocated asynchronously a barrier
// will be inserted to preserve fence timelines.
//
// Deallocations will only be queue-ordered if the |buffer| was originally
// allocated with iree_hal_device_queue_alloca. Any synchronous allocations will
// be ignored and deallocated when the |buffer| has been released but a queue
// barrier will be inserted to preserve the timeline.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_dealloca(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags);

// Enqueues a single queue-ordered fill operation.
// The |target_buffer| must be visible to the device queue performing the fill.
//
// WARNING: individual fills have a high overhead and batching should be
// performed by the caller instead of calling this multiple times. The
// iree_hal_create_transfer_command_buffer utility makes it easy to create
// batches of transfer operations (fill, update, copy) and is only a few lines
// more code.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_fill(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags);

// Enqueues a single queue-ordered buffer update operation.
// The provided |source_buffer| will be captured and need not remain live or
// unchanged while the operation is queued. The |target_buffer| must be visible
// to the device queue performing the update.
//
// Some implementations may have limits on the size of the update or may perform
// poorly if the size is larger than an implementation-defined limit. Updates
// should be kept as small and infrequent as possible.
//
// WARNING: individual copies have a high overhead and batching should be
// performed by the caller instead of calling this multiple times. The
// iree_hal_create_transfer_command_buffer utility makes it easy to create
// batches of transfer operations (fill, update, copy) and is only a few lines
// more code.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_update(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags);

// Enqueues a single queue-ordered copy operation.
// The |source_buffer| and |target_buffer| must both be visible to the device
// queue performing the copy.
//
// WARNING: individual copies have a high overhead and batching should be
// performed by the caller instead of calling this multiple times. The
// iree_hal_create_transfer_command_buffer utility makes it easy to create
// batches of transfer operations (fill, update, copy) and is only a few lines
// more code.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_copy(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags);

// Enqueues a file read operation that streams a segment of the |source_file|
// defined by the |source_offset| and |length| into the HAL |target_buffer| at
// the specified |target_offset|. The |queue_affinity| should be set to where
// the target buffer will be consumed. The source file must have read permission
// and the target buffer must have transfer-target usage.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_read(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags);

// Enqueues a file write operation that streams a segment of the HAL
// |source_buffer| defined by the |source_offset| and |length| into the
// |target_file| at the specified |target_offset|. The |queue_affinity| should
// be set to where the source buffer was produced. The source buffer must have
// transfer-source usage and the target file must have write permission.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_write(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags);

// Executes a command buffer on a device queue.
// No commands will execute until the wait fence has been reached and the signal
// fence will be signaled when all commands have completed. If a command buffer
// is omitted this will act as a barrier.
//
// The queue is selected based on the command buffer submitted and the
// |queue_affinity|. As the number of available queues can vary the
// |queue_affinity| is used to hash into the available queues for the required
// categories. For example if 2 queues support transfer commands and the
// affinity is 5 the resulting queue could be index hash(5)=1. The affinity can
// thus be treated as just a way to indicate whether two submissions must be
// placed on to the same queue. Note that the exact hashing function is
// implementation dependent.
//
// A optional binding table must be provided if the command buffer has indirect
// bindings and may otherwise be `iree_hal_buffer_binding_table_empty()`. The
// binding table contents will be captured during the call and need not persist
// after the call returns.
//
// The submission behavior matches Vulkan's vkQueueSubmit, with each submission
// executing its command buffers in the order they are defined but allowing the
// command buffers to complete out-of-order. See:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueueSubmit.html
IREE_API_EXPORT iree_status_t iree_hal_device_queue_execute(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags);

// Enqueues a barrier waiting for |wait_semaphore_list| and signaling
// |signal_semaphore_list| when reached.
// Equivalent to iree_hal_device_queue_execute with no command buffers.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_barrier(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_execute_flags_t flags);

// Flushes any locally-pending submissions in the queue.
// When submitting many queue operations this can be used to eagerly flush
// earlier submissions while later ones are still being constructed.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_flush(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity);

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
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout);

// Begins a profile capture on |device| with the given |options|.
// This will use an implementation-defined profiling API to capture all
// supported device operations until the iree_hal_device_profiling_end is
// called. If the device or current build configuration do not support profiling
// this method is a no-op. See implementation-specific device creation APIs and
// driver module registration for more information.
//
// WARNING: the device must be idle before calling this method. Behavior is
// undefined if there are any in-flight or pending queue operations or access
// from another thread while profiling is starting/stopping.
//
// WARNING: profiling in any mode can dramatically increase overhead with some
// modes being significantly more expensive in both host and device time enough
// to invalidate performance numbers from other mechanisms (perf/tracy/etc).
// When measuring end-to-end performance use only
// IREE_HAL_DEVICE_PROFILING_MODE_QUEUE_OPERATIONS.
//
// Examples of APIs this maps to (where supported):
// - CPU: perf_event_open/close or vendor APIs
// - CUDA: cuProfilerStart/cuProfilerStop
// - Direct3D: PIXBeginCapture/PIXEndCapture
// - Metal: [MTLCaptureManager startCapture/stopCapture]
// - Vulkan: vkAcquireProfilingLockKHR/vkReleaseProfilingLockKHR +
//           RenderDoc StartFrameCapture/EndFrameCapture
IREE_API_EXPORT iree_status_t iree_hal_device_profiling_begin(
    iree_hal_device_t* device,
    const iree_hal_device_profiling_options_t* options);

// Flushes any pending profiling data. May be a no-op.
IREE_API_EXPORT iree_status_t
iree_hal_device_profiling_flush(iree_hal_device_t* device);

// Ends a profile previous started with iree_hal_device_profiling_begin.
// The device must be idle before calling this method.
IREE_API_EXPORT iree_status_t
iree_hal_device_profiling_end(iree_hal_device_t* device);

//===----------------------------------------------------------------------===//
// iree_hal_device_list_t
//===----------------------------------------------------------------------===//

// A fixed-size list of retained devices.
typedef struct iree_hal_device_list_t {
  iree_allocator_t host_allocator;
  iree_host_size_t capacity;
  iree_host_size_t count;
  iree_hal_device_t* devices[];
} iree_hal_device_list_t;

// Allocates an empty device list with the given capacity.
IREE_API_EXPORT iree_status_t iree_hal_device_list_allocate(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    iree_hal_device_list_t** out_list);

// Frees a device |list|.
IREE_API_EXPORT void iree_hal_device_list_free(iree_hal_device_list_t* list);

// Pushes a |device| onto the |list| and retains it.
IREE_API_EXPORT iree_status_t iree_hal_device_list_push_back(
    iree_hal_device_list_t* list, iree_hal_device_t* device);

// Returns the device at index |i| in the |list| or NULL if out of range.
// Callers must retain the device if it's possible for the returned pointer to
// live beyond the list.
IREE_API_EXPORT iree_hal_device_t* iree_hal_device_list_at(
    const iree_hal_device_list_t* list, iree_host_size_t i);

//===----------------------------------------------------------------------===//
// iree_hal_device_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_device_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_device_t* device);

  iree_string_view_t(IREE_API_PTR* id)(iree_hal_device_t* device);

  iree_allocator_t(IREE_API_PTR* host_allocator)(iree_hal_device_t* device);
  iree_hal_allocator_t*(IREE_API_PTR* device_allocator)(
      iree_hal_device_t* device);
  void(IREE_API_PTR* replace_device_allocator)(
      iree_hal_device_t* device, iree_hal_allocator_t* new_allocator);
  void(IREE_API_PTR* replace_channel_provider)(
      iree_hal_device_t* device, iree_hal_channel_provider_t* new_provider);

  iree_status_t(IREE_API_PTR* trim)(iree_hal_device_t* device);

  iree_status_t(IREE_API_PTR* query_i64)(iree_hal_device_t* device,
                                         iree_string_view_t category,
                                         iree_string_view_t key,
                                         int64_t* out_value);

  iree_status_t(IREE_API_PTR* create_channel)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      iree_hal_channel_params_t params, iree_hal_channel_t** out_channel);

  iree_status_t(IREE_API_PTR* create_command_buffer)(
      iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
      iree_hal_command_category_t command_categories,
      iree_hal_queue_affinity_t queue_affinity,
      iree_host_size_t binding_capacity,
      iree_hal_command_buffer_t** out_command_buffer);

  iree_status_t(IREE_API_PTR* create_event)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      iree_hal_event_flags_t flags, iree_hal_event_t** out_event);

  iree_status_t(IREE_API_PTR* create_executable_cache)(
      iree_hal_device_t* device, iree_string_view_t identifier,
      iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache);

  iree_status_t(IREE_API_PTR* import_file)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
      iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file);

  iree_status_t(IREE_API_PTR* create_semaphore)(
      iree_hal_device_t* device, uint64_t initial_value,
      iree_hal_semaphore_flags_t flags, iree_hal_semaphore_t** out_semaphore);

  iree_hal_semaphore_compatibility_t(
      IREE_API_PTR* query_semaphore_compatibility)(
      iree_hal_device_t* device, iree_hal_semaphore_t* semaphore);

  iree_status_t(IREE_API_PTR* queue_alloca)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
      iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
      iree_hal_buffer_t** IREE_RESTRICT out_buffer);

  iree_status_t(IREE_API_PTR* queue_dealloca)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_fill)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_device_size_t length, const void* pattern,
      iree_host_size_t pattern_length, iree_hal_fill_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_update)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      const void* source_buffer, iree_host_size_t source_offset,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_device_size_t length, iree_hal_update_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_copy)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_device_size_t length, iree_hal_copy_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_read)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_file_t* source_file, uint64_t source_offset,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_device_size_t length, iree_hal_read_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_write)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
      iree_hal_file_t* target_file, uint64_t target_offset,
      iree_device_size_t length, iree_hal_write_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_execute)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_binding_table_t binding_table,
      iree_hal_execute_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_flush)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity);

  iree_status_t(IREE_API_PTR* wait_semaphores)(
      iree_hal_device_t* device, iree_hal_wait_mode_t wait_mode,
      const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout);

  iree_status_t(IREE_API_PTR* profiling_begin)(
      iree_hal_device_t* device,
      const iree_hal_device_profiling_options_t* options);
  iree_status_t(IREE_API_PTR* profiling_flush)(iree_hal_device_t* device);
  iree_status_t(IREE_API_PTR* profiling_end)(iree_hal_device_t* device);
} iree_hal_device_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_device_vtable_t);

IREE_API_EXPORT void iree_hal_device_destroy(iree_hal_device_t* device);

IREE_API_EXPORT iree_status_t iree_hal_device_queue_emulated_fill(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags);

IREE_API_EXPORT iree_status_t iree_hal_device_queue_emulated_update(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags);

IREE_API_EXPORT iree_status_t iree_hal_device_queue_emulated_copy(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DEVICE_H_
