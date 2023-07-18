// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_TRACING_H_
#define IREE_HAL_DRIVERS_VULKAN_TRACING_H_

// clang-format off: must be included before all other headers.
#include "iree/hal/drivers/vulkan/vulkan_headers.h"
// clang-format on

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/handle_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Per-queue Vulkan tracing context.
// No-op if IREE tracing is not enabled.
//
// Use the IREE_VULKAN_TRACE_* macros to trace a contiguous set of command
// buffer operations. Unlike the normal tracy macros there are no zone IDs and
// instead each queue gets an ID allocated once and passed to all tracing
// macros.
//
// Usage:
//   IREE_VULKAN_TRACE_ZONE_BEGIN(device->tracing_context, command_buffer);
//   vkCmdDispatch(command_buffer, ...);
//   IREE_VULKAN_TRACE_ZONE_END(queue->tracing_context, command_buffer);
//   ...
//   iree_hal_vulkan_tracing_context_collect(queue->tracing_context,
//                                           command_buffer);
//   vkQueueSubmit(...command_buffer...);
//
// NOTE: timestamps have non-trivial side-effecting behavior on the device:
// inserting a timestamp is in the worst (and average) case just as bad as
// inserting a full global execution barrier. If two command buffer operations
// that could overlap (no barrier between them) have tracing zones placed around
// them they will execute sequentially.
//
// TODO(benvanik):
//   Each queue needs a context and maintains its own query pool. In the future
//   this should be changed to have a single query pool per device to reduce
//   bookkeeping overhead.
//
// TODO(benvanik):
//   Both a zone begin and zone end always insert timestamps leading to N*2
//   total queries, however within command buffers the end of one zone and the
//   begin of another share the same point in time. By inserting the timestamps
//   at barriers in the command buffer the query count can be reduced to N+1.
//
// TODO(benvanik):
//   vkCmdCopyQueryPoolResults is really what we should be using to do this -
//   that inserts a device-side transfer to a buffer (conceptually) that is
//   in-stream with all submissions to a queue. This changes things to a push
//   model vs. the pull one in _collect and allows us to pipeline the readbacks.
//   Instead of being limited to the query pool slots we'd only be limited by
//   the size of the buffer the copy targets allowing us to perform collection
//   much more infrequently.
//
// Thread-compatible: external synchronization is required if using from
// multiple threads (same as with VkQueue itself).
typedef struct iree_hal_vulkan_tracing_context_t
    iree_hal_vulkan_tracing_context_t;

// Allocates a tracing context for the given Vulkan queue.
// Each context must only be used with the queue it was created with.
//
// |maintenance_dispatch_queue| may be used to perform query pool maintenance
// tasks and must support graphics or compute commands.
iree_status_t iree_hal_vulkan_tracing_context_allocate(
    VkPhysicalDevice physical_device,
    iree::hal::vulkan::VkDeviceHandle* logical_device, VkQueue queue,
    iree_string_view_t queue_name, VkQueue maintenance_dispatch_queue,
    iree::hal::vulkan::VkCommandPoolHandle* maintenance_command_pool,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_tracing_context_t** out_context);

// Frees a tracing context and all associated Vulkan resources.
// All submissions using the resources must be completed prior to calling.
void iree_hal_vulkan_tracing_context_free(
    iree_hal_vulkan_tracing_context_t* context);

// Collects in-flight timestamp queries from the queue and feeds them to tracy.
// Must be called frequently (every submission, etc) to drain the backlog;
// tracing may start failing if the internal ringbuffer is exceeded.
//
// The provided |command_buffer| may receive additional bookkeeping commands
// that should have no impact on correctness or behavior. If VK_NULL_HANDLE is
// provided then collection will occur synchronously.
void iree_hal_vulkan_tracing_context_collect(
    iree_hal_vulkan_tracing_context_t* context, VkCommandBuffer command_buffer);

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

// Begins a normal zone derived on the calling |src_loc|.
// Must be perfectly nested and paired with a corresponding zone end.
void iree_hal_vulkan_tracing_zone_begin_impl(
    iree_hal_vulkan_tracing_context_t* context, VkCommandBuffer command_buffer,
    const iree_tracing_location_t* src_loc);

// Begins an external zone using the given source information.
// The provided strings will be copied into the tracy buffer.
void iree_hal_vulkan_tracing_zone_begin_external_impl(
    iree_hal_vulkan_tracing_context_t* context, VkCommandBuffer command_buffer,
    const char* file_name, size_t file_name_length, uint32_t line,
    const char* function_name, size_t function_name_length, const char* name,
    size_t name_length);

void iree_hal_vulkan_tracing_zone_end_impl(
    iree_hal_vulkan_tracing_context_t* context, VkCommandBuffer command_buffer);

// Begins a new zone with the parent function name.
#define IREE_VULKAN_TRACE_ZONE_BEGIN(context, command_buffer)                 \
  static const iree_tracing_location_t TracyConcat(                           \
      __tracy_source_location, __LINE__) = {name_literal, __FUNCTION__,       \
                                            __FILE__, (uint32_t)__LINE__, 0}; \
  iree_hal_vulkan_tracing_zone_begin_impl(                                    \
      context, command_buffer,                                                \
      &TracyConcat(__tracy_source_location, __LINE__));

// Begins an externally defined zone with a dynamic source location.
// The |file_name|, |function_name|, and optional |name| strings will be copied
// into the trace buffer and do not need to persist.
#define IREE_VULKAN_TRACE_ZONE_BEGIN_EXTERNAL(                                 \
    context, command_buffer, file_name, file_name_length, line, function_name, \
    function_name_length, name, name_length)                                   \
  iree_hal_vulkan_tracing_zone_begin_external_impl(                            \
      context, command_buffer, file_name, file_name_length, line,              \
      function_name, function_name_length, name, name_length)

// Ends the current zone. Must be passed the |zone_id| from the _BEGIN.
#define IREE_VULKAN_TRACE_ZONE_END(context, command_buffer) \
  iree_hal_vulkan_tracing_zone_end_impl(context, command_buffer)

#else

#define IREE_VULKAN_TRACE_ZONE_BEGIN(context, command_buffer)
#define IREE_VULKAN_TRACE_ZONE_BEGIN_EXTERNAL(                                 \
    context, command_buffer, file_name, file_name_length, line, function_name, \
    function_name_length, name, name_length)
#define IREE_VULKAN_TRACE_ZONE_END(context, command_buffer)

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_TRACING_H_
