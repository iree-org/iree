// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/direct_command_buffer.h"

#import <Metal/Metal.h>

#include "iree/base/api.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/metal/builtin_executables.h"
#include "iree/hal/drivers/metal/executable.h"
#include "iree/hal/drivers/metal/metal_buffer.h"
#include "iree/hal/drivers/metal/metal_device.h"
#include "iree/hal/drivers/metal/staging_buffer.h"
#include "iree/hal/utils/resource_set.h"

//===------------------------------------------------------------------------------------------===//
// Segmented submission management
//===------------------------------------------------------------------------------------------===//

// Unlike Vulkan, Metal adopts a multi-level command recording model--memory/dispatch commands are
// not directly recorded into a command buffer; rather, they must go through the additional level of
// blit/compute encoders. IREE's HAL follows the flat Vulkan command buffer recording model, so we
// have a mismatch here. Implementing IREE's HAL using Metal would require switching encoders for
// interleaved memory and dispatch commands. Additionally, certain IREE HAL API features do not have
// direct mapping in Metal APIs, e.g., various forms of IREE HAL execution/memory barriers.
// Translating them would require looking at both previous and next commands to decide the proper
// mapping.
//
// Due to these reasons, it's beneficial to have a complete view of the full command buffer and
// extra flexibility during recording, in order to fixup past commands, or inspect future commands.
//
// Therefore, to implement IREE HAL command buffers using Metal, we perform two steps using a linked
// list of command segments. First we create segments (iree_hal_metal_command_buffer_prepare_* and
// iree_hal_metal_command_segment_create_*) to keep track of all IREE HAL commands and the
// associated data, and then, when finalizing the command buffer, we iterate through all the
// segments and record their contents (iree_hal_metal_command_segment_record_*) into a proper Metal
// command buffer . A linked list gives us the flexibility to organize command sequence in low
// overhead; and a deferred recording gives us the complete picture of the command buffer when
// really started recording.

// Command action kind of a command segment.
typedef enum iree_hal_metal_command_segment_action_e {
  IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_BARRIER,      // Execution/memory barrier command
  IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_DISPATCH,     // Dispatch command
  IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_FILL_BUFFER,  // Fill buffer command
  IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_COPY_BUFFER,  // Copy buffer command
} iree_hal_metal_command_segment_action_t;

// API data for execution/memory barrier command segments.
typedef struct iree_hal_metal_barrier_segment_t {
  iree_host_size_t memory_barrier_count;  // Total number of memory barriers
  iree_host_size_t buffer_barrier_count;  // Total number of buffer barriers
  // The list of buffer barriers, pointing to the end of the segment allocation.
  const iree_hal_buffer_barrier_t* buffer_barriers;
} iree_hal_metal_barrier_segment_t;
// + Additional inline allocation for holding all buffer barriers.

typedef struct iree_hal_metal_descriptor_t {
  uint32_t binding;
  iree_hal_buffer_t* buffer;
  iree_device_size_t offset;
  MTLResourceUsage usage;
} iree_hal_metal_descriptor_t;

// API data for dispatch command segments.
typedef struct iree_hal_metal_dispatch_segment_t {
  // Compute function pipeline with information required to dispatch it.
  const iree_hal_metal_pipeline_t* pipeline;

  // Threadgroup size required during dispatch.
  MTLSize threadgroup_size;

  // Workgroup count information--if |workgroups_buffer| is not nil, then indirect dispatch;
  // otherwise uses |workgroup_count| for direct dispatch.
  id<MTLBuffer> workgroups_buffer;
  iree_device_size_t workgroups_offset;
  MTLSize workgroup_count;

  // The number of descriptors bound for this dispatch.
  iree_host_size_t descriptor_count;
  // The list of bound descriptors, pointing to the end of the segment allocation.
  iree_hal_metal_descriptor_t* descriptors;

  // The number of push constant values.
  iree_host_size_t constant_count;
  // The list of push constants, pointing to the end of the segment allocation.
  int32_t* constants;
} iree_hal_metal_dispatch_segment_t;
// + Additional inline allocation for holding all bound descriptors.
// + Additional inline allocation for holding all push constants.

// API data for fill buffer command segments.
typedef struct iree_hal_metal_fill_buffer_segment_t {
  id<MTLBuffer> target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
  // The fill pattern, pointing to the end of the segment allocation.
  const void* pattern;
  iree_host_size_t pattern_length;
} iree_hal_metal_fill_buffer_segment_t;
// + Additional inline allocation for holding the fill pattern.

// API data for copy buffer command segments.
typedef struct iree_hal_metal_copy_buffer_segment_t {
  id<MTLBuffer> source_buffer;
  iree_device_size_t source_offset;
  id<MTLBuffer> target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
} iree_hal_metal_copy_buffer_segment_t;

struct iree_hal_metal_command_segment_t;
typedef struct iree_hal_metal_command_segment_t {
  struct iree_hal_metal_command_segment_t* next_segment;
  iree_hal_metal_command_segment_action_t action;
  union {
    iree_hal_metal_barrier_segment_t barrier;
    iree_hal_metal_dispatch_segment_t dispatch;
    iree_hal_metal_fill_buffer_segment_t fill_buffer;
    iree_hal_metal_copy_buffer_segment_t copy_buffer;
  };
} iree_hal_metal_command_segment_t;

typedef struct iree_hal_metal_command_segment_list_t {
  iree_hal_metal_command_segment_t* head;
  iree_hal_metal_command_segment_t* tail;
} iree_hal_metal_command_segment_list_t;

static void iree_hal_metal_command_segment_list_reset(iree_hal_metal_command_segment_list_t* list) {
  memset(list, 0, sizeof(*list));
}

static void iree_hal_metal_command_segment_list_push_front(
    iree_hal_metal_command_segment_list_t* list, iree_hal_metal_command_segment_t* segment) {
  segment->next_segment = list->head;
  list->head = segment;
  if (!list->tail) list->tail = segment;
}

static void iree_hal_metal_command_segment_list_push_back(
    iree_hal_metal_command_segment_list_t* list, iree_hal_metal_command_segment_t* segment) {
  segment->next_segment = NULL;
  if (list->tail) {
    list->tail->next_segment = segment;
    list->tail = segment;
  } else {
    list->head = list->tail = segment;
  }
}

//===------------------------------------------------------------------------------------------===//
// iree_hal_metal_command_buffer_t
//===------------------------------------------------------------------------------------------===//

typedef struct iree_hal_metal_command_buffer_t {
  iree_hal_command_buffer_t base;

  // The HAL device owning this command buffer. We need to retain it to make sure it outlive this
  // command buffer to allow access to shared resources.
  iree_hal_device_t* device;

  // The Metal command queue owning this command buffer.
  id<MTLCommandQueue> queue;

  // For polyfilling fill/copy/update buffers that are not directly supported by Metal APIs.
  iree_hal_metal_builtin_executable_t* builtin_executable;

  // Arena used for all allocations; references the shared device block pool.
  iree_arena_allocator_t arena;

  // Per-queue shared uniform staging buffer for uploading parameters to the GPU, including argument
  // buffers and buffer update source buffers.
  iree_hal_metal_staging_buffer_t* staging_buffer;

  iree_allocator_t host_allocator;

  // Maintains a reference to all resources used within the command buffer. Resets on each begin.
  iree_hal_resource_set_t* resource_set;

  // Linked list of command segments to be recorded into a command buffer.
  iree_hal_metal_command_segment_list_t segments;

  id<MTLCommandBuffer> command_buffer;

  MTLDispatchType dispatch_type;

  struct {
    // The current active compute/blit encoders for encoding compute for memory operations.
    // Metal commands are encoded into the command buffer with such encoders, and each encoder can
    // only encode the specific type of operations it supports.
    id<MTLComputeCommandEncoder> compute_encoder;
    id<MTLBlitCommandEncoder> blit_encoder;

    // MTLEven used for synchronization when we switch between blit and compute encoders.
    // Normally we would use MTLFence objects, but the difference between IREE HAL and Metal API
    // means we may see many encoder switches. It would require creating a lot GPU objects. In order
    // to avoid the cost, we just use one MTLEvent with different values for different switches.
    id<MTLEvent> encoder_event;
    // The next available encoder event value to signal/wait to/on.
    uint64_t next_encoder_event_value;
  } state;
} iree_hal_metal_command_buffer_t;

//===------------------------------------------------------------------------------------------===//
// iree_hal_metal_command_buffer_vtable APIs
//===------------------------------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t iree_hal_metal_command_buffer_vtable;

static iree_hal_metal_command_buffer_t* iree_hal_metal_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_command_buffer_vtable);
  return (iree_hal_metal_command_buffer_t*)base_value;
}

static const iree_hal_metal_command_buffer_t* iree_hal_metal_command_buffer_const_cast(
    const iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_command_buffer_vtable);
  return (const iree_hal_metal_command_buffer_t*)base_value;
}

id<MTLCommandBuffer> iree_hal_metal_direct_command_buffer_handle(
    const iree_hal_command_buffer_t* base_command_buffer) {
  const iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_const_cast(base_command_buffer);
  return command_buffer->command_buffer;
}

static void iree_hal_metal_end_compute_encoder(iree_hal_metal_command_buffer_t* command_buffer) {
  if (command_buffer->state.compute_encoder) {
    [command_buffer->state.compute_encoder endEncoding];
    [command_buffer->state.compute_encoder release];  // -1
    command_buffer->state.compute_encoder = nil;
  }
}

static void iree_hal_metal_end_blit_encoder(iree_hal_metal_command_buffer_t* command_buffer) {
  if (command_buffer->state.blit_encoder) {
    [command_buffer->state.blit_encoder endEncoding];
    [command_buffer->state.blit_encoder release];  // -1
    command_buffer->state.blit_encoder = nil;
  }
}

static void iree_hal_metal_command_buffer_reset(iree_hal_metal_command_buffer_t* command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_metal_end_blit_encoder(command_buffer);
  iree_hal_metal_end_compute_encoder(command_buffer);
  iree_hal_metal_command_segment_list_reset(&command_buffer->segments);
  iree_arena_reset(&command_buffer->arena);
  IREE_TRACE_ZONE_END(z0);
}

static id<MTLComputeCommandEncoder> iree_hal_metal_get_or_begin_compute_encoder(
    iree_hal_metal_command_buffer_t* command_buffer) {
  id<MTLCommandBuffer> metal_handle = command_buffer->command_buffer;

  // If we are switching encoders, we would need to use a fence to synchronize "one or more
  // resources across different passes within a command buffer."
  // https://developer.apple.com/documentation/metal/resource_synchronization
  uint64_t encoder_event_value = 0;
  if (command_buffer->state.blit_encoder) {
    iree_hal_metal_end_blit_encoder(command_buffer);
    encoder_event_value = command_buffer->state.next_encoder_event_value++;
    [metal_handle encodeSignalEvent:command_buffer->state.encoder_event value:encoder_event_value];
  }

  if (!command_buffer->state.compute_encoder) {
    if (encoder_event_value != 0) {
      [metal_handle encodeWaitForEvent:command_buffer->state.encoder_event
                                 value:encoder_event_value];
    }
    @autoreleasepool {  // Use @autoreleasepool to trigger the autorelease within encoder creation.
      // We manage commands dependencies and insert barriers explicitly in IREE; so use the
      // concurrent dispatch type for compute encoders.
      command_buffer->state.compute_encoder = [[metal_handle
          computeCommandEncoderWithDispatchType:command_buffer->dispatch_type] retain];  // +1
    }
  }

  return command_buffer->state.compute_encoder;
}

static id<MTLBlitCommandEncoder> iree_hal_metal_get_or_begin_blit_encoder(
    iree_hal_metal_command_buffer_t* command_buffer) {
  id<MTLCommandBuffer> metal_handle = command_buffer->command_buffer;

  // If we are switching encoders, we would need to use a fence to synchronize "one or more
  // resources across different passes within a command buffer."
  // https://developer.apple.com/documentation/metal/resource_synchronization
  uint64_t encoder_event_value = 0;
  if (command_buffer->state.compute_encoder) {
    iree_hal_metal_end_compute_encoder(command_buffer);
    encoder_event_value = command_buffer->state.next_encoder_event_value++;
    [metal_handle encodeSignalEvent:command_buffer->state.encoder_event value:encoder_event_value];
  }

  if (!command_buffer->state.blit_encoder) {
    if (encoder_event_value != 0) {
      [metal_handle encodeWaitForEvent:command_buffer->state.encoder_event
                                 value:encoder_event_value];
    }
    @autoreleasepool {  // Use @autoreleasepool to trigger the autorelease within encoder creation.
      command_buffer->state.blit_encoder = [[metal_handle blitCommandEncoder] retain];  // +1
    }
  }

  return command_buffer->state.blit_encoder;
}

// Destroys the given |base_command_buffer| itself, without decreasing refcount in the shared
// staging buffer yet.
static void iree_hal_metal_command_buffer_destroy_internal(
    iree_hal_command_buffer_t* base_command_buffer);

iree_status_t iree_hal_metal_direct_command_buffer_create(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories, iree_host_size_t binding_capacity,
    iree_hal_metal_command_buffer_resource_reference_mode_t resource_reference_mode,
    id<MTLCommandQueue> queue, iree_arena_block_pool_t* block_pool,
    iree_hal_metal_staging_buffer_t* staging_buffer,
    iree_hal_metal_builtin_executable_t* builtin_executable, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  IREE_ASSERT_TRUE(iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT));
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "indirect command buffer not yet supported");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_metal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*command_buffer) + iree_hal_command_buffer_validation_state_size(
                                                          mode, binding_capacity),
                            (void**)&command_buffer));

  iree_hal_command_buffer_initialize(iree_hal_device_allocator(device), mode, command_categories,
                                     IREE_HAL_QUEUE_AFFINITY_ANY, binding_capacity,
                                     (uint8_t*)command_buffer + sizeof(*command_buffer),
                                     &iree_hal_metal_command_buffer_vtable, &command_buffer->base);
  command_buffer->device = device;
  command_buffer->queue = [queue retain];  // +1
  command_buffer->builtin_executable = builtin_executable;
  iree_arena_initialize(block_pool, &command_buffer->arena);
  command_buffer->staging_buffer = staging_buffer;
  command_buffer->host_allocator = host_allocator;
  iree_status_t status = iree_ok_status();
  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
    status = iree_hal_resource_set_allocate(block_pool, &command_buffer->resource_set);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_metal_command_segment_list_reset(&command_buffer->segments);
    @autoreleasepool {  // Use @autoreleasepool to trigger the autorelease within encoder creation.
      // We track resource lifetime by ourselves in IREE; so just do unretained references to
      // resources in Metal command buffer, which avoids overhead and gives better performance.
      MTLCommandBufferDescriptor* descriptor = [MTLCommandBufferDescriptor new];  // +1
      descriptor.retainedReferences =
          resource_reference_mode == IREE_HAL_METAL_COMMAND_BUFFER_RESOURCE_REFERENCE_MODE_RETAINED;
      descriptor.errorOptions = MTLCommandBufferErrorOptionNone;
      command_buffer->command_buffer =
          [[queue commandBufferWithDescriptor:descriptor] retain];  // +1
      [descriptor release];                                         // -1
    }
    const iree_hal_metal_device_params_t* params = iree_hal_metal_device_params(device);
    command_buffer->dispatch_type =
        params->command_dispatch_type == IREE_HAL_METAL_COMMAND_DISPATCH_TYPE_CONCURRENT
            ? MTLDispatchTypeConcurrent
            : MTLDispatchTypeSerial;
    command_buffer->state.compute_encoder = nil;
    command_buffer->state.blit_encoder = nil;
    command_buffer->state.encoder_event = [queue.device newEvent];  // +1
    command_buffer->state.next_encoder_event_value = 1;
  }

  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;

    // Increase command buffer refcount in the shared staging buffer. We tie this to the command
    // buffer's lifetime to avoid resource leak.
    iree_hal_metal_staging_buffer_increase_command_buffer_refcount(staging_buffer);
    // Retain the device given that we refer to builtin executables and staging buffers whose
    // lifetime is associated with the device.
    iree_hal_device_retain(device);
  } else {
    iree_hal_metal_command_buffer_destroy_internal(&command_buffer->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_command_buffer_destroy_internal(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);

  iree_hal_metal_command_buffer_reset(command_buffer);
  [command_buffer->state.encoder_event release];  // -1
  IREE_ASSERT_EQ(command_buffer->state.compute_encoder, nil);
  IREE_ASSERT_EQ(command_buffer->state.blit_encoder, nil);
  [command_buffer->command_buffer release];  // -1
  [command_buffer->queue release];           // -1
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(command_buffer->host_allocator, command_buffer);
}

static void iree_hal_metal_command_buffer_destroy(iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  iree_hal_device_t* device = command_buffer->device;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Decrease command buffer refcount in the shared staging buffer, and potentially reclaim
  // resources. We tie this to the command buffer's lifetime to avoid resource leak.
  if (command_buffer->staging_buffer) {
    iree_hal_metal_staging_buffer_decrease_command_buffer_refcount(command_buffer->staging_buffer);
  }

  iree_hal_metal_command_buffer_destroy_internal(base_command_buffer);

  iree_hal_device_release(device);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_metal_command_buffer_isa(iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource, &iree_hal_metal_command_buffer_vtable);
}

static iree_status_t iree_hal_metal_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color, const iree_hal_label_location_t* location) {
  // TODO(antiagainst): implement support for debug group
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  // TODO(antiagainst): implement support for debug group
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_buffer_prepare_barrier(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask, iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count, const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count, const iree_hal_buffer_barrier_t* buffer_barriers) {
  if (iree_any_bit_set(source_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST) ||
      iree_any_bit_set(target_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "barrier involving host not yet supported");
  }

  if (flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "non-zero barrier flag not yet supported");
  }

  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the command segment and keep track of all necessary API data.
  uint8_t* storage_base = NULL;
  iree_hal_metal_command_segment_t* segment = NULL;
  iree_host_size_t buffer_barrier_length = buffer_barrier_count * sizeof(iree_hal_buffer_barrier_t);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, sizeof(*segment) + buffer_barrier_length,
                              (void**)&storage_base));

  // Copy the buffer barriers to the end of the current segments for later access. We don't copy
  // memory barriers because in Metal there is only coarse-grained full memory barrier affecting
  // all buffers, regardless of the fine-grained details from IREE HAL barriers.
  uint8_t* barrier_ptr = storage_base + sizeof(*segment);
  memcpy(barrier_ptr, (const uint8_t*)buffer_barriers, buffer_barrier_length);

  // Compose and push the barrier segment.
  segment = (iree_hal_metal_command_segment_t*)storage_base;
  segment->action = IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_BARRIER;
  iree_hal_metal_command_segment_list_push_back(&command_buffer->segments, segment);

  segment->barrier.memory_barrier_count = memory_barrier_count;
  segment->barrier.buffer_barrier_count = buffer_barrier_count;
  segment->barrier.buffer_barriers = (const iree_hal_buffer_barrier_t*)barrier_ptr;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_segment_record_barrier(
    iree_hal_metal_command_buffer_t* command_buffer, iree_hal_metal_barrier_segment_t* segment) {
  // TODO(antiagainst): Analyze segments before and after to optimize barriers, e.g., switching
  // encoders would require its own synchronization; so we don't need extract barriers in the
  // middle.
  if (segment->memory_barrier_count == 0 && segment->buffer_barrier_count == 0) {
    // There is no direct corresponding APIs for execution only barrier in Metal. We just signal and
    // wait on the same value of a MTLEvent here.
    iree_hal_metal_end_blit_encoder(command_buffer);
    iree_hal_metal_end_compute_encoder(command_buffer);
    id<MTLCommandBuffer> metal_handle = command_buffer->command_buffer;
    uint64_t event_value = command_buffer->state.next_encoder_event_value++;
    [metal_handle encodeSignalEvent:command_buffer->state.encoder_event value:event_value];
    [metal_handle encodeWaitForEvent:command_buffer->state.encoder_event value:event_value];
    return iree_ok_status();
  }

  id<MTLComputeCommandEncoder> encoder =
      iree_hal_metal_get_or_begin_compute_encoder(command_buffer);

  if (segment->memory_barrier_count != 0) {
    // If there is a memory barrier specified, we have to place a catch-all barrier for all buffers.
    // Metal does not provide a more fine-grained control here.
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return iree_ok_status();
  }

  if (segment->buffer_barrier_count != 0) {
    // But we do have the option to specify a list of buffers to synchronize if only buffer barriers
    // are specified.
    id<MTLResource>* resources =
        (id<MTLResource>*)iree_alloca(sizeof(id<MTLResource>) * segment->buffer_barrier_count);
    for (iree_host_size_t i = 0; i < segment->buffer_barrier_count; ++i) {
      resources[i] = iree_hal_metal_buffer_handle(
          iree_hal_buffer_allocated_buffer(segment->buffer_barriers[i].buffer_ref.buffer));
    }
    [encoder memoryBarrierWithResources:resources count:segment->buffer_barrier_count];
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_metal_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_metal_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer, iree_host_size_t event_count,
    const iree_hal_event_t** events, iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask, iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers, iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_metal_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_ref_t buffer_ref,
    iree_hal_memory_advise_flags_t flags, uint64_t arg0, uint64_t arg1) {
  // This is a hint to the device and we have nothing to do for Metal.
  return iree_ok_status();
}

// Fills |value| with the duplicated single byte value and return true if the given |pattern| has
// duplicated values for each of its |pattern_length| bytes.
static bool iree_hal_metal_get_duplicated_single_byte_value(const void* pattern,
                                                            size_t pattern_length, uint8_t* value) {
  switch (pattern_length) {
    case 1: {
      *value = *(uint8_t*)pattern;
      return true;
    }
    case 2: {
      uint16_t two_bytes = *(uint16_t*)pattern;
      uint16_t byte0 = two_bytes & 0xffu;
      uint16_t byte1 = two_bytes >> 8u;
      if (byte0 == byte1) {
        *value = (int8_t)byte0;
        return true;
      }
      break;
    }
    case 4: {
      uint32_t four_bytes = *(uint32_t*)pattern;
      uint32_t byte0 = four_bytes & 0xffu;
      uint32_t byte1 = (four_bytes >> 8u) & 0xffu;
      uint32_t byte2 = (four_bytes >> 16u) & 0xffu;
      uint32_t byte3 = four_bytes >> 24u;
      if (byte0 == byte1 && byte0 == byte2 && byte0 == byte3) {
        *value = (int8_t)byte0;
        return true;
      }
      break;
    }
    default:
      break;
  }
  return false;
}

// Duplicates the given |pattern| into 4-bytes and returns the value.
static uint32_t iree_hal_metal_duplicate_to_four_byte_value(const void* pattern,
                                                            size_t pattern_length) {
  if (pattern_length == 1) {
    uint8_t single_byte = *(uint8_t*)pattern;
    uint32_t value = (uint32_t)single_byte;
    value |= (value << 8u);
    value |= (value << 16u);
    return value;
  }

  if (pattern_length == 2) {
    uint16_t two_bytes = *(uint16_t*)pattern;
    uint32_t value = (uint32_t)two_bytes;
    value |= (value << 16u);
    return value;
  }

  IREE_ASSERT(pattern_length == 4);
  return *(uint32_t*)pattern;
}

static iree_status_t iree_hal_metal_command_buffer_prepare_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_ref_t target_ref,
    const void* pattern, iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  id<MTLBuffer> target_device_buffer =
      iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;

  // Allocate the command segment and keep track of all necessary API data.
  uint8_t* storage_base = NULL;
  iree_hal_metal_command_segment_t* segment = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, sizeof(*segment) + pattern_length,
                              (void**)&storage_base));

  // Copy the patttern to the end of the segment for later access.
  uint8_t* pattern_ptr = storage_base + sizeof(*segment);
  memcpy(pattern_ptr, (const uint8_t*)pattern, pattern_length);

  // Compose and push the fill buffer segment.
  segment = (iree_hal_metal_command_segment_t*)storage_base;
  segment->action = IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_FILL_BUFFER;
  iree_hal_metal_command_segment_list_push_back(&command_buffer->segments, segment);

  segment->fill_buffer.target_buffer = target_device_buffer;
  segment->fill_buffer.target_offset = target_offset;
  segment->fill_buffer.length = target_ref.length;
  segment->fill_buffer.pattern = (const void*)pattern_ptr;
  segment->fill_buffer.pattern_length = pattern_length;

  iree_status_t status =
      iree_hal_resource_set_insert(command_buffer->resource_set, 1, &target_ref.buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_command_segment_record_fill_buffer(
    iree_hal_metal_command_buffer_t* command_buffer,
    iree_hal_metal_fill_buffer_segment_t* segment) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Note that fillBuffer:range:value: only accepts a single byte as the pattern but FillBuffer
  // can accept 1/2/4 bytes. If the pattern itself contains repeated bytes, we can call into
  // fillBuffer:range:value:. Otherwise we need to emulate the support.
  uint8_t pattern_1byte = 0u;

  // Per the spec for fillBuffer:range:value: "The alignment and length of the range must both be a
  // multiple of 4 bytes in macOS, and 1 byte in iOS and tvOS."
#if defined(IREE_PLATFORM_MACOS)
  const bool can_use_metal_api = segment->target_offset % 4 == 0 && segment->length % 4 == 0 &&
                                 iree_hal_metal_get_duplicated_single_byte_value(
                                     segment->pattern, segment->pattern_length, &pattern_1byte);
#else
  const bool can_use_metal_api = iree_hal_metal_get_duplicated_single_byte_value(
      segment->pattern, segment->pattern_length, &pattern_1byte);
#endif

  if (can_use_metal_api) {
    id<MTLBlitCommandEncoder> encoder = iree_hal_metal_get_or_begin_blit_encoder(command_buffer);
    [encoder fillBuffer:segment->target_buffer
                  range:NSMakeRange(segment->target_offset, segment->length)
                  value:pattern_1byte];
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  id<MTLComputeCommandEncoder> compute_encoder =
      iree_hal_metal_get_or_begin_compute_encoder(command_buffer);
  uint32_t pattern_4byte =
      iree_hal_metal_duplicate_to_four_byte_value(segment->pattern, segment->pattern_length);
  iree_status_t status = iree_hal_metal_builtin_executable_fill_buffer(
      command_buffer->builtin_executable, compute_encoder, segment->target_buffer,
      segment->target_offset, segment->length, pattern_4byte);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_command_segment_create_copy_buffer(
    iree_hal_metal_command_buffer_t* command_buffer, id<MTLBuffer> source_device_buffer,
    iree_device_size_t source_offset, id<MTLBuffer> target_device_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the command segment and keep track of all necessary API data.
  uint8_t* storage_base = NULL;
  iree_hal_metal_command_segment_t* segment = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, sizeof(*segment), (void**)&storage_base));

  // Compose and push the barrier segment.
  segment = (iree_hal_metal_command_segment_t*)storage_base;
  segment->action = IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_COPY_BUFFER;
  iree_hal_metal_command_segment_list_push_back(&command_buffer->segments, segment);

  segment->copy_buffer.source_buffer = source_device_buffer;
  segment->copy_buffer.source_offset = source_offset;
  segment->copy_buffer.target_buffer = target_device_buffer;
  segment->copy_buffer.target_offset = target_offset;
  segment->copy_buffer.length = length;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_segment_record_copy_buffer(
    iree_hal_metal_command_buffer_t* command_buffer,
    iree_hal_metal_copy_buffer_segment_t* segment) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Per the spec for copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size, the source/target
  // offset and length must be a multiple of 4 bytes in macOS, and 1 byte in iOS and tvOS.
#if defined(IREE_PLATFORM_MACOS)
  bool can_use_metal_api = segment->source_offset % 4 == 0 && segment->target_offset % 4 == 0 &&
                           segment->length % 4 == 0;
#else
  bool can_use_metal_api = true;
#endif

  iree_status_t status = iree_ok_status();
  if (can_use_metal_api) {
    id<MTLBlitCommandEncoder> encoder = iree_hal_metal_get_or_begin_blit_encoder(command_buffer);
    [encoder copyFromBuffer:segment->source_buffer
               sourceOffset:segment->source_offset
                   toBuffer:segment->target_buffer
          destinationOffset:segment->target_offset
                       size:segment->length];
  } else {
    id<MTLComputeCommandEncoder> encoder =
        iree_hal_metal_get_or_begin_compute_encoder(command_buffer);
    status = iree_hal_metal_builtin_executable_copy_buffer(
        command_buffer->builtin_executable, encoder, segment->source_buffer, segment->source_offset,
        segment->target_buffer, segment->target_offset, segment->length);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_command_buffer_prepare_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // There are no direct corresponding APIs in Metal. We update the source buffer data to the
  // staging buffer and then copy over.

  iree_const_byte_span_t source_data_span =
      iree_make_const_byte_span((uint8_t*)source_buffer + source_offset, target_ref.length);
  uint32_t offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_metal_staging_buffer_append(command_buffer->staging_buffer, source_data_span,
                                               /*alignment=*/4, &offset));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1, &target_ref.buffer));

  id<MTLBuffer> target_device_buffer =
      iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;

  iree_status_t status = iree_hal_metal_command_segment_create_copy_buffer(
      command_buffer, command_buffer->staging_buffer->metal_buffer, offset, target_device_buffer,
      target_offset, target_ref.length);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_command_buffer_prepare_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_ref_t source_ref,
    iree_hal_buffer_ref_t target_ref, iree_hal_copy_flags_t flags) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_buffer_t* resources[2] = {source_ref.buffer, target_ref.buffer};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, IREE_ARRAYSIZE(resources),
                                       resources));

  id<MTLBuffer> source_device_buffer =
      iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(source_ref.buffer));
  id<MTLBuffer> target_device_buffer =
      iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(target_ref.buffer));

  iree_device_size_t source_offset =
      iree_hal_buffer_byte_offset(source_ref.buffer) + source_ref.offset;
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;

  iree_status_t status = iree_hal_metal_command_segment_create_copy_buffer(
      command_buffer, source_device_buffer, source_offset, target_device_buffer, target_offset,
      target_ref.length);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "collectives not yet supported");
}

// Prepares kernels and argument buffers needed for kernel dispatches.
static iree_status_t iree_hal_metal_command_buffer_prepare_dispatch(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal, const iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO: support custom args (should be easy) and indirect arguments (via
  // setKernelBuffer:offset:atIndex:).
  if (iree_hal_dispatch_uses_custom_arguments(flags)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "direct/indirect arguments are not supported on Metal");
  }

  iree_host_size_t resource_count = 1;
  const void* resources[2] = {executable, NULL};
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    resources[resource_count++] = config.workgroup_count_ref.buffer;
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, resource_count, &executable));

  const iree_hal_metal_pipeline_t* pipeline = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_metal_executable_lookup_pipeline(executable, export_ordinal, &pipeline));

  // Allocate the command segment and keep track of all necessary API data.
  uint8_t* storage_base = NULL;
  iree_hal_metal_command_segment_t* segment = NULL;
  iree_host_size_t descriptor_length = bindings.count * sizeof(iree_hal_metal_descriptor_t);
  iree_host_size_t total_size = sizeof(*segment) + descriptor_length + constants.data_length;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, total_size, (void**)&storage_base));

  // Compose and push the dispatch segment.
  segment = (iree_hal_metal_command_segment_t*)storage_base;
  memset(segment, 0, sizeof(*segment));
  segment->action = IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_DISPATCH;
  iree_hal_metal_command_segment_list_push_back(&command_buffer->segments, segment);

  segment->dispatch.pipeline = pipeline;
  segment->dispatch.threadgroup_size =
      config.workgroup_size[0] ? MTLSizeMake(config.workgroup_size[0], config.workgroup_size[1],
                                             config.workgroup_size[2])
                               : pipeline->threadgroup_size;

  // Copy descriptors from all sets to the end of the current segment for later access.
  segment->dispatch.descriptor_count = bindings.count;
  segment->dispatch.descriptors = (iree_hal_metal_descriptor_t*)(storage_base + sizeof(*segment));
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    iree_hal_metal_descriptor_t* descriptor = &segment->dispatch.descriptors[i];

    descriptor->binding = i;
    descriptor->buffer = bindings.values[i].buffer;
    descriptor->offset = bindings.values[i].offset;

    MTLResourceUsage usage = MTLResourceUsageRead;
    uint64_t binding_bit = 1ull << i;
    if (iree_any_bit_set(pipeline->binding_read_only_bits, binding_bit)) {
      usage |= MTLResourceUsageWrite;
    }
    descriptor->usage = usage;

    if (descriptor->buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1, &descriptor->buffer));
    }
  }

  // Copy push constants to the end of the current segment for later access.
  segment->dispatch.constant_count = constants.data_length / sizeof(int32_t);
  uint8_t* constant_ptr = storage_base + sizeof(*segment) + descriptor_length;
  segment->dispatch.constants = (int32_t*)constant_ptr;
  memcpy(constant_ptr, constants.data, constants.data_length);

  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    segment->dispatch.workgroups_buffer = iree_hal_metal_buffer_handle(
        iree_hal_buffer_allocated_buffer(config.workgroup_count_ref.buffer));
    segment->dispatch.workgroups_offset = config.workgroup_count_ref.offset;
  } else {
    segment->dispatch.workgroup_count = MTLSizeMake(
        config.workgroup_count[0], config.workgroup_count[1], config.workgroup_count[2]);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_segment_record_dispatch(
    iree_hal_metal_command_buffer_t* command_buffer, iree_hal_metal_dispatch_segment_t* segment) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Set the compute kernel to dispatch.
  id<MTLComputeCommandEncoder> compute_encoder =
      iree_hal_metal_get_or_begin_compute_encoder(command_buffer);
  [compute_encoder setComputePipelineState:segment->pipeline->pipeline_state];

  // Record push constants.
  if (segment->constant_count != 0) {
    [compute_encoder setBytes:(void*)segment->constants
                       length:segment->constant_count * sizeof(int32_t)
                      atIndex:IREE_HAL_METAL_PUSH_CONSTANT_BUFFER_INDEX];
  }

  // Record argument buffers for all descriptors and record buffer usages.
  iree_hal_metal_descriptor_t* descriptors = segment->descriptors;

  // Build argument encoder and argument buffer for the current descriptor set.
  // TODO(antiagainst): Use a cache layer to cache and reuse argument buffers with the same
  // content, to avoid duplicating overhead.
  id<MTLBuffer> argument_buffer = command_buffer->staging_buffer->metal_buffer;
  id<MTLArgumentEncoder> argument_encoder =
      [segment->pipeline->function newArgumentEncoderWithBufferIndex:0];  // +1
  IREE_ASSERT(argument_encoder != nil);

  // Reserve space for the argument buffer from shared staging buffer.
  iree_byte_span_t reservation = iree_byte_span_empty();
  uint32_t argument_buffer_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_metal_staging_buffer_reserve(
              command_buffer->staging_buffer, argument_encoder.encodedLength,
              argument_encoder.alignment, &reservation, &argument_buffer_offset));
  [argument_encoder setArgumentBuffer:argument_buffer offset:argument_buffer_offset];

  // Now record all bound buffers belonging to the current set into the argument buffer.
  for (iree_host_size_t i = 0; i < segment->descriptor_count; ++i) {
    uint32_t current_binding = descriptors[i].binding;
    id<MTLBuffer> current_buffer =
        iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(descriptors[i].buffer));
    iree_host_size_t offset =
        iree_hal_buffer_byte_offset(descriptors[i].buffer) + descriptors[i].offset;
    [argument_encoder setBuffer:current_buffer offset:offset atIndex:current_binding];

    // Also record buffer usages.
    [compute_encoder useResource:current_buffer usage:descriptors[i].usage];
  }
  // Record the argument buffer.
  [compute_encoder setBuffer:argument_buffer offset:argument_buffer_offset atIndex:0];

  [argument_encoder release];  // -1

  // Record the dispatch, either direct or indirect.
  if (segment->workgroups_buffer == nil) {
    // Direct dispatch of a fixed workgroup count.
    [compute_encoder dispatchThreadgroups:segment->workgroup_count
                    threadsPerThreadgroup:segment->threadgroup_size];
  } else {
    // Indirect dispatch using a workgroup count from buffers.
    [compute_encoder dispatchThreadgroupsWithIndirectBuffer:segment->workgroups_buffer
                                       indirectBufferOffset:segment->workgroups_offset
                                      threadsPerThreadgroup:segment->threadgroup_size];
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_segment_record(
    iree_hal_metal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_hal_metal_command_segment_t* segment = command_buffer->segments.head; segment;
       segment = segment->next_segment) {
    switch (segment->action) {
      case IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_BARRIER: {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_metal_command_segment_record_barrier(command_buffer, &segment->barrier));
      } break;
      case IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_DISPATCH: {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_metal_command_segment_record_dispatch(command_buffer, &segment->dispatch));
      } break;
      case IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_FILL_BUFFER: {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_metal_command_segment_record_fill_buffer(
                                                  command_buffer, &segment->fill_buffer));
      } break;
      case IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_COPY_BUFFER: {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_metal_command_segment_record_copy_buffer(
                                                  command_buffer, &segment->copy_buffer));
      } break;
      default:
        IREE_ASSERT(false, "unhandled command segment kind");
        break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  iree_hal_metal_command_buffer_reset(command_buffer);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_metal_command_segment_record(command_buffer));
  iree_hal_metal_end_blit_encoder(command_buffer);
  iree_hal_metal_end_compute_encoder(command_buffer);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static const iree_hal_command_buffer_vtable_t iree_hal_metal_command_buffer_vtable = {
    .destroy = iree_hal_metal_command_buffer_destroy,
    .begin = iree_hal_metal_command_buffer_begin,
    .end = iree_hal_metal_command_buffer_end,
    .begin_debug_group = iree_hal_metal_command_buffer_begin_debug_group,
    .end_debug_group = iree_hal_metal_command_buffer_end_debug_group,
    .execution_barrier = iree_hal_metal_command_buffer_prepare_barrier,
    .signal_event = iree_hal_metal_command_buffer_signal_event,
    .reset_event = iree_hal_metal_command_buffer_reset_event,
    .wait_events = iree_hal_metal_command_buffer_wait_events,
    .advise_buffer = iree_hal_metal_command_buffer_advise_buffer,
    .fill_buffer = iree_hal_metal_command_buffer_prepare_fill_buffer,
    .update_buffer = iree_hal_metal_command_buffer_prepare_update_buffer,
    .copy_buffer = iree_hal_metal_command_buffer_prepare_copy_buffer,
    .collective = iree_hal_metal_command_buffer_collective,
    .dispatch = iree_hal_metal_command_buffer_prepare_dispatch,
};
