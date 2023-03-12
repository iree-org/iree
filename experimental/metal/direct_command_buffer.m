// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/metal/direct_command_buffer.h"

#import <Metal/Metal.h>

#include "experimental/metal/builtin_executables.h"
#include "experimental/metal/metal_buffer.h"
#include "experimental/metal/metal_device.h"
#include "experimental/metal/metal_fence.h"
#include "experimental/metal/metal_kernel_library.h"
#include "experimental/metal/pipeline_layout.h"
#include "iree/base/api.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/resource_set.h"

typedef struct iree_hal_metal_descriptor_t {
  uint32_t set;
  uint32_t binding;
  iree_hal_buffer_t* buffer;
  iree_host_size_t offset;

} iree_hal_metal_descriptor_t;

typedef struct iree_hal_metal_command_buffer_t {
  iree_hal_command_buffer_t base;

  // The Metal command queue owning this command buffer.
  id<MTLCommandQueue> queue;

  // For polyfilling fill/copy/update buffers that are not directly supported by Metal APIs.
  iree_hal_metal_builtin_executable_t* builtin_executable;

  id<MTLCommandBuffer> command_buffer;

  MTLDispatchType dispatch_type;

  // The current active compute/blit encoders for encoding compute for memory operations.
  // Metal commands are encoded into the command buffer with such encoders, and each encoder can
  // only encode the specific type of operations it supports.
  id<MTLComputeCommandEncoder> compute_encoder;
  id<MTLBlitCommandEncoder> blit_encoder;

  // Metal APIs mandate we create argument bufffers (for descriptor sets) from compiled kernel
  // function. That means we need to bind the compute kernel first before setting descriptors and
  // binding buffers. So we need to cache the descriptor information by ourselves and apply them in
  // a delayed manner.

  // A sorted flat list of descriptors from all pushed descriptor sets.
  iree_hal_metal_descriptor_t current_descriptors[IREE_HAL_METAL_MAX_BINDING_COUNT];
  // The total used slot count / next unused slot index in |current_descriptors|.
  int current_total_binding_count;
  // The max descriptor set number we have seen thus far.
  int current_max_set_number;

  // All available push constants updated each time push_constants is called. Reset only with the
  // command buffer and otherwise will maintain its values during recording to allow for partial
  // push_constants updates.
  int32_t push_constants[IREE_HAL_METAL_MAX_PUSH_CONSTANT_COUNT];

  // The current pipeline layout used for push descriptors.
  iree_hal_pipeline_layout_t* current_pipeline_layout;

  iree_allocator_t host_allocator;

  // Maintains a reference to all resources used within the command buffer. Resets on each begin.
  iree_hal_resource_set_t* resource_set;
} iree_hal_metal_command_buffer_t;

static const iree_hal_command_buffer_vtable_t iree_hal_metal_command_buffer_vtable;

static iree_hal_metal_command_buffer_t* iree_hal_metal_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_command_buffer_vtable);
  return (iree_hal_metal_command_buffer_t*)base_value;
}

id<MTLCommandBuffer> iree_hal_metal_direct_command_buffer_handle(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  return command_buffer->command_buffer;
}

static void iree_hal_metal_end_compute_encoder(iree_hal_metal_command_buffer_t* command_buffer) {
  if (command_buffer->compute_encoder) {
    [command_buffer->compute_encoder endEncoding];
    [command_buffer->compute_encoder release];  // -1
    command_buffer->compute_encoder = nil;
  }
}

static void iree_hal_metal_end_blit_encoder(iree_hal_metal_command_buffer_t* command_buffer) {
  if (command_buffer->blit_encoder) {
    [command_buffer->blit_encoder endEncoding];
    [command_buffer->blit_encoder release];  // -1
    command_buffer->blit_encoder = nil;
  }
}

static id<MTLComputeCommandEncoder> iree_hal_metal_get_or_begin_compute_encoder(
    iree_hal_metal_command_buffer_t* command_buffer) {
  id<MTLFence> encoder_fence = nil;
  if (command_buffer->blit_encoder) {
    // We would need to use a fence to synchronize "one or more resources across different passes
    // within a command buffer."
    // https://developer.apple.com/documentation/metal/resource_synchronization
    encoder_fence = [command_buffer->command_buffer.device newFence];  // +1
    [command_buffer->command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      [encoder_fence release];  // -1
    }];
    [command_buffer->blit_encoder updateFence:encoder_fence];
    iree_hal_metal_end_blit_encoder(command_buffer);
  }

  @autoreleasepool {  // Use @autoreleasepool to trigger the autorelease within encoder creation.
    if (!command_buffer->compute_encoder) {
      // We manage commands dependencies and insert barriers explicitly in IREE; so use the
      // concurrent dispatch type for compute encoders.
      command_buffer->compute_encoder = [[command_buffer->command_buffer
          computeCommandEncoderWithDispatchType:command_buffer->dispatch_type] retain];  // +1
    }
  }

  if (encoder_fence != nil) {
    [command_buffer->compute_encoder waitForFence:encoder_fence];
  }
  return command_buffer->compute_encoder;
}

static id<MTLBlitCommandEncoder> iree_hal_metal_get_or_begin_blit_encoder(
    iree_hal_metal_command_buffer_t* command_buffer) {
  id<MTLFence> encoder_fence = nil;
  if (command_buffer->compute_encoder) {
    // We would need to use a fence to synchronize "one or more resources across different passes
    // within a command buffer."
    // https://developer.apple.com/documentation/metal/resource_synchronization
    encoder_fence = [command_buffer->command_buffer.device newFence];  // +1
    [command_buffer->command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      [encoder_fence release];  // -1
    }];
    [command_buffer->compute_encoder updateFence:encoder_fence];
    iree_hal_metal_end_compute_encoder(command_buffer);
  }

  @autoreleasepool {  // Use @autoreleasepool to trigger the autorelease within encoder creation.
    if (!command_buffer->blit_encoder) {
      command_buffer->blit_encoder =
          [[command_buffer->command_buffer blitCommandEncoder] retain];  // +1
    }
  }

  if (encoder_fence != nil) {
    [command_buffer->blit_encoder waitForFence:encoder_fence];
  }
  return command_buffer->blit_encoder;
}

iree_status_t iree_hal_metal_direct_command_buffer_create(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories, iree_host_size_t binding_capacity,
    iree_hal_metal_command_buffer_resource_reference_mode_t resource_reference_mode,
    id<MTLCommandQueue> queue, iree_allocator_t host_allocator, iree_arena_block_pool_t* block_pool,
    iree_hal_metal_builtin_executable_t* builtin_executable,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  IREE_ASSERT_TRUE(iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT));
  IREE_ASSERT_TRUE(!iree_any_bit_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_NESTED));
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplemented indirect command buffers");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_metal_command_buffer_t* command_buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*command_buffer), (void**)&command_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_command_buffer_initialize(
        device, mode, command_categories, IREE_HAL_QUEUE_AFFINITY_ANY, binding_capacity,
        &iree_hal_metal_command_buffer_vtable, &command_buffer->base);
    command_buffer->queue = [queue retain];  // +1
    command_buffer->builtin_executable = builtin_executable;
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
    command_buffer->compute_encoder = nil;
    command_buffer->blit_encoder = nil;
    memset(command_buffer->current_descriptors, 0,
           IREE_HAL_METAL_MAX_BINDING_COUNT * sizeof(command_buffer->current_descriptors[0]));
    command_buffer->current_total_binding_count = 0;
    command_buffer->current_max_set_number = -1;
    memset(command_buffer->push_constants, 0, sizeof(command_buffer->push_constants));
    command_buffer->current_pipeline_layout = NULL;
    command_buffer->host_allocator = host_allocator;
    status = iree_hal_resource_set_allocate(block_pool, &command_buffer->resource_set);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_command_buffer_destroy(iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_ASSERT_EQ(command_buffer->compute_encoder, nil);
  IREE_ASSERT_EQ(command_buffer->blit_encoder, nil);
  [command_buffer->command_buffer release];  // -1
  [command_buffer->queue release];           // -1
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_allocator_free(command_buffer->host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_metal_command_buffer_isa(iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource, &iree_hal_metal_command_buffer_vtable);
}

static iree_status_t iree_hal_metal_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  // Nothing to do.
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  iree_hal_metal_end_blit_encoder(command_buffer);
  iree_hal_metal_end_compute_encoder(command_buffer);
  return iree_ok_status();
}

static void iree_hal_metal_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color, const iree_hal_label_location_t* location) {
  // TODO(antiagainst): implement support for debug group
}

static void iree_hal_metal_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  // TODO(antiagainst): implement support for debug group
}

static iree_status_t iree_hal_metal_command_buffer_execution_barrier(
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
  id<MTLComputeCommandEncoder> encoder =
      iree_hal_metal_get_or_begin_compute_encoder(command_buffer);

  if (memory_barrier_count != 0) {
    // If there is a memory barrier specified, we have to place a catch-all barrier for all buffers.
    // Metal does not provide a more fine-grained control here.
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return iree_ok_status();
  }

  if (buffer_barrier_count != 0) {
    // But we do have the option to specify a list of buffers to synchronize if only buffer barriers
    // are specified.
    id<MTLResource>* resources =
        (id<MTLResource>*)iree_alloca(sizeof(id<MTLResource>) * buffer_barrier_count);
    for (unsigned i = 0; i < buffer_barrier_count; ++i) {
      resources[i] =
          iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(buffer_barriers[i].buffer));
    }
    [encoder memoryBarrierWithResources:resources count:buffer_barrier_count];
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  id<MTLFence> fence = iree_hal_metal_fence_handle(event);

  // In Metal compute pipelines, fences are more course grained than Vulkan--we only have one
  // execution stage instead of multiple stages in Vulkan.
  if (command_buffer->blit_encoder != nil) {
    [command_buffer->blit_encoder updateFence:fence];
  } else {
    id<MTLComputeCommandEncoder> compute_encoder =
        iree_hal_metal_get_or_begin_compute_encoder(command_buffer);
    [compute_encoder updateFence:fence];
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  // In Metal, MTLFence does not support reset. We just create a new fence every time IREE event
  // reset API is called. This assumes IREE fences are not waited anymore when reset is called,
  // which should be true for proper IREE event usage.
  return iree_hal_metal_fence_recreate(event);
}

static iree_status_t iree_hal_metal_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer, iree_host_size_t event_count,
    const iree_hal_event_t** events, iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask, iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers, iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);

  // In Metal compute pipelines, fences are more course grained than Vulkan--we only have one
  // execution stage instead of multiple stages in Vulkan, and we don't have separate memory
  // barriers to control.
  if (command_buffer->blit_encoder != nil) {
    for (iree_host_size_t i = 0; i < event_count; ++i) {
      id<MTLFence> fence = iree_hal_metal_fence_handle(events[i]);
      [command_buffer->blit_encoder waitForFence:fence];
    }
  } else {
    id<MTLComputeCommandEncoder> compute_encoder =
        iree_hal_metal_get_or_begin_compute_encoder(command_buffer);
    for (iree_host_size_t i = 0; i < event_count; ++i) {
      id<MTLFence> fence = iree_hal_metal_fence_handle(events[i]);
      [compute_encoder waitForFence:fence];
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
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

// Fills |value| by duplicating the given |pattern| into 4-bytes.
static iree_status_t iree_hal_metal_duplicate_to_four_byte_value(const void* pattern,
                                                                 size_t pattern_length,
                                                                 uint32_t* value) {
  switch (pattern_length) {
    case 1: {
      uint8_t single_byte = *(uint8_t*)pattern;
      *value = (uint32_t)single_byte;
      *value |= (*value << 8u);
      *value |= (*value << 16u);
      return iree_ok_status();
    }
    case 2: {
      uint16_t two_bytes = *(uint16_t*)pattern;
      *value = (uint32_t)two_bytes;
      *value |= (*value << 16u);
      return iree_ok_status();
    }
    case 4: {
      *value = *(uint32_t*)pattern;
      return iree_ok_status();
    }

    default:
      break;
  }
  return iree_make_status(IREE_STATUS_INTERNAL, "fill pattern should have 1/2/4 bytes");
}

static iree_status_t iree_hal_metal_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  id<MTLBuffer> target_device_buffer =
      iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);

  // Per the spec for fillBuffer:range:value: "The alignment and length of the range must both be a
  // multiple of 4 bytes in macOS, and 1 byte in iOS and tvOS."
#if defined(IREE_PLATFORM_MACOS)
  bool can_use_metal_api = target_offset % 4 == 0 && length % 4 == 0;
#else
  bool can_use_metal_api = true;
#endif

  // Note that fillBuffer:range:value: only accepts a single byte as the pattern but FillBuffer
  // can accept 1/2/4 bytes. If the pattern itself contains repeated bytes, we can call into
  // fillBuffer:range:value:. Otherwise we need to emulate the support.
  uint8_t single_byte_value = 0u;
  if (can_use_metal_api) {
    can_use_metal_api &= iree_hal_metal_get_duplicated_single_byte_value(pattern, pattern_length,
                                                                         &single_byte_value);
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_resource_set_insert(command_buffer->resource_set, 1, &target_buffer));

  iree_status_t status = iree_ok_status();
  if (can_use_metal_api) {
    id<MTLBlitCommandEncoder> encoder = iree_hal_metal_get_or_begin_blit_encoder(command_buffer);
    [encoder fillBuffer:target_device_buffer
                  range:NSMakeRange(target_offset, length)
                  value:single_byte_value];
  } else {
    id<MTLComputeCommandEncoder> compute_encoder =
        iree_hal_metal_get_or_begin_compute_encoder(command_buffer);
    uint32_t pattern_4byte = 0;
    status = iree_hal_metal_duplicate_to_four_byte_value(pattern, pattern_length, &pattern_4byte);
    if (iree_status_is_ok(status)) {
      status = iree_hal_metal_builtin_executable_fill_buffer(command_buffer->builtin_executable,
                                                             compute_encoder, target_device_buffer,
                                                             target_offset, length, pattern_4byte);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_command_buffer_copy_buffer_internal(
    iree_hal_metal_command_buffer_t* command_buffer, id<MTLBuffer> source_device_buffer,
    iree_device_size_t source_offset, id<MTLBuffer> target_device_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  // Per the spec for copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size, the source/target
  // offset and length must be a multiple of 4 bytes in macOS, and 1 byte in iOS and tvOS.
#if defined(IREE_PLATFORM_MACOS)
  bool can_use_metal_api = source_offset % 4 == 0 && target_offset % 4 == 0 && length % 4 == 0;
#else
  bool can_use_metal_api = true;
#endif

  iree_status_t status = iree_ok_status();
  if (can_use_metal_api) {
    id<MTLBlitCommandEncoder> encoder = iree_hal_metal_get_or_begin_blit_encoder(command_buffer);
    [encoder copyFromBuffer:source_device_buffer
               sourceOffset:source_offset
                   toBuffer:target_device_buffer
          destinationOffset:target_offset
                       size:length];
  } else {
    id<MTLComputeCommandEncoder> encoder =
        iree_hal_metal_get_or_begin_compute_encoder(command_buffer);
    status = iree_hal_metal_builtin_executable_copy_buffer(
        command_buffer->builtin_executable, encoder, source_device_buffer, source_offset,
        target_device_buffer, target_offset, length);
  }

  return status;
}

static iree_status_t iree_hal_metal_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  // There are no direct corresponding APIs in Metal. We emulate it by creating a buffer with the
  // content and then copy it over.
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  id<MTLDevice> device = command_buffer->command_buffer.device;
  MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
  id<MTLBuffer> data_buffer = [device newBufferWithBytes:((uint8_t*)source_buffer + source_offset)
                                                  length:length
                                                 options:options];  // +1
  [command_buffer->command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cmdbuf) {
    [data_buffer release];  // -1
  }];

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1, &target_buffer));

  id<MTLBuffer> target_device_buffer =
      iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);

  iree_status_t status = iree_hal_metal_command_buffer_copy_buffer_internal(
      command_buffer, data_buffer, /*source_offset=*/0, target_device_buffer, target_offset,
      length);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_buffer_t* buffers[2] = {source_buffer, target_buffer};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 2, buffers));

  id<MTLBuffer> source_device_buffer =
      iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(source_buffer));
  id<MTLBuffer> target_device_buffer =
      iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(target_buffer));

  source_offset += iree_hal_buffer_byte_offset(source_buffer);
  target_offset += iree_hal_buffer_byte_offset(target_buffer);

  iree_status_t status = iree_hal_metal_command_buffer_copy_buffer_internal(
      command_buffer, source_device_buffer, source_offset, target_device_buffer, target_offset,
      length);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "collectives not yet supported");
}

static iree_status_t iree_hal_metal_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_pipeline_layout_t* pipeline_layout,
    iree_host_size_t offset, const void* values, iree_host_size_t values_length) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);

  // "Binding a pipeline with a layout that is not compatible with the push constant layout does not
  // disturb the push constant values." So we don't need to check whether the pipeline layout
  // compatibility and invalidate existing values.
  // See https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdPushConstants.html

  if (IREE_UNLIKELY(offset + values_length >= sizeof(command_buffer->push_constants))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "push constant range [%zu, %zu) out of range", offset,
                            offset + values_length);
  }

  memcpy((uint8_t*)&command_buffer->push_constants + offset, values, values_length);
  command_buffer->current_pipeline_layout = pipeline_layout;

  return iree_ok_status();
}

static int compare_descriptor(const void* a, const void* b) {
  const iree_hal_metal_descriptor_t* buffer_a = (const iree_hal_metal_descriptor_t*)a;
  const iree_hal_metal_descriptor_t* buffer_b = (const iree_hal_metal_descriptor_t*)b;
  if (buffer_a->set != buffer_b->set) return buffer_a->set - buffer_b->set;
  return buffer_a->binding - buffer_b->binding;
}

// Returns true if the given |descriptors| array contains descriptors in ascending binding slot
// order and there is no duplicated binding slots.
static bool iree_hal_metal_is_sorted_unique_descriptors(iree_hal_metal_descriptor_t* descriptors,
                                                        int descriptor_count) {
  for (int i = 1; i < descriptor_count; ++i) {
    if (compare_descriptor(&descriptors[i - 1], &descriptors[i]) >= 0) return false;
  }
  return true;
}

static iree_status_t iree_hal_metal_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_pipeline_layout_t* pipeline_layout,
    uint32_t set, iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (set == IREE_HAL_METAL_PUSH_CONSTANT_BUFFER_INDEX) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "descriptor set #%d reserved for push constant emulation",
                            IREE_HAL_METAL_PUSH_CONSTANT_BUFFER_INDEX);
  }

  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    if (bindings[i].buffer) continue;
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unimplemented null buffer in push descriptor set");
  }

  iree_hal_metal_descriptor_t* descriptors = command_buffer->current_descriptors;
  IREE_ASSERT(iree_hal_metal_is_sorted_unique_descriptors(
      descriptors, command_buffer->current_total_binding_count));

  if (command_buffer->current_max_set_number >= (int)set) {
    // We are pushing an already seen set. This would invalidate all sets with the given number and
    // larger ones. So clear all affected bindings.
    // TODO(antiagainst): We should actually check current pipeline's layout compatibility with
    // previous one and decide whether we should invalidate lower numbered sets too. For now we
    // assume the compiler side is doing proper job of guaranteeing that.
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap14.html#descriptorsets-compatibility
    int* count = &command_buffer->current_total_binding_count;
    while (*count > 0 && descriptors[*count - 1].set >= (int)set) --(*count);
    command_buffer->current_max_set_number = (*count == 0) ? -1 : descriptors[*count - 1].set;
  }

  // Pushing a new set with a larger number. All sets with smaller number remain active. Just sort
  // the current one and copy over the data. This is the expected usage pattern in IREE, where the
  // compiler sorts/deduplicates descriptor sets, and pushes them in ascending order.
  if (binding_count + command_buffer->current_total_binding_count >
      IREE_HAL_METAL_MAX_BINDING_COUNT) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "exceeded available binding slots for push descriptor sets");
  }

  int start_index = command_buffer->current_total_binding_count;
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    iree_hal_metal_descriptor_t* descriptor = &descriptors[start_index + i];
    descriptor->set = set;
    descriptor->binding = bindings[i].binding;
    descriptor->buffer = bindings[i].buffer;
    descriptor->offset = bindings[i].offset;
  }
  qsort(&descriptors[start_index], binding_count, sizeof(descriptors[0]), compare_descriptor);

  command_buffer->current_max_set_number = set;
  command_buffer->current_total_binding_count += binding_count;

  IREE_ASSERT(iree_hal_metal_is_sorted_unique_descriptors(
      descriptors, command_buffer->current_total_binding_count));

  // Retain all buffers bound in this descriptor set.
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    if (bindings[i].buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1, &bindings[i].buffer));
    }
  }

  command_buffer->current_pipeline_layout = pipeline_layout;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1, &pipeline_layout));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static inline MTLResourceUsage iree_hal_metal_get_metal_resource_usage(
    iree_hal_descriptor_set_layout_binding_t* binding) {
  MTLResourceUsage usage = MTLResourceUsageRead;
  if (binding->flags != IREE_HAL_DESCRIPTOR_FLAG_READ_ONLY) usage |= MTLResourceUsageWrite;
  return usage;
}

// Creates an argument encoder and its backing argument buffer for the given kernel |function|'s
// |buffer_index|. The argument encoder will be set to encode into the newly created argument
// buffer. Callers are expected to release both the argument encoder and buffer.
static iree_status_t iree_hal_metal_create_argument_encoder(
    id<MTLDevice> device, id<MTLCommandBuffer> command_buffer, id<MTLFunction> function,
    uint32_t buffer_index, id<MTLArgumentEncoder>* out_encoder, id<MTLBuffer>* out_buffer) {
  id<MTLArgumentEncoder> argument_encoder =
      [function newArgumentEncoderWithBufferIndex:buffer_index];  // +1
  if (!argument_encoder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid argument buffer index #%u",
                            buffer_index);
  }

  __block id<MTLBuffer> argument_buffer =
      [device newBufferWithLength:argument_encoder.encodedLength
                          options:MTLResourceStorageModeShared];  // +1
  if (!argument_buffer) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to create argument buffer with size = %ld bytes",
                            argument_encoder.encodedLength);
  }

  // The arugment encoder and buffer can be deleted once the command buffer completes.
  [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cmdbuf) {
    [argument_buffer release];   // -1
    [argument_encoder release];  // -1
  }];

  [argument_encoder setArgumentBuffer:argument_buffer offset:0];
  *out_encoder = argument_encoder;
  *out_buffer = argument_buffer;
  return iree_ok_status();
}

// Prepares kernels and argument buffers needed for kernel dispatches.
static iree_status_t iree_hal_metal_command_buffer_prepare_dispatch(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_executable_t* executable,
    int32_t entry_point, iree_hal_metal_kernel_params_t* kernel_params) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_metal_kernel_library_entry_point_kernel_params(
                                            executable, entry_point, kernel_params));
  if (!command_buffer->current_pipeline_layout) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing pipeline layout when dispatch");
  }

  // Set the compute kernel to dispatch.
  id<MTLComputeCommandEncoder> compute_encoder =
      iree_hal_metal_get_or_begin_compute_encoder(command_buffer);
  [compute_encoder setComputePipelineState:kernel_params->pso];

  iree_status_t status = iree_ok_status();

  // Bind all buffers in all descriptor sets.
  iree_hal_metal_descriptor_t* descriptors = command_buffer->current_descriptors;
  int binding_count = command_buffer->current_total_binding_count;
  int i = 0;
  while (i < binding_count) {
    // Build argument encoder and argument buffer for the current descriptor set.
    uint32_t current_set = descriptors[i].set;

    id<MTLArgumentEncoder> argument_encoder;
    id<MTLBuffer> argument_buffer;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_metal_create_argument_encoder(
                command_buffer->command_buffer.device, command_buffer->command_buffer,
                kernel_params->function, current_set, &argument_encoder, &argument_buffer));

    iree_hal_descriptor_set_layout_t* set_layout =
        iree_hal_metal_pipeline_layout_descriptor_set_layout(
            command_buffer->current_pipeline_layout, current_set);
    if (!set_layout) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "cannot find descriptor set layout for set #%u", current_set);
      break;
    }

    // Now put all bound buffers belonging to the current descriptor set into the argument buffer.
    for (; i < binding_count && descriptors[i].set == current_set; ++i) {
      unsigned current_binding = descriptors[i].binding;
      id<MTLBuffer> current_buffer =
          iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(descriptors[i].buffer));
      iree_host_size_t offset =
          iree_hal_buffer_byte_offset(descriptors[i].buffer) + descriptors[i].offset;
      [argument_encoder setBuffer:current_buffer offset:offset atIndex:current_binding];

      iree_hal_descriptor_set_layout_binding_t* binding_params =
          iree_hal_metal_descriptor_set_layout_binding(set_layout, current_binding);
      if (!binding_params) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "cannot find information for binding #%u in set #%u",
                                  current_binding, current_set);
        break;
      }
      [compute_encoder useResource:current_buffer
                             usage:iree_hal_metal_get_metal_resource_usage(binding_params)];
    }
    if (!iree_status_is_ok(status)) break;

    [compute_encoder setBuffer:argument_buffer offset:0 atIndex:current_set];
  }

  if (iree_hal_metal_pipeline_layout_push_constant_count(command_buffer->current_pipeline_layout)) {
    [compute_encoder setBytes:(void*)command_buffer->push_constants
                       length:sizeof(command_buffer->push_constants)
                      atIndex:IREE_HAL_METAL_PUSH_CONSTANT_BUFFER_INDEX];
  }

  if (iree_status_is_ok(status)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1, &executable));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_executable_t* executable,
    int32_t entry_point, uint32_t workgroup_count_x, uint32_t workgroup_count_y,
    uint32_t workgroup_count_z) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_metal_kernel_params_t kernel_params;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_metal_command_buffer_prepare_dispatch(base_command_buffer, executable,
                                                         entry_point, &kernel_params));

  id<MTLComputeCommandEncoder> compute_encoder =
      iree_hal_metal_get_or_begin_compute_encoder(command_buffer);
  uint32_t* workgroup_size = kernel_params.threadgroup_size;
  [compute_encoder
       dispatchThreadgroups:MTLSizeMake(workgroup_count_x, workgroup_count_y, workgroup_count_z)
      threadsPerThreadgroup:MTLSizeMake(workgroup_size[0], workgroup_size[1], workgroup_size[2])];

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_executable_t* executable,
    int32_t entry_point, iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_metal_kernel_params_t kernel_params;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_metal_command_buffer_prepare_dispatch(base_command_buffer, executable,
                                                         entry_point, &kernel_params));

  id<MTLComputeCommandEncoder> compute_encoder =
      iree_hal_metal_get_or_begin_compute_encoder(command_buffer);
  uint32_t* workgroup_size = kernel_params.threadgroup_size;
  id<MTLBuffer> metal_buffer =
      iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(workgroups_buffer));
  [compute_encoder
      dispatchThreadgroupsWithIndirectBuffer:metal_buffer
                        indirectBufferOffset:workgroups_offset
                       threadsPerThreadgroup:MTLSizeMake(workgroup_size[0], workgroup_size[1],
                                                         workgroup_size[2])];

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_buffer_execute_commands(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_command_buffer_t* base_commands,
    iree_hal_buffer_binding_table_t binding_table) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "secondary command buffer not yet supported");
}

static const iree_hal_command_buffer_vtable_t iree_hal_metal_command_buffer_vtable = {
    .destroy = iree_hal_metal_command_buffer_destroy,
    .begin = iree_hal_metal_command_buffer_begin,
    .end = iree_hal_metal_command_buffer_end,
    .begin_debug_group = iree_hal_metal_command_buffer_begin_debug_group,
    .end_debug_group = iree_hal_metal_command_buffer_end_debug_group,
    .execution_barrier = iree_hal_metal_command_buffer_execution_barrier,
    .signal_event = iree_hal_metal_command_buffer_signal_event,
    .reset_event = iree_hal_metal_command_buffer_reset_event,
    .wait_events = iree_hal_metal_command_buffer_wait_events,
    .discard_buffer = iree_hal_metal_command_buffer_discard_buffer,
    .fill_buffer = iree_hal_metal_command_buffer_fill_buffer,
    .update_buffer = iree_hal_metal_command_buffer_update_buffer,
    .copy_buffer = iree_hal_metal_command_buffer_copy_buffer,
    .collective = iree_hal_metal_command_buffer_collective,
    .push_constants = iree_hal_metal_command_buffer_push_constants,
    .push_descriptor_set = iree_hal_metal_command_buffer_push_descriptor_set,
    .dispatch = iree_hal_metal_command_buffer_dispatch,
    .dispatch_indirect = iree_hal_metal_command_buffer_dispatch_indirect,
    .execute_commands = iree_hal_metal_command_buffer_execute_commands,
};
