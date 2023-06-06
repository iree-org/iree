// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/webgpu/command_buffer.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "experimental/webgpu/buffer.h"
#include "experimental/webgpu/executable.h"
#include "experimental/webgpu/pipeline_layout.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// Segmented submission management
//===----------------------------------------------------------------------===//
// WebGPU - like Metal - has a rather obtuse multi-level recording model with
// the most obtuse design point being that DMA operations happen on the queue
// directly. In trying to model a single command buffer we may need to make
// multiple ordered submissions to the device queue, which is unfortunate as
// the queue submission routine only takes command buffers and we need to
// interleave the command buffer submissions with other queue operations.

typedef enum iree_hal_webgpu_command_segment_action_e {
  // wgpuQueueSubmit of a command buffer.
  IREE_HAL_WEBGPU_COMMAND_SEGMENT_ACTION_EXECUTE,
  // wgpuQueueWriteBuffer for a host->device transfer.
  IREE_HAL_WEBGPU_COMMAND_SEGMENT_ACTION_WRITE_BUFFER,
} iree_hal_webgpu_command_segment_action_t;

struct iree_hal_webgpu_command_segment_t;
typedef struct iree_hal_webgpu_command_segment_t {
  struct iree_hal_webgpu_command_segment_t* next_segment;
  iree_hal_webgpu_command_segment_action_t action;
  union {
    struct {
      WGPUCommandBuffer command_buffer;
    } execute;
    struct {
      const void* source_buffer;
      iree_host_size_t source_offset;
      WGPUBuffer target_buffer;
      iree_device_size_t target_offset;
      iree_host_size_t length;
    } write_buffer;
  };
} iree_hal_webgpu_command_segment_t;

typedef struct iree_hal_webgpu_command_segment_list_t {
  iree_hal_webgpu_command_segment_t* head;
  iree_hal_webgpu_command_segment_t* tail;
} iree_hal_webgpu_command_segment_list_t;

static void iree_hal_webgpu_command_segment_list_reset(
    iree_hal_webgpu_command_segment_list_t* list) {
  for (iree_hal_webgpu_command_segment_t* segment = list->head; segment;
       segment = segment->next_segment) {
    switch (segment->action) {
      case IREE_HAL_WEBGPU_COMMAND_SEGMENT_ACTION_WRITE_BUFFER:
        iree_wgpuCommandBufferDrop(segment->execute.command_buffer);
        break;
      default:
      case IREE_HAL_WEBGPU_COMMAND_SEGMENT_ACTION_EXECUTE:
        // Nothing to do.
        break;
    }
  }
  memset(list, 0, sizeof(*list));
}

static void iree_hal_webgpu_command_segment_list_push_front(
    iree_hal_webgpu_command_segment_list_t* list,
    iree_hal_webgpu_command_segment_t* segment) {
  segment->next_segment = list->head;
  list->head = segment;
  if (!list->tail) list->tail = segment;
}

static void iree_hal_webgpu_command_segment_list_push_back(
    iree_hal_webgpu_command_segment_list_t* list,
    iree_hal_webgpu_command_segment_t* segment) {
  segment->next_segment = NULL;
  if (list->tail) {
    list->tail->next_segment = segment;
    list->tail = segment;
  } else {
    list->head = list->tail = segment;
  }
}

static void iree_hal_webgpu_command_segment_issue_execute(
    iree_hal_webgpu_command_segment_t* segment, WGPUQueue queue) {
  IREE_TRACE_ZONE_BEGIN(z0);
  wgpuQueueSubmit(queue, 1, &segment->execute.command_buffer);
  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_webgpu_command_segment_issue_write_buffer(
    iree_hal_webgpu_command_segment_t* segment, WGPUQueue queue) {
  IREE_TRACE_ZONE_BEGIN(z0);
  wgpuQueueWriteBuffer(queue, segment->write_buffer.target_buffer,
                       segment->write_buffer.target_offset,
                       ((const uint8_t*)segment->write_buffer.source_buffer) +
                           segment->write_buffer.source_offset,
                       segment->write_buffer.length);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_command_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  WGPUDevice device;

  // Shared staging uniform buffer with queue-ordered data. We use this
  // for push constant emulation by recording all of the push constants per
  // dispatch and then updating the buffer prior to issuing the commands using
  // it. This works because there's no out-of-order or overlapping execution in
  // WebGPU (unfortunately) and we know that if we write in queue-order the
  // updates will be visible to the subsequently issued commands.
  iree_hal_webgpu_staging_buffer_t* staging_buffer;

  // Device-shared WGPUBindGroup cache.
  iree_hal_webgpu_bind_group_cache_t* bind_group_cache;

  // Shaders emulating functionality not present in WebGPU.
  // Owned by the parent device.
  iree_hal_webgpu_builtins_t* builtins;

  // Arena used for all allocations; references the shared device block pool.
  iree_arena_allocator_t arena;

  // Linked list of queue submission actions.
  iree_hal_webgpu_command_segment_list_t segments;

  struct {
    // Valid only when recording.
    WGPUCommandEncoder encoder;
    // Currently open pass - NULL if no open pass.
    WGPUComputePassEncoder compute_pass;

    // All available push constants updated each time push_constants is called.
    // Reset only with the command buffer and otherwise will maintain its values
    // during recording to allow for partial push_constants updates.
    uint32_t push_constants[IREE_HAL_WEBGPU_MAX_PUSH_CONSTANT_COUNT];

    // TODO(benvanik): add a push_constants dirty bit so we know if we need to
    // upload more. Today we'll stage the same values for each dispatch.

    // Snapshot of descriptor sets as populated by push_descriptor_set.
    // Each push_descriptor_set will invalidate the bind group handle and
    // subsequent dispatches will acquire new bind groups from the cache. If
    // future updates are no-ops the same bind group handle can be used.
    struct {
      WGPUBindGroup handle;
      iree_hal_webgpu_bind_group_binding_t
          bindings[IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_BINDING_COUNT];
    } bind_groups[IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_COUNT];

    // Bitfield tracking which bind groups are set to an empty group.
    uint64_t bind_groups_empty;
  } state;
} iree_hal_webgpu_command_buffer_t;

extern const iree_hal_command_buffer_vtable_t
    iree_hal_webgpu_command_buffer_vtable;

static iree_hal_webgpu_command_buffer_t* iree_hal_webgpu_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_command_buffer_vtable);
  return (iree_hal_webgpu_command_buffer_t*)base_value;
}

iree_status_t iree_hal_webgpu_command_buffer_create(
    iree_hal_device_t* device, WGPUDevice device_handle,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool,
    iree_hal_webgpu_staging_buffer_t* staging_buffer,
    iree_hal_webgpu_bind_group_cache_t* bind_group_cache,
    iree_hal_webgpu_builtins_t* builtins, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(staging_buffer);
  IREE_ASSERT_ARGUMENT(bind_group_cache);
  IREE_ASSERT_ARGUMENT(builtins);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_webgpu_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*command_buffer), (void**)&command_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_command_buffer_initialize(
        device, mode, command_categories, queue_affinity, binding_capacity,
        &iree_hal_webgpu_command_buffer_vtable, &command_buffer->base);
    command_buffer->host_allocator = host_allocator;
    command_buffer->device = device_handle;
    command_buffer->staging_buffer = staging_buffer;
    command_buffer->bind_group_cache = bind_group_cache;
    command_buffer->builtins = builtins;

    iree_arena_initialize(block_pool, &command_buffer->arena);
    iree_hal_webgpu_command_segment_list_reset(&command_buffer->segments);

    *out_command_buffer = &command_buffer->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

bool iree_hal_webgpu_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_webgpu_command_buffer_vtable);
}

static void* iree_hal_webgpu_command_buffer_dyn_cast(
    iree_hal_command_buffer_t* command_buffer, const void* vtable) {
  if (vtable == &iree_hal_webgpu_command_buffer_vtable) {
    IREE_HAL_ASSERT_TYPE(command_buffer, vtable);
    return command_buffer;
  }
  return NULL;
}

static void iree_hal_webgpu_command_buffer_reset(
    iree_hal_webgpu_command_buffer_t* command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (command_buffer->state.compute_pass) {
    wgpuComputePassEncoderEnd(command_buffer->state.compute_pass);
  }
  if (command_buffer->state.encoder) {
    const WGPUCommandBufferDescriptor descriptor = {
        .nextInChain = NULL,
        .label = NULL,
    };
    iree_wgpuCommandBufferDrop(
        wgpuCommandEncoderFinish(command_buffer->state.encoder, &descriptor));
    command_buffer->state.encoder = NULL;
  }

  command_buffer->state.bind_groups_empty = 0;

  iree_hal_webgpu_staging_buffer_reset(command_buffer->staging_buffer);
  iree_hal_webgpu_command_segment_list_reset(&command_buffer->segments);
  iree_arena_reset(&command_buffer->arena);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_webgpu_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_webgpu_command_buffer_reset(command_buffer);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_webgpu_command_buffer_issue(
    iree_hal_command_buffer_t* base_command_buffer, WGPUQueue queue) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);
  IREE_ASSERT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_hal_webgpu_command_segment_t* segment =
           command_buffer->segments.head;
       segment; segment = segment->next_segment) {
    switch (segment->action) {
      case IREE_HAL_WEBGPU_COMMAND_SEGMENT_ACTION_EXECUTE:
        iree_hal_webgpu_command_segment_issue_execute(segment, queue);
        break;
      case IREE_HAL_WEBGPU_COMMAND_SEGMENT_ACTION_WRITE_BUFFER:
        iree_hal_webgpu_command_segment_issue_write_buffer(segment, queue);
        break;
      default:
        break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_flush_encoder(
    iree_hal_webgpu_command_buffer_t* command_buffer) {
  if (!command_buffer->state.encoder) return iree_ok_status();

  // End any open compute pass.
  if (command_buffer->state.compute_pass) {
    wgpuComputePassEncoderEnd(command_buffer->state.compute_pass);
    command_buffer->state.compute_pass = NULL;
  }

  // Finalize encoder and produce a command buffer.
  const WGPUCommandBufferDescriptor descriptor = {
      .nextInChain = NULL,
      .label = NULL,
  };
  WGPUCommandBuffer handle =
      wgpuCommandEncoderFinish(command_buffer->state.encoder, &descriptor);
  command_buffer->state.encoder = NULL;

  iree_hal_webgpu_command_segment_t* segment = NULL;
  iree_status_t status = iree_arena_allocate(
      &command_buffer->arena, sizeof(*segment), (void**)&segment);
  if (iree_status_is_ok(status)) {
    // Attach the command buffer segment.
    segment->action = IREE_HAL_WEBGPU_COMMAND_SEGMENT_ACTION_EXECUTE;
    segment->execute.command_buffer = handle;
    iree_hal_webgpu_command_segment_list_push_back(&command_buffer->segments,
                                                   segment);
  } else {
    iree_wgpuCommandBufferDrop(handle);
  }
  return status;
}

static iree_status_t iree_hal_webgpu_command_buffer_acquire_command_encoder(
    iree_hal_webgpu_command_buffer_t* command_buffer,
    WGPUCommandEncoder* out_command_encoder) {
  // Close active compute pass, if any.
  if (command_buffer->state.compute_pass) {
    wgpuComputePassEncoderEnd(command_buffer->state.compute_pass);
    command_buffer->state.compute_pass = NULL;
  }

  // Reuse an open encoder, if any.
  if (command_buffer->state.encoder) {
    *out_command_encoder = command_buffer->state.encoder;
    return iree_ok_status();
  }

  // Open a new encoder.
  const WGPUCommandEncoderDescriptor descriptor = {
      .nextInChain = NULL,
      .label = NULL,
  };
  command_buffer->state.encoder =
      wgpuDeviceCreateCommandEncoder(command_buffer->device, &descriptor);
  *out_command_encoder = command_buffer->state.encoder;

  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_acquire_compute_pass(
    iree_hal_webgpu_command_buffer_t* command_buffer,
    WGPUComputePassEncoder* out_compute_pass) {
  // Reuse an open compute pass, if any.
  if (command_buffer->state.compute_pass) {
    *out_compute_pass = command_buffer->state.compute_pass;
    return iree_ok_status();
  }

  // Open/reuse an encoder.
  WGPUCommandEncoder command_encoder = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_command_buffer_acquire_command_encoder(
      command_buffer, &command_encoder));

  // Open a new compute pass.
  const WGPUComputePassDescriptor descriptor = {
      .nextInChain = NULL,
      .label = NULL,
  };
  command_buffer->state.compute_pass =
      wgpuCommandEncoderBeginComputePass(command_encoder, &descriptor);
  *out_compute_pass = command_buffer->state.compute_pass;

  // Reset all device-side state for the compute pass - nothing carries over
  // across passes and we will need to rebind things.
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(command_buffer->state.bind_groups); ++i) {
    command_buffer->state.bind_groups[i].handle = NULL;
  }
  command_buffer->state.bind_groups_empty = 0;

  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_flush(
    iree_hal_webgpu_command_buffer_t* command_buffer) {
  // Flush any active encoder as we are beginning a new segment.
  IREE_RETURN_IF_ERROR(
      iree_hal_webgpu_command_buffer_flush_encoder(command_buffer));

  // Flush the staging buffer to get the upload parameters.
  void* source_buffer = NULL;
  WGPUBuffer target_buffer = NULL;
  iree_host_size_t upload_length = 0;
  iree_hal_webgpu_staging_buffer_flush(command_buffer->staging_buffer,
                                       &source_buffer, &target_buffer,
                                       &upload_length);

  // Enqueue new segment.
  uint8_t* storage_base = NULL;
  iree_hal_webgpu_command_segment_t* segment = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&command_buffer->arena,
                                           sizeof(*segment) + upload_length,
                                           (void**)&storage_base));

  // Copy the staging upload data into the command buffer so the host staging
  // buffer can be reused immediately. This results in an extra copy but this
  // is mostly small. We could - if executing inline - submit this to the
  // queue immediately without the segment overhead.
  uint8_t* storage_buffer = storage_base + sizeof(*segment);
  memcpy(storage_buffer, source_buffer, upload_length);

  // Attach the write_buffer segment.
  segment = (iree_hal_webgpu_command_segment_t*)storage_base;
  segment->action = IREE_HAL_WEBGPU_COMMAND_SEGMENT_ACTION_WRITE_BUFFER;
  segment->write_buffer.source_buffer = storage_buffer;
  segment->write_buffer.source_offset = 0;
  segment->write_buffer.target_buffer = target_buffer;
  segment->write_buffer.target_offset = 0;
  segment->write_buffer.length = upload_length;
  iree_hal_webgpu_command_segment_list_push_back(&command_buffer->segments,
                                                 segment);

  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_append_parameters(
    iree_hal_webgpu_command_buffer_t* command_buffer,
    iree_const_byte_span_t source, uint32_t* out_offset) {
  // Try to append the parameters - this may fail if the staging buffer is
  // exhausted and needs to be flushed. If so we flush and then try again.
  iree_status_t try_status = iree_hal_webgpu_staging_buffer_append(
      command_buffer->staging_buffer, source, out_offset);
  if (iree_status_is_ok(try_status) ||
      !iree_status_is_resource_exhausted(try_status)) {
    return try_status;  // NOTE: may be a failure.
  }

  // Flush any pending commands and the current staging buffer state.
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_command_buffer_flush(command_buffer));

  // Try to stage the parameters again. If this fails it's not because it needed
  // a flush.
  return iree_hal_webgpu_staging_buffer_append(command_buffer->staging_buffer,
                                               source, out_offset);
}

static iree_status_t iree_hal_webgpu_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);
  iree_hal_webgpu_command_buffer_reset(command_buffer);
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);
  return iree_hal_webgpu_command_buffer_flush(command_buffer);
}

static void iree_hal_webgpu_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);

  WGPUCommandEncoder command_encoder = NULL;
  iree_status_t status = iree_hal_webgpu_command_buffer_acquire_command_encoder(
      command_buffer, &command_encoder);
  if (!iree_status_is_ok(status)) {
    // TODO(benvanik): mark recording as failed.
    iree_status_ignore(status);
    return;
  }

  // TODO(benvanik): ensure this works right when in a compute pass.
  char label_str[128] = {0};
  memcpy(label_str, label.data, iree_min(sizeof(label_str) - 1, label.size));
  wgpuCommandEncoderPushDebugGroup(command_encoder, label_str);
}

static void iree_hal_webgpu_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);

  WGPUCommandEncoder command_encoder = NULL;
  iree_status_t status = iree_hal_webgpu_command_buffer_acquire_command_encoder(
      command_buffer, &command_encoder);
  if (!iree_status_is_ok(status)) {
    // TODO(benvanik): mark recording as failed.
    iree_status_ignore(status);
    return;
  }

  wgpuCommandEncoderPopDebugGroup(command_encoder);
}

static iree_status_t iree_hal_webgpu_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  // No-op: barriers are automatic in WebGPU.
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  // No-op: no events in WebGPU.
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  // No-op: no events in WebGPU.
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  // No-op: no events in WebGPU.
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
  // No-op: though maybe it'd be a useful addition to the spec as otherwise
  // false dependencies can creep in.
  return iree_ok_status();
}

// Splats a pattern value of 1, 2, or 4 bytes out to a 4 byte value.
static uint32_t iree_hal_webgpu_splat_pattern(const void* pattern,
                                              size_t pattern_length) {
  switch (pattern_length) {
    case 1: {
      uint32_t pattern_value = *(const uint8_t*)(pattern);
      return (pattern_value << 24) | (pattern_value << 16) |
             (pattern_value << 8) | pattern_value;
    }
    case 2: {
      uint32_t pattern_value = *(const uint16_t*)(pattern);
      return (pattern_value << 16) | pattern_value;
    }
    case 4: {
      uint32_t pattern_value = *(const uint32_t*)(pattern);
      return pattern_value;
    }
    default:
      return 0;  // Already verified that this should not be possible.
  }
}

static iree_status_t iree_hal_webgpu_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);

  iree_hal_webgpu_builtin_fill_buffer_t* builtin =
      &command_buffer->builtins->fill_buffer;
  target_offset += iree_hal_buffer_byte_offset(target_buffer);

  // TODO(scotttodd): change to using what the vulkan emulation does
  uint32_t dword_pattern =
      iree_hal_webgpu_splat_pattern(pattern, pattern_length);

  // If the pattern is zero and both the offset and length are multiples of 4,
  // we can use the native wgpuCommandEncoderClearBuffer function. Otherwise,
  // we dispatch our own fill emulation shader.
  uint32_t zero_pattern = 0;
  if (memcmp(&dword_pattern, &zero_pattern, pattern_length) == 0 &&
      target_offset % 4 == 0 && length % 4 == 0) {
    WGPUCommandEncoder command_encoder = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_webgpu_command_buffer_acquire_command_encoder(
        command_buffer, &command_encoder));

    wgpuCommandEncoderClearBuffer(
        command_encoder,
        iree_hal_webgpu_buffer_handle(
            iree_hal_buffer_allocated_buffer(target_buffer)),
        target_offset, length);
    return iree_ok_status();
  }

  // need to handle %4!=0 offset and pattern length as with vulkan

  // Upload push constant data - this may incur a segment flush if the staging
  // buffer is exhausted.
  const uint32_t params_data[] = {
      /*offset=*/target_offset,
      /*length=*/length,
      /*pattern=*/dword_pattern,
  };
  uint32_t params_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_command_buffer_append_parameters(
      command_buffer,
      iree_make_const_byte_span(params_data, sizeof(params_data)),
      &params_offset));

  // Acquire the compute pass we'll encode the dispatch into - this may be
  // fresh or reused from prior commands.
  WGPUComputePassEncoder compute_pass = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_command_buffer_acquire_compute_pass(
      command_buffer, &compute_pass));
  wgpuComputePassEncoderSetPipeline(compute_pass, builtin->pipeline);

  // Bind the push constant emulation bind group at the staging buffer relative
  // offset for this dispatch.
  wgpuComputePassEncoderSetBindGroup(compute_pass, /*groupIndex=*/0,
                                     command_buffer->staging_buffer->bind_group,
                                     1, &params_offset);
  command_buffer->state.bind_groups[0].handle = NULL;

  // Grab a (probably uncached) bind group for the target buffer binding.
  const iree_hal_webgpu_bind_group_binding_t buffer_binding = {
      .type = WGPUBufferBindingType_Storage,
      .buffer = iree_hal_webgpu_buffer_handle(
          iree_hal_buffer_allocated_buffer(target_buffer)),
      .offset = 0,
      .length = length,
  };
  WGPUBindGroup buffer_group = iree_hal_webgpu_bind_group_cache_acquire(
      command_buffer->bind_group_cache, builtin->buffer_group_layout,
      &buffer_binding, /*binding_mask=*/1);
  wgpuComputePassEncoderSetBindGroup(compute_pass, /*groupIndex=*/1,
                                     buffer_group, 0, NULL);
  command_buffer->state.bind_groups[1].handle = NULL;

  // NOTE: this is not the right way to do this - we need to be tiling inside
  // the fill.
  wgpuComputePassEncoderDispatchWorkgroups(compute_pass, length, 1, 1);

  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);

  // Flush any active encoder as we are beginning a new segment.
  IREE_RETURN_IF_ERROR(
      iree_hal_webgpu_command_buffer_flush_encoder(command_buffer));

  // Enqueue new segment.
  uint8_t* storage_base = NULL;
  iree_hal_webgpu_command_segment_t* segment = NULL;
  iree_status_t status = iree_arena_allocate(
      &command_buffer->arena, sizeof(*segment) + length, (void**)&storage_base);
  if (iree_status_is_ok(status)) {
    // Copy the update data into the command buffer so the user can change
    // it immediately after this call returns. This results in a double copy
    // because we need to put it in our command buffer and then when issuing
    // copy again into the WebGPU queue. Thankfully these updates are restricted
    // to a handful of KB so that's not really our biggest inefficiency.
    uint8_t* storage_buffer = storage_base + sizeof(*segment);
    memcpy(storage_buffer, (const uint8_t*)source_buffer + source_offset,
           length);

    // Attach the write_buffer segment.
    segment = (iree_hal_webgpu_command_segment_t*)storage_base;
    segment->action = IREE_HAL_WEBGPU_COMMAND_SEGMENT_ACTION_WRITE_BUFFER;
    segment->write_buffer.source_buffer = storage_buffer;
    segment->write_buffer.source_offset = 0;
    segment->write_buffer.target_buffer =
        iree_hal_webgpu_buffer_handle(target_buffer);
    segment->write_buffer.target_offset = target_offset;
    segment->write_buffer.length = length;
    iree_hal_webgpu_command_segment_list_push_back(&command_buffer->segments,
                                                   segment);
  }
  return status;
}

static iree_status_t iree_hal_webgpu_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);

  WGPUCommandEncoder command_encoder = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_command_buffer_acquire_command_encoder(
      command_buffer, &command_encoder));

  wgpuCommandEncoderCopyBufferToBuffer(
      command_encoder, iree_hal_webgpu_buffer_handle(source_buffer),
      source_offset, iree_hal_webgpu_buffer_handle(target_buffer),
      target_offset, length);

  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);

  if (IREE_UNLIKELY(offset + values_length >=
                    sizeof(command_buffer->state.push_constants))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "push constant range %zu (length=%zu) out of range",
                            offset, values_length);
  }

  // NOTE: command buffer state change only; enqueues no tasks.
  memcpy((uint8_t*)&command_buffer->state.push_constants + offset, values,
         values_length);

  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);

  // NOTE: we don't check for redundant sets here as the compiler should have
  // done that for us.
  command_buffer->state.bind_groups[set].handle = NULL;
  iree_hal_webgpu_bind_group_binding_t* group_bindings =
      command_buffer->state.bind_groups[set].bindings;
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    uint32_t ordinal = bindings[i].binding;
    if (ordinal >= IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_BINDING_COUNT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "binding ordinal %d is out of range, must be 0-%d", ordinal,
          IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_BINDING_COUNT);
    }
    iree_hal_webgpu_bind_group_binding_t* group_binding =
        &group_bindings[bindings[i].binding];

    // TODO(benvanik): lookup binding type from layout. We should also be
    // tagging whether it's dynamic here.
    group_binding->type = WGPUBufferBindingType_Storage;

    group_binding->buffer =
        bindings[i].buffer ? iree_hal_webgpu_buffer_handle(bindings[i].buffer)
                           : NULL;
    group_binding->offset = bindings[i].offset;
    group_binding->length = bindings[i].length;
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_prepare_dispatch(
    iree_hal_webgpu_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, uint32_t ordinal,
    WGPUComputePassEncoder* out_compute_pass) {
  const iree_hal_webgpu_entry_point_t* entry_point =
      iree_hal_webgpu_executable_lookup_entry_point(executable, ordinal);

  // Upload push constant data - this may incur a segment flush if the staging
  // buffer is exhausted.
  iree_host_size_t push_constant_count =
      iree_hal_webgpu_pipeline_layout_push_constant_count(entry_point->layout);
  iree_const_byte_span_t push_constant_data = iree_make_const_byte_span(
      command_buffer->state.push_constants,
      push_constant_count * sizeof(command_buffer->state.push_constants[0]));
  uint32_t params_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_command_buffer_append_parameters(
      command_buffer, push_constant_data, &params_offset));

  // Acquire the compute pass we'll encode the dispatch into - this may be
  // fresh or reused from prior commands.
  WGPUComputePassEncoder compute_pass = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_command_buffer_acquire_compute_pass(
      command_buffer, &compute_pass));
  wgpuComputePassEncoderSetPipeline(compute_pass, entry_point->pipeline);

  if (push_constant_count > 0) {
    // Bind the push constant emulation bind group at the staging buffer
    // relative offset for this dispatch.
    wgpuComputePassEncoderSetBindGroup(
        compute_pass, IREE_HAL_WEBGPU_PARAMS_BIND_GROUP_INDEX,
        command_buffer->staging_buffer->bind_group, 1, &params_offset);
  }

  // Set all bindings.
  const iree_hal_webgpu_set_binding_info_t* binding_info =
      iree_hal_webgpu_pipeline_layout_set_binding_info(entry_point->layout);
  for (iree_host_size_t i = 0; i < binding_info->set_count; ++i) {
    // If there are no bindings in this set we can skip it.
    if (binding_info->set_masks[i] == 0) continue;

    // If there is a bind group handle then it means we've done the lookup and
    // set the bind group on the device already - we can skip.
    if (command_buffer->state.bind_groups[i].handle) continue;

    // Acquire the bind group to use for the current descriptor set.
    WGPUBindGroup handle = iree_hal_webgpu_bind_group_cache_acquire(
        command_buffer->bind_group_cache, binding_info->set_layouts[i],
        command_buffer->state.bind_groups[i].bindings,
        binding_info->set_masks[i]);

    // NOTE: today we don't support dynamic offsets for push descriptor sets.
    // This will be a larger change we'll need to handle in the compiler. If we
    // wanted to improve caching we could make all the bindings dynamic and then
    // always cache the base offsets, however
    // maxDynamicStorageBuffersPerPipelineLayout is minimally 4 and that's not
    // a lot of bindings.
    wgpuComputePassEncoderSetBindGroup(compute_pass, (uint32_t)i, handle, 0,
                                       NULL);
    command_buffer->state.bind_groups[i].handle = handle;
    command_buffer->state.bind_groups_empty &= ~(1ull << i);
  }

  if (push_constant_count > 0) {
    // Pad up to IREE_HAL_WEBGPU_PARAMS_BIND_GROUP_INDEX with empty bind groups.
    WGPUBindGroup empty_handle =
        command_buffer->staging_buffer->empty_bind_group;
    for (iree_host_size_t i = binding_info->set_count;
         i < IREE_HAL_WEBGPU_PARAMS_BIND_GROUP_INDEX; ++i) {
      // Skip if an empty group is already set at this index.
      if ((command_buffer->state.bind_groups_empty >> i) & 1ull) continue;

      wgpuComputePassEncoderSetBindGroup(compute_pass, (uint32_t)i,
                                         empty_handle, 0, NULL);
      command_buffer->state.bind_groups[i].handle = empty_handle;
      command_buffer->state.bind_groups_empty |= 1ull << i;
    }
  }

  *out_compute_pass = compute_pass;
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);

  WGPUComputePassEncoder compute_pass = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_command_buffer_prepare_dispatch(
      command_buffer, executable, entry_point, &compute_pass));
  wgpuComputePassEncoderDispatchWorkgroups(compute_pass, workgroup_x,
                                           workgroup_y, workgroup_z);

  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);

  WGPUComputePassEncoder compute_pass = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_command_buffer_prepare_dispatch(
      command_buffer, executable, entry_point, &compute_pass));
  wgpuComputePassEncoderDispatchWorkgroupsIndirect(
      compute_pass, iree_hal_webgpu_buffer_handle(workgroups_buffer),
      workgroups_offset);

  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_execute_commands(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_command_buffer_t* base_commands,
    iree_hal_buffer_binding_table_t binding_table) {
  // TODO(#10144): support indirect command buffers via deferred command buffers
  // as WebGPU has no concept of reusable dispatch command encoders. One day
  // hopefully there's an equivalent of GPURenderBundle but given WebGPU's other
  // limitations it may not be useful.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "indirect command buffers not yet implemented");
}

const iree_hal_command_buffer_vtable_t iree_hal_webgpu_command_buffer_vtable = {
    .destroy = iree_hal_webgpu_command_buffer_destroy,
    .begin = iree_hal_webgpu_command_buffer_begin,
    .end = iree_hal_webgpu_command_buffer_end,
    .begin_debug_group = iree_hal_webgpu_command_buffer_begin_debug_group,
    .end_debug_group = iree_hal_webgpu_command_buffer_end_debug_group,
    .execution_barrier = iree_hal_webgpu_command_buffer_execution_barrier,
    .signal_event = iree_hal_webgpu_command_buffer_signal_event,
    .reset_event = iree_hal_webgpu_command_buffer_reset_event,
    .wait_events = iree_hal_webgpu_command_buffer_wait_events,
    .discard_buffer = iree_hal_webgpu_command_buffer_discard_buffer,
    .fill_buffer = iree_hal_webgpu_command_buffer_fill_buffer,
    .update_buffer = iree_hal_webgpu_command_buffer_update_buffer,
    .copy_buffer = iree_hal_webgpu_command_buffer_copy_buffer,
    .push_constants = iree_hal_webgpu_command_buffer_push_constants,
    .push_descriptor_set = iree_hal_webgpu_command_buffer_push_descriptor_set,
    .dispatch = iree_hal_webgpu_command_buffer_dispatch,
    .dispatch_indirect = iree_hal_webgpu_command_buffer_dispatch_indirect,
    .execute_commands = iree_hal_webgpu_command_buffer_execute_commands,
};
