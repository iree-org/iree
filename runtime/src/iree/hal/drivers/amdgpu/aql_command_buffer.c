// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/blit.h"
#include "iree/hal/drivers/amdgpu/util/kernarg_ring.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_aql_command_buffer_t
//===----------------------------------------------------------------------===//

enum {
  IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_PAGE_CAPACITY_LOG2 = 9,
  IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_PAGE_CAPACITY = 512,
  IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_PAGE_MASK =
      IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_PAGE_CAPACITY - 1,
};

typedef struct iree_hal_amdgpu_aql_command_buffer_rodata_segment_t {
  // Command-buffer-owned immutable payload bytes.
  const uint8_t* data;
  // Byte length of |data|.
  uint32_t length;
  // Reserved bits for future rodata storage strategy flags.
  uint32_t reserved0;
} iree_hal_amdgpu_aql_command_buffer_rodata_segment_t;

typedef struct iree_hal_amdgpu_aql_command_buffer_rodata_segment_page_t {
  // Next page in ordinal order.
  struct iree_hal_amdgpu_aql_command_buffer_rodata_segment_page_t* next;
  // Number of valid entries in |segments|.
  uint32_t count;
  // Reserved bits for future page metadata.
  uint32_t reserved0;
  // Fixed-capacity rodata segment descriptors.
  iree_hal_amdgpu_aql_command_buffer_rodata_segment_t
      segments[IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_PAGE_CAPACITY];
} iree_hal_amdgpu_aql_command_buffer_rodata_segment_page_t;
static_assert((IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_PAGE_CAPACITY &
               IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_PAGE_MASK) ==
                  0,
              "rodata segment page capacity must be a power-of-two");
static_assert(
    sizeof(iree_hal_amdgpu_aql_command_buffer_rodata_segment_page_t) <=
        IREE_HAL_AMDGPU_AQL_PROGRAM_DEFAULT_BLOCK_SIZE,
    "rodata segment page should fit in a default command-buffer block");

typedef struct iree_hal_amdgpu_aql_command_buffer_t {
  // Base HAL command-buffer resource.
  iree_hal_command_buffer_t base;
  // Host allocator used to allocate the command-buffer object.
  iree_allocator_t host_allocator;
  // Block pool used for durable command-buffer program blocks.
  iree_arena_block_pool_t* block_pool;
  // Resource set retaining direct buffers and executables when not unretained.
  iree_hal_resource_set_t* resource_set;
  // Direct buffers referenced by static command records.
  iree_hal_buffer_t** static_buffers;
  // Allocated entries in |static_buffers|.
  uint32_t static_buffer_capacity;
  // Valid entries in |static_buffers|.
  uint32_t static_buffer_count;
  // Immutable payload storage captured while recording.
  struct {
    // Arena that owns segment pages and payload bytes.
    iree_arena_allocator_t arena;
    // First segment page in ordinal order.
    iree_hal_amdgpu_aql_command_buffer_rodata_segment_page_t* first_page;
    // Last segment page in ordinal order.
    iree_hal_amdgpu_aql_command_buffer_rodata_segment_page_t* current_page;
    // Total segment descriptors assigned ordinals.
    uint32_t segment_count;
  } rodata;
  // Builder used only during begin/end recording.
  iree_hal_amdgpu_aql_program_builder_t builder;
  // Program produced by end() and consumed by queue execution.
  iree_hal_amdgpu_aql_program_t program;
} iree_hal_amdgpu_aql_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_amdgpu_aql_command_buffer_vtable;

static iree_hal_amdgpu_aql_command_buffer_t*
iree_hal_amdgpu_aql_command_buffer_cast(iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_aql_command_buffer_vtable);
  return (iree_hal_amdgpu_aql_command_buffer_t*)base_value;
}

static bool iree_hal_amdgpu_aql_command_buffer_retains_resources(
    const iree_hal_amdgpu_aql_command_buffer_t* command_buffer) {
  return !iree_all_bits_set(command_buffer->base.mode,
                            IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED);
}

static void iree_hal_amdgpu_aql_command_buffer_reset_resources(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer) {
  iree_hal_resource_set_free(command_buffer->resource_set);
  command_buffer->resource_set = NULL;
  iree_allocator_free(command_buffer->host_allocator,
                      command_buffer->static_buffers);
  command_buffer->static_buffers = NULL;
  command_buffer->static_buffer_capacity = 0;
  command_buffer->static_buffer_count = 0;
  iree_arena_reset(&command_buffer->rodata.arena);
  command_buffer->rodata.first_page = NULL;
  command_buffer->rodata.current_page = NULL;
  command_buffer->rodata.segment_count = 0;
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_ensure_resource_set(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer) {
  if (!iree_hal_amdgpu_aql_command_buffer_retains_resources(command_buffer) ||
      command_buffer->resource_set) {
    return iree_ok_status();
  }
  return iree_hal_resource_set_allocate(command_buffer->block_pool,
                                        &command_buffer->resource_set);
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_ensure_static_buffers(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer) {
  if (command_buffer->static_buffer_count <
      command_buffer->static_buffer_capacity) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(command_buffer->static_buffer_capacity > UINT32_MAX / 2)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command-buffer static buffer table overflow");
  }
  iree_host_size_t capacity = command_buffer->static_buffer_capacity;
  iree_status_t status = iree_allocator_grow_array(
      command_buffer->host_allocator, /*minimum_capacity=*/16,
      sizeof(*command_buffer->static_buffers), &capacity,
      (void**)&command_buffer->static_buffers);
  if (iree_status_is_ok(status)) {
    command_buffer->static_buffer_capacity = (uint32_t)capacity;
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_record_static_buffer(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    iree_hal_buffer_t* buffer, uint32_t* out_ordinal) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_ensure_resource_set(command_buffer));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_ensure_static_buffers(command_buffer));
  if (command_buffer->resource_set) {
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
        command_buffer->resource_set, /*count=*/1, &buffer));
  }
  *out_ordinal = command_buffer->static_buffer_count;
  command_buffer->static_buffers[command_buffer->static_buffer_count++] =
      buffer;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_append_rodata_segment(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer, const uint8_t* data,
    iree_host_size_t byte_length, uint64_t* out_rodata_ordinal) {
  *out_rodata_ordinal = 0;
  if (IREE_UNLIKELY(byte_length > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command-buffer rodata segment too large");
  }
  if (IREE_UNLIKELY(command_buffer->rodata.segment_count == UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command-buffer rodata segment count overflow");
  }

  iree_hal_amdgpu_aql_command_buffer_rodata_segment_page_t* page =
      command_buffer->rodata.current_page;
  if (!page ||
      page->count ==
          IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_PAGE_CAPACITY) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(&command_buffer->rodata.arena,
                                             sizeof(*page), (void**)&page));
    memset(page, 0, sizeof(*page));
    if (command_buffer->rodata.current_page) {
      command_buffer->rodata.current_page->next = page;
    } else {
      command_buffer->rodata.first_page = page;
    }
    command_buffer->rodata.current_page = page;
  }

  const uint32_t ordinal = command_buffer->rodata.segment_count++;
  page->segments[page->count++] =
      (iree_hal_amdgpu_aql_command_buffer_rodata_segment_t){
          .data = data,
          .length = (uint32_t)byte_length,
      };
  *out_rodata_ordinal = ordinal;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_record_rodata(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_host_size_t source_length, uint64_t* out_rodata_ordinal) {
  *out_rodata_ordinal = 0;
  iree_host_size_t source_end = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(source_offset, source_length,
                                                &source_end))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "command-buffer update source span overflows host size "
        "(offset=%" PRIhsz ", length=%" PRIhsz ")",
        source_offset, source_length);
  }
  (void)source_end;

  uint8_t* rodata = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &command_buffer->rodata.arena,
      iree_max((iree_host_size_t)1, source_length), (void**)&rodata));
  if (source_length > 0) {
    memcpy(rodata, (const uint8_t*)source_buffer + source_offset,
           source_length);
  }
  return iree_hal_amdgpu_aql_command_buffer_append_rodata_segment(
      command_buffer, rodata, source_length, out_rodata_ordinal);
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_aql_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer);

iree_status_t iree_hal_amdgpu_aql_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (iree_any_bit_set(mode,
                       IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION) &&
      !iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ALLOW_INLINE_EXECUTION requires ONE_SHOT mode");
  }
  if (IREE_UNLIKELY(!block_pool)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command-buffer block pool is required");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t total_size = 0;
  iree_host_size_t validation_state_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_hal_amdgpu_aql_command_buffer_t), &total_size,
              IREE_STRUCT_FIELD(iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                                uint8_t, &validation_state_offset)));

  iree_hal_amdgpu_aql_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                (void**)&command_buffer));
  memset(command_buffer, 0, sizeof(*command_buffer));
  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t*)command_buffer + validation_state_offset,
      &iree_hal_amdgpu_aql_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->block_pool = block_pool;
  iree_arena_initialize(block_pool, &command_buffer->rodata.arena);
  iree_hal_amdgpu_aql_program_builder_initialize(block_pool,
                                                 &command_buffer->builder);

  *out_command_buffer = &command_buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdgpu_aql_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_aql_program_release(&command_buffer->program);
  iree_hal_amdgpu_aql_program_builder_deinitialize(&command_buffer->builder);
  iree_hal_amdgpu_aql_command_buffer_reset_resources(command_buffer);
  iree_arena_deinitialize(&command_buffer->rodata.arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_amdgpu_aql_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_amdgpu_aql_command_buffer_vtable);
}

const iree_hal_amdgpu_aql_program_t* iree_hal_amdgpu_aql_command_buffer_program(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  return &command_buffer->program;
}

iree_hal_buffer_t* iree_hal_amdgpu_aql_command_buffer_static_buffer(
    iree_hal_command_buffer_t* base_command_buffer, uint32_t ordinal) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  return ordinal < command_buffer->static_buffer_count
             ? command_buffer->static_buffers[ordinal]
             : NULL;
}

const uint8_t* iree_hal_amdgpu_aql_command_buffer_rodata(
    iree_hal_command_buffer_t* base_command_buffer, uint64_t ordinal,
    uint32_t length) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  if (IREE_UNLIKELY(ordinal >= command_buffer->rodata.segment_count)) {
    return NULL;
  }

  uint64_t page_ordinal =
      ordinal >>
      IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_PAGE_CAPACITY_LOG2;
  const uint32_t segment_ordinal =
      (uint32_t)(ordinal &
                 IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_PAGE_MASK);
  iree_hal_amdgpu_aql_command_buffer_rodata_segment_page_t* page =
      command_buffer->rodata.first_page;
  while (page_ordinal > 0 && page) {
    page = page->next;
    --page_ordinal;
  }
  if (IREE_UNLIKELY(!page || segment_ordinal >= page->count)) return NULL;

  const iree_hal_amdgpu_aql_command_buffer_rodata_segment_t* segment =
      &page->segments[segment_ordinal];
  return length == segment->length ? segment->data : NULL;
}

//===----------------------------------------------------------------------===//
// Recording Session
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_aql_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  iree_hal_amdgpu_aql_program_release(&command_buffer->program);
  iree_hal_amdgpu_aql_command_buffer_reset_resources(command_buffer);
  return iree_hal_amdgpu_aql_program_builder_begin(&command_buffer->builder);
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  iree_status_t status = iree_hal_amdgpu_aql_program_builder_end(
      &command_buffer->builder, &command_buffer->program);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_set_freeze(command_buffer->resource_set);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Debug Groups
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_aql_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Barriers and Events
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_aql_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);

  iree_hal_amdgpu_command_buffer_command_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_builder_append_command(
      &command_buffer->builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_barrier_command_t),
      /*fixup_count=*/0, /*aql_packet_count=*/0, /*kernarg_length=*/0, &header,
      /*out_fixups=*/NULL));

  iree_hal_amdgpu_command_buffer_barrier_command_t* barrier =
      (iree_hal_amdgpu_command_buffer_barrier_command_t*)header;
  barrier->acquire_scope = 0;
  barrier->release_scope = 0;
  barrier->barrier_flags = 0;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU command-buffer events not implemented");
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU command-buffer events not implemented");
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU command-buffer events not implemented");
}

//===----------------------------------------------------------------------===//
// Buffer Reference Recording
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_aql_command_buffer_record_buffer_ref(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_buffer_usage_t required_usage,
    iree_hal_memory_access_t required_access, uint8_t* out_kind,
    uint32_t* out_ordinal, uint64_t* out_offset, uint64_t* out_length) {
  *out_kind = IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_INVALID;
  *out_ordinal = 0;
  *out_offset = 0;
  *out_length = 0;

  if (IREE_UNLIKELY(buffer_ref.reserved != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command-buffer buffer reference reserved bits "
                            "must be zero");
  }

  if (!buffer_ref.buffer) {
    if (IREE_UNLIKELY(buffer_ref.buffer_slot >=
                      command_buffer->base.binding_capacity)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "indirect command-buffer buffer slot %u exceeds binding capacity %u",
          buffer_ref.buffer_slot, command_buffer->base.binding_capacity);
    }
    command_buffer->base.binding_count = iree_max(
        command_buffer->base.binding_count, buffer_ref.buffer_slot + 1);
    *out_kind = IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_DYNAMIC;
    *out_ordinal = buffer_ref.buffer_slot;
    *out_offset = buffer_ref.offset;
    *out_length = buffer_ref.length;
    return iree_ok_status();
  }

  if (IREE_UNLIKELY(buffer_ref.buffer_slot != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "direct command-buffer buffer references must not "
                            "specify a binding table slot");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(buffer_ref.buffer), required_usage));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(buffer_ref.buffer), required_access));

  iree_device_size_t resolved_offset = 0;
  iree_device_size_t resolved_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      /*base_offset=*/0, iree_hal_buffer_byte_length(buffer_ref.buffer),
      buffer_ref.offset, buffer_ref.length, &resolved_offset,
      &resolved_length));
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer_ref.buffer);
  if (IREE_UNLIKELY(!iree_hal_amdgpu_buffer_device_pointer(allocated_buffer))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command-buffer buffer must be backed by an AMDGPU allocation");
  }

  uint32_t ordinal = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_record_static_buffer(
      command_buffer, buffer_ref.buffer, &ordinal));

  *out_kind = IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_STATIC;
  *out_ordinal = ordinal;
  *out_offset = resolved_offset;
  *out_length = resolved_length;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Buffer Commands
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_aql_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  if (IREE_UNLIKELY(!pattern)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fill pattern must be non-null");
  }
  if (IREE_UNLIKELY(pattern_length != 1 && pattern_length != 2 &&
                    pattern_length != 4)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fill patterns must be 1, 2, or 4 bytes (got %" PRIhsz ")",
        pattern_length);
  }
  if (IREE_UNLIKELY(flags != IREE_HAL_FILL_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported fill flags: 0x%" PRIx64, flags);
  }

  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  uint8_t target_kind = 0;
  uint32_t target_ordinal = 0;
  uint64_t target_offset = 0;
  uint64_t length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_record_buffer_ref(
      command_buffer, target_ref, IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
      IREE_HAL_MEMORY_ACCESS_WRITE, &target_kind, &target_ordinal,
      &target_offset, &length));

  uint64_t pattern_bits = 0;
  memcpy(&pattern_bits, pattern, pattern_length);
  iree_hal_amdgpu_command_buffer_command_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_builder_append_command(
      &command_buffer->builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_fill_command_t),
      /*fixup_count=*/0, /*aql_packet_count=*/1,
      sizeof(iree_hal_amdgpu_kernarg_block_t), &header,
      /*out_fixups=*/NULL));

  iree_hal_amdgpu_command_buffer_fill_command_t* fill_command =
      (iree_hal_amdgpu_command_buffer_fill_command_t*)header;
  fill_command->target_offset = target_offset;
  fill_command->length = length;
  fill_command->pattern = pattern_bits;
  fill_command->target_ordinal = target_ordinal;
  fill_command->target_kind = target_kind;
  fill_command->pattern_length = (uint8_t)pattern_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  if (IREE_UNLIKELY(!source_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "update source buffer must be non-null");
  }
  if (IREE_UNLIKELY(target_ref.length >
                    IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command-buffer update length %" PRIdsz
                            " exceeds maximum update size %" PRIdsz,
                            target_ref.length,
                            IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE);
  }
  if (IREE_UNLIKELY(flags != IREE_HAL_UPDATE_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported update flags: 0x%" PRIx64, flags);
  }
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  uint8_t target_kind = 0;
  uint32_t target_ordinal = 0;
  uint64_t target_offset = 0;
  uint64_t target_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_record_buffer_ref(
      command_buffer, target_ref, IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
      IREE_HAL_MEMORY_ACCESS_WRITE, &target_kind, &target_ordinal,
      &target_offset, &target_length));

  uint64_t rodata_ordinal = 0;
  const iree_host_size_t source_length = (iree_host_size_t)target_length;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_record_rodata(
      command_buffer, source_buffer, source_offset, source_length,
      &rodata_ordinal));

  const iree_host_size_t source_payload_offset =
      IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_OFFSET;
  iree_host_size_t kernarg_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          source_payload_offset, source_length, &kernarg_length))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "command-buffer update staging payload overflows host size "
        "(offset=%" PRIhsz ", source_length=%" PRIhsz ")",
        source_payload_offset, source_length);
  }
  const iree_host_size_t kernarg_block_count = iree_host_size_ceil_div(
      kernarg_length, sizeof(iree_hal_amdgpu_kernarg_block_t));
  iree_host_size_t kernarg_block_length = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(kernarg_block_count,
                                      sizeof(iree_hal_amdgpu_kernarg_block_t),
                                      &kernarg_block_length) ||
          kernarg_block_length > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "command-buffer update staging payload requires too many kernarg "
        "bytes (%" PRIhsz ")",
        kernarg_block_length);
  }

  iree_hal_amdgpu_command_buffer_command_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_builder_append_command(
      &command_buffer->builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_UPDATE,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_update_command_t),
      /*fixup_count=*/0, /*aql_packet_count=*/1, (uint32_t)kernarg_block_length,
      &header, /*out_fixups=*/NULL));

  iree_hal_amdgpu_command_buffer_update_command_t* update_command =
      (iree_hal_amdgpu_command_buffer_update_command_t*)header;
  update_command->rodata_ordinal = rodata_ordinal;
  update_command->target_offset = target_offset;
  update_command->length = (uint32_t)target_length;
  update_command->target_ordinal = target_ordinal;
  update_command->target_kind = target_kind;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  if (IREE_UNLIKELY(flags != IREE_HAL_COPY_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported copy flags: 0x%" PRIx64, flags);
  }

  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  uint8_t source_kind = 0;
  uint32_t source_ordinal = 0;
  uint64_t source_offset = 0;
  uint64_t source_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_record_buffer_ref(
      command_buffer, source_ref, IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE,
      IREE_HAL_MEMORY_ACCESS_READ, &source_kind, &source_ordinal,
      &source_offset, &source_length));

  uint8_t target_kind = 0;
  uint32_t target_ordinal = 0;
  uint64_t target_offset = 0;
  uint64_t target_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_record_buffer_ref(
      command_buffer, target_ref, IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
      IREE_HAL_MEMORY_ACCESS_WRITE, &target_kind, &target_ordinal,
      &target_offset, &target_length));

  if (IREE_UNLIKELY(source_length != target_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "copy spans between source and target must match "
                            "(source_length=%" PRIu64 ", target_length=%" PRIu64
                            ")",
                            source_length, target_length);
  }
  if (source_ref.buffer && target_ref.buffer &&
      IREE_UNLIKELY(iree_hal_buffer_test_overlap(
                        source_ref.buffer, source_offset, source_length,
                        target_ref.buffer, target_offset,
                        target_length) != IREE_HAL_BUFFER_OVERLAP_DISJOINT)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source and target ranges must not overlap within the same buffer");
  }

  iree_hal_amdgpu_command_buffer_command_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_builder_append_command(
      &command_buffer->builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COPY,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_copy_command_t),
      /*fixup_count=*/0, /*aql_packet_count=*/1,
      sizeof(iree_hal_amdgpu_kernarg_block_t), &header,
      /*out_fixups=*/NULL));

  iree_hal_amdgpu_command_buffer_copy_command_t* copy_command =
      (iree_hal_amdgpu_command_buffer_copy_command_t*)header;
  copy_command->length = source_length;
  copy_command->source_offset = source_offset;
  copy_command->target_offset = target_offset;
  copy_command->source_ordinal = source_ordinal;
  copy_command->target_ordinal = target_ordinal;
  copy_command->source_kind = source_kind;
  copy_command->target_kind = target_kind;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU collectives not implemented");
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU command-buffer dispatch replay not implemented");
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t
    iree_hal_amdgpu_aql_command_buffer_vtable = {
        .destroy = iree_hal_amdgpu_aql_command_buffer_destroy,
        .begin = iree_hal_amdgpu_aql_command_buffer_begin,
        .end = iree_hal_amdgpu_aql_command_buffer_end,
        .begin_debug_group =
            iree_hal_amdgpu_aql_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_amdgpu_aql_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_amdgpu_aql_command_buffer_execution_barrier,
        .signal_event = iree_hal_amdgpu_aql_command_buffer_signal_event,
        .reset_event = iree_hal_amdgpu_aql_command_buffer_reset_event,
        .wait_events = iree_hal_amdgpu_aql_command_buffer_wait_events,
        .advise_buffer = iree_hal_amdgpu_aql_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_amdgpu_aql_command_buffer_fill_buffer,
        .update_buffer = iree_hal_amdgpu_aql_command_buffer_update_buffer,
        .copy_buffer = iree_hal_amdgpu_aql_command_buffer_copy_buffer,
        .collective = iree_hal_amdgpu_aql_command_buffer_collective,
        .dispatch = iree_hal_amdgpu_aql_command_buffer_dispatch,
};
