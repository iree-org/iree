// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"

#include <string.h>

#include "iree/base/alignment.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/blit.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/transient_buffer.h"
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

typedef enum iree_hal_amdgpu_aql_command_buffer_rodata_segment_flag_bits_e {
  IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_FLAG_NONE = 0u,
  IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_FLAG_PREPUBLISHED_KERNARGS =
      1u << 0,
} iree_hal_amdgpu_aql_command_buffer_rodata_segment_flag_bits_t;

typedef struct iree_hal_amdgpu_aql_command_buffer_rodata_segment_t {
  // Command-buffer-owned immutable payload bytes.
  uint8_t* data;
  // Device pointer to prepublished payload bytes, or 0 when host-only.
  uint64_t device_pointer;
  // Byte length of |data|.
  uint32_t length;
  // Required alignment for the device pointer when materialized.
  uint32_t alignment;
  // Segment flags from
  // iree_hal_amdgpu_aql_command_buffer_rodata_segment_flag_bits_t.
  uint32_t flags;
  // Reserved bytes that must be zero.
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
  // Borrowed device allocator used during recording finalization.
  iree_hal_allocator_t* device_allocator;
  // Block pool used for durable command-buffer program blocks.
  iree_arena_block_pool_t* block_pool;
  // Physical device ordinal selected from the command buffer's queue affinity.
  uint32_t device_ordinal;
  // Reserved bytes for stable layout.
  uint32_t reserved0;
  // Resource set retaining direct buffers and executables when not unretained.
  iree_hal_resource_set_t* resource_set;
  // Direct buffers referenced by static command records.
  iree_hal_buffer_t** static_buffers;
  // Allocated entries in |static_buffers|.
  uint32_t static_buffer_capacity;
  // Valid entries in |static_buffers|.
  uint32_t static_buffer_count;
  // Device-visible storage containing prepublished static dispatch kernargs.
  iree_hal_buffer_t* prepublished_kernarg_buffer;
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

static bool iree_hal_amdgpu_aql_command_buffer_validates(
    const iree_hal_amdgpu_aql_command_buffer_t* command_buffer) {
#if IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
  return !iree_any_bit_set(command_buffer->base.mode,
                           IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED);
#else
  (void)command_buffer;
  return false;
#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
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
  iree_hal_buffer_release(command_buffer->prepublished_kernarg_buffer);
  command_buffer->prepublished_kernarg_buffer = NULL;
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
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_resource_set_allocate(
      command_buffer->block_pool, &command_buffer->resource_set);
  IREE_TRACE_ZONE_END(z0);
  return status;
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
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, command_buffer->static_buffer_capacity);
  iree_host_size_t capacity = command_buffer->static_buffer_capacity;
  iree_status_t status = iree_allocator_grow_array(
      command_buffer->host_allocator, /*minimum_capacity=*/16,
      sizeof(*command_buffer->static_buffers), &capacity,
      (void**)&command_buffer->static_buffers);
  if (iree_status_is_ok(status)) {
    command_buffer->static_buffer_capacity = (uint32_t)capacity;
  }
  IREE_TRACE_ZONE_END(z0);
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

static iree_hal_amdgpu_aql_command_buffer_rodata_segment_t*
iree_hal_amdgpu_aql_command_buffer_rodata_segment_for_ordinal(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer, uint64_t ordinal) {
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
  return &page->segments[segment_ordinal];
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_append_rodata_segment(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer, uint8_t* data,
    iree_host_size_t byte_length, uint32_t alignment, uint32_t flags,
    uint64_t* out_rodata_ordinal) {
  *out_rodata_ordinal = 0;
  if (IREE_UNLIKELY(byte_length > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command-buffer rodata segment too large");
  }
  if (IREE_UNLIKELY(!alignment || !iree_host_size_is_power_of_two(alignment))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command-buffer rodata segment alignment must be a non-zero power of "
        "two");
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
    IREE_TRACE_ZONE_BEGIN(z0);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&command_buffer->rodata.arena, sizeof(*page),
                                (void**)&page));
    memset(page, 0, sizeof(*page));
    if (command_buffer->rodata.current_page) {
      command_buffer->rodata.current_page->next = page;
    } else {
      command_buffer->rodata.first_page = page;
    }
    command_buffer->rodata.current_page = page;
    IREE_TRACE_ZONE_END(z0);
  }

  const uint32_t ordinal = command_buffer->rodata.segment_count++;
  page->segments[page->count++] =
      (iree_hal_amdgpu_aql_command_buffer_rodata_segment_t){
          .data = data,
          .device_pointer = 0,
          .length = (uint32_t)byte_length,
          .alignment = alignment,
          .flags = flags,
      };
  *out_rodata_ordinal = ordinal;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_allocate_rodata_segment(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    iree_host_size_t byte_length, uint32_t alignment, uint32_t flags,
    uint8_t** out_data, uint64_t* out_rodata_ordinal) {
  *out_data = NULL;
  *out_rodata_ordinal = 0;
  uint8_t* rodata = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, byte_length);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_arena_allocate_aligned(&command_buffer->rodata.arena,
                                  iree_max((iree_host_size_t)1, byte_length),
                                  alignment, (void**)&rodata));
  iree_status_t status =
      iree_hal_amdgpu_aql_command_buffer_append_rodata_segment(
          command_buffer, rodata, byte_length, alignment, flags,
          out_rodata_ordinal);
  if (iree_status_is_ok(status)) {
    *out_data = rodata;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
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
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_allocate_rodata_segment(
          command_buffer, source_length, /*alignment=*/1,
          IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_FLAG_NONE, &rodata,
          out_rodata_ordinal));
  if (source_length > 0) {
    memcpy(rodata, (const uint8_t*)source_buffer + source_offset,
           source_length);
  }
  return iree_ok_status();
}

static bool iree_hal_amdgpu_aql_command_buffer_rodata_is_prepublished_kernarg(
    const iree_hal_amdgpu_aql_command_buffer_rodata_segment_t* segment) {
  return iree_all_bits_set(
      segment->flags,
      IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_FLAG_PREPUBLISHED_KERNARGS);
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_count_prepublished_kernargs(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    iree_host_size_t* out_template_count, iree_host_size_t* out_payload_length,
    uint32_t* out_max_alignment) {
  *out_template_count = 0;
  *out_payload_length = 0;
  *out_max_alignment = 1;

  iree_host_size_t template_count = 0;
  iree_host_size_t payload_length = 0;
  uint32_t max_alignment = 1;
  for (iree_hal_amdgpu_aql_command_buffer_rodata_segment_page_t* page =
           command_buffer->rodata.first_page;
       page; page = page->next) {
    for (uint32_t i = 0; i < page->count; ++i) {
      const iree_hal_amdgpu_aql_command_buffer_rodata_segment_t* segment =
          &page->segments[i];
      if (!iree_hal_amdgpu_aql_command_buffer_rodata_is_prepublished_kernarg(
              segment)) {
        continue;
      }
      iree_host_size_t aligned_payload_length = 0;
      if (IREE_UNLIKELY(!iree_host_size_checked_align(
              payload_length, segment->alignment, &aligned_payload_length))) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "prepublished command-buffer kernarg offset overflow");
      }
      payload_length = aligned_payload_length;
      if (IREE_UNLIKELY(!iree_host_size_checked_add(
              payload_length,
              iree_max((iree_host_size_t)1, (iree_host_size_t)segment->length),
              &payload_length))) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "prepublished command-buffer kernarg storage overflow");
      }
      max_alignment = iree_max(max_alignment, segment->alignment);
      ++template_count;
    }
  }

  *out_template_count = template_count;
  *out_payload_length = payload_length;
  *out_max_alignment = max_alignment;
  return iree_ok_status();
}

static void iree_hal_amdgpu_aql_command_buffer_copy_prepublished_kernargs(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    iree_hal_buffer_mapping_t* mapping, uint8_t* device_base) {
  uint8_t* host_base = mapping->contents.data;
  const uintptr_t device_base_address = (uintptr_t)device_base;
  iree_host_size_t payload_offset = 0;
  for (iree_hal_amdgpu_aql_command_buffer_rodata_segment_page_t* page =
           command_buffer->rodata.first_page;
       page; page = page->next) {
    for (uint32_t i = 0; i < page->count; ++i) {
      iree_hal_amdgpu_aql_command_buffer_rodata_segment_t* segment =
          &page->segments[i];
      if (!iree_hal_amdgpu_aql_command_buffer_rodata_is_prepublished_kernarg(
              segment)) {
        continue;
      }
      const uintptr_t unaligned_address = device_base_address + payload_offset;
      const uintptr_t aligned_address =
          (unaligned_address + segment->alignment - 1) &
          ~((uintptr_t)segment->alignment - 1);
      payload_offset =
          (iree_host_size_t)(aligned_address - device_base_address);
      if (segment->length > 0) {
        memcpy(host_base + payload_offset, segment->data, segment->length);
      }
      segment->device_pointer = (uint64_t)aligned_address;
      payload_offset +=
          iree_max((iree_host_size_t)1, (iree_host_size_t)segment->length);
    }
  }
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_materialize_prepublished_kernargs(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer) {
  iree_host_size_t template_count = 0;
  iree_host_size_t payload_length = 0;
  uint32_t max_alignment = 1;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_count_prepublished_kernargs(
          command_buffer, &template_count, &payload_length, &max_alignment));
  if (template_count == 0) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(payload_length > (iree_host_size_t)IREE_DEVICE_SIZE_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "prepublished command-buffer kernarg storage length %" PRIhsz
        " exceeds device size max %" PRIdsz,
        payload_length, IREE_DEVICE_SIZE_MAX);
  }

  iree_host_size_t allocation_length = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(payload_length, max_alignment - 1,
                                      &allocation_length) ||
          allocation_length > (iree_host_size_t)IREE_DEVICE_SIZE_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "prepublished command-buffer kernarg allocation length overflow");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, allocation_length);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, template_count);

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ |
                 IREE_HAL_BUFFER_USAGE_MAPPING;

  iree_hal_buffer_t* template_buffer = NULL;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      command_buffer->device_allocator, params,
      (iree_device_size_t)allocation_length, &template_buffer);
  iree_hal_buffer_mapping_t mapping;
  memset(&mapping, 0, sizeof(mapping));
  uint8_t* device_base = NULL;
  if (iree_status_is_ok(status)) {
    device_base =
        (uint8_t*)iree_hal_amdgpu_buffer_device_pointer(template_buffer);
    if (IREE_UNLIKELY(!device_base)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "prepublished command-buffer kernarg buffer must be backed by an "
          "AMDGPU allocation");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_range(
        template_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, /*byte_offset=*/0,
        (iree_device_size_t)allocation_length, &mapping);
  }
  if (iree_status_is_ok(status)) {
    memset(mapping.contents.data, 0, allocation_length);
    iree_hal_amdgpu_aql_command_buffer_copy_prepublished_kernargs(
        command_buffer, &mapping, device_base);
  }
  if (iree_status_is_ok(status) &&
      !iree_all_bits_set(iree_hal_buffer_memory_type(template_buffer),
                         IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    status = iree_hal_buffer_mapping_flush_range(&mapping, /*byte_offset=*/0,
                                                 IREE_HAL_WHOLE_BUFFER);
  }
  if (mapping.buffer) {
    status = iree_status_join(status, iree_hal_buffer_unmap_range(&mapping));
  }
  if (iree_status_is_ok(status)) {
    command_buffer->prepublished_kernarg_buffer = template_buffer;
  } else {
    iree_hal_buffer_release(template_buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
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
    iree_host_size_t device_ordinal, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator,
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
  if (IREE_UNLIKELY(device_ordinal > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command-buffer device ordinal %" PRIhsz
                            " exceeds uint32_t storage",
                            device_ordinal);
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
  command_buffer->device_allocator = device_allocator;
  command_buffer->block_pool = block_pool;
  command_buffer->device_ordinal = (uint32_t)device_ordinal;
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

iree_host_size_t iree_hal_amdgpu_aql_command_buffer_device_ordinal(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  return command_buffer->device_ordinal;
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
  const iree_hal_amdgpu_aql_command_buffer_rodata_segment_t* segment =
      iree_hal_amdgpu_aql_command_buffer_rodata_segment_for_ordinal(
          command_buffer, ordinal);
  if (IREE_UNLIKELY(!segment)) {
    return NULL;
  }
  return length == segment->length ? segment->data : NULL;
}

void* iree_hal_amdgpu_aql_command_buffer_prepublished_kernarg(
    iree_hal_command_buffer_t* base_command_buffer, uint32_t ordinal,
    uint32_t length) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  const iree_hal_amdgpu_aql_command_buffer_rodata_segment_t* segment =
      iree_hal_amdgpu_aql_command_buffer_rodata_segment_for_ordinal(
          command_buffer, ordinal);
  if (IREE_UNLIKELY(!segment || length != segment->length)) {
    return NULL;
  }
  if (IREE_UNLIKELY(!iree_all_bits_set(
          segment->flags,
          IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_FLAG_PREPUBLISHED_KERNARGS))) {
    return NULL;
  }
  return (void*)(uintptr_t)segment->device_pointer;
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
    status =
        iree_hal_amdgpu_aql_command_buffer_materialize_prepublished_kernargs(
            command_buffer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_resource_set_freeze(command_buffer->resource_set);
  } else if (command_buffer->program.first_block) {
    iree_hal_amdgpu_aql_program_release(&command_buffer->program);
    iree_hal_amdgpu_aql_command_buffer_reset_resources(command_buffer);
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
      /*binding_source_count=*/0, /*aql_packet_count=*/0,
      /*kernarg_length=*/0, &header, /*out_binding_sources=*/NULL));

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

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_resolve_static_buffer_ref(
    const iree_hal_buffer_ref_t* buffer_ref, uint64_t* out_device_pointer) {
  *out_device_pointer = 0;
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer_ref->buffer);
  if (iree_hal_amdgpu_transient_buffer_isa(allocated_buffer)) {
    iree_hal_buffer_t* committed_backing = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_transient_buffer_resolve_committed_backing(
            allocated_buffer, &committed_backing));
    allocated_buffer = iree_hal_buffer_allocated_buffer(committed_backing);
  }
  void* device_ptr = iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command-buffer buffer reference must be backed by an AMDGPU "
        "allocation");
  }
  iree_device_size_t device_offset = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_add(
          iree_hal_buffer_byte_offset(buffer_ref->buffer), buffer_ref->offset,
          &device_offset))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "command-buffer buffer reference device pointer offset overflows "
        "device size");
  }
  if (IREE_UNLIKELY(device_offset > UINTPTR_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "command-buffer buffer reference device pointer offset exceeds host "
        "pointer size");
  }
  *out_device_pointer =
      (uint64_t)((uintptr_t)device_ptr + (uintptr_t)device_offset);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_record_buffer_ref(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t buffer_ref,
    iree_hal_amdgpu_command_buffer_binding_kind_t* out_kind,
    uint32_t* out_ordinal, uint64_t* out_offset, uint64_t* out_length) {
  *out_kind = IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_INVALID;
  *out_ordinal = 0;
  *out_offset = 0;
  *out_length = 0;

  if (!buffer_ref.buffer) {
    if (IREE_UNLIKELY(buffer_ref.buffer_slot == UINT32_MAX)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "indirect command-buffer buffer slot %u exceeds binding count "
          "storage",
          buffer_ref.buffer_slot);
    }
    command_buffer->base.binding_count = iree_max(
        command_buffer->base.binding_count, buffer_ref.buffer_slot + 1);
    *out_kind = IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_DYNAMIC;
    *out_ordinal = buffer_ref.buffer_slot;
    *out_offset = buffer_ref.offset;
    *out_length = buffer_ref.length;
    return iree_ok_status();
  }

  iree_device_size_t resolved_offset = 0;
  iree_device_size_t resolved_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      /*base_offset=*/0, iree_hal_buffer_byte_length(buffer_ref.buffer),
      buffer_ref.offset, buffer_ref.length, &resolved_offset,
      &resolved_length));
  uint64_t unused_device_pointer = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_resolve_static_buffer_ref(
          &buffer_ref, &unused_device_pointer));

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
// Dispatch Recording
//===----------------------------------------------------------------------===//

static bool iree_hal_amdgpu_dispatch_config_has_workgroup_size_override(
    const iree_hal_dispatch_config_t config) {
  return config.workgroup_size[0] || config.workgroup_size[1] ||
         config.workgroup_size[2];
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_check_dispatch_flags(
    iree_hal_dispatch_flags_t flags) {
  if (iree_hal_dispatch_uses_indirect_arguments(flags)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "indirect dispatch arguments are not supported by AMDGPU command "
        "buffers yet");
  }
  const iree_hal_dispatch_flags_t supported_flags =
      IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS |
      IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS |
      IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS |
      IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION;
  if (IREE_UNLIKELY(iree_any_bit_set(flags, ~supported_flags))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported dispatch flags: 0x%" PRIx64, flags);
  }
  return iree_ok_status();
}

static void iree_hal_amdgpu_aql_command_buffer_select_dispatch_kernel_args(
    const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor,
    const iree_hal_dispatch_config_t config,
    iree_hal_amdgpu_device_kernel_args_t* override_kernel_args,
    const iree_hal_amdgpu_device_kernel_args_t** out_kernel_args) {
  *out_kernel_args = &descriptor->kernel_args;
  if (!iree_hal_amdgpu_dispatch_config_has_workgroup_size_override(config)) {
    return;
  }

  *override_kernel_args = descriptor->kernel_args;
  for (iree_host_size_t i = 0; i < 3; ++i) {
    override_kernel_args->workgroup_size[i] =
        (uint16_t)config.workgroup_size[i];
  }

  *out_kernel_args = override_kernel_args;
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_validate_dispatch_shape(
    const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor,
    const iree_hal_dispatch_config_t config, iree_hal_dispatch_flags_t flags) {
  const bool uses_indirect_parameters =
      iree_hal_dispatch_uses_indirect_parameters(flags);
  if (iree_hal_amdgpu_dispatch_config_has_workgroup_size_override(config)) {
    for (iree_host_size_t i = 0; i < 3; ++i) {
      if (IREE_UNLIKELY(!config.workgroup_size[i])) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "dispatch workgroup size override must specify all dimensions");
      }
      if (IREE_UNLIKELY(config.workgroup_size[i] > UINT16_MAX)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "dispatch workgroup size override dimension %" PRIhsz
            " value %u exceeds %u",
            i, config.workgroup_size[i], UINT16_MAX);
      }
      if (!uses_indirect_parameters) {
        const uint64_t grid_size =
            (uint64_t)config.workgroup_count[i] * config.workgroup_size[i];
        if (IREE_UNLIKELY(grid_size > UINT32_MAX)) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "dispatch grid dimension %" PRIhsz
              " overflows uint32_t (workgroup_count=%u, workgroup_size=%u)",
              i, config.workgroup_count[i], config.workgroup_size[i]);
        }
      }
    }
  } else if (!uses_indirect_parameters) {
    for (iree_host_size_t i = 0; i < 3; ++i) {
      if (IREE_UNLIKELY(config.workgroup_count[i] >
                        descriptor->max_workgroup_count[i])) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "dispatch grid dimension %" PRIhsz
            " overflows uint32_t (workgroup_count=%u, workgroup_size=%u)",
            i, config.workgroup_count[i],
            descriptor->kernel_args.workgroup_size[i]);
      }
    }
  }
  if (IREE_UNLIKELY(config.dynamic_workgroup_local_memory >
                    descriptor->max_dynamic_workgroup_local_memory)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch group segment size overflows uint32_t "
                            "(static=%u, dynamic=%u)",
                            descriptor->kernel_args.group_segment_size,
                            config.dynamic_workgroup_local_memory);
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_prepare_dispatch_binding_sources(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_buffer_ref_list_t bindings) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_ensure_resource_set(command_buffer));

  iree_host_size_t binding_count = command_buffer->base.binding_count;
  iree_status_t status = iree_ok_status();
  iree_host_size_t failed_index = 0;
  for (iree_host_size_t i = 0; i < bindings.count && iree_status_is_ok(status);
       ++i) {
    failed_index = i;
    const iree_hal_buffer_ref_t* binding = &bindings.values[i];
    if (!binding->buffer) {
      if (IREE_UNLIKELY(binding->buffer_slot == UINT32_MAX)) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "indirect command-buffer dispatch binding slot %u exceeds binding "
            "count storage",
            binding->buffer_slot);
      } else {
        binding_count = iree_max(binding_count, binding->buffer_slot + 1);
      }
      continue;
    }

    uint64_t unused_device_pointer = 0;
    status = iree_hal_amdgpu_aql_command_buffer_resolve_static_buffer_ref(
        binding, &unused_device_pointer);
    if (iree_status_is_ok(status) && command_buffer->resource_set) {
      status = iree_hal_resource_set_insert(command_buffer->resource_set,
                                            /*count=*/1, &binding->buffer);
    }
  }
  if (iree_status_is_ok(status)) {
    command_buffer->base.binding_count = (uint32_t)binding_count;
  } else {
    status =
        iree_status_annotate_f(status, "binding[%" PRIhsz "]", failed_index);
  }
  return status;
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_write_dispatch_binding_sources(
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < bindings.count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_buffer_ref_t* binding = &bindings.values[i];
    iree_hal_amdgpu_command_buffer_binding_source_t* binding_source =
        &binding_sources[i];
    if (!binding->buffer) {
      binding_source->offset_or_pointer = binding->offset;
      binding_source->slot = binding->buffer_slot;
      binding_source->flags =
          IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC;
      continue;
    }

    status = iree_hal_amdgpu_aql_command_buffer_resolve_static_buffer_ref(
        binding, &binding_source->offset_or_pointer);
    if (iree_status_is_ok(status)) {
      binding_source->slot = 0;
      binding_source->flags =
          IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_NONE;
    }
  }
  return status;
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_check_indirect_workgroup_count_ref(
    iree_hal_buffer_ref_t buffer_ref) {
  const iree_device_size_t workgroup_count_length = sizeof(uint32_t[3]);
  if (IREE_UNLIKELY((buffer_ref.offset % sizeof(uint32_t)) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "indirect workgroup count offset must be 4-byte aligned");
  }
  if (IREE_UNLIKELY(buffer_ref.length != IREE_HAL_WHOLE_BUFFER &&
                    buffer_ref.length < workgroup_count_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "indirect workgroup count buffer must contain at least uint32_t[3]");
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_write_indirect_parameter_source(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t buffer_ref,
    iree_hal_amdgpu_command_buffer_binding_source_t* binding_source) {
  memset(binding_source, 0, sizeof(*binding_source));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_check_indirect_workgroup_count_ref(
          buffer_ref));

  if (!buffer_ref.buffer) {
    if (IREE_UNLIKELY(buffer_ref.buffer_slot == UINT32_MAX)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "indirect workgroup count binding slot %u exceeds binding count "
          "storage",
          buffer_ref.buffer_slot);
    }
    command_buffer->base.binding_count = iree_max(
        command_buffer->base.binding_count, buffer_ref.buffer_slot + 1);
    binding_source->offset_or_pointer = buffer_ref.offset;
    binding_source->slot = buffer_ref.buffer_slot;
    binding_source->flags =
        IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC |
        IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(buffer_ref.buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(buffer_ref.buffer),
      IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(buffer_ref.buffer),
      IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
      buffer_ref.buffer, buffer_ref.offset, sizeof(uint32_t[3])));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_ensure_resource_set(command_buffer));
  if (command_buffer->resource_set) {
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
        command_buffer->resource_set, /*count=*/1, &buffer_ref.buffer));
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_resolve_static_buffer_ref(
          &buffer_ref, &binding_source->offset_or_pointer));
  binding_source->slot = 0;
  binding_source->flags =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_qword_length(
    iree_host_size_t byte_length, const char* label, uint16_t* out_qwords,
    iree_host_size_t* out_padded_length) {
  if (IREE_UNLIKELY(byte_length > IREE_HOST_SIZE_MAX - 7)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%s byte length %" PRIhsz
                            " overflows 8-byte alignment",
                            label, byte_length);
  }
  const iree_host_size_t padded_length = iree_host_align(byte_length, 8);
  const iree_host_size_t qword_length = padded_length / 8;
  if (IREE_UNLIKELY(qword_length > UINT16_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%s byte length %" PRIhsz
                            " exceeds uint16_t qword storage",
                            label, byte_length);
  }
  *out_qwords = (uint16_t)qword_length;
  if (out_padded_length) *out_padded_length = padded_length;
  return iree_ok_status();
}

static void iree_hal_amdgpu_aql_command_buffer_write_implicit_args(
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    const iree_hal_dispatch_config_t config,
    iree_amdgpu_kernel_implicit_args_t* implicit_args) {
  implicit_args->block_count[0] = config.workgroup_count[0];
  implicit_args->block_count[1] = config.workgroup_count[1];
  implicit_args->block_count[2] = config.workgroup_count[2];
  implicit_args->group_size[0] = kernel_args->workgroup_size[0];
  implicit_args->group_size[1] = kernel_args->workgroup_size[1];
  implicit_args->group_size[2] = kernel_args->workgroup_size[2];
  implicit_args->grid_dims = 3;
  implicit_args->printf_buffer = NULL;
  implicit_args->hostcall_buffer = NULL;
  implicit_args->dynamic_lds_size = config.dynamic_workgroup_local_memory;
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_write_dispatch_tail(
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* layout,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_amdgpu_command_buffer_kernarg_strategy_t kernarg_strategy,
    uint8_t* tail_payload) {
  switch (kernarg_strategy) {
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_HAL: {
      const iree_host_size_t binding_bytes =
          (iree_host_size_t)kernel_args->binding_count * sizeof(uint64_t);
      if (constants.data_length > 0) {
        memcpy(tail_payload, constants.data, constants.data_length);
      }
      if (layout->has_implicit_args) {
        iree_amdgpu_kernel_implicit_args_t* implicit_args =
            (iree_amdgpu_kernel_implicit_args_t*)(tail_payload +
                                                  layout->implicit_args_offset -
                                                  binding_bytes);
        iree_hal_amdgpu_aql_command_buffer_write_implicit_args(
            kernel_args, config, implicit_args);
      }
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_CUSTOM_DIRECT:
      if (constants.data_length > 0) {
        memcpy(tail_payload, constants.data, constants.data_length);
      }
      return iree_ok_status();
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_INDIRECT:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "indirect dispatch arguments are not supported by AMDGPU command "
          "buffers yet");
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported command-buffer kernarg strategy %u",
                              kernarg_strategy);
  }
}

static bool iree_hal_amdgpu_aql_command_buffer_has_dynamic_dispatch_bindings(
    iree_hal_buffer_ref_list_t bindings) {
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    if (!bindings.values[i].buffer) return true;
  }
  return false;
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_record_prepublished_dispatch_kernargs(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* layout,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings,
    iree_hal_amdgpu_command_buffer_kernarg_strategy_t kernarg_strategy,
    iree_host_size_t kernarg_padded_length, uint64_t* out_rodata_ordinal) {
  *out_rodata_ordinal = 0;
  uint8_t* kernarg_data = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_allocate_rodata_segment(
      command_buffer, kernarg_padded_length, kernel_args->kernarg_alignment,
      IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_FLAG_PREPUBLISHED_KERNARGS,
      &kernarg_data, out_rodata_ordinal));
  memset(kernarg_data, 0, iree_max((iree_host_size_t)1, kernarg_padded_length));

  switch (kernarg_strategy) {
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_HAL: {
      uint64_t* binding_dst = (uint64_t*)kernarg_data;
      for (iree_host_size_t i = 0; i < bindings.count; ++i) {
        IREE_RETURN_IF_ERROR(
            iree_hal_amdgpu_aql_command_buffer_resolve_static_buffer_ref(
                &bindings.values[i], &binding_dst[i]));
      }
      const iree_host_size_t binding_bytes =
          (iree_host_size_t)kernel_args->binding_count * sizeof(uint64_t);
      return iree_hal_amdgpu_aql_command_buffer_write_dispatch_tail(
          kernel_args, layout, config, constants, kernarg_strategy,
          kernarg_data + binding_bytes);
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_CUSTOM_DIRECT:
      return iree_hal_amdgpu_aql_command_buffer_write_dispatch_tail(
          kernel_args, layout, config, constants, kernarg_strategy,
          kernarg_data);
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unsupported prepublished command-buffer kernarg strategy %u",
          kernarg_strategy);
  }
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
  iree_hal_amdgpu_command_buffer_binding_kind_t target_kind =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_INVALID;
  uint32_t target_ordinal = 0;
  uint64_t target_offset = 0;
  uint64_t length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_record_buffer_ref(
      command_buffer, target_ref, &target_kind, &target_ordinal, &target_offset,
      &length));

  uint64_t pattern_bits = 0;
  memcpy(&pattern_bits, pattern, pattern_length);
  iree_hal_amdgpu_command_buffer_command_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_builder_append_command(
      &command_buffer->builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_fill_command_t),
      /*binding_source_count=*/0, /*aql_packet_count=*/1,
      sizeof(iree_hal_amdgpu_kernarg_block_t), &header,
      /*out_binding_sources=*/NULL));

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
  iree_hal_amdgpu_command_buffer_binding_kind_t target_kind =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_INVALID;
  uint32_t target_ordinal = 0;
  uint64_t target_offset = 0;
  uint64_t target_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_record_buffer_ref(
      command_buffer, target_ref, &target_kind, &target_ordinal, &target_offset,
      &target_length));

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
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &kernarg_block_length,
      IREE_STRUCT_FIELD(kernarg_block_count, iree_hal_amdgpu_kernarg_block_t,
                        NULL)));
  if (IREE_UNLIKELY(kernarg_block_length > UINT32_MAX)) {
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
      /*binding_source_count=*/0, /*aql_packet_count=*/1,
      (uint32_t)kernarg_block_length, &header,
      /*out_binding_sources=*/NULL));

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
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  if (IREE_UNLIKELY(flags != IREE_HAL_COPY_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported copy flags: 0x%" PRIx64, flags);
  }
  iree_hal_amdgpu_command_buffer_binding_kind_t source_kind =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_INVALID;
  uint32_t source_ordinal = 0;
  uint64_t source_offset = 0;
  uint64_t source_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_record_buffer_ref(
      command_buffer, source_ref, &source_kind, &source_ordinal, &source_offset,
      &source_length));

  iree_hal_amdgpu_command_buffer_binding_kind_t target_kind =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_INVALID;
  uint32_t target_ordinal = 0;
  uint64_t target_offset = 0;
  uint64_t target_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_record_buffer_ref(
      command_buffer, target_ref, &target_kind, &target_ordinal, &target_offset,
      &target_length));

  if (IREE_UNLIKELY(source_length != target_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "copy spans between source and target must match "
                            "(source_length=%" PRIu64 ", target_length=%" PRIu64
                            ")",
                            source_length, target_length);
  }
  iree_hal_amdgpu_command_buffer_command_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_builder_append_command(
      &command_buffer->builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COPY,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_copy_command_t),
      /*binding_source_count=*/0, /*aql_packet_count=*/1,
      sizeof(iree_hal_amdgpu_kernarg_block_t), &header,
      /*out_binding_sources=*/NULL));

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
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  const bool validates =
      iree_hal_amdgpu_aql_command_buffer_validates(command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_check_dispatch_flags(flags));

  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_device(
          executable, export_ordinal, command_buffer->device_ordinal,
          &descriptor));

  if (validates) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_aql_command_buffer_validate_dispatch_shape(
            descriptor, config, flags));
  }

  iree_hal_amdgpu_device_kernel_args_t override_kernel_args;
  const iree_hal_amdgpu_device_kernel_args_t* kernel_args = NULL;
  iree_hal_amdgpu_aql_command_buffer_select_dispatch_kernel_args(
      descriptor, config, &override_kernel_args, &kernel_args);

  const bool uses_custom_direct_arguments =
      iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
  const bool uses_indirect_parameters =
      iree_hal_dispatch_uses_indirect_parameters(flags);
  bool has_dynamic_bindings = false;
  if (IREE_UNLIKELY(constants.data_length > 0 && !constants.data)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch constant data must be non-null when length is non-zero");
  }

  const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* layout = NULL;
  uint32_t kernarg_block_count = 0;
  iree_hal_amdgpu_command_buffer_kernarg_strategy_t kernarg_strategy =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_HAL;
  if (uses_custom_direct_arguments) {
    if (IREE_UNLIKELY(constants.data_length !=
                      descriptor->kernel_args.kernarg_size)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "custom dispatch argument length mismatch; expected %u but got "
          "%" PRIhsz,
          descriptor->kernel_args.kernarg_size, constants.data_length);
    }
    layout = &descriptor->custom_kernarg_layout;
    kernarg_block_count = iree_max(1u, descriptor->custom_kernarg_block_count);
    kernarg_strategy =
        IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_CUSTOM_DIRECT;
  } else {
    const iree_host_size_t expected_constant_length =
        (iree_host_size_t)descriptor->kernel_args.constant_count *
        sizeof(uint32_t);
    if (IREE_UNLIKELY(constants.data_length != expected_constant_length)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch constant count mismatch; expected %u but got %" PRIhsz,
          (uint32_t)descriptor->kernel_args.constant_count,
          constants.data_length / sizeof(uint32_t));
    }
    if (IREE_UNLIKELY(bindings.count !=
                      descriptor->kernel_args.binding_count)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch binding count mismatch; expected %u but got %" PRIhsz,
          (uint32_t)descriptor->kernel_args.binding_count, bindings.count);
    }
    if (IREE_UNLIKELY(bindings.count > 0 && !bindings.values)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch bindings must be non-null when count is non-zero");
    }
    has_dynamic_bindings =
        iree_hal_amdgpu_aql_command_buffer_has_dynamic_dispatch_bindings(
            bindings);
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_aql_command_buffer_prepare_dispatch_binding_sources(
            command_buffer, bindings));
    layout = &descriptor->hal_kernarg_layout;
    kernarg_block_count = iree_max(1u, descriptor->hal_kernarg_block_count);
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_ensure_resource_set(command_buffer));
  if (command_buffer->resource_set) {
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
        command_buffer->resource_set, /*count=*/1, &executable));
  }

  const iree_host_size_t binding_bytes =
      uses_custom_direct_arguments
          ? 0
          : (iree_host_size_t)kernel_args->binding_count * sizeof(uint64_t);
  const iree_host_size_t tail_byte_length =
      layout->total_kernarg_size - binding_bytes;
  uint16_t tail_length_qwords = 0;
  iree_host_size_t tail_padded_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_qword_length(
      tail_byte_length, "dispatch tail payload", &tail_length_qwords,
      &tail_padded_length));
  uint16_t kernarg_length_qwords = 0;
  iree_host_size_t kernarg_padded_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_qword_length(
      layout->total_kernarg_size, "dispatch kernarg", &kernarg_length_qwords,
      &kernarg_padded_length));
  const uint16_t implicit_args_offset_qwords =
      layout->has_implicit_args ? (uint16_t)(layout->implicit_args_offset / 8)
                                : UINT16_MAX;
  if (IREE_UNLIKELY(kernarg_block_count >
                    UINT32_MAX / sizeof(iree_hal_amdgpu_kernarg_block_t))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch kernargs require too many kernarg blocks (%" PRIu32 ")",
        kernarg_block_count);
  }
  const uint32_t dispatch_kernarg_block_length =
      kernarg_block_count * sizeof(iree_hal_amdgpu_kernarg_block_t);
  const uint32_t patch_kernarg_block_length =
      uses_indirect_parameters ? sizeof(iree_hal_amdgpu_kernarg_block_t) : 0;
  const uint32_t kernarg_block_length =
      patch_kernarg_block_length + dispatch_kernarg_block_length;
  const bool prepublish_kernargs =
      !uses_indirect_parameters && !has_dynamic_bindings;
  const uint32_t queue_kernarg_block_length =
      prepublish_kernargs ? 0 : kernarg_block_length;

  iree_host_size_t command_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amdgpu_command_buffer_dispatch_command_t),
      &command_length,
      IREE_STRUCT_FIELD(prepublish_kernargs ? 0 : tail_padded_length, uint8_t,
                        NULL)));

  const uint16_t binding_source_count =
      prepublish_kernargs
          ? 0
          : (uint16_t)((uses_custom_direct_arguments ? 0 : bindings.count) +
                       (uses_indirect_parameters ? 1 : 0));
  iree_hal_amdgpu_command_buffer_command_header_t* header = NULL;
  iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_builder_append_command(
      &command_buffer->builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE, command_length,
      binding_source_count, uses_indirect_parameters ? 2 : 1,
      queue_kernarg_block_length, &header, &binding_sources));

  uint64_t prepublished_rodata_ordinal = 0;
  if (prepublish_kernargs) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_aql_command_buffer_record_prepublished_dispatch_kernargs(
            command_buffer, kernel_args, layout, config, constants, bindings,
            kernarg_strategy, kernarg_padded_length,
            &prepublished_rodata_ordinal));
    if (IREE_UNLIKELY(prepublished_rodata_ordinal > UINT32_MAX)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "prepublished command-buffer kernarg rodata "
                              "ordinal exceeds uint32_t");
    }
  }

  iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command =
      (iree_hal_amdgpu_command_buffer_dispatch_command_t*)header;
  dispatch_command->kernel_object = kernel_args->kernel_object;
  dispatch_command->binding_source_offset =
      binding_sources
          ? (uint32_t)((uint8_t*)binding_sources -
                       (uint8_t*)command_buffer->builder.current_block)
          : 0;
  dispatch_command->payload_reference =
      prepublish_kernargs
          ? (uint32_t)prepublished_rodata_ordinal
          : sizeof(iree_hal_amdgpu_command_buffer_dispatch_command_t);
  dispatch_command->binding_count = (uint16_t)bindings.count;
  dispatch_command->kernarg_length_qwords = kernarg_length_qwords;
  dispatch_command->tail_length_qwords =
      prepublish_kernargs ? 0 : tail_length_qwords;
  dispatch_command->kernarg_strategy =
      prepublish_kernargs
          ? IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_PREPUBLISHED
          : (uint8_t)kernarg_strategy;
  dispatch_command->dispatch_flags =
      uses_indirect_parameters
          ? IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_INDIRECT_PARAMETERS
          : IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_NONE;
  dispatch_command->setup = kernel_args->setup;
  dispatch_command->export_ordinal = export_ordinal;
  dispatch_command->workgroup_size[0] = kernel_args->workgroup_size[0];
  dispatch_command->workgroup_size[1] = kernel_args->workgroup_size[1];
  dispatch_command->workgroup_size[2] = kernel_args->workgroup_size[2];
  dispatch_command->implicit_args_offset_qwords = implicit_args_offset_qwords;
  dispatch_command->grid_size[0] =
      uses_indirect_parameters
          ? 0
          : config.workgroup_count[0] * kernel_args->workgroup_size[0];
  dispatch_command->grid_size[1] =
      uses_indirect_parameters
          ? 0
          : config.workgroup_count[1] * kernel_args->workgroup_size[1];
  dispatch_command->grid_size[2] =
      uses_indirect_parameters
          ? 0
          : config.workgroup_count[2] * kernel_args->workgroup_size[2];
  dispatch_command->private_segment_size = kernel_args->private_segment_size;
  dispatch_command->group_segment_size =
      kernel_args->group_segment_size + config.dynamic_workgroup_local_memory;

  if (binding_sources && !uses_custom_direct_arguments) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_aql_command_buffer_write_dispatch_binding_sources(
            bindings, binding_sources));
  }
  if (uses_indirect_parameters) {
    iree_hal_amdgpu_command_buffer_binding_source_t* parameter_source =
        binding_sources + (uses_custom_direct_arguments ? 0 : bindings.count);
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_aql_command_buffer_write_indirect_parameter_source(
            command_buffer, config.workgroup_count_ref, parameter_source));
  }
  if (prepublish_kernargs) {
    return iree_ok_status();
  }

  uint8_t* tail_payload =
      (uint8_t*)dispatch_command + dispatch_command->payload_reference;
  return iree_hal_amdgpu_aql_command_buffer_write_dispatch_tail(
      kernel_args, layout, config, constants, kernarg_strategy, tail_payload);
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
