// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"

#include <string.h>

#include "iree/base/alignment.h"
#include "iree/hal/drivers/amdgpu/abi/queue.h"
#include "iree/hal/drivers/amdgpu/aql_command_buffer_profile.h"
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
  IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_STATIC_BUFFER_PAGE_CAPACITY_LOG2 = 9,
  IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_STATIC_BUFFER_PAGE_CAPACITY = 512,
  IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_STATIC_BUFFER_PAGE_MASK =
      IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_STATIC_BUFFER_PAGE_CAPACITY - 1,
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

typedef uint32_t iree_hal_amdgpu_aql_command_buffer_rodata_segment_flags_t;

typedef enum iree_hal_amdgpu_aql_command_buffer_recording_state_e {
  IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RECORDING_STATE_INITIAL = 0,
  IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RECORDING_STATE_RECORDING = 1,
  IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RECORDING_STATE_FINALIZED = 2,
  IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RECORDING_STATE_FAILED = 3,
} iree_hal_amdgpu_aql_command_buffer_recording_state_t;

typedef struct iree_hal_amdgpu_aql_command_buffer_static_buffer_page_t {
  // Next page in ordinal order.
  struct iree_hal_amdgpu_aql_command_buffer_static_buffer_page_t* next;
  // Number of valid entries in |buffers|.
  uint32_t count;
  // Reserved bits for future page metadata.
  uint32_t reserved0;
  // Fixed-capacity direct buffer table.
  iree_hal_buffer_t*
      buffers[IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_STATIC_BUFFER_PAGE_CAPACITY];
} iree_hal_amdgpu_aql_command_buffer_static_buffer_page_t;
static_assert((IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_STATIC_BUFFER_PAGE_CAPACITY &
               IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_STATIC_BUFFER_PAGE_MASK) == 0,
              "static buffer page capacity must be a power-of-two");
static_assert(
    sizeof(iree_hal_amdgpu_aql_command_buffer_static_buffer_page_t) <=
        IREE_HAL_AMDGPU_AQL_PROGRAM_DEFAULT_BLOCK_SIZE,
    "static buffer page should fit in a default command-buffer block");

typedef struct iree_hal_amdgpu_aql_command_buffer_rodata_segment_t {
  // Command-buffer-owned immutable payload bytes.
  uint8_t* data;
  // Prepublished kernarg metadata used only when the segment flag is set.
  struct {
    // Dispatch command whose payload reference is patched during finalization.
    iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command;
  } prepublished;
  // Byte length of |data|.
  uint32_t length;
  // Required alignment for the device pointer when materialized.
  uint32_t alignment;
  // Segment flags from
  // iree_hal_amdgpu_aql_command_buffer_rodata_segment_flag_bits_t.
  iree_hal_amdgpu_aql_command_buffer_rodata_segment_flags_t flags;
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

typedef struct iree_hal_amdgpu_aql_command_buffer_dispatch_summary_block_t {
  // Next block summary in recording order.
  struct iree_hal_amdgpu_aql_command_buffer_dispatch_summary_block_t* next;
  // Recorded command-buffer block this summary describes.
  const iree_hal_amdgpu_command_buffer_block_header_t* header;
  // Retained dispatch summaries for this block.
  struct {
    // First dispatch summary in command order.
    iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t* first;
    // Final dispatch summary in command order.
    iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t* last;
    // Number of dispatch summaries in this block.
    uint32_t count;
  } dispatch;
} iree_hal_amdgpu_aql_command_buffer_dispatch_summary_block_t;

typedef struct iree_hal_amdgpu_aql_command_buffer_t {
  // Base HAL command-buffer resource.
  iree_hal_command_buffer_t base;
  // Host allocator used to allocate the command-buffer object.
  iree_allocator_t host_allocator;
  // Borrowed device allocator used during recording finalization.
  iree_hal_allocator_t* device_allocator;
  // Borrowed block pools used for command-buffer-owned storage.
  struct {
    // Block pool used for durable command-buffer program blocks.
    iree_arena_block_pool_t* program;
    // Block pool used for retained HAL resource sets.
    iree_arena_block_pool_t* resource_set;
  } block_pools;
  // Physical device ordinal selected from the command buffer's queue affinity.
  uint32_t device_ordinal;
  // One-shot lifecycle state enforced even when generic HAL validation is off.
  iree_hal_amdgpu_aql_command_buffer_recording_state_t recording_state;
  // Arena owning recording-lifetime static buffer pages, rodata pages, and
  // rodata payload bytes referenced by finalized program command records.
  iree_arena_allocator_t recording_arena;
  // Resource set retaining direct buffers and executables when not unretained.
  iree_hal_resource_set_t* resource_set;
  // Direct buffer ordinal table captured while recording.
  struct {
    // First static buffer page in ordinal order.
    iree_hal_amdgpu_aql_command_buffer_static_buffer_page_t* first_page;
    // Last static buffer page in ordinal order.
    iree_hal_amdgpu_aql_command_buffer_static_buffer_page_t* current_page;
    // Total direct buffer ordinals assigned.
    uint32_t count;
    // Reserved bytes for stable layout.
    uint32_t reserved0;
  } static_buffers;
  // Device-visible storage containing prepublished static dispatch kernargs.
  struct {
    // Cold-path storage strategy selected during command-buffer creation.
    iree_hal_amdgpu_aql_prepublished_kernarg_storage_t storage;
    // Recording-time materialization plan for immutable kernarg templates.
    struct {
      // Number of prepublished kernarg templates recorded.
      iree_host_size_t count;
      // Minimum byte length required before base-alignment slack.
      iree_host_size_t payload_length;
      // Maximum device pointer alignment required by any template.
      uint32_t max_alignment;
    } templates;
    // Materialized device-visible kernarg template allocation.
    struct {
      // Retained buffer containing all prepublished kernarg templates.
      iree_hal_buffer_t* buffer;
      // Device pointer to the first byte of |buffer|.
      uint8_t* device_base;
      // Allocated byte length of |buffer|.
      iree_device_size_t byte_length;
    } materialized;
  } prepublished_kernargs;
  // Immutable payload ordinal table captured while recording.
  struct {
    // First segment page in ordinal order.
    iree_hal_amdgpu_aql_command_buffer_rodata_segment_page_t* first_page;
    // Last segment page in ordinal order.
    iree_hal_amdgpu_aql_command_buffer_rodata_segment_page_t* current_page;
    // Total segment descriptors assigned ordinals.
    uint32_t segment_count;
  } rodata;
  // Command-buffer profile metadata retained for profiling-enabled recording.
  struct {
    // Borrowed logical-device profiling metadata registry.
    iree_hal_amdgpu_profile_metadata_registry_t* metadata;
    // Producer-local profile command-buffer id, or 0 when profile metadata is
    // not retained for this command buffer.
    uint64_t id;
  } profile;
  // Recording-time sidecars retained for profiling and timestamp planning.
  struct {
    // Block-level dispatch summary list.
    struct {
      // First block with retained dispatch summaries.
      iree_hal_amdgpu_aql_command_buffer_dispatch_summary_block_t* first;
      // Current recording block with retained dispatch summaries.
      iree_hal_amdgpu_aql_command_buffer_dispatch_summary_block_t* current;
    } block;
    // Total retained dispatch summaries across all blocks.
    uint32_t count;
  } dispatch_summaries;
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

static bool iree_hal_amdgpu_aql_command_buffer_retains_profile_metadata(
    const iree_hal_amdgpu_aql_command_buffer_t* command_buffer) {
  return iree_all_bits_set(
      command_buffer->base.mode,
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA);
}

static bool iree_hal_amdgpu_aql_command_buffer_retains_dispatch_summaries(
    const iree_hal_amdgpu_aql_command_buffer_t* command_buffer) {
  return iree_any_bit_set(
      command_buffer->base.mode,
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA |
          IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA);
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

static bool iree_hal_amdgpu_aql_command_buffer_prepublish_enabled(
    const iree_hal_amdgpu_aql_command_buffer_t* command_buffer) {
  return command_buffer->prepublished_kernargs.storage.strategy !=
         IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_STRATEGY_DISABLED;
}

static void iree_hal_amdgpu_aql_command_buffer_reset_resources(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer) {
  iree_hal_resource_set_free(command_buffer->resource_set);
  command_buffer->resource_set = NULL;
  command_buffer->static_buffers.first_page = NULL;
  command_buffer->static_buffers.current_page = NULL;
  command_buffer->static_buffers.count = 0;
  iree_hal_buffer_release(
      command_buffer->prepublished_kernargs.materialized.buffer);
  command_buffer->prepublished_kernargs.templates.count = 0;
  command_buffer->prepublished_kernargs.templates.payload_length = 0;
  command_buffer->prepublished_kernargs.templates.max_alignment = 1;
  command_buffer->prepublished_kernargs.materialized.buffer = NULL;
  command_buffer->prepublished_kernargs.materialized.device_base = NULL;
  command_buffer->prepublished_kernargs.materialized.byte_length = 0;
  iree_arena_reset(&command_buffer->recording_arena);
  command_buffer->rodata.first_page = NULL;
  command_buffer->rodata.current_page = NULL;
  command_buffer->rodata.segment_count = 0;
  command_buffer->dispatch_summaries.block.first = NULL;
  command_buffer->dispatch_summaries.block.current = NULL;
  command_buffer->dispatch_summaries.count = 0;
}

static void iree_hal_amdgpu_aql_command_buffer_discard_recording(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer) {
  iree_hal_amdgpu_aql_program_release(&command_buffer->program);
  iree_hal_amdgpu_aql_program_builder_deinitialize(&command_buffer->builder);
  iree_hal_amdgpu_aql_program_builder_initialize(
      command_buffer->block_pools.program, &command_buffer->builder);
  iree_hal_amdgpu_aql_command_buffer_reset_resources(command_buffer);
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_ensure_resource_set(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer) {
  if (!iree_hal_amdgpu_aql_command_buffer_retains_resources(command_buffer) ||
      command_buffer->resource_set) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_resource_set_allocate(
      command_buffer->block_pools.resource_set, &command_buffer->resource_set);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_hal_buffer_t*
iree_hal_amdgpu_aql_command_buffer_static_buffer_for_ordinal(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer, uint32_t ordinal) {
  if (IREE_UNLIKELY(ordinal >= command_buffer->static_buffers.count)) {
    return NULL;
  }

  uint32_t page_ordinal =
      ordinal >>
      IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_STATIC_BUFFER_PAGE_CAPACITY_LOG2;
  const uint32_t buffer_ordinal =
      ordinal & IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_STATIC_BUFFER_PAGE_MASK;
  iree_hal_amdgpu_aql_command_buffer_static_buffer_page_t* page =
      command_buffer->static_buffers.first_page;
  while (page_ordinal > 0 && page) {
    page = page->next;
    --page_ordinal;
  }
  if (IREE_UNLIKELY(!page || buffer_ordinal >= page->count)) return NULL;
  return page->buffers[buffer_ordinal];
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_ensure_static_buffer_page(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    iree_hal_amdgpu_aql_command_buffer_static_buffer_page_t** out_page) {
  *out_page = NULL;
  if (IREE_UNLIKELY(command_buffer->static_buffers.count == UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command-buffer static buffer table overflow");
  }
  iree_hal_amdgpu_aql_command_buffer_static_buffer_page_t* page =
      command_buffer->static_buffers.current_page;
  if (page &&
      page->count <
          IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_STATIC_BUFFER_PAGE_CAPACITY) {
    *out_page = page;
    return iree_ok_status();
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, command_buffer->static_buffers.count);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->recording_arena, sizeof(*page),
                              (void**)&page));
  memset(page, 0, sizeof(*page));
  if (command_buffer->static_buffers.current_page) {
    command_buffer->static_buffers.current_page->next = page;
  } else {
    command_buffer->static_buffers.first_page = page;
  }
  command_buffer->static_buffers.current_page = page;
  *out_page = page;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_allocate_static_buffer(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    iree_hal_buffer_t* buffer, uint32_t* out_ordinal) {
  *out_ordinal = 0;
  iree_hal_amdgpu_aql_command_buffer_static_buffer_page_t* page = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_ensure_static_buffer_page(
          command_buffer, &page));
  *out_ordinal = command_buffer->static_buffers.count;
  ++command_buffer->static_buffers.count;
  page->buffers[page->count++] = buffer;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_record_static_buffer(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    iree_hal_buffer_t* buffer, uint32_t* out_ordinal) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_ensure_resource_set(command_buffer));
  if (command_buffer->resource_set) {
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
        command_buffer->resource_set, /*count=*/1, &buffer));
  }
  return iree_hal_amdgpu_aql_command_buffer_allocate_static_buffer(
      command_buffer, buffer, out_ordinal);
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
    iree_host_size_t byte_length, uint32_t alignment,
    iree_hal_amdgpu_aql_command_buffer_rodata_segment_flags_t flags,
    uint64_t* out_rodata_ordinal,
    iree_hal_amdgpu_aql_command_buffer_rodata_segment_t** out_segment) {
  *out_rodata_ordinal = 0;
  if (out_segment) *out_segment = NULL;
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
        z0, iree_arena_allocate(&command_buffer->recording_arena, sizeof(*page),
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
  iree_hal_amdgpu_aql_command_buffer_rodata_segment_t* segment =
      &page->segments[page->count++];
  *segment = (iree_hal_amdgpu_aql_command_buffer_rodata_segment_t){
      .data = data,
      .length = (uint32_t)byte_length,
      .alignment = alignment,
      .flags = flags,
  };
  *out_rodata_ordinal = ordinal;
  if (out_segment) *out_segment = segment;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_allocate_rodata_segment(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    iree_host_size_t byte_length, uint32_t alignment,
    iree_hal_amdgpu_aql_command_buffer_rodata_segment_flags_t flags,
    uint8_t** out_data, uint64_t* out_rodata_ordinal,
    iree_hal_amdgpu_aql_command_buffer_rodata_segment_t** out_segment) {
  *out_data = NULL;
  *out_rodata_ordinal = 0;
  if (out_segment) *out_segment = NULL;
  uint8_t* rodata = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_aligned(
      &command_buffer->recording_arena,
      iree_max((iree_host_size_t)1, byte_length), alignment, (void**)&rodata));
  iree_status_t status =
      iree_hal_amdgpu_aql_command_buffer_append_rodata_segment(
          command_buffer, rodata, byte_length, alignment, flags,
          out_rodata_ordinal, out_segment);
  if (iree_status_is_ok(status)) {
    *out_data = rodata;
  }
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
          out_rodata_ordinal, /*out_segment=*/NULL));
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
iree_hal_amdgpu_aql_command_buffer_append_prepublished_kernarg_template(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_aql_command_buffer_rodata_segment_t* segment) {
  iree_host_size_t aligned_payload_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_align(
          command_buffer->prepublished_kernargs.templates.payload_length,
          segment->alignment, &aligned_payload_length))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "prepublished command-buffer kernarg offset overflow");
  }
  command_buffer->prepublished_kernargs.templates.payload_length =
      aligned_payload_length;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          command_buffer->prepublished_kernargs.templates.payload_length,
          iree_max((iree_host_size_t)1, (iree_host_size_t)segment->length),
          &command_buffer->prepublished_kernargs.templates.payload_length))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "prepublished command-buffer kernarg storage overflow");
  }
  command_buffer->prepublished_kernargs.templates.max_alignment =
      iree_max(command_buffer->prepublished_kernargs.templates.max_alignment,
               segment->alignment);
  ++command_buffer->prepublished_kernargs.templates.count;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_copy_prepublished_kernargs(
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
      if (IREE_UNLIKELY(payload_offset > UINT32_MAX)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "prepublished command-buffer kernarg offset exceeds uint32_t");
      }
      if (IREE_UNLIKELY(!segment->prepublished.dispatch_command)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "prepublished command-buffer kernarg has no dispatch command");
      }
      if (segment->length > 0) {
        memcpy(host_base + payload_offset, segment->data, segment->length);
      }
      segment->prepublished.dispatch_command->payload_reference =
          (uint32_t)payload_offset;
      payload_offset +=
          iree_max((iree_host_size_t)1, (iree_host_size_t)segment->length);
    }
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_verify_prepublished_kernarg_storage(
    const iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    iree_hal_memory_type_t required_type, iree_hal_buffer_t* buffer) {
  const iree_hal_memory_type_t actual_type =
      iree_hal_buffer_memory_type(buffer);
  if (IREE_LIKELY(iree_all_bits_set(actual_type, required_type))) {
    return iree_ok_status();
  }
#if IREE_STATUS_MODE
  iree_bitfield_string_temp_t required_temp;
  iree_bitfield_string_temp_t actual_temp;
  const iree_string_view_t required_string =
      iree_hal_memory_type_format(required_type, &required_temp);
  const iree_string_view_t actual_string =
      iree_hal_memory_type_format(actual_type, &actual_temp);
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "prepublished command-buffer kernarg strategy %u requires "
      "memory_type=%.*s but allocation returned memory_type=%.*s",
      command_buffer->prepublished_kernargs.storage.strategy,
      (int)required_string.size, required_string.data, (int)actual_string.size,
      actual_string.data);
#else
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "prepublished command-buffer kernarg allocation returned incompatible "
      "memory type");
#endif  // IREE_STATUS_MODE
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_materialize_prepublished_kernargs(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer) {
  const iree_host_size_t template_count =
      command_buffer->prepublished_kernargs.templates.count;
  if (template_count == 0) {
    return iree_ok_status();
  }
  const iree_host_size_t payload_length =
      command_buffer->prepublished_kernargs.templates.payload_length;
  const uint32_t max_alignment =
      command_buffer->prepublished_kernargs.templates.max_alignment;
  if (IREE_UNLIKELY(!iree_hal_amdgpu_aql_command_buffer_prepublish_enabled(
          command_buffer))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "command buffer recorded prepublished kernargs without a storage "
        "strategy");
  }
  if (IREE_UNLIKELY(payload_length > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "prepublished command-buffer kernarg payload length %" PRIhsz
        " exceeds uint32_t reference max",
        payload_length);
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

  iree_hal_buffer_params_t params =
      command_buffer->prepublished_kernargs.storage.buffer_params;
  params.queue_affinity = command_buffer->base.queue_affinity;

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
    status =
        iree_hal_amdgpu_aql_command_buffer_verify_prepublished_kernarg_storage(
            command_buffer, params.type, template_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_range(
        template_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, /*byte_offset=*/0,
        (iree_device_size_t)allocation_length, &mapping);
  }
  if (iree_status_is_ok(status)) {
    memset(mapping.contents.data, 0, allocation_length);
    status = iree_hal_amdgpu_aql_command_buffer_copy_prepublished_kernargs(
        command_buffer, &mapping, device_base);
  }
  if (mapping.buffer) {
    status = iree_status_join(status, iree_hal_buffer_unmap_range(&mapping));
  }
  if (iree_status_is_ok(status)) {
    command_buffer->prepublished_kernargs.materialized.buffer = template_buffer;
    command_buffer->prepublished_kernargs.materialized.device_base =
        device_base;
    command_buffer->prepublished_kernargs.materialized.byte_length =
        (iree_device_size_t)allocation_length;
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
    iree_host_size_t device_ordinal,
    iree_hal_amdgpu_aql_prepublished_kernarg_storage_t
        prepublished_kernarg_storage,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    iree_arena_block_pool_t* program_block_pool,
    iree_arena_block_pool_t* resource_set_block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (iree_any_bit_set(mode,
                       IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION) &&
      !iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ALLOW_INLINE_EXECUTION requires ONE_SHOT mode");
  }
  if (IREE_UNLIKELY(!program_block_pool)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command-buffer program block pool is required");
  }
  if (IREE_UNLIKELY(!resource_set_block_pool)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command-buffer resource set block pool is "
                            "required");
  }
  const bool retain_profile_metadata = iree_all_bits_set(
      mode, IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA);
  if (IREE_UNLIKELY(retain_profile_metadata && !profile_metadata)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command-buffer profile metadata is required");
  }
  if (IREE_UNLIKELY(device_ordinal > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command-buffer device ordinal %" PRIhsz
                            " exceeds uint32_t storage",
                            device_ordinal);
  }
  switch (prepublished_kernarg_storage.strategy) {
    case IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_STRATEGY_DISABLED:
    case IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_STRATEGY_DEVICE_FINE_HOST_COHERENT:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unsupported prepublished command-buffer kernarg storage strategy %u",
          prepublished_kernarg_storage.strategy);
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
  command_buffer->block_pools.program = program_block_pool;
  command_buffer->block_pools.resource_set = resource_set_block_pool;
  command_buffer->profile.metadata = profile_metadata;
  command_buffer->device_ordinal = (uint32_t)device_ordinal;
  command_buffer->prepublished_kernargs.storage = prepublished_kernarg_storage;
  command_buffer->prepublished_kernargs.templates.max_alignment = 1;
  iree_arena_initialize(program_block_pool, &command_buffer->recording_arena);
  iree_hal_amdgpu_aql_program_builder_initialize(program_block_pool,
                                                 &command_buffer->builder);

  iree_status_t status = iree_ok_status();
  if (retain_profile_metadata) {
    status = iree_hal_amdgpu_profile_metadata_register_command_buffer(
        profile_metadata, mode, command_categories, queue_affinity,
        device_ordinal, &command_buffer->profile.id);
  }
  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_amdgpu_aql_command_buffer_destroy(&command_buffer->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
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
  iree_arena_deinitialize(&command_buffer->recording_arena);
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

uint64_t iree_hal_amdgpu_aql_command_buffer_profile_id(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  return command_buffer->profile.id;
}

const iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t*
iree_hal_amdgpu_aql_command_buffer_dispatch_summaries(
    iree_hal_command_buffer_t* base_command_buffer,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    uint32_t* out_count) {
  IREE_ASSERT_ARGUMENT(out_count);
  *out_count = 0;
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  for (const iree_hal_amdgpu_aql_command_buffer_dispatch_summary_block_t*
           summary_block = command_buffer->dispatch_summaries.block.first;
       summary_block; summary_block = summary_block->next) {
    if (summary_block->header == block) {
      *out_count = summary_block->dispatch.count;
      return summary_block->dispatch.first;
    }
    if (summary_block->header->block_ordinal > block->block_ordinal) {
      break;
    }
  }
  return NULL;
}

iree_hal_buffer_t* iree_hal_amdgpu_aql_command_buffer_static_buffer(
    iree_hal_command_buffer_t* base_command_buffer, uint32_t ordinal) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  return iree_hal_amdgpu_aql_command_buffer_static_buffer_for_ordinal(
      command_buffer, ordinal);
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
    iree_hal_command_buffer_t* base_command_buffer, uint32_t byte_offset,
    uint32_t length) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  if (IREE_UNLIKELY(
          !command_buffer->prepublished_kernargs.materialized.buffer)) {
    return NULL;
  }
  const iree_device_size_t required_length =
      iree_max((iree_device_size_t)1, (iree_device_size_t)length);
  iree_device_size_t end_offset = 0;
  if (IREE_UNLIKELY(
          !iree_device_size_checked_add((iree_device_size_t)byte_offset,
                                        required_length, &end_offset) ||
          end_offset >
              command_buffer->prepublished_kernargs.materialized.byte_length)) {
    return NULL;
  }
  return command_buffer->prepublished_kernargs.materialized.device_base +
         byte_offset;
}

//===----------------------------------------------------------------------===//
// Recording Session
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_aql_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  switch (command_buffer->recording_state) {
    case IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RECORDING_STATE_INITIAL:
      break;
    case IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RECORDING_STATE_RECORDING:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "command buffer is already in a recording state");
    case IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RECORDING_STATE_FINALIZED:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "command buffer has already been recorded; "
                              "re-recording command buffers is not allowed");
    case IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RECORDING_STATE_FAILED:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "command buffer recording failed and cannot be reused");
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "invalid command-buffer recording state %d",
                              (int)command_buffer->recording_state);
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_program_builder_begin(&command_buffer->builder));
  command_buffer->recording_state =
      IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RECORDING_STATE_RECORDING;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  if (IREE_UNLIKELY(
          command_buffer->recording_state !=
          IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RECORDING_STATE_RECORDING)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffer is not in a recording state");
  }
  iree_status_t status = iree_hal_amdgpu_aql_program_builder_end(
      &command_buffer->builder, &command_buffer->program);
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_aql_command_buffer_materialize_prepublished_kernargs(
            command_buffer);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_aql_command_buffer_retains_profile_metadata(
          command_buffer)) {
    status = iree_hal_amdgpu_aql_command_buffer_register_profile_operations(
        command_buffer->profile.metadata, command_buffer->profile.id,
        &command_buffer->program, command_buffer->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_resource_set_freeze(command_buffer->resource_set);
    command_buffer->recording_state =
        IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RECORDING_STATE_FINALIZED;
  } else {
    iree_hal_amdgpu_aql_command_buffer_discard_recording(command_buffer);
    command_buffer->recording_state =
        IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RECORDING_STATE_FAILED;
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

static iree_hsa_fence_scope_t
iree_hal_amdgpu_aql_command_buffer_access_scope_fence_scope(
    iree_hal_access_scope_t access_scope) {
  if (access_scope == 0) return IREE_HSA_FENCE_SCOPE_NONE;
  // Resolve HAL memory visibility to HSA fence scope while recording so replay
  // can consume compact command flags without re-inspecting barrier operands.
  // Same-agent device producer/consumer edges use AGENT; host/system-visible
  // edges use SYSTEM; execution-only barriers carry no acquire/release scope.
  const iree_hal_access_scope_t system_scopes =
      IREE_HAL_ACCESS_SCOPE_HOST_READ | IREE_HAL_ACCESS_SCOPE_HOST_WRITE |
      IREE_HAL_ACCESS_SCOPE_MEMORY_READ | IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE;
  return iree_any_bit_set(access_scope, system_scopes)
             ? IREE_HSA_FENCE_SCOPE_SYSTEM
             : IREE_HSA_FENCE_SCOPE_AGENT;
}

static void iree_hal_amdgpu_aql_command_buffer_accumulate_barrier_scopes(
    iree_hal_access_scope_t source_scope, iree_hal_access_scope_t target_scope,
    iree_hsa_fence_scope_t* release_scope,
    iree_hsa_fence_scope_t* acquire_scope) {
  const iree_hsa_fence_scope_t source_fence_scope =
      iree_hal_amdgpu_aql_command_buffer_access_scope_fence_scope(source_scope);
  const iree_hsa_fence_scope_t target_fence_scope =
      iree_hal_amdgpu_aql_command_buffer_access_scope_fence_scope(target_scope);
  const iree_hsa_fence_scope_t fence_scope =
      source_fence_scope > target_fence_scope ? source_fence_scope
                                              : target_fence_scope;
  if (source_scope != 0 && fence_scope > *release_scope) {
    *release_scope = fence_scope;
  }
  if (target_scope != 0 && fence_scope > *acquire_scope) {
    *acquire_scope = fence_scope;
  }
}

static void iree_hal_amdgpu_aql_command_buffer_resolve_barrier_scopes(
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers,
    iree_hsa_fence_scope_t* out_acquire_scope,
    iree_hsa_fence_scope_t* out_release_scope) {
  iree_hsa_fence_scope_t acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
  iree_hsa_fence_scope_t release_scope = IREE_HSA_FENCE_SCOPE_NONE;
  if (iree_any_bit_set(source_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    acquire_scope = IREE_HSA_FENCE_SCOPE_SYSTEM;
  }
  if (iree_any_bit_set(target_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    release_scope = IREE_HSA_FENCE_SCOPE_SYSTEM;
  }
  for (iree_host_size_t i = 0; i < memory_barrier_count; ++i) {
    iree_hal_amdgpu_aql_command_buffer_accumulate_barrier_scopes(
        memory_barriers[i].source_scope, memory_barriers[i].target_scope,
        &release_scope, &acquire_scope);
  }
  for (iree_host_size_t i = 0; i < buffer_barrier_count; ++i) {
    iree_hal_amdgpu_aql_command_buffer_accumulate_barrier_scopes(
        buffer_barriers[i].source_scope, buffer_barriers[i].target_scope,
        &release_scope, &acquire_scope);
  }
  *out_acquire_scope = acquire_scope;
  *out_release_scope = release_scope;
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  if (IREE_UNLIKELY(flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported execution barrier flags");
  }

  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);

  iree_hsa_fence_scope_t acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
  iree_hsa_fence_scope_t release_scope = IREE_HSA_FENCE_SCOPE_NONE;
  iree_hal_amdgpu_aql_command_buffer_resolve_barrier_scopes(
      source_stage_mask, target_stage_mask, memory_barrier_count,
      memory_barriers, buffer_barrier_count, buffer_barriers, &acquire_scope,
      &release_scope);

  iree_hal_amdgpu_command_buffer_command_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_builder_append_command(
      &command_buffer->builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_barrier_command_t),
      /*binding_source_count=*/0, /*aql_packet_count=*/0,
      /*kernarg_length=*/0, &header, /*out_binding_sources=*/NULL));

  iree_hal_amdgpu_command_buffer_barrier_command_t* barrier =
      (iree_hal_amdgpu_command_buffer_barrier_command_t*)header;
  barrier->acquire_scope = (uint8_t)acquire_scope;
  barrier->release_scope = (uint8_t)release_scope;
  barrier->barrier_flags = (uint16_t)flags;
  iree_hal_amdgpu_aql_program_builder_set_pending_barrier_scopes(
      &command_buffer->builder, barrier->acquire_scope, barrier->release_scope);
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

static bool
iree_hal_amdgpu_aql_command_buffer_allows_staged_transient_buffer_refs(
    const iree_hal_amdgpu_aql_command_buffer_t* command_buffer) {
  // One-shot command buffers are recorded for a single queued execution and may
  // capture transient backing staged by a preceding queue_alloca before the
  // user-visible alloca signal is published. Reusable command buffers require
  // committed backing because the captured pointer can be replayed later.
  return iree_all_bits_set(command_buffer->base.mode,
                           IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT);
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_resolve_static_buffer_ref(
    const iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_buffer_ref_t* buffer_ref, uint64_t* out_device_pointer) {
  *out_device_pointer = 0;
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer_ref->buffer);
  if (iree_hal_amdgpu_transient_buffer_isa(allocated_buffer)) {
    iree_hal_buffer_t* backing_buffer = NULL;
    if (iree_hal_amdgpu_aql_command_buffer_allows_staged_transient_buffer_refs(
            command_buffer)) {
      backing_buffer =
          iree_hal_amdgpu_transient_buffer_backing_buffer(allocated_buffer);
      if (IREE_UNLIKELY(!backing_buffer)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "one-shot command-buffer buffer reference has no staged AMDGPU "
            "backing");
      }
    } else {
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_transient_buffer_resolve_committed_backing(
              allocated_buffer, &backing_buffer));
    }
    allocated_buffer = iree_hal_buffer_allocated_buffer(backing_buffer);
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
      IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION |
      IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES;
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

static bool iree_hal_amdgpu_aql_command_buffer_should_defer_static_buffer_ref(
    const iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_buffer_ref_t* buffer_ref) {
  if (!iree_hal_amdgpu_aql_command_buffer_allows_staged_transient_buffer_refs(
          command_buffer)) {
    return false;
  }
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer_ref->buffer);
  return iree_hal_amdgpu_transient_buffer_isa(allocated_buffer) &&
         !iree_hal_amdgpu_transient_buffer_backing_buffer(allocated_buffer);
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

    iree_device_size_t unused_offset = 0;
    iree_device_size_t unused_length = 0;
    status = iree_hal_buffer_calculate_range(
        /*base_offset=*/0, iree_hal_buffer_byte_length(binding->buffer),
        binding->offset, binding->length, &unused_offset, &unused_length);
    if (iree_status_is_ok(status) &&
        !iree_hal_amdgpu_aql_command_buffer_should_defer_static_buffer_ref(
            command_buffer, binding)) {
      uint64_t unused_device_pointer = 0;
      status = iree_hal_amdgpu_aql_command_buffer_resolve_static_buffer_ref(
          command_buffer, binding, &unused_device_pointer);
    }
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
iree_hal_amdgpu_aql_command_buffer_record_deferred_dispatch_binding_source(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_buffer_ref_t* binding,
    iree_hal_amdgpu_command_buffer_binding_source_t* binding_source) {
  iree_device_size_t resolved_offset = 0;
  iree_device_size_t resolved_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      /*base_offset=*/0, iree_hal_buffer_byte_length(binding->buffer),
      binding->offset, binding->length, &resolved_offset, &resolved_length));
  (void)resolved_length;

  uint32_t static_buffer_ordinal = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_allocate_static_buffer(
          command_buffer, binding->buffer, &static_buffer_ordinal));
  binding_source->offset_or_pointer = resolved_offset;
  binding_source->slot = static_buffer_ordinal;
  binding_source->flags =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_STATIC_BUFFER;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_write_dispatch_binding_sources(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < bindings.count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_buffer_ref_t* binding = &bindings.values[i];
    iree_hal_amdgpu_command_buffer_binding_source_t* binding_source =
        &binding_sources[i];
    binding_source->target_binding_ordinal = (uint16_t)i;
    if (!binding->buffer) {
      binding_source->offset_or_pointer = binding->offset;
      binding_source->slot = binding->buffer_slot;
      binding_source->flags =
          IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC;
      continue;
    }

    if (iree_hal_amdgpu_aql_command_buffer_should_defer_static_buffer_ref(
            command_buffer, binding)) {
      status =
          iree_hal_amdgpu_aql_command_buffer_record_deferred_dispatch_binding_source(
              command_buffer, binding, binding_source);
      continue;
    }

    status = iree_hal_amdgpu_aql_command_buffer_resolve_static_buffer_ref(
        command_buffer, binding, &binding_source->offset_or_pointer);
    if (iree_status_is_ok(status)) {
      binding_source->slot = 0;
      binding_source->flags =
          IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_NONE;
    }
  }
  return status;
}

static void
iree_hal_amdgpu_aql_command_buffer_write_dynamic_dispatch_binding_sources(
    iree_hal_buffer_ref_list_t bindings,
    iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources) {
  iree_host_size_t source_index = 0;
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    const iree_hal_buffer_ref_t* binding = &bindings.values[i];
    if (binding->buffer) continue;

    iree_hal_amdgpu_command_buffer_binding_source_t* binding_source =
        &binding_sources[source_index++];
    binding_source->offset_or_pointer = binding->offset;
    binding_source->slot = binding->buffer_slot;
    binding_source->flags =
        IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC;
    binding_source->target_binding_ordinal = (uint16_t)i;
  }
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

  if (iree_hal_amdgpu_aql_command_buffer_should_defer_static_buffer_ref(
          command_buffer, &buffer_ref)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_aql_command_buffer_record_deferred_dispatch_binding_source(
            command_buffer, &buffer_ref, binding_source));
    binding_source->flags |=
        IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_resolve_static_buffer_ref(
          command_buffer, &buffer_ref, &binding_source->offset_or_pointer));
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
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_HAL:
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_DYNAMIC_BINDINGS: {
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

static uint16_t
iree_hal_amdgpu_aql_command_buffer_count_dynamic_dispatch_bindings(
    iree_hal_buffer_ref_list_t bindings) {
  uint16_t count = 0;
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    if (!bindings.values[i].buffer) ++count;
  }
  return count;
}

typedef uint32_t iree_hal_amdgpu_aql_command_buffer_kernarg_template_flags_t;
enum iree_hal_amdgpu_aql_command_buffer_kernarg_template_flag_bits_t {
  IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_KERNARG_TEMPLATE_FLAG_NONE = 0u,
  IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_KERNARG_TEMPLATE_FLAG_ALLOW_DYNAMIC_BINDINGS =
      1u << 0,
};

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_write_dispatch_kernarg_template(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* layout,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings,
    iree_hal_amdgpu_command_buffer_kernarg_strategy_t kernarg_strategy,
    iree_hal_amdgpu_aql_command_buffer_kernarg_template_flags_t flags,
    uint8_t* kernarg_data) {
  switch (kernarg_strategy) {
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_HAL: {
      uint64_t* binding_dst = (uint64_t*)kernarg_data;
      for (iree_host_size_t i = 0; i < bindings.count; ++i) {
        if (!bindings.values[i].buffer) {
          if (IREE_UNLIKELY(!iree_any_bit_set(
                  flags,
                  IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_KERNARG_TEMPLATE_FLAG_ALLOW_DYNAMIC_BINDINGS))) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "prepublished command-buffer kernarg template cannot contain "
                "dynamic bindings");
          }
          binding_dst[i] = 0;
          continue;
        }
        IREE_RETURN_IF_ERROR(
            iree_hal_amdgpu_aql_command_buffer_resolve_static_buffer_ref(
                command_buffer, &bindings.values[i], &binding_dst[i]));
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
          "unsupported command-buffer kernarg template strategy %u",
          kernarg_strategy);
  }
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_record_prepublished_dispatch_kernargs(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* layout,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings,
    iree_hal_amdgpu_command_buffer_kernarg_strategy_t kernarg_strategy,
    iree_host_size_t kernarg_padded_length, uint64_t* out_rodata_ordinal) {
  *out_rodata_ordinal = 0;
  uint8_t* kernarg_data = NULL;
  iree_hal_amdgpu_aql_command_buffer_rodata_segment_t* segment = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_allocate_rodata_segment(
      command_buffer, kernarg_padded_length, kernel_args->kernarg_alignment,
      IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_FLAG_PREPUBLISHED_KERNARGS,
      &kernarg_data, out_rodata_ordinal, &segment));
  segment->prepublished.dispatch_command = dispatch_command;
  memset(kernarg_data, 0, iree_max((iree_host_size_t)1, kernarg_padded_length));

  iree_status_t status =
      iree_hal_amdgpu_aql_command_buffer_write_dispatch_kernarg_template(
          command_buffer, kernel_args, layout, config, constants, bindings,
          kernarg_strategy,
          IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_KERNARG_TEMPLATE_FLAG_NONE,
          kernarg_data);
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_aql_command_buffer_append_prepublished_kernarg_template(
            command_buffer, segment);
  }
  return status;
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_record_patched_dispatch_kernargs(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* layout,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings,
    iree_hal_amdgpu_command_buffer_kernarg_strategy_t kernarg_strategy,
    iree_host_size_t kernarg_padded_length, uint64_t* out_rodata_ordinal) {
  *out_rodata_ordinal = 0;
  uint8_t* kernarg_data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_allocate_rodata_segment(
          command_buffer, kernarg_padded_length, kernel_args->kernarg_alignment,
          IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_RODATA_SEGMENT_FLAG_NONE,
          &kernarg_data, out_rodata_ordinal, /*out_segment=*/NULL));
  memset(kernarg_data, 0, iree_max((iree_host_size_t)1, kernarg_padded_length));
  return iree_hal_amdgpu_aql_command_buffer_write_dispatch_kernarg_template(
      command_buffer, kernel_args, layout, config, constants, bindings,
      kernarg_strategy,
      IREE_HAL_AMDGPU_AQL_COMMAND_BUFFER_KERNARG_TEMPLATE_FLAG_ALLOW_DYNAMIC_BINDINGS,
      kernarg_data);
}

typedef enum iree_hal_amdgpu_aql_dispatch_plan_flag_bits_e {
  IREE_HAL_AMDGPU_AQL_DISPATCH_PLAN_FLAG_NONE = 0u,
  IREE_HAL_AMDGPU_AQL_DISPATCH_PLAN_FLAG_CUSTOM_DIRECT_ARGUMENTS = 1u << 0,
  IREE_HAL_AMDGPU_AQL_DISPATCH_PLAN_FLAG_INDIRECT_PARAMETERS = 1u << 1,
  IREE_HAL_AMDGPU_AQL_DISPATCH_PLAN_FLAG_DYNAMIC_BINDINGS = 1u << 2,
} iree_hal_amdgpu_aql_dispatch_plan_flag_bits_t;

typedef uint32_t iree_hal_amdgpu_aql_dispatch_plan_flags_t;

typedef enum iree_hal_amdgpu_aql_dispatch_layout_flag_bits_e {
  IREE_HAL_AMDGPU_AQL_DISPATCH_LAYOUT_FLAG_NONE = 0u,
  IREE_HAL_AMDGPU_AQL_DISPATCH_LAYOUT_FLAG_PREPUBLISH_KERNARGS = 1u << 0,
  IREE_HAL_AMDGPU_AQL_DISPATCH_LAYOUT_FLAG_PATCH_KERNARG_TEMPLATE = 1u << 1,
} iree_hal_amdgpu_aql_dispatch_layout_flag_bits_t;

typedef uint32_t iree_hal_amdgpu_aql_dispatch_layout_flags_t;

typedef struct iree_hal_amdgpu_aql_dispatch_inputs_t {
  // Executable containing the requested export.
  iree_hal_executable_t* executable;

  // Export ordinal within |executable|.
  iree_hal_executable_export_ordinal_t export_ordinal;

  // HAL dispatch configuration.
  iree_hal_dispatch_config_t config;

  // Borrowed constant bytes passed to the dispatch.
  iree_const_byte_span_t constants;

  // Borrowed binding list passed to the dispatch.
  iree_hal_buffer_ref_list_t bindings;

  // HAL dispatch flags.
  iree_hal_dispatch_flags_t flags;
} iree_hal_amdgpu_aql_dispatch_inputs_t;

typedef struct iree_hal_amdgpu_aql_dispatch_plan_t {
  // Descriptor resolved for the selected physical device/export pair.
  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor;

  // Workgroup-size override storage used when |kernel_args| points here.
  iree_hal_amdgpu_device_kernel_args_t override_kernel_args;

  // Kernel argument descriptor selected from |descriptor|.
  const iree_hal_amdgpu_device_kernel_args_t* kernel_args;

  // Kernarg layout selected for HAL or custom-direct arguments.
  const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* layout;

  // Number of kernarg blocks required by the selected descriptor path.
  uint32_t kernarg_block_count;

  // Binding-source plan for this dispatch.
  struct {
    // Number of dynamic binding table sources used by this dispatch.
    uint16_t dynamic_count;
  } bindings;

  // Command-buffer kernarg strategy used for this dispatch.
  iree_hal_amdgpu_command_buffer_kernarg_strategy_t kernarg_strategy;

  // Plan flags from iree_hal_amdgpu_aql_dispatch_plan_flag_bits_t.
  iree_hal_amdgpu_aql_dispatch_plan_flags_t flags;
} iree_hal_amdgpu_aql_dispatch_plan_t;

static bool iree_hal_amdgpu_aql_dispatch_plan_uses_custom_direct_arguments(
    const iree_hal_amdgpu_aql_dispatch_plan_t* plan) {
  return iree_any_bit_set(
      plan->flags,
      IREE_HAL_AMDGPU_AQL_DISPATCH_PLAN_FLAG_CUSTOM_DIRECT_ARGUMENTS);
}

static bool iree_hal_amdgpu_aql_dispatch_plan_uses_indirect_parameters(
    const iree_hal_amdgpu_aql_dispatch_plan_t* plan) {
  return iree_any_bit_set(
      plan->flags, IREE_HAL_AMDGPU_AQL_DISPATCH_PLAN_FLAG_INDIRECT_PARAMETERS);
}

static bool iree_hal_amdgpu_aql_dispatch_plan_has_dynamic_bindings(
    const iree_hal_amdgpu_aql_dispatch_plan_t* plan) {
  return iree_any_bit_set(
      plan->flags, IREE_HAL_AMDGPU_AQL_DISPATCH_PLAN_FLAG_DYNAMIC_BINDINGS);
}

typedef struct iree_hal_amdgpu_aql_dispatch_layout_t {
  // Command record and binding-source layout in the AQL program.
  struct {
    // Byte length of the command record allocation.
    iree_host_size_t byte_length;

    // Number of binding-source records following the command.
    uint16_t binding_source_count;

    // Worst-case AQL packets required by replay.
    uint32_t aql_packet_count;
  } command;

  // Queue-time kernarg allocation requirements.
  struct {
    // Total kernarg qword length.
    uint16_t total_length_qwords;

    // Tail payload qword length stored after the command record.
    uint16_t tail_length_qwords;

    // Implicit-argument qword offset, or UINT16_MAX when absent.
    uint16_t implicit_args_offset_qwords;

    // Total kernarg length padded to qword alignment.
    iree_host_size_t total_padded_length;

    // Tail payload length padded to qword alignment.
    iree_host_size_t tail_padded_length;

    // Queue-time kernarg block bytes reserved by replay.
    uint32_t queue_block_length;
  } kernarg;

  // Layout flags from iree_hal_amdgpu_aql_dispatch_layout_flag_bits_t.
  iree_hal_amdgpu_aql_dispatch_layout_flags_t flags;
} iree_hal_amdgpu_aql_dispatch_layout_t;

static bool iree_hal_amdgpu_aql_dispatch_layout_prepublishes_kernargs(
    const iree_hal_amdgpu_aql_dispatch_layout_t* layout) {
  return iree_any_bit_set(
      layout->flags,
      IREE_HAL_AMDGPU_AQL_DISPATCH_LAYOUT_FLAG_PREPUBLISH_KERNARGS);
}

static bool iree_hal_amdgpu_aql_dispatch_layout_patches_kernarg_template(
    const iree_hal_amdgpu_aql_dispatch_layout_t* layout) {
  return iree_any_bit_set(
      layout->flags,
      IREE_HAL_AMDGPU_AQL_DISPATCH_LAYOUT_FLAG_PATCH_KERNARG_TEMPLATE);
}

static bool iree_hal_amdgpu_aql_dispatch_layout_uses_kernarg_template(
    const iree_hal_amdgpu_aql_dispatch_layout_t* layout) {
  return iree_any_bit_set(
      layout->flags,
      IREE_HAL_AMDGPU_AQL_DISPATCH_LAYOUT_FLAG_PREPUBLISH_KERNARGS |
          IREE_HAL_AMDGPU_AQL_DISPATCH_LAYOUT_FLAG_PATCH_KERNARG_TEMPLATE);
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_prepare_dispatch_plan(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_aql_dispatch_inputs_t* inputs, bool validates,
    iree_hal_amdgpu_aql_dispatch_plan_t* out_plan) {
  *out_plan = (iree_hal_amdgpu_aql_dispatch_plan_t){0};
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_check_dispatch_flags(inputs->flags));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_device(
          inputs->executable, inputs->export_ordinal,
          command_buffer->device_ordinal, &out_plan->descriptor));

  if (validates) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_aql_command_buffer_validate_dispatch_shape(
            out_plan->descriptor, inputs->config, inputs->flags));
  }

  iree_hal_amdgpu_aql_command_buffer_select_dispatch_kernel_args(
      out_plan->descriptor, inputs->config, &out_plan->override_kernel_args,
      &out_plan->kernel_args);

  if (iree_any_bit_set(inputs->flags,
                       IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS)) {
    out_plan->flags |=
        IREE_HAL_AMDGPU_AQL_DISPATCH_PLAN_FLAG_CUSTOM_DIRECT_ARGUMENTS;
  }
  if (iree_hal_dispatch_uses_indirect_parameters(inputs->flags)) {
    out_plan->flags |=
        IREE_HAL_AMDGPU_AQL_DISPATCH_PLAN_FLAG_INDIRECT_PARAMETERS;
  }
  out_plan->kernarg_strategy =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_HAL;
  if (IREE_UNLIKELY(inputs->constants.data_length > 0 &&
                    !inputs->constants.data)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch constant data must be non-null when length is non-zero");
  }

  if (iree_hal_amdgpu_aql_dispatch_plan_uses_custom_direct_arguments(
          out_plan)) {
    if (IREE_UNLIKELY(inputs->constants.data_length !=
                      out_plan->descriptor->kernel_args.kernarg_size)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "custom dispatch argument length mismatch; expected %u but got "
          "%" PRIhsz,
          out_plan->descriptor->kernel_args.kernarg_size,
          inputs->constants.data_length);
    }
    out_plan->layout = &out_plan->descriptor->custom_kernarg_layout;
    out_plan->kernarg_block_count =
        iree_max(1u, out_plan->descriptor->custom_kernarg_block_count);
    out_plan->kernarg_strategy =
        IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_CUSTOM_DIRECT;
    return iree_ok_status();
  }

  const iree_host_size_t expected_constant_length =
      (iree_host_size_t)out_plan->descriptor->kernel_args.constant_count *
      sizeof(uint32_t);
  if (IREE_UNLIKELY(inputs->constants.data_length !=
                    expected_constant_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch constant count mismatch; expected %u but got %" PRIhsz,
        (uint32_t)out_plan->descriptor->kernel_args.constant_count,
        inputs->constants.data_length / sizeof(uint32_t));
  }
  if (IREE_UNLIKELY(inputs->bindings.count !=
                    out_plan->descriptor->kernel_args.binding_count)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch binding count mismatch; expected %u but got %" PRIhsz,
        (uint32_t)out_plan->descriptor->kernel_args.binding_count,
        inputs->bindings.count);
  }
  if (IREE_UNLIKELY(inputs->bindings.count > 0 && !inputs->bindings.values)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch bindings must be non-null when count is non-zero");
  }
  out_plan->bindings.dynamic_count =
      iree_hal_amdgpu_aql_command_buffer_count_dynamic_dispatch_bindings(
          inputs->bindings);
  if (out_plan->bindings.dynamic_count != 0) {
    out_plan->flags |= IREE_HAL_AMDGPU_AQL_DISPATCH_PLAN_FLAG_DYNAMIC_BINDINGS;
  }
  if (out_plan->bindings.dynamic_count != 0 &&
      !iree_hal_amdgpu_aql_dispatch_plan_uses_indirect_parameters(out_plan) &&
      out_plan->bindings.dynamic_count == inputs->bindings.count) {
    out_plan->kernarg_strategy =
        IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_DYNAMIC_BINDINGS;
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_prepare_dispatch_binding_sources(
          command_buffer, inputs->bindings));
  out_plan->layout = &out_plan->descriptor->hal_kernarg_layout;
  out_plan->kernarg_block_count =
      iree_max(1u, out_plan->descriptor->hal_kernarg_block_count);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_retain_dispatch_inputs(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_aql_dispatch_inputs_t* inputs) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_ensure_resource_set(command_buffer));
  if (!command_buffer->resource_set) return iree_ok_status();
  return iree_hal_resource_set_insert(command_buffer->resource_set,
                                      /*count=*/1, &inputs->executable);
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_calculate_dispatch_layout(
    const iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_aql_dispatch_inputs_t* inputs,
    const iree_hal_amdgpu_aql_dispatch_plan_t* plan,
    iree_hal_amdgpu_aql_dispatch_layout_t* out_layout) {
  *out_layout = (iree_hal_amdgpu_aql_dispatch_layout_t){0};

  const iree_host_size_t binding_bytes =
      iree_hal_amdgpu_aql_dispatch_plan_uses_custom_direct_arguments(plan)
          ? 0
          : (iree_host_size_t)plan->kernel_args->binding_count *
                sizeof(uint64_t);
  const iree_host_size_t tail_byte_length =
      plan->layout->total_kernarg_size - binding_bytes;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_qword_length(
      tail_byte_length, "dispatch tail payload",
      &out_layout->kernarg.tail_length_qwords,
      &out_layout->kernarg.tail_padded_length));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_qword_length(
      plan->layout->total_kernarg_size, "dispatch kernarg",
      &out_layout->kernarg.total_length_qwords,
      &out_layout->kernarg.total_padded_length));
  out_layout->kernarg.implicit_args_offset_qwords =
      plan->layout->has_implicit_args
          ? (uint16_t)(plan->layout->implicit_args_offset / 8)
          : UINT16_MAX;
  if (IREE_UNLIKELY(plan->kernarg_block_count >
                    UINT32_MAX / sizeof(iree_hal_amdgpu_kernarg_block_t))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch kernargs require too many kernarg blocks (%" PRIu32 ")",
        plan->kernarg_block_count);
  }

  const uint32_t dispatch_kernarg_block_length =
      plan->kernarg_block_count * sizeof(iree_hal_amdgpu_kernarg_block_t);
  const bool uses_indirect_parameters =
      iree_hal_amdgpu_aql_dispatch_plan_uses_indirect_parameters(plan);
  const bool uses_custom_direct_arguments =
      iree_hal_amdgpu_aql_dispatch_plan_uses_custom_direct_arguments(plan);
  const uint32_t patch_kernarg_block_length =
      uses_indirect_parameters ? sizeof(iree_hal_amdgpu_kernarg_block_t) : 0;
  const uint32_t kernarg_block_length =
      patch_kernarg_block_length + dispatch_kernarg_block_length;
  // Prepublication is a reusable-command-buffer strategy for immutable
  // kernargs. It materializes static kernargs once at end() so replay avoids
  // queue-time kernarg reservation, binding patching, and block growth.
  if (iree_hal_amdgpu_aql_command_buffer_prepublish_enabled(command_buffer) &&
      !iree_all_bits_set(command_buffer->base.mode,
                         IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT) &&
      !uses_indirect_parameters &&
      !iree_hal_amdgpu_aql_dispatch_plan_has_dynamic_bindings(plan)) {
    out_layout->flags |=
        IREE_HAL_AMDGPU_AQL_DISPATCH_LAYOUT_FLAG_PREPUBLISH_KERNARGS;
  }
  // Mixed static/dynamic reusable dispatches keep an immutable host template
  // and patch only the dynamic binding qwords at replay time. All-dynamic
  // dispatches stay on the compact inline form but use a dynamic-only replay
  // strategy so packet processing does not branch over impossible static
  // binding source cases.
  if (!iree_all_bits_set(command_buffer->base.mode,
                         IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT) &&
      !uses_indirect_parameters && !uses_custom_direct_arguments &&
      iree_hal_amdgpu_aql_dispatch_plan_has_dynamic_bindings(plan) &&
      plan->bindings.dynamic_count < inputs->bindings.count) {
    out_layout->flags |=
        IREE_HAL_AMDGPU_AQL_DISPATCH_LAYOUT_FLAG_PATCH_KERNARG_TEMPLATE;
  }
  const bool prepublishes_kernargs =
      iree_hal_amdgpu_aql_dispatch_layout_prepublishes_kernargs(out_layout);
  const bool uses_kernarg_template =
      iree_hal_amdgpu_aql_dispatch_layout_uses_kernarg_template(out_layout);
  const bool patches_kernarg_template =
      iree_hal_amdgpu_aql_dispatch_layout_patches_kernarg_template(out_layout);
  out_layout->kernarg.queue_block_length =
      prepublishes_kernargs ? 0 : kernarg_block_length;

  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amdgpu_command_buffer_dispatch_command_t),
      &out_layout->command.byte_length,
      IREE_STRUCT_FIELD(
          uses_kernarg_template ? 0 : out_layout->kernarg.tail_padded_length,
          uint8_t, NULL)));
  out_layout->command.binding_source_count =
      prepublishes_kernargs
          ? 0
          : (patches_kernarg_template
                 ? plan->bindings.dynamic_count
                 : (uint16_t)((uses_custom_direct_arguments
                                   ? 0
                                   : inputs->bindings.count) +
                              (uses_indirect_parameters ? 1 : 0)));
  out_layout->command.aql_packet_count = uses_indirect_parameters ? 2 : 1;
  return iree_ok_status();
}

static void iree_hal_amdgpu_aql_command_buffer_initialize_dispatch_command(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_aql_dispatch_inputs_t* inputs,
    const iree_hal_amdgpu_aql_dispatch_plan_t* plan,
    const iree_hal_amdgpu_aql_dispatch_layout_t* layout,
    uint64_t kernarg_template_reference,
    iree_hal_amdgpu_command_buffer_command_header_t* header,
    iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources) {
  iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command =
      (iree_hal_amdgpu_command_buffer_dispatch_command_t*)header;
  dispatch_command->kernel_object = plan->kernel_args->kernel_object;
  dispatch_command->binding_source_offset =
      binding_sources
          ? (uint32_t)((uint8_t*)binding_sources -
                       (uint8_t*)command_buffer->builder.current_block.header)
          : 0;
  const bool prepublishes_kernargs =
      iree_hal_amdgpu_aql_dispatch_layout_prepublishes_kernargs(layout);
  const bool patches_kernarg_template =
      iree_hal_amdgpu_aql_dispatch_layout_patches_kernarg_template(layout);
  const bool uses_kernarg_template =
      prepublishes_kernargs || patches_kernarg_template;
  dispatch_command->payload_reference =
      uses_kernarg_template
          ? (uint32_t)kernarg_template_reference
          : sizeof(iree_hal_amdgpu_command_buffer_dispatch_command_t);
  dispatch_command->binding_count = (uint16_t)inputs->bindings.count;
  dispatch_command->kernarg_length_qwords = layout->kernarg.total_length_qwords;
  dispatch_command->payload.tail_length_qwords =
      uses_kernarg_template ? 0 : layout->kernarg.tail_length_qwords;
  if (patches_kernarg_template) {
    dispatch_command->payload.patch_source_count = plan->bindings.dynamic_count;
  }
  if (prepublishes_kernargs) {
    dispatch_command->kernarg_strategy =
        IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_PREPUBLISHED;
  } else if (patches_kernarg_template) {
    dispatch_command->kernarg_strategy =
        IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_PATCHED_TEMPLATE;
  } else {
    dispatch_command->kernarg_strategy = (uint8_t)plan->kernarg_strategy;
  }
  const bool uses_indirect_parameters =
      iree_hal_amdgpu_aql_dispatch_plan_uses_indirect_parameters(plan);
  dispatch_command->dispatch_flags =
      uses_indirect_parameters
          ? IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_INDIRECT_PARAMETERS
          : IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_NONE;
  dispatch_command->setup = plan->kernel_args->setup;
  dispatch_command->export_ordinal = inputs->export_ordinal;
  dispatch_command->workgroup_size[0] = plan->kernel_args->workgroup_size[0];
  dispatch_command->workgroup_size[1] = plan->kernel_args->workgroup_size[1];
  dispatch_command->workgroup_size[2] = plan->kernel_args->workgroup_size[2];
  dispatch_command->implicit_args_offset_qwords =
      layout->kernarg.implicit_args_offset_qwords;
  dispatch_command->grid_size[0] =
      uses_indirect_parameters ? 0
                               : inputs->config.workgroup_count[0] *
                                     plan->kernel_args->workgroup_size[0];
  dispatch_command->grid_size[1] =
      uses_indirect_parameters ? 0
                               : inputs->config.workgroup_count[1] *
                                     plan->kernel_args->workgroup_size[1];
  dispatch_command->grid_size[2] =
      uses_indirect_parameters ? 0
                               : inputs->config.workgroup_count[2] *
                                     plan->kernel_args->workgroup_size[2];
  dispatch_command->private_segment_size =
      plan->kernel_args->private_segment_size;
  dispatch_command->group_segment_size =
      plan->kernel_args->group_segment_size +
      inputs->config.dynamic_workgroup_local_memory;
  dispatch_command->executable_id =
      iree_hal_amdgpu_executable_profile_id(inputs->executable);
}

static iree_status_t
iree_hal_amdgpu_aql_command_buffer_ensure_current_dispatch_summary_block(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_amdgpu_aql_command_buffer_dispatch_summary_block_t**
        out_summary_block) {
  *out_summary_block = command_buffer->dispatch_summaries.block.current;
  if (*out_summary_block && (*out_summary_block)->header == block) {
    return iree_ok_status();
  }

  iree_hal_amdgpu_aql_command_buffer_dispatch_summary_block_t* summary_block =
      NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&command_buffer->recording_arena,
                                           sizeof(*summary_block),
                                           (void**)&summary_block));
  memset(summary_block, 0, sizeof(*summary_block));
  summary_block->header = block;
  if (command_buffer->dispatch_summaries.block.current) {
    command_buffer->dispatch_summaries.block.current->next = summary_block;
  } else {
    command_buffer->dispatch_summaries.block.first = summary_block;
  }
  command_buffer->dispatch_summaries.block.current = summary_block;
  *out_summary_block = summary_block;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_record_dispatch_summary(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    const iree_hal_amdgpu_aql_dispatch_layout_t* layout) {
  if (!iree_hal_amdgpu_aql_command_buffer_retains_dispatch_summaries(
          command_buffer)) {
    return iree_ok_status();
  }

  iree_hal_amdgpu_aql_command_buffer_dispatch_summary_block_t* summary_block =
      NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_ensure_current_dispatch_summary_block(
          command_buffer, command_buffer->builder.current_block.header,
          &summary_block));

  iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t* summary = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&command_buffer->recording_arena,
                                           sizeof(*summary), (void**)&summary));
  memset(summary, 0, sizeof(*summary));
  const uint32_t first_packet_ordinal =
      command_buffer->builder.current_block.aql_packet_count -
      layout->command.aql_packet_count;
  summary->packets.first_ordinal = first_packet_ordinal;
  const bool uses_indirect_parameters = iree_any_bit_set(
      dispatch_command->dispatch_flags,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_INDIRECT_PARAMETERS);
  summary->packets.dispatch_ordinal =
      first_packet_ordinal + (uses_indirect_parameters ? 1u : 0u);
  summary->metadata.executable_id = dispatch_command->executable_id;
  summary->metadata.command_index = dispatch_command->header.command_index;
  summary->metadata.export_ordinal = dispatch_command->export_ordinal;
  summary->metadata.dispatch_flags = dispatch_command->dispatch_flags;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(summary->workgroup.size);
       ++i) {
    summary->workgroup.size[i] = dispatch_command->workgroup_size[i];
    if (!uses_indirect_parameters && dispatch_command->workgroup_size[i] != 0) {
      summary->workgroup.count[i] =
          dispatch_command->grid_size[i] / dispatch_command->workgroup_size[i];
    }
  }

  if (summary_block->dispatch.last) {
    summary_block->dispatch.last->next = summary;
  } else {
    summary_block->dispatch.first = summary;
  }
  summary_block->dispatch.last = summary;
  ++summary_block->dispatch.count;
  ++command_buffer->dispatch_summaries.count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_append_dispatch_command(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_aql_dispatch_inputs_t* inputs,
    const iree_hal_amdgpu_aql_dispatch_plan_t* plan,
    const iree_hal_amdgpu_aql_dispatch_layout_t* layout,
    iree_hal_amdgpu_command_buffer_dispatch_command_t** out_dispatch_command,
    iree_hal_amdgpu_command_buffer_binding_source_t** out_binding_sources) {
  *out_dispatch_command = NULL;
  *out_binding_sources = NULL;

  iree_hal_amdgpu_command_buffer_command_header_t* header = NULL;
  const bool uses_indirect_parameters =
      iree_hal_amdgpu_aql_dispatch_plan_uses_indirect_parameters(plan);
  const uint8_t command_flags =
      uses_indirect_parameters
          ? IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER
          : IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_builder_append_command(
      &command_buffer->builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH,
      command_flags, layout->command.byte_length,
      layout->command.binding_source_count, layout->command.aql_packet_count,
      layout->kernarg.queue_block_length, &header, out_binding_sources));

  uint64_t kernarg_template_reference = 0;
  if (iree_hal_amdgpu_aql_dispatch_layout_prepublishes_kernargs(layout)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_aql_command_buffer_record_prepublished_dispatch_kernargs(
            command_buffer,
            (iree_hal_amdgpu_command_buffer_dispatch_command_t*)header,
            plan->kernel_args, plan->layout, inputs->config, inputs->constants,
            inputs->bindings, plan->kernarg_strategy,
            layout->kernarg.total_padded_length, &kernarg_template_reference));
    if (IREE_UNLIKELY(kernarg_template_reference > UINT32_MAX)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "prepublished command-buffer kernarg rodata "
                              "ordinal exceeds uint32_t");
    }
  } else if (iree_hal_amdgpu_aql_dispatch_layout_patches_kernarg_template(
                 layout)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_aql_command_buffer_record_patched_dispatch_kernargs(
            command_buffer, plan->kernel_args, plan->layout, inputs->config,
            inputs->constants, inputs->bindings, plan->kernarg_strategy,
            layout->kernarg.total_padded_length, &kernarg_template_reference));
    if (IREE_UNLIKELY(kernarg_template_reference > UINT32_MAX)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "patched command-buffer kernarg template rodata "
                              "ordinal exceeds uint32_t");
    }
  }

  iree_hal_amdgpu_aql_command_buffer_initialize_dispatch_command(
      command_buffer, inputs, plan, layout, kernarg_template_reference, header,
      *out_binding_sources);
  if (uses_indirect_parameters) {
    ++command_buffer->builder.current_block.indirect_dispatch_count;
  }
  *out_dispatch_command =
      (iree_hal_amdgpu_command_buffer_dispatch_command_t*)header;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_write_dispatch_payload(
    iree_hal_amdgpu_aql_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_aql_dispatch_inputs_t* inputs,
    const iree_hal_amdgpu_aql_dispatch_plan_t* plan,
    const iree_hal_amdgpu_aql_dispatch_layout_t* layout,
    iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources) {
  const bool uses_custom_direct_arguments =
      iree_hal_amdgpu_aql_dispatch_plan_uses_custom_direct_arguments(plan);
  if (binding_sources && !uses_custom_direct_arguments) {
    if (iree_hal_amdgpu_aql_dispatch_layout_patches_kernarg_template(layout)) {
      iree_hal_amdgpu_aql_command_buffer_write_dynamic_dispatch_binding_sources(
          inputs->bindings, binding_sources);
    } else {
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_aql_command_buffer_write_dispatch_binding_sources(
              command_buffer, inputs->bindings, binding_sources));
    }
  }
  if (iree_hal_amdgpu_aql_dispatch_plan_uses_indirect_parameters(plan)) {
    iree_hal_amdgpu_command_buffer_binding_source_t* parameter_source =
        binding_sources +
        (uses_custom_direct_arguments ? 0 : inputs->bindings.count);
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_aql_command_buffer_write_indirect_parameter_source(
            command_buffer, inputs->config.workgroup_count_ref,
            parameter_source));
  }
  if (iree_hal_amdgpu_aql_dispatch_layout_uses_kernarg_template(layout)) {
    return iree_ok_status();
  }

  uint8_t* tail_payload =
      (uint8_t*)dispatch_command + dispatch_command->payload_reference;
  return iree_hal_amdgpu_aql_command_buffer_write_dispatch_tail(
      plan->kernel_args, plan->layout, inputs->config, inputs->constants,
      plan->kernarg_strategy, tail_payload);
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
  const iree_hal_amdgpu_aql_dispatch_inputs_t inputs = {
      .executable = executable,
      .export_ordinal = export_ordinal,
      .config = config,
      .constants = constants,
      .bindings = bindings,
      .flags = flags,
  };
  const bool validates =
      iree_hal_amdgpu_aql_command_buffer_validates(command_buffer);
  iree_hal_amdgpu_aql_dispatch_plan_t plan;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_command_buffer_prepare_dispatch_plan(
      command_buffer, &inputs, validates, &plan));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_retain_dispatch_inputs(command_buffer,
                                                                &inputs));

  iree_hal_amdgpu_aql_dispatch_layout_t layout;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_calculate_dispatch_layout(
          command_buffer, &inputs, &plan, &layout));

  iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command = NULL;
  iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_append_dispatch_command(
          command_buffer, &inputs, &plan, &layout, &dispatch_command,
          &binding_sources));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_command_buffer_write_dispatch_payload(
          command_buffer, &inputs, &plan, &layout, dispatch_command,
          binding_sources));
  return iree_hal_amdgpu_aql_command_buffer_record_dispatch_summary(
      command_buffer, dispatch_command, &layout);
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
