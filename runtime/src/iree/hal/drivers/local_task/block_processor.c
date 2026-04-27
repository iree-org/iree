// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/block_processor.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/internal/cpu.h"
#include "iree/base/threading/wait_address.h"
#include "iree/hal/local/local_executable.h"
#include "iree/task/executor.h"

//===----------------------------------------------------------------------===//
// Tuning
//===----------------------------------------------------------------------===//

// Maximum sequential tiles reserved by one successful dispatch tile claim.
// Larger reservations reduce shared tile_index contention and improve locality
// on large grids; small grids keep single-tile reservations so all workers can
// participate.
#define IREE_HAL_CMD_DISPATCH_MAX_TILES_PER_RESERVATION (8)

// Region lookahead width bucket required before no-work drainers request the
// longer warm-spin handoff window from the task worker. Buckets at or below the
// normal tail-retention cap are handled by the generic task policy.
#define IREE_HAL_CMD_BLOCK_PROCESSOR_WARM_SPIN_LOOKAHEAD_WIDTH (8)

//===----------------------------------------------------------------------===//
// Command execution helpers
//===----------------------------------------------------------------------===//

// Fills |length| bytes at |target| with a repeating pattern of
// |pattern_length| bytes from |pattern|. |length| must be a multiple of
// |pattern_length|. |pattern_length| must be 1, 2, or 4.
static void iree_hal_cmd_fill_pattern(uint8_t* target,
                                      iree_device_size_t length,
                                      uint64_t pattern,
                                      uint8_t pattern_length) {
  switch (pattern_length) {
    case 1: {
      memset(target, (uint8_t)pattern, (size_t)length);
      break;
    }
    case 2: {
      uint16_t value = (uint16_t)pattern;
      for (iree_device_size_t i = 0; i < length; i += 2) {
        memcpy(target + i, &value, 2);
      }
      break;
    }
    case 4: {
      uint32_t value = (uint32_t)pattern;
      for (iree_device_size_t i = 0; i < length; i += 4) {
        memcpy(target + i, &value, 4);
      }
      break;
    }
    default:
      IREE_ASSERT(false, "unsupported fill pattern length");
      break;
  }
}

// Advances the task-process retention epoch, if one is attached.
static void iree_hal_cmd_block_processor_advance_retention_epoch(
    iree_hal_cmd_block_processor_context_t* context) {
  if (context->retention_epoch_ptr) {
    iree_atomic_fetch_add(context->retention_epoch_ptr, 1,
                          iree_memory_order_release);
    if (context->retention_sleepers_ptr &&
        iree_atomic_load(context->retention_sleepers_ptr,
                         iree_memory_order_acquire) > 0) {
      IREE_TRACE_ZONE_BEGIN_NAMED(
          z_wake, "iree_hal_local_task_wake_retention_sleepers");
      iree_wait_address_wake_all(context->retention_epoch_ptr);
      IREE_TRACE_ZONE_END(z_wake);
    }
  }
}

static iree_time_t iree_hal_cmd_block_processor_retention_deadline_after(
    iree_time_t now, iree_duration_t duration) {
  if (duration <= IREE_DURATION_ZERO) return now;
  iree_time_t deadline = now + duration;
  return deadline < now ? IREE_TIME_INFINITE_FUTURE : deadline;
}

static void iree_hal_cmd_block_processor_extend_retention_deadline(
    iree_hal_cmd_block_processor_context_t* context, iree_time_t deadline) {
  if (!context->retention_spin_deadline_ns_ptr) return;
  int64_t current_deadline = iree_atomic_load(
      context->retention_spin_deadline_ns_ptr, iree_memory_order_relaxed);
  while (current_deadline < deadline) {
    if (iree_atomic_compare_exchange_weak(
            context->retention_spin_deadline_ns_ptr, &current_deadline,
            deadline, iree_memory_order_release, iree_memory_order_relaxed)) {
      return;
    }
  }
}

// Keeps warm task workers alive across region-transition bookkeeping.
static void iree_hal_cmd_block_processor_begin_retention_transition(
    iree_hal_cmd_block_processor_context_t* context) {
  if (context->retention_transition_spin_ns <= IREE_DURATION_ZERO) return;
  if (!context->retention_spin_deadline_ns_ptr) return;
  const iree_time_t now = iree_time_now();
  iree_hal_cmd_block_processor_extend_retention_deadline(
      context, iree_hal_cmd_block_processor_retention_deadline_after(
                   now, context->retention_transition_spin_ns));
}

// Marks the processor completed and releases any warm task workers.
static void iree_hal_cmd_block_processor_mark_completed(
    iree_hal_cmd_block_processor_context_t* context) {
  iree_atomic_store(&context->completed, 1, iree_memory_order_release);
  iree_hal_cmd_block_processor_advance_retention_epoch(context);
}

// Records the first error encountered by any worker. Thread-safe via CAS.
// If another worker already recorded an error, the losing reporter frees its
// status. Also sets the completed flag so that all workers exit on their next
// drain() call.
static void iree_hal_cmd_block_processor_report_error(
    iree_hal_cmd_block_processor_context_t* context, iree_status_t status) {
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &context->error_status, &expected, (intptr_t)status,
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    iree_status_free(status);
  }
  iree_hal_cmd_block_processor_mark_completed(context);
}

// Returns true if any worker has reported an error (early exit check).
static bool iree_hal_cmd_block_processor_has_error(
    const iree_hal_cmd_block_processor_context_t* context) {
  return iree_atomic_load(&context->error_status, iree_memory_order_relaxed) !=
         0;
}

static iree_hal_cmd_block_state_t* iree_hal_cmd_block_processor_state_at(
    iree_hal_cmd_block_processor_context_t* context, uint16_t state_index) {
  return (iree_hal_cmd_block_state_t*)((uint8_t*)context->state_storage +
                                       (iree_host_size_t)state_index *
                                           context->state_stride);
}

static iree_hal_cmd_block_state_t* iree_hal_cmd_block_processor_next_state(
    iree_hal_cmd_block_processor_context_t* context) {
  if (context->next_block_state_index >= context->state_count) return NULL;
  iree_hal_cmd_block_state_t* state = iree_hal_cmd_block_processor_state_at(
      context, context->next_block_state_index);
  ++context->next_block_state_index;
  return state;
}

static void iree_hal_cmd_block_processor_publish_state(
    iree_hal_cmd_block_processor_context_t* context,
    iree_hal_cmd_block_state_t* state) {
  iree_atomic_store(&context->current_state, (intptr_t)state,
                    iree_memory_order_release);
}

static iree_hal_cmd_block_state_t* iree_hal_cmd_block_processor_current_state(
    const iree_hal_cmd_block_processor_context_t* context) {
  return (iree_hal_cmd_block_state_t*)iree_atomic_load(
      &context->current_state, iree_memory_order_acquire);
}

// Resolves all binding pointers and lengths for a block via its fixup table.
// Populates state->binding_ptrs[] and binding_lengths[] for each fixup entry.
static void iree_hal_cmd_block_processor_resolve_bindings(
    const iree_hal_cmd_block_header_t* block, iree_hal_cmd_block_state_t* state,
    uint16_t max_region_dispatch_count, uint16_t total_binding_count,
    const iree_hal_cmd_binding_entry_t* binding_table,
    iree_host_size_t binding_table_length) {
  void** binding_ptrs =
      iree_hal_cmd_block_state_binding_ptrs(state, max_region_dispatch_count);
  size_t* binding_lengths = iree_hal_cmd_block_state_binding_lengths(
      state, max_region_dispatch_count, total_binding_count);

  const iree_hal_cmd_fixup_t* fixups = iree_hal_cmd_block_fixups(block);
  for (uint16_t i = 0; i < block->fixup_count; ++i) {
    const iree_hal_cmd_fixup_t* fixup = &fixups[i];
    if (!fixup->host_ptr) {
      // Indirect fixup (fast path): look up in the binding table.
      IREE_ASSERT(fixup->slot < binding_table_length);
      binding_ptrs[fixup->data_index] =
          (uint8_t*)binding_table[fixup->slot].base + fixup->offset;
      binding_lengths[fixup->data_index] =
          binding_table[fixup->slot].length - (size_t)fixup->offset;
    } else if (iree_any_bit_set(fixup->flags, IREE_HAL_CMD_FIXUP_FLAG_SPAN)) {
      // Span fixup (rare): async/registered region dereference.
      binding_ptrs[fixup->data_index] =
          iree_async_span_ptr(*fixup->span) + fixup->offset;
      binding_lengths[fixup->data_index] = (size_t)fixup->length;
    } else {
      // Direct inline fixup: host pointer already resolved at recording time.
      binding_ptrs[fixup->data_index] =
          (uint8_t*)fixup->host_ptr + fixup->offset;
      binding_lengths[fixup->data_index] = (size_t)fixup->length;
    }
  }
}

static void iree_hal_cmd_dispatch_read_workgroup_count(
    const iree_hal_cmd_dispatch_t* dispatch, void** binding_ptrs,
    uint32_t out_workgroup_count[3]) {
  if (dispatch->header.flags & IREE_HAL_CMD_FLAG_INDIRECT) {
    const void* params_buffer =
        binding_ptrs[dispatch->params.indirect.params_binding];
    const iree_hal_dispatch_params_t* params =
        (const iree_hal_dispatch_params_t*)((const uint8_t*)params_buffer +
                                            dispatch->params.indirect
                                                .params_offset);
    out_workgroup_count[0] = params->workgroup_count[0];
    out_workgroup_count[1] = params->workgroup_count[1];
    out_workgroup_count[2] = params->workgroup_count[2];
  } else {
    out_workgroup_count[0] = dispatch->params.direct.workgroup_count[0];
    out_workgroup_count[1] = dispatch->params.direct.workgroup_count[1];
    out_workgroup_count[2] = dispatch->params.direct.workgroup_count[2];
  }
}

static uint32_t iree_hal_cmd_dispatch_tile_count(
    const iree_hal_cmd_dispatch_t* dispatch, void** binding_ptrs) {
  uint32_t workgroup_count[3];
  iree_hal_cmd_dispatch_read_workgroup_count(dispatch, binding_ptrs,
                                             workgroup_count);
  return workgroup_count[0] * workgroup_count[1] * workgroup_count[2];
}

static iree_hal_fill_params_t iree_hal_cmd_fill_read_params(
    const iree_hal_cmd_fill_t* fill, void** binding_ptrs) {
  if (fill->header.flags & IREE_HAL_CMD_FLAG_INDIRECT) {
    const void* params_buffer =
        binding_ptrs[fill->params.indirect.params_binding];
    return *(
        const iree_hal_fill_params_t*)((const uint8_t*)params_buffer +
                                       fill->params.indirect.params_offset);
  }
  return fill->params.direct;
}

static iree_hal_copy_params_t iree_hal_cmd_copy_read_params(
    const iree_hal_cmd_copy_t* copy, void** binding_ptrs) {
  if (copy->header.flags & IREE_HAL_CMD_FLAG_INDIRECT) {
    const void* params_buffer =
        binding_ptrs[copy->params.indirect.params_binding];
    return *(
        const iree_hal_copy_params_t*)((const uint8_t*)params_buffer +
                                       copy->params.indirect.params_offset);
  }
  return copy->params.direct;
}

static uint32_t iree_hal_cmd_region_tile_count(
    const iree_hal_cmd_barrier_t* barrier, void** binding_ptrs) {
  uint32_t tile_count = 0;
  const iree_hal_cmd_header_t* cmd = iree_hal_cmd_next(&barrier->header);
  for (uint8_t d = 0; d < barrier->dispatch_count; ++d) {
    switch (cmd->opcode) {
      case IREE_HAL_CMD_DISPATCH:
        tile_count += iree_hal_cmd_dispatch_tile_count(
            (const iree_hal_cmd_dispatch_t*)cmd, binding_ptrs);
        break;
      case IREE_HAL_CMD_FILL: {
        const iree_hal_cmd_fill_t* fill = (const iree_hal_cmd_fill_t*)cmd;
        const iree_hal_fill_params_t params =
            iree_hal_cmd_fill_read_params(fill, binding_ptrs);
        tile_count += iree_hal_cmd_transfer_tile_count(params.length);
        break;
      }
      case IREE_HAL_CMD_COPY: {
        const iree_hal_cmd_copy_t* copy = (const iree_hal_cmd_copy_t*)cmd;
        const iree_hal_copy_params_t params =
            iree_hal_cmd_copy_read_params(copy, binding_ptrs);
        tile_count += iree_hal_cmd_transfer_tile_count(params.length);
        break;
      }
      case IREE_HAL_CMD_UPDATE:
        tile_count += iree_hal_cmd_transfer_tile_count(
            ((const iree_hal_cmd_update_t*)cmd)->length);
        break;
      default:
        break;
    }
    cmd = iree_hal_cmd_next(cmd);
  }
  return tile_count;
}

static void iree_hal_cmd_block_processor_profile_reset_dispatch(
    iree_hal_cmd_block_processor_profile_dispatch_t* dispatch_profile) {
  iree_atomic_store(&dispatch_profile->start_host_time_ns, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&dispatch_profile->end_host_time_ns, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&dispatch_profile->tile_count, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&dispatch_profile->tile_duration_sum_ns, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&dispatch_profile->status_code, IREE_STATUS_OK,
                    iree_memory_order_relaxed);
}

static void iree_hal_cmd_block_processor_profile_reset_dispatches(
    iree_hal_cmd_block_processor_context_t* context) {
  if (!context->profile.dispatches) return;
  for (iree_host_size_t i = 0; i < context->profile.dispatch_capacity; ++i) {
    iree_hal_cmd_block_processor_profile_reset_dispatch(
        &context->profile.dispatches[i]);
  }
}

IREE_ATTRIBUTE_ALWAYS_INLINE static inline bool
iree_hal_cmd_block_processor_profile_records_regions(
    const iree_hal_cmd_block_processor_context_t* context) {
  return context->profile.command_region.events_enabled;
}

static void iree_hal_cmd_block_processor_profile_atomic_min_i64(
    iree_atomic_int64_t* target, int64_t value) {
  int64_t current = iree_atomic_load(target, iree_memory_order_relaxed);
  while ((current == 0 || value < current) &&
         !iree_atomic_compare_exchange_weak(target, &current, value,
                                            iree_memory_order_relaxed,
                                            iree_memory_order_relaxed)) {
  }
}

static void iree_hal_cmd_block_processor_profile_atomic_min_i32(
    iree_atomic_int32_t* target, int32_t value) {
  int32_t current = iree_atomic_load(target, iree_memory_order_relaxed);
  while ((current == 0 || value < current) &&
         !iree_atomic_compare_exchange_weak(target, &current, value,
                                            iree_memory_order_relaxed,
                                            iree_memory_order_relaxed)) {
  }
}

static void iree_hal_cmd_block_processor_profile_atomic_max_i64(
    iree_atomic_int64_t* target, int64_t value) {
  int64_t current = iree_atomic_load(target, iree_memory_order_relaxed);
  while (value > current &&
         !iree_atomic_compare_exchange_weak(target, &current, value,
                                            iree_memory_order_relaxed,
                                            iree_memory_order_relaxed)) {
  }
}

static void iree_hal_cmd_block_processor_profile_atomic_max_i32(
    iree_atomic_int32_t* target, int32_t value) {
  int32_t current = iree_atomic_load(target, iree_memory_order_relaxed);
  while (value > current &&
         !iree_atomic_compare_exchange_weak(target, &current, value,
                                            iree_memory_order_relaxed,
                                            iree_memory_order_relaxed)) {
  }
}

static uint32_t
iree_hal_cmd_block_processor_profile_remaining_tile_bucket_index(
    uint32_t remaining_tile_count) {
  if (remaining_tile_count <= 2) return remaining_tile_count;
  uint32_t bucket_index = 3;
  uint32_t bucket_limit = 4;
  while (remaining_tile_count > bucket_limit &&
         bucket_index + 1 <
             IREE_HAL_PROFILE_COMMAND_REGION_REMAINING_TILE_BUCKET_COUNT) {
    ++bucket_index;
    bucket_limit <<= 1;
  }
  return bucket_index;
}

typedef struct iree_hal_cmd_block_processor_profile_region_snapshot_t {
  // Snapshot of the scheduler-visible command-buffer region.
  struct {
    // Host timestamp when the region became claimable.
    iree_time_t start_host_time_ns;

    // Initial tile count observed when the region became claimable.
    uint32_t tile_count;

    // Number of drains that executed one or more tiles.
    uint32_t useful_drain_count;

    // Number of drains that found no claimable tiles.
    uint32_t no_work_drain_count;

    // Tail no-work observations collected after tile claiming.
    struct {
      // Number of active-region drains that found no claimable tile.
      uint32_t count;

      // Unfinished tile counts observed by tail no-work drains.
      struct {
        // Minimum unfinished tile count observed.
        uint32_t min;

        // Maximum unfinished tile count observed.
        uint32_t max;

        // Power-of-two unfinished tile count histogram.
        uint32_t bucket_counts
            [IREE_HAL_PROFILE_COMMAND_REGION_REMAINING_TILE_BUCKET_COUNT];
      } remaining_tiles;

      // Host timestamp when the first drain began, or 0.
      iree_time_t first_start_host_time_ns;

      // Host timestamp when the last drain ended, or 0.
      iree_time_t last_end_host_time_ns;

      // Accumulated region-relative time values for tail no-work drains.
      struct {
        // Sum of no-work drain start offsets from the region start.
        iree_time_t start_offset_ns;

        // Sum of no-work drain durations.
        iree_time_t drain_duration_ns;
      } time_sums;
    } tail_no_work;

    // Host timestamp when the first useful drain began, or 0.
    iree_time_t first_useful_drain_start_host_time_ns;

    // Host timestamp when the last useful drain ended, or 0.
    iree_time_t last_useful_drain_end_host_time_ns;

    // Warm-worker retention behavior observed while this region was active.
    struct {
      // No-work drains that kept the process active after advancement.
      uint32_t keep_active_count;

      // No-work drains that explicitly republished process activity.
      uint32_t publish_keep_active_count;

      // No-work drains that waited warm on the process retention epoch.
      uint32_t keep_warm_count;
    } retention;
  } command_region;
} iree_hal_cmd_block_processor_profile_region_snapshot_t;

static iree_hal_cmd_block_processor_profile_region_snapshot_t
iree_hal_cmd_block_processor_profile_snapshot_active_region(
    iree_hal_cmd_block_processor_context_t* context) {
  iree_hal_cmd_block_processor_profile_region_snapshot_t snapshot = {0};
  if (!iree_hal_cmd_block_processor_profile_records_regions(context)) {
    return snapshot;
  }
  snapshot.command_region.start_host_time_ns =
      iree_atomic_load(&context->profile.command_region.start_host_time_ns,
                       iree_memory_order_relaxed);
  const int32_t tile_count = iree_atomic_load(
      &context->profile.command_region.tile_count, iree_memory_order_relaxed);
  const int32_t useful_drain_count =
      iree_atomic_load(&context->profile.command_region.useful_drain_count,
                       iree_memory_order_relaxed);
  const int32_t no_work_drain_count =
      iree_atomic_load(&context->profile.command_region.no_work_drain_count,
                       iree_memory_order_relaxed);
  const int32_t tail_no_work_remaining_tile_min = iree_atomic_load(
      &context->profile.command_region.tail_no_work.remaining_tiles.min,
      iree_memory_order_relaxed);
  const int32_t tail_no_work_remaining_tile_max = iree_atomic_load(
      &context->profile.command_region.tail_no_work.remaining_tiles.max,
      iree_memory_order_relaxed);
  const iree_time_t first_useful_drain_start_host_time_ns = iree_atomic_load(
      &context->profile.command_region.first_useful_drain_start_host_time_ns,
      iree_memory_order_relaxed);
  const iree_time_t last_useful_drain_end_host_time_ns = iree_atomic_load(
      &context->profile.command_region.last_useful_drain_end_host_time_ns,
      iree_memory_order_relaxed);
  const iree_time_t tail_no_work_first_start_host_time_ns = iree_atomic_load(
      &context->profile.command_region.tail_no_work.first_start_host_time_ns,
      iree_memory_order_relaxed);
  const iree_time_t tail_no_work_last_end_host_time_ns = iree_atomic_load(
      &context->profile.command_region.tail_no_work.last_end_host_time_ns,
      iree_memory_order_relaxed);
  const iree_time_t tail_no_work_start_offset_ns_sum = iree_atomic_load(
      &context->profile.command_region.tail_no_work.time_sums.start_offset_ns,
      iree_memory_order_relaxed);
  const iree_time_t tail_no_work_drain_duration_ns_sum = iree_atomic_load(
      &context->profile.command_region.tail_no_work.time_sums.drain_duration_ns,
      iree_memory_order_relaxed);
  const int32_t keep_active_count = iree_atomic_load(
      &context->profile.command_region.retention.keep_active_count,
      iree_memory_order_relaxed);
  const int32_t publish_keep_active_count = iree_atomic_load(
      &context->profile.command_region.retention.publish_keep_active_count,
      iree_memory_order_relaxed);
  const int32_t keep_warm_count = iree_atomic_load(
      &context->profile.command_region.retention.keep_warm_count,
      iree_memory_order_relaxed);
  snapshot.command_region.first_useful_drain_start_host_time_ns =
      first_useful_drain_start_host_time_ns;
  snapshot.command_region.last_useful_drain_end_host_time_ns =
      last_useful_drain_end_host_time_ns;
  snapshot.command_region.tile_count =
      tile_count > 0 ? (uint32_t)tile_count : 0;
  snapshot.command_region.useful_drain_count =
      useful_drain_count > 0 ? (uint32_t)useful_drain_count : 0;
  snapshot.command_region.no_work_drain_count =
      no_work_drain_count > 0 ? (uint32_t)no_work_drain_count : 0;
  snapshot.command_region.tail_no_work.remaining_tiles.min =
      tail_no_work_remaining_tile_min > 0
          ? (uint32_t)tail_no_work_remaining_tile_min
          : 0;
  snapshot.command_region.tail_no_work.remaining_tiles.max =
      tail_no_work_remaining_tile_max > 0
          ? (uint32_t)tail_no_work_remaining_tile_max
          : 0;
  for (iree_host_size_t i = 0;
       i <
       IREE_ARRAYSIZE(
           snapshot.command_region.tail_no_work.remaining_tiles.bucket_counts);
       ++i) {
    const int32_t bucket_count =
        iree_atomic_load(&context->profile.command_region.tail_no_work
                              .remaining_tiles.bucket_counts[i],
                         iree_memory_order_relaxed);
    const uint32_t normalized_bucket_count =
        bucket_count > 0 ? (uint32_t)bucket_count : 0;
    snapshot.command_region.tail_no_work.remaining_tiles.bucket_counts[i] =
        normalized_bucket_count;
    snapshot.command_region.tail_no_work.count += normalized_bucket_count;
  }
  snapshot.command_region.tail_no_work.first_start_host_time_ns =
      tail_no_work_first_start_host_time_ns;
  snapshot.command_region.tail_no_work.last_end_host_time_ns =
      tail_no_work_last_end_host_time_ns;
  snapshot.command_region.tail_no_work.time_sums.start_offset_ns =
      tail_no_work_start_offset_ns_sum;
  snapshot.command_region.tail_no_work.time_sums.drain_duration_ns =
      tail_no_work_drain_duration_ns_sum;
  snapshot.command_region.retention.keep_active_count =
      keep_active_count > 0 ? (uint32_t)keep_active_count : 0;
  snapshot.command_region.retention.publish_keep_active_count =
      publish_keep_active_count > 0 ? (uint32_t)publish_keep_active_count : 0;
  snapshot.command_region.retention.keep_warm_count =
      keep_warm_count > 0 ? (uint32_t)keep_warm_count : 0;
  return snapshot;
}

static void iree_hal_cmd_block_processor_profile_begin_region(
    iree_hal_cmd_block_processor_context_t* context, uint32_t tile_count) {
  if (!iree_hal_cmd_block_processor_profile_records_regions(context)) return;
  iree_atomic_store(&context->profile.command_region.start_host_time_ns,
                    iree_time_now(), iree_memory_order_relaxed);
  iree_atomic_store(&context->profile.command_region.tile_count,
                    (int32_t)tile_count, iree_memory_order_relaxed);
  iree_atomic_store(&context->profile.command_region.useful_drain_count, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&context->profile.command_region.no_work_drain_count, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(
      &context->profile.command_region.tail_no_work.remaining_tiles.min, 0,
      iree_memory_order_relaxed);
  iree_atomic_store(
      &context->profile.command_region.tail_no_work.remaining_tiles.max, 0,
      iree_memory_order_relaxed);
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(context->profile.command_region.tail_no_work
                              .remaining_tiles.bucket_counts);
       ++i) {
    iree_atomic_store(&context->profile.command_region.tail_no_work
                           .remaining_tiles.bucket_counts[i],
                      0, iree_memory_order_relaxed);
  }
  iree_atomic_store(
      &context->profile.command_region.first_useful_drain_start_host_time_ns, 0,
      iree_memory_order_relaxed);
  iree_atomic_store(
      &context->profile.command_region.last_useful_drain_end_host_time_ns, 0,
      iree_memory_order_relaxed);
  iree_atomic_store(
      &context->profile.command_region.tail_no_work.first_start_host_time_ns, 0,
      iree_memory_order_relaxed);
  iree_atomic_store(
      &context->profile.command_region.tail_no_work.last_end_host_time_ns, 0,
      iree_memory_order_relaxed);
  iree_atomic_store(
      &context->profile.command_region.tail_no_work.time_sums.start_offset_ns,
      0, iree_memory_order_relaxed);
  iree_atomic_store(
      &context->profile.command_region.tail_no_work.time_sums.drain_duration_ns,
      0, iree_memory_order_relaxed);
  iree_atomic_store(
      &context->profile.command_region.retention.keep_active_count, 0,
      iree_memory_order_relaxed);
  iree_atomic_store(
      &context->profile.command_region.retention.publish_keep_active_count, 0,
      iree_memory_order_relaxed);
  iree_atomic_store(&context->profile.command_region.retention.keep_warm_count,
                    0, iree_memory_order_relaxed);
}

static void iree_hal_cmd_block_processor_profile_record_drain(
    iree_hal_cmd_block_processor_context_t* context, uint32_t tile_count,
    uint32_t remaining_tile_count, iree_time_t start_host_time_ns,
    iree_time_t end_host_time_ns) {
  if (tile_count == 0) {
    const uint32_t bucket_index =
        iree_hal_cmd_block_processor_profile_remaining_tile_bucket_index(
            remaining_tile_count);
    iree_hal_cmd_block_processor_profile_atomic_min_i32(
        &context->profile.command_region.tail_no_work.remaining_tiles.min,
        (int32_t)remaining_tile_count);
    iree_hal_cmd_block_processor_profile_atomic_max_i32(
        &context->profile.command_region.tail_no_work.remaining_tiles.max,
        (int32_t)remaining_tile_count);
    iree_atomic_fetch_add(&context->profile.command_region.tail_no_work
                               .remaining_tiles.bucket_counts[bucket_index],
                          1, iree_memory_order_relaxed);
    iree_hal_cmd_block_processor_profile_atomic_min_i64(
        &context->profile.command_region.tail_no_work.first_start_host_time_ns,
        start_host_time_ns);
    iree_hal_cmd_block_processor_profile_atomic_max_i64(
        &context->profile.command_region.tail_no_work.last_end_host_time_ns,
        end_host_time_ns);
    const iree_time_t region_start_host_time_ns =
        iree_atomic_load(&context->profile.command_region.start_host_time_ns,
                         iree_memory_order_relaxed);
    const iree_time_t start_offset_ns =
        start_host_time_ns > region_start_host_time_ns
            ? start_host_time_ns - region_start_host_time_ns
            : 0;
    const iree_time_t drain_duration_ns =
        end_host_time_ns > start_host_time_ns
            ? end_host_time_ns - start_host_time_ns
            : 0;
    iree_atomic_fetch_add(
        &context->profile.command_region.tail_no_work.time_sums.start_offset_ns,
        start_offset_ns, iree_memory_order_relaxed);
    iree_atomic_fetch_add(&context->profile.command_region.tail_no_work
                               .time_sums.drain_duration_ns,
                          drain_duration_ns, iree_memory_order_relaxed);
    return;
  }
  iree_atomic_fetch_add(&context->profile.command_region.useful_drain_count, 1,
                        iree_memory_order_relaxed);
  iree_hal_cmd_block_processor_profile_atomic_min_i64(
      &context->profile.command_region.first_useful_drain_start_host_time_ns,
      start_host_time_ns);
  iree_hal_cmd_block_processor_profile_atomic_max_i64(
      &context->profile.command_region.last_useful_drain_end_host_time_ns,
      end_host_time_ns);
}

//===----------------------------------------------------------------------===//
// Per-command tile execution
//===----------------------------------------------------------------------===//

// Executes a single tile of a dispatch command. Dispatches through the native
// function pointer when available (the fast path), or falls back to the
// executable's issue_call vtable for VM-based backends (VMVX, JIT, etc.).
static inline iree_status_t iree_hal_cmd_execute_dispatch_tile(
    const iree_hal_cmd_dispatch_t* dispatch,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    iree_hal_executable_workgroup_state_v0_t* workgroup_state,
    uint32_t worker_id) {
  if (IREE_LIKELY(dispatch->function)) {
    int ret = dispatch->function(&dispatch->executable->environment,
                                 dispatch_state, workgroup_state);
    if (IREE_UNLIKELY(ret != 0)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "dispatch kernel returned non-zero (%d)", ret);
    }
    return iree_ok_status();
  }
  return iree_hal_local_executable_issue_call(
      dispatch->executable, dispatch->export_ordinal, dispatch_state,
      workgroup_state, worker_id);
}

static inline void iree_hal_cmd_dispatch_initialize_workgroup_state(
    uint32_t tile, const uint32_t workgroup_count[3], uint32_t xy_count,
    const iree_hal_cmd_block_processor_worker_context_t* worker_context,
    uint32_t local_memory_size,
    iree_hal_executable_workgroup_state_v0_t* out_workgroup_state) {
  memset(out_workgroup_state, 0, sizeof(*out_workgroup_state));
  out_workgroup_state->workgroup_id_x = tile % workgroup_count[0];
  out_workgroup_state->workgroup_id_y =
      (tile / workgroup_count[0]) % workgroup_count[1];
  out_workgroup_state->workgroup_id_z = (uint16_t)(tile / xy_count);
  out_workgroup_state->processor_id = worker_context->worker_index;
  out_workgroup_state->local_memory =
      local_memory_size > 0 ? worker_context->local_memory.data : NULL;
  out_workgroup_state->local_memory_size = local_memory_size;
}

static uint32_t iree_hal_cmd_dispatch_tiles_per_reservation(
    uint32_t tile_count, uint32_t worker_count, uint32_t explicit_value) {
  if (explicit_value != 0) return explicit_value;
  if (tile_count <
      worker_count * IREE_HAL_CMD_DISPATCH_MAX_TILES_PER_RESERVATION) {
    return 1;
  }
  return IREE_HAL_CMD_DISPATCH_MAX_TILES_PER_RESERVATION;
}

// Executes tiles for a DISPATCH command. Claims tiles by CAS-advancing the
// command's shared epoch-tagged tile_index entry, executes each tile by calling
// the kernel function, and returns the total number of tiles completed.
//
// For single-worker mode (worker_count==1): the tile_index atomic is not
// touched (unnecessary overhead). All tiles are executed sequentially.
static iree_status_t iree_hal_cmd_execute_dispatch_tiles(
    const iree_hal_cmd_dispatch_t* dispatch, void** binding_ptrs,
    size_t* binding_lengths, iree_atomic_int64_t* tile_index,
    int32_t region_epoch,
    const iree_hal_cmd_block_processor_worker_context_t* worker_context,
    uint32_t worker_count, uint32_t* out_tiles_completed) {
  *out_tiles_completed = 0;

  uint32_t workgroup_count[3];
  iree_hal_cmd_dispatch_read_workgroup_count(dispatch, binding_ptrs,
                                             workgroup_count);

  const uint32_t tile_count =
      workgroup_count[0] * workgroup_count[1] * workgroup_count[2];
  if (tile_count == 0) return iree_ok_status();
  if (IREE_UNLIKELY(dispatch->local_memory_size >
                    worker_context->local_memory.data_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dispatch requires %ub of local memory but only "
                            "%" PRIhsz "b is available per-worker",
                            dispatch->local_memory_size,
                            worker_context->local_memory.data_length);
  }
  if (IREE_UNLIKELY(worker_context->local_memory.data_length > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "worker local memory size %" PRIhsz
                            " exceeds the dispatch ABI maximum",
                            worker_context->local_memory.data_length);
  }

  IREE_TRACE(iree_string_view_t trace_name =
                 iree_hal_local_executable_export_name(
                     dispatch->executable, dispatch->export_ordinal));
  IREE_TRACE(if (iree_string_view_is_empty(trace_name)) {
    trace_name = iree_make_cstring_view("iree_hal_local_task_dispatch");
  });

  // Build the dispatch state on the stack. Every pointer field is a direct
  // assignment into .text or .data — no layout computation, no memcpy.
  iree_hal_executable_dispatch_state_v0_t dispatch_state;
  memset(&dispatch_state, 0, sizeof(dispatch_state));
  dispatch_state.workgroup_size_x = dispatch->workgroup_size[0];
  dispatch_state.workgroup_size_y = dispatch->workgroup_size[1];
  dispatch_state.workgroup_size_z = (uint16_t)dispatch->workgroup_size[2];
  dispatch_state.workgroup_count_x = workgroup_count[0];
  dispatch_state.workgroup_count_y = workgroup_count[1];
  dispatch_state.workgroup_count_z = (uint16_t)workgroup_count[2];
  dispatch_state.constant_count = (uint16_t)dispatch->constant_count;
  dispatch_state.constants = dispatch->constants;
  dispatch_state.max_concurrency = (uint8_t)worker_count;
  dispatch_state.binding_count = dispatch->binding_count;
  dispatch_state.binding_ptrs =
      (void* const*)&binding_ptrs[dispatch->binding_data_base];
  dispatch_state.binding_lengths =
      (const size_t*)&binding_lengths[dispatch->binding_data_base];

  const uint32_t xy_count = workgroup_count[0] * workgroup_count[1];
  const uint32_t tiles_per_reservation =
      iree_hal_cmd_dispatch_tiles_per_reservation(
          tile_count, worker_count, dispatch->tiles_per_reservation);

  if (worker_count == 1) {
    IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(z_dispatch, trace_name.data,
                                        trace_name.size);

    // Single-worker fast path: no tile claiming atomics needed.
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
      iree_hal_executable_workgroup_state_v0_t workgroup_state;
      iree_hal_cmd_dispatch_initialize_workgroup_state(
          tile, workgroup_count, xy_count, worker_context,
          dispatch->local_memory_size, &workgroup_state);
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z_dispatch, iree_hal_cmd_execute_dispatch_tile(
                          dispatch, &dispatch_state, &workgroup_state,
                          worker_context->worker_index));
    }
    *out_tiles_completed = tile_count;
    IREE_TRACE_ZONE_END(z_dispatch);
  } else {
    // Multi-worker: claim tiles by advancing the epoch-tagged shared tile
    // counter. Stale workers from previous regions fail the epoch check instead
    // of claiming from a reset counter.
    uint32_t completed = 0;
    while (true) {
      int64_t current = iree_atomic_load(tile_index, iree_memory_order_relaxed);
      uint32_t counter = 0;
      uint32_t new_counter = 0;
      while (true) {
        if ((current >> 32) != region_epoch) {
          *out_tiles_completed = completed;
          return iree_ok_status();
        }
        counter = (uint32_t)current;
        if (counter >= tile_count) {
          *out_tiles_completed = completed;
          return iree_ok_status();
        }
        new_counter = counter + tiles_per_reservation;
        if (new_counter > tile_count) new_counter = tile_count;
        const int64_t desired =
            ((int64_t)region_epoch << 32) | (int64_t)new_counter;
        if (iree_atomic_compare_exchange_weak(tile_index, &current, desired,
                                              iree_memory_order_relaxed,
                                              iree_memory_order_relaxed)) {
          break;
        }
      }
      if (counter >= tile_count) break;

      IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(z_dispatch, trace_name.data,
                                          trace_name.size);

      for (uint32_t tile = counter; tile < new_counter; ++tile) {
        iree_hal_executable_workgroup_state_v0_t workgroup_state;
        iree_hal_cmd_dispatch_initialize_workgroup_state(
            tile, workgroup_count, xy_count, worker_context,
            dispatch->local_memory_size, &workgroup_state);
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z_dispatch, iree_hal_cmd_execute_dispatch_tile(
                            dispatch, &dispatch_state, &workgroup_state,
                            worker_context->worker_index));
        ++completed;
      }
      IREE_TRACE_ZONE_END(z_dispatch);
    }
    *out_tiles_completed = completed;
  }

  return iree_ok_status();
}

static bool iree_hal_cmd_transfer_claim_tile(iree_atomic_int64_t* tile_index,
                                             int32_t region_epoch,
                                             uint32_t tile_count,
                                             uint32_t* out_tile) {
  int64_t current = iree_atomic_load(tile_index, iree_memory_order_relaxed);
  while (true) {
    if ((current >> 32) != region_epoch) return false;
    const uint32_t tile = (uint32_t)current;
    if (tile >= tile_count) return false;
    const int64_t desired = ((int64_t)region_epoch << 32) | (int64_t)(tile + 1);
    if (iree_atomic_compare_exchange_weak(tile_index, &current, desired,
                                          iree_memory_order_relaxed,
                                          iree_memory_order_relaxed)) {
      *out_tile = tile;
      return true;
    }
  }
}

static void iree_hal_cmd_block_processor_profile_accumulate_dispatch(
    iree_hal_cmd_block_processor_context_t* context,
    const iree_hal_cmd_dispatch_t* dispatch, iree_status_t status,
    iree_time_t start_host_time_ns, iree_time_t end_host_time_ns,
    uint32_t tile_count) {
  iree_hal_cmd_block_processor_profile_dispatch_t* dispatch_profile =
      &context->profile.dispatches[dispatch->header.dispatch_index];
  iree_hal_cmd_block_processor_profile_atomic_min_i64(
      &dispatch_profile->start_host_time_ns, start_host_time_ns);
  iree_hal_cmd_block_processor_profile_atomic_max_i64(
      &dispatch_profile->end_host_time_ns, end_host_time_ns);
  iree_atomic_fetch_add(&dispatch_profile->tile_count, (int64_t)tile_count,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&dispatch_profile->tile_duration_sum_ns,
                        (int64_t)(end_host_time_ns - start_host_time_ns),
                        iree_memory_order_relaxed);
  const iree_status_code_t status_code = iree_status_code(status);
  if (status_code != IREE_STATUS_OK) {
    int32_t expected_status_code = IREE_STATUS_OK;
    iree_atomic_compare_exchange_strong(
        &dispatch_profile->status_code, &expected_status_code,
        (int32_t)status_code, iree_memory_order_relaxed,
        iree_memory_order_relaxed);
  }
}

static void iree_hal_cmd_block_processor_profile_make_dispatch_event(
    iree_hal_cmd_block_processor_context_t* context,
    const iree_hal_cmd_dispatch_t* dispatch, void** binding_ptrs,
    uint64_t tile_count, int64_t tile_duration_sum_ns,
    iree_status_code_t status_code, iree_time_t start_host_time_ns,
    iree_time_t end_host_time_ns,
    iree_hal_local_profile_host_execution_event_info_t* out_event_info) {
  uint32_t workgroup_count[3];
  iree_hal_cmd_dispatch_read_workgroup_count(dispatch, binding_ptrs,
                                             workgroup_count);

  iree_hal_local_profile_host_execution_event_info_t event_info =
      iree_hal_local_profile_host_execution_event_info_default();
  event_info.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
  event_info.flags = IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_COMMAND_BUFFER |
                     IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_DEFERRED;
  if (iree_any_bit_set(dispatch->header.flags, IREE_HAL_CMD_FLAG_INDIRECT)) {
    event_info.flags |=
        IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_INDIRECT_PARAMETERS;
  }
  event_info.status_code = status_code;
  event_info.scope = context->profile.scope;
  event_info.submission_id = context->profile.submission_id;
  event_info.command_buffer_id = context->profile.command_buffer_id;
  event_info.executable_id =
      iree_hal_local_executable_profile_id(dispatch->executable);
  event_info.command_index = dispatch->profile.command_index;
  event_info.export_ordinal = dispatch->export_ordinal;
  memcpy(event_info.workgroup_count, workgroup_count,
         sizeof(event_info.workgroup_count));
  event_info.workgroup_size[0] =
      dispatch->workgroup_size[0] ? dispatch->workgroup_size[0] : 1;
  event_info.workgroup_size[1] =
      dispatch->workgroup_size[1] ? dispatch->workgroup_size[1] : 1;
  event_info.workgroup_size[2] =
      dispatch->workgroup_size[2] ? dispatch->workgroup_size[2] : 1;
  event_info.start_host_time_ns = start_host_time_ns;
  event_info.end_host_time_ns = end_host_time_ns;
  event_info.tile_count = tile_count;
  event_info.tile_duration_sum_ns = tile_duration_sum_ns;
  event_info.operation_count = 1;
  *out_event_info = event_info;
}

static void iree_hal_cmd_block_processor_profile_append_dispatch_event(
    iree_hal_cmd_block_processor_context_t* context,
    const iree_hal_cmd_dispatch_t* dispatch, void** binding_ptrs,
    uint64_t tile_count, int64_t tile_duration_sum_ns,
    iree_status_code_t status_code, iree_time_t start_host_time_ns,
    iree_time_t end_host_time_ns) {
  iree_hal_local_profile_host_execution_event_info_t event_info;
  iree_hal_cmd_block_processor_profile_make_dispatch_event(
      context, dispatch, binding_ptrs, tile_count, tile_duration_sum_ns,
      status_code, start_host_time_ns, end_host_time_ns, &event_info);
  iree_hal_local_profile_recorder_append_host_execution_event(
      context->profile.recorder, &event_info, /*out_event_id=*/NULL);
}

static iree_status_t iree_hal_cmd_execute_dispatch_tiles_profiled(
    iree_hal_cmd_block_processor_context_t* context,
    const iree_hal_cmd_dispatch_t* dispatch, void** binding_ptrs,
    size_t* binding_lengths, iree_atomic_int64_t* tile_index,
    int32_t region_epoch,
    const iree_hal_cmd_block_processor_worker_context_t* worker_context,
    uint32_t worker_count, uint32_t* out_tiles_completed) {
  *out_tiles_completed = 0;

  iree_hal_local_profile_recorder_t* recorder = context->profile.recorder;
  const bool profile_host_execution =
      iree_hal_local_profile_recorder_is_enabled(
          recorder, IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS);
  const bool aggregate_host_execution = profile_host_execution &&
                                        context->worker_count > 1 &&
                                        context->profile.dispatches != NULL;
  if (IREE_UNLIKELY(aggregate_host_execution &&
                    dispatch->header.dispatch_index >=
                        context->profile.dispatch_capacity)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "dispatch index %u exceeds profile dispatch capacity %zu",
        dispatch->header.dispatch_index,
        (size_t)context->profile.dispatch_capacity);
  }

  const iree_time_t start_host_time_ns =
      profile_host_execution ? iree_time_now() : 0;
  iree_status_t status = iree_hal_cmd_execute_dispatch_tiles(
      dispatch, binding_ptrs, binding_lengths, tile_index, region_epoch,
      worker_context, worker_count, out_tiles_completed);
  const iree_time_t end_host_time_ns =
      profile_host_execution ? iree_time_now() : 0;

  if (IREE_UNLIKELY(profile_host_execution && (*out_tiles_completed != 0 ||
                                               !iree_status_is_ok(status)))) {
    if (aggregate_host_execution) {
      iree_hal_cmd_block_processor_profile_accumulate_dispatch(
          context, dispatch, status, start_host_time_ns, end_host_time_ns,
          *out_tiles_completed);
    } else {
      iree_hal_cmd_block_processor_profile_append_dispatch_event(
          context, dispatch, binding_ptrs, *out_tiles_completed,
          (int64_t)(end_host_time_ns - start_host_time_ns),
          iree_status_code(status), start_host_time_ns, end_host_time_ns);
    }
  }

  return status;
}

static iree_host_size_t iree_hal_cmd_block_processor_profile_snapshot_region(
    iree_hal_cmd_block_processor_context_t* context,
    const iree_hal_cmd_barrier_t* barrier, void** binding_ptrs,
    iree_hal_local_profile_host_execution_event_info_t* out_events,
    iree_host_size_t event_capacity) {
  if (!context->profile.dispatches ||
      !iree_hal_local_profile_recorder_is_enabled(
          context->profile.recorder,
          IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS)) {
    return 0;
  }

  iree_host_size_t event_count = 0;
  const iree_hal_cmd_header_t* cmd = iree_hal_cmd_next(&barrier->header);
  for (uint8_t d = 0; d < barrier->dispatch_count; ++d) {
    if (cmd->opcode == IREE_HAL_CMD_DISPATCH) {
      const iree_hal_cmd_dispatch_t* dispatch =
          (const iree_hal_cmd_dispatch_t*)cmd;
      IREE_ASSERT(dispatch->header.dispatch_index <
                  context->profile.dispatch_capacity);
      if (IREE_LIKELY(dispatch->header.dispatch_index <
                      context->profile.dispatch_capacity)) {
        iree_hal_cmd_block_processor_profile_dispatch_t* dispatch_profile =
            &context->profile.dispatches[dispatch->header.dispatch_index];
        const int64_t tile_count = iree_atomic_load(
            &dispatch_profile->tile_count, iree_memory_order_relaxed);
        const int32_t status_code = iree_atomic_load(
            &dispatch_profile->status_code, iree_memory_order_relaxed);
        if (tile_count != 0 || status_code != IREE_STATUS_OK) {
          const iree_time_t start_host_time_ns = iree_atomic_load(
              &dispatch_profile->start_host_time_ns, iree_memory_order_relaxed);
          const iree_time_t end_host_time_ns = iree_atomic_load(
              &dispatch_profile->end_host_time_ns, iree_memory_order_relaxed);
          const int64_t tile_duration_sum_ns =
              iree_atomic_load(&dispatch_profile->tile_duration_sum_ns,
                               iree_memory_order_relaxed);
          if (IREE_LIKELY(event_count < event_capacity)) {
            iree_hal_cmd_block_processor_profile_make_dispatch_event(
                context, dispatch, binding_ptrs, (uint64_t)tile_count,
                tile_duration_sum_ns, (iree_status_code_t)status_code,
                start_host_time_ns, end_host_time_ns,
                &out_events[event_count++]);
          }
        }
        iree_hal_cmd_block_processor_profile_reset_dispatch(dispatch_profile);
      }
    }
    cmd = iree_hal_cmd_next(cmd);
  }
  return event_count;
}

static void iree_hal_cmd_block_processor_profile_append_host_execution_events(
    iree_hal_cmd_block_processor_context_t* context,
    const iree_hal_local_profile_host_execution_event_info_t* events,
    iree_host_size_t event_count) {
  for (iree_host_size_t i = 0; i < event_count; ++i) {
    iree_hal_local_profile_recorder_append_host_execution_event(
        context->profile.recorder, &events[i], /*out_event_id=*/NULL);
  }
}

static void iree_hal_cmd_execute_fill_tile(const iree_hal_cmd_fill_t* fill,
                                           uint8_t* target,
                                           const iree_hal_fill_params_t* params,
                                           uint32_t tile) {
  const iree_device_size_t tile_offset =
      iree_hal_cmd_transfer_tile_offset(tile);
  const iree_device_size_t tile_length =
      iree_hal_cmd_transfer_tile_length(params->length, tile);
  iree_hal_cmd_fill_pattern(target + params->target_offset + tile_offset,
                            tile_length, params->pattern, fill->pattern_length);
}

// Executes a FILL command by claiming transfer tiles with the same epoch-tagged
// CAS machinery used for dispatch workgroups.
static uint32_t iree_hal_cmd_execute_fill(const iree_hal_cmd_fill_t* fill,
                                          void** binding_ptrs,
                                          iree_atomic_int64_t* tile_index,
                                          int32_t region_epoch,
                                          uint32_t worker_count) {
  uint8_t* target = (uint8_t*)binding_ptrs[fill->target_binding];
  const iree_hal_fill_params_t params =
      iree_hal_cmd_fill_read_params(fill, binding_ptrs);
  const uint32_t tile_count = iree_hal_cmd_transfer_tile_count(params.length);
  if (tile_count == 0) return 0;

  uint32_t completed = 0;
  if (worker_count == 1) {
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
      iree_hal_cmd_execute_fill_tile(fill, target, &params, tile);
    }
    completed = tile_count;
  } else {
    uint32_t tile = 0;
    while (iree_hal_cmd_transfer_claim_tile(tile_index, region_epoch,
                                            tile_count, &tile)) {
      iree_hal_cmd_execute_fill_tile(fill, target, &params, tile);
      ++completed;
    }
  }
  return completed;
}

static void iree_hal_cmd_execute_copy_tile(const uint8_t* source,
                                           uint8_t* target,
                                           const iree_hal_copy_params_t* params,
                                           uint32_t tile) {
  const iree_device_size_t tile_offset =
      iree_hal_cmd_transfer_tile_offset(tile);
  const iree_device_size_t tile_length =
      iree_hal_cmd_transfer_tile_length(params->length, tile);
  memcpy(target + params->target_offset + tile_offset,
         source + params->source_offset + tile_offset, (size_t)tile_length);
}

// Executes a COPY command by claiming transfer tiles with the same epoch-tagged
// CAS machinery used for dispatch workgroups.
static uint32_t iree_hal_cmd_execute_copy(const iree_hal_cmd_copy_t* copy,
                                          void** binding_ptrs,
                                          iree_atomic_int64_t* tile_index,
                                          int32_t region_epoch,
                                          uint32_t worker_count) {
  const uint8_t* source = (const uint8_t*)binding_ptrs[copy->source_binding];
  uint8_t* target = (uint8_t*)binding_ptrs[copy->target_binding];
  const iree_hal_copy_params_t params =
      iree_hal_cmd_copy_read_params(copy, binding_ptrs);
  const uint32_t tile_count = iree_hal_cmd_transfer_tile_count(params.length);
  if (tile_count == 0) return 0;

  uint32_t completed = 0;
  if (worker_count == 1) {
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
      iree_hal_cmd_execute_copy_tile(source, target, &params, tile);
    }
    completed = tile_count;
  } else {
    uint32_t tile = 0;
    while (iree_hal_cmd_transfer_claim_tile(tile_index, region_epoch,
                                            tile_count, &tile)) {
      iree_hal_cmd_execute_copy_tile(source, target, &params, tile);
      ++completed;
    }
  }
  return completed;
}

// Executes an UPDATE command. Copies inline host data from .text to a device
// buffer. UPDATE commands are at most one transfer tile because the inline
// source payload is capped by the block ISA command size.
static uint32_t iree_hal_cmd_execute_update(const iree_hal_cmd_update_t* update,
                                            void** binding_ptrs,
                                            iree_atomic_int64_t* tile_index,
                                            int32_t region_epoch,
                                            uint32_t worker_count) {
  const uint32_t tile_count = iree_hal_cmd_transfer_tile_count(update->length);
  if (tile_count == 0) return 0;
  if (worker_count > 1) {
    int64_t current = iree_atomic_load(tile_index, iree_memory_order_relaxed);
    while (true) {
      if ((current >> 32) != region_epoch) return 0;
      const uint32_t tile = (uint32_t)current;
      if (tile != 0) return 0;
      const int64_t desired =
          ((int64_t)region_epoch << 32) | (int64_t)tile_count;
      if (iree_atomic_compare_exchange_weak(tile_index, &current, desired,
                                            iree_memory_order_relaxed,
                                            iree_memory_order_relaxed)) {
        break;
      }
    }
  }

  uint8_t* target = (uint8_t*)binding_ptrs[update->target_binding];
  memcpy(target + update->target_offset, update->source_data,
         (size_t)update->length);

  return tile_count;
}

//===----------------------------------------------------------------------===//
// Block and region processing
//===----------------------------------------------------------------------===//

// Processes one barrier-delimited region cooperatively. Each worker scans the
// dispatch_count commands following the barrier and claims tiles from each.
// Commands in an open synchronization scope can make progress concurrently.
//
// Tile claiming validates the active region epoch in each tile_index entry. The
// region completer resets the region-local entries before publishing the next
// epoch, so stale workers from an older region cannot mutate the next region's
// counters.
//
// On kernel error, reports the error to the context and returns immediately.
// Other workers will see the error flag and exit their loops.
static uint32_t iree_hal_cmd_block_processor_process_region(
    const iree_hal_cmd_barrier_t* barrier, int32_t region_epoch,
    const iree_hal_cmd_block_processor_worker_context_t* worker_context,
    iree_hal_cmd_block_processor_context_t* context,
    iree_hal_cmd_block_state_t* state) {
  IREE_TRACE_ZONE_BEGIN_NAMED(z_region, "iree_hal_local_task_process_region");

  uint32_t tiles_completed = 0;
  void** binding_ptrs = iree_hal_cmd_block_state_binding_ptrs(
      state, context->max_region_dispatch_count);
  size_t* binding_lengths = iree_hal_cmd_block_state_binding_lengths(
      state, context->max_region_dispatch_count,
      context->max_total_binding_count);

  const uint8_t dispatch_count = barrier->dispatch_count;
  if (dispatch_count == 0) {
    IREE_TRACE_ZONE_END(z_region);
    return 0;
  }

  const iree_hal_cmd_header_t* cmd = iree_hal_cmd_next(&barrier->header);
  for (uint8_t d = 0; d < dispatch_count; ++d) {
    if (IREE_UNLIKELY(iree_hal_cmd_block_processor_has_error(context))) break;

    iree_atomic_int64_t* tile_idx =
        iree_hal_cmd_block_state_tile_index(state, cmd->dispatch_index);

    switch (cmd->opcode) {
      case IREE_HAL_CMD_DISPATCH: {
        uint32_t dispatch_tiles = 0;
        iree_status_t status = iree_ok_status();
        if (IREE_UNLIKELY(context->profile.recorder != NULL)) {
          status = iree_hal_cmd_execute_dispatch_tiles_profiled(
              context, (const iree_hal_cmd_dispatch_t*)cmd, binding_ptrs,
              binding_lengths, tile_idx, region_epoch, worker_context,
              context->worker_count, &dispatch_tiles);
        } else {
          status = iree_hal_cmd_execute_dispatch_tiles(
              (const iree_hal_cmd_dispatch_t*)cmd, binding_ptrs,
              binding_lengths, tile_idx, region_epoch, worker_context,
              context->worker_count, &dispatch_tiles);
        }
        if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
          iree_hal_cmd_block_processor_report_error(context, status);
          IREE_TRACE_ZONE_END(z_region);
          return tiles_completed;
        }
        tiles_completed += dispatch_tiles;
        break;
      }
      case IREE_HAL_CMD_FILL: {
        tiles_completed += iree_hal_cmd_execute_fill(
            (const iree_hal_cmd_fill_t*)cmd, binding_ptrs, tile_idx,
            region_epoch, context->worker_count);
        break;
      }
      case IREE_HAL_CMD_COPY: {
        tiles_completed += iree_hal_cmd_execute_copy(
            (const iree_hal_cmd_copy_t*)cmd, binding_ptrs, tile_idx,
            region_epoch, context->worker_count);
        break;
      }
      case IREE_HAL_CMD_UPDATE: {
        tiles_completed += iree_hal_cmd_execute_update(
            (const iree_hal_cmd_update_t*)cmd, binding_ptrs, tile_idx,
            region_epoch, context->worker_count);
        break;
      }
      default: {
        iree_hal_cmd_block_processor_report_error(
            context,
            iree_make_status(IREE_STATUS_INTERNAL,
                             "unexpected opcode %d in region", cmd->opcode));
        break;
      }
    }

    cmd = iree_hal_cmd_next(cmd);
  }

  IREE_TRACE_ZONE_END(z_region);
  return tiles_completed;
}

static int32_t iree_hal_cmd_block_processor_calculate_wake_budget(
    const iree_hal_cmd_block_processor_context_t* context,
    uint32_t remaining_tiles) {
  uint32_t wake_budget = remaining_tiles;
  if (wake_budget == 0) wake_budget = 1;
  if (wake_budget > context->worker_count) wake_budget = context->worker_count;
  return (int32_t)wake_budget;
}

static bool iree_hal_cmd_block_processor_region_prefers_warm_spin(
    const iree_hal_cmd_block_header_t* block, int32_t region_index) {
  if (region_index < 0 || region_index >= (int32_t)block->region_count) {
    return false;
  }
  const iree_hal_cmd_region_summary_t* summary =
      &iree_hal_cmd_block_region_summaries(block)[region_index];
  if (summary->next_candidate_region == IREE_HAL_CMD_REGION_INDEX_NONE ||
      summary->next_candidate_region >= block->region_count) {
    return false;
  }
  const iree_hal_cmd_region_summary_t* next_summary =
      &iree_hal_cmd_block_region_summaries(
          block)[summary->next_candidate_region];
  return next_summary->width_bucket >
         IREE_HAL_CMD_BLOCK_PROCESSOR_WARM_SPIN_LOOKAHEAD_WIDTH;
}

static int32_t iree_hal_cmd_block_processor_region_warm_retainer_limit(
    const iree_hal_cmd_block_processor_context_t* context,
    const iree_hal_cmd_block_header_t* block, int32_t region_index) {
  if (region_index < 0 || region_index >= (int32_t)block->region_count) {
    return 0;
  }
  const iree_hal_cmd_region_summary_t* summary =
      &iree_hal_cmd_block_region_summaries(block)[region_index];
  if (summary->lookahead_width_bucket <=
      IREE_HAL_CMD_BLOCK_PROCESSOR_WARM_SPIN_LOOKAHEAD_WIDTH) {
    return 0;
  }
  return iree_min((int32_t)context->worker_count,
                  (int32_t)summary->lookahead_width_bucket);
}

// Finds the first executable region at or after |start_region_index| using the
// immutable region summary table for navigation. Only potentially active
// candidate regions require dynamic tile-count evaluation; definitely empty
// regions are skipped by indexed summary links instead of command-stream scans.
static int32_t iree_hal_cmd_block_processor_find_active_region(
    const iree_hal_cmd_block_header_t* block, void** binding_ptrs,
    uint16_t start_region_index, uint32_t* out_remaining_tiles) {
  const iree_hal_cmd_region_summary_t* summaries =
      iree_hal_cmd_block_region_summaries(block);
  uint16_t region_index = start_region_index;
  while (region_index < block->region_count) {
    const iree_hal_cmd_region_summary_t* summary = &summaries[region_index];
    if (summary->width_bucket == 0) {
      region_index = summary->next_candidate_region;
      continue;
    }

    const iree_hal_cmd_barrier_t* barrier =
        iree_hal_cmd_block_region_barrier(block, region_index);
    const uint32_t remaining_tiles =
        iree_hal_cmd_region_tile_count(barrier, binding_ptrs);
    if (remaining_tiles != 0) {
      *out_remaining_tiles = remaining_tiles;
      return region_index;
    }

    region_index = summary->next_candidate_region;
  }

  *out_remaining_tiles = 0;
  return block->region_count;
}

// Initializes .data for a new block. Binding pointers are block-local state and
// multi-worker contexts use a fresh state slot for each block so stale workers
// from the previous block can finish without racing fixup writes here.
//
// region_state is set to the first non-empty region. Workers skip empty
// regions; the active region index only gates non-empty regions. If all regions
// are empty, the active region index is set to region_count and the completer
// must handle the block terminator.
//
// remaining_tiles is set to the first active region's total tiles so
// that workers can begin completer election immediately.
static void iree_hal_cmd_block_processor_init_block(
    const iree_hal_cmd_block_header_t* block,
    iree_hal_cmd_block_processor_context_t* context,
    iree_hal_cmd_block_state_t* state) {
  iree_hal_cmd_block_processor_profile_reset_dispatches(context);

  // Assign a unique global epoch for this block's first active region. Workers
  // acquire this to observe the freshly initialized .data, and no-work warm
  // retainers use it to detect advancement.
  int32_t block_epoch = context->next_epoch;
  context->next_epoch = block_epoch + 1;

  // Reset all tile counters. All slots are set because
  // max_region_dispatch_count covers the largest region in the block.
  for (uint16_t i = 0; i < context->max_region_dispatch_count; ++i) {
    iree_atomic_store(iree_hal_cmd_block_state_tile_index(state, i),
                      (int64_t)block_epoch << 32, iree_memory_order_relaxed);
  }

  // Resolve bindings before computing active regions: indirect dispatch
  // parameters are ordinary bindings and their tile counts are only known
  // after .data is populated.
  iree_hal_cmd_block_processor_resolve_bindings(
      block, state, context->max_region_dispatch_count,
      context->max_total_binding_count, context->binding_table,
      context->binding_table_length);
  void** binding_ptrs = iree_hal_cmd_block_state_binding_ptrs(
      state, context->max_region_dispatch_count);

  uint32_t first_remaining_tiles = 0;
  const int32_t first_active = iree_hal_cmd_block_processor_find_active_region(
      block, binding_ptrs, /*start_region_index=*/0, &first_remaining_tiles);

  {
    int64_t remaining_tagged =
        ((int64_t)block_epoch << 32) | (int64_t)first_remaining_tiles;
    iree_atomic_store(&state->remaining_tiles, remaining_tagged,
                      iree_memory_order_relaxed);
  }

  if (first_active < block->region_count) {
    iree_atomic_store(&context->current_wake_budget,
                      iree_hal_cmd_block_processor_calculate_wake_budget(
                          context, first_remaining_tiles),
                      iree_memory_order_relaxed);
    iree_hal_cmd_block_processor_profile_begin_region(context,
                                                      first_remaining_tiles);
  } else {
    iree_atomic_store(&context->current_wake_budget, 1,
                      iree_memory_order_relaxed);
  }
  iree_atomic_store(
      &state->region_state,
      iree_hal_cmd_block_region_state_pack(block_epoch, first_active),
      iree_memory_order_release);
}

// Prepares .data for the next region at a region transition. Called by the
// completer after remaining_tiles reaches zero, guaranteeing all tiles in
// the previous region have been executed.
//
// |region_epoch| is the global epoch for the new region (from
// context->next_epoch, pre-incremented by the caller). All tile_index entries
// are reset to (region_epoch << 32 | 0). All max_region_dispatch_count entries
// are set, not just the previous region's count, because the next region may
// have more dispatches.
//
// region_state is stored last with release semantics, publishing the new active
// region together with tile_indices and remaining_tiles. Workers acquire
// region_state to observe one consistent region identity.
static void iree_hal_cmd_block_processor_init_region(
    iree_hal_cmd_block_processor_context_t* context,
    iree_hal_cmd_block_state_t* state, int32_t region_epoch,
    int32_t active_region, uint32_t next_remaining_tiles) {
  // Reset tile counters for the new region.
  const int64_t epoch_shifted = (int64_t)region_epoch << 32;
  for (uint16_t i = 0; i < context->max_region_dispatch_count; ++i) {
    iree_atomic_store(iree_hal_cmd_block_state_tile_index(state, i),
                      epoch_shifted, iree_memory_order_relaxed);
  }
  // Set remaining tiles for the new region with the epoch sideband.
  int64_t remaining_tagged =
      ((int64_t)region_epoch << 32) | (int64_t)(uint32_t)next_remaining_tiles;
  iree_atomic_store(&state->remaining_tiles, remaining_tagged,
                    iree_memory_order_relaxed);
  iree_hal_cmd_block_processor_profile_begin_region(context,
                                                    next_remaining_tiles);
  iree_atomic_store(
      &state->region_state,
      iree_hal_cmd_block_region_state_pack(region_epoch, active_region),
      iree_memory_order_release);
}

//===----------------------------------------------------------------------===//
// Drain paths
//===----------------------------------------------------------------------===//

// Executes the entire recording on a single thread. No atomics, no spinning,
// no synchronization — just a straight-line interpreter. This is the path
// for inline execution and small command buffers.
static iree_status_t iree_hal_cmd_block_processor_execute_single_worker(
    iree_hal_cmd_block_processor_context_t* context,
    const iree_hal_cmd_block_processor_worker_context_t* worker_context,
    uint32_t* out_tiles_executed) {
  const iree_hal_cmd_block_header_t* block = context->recording->first_block;
  iree_hal_cmd_block_state_t* state = context->state_storage;

  while (block) {
    iree_hal_cmd_block_processor_init_block(block, context, state);

    void** binding_ptrs = iree_hal_cmd_block_state_binding_ptrs(
        state, context->max_region_dispatch_count);
    size_t* binding_lengths = iree_hal_cmd_block_state_binding_lengths(
        state, context->max_region_dispatch_count,
        context->max_total_binding_count);

    const iree_hal_cmd_header_t* cmd = iree_hal_cmd_block_commands(block);
    const uint8_t* stream_end = (const uint8_t*)cmd + block->used_bytes;
    bool branch_taken = false;

    while ((const uint8_t*)cmd < stream_end && !branch_taken) {
      switch (cmd->opcode) {
        case IREE_HAL_CMD_DISPATCH: {
          uint32_t tiles = 0;
          iree_status_t status = iree_ok_status();
          if (IREE_UNLIKELY(context->profile.recorder != NULL)) {
            status = iree_hal_cmd_execute_dispatch_tiles_profiled(
                context, (const iree_hal_cmd_dispatch_t*)cmd, binding_ptrs,
                binding_lengths, NULL, 0, worker_context, 1, &tiles);
          } else {
            status = iree_hal_cmd_execute_dispatch_tiles(
                (const iree_hal_cmd_dispatch_t*)cmd, binding_ptrs,
                binding_lengths, NULL, 0, worker_context, 1, &tiles);
          }
          *out_tiles_executed += tiles;
          IREE_RETURN_IF_ERROR(status);
          break;
        }
        case IREE_HAL_CMD_FILL: {
          *out_tiles_executed += iree_hal_cmd_execute_fill(
              (const iree_hal_cmd_fill_t*)cmd, binding_ptrs, NULL, 0, 1);
          break;
        }
        case IREE_HAL_CMD_COPY: {
          *out_tiles_executed += iree_hal_cmd_execute_copy(
              (const iree_hal_cmd_copy_t*)cmd, binding_ptrs, NULL, 0, 1);
          break;
        }
        case IREE_HAL_CMD_UPDATE: {
          *out_tiles_executed += iree_hal_cmd_execute_update(
              (const iree_hal_cmd_update_t*)cmd, binding_ptrs, NULL, 0, 1);
          break;
        }
        case IREE_HAL_CMD_BARRIER: {
          // Single-worker: barriers are no-ops (execution is sequential).
          break;
        }
        case IREE_HAL_CMD_BRANCH: {
          block = ((const iree_hal_cmd_branch_t*)cmd)->target;
          branch_taken = true;
          break;
        }
        case IREE_HAL_CMD_RETURN: {
          return iree_ok_status();
        }
        default: {
          return iree_make_status(
              IREE_STATUS_INTERNAL, "unknown opcode %d at stream offset %zu",
              cmd->opcode,
              (size_t)((const uint8_t*)cmd - (const uint8_t*)block));
        }
      }
      if (!branch_taken) cmd = iree_hal_cmd_next(cmd);
    }

    if (branch_taken) continue;

    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "block command stream ended without BRANCH or RETURN");
  }

  return iree_ok_status();
}

// N workers call drain() concurrently, each processing one region pass per
// call. The algorithm uses two levels of synchronization:
//   - Per-dispatch:  epoch-tagged tile_indices[] (CAS work stealing)
//   - Per-region:    packed region_state (epoch + active region) +
//                    epoch-tagged remaining_tiles (completer election)
//   - Per-block:     block_sequence + current_state (block transition
//                    publication)
//
// COMPLETER ELECTION
//
// After processing a region, each worker decrements remaining_tiles
// by the number of tiles it completed using an epoch-validating CAS. The
// worker whose decrement drives the count to zero becomes the completer.
// Workers with 0 tiles skip the election to avoid false positives.
//
// TILE CLAIMING
//
// Each tile_index is a 64-bit atomic counter padded to its own cache line.
// Workers claim tiles with CAS against (region_epoch << 32 | tile_index).
// A stale worker that races a region transition observes an epoch mismatch and
// leaves the next region's counter untouched.
//
// REGION TRANSITIONS
//
// Region transitions reuse the same block-state slot because binding_ptrs and
// binding_lengths are block-local and do not change across regions. The
// completer writes tile_indices and remaining_tiles, then publishes the new
// epoch and active region index together by storing region_state with release
// semantics. Workers acquire region_state as one value so they cannot combine
// an old epoch with a new region index.
//
// BLOCK TRANSITIONS (SEQLOCK)
//
// When all regions in a block are done, the completer handles the block
// terminator:
//   - BRANCH: uses a seqlock on block_sequence to protect init_block.
//     The completer increments block_sequence to even (in progress),
//     initializes the next block into a fresh state slot, publishes
//     current_block/current_state, then increments to odd (ready). Workers
//     check parity: even = bail immediately, odd = proceed. A re-check after
//     sampling current_block/current_state detects transitions that started
//     between the initial read and state access.
//   - RETURN: sets completed. All subsequent drain() calls return
//     completed=true.
//
// Per-block state slots are necessary because workers may still be executing
// code that reads old binding_ptrs/binding_lengths after the completer advances
// to the next block. The seqlock publishes which slot is current; the separate
// slots keep old block-local data stable for stale readers.
//
// Empty block chains (all regions empty) are followed iteratively by the
// completer while block_sequence remains even (in progress). Workers stay
// out until a block with work is found and the ready increment fires.
//
// DEADLOCK FREEDOM
//
// There are no internal waits. Non-completer workers simply return and the
// caller decides whether to yield, scan other contexts, or retry. Workers that
// observe an even block_sequence bail and retry through the caller.
//
// INVARIANTS
//
//   I1: The active region index is monotonically non-decreasing within a block.
//   I2: tile_indices, remaining_tiles, and the active region index are set
//       BEFORE region_state (the release point). Workers acquire region_state,
//       guaranteeing all prior writes are visible and that the epoch/index pair
//       is not torn across transitions.
//   I3: Empty regions are skipped by init_block and the completer.
//   I4: block_sequence uses seqlock parity: odd = ready, even = transition
//       in progress. Workers check parity on entry and re-check after
//       sampling current_block/current_state. All init_block writes are
//       published before the odd release increment.
//   I5: Multi-worker block transitions never reuse a block-state slot.

typedef struct iree_hal_cmd_block_processor_budget_update_t {
  // Previous wake budget observed before the transition.
  int32_t old_budget;

  // Wake budget selected for the next active region.
  int32_t new_budget;

  // Additional wake credits published for the executor wake tree.
  int32_t wake_delta;
} iree_hal_cmd_block_processor_budget_update_t;

// Updates the wake budget for a new region. Called by the completer at
// region/block transitions. If the new region needs more workers than are
// currently budgeted, adds wake credits so the relay mechanism brings up
// additional workers.
static iree_hal_cmd_block_processor_budget_update_t
iree_hal_cmd_block_processor_update_budget(
    iree_hal_cmd_block_processor_context_t* context,
    uint32_t next_remaining_tiles) {
  iree_hal_cmd_block_processor_budget_update_t update = {0, 0, 0};
  int32_t new_budget = iree_hal_cmd_block_processor_calculate_wake_budget(
      context, next_remaining_tiles);
  iree_atomic_store(&context->current_wake_budget, new_budget,
                    iree_memory_order_relaxed);
  if (!context->wake_budget_ptr) return update;
  int32_t old_budget = iree_atomic_exchange(
      context->wake_budget_ptr, new_budget, iree_memory_order_relaxed);
  update.old_budget = old_budget;
  update.new_budget = new_budget;
  if (new_budget > old_budget) {
    update.wake_delta = new_budget - old_budget;
  }
  return update;
}

static void iree_hal_cmd_block_processor_wake_additional_workers(
    iree_hal_cmd_block_processor_context_t* context, int32_t wake_delta) {
  if (wake_delta > 0 && context->wake_executor) {
    iree_task_executor_wake_workers(context->wake_executor, wake_delta);
  }
}

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
static void iree_hal_cmd_block_processor_trace_region_transition(void) {
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_local_task_region_transition");
  IREE_TRACE_ZONE_END(z0);
}
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

static void iree_hal_cmd_block_processor_profile_append_command_region_event(
    iree_hal_cmd_block_processor_context_t* context,
    const iree_hal_cmd_block_header_t* completed_block,
    uint32_t completed_block_sequence, uint32_t completed_region_epoch,
    int32_t completed_region_index, iree_time_t completed_region_end_host_time,
    iree_hal_cmd_block_processor_profile_region_snapshot_t region_snapshot,
    const iree_hal_cmd_block_header_t* next_block, int32_t next_region_index,
    uint32_t next_remaining_tiles,
    iree_hal_cmd_block_processor_budget_update_t budget_update,
    iree_hal_profile_command_region_event_flags_t transition_flags) {
  if (!iree_hal_cmd_block_processor_profile_records_regions(context)) return;

  const iree_hal_cmd_region_summary_t* completed_summary = NULL;
  if (completed_region_index >= 0 &&
      completed_region_index < (int32_t)completed_block->region_count) {
    completed_summary = &iree_hal_cmd_block_region_summaries(
        completed_block)[completed_region_index];
  }

  const iree_hal_cmd_region_summary_t* next_summary = NULL;
  if (next_block && next_region_index >= 0 &&
      next_region_index < (int32_t)next_block->region_count) {
    next_summary =
        &iree_hal_cmd_block_region_summaries(next_block)[next_region_index];
    transition_flags |=
        IREE_HAL_PROFILE_COMMAND_REGION_EVENT_FLAG_HAS_NEXT_REGION;
  }

  iree_hal_local_profile_command_region_event_info_t event_info =
      iree_hal_local_profile_command_region_event_info_default();
  event_info.flags = IREE_HAL_PROFILE_COMMAND_REGION_EVENT_FLAG_COMMAND_BUFFER |
                     transition_flags;
  event_info.scope = context->profile.scope;
  event_info.submission_id = context->profile.submission_id;
  event_info.command_buffer_id = context->profile.command_buffer_id;
  event_info.command_region.block_sequence = completed_block_sequence;
  event_info.command_region.epoch = completed_region_epoch;
  event_info.command_region.index = completed_region_index;
  event_info.command_region.dispatch_count =
      completed_summary ? completed_summary->dispatch_count : 0;
  event_info.command_region.tile_count =
      region_snapshot.command_region.tile_count;
  event_info.command_region.width_bucket =
      completed_summary ? completed_summary->width_bucket : 0;
  event_info.command_region.lookahead_width_bucket =
      completed_summary ? completed_summary->lookahead_width_bucket : 0;
  event_info.command_region.useful_drain_count =
      region_snapshot.command_region.useful_drain_count;
  event_info.command_region.no_work_drain_count =
      region_snapshot.command_region.no_work_drain_count;
  event_info.command_region.tail_no_work.count =
      region_snapshot.command_region.tail_no_work.count;
  event_info.command_region.tail_no_work.remaining_tiles.min =
      region_snapshot.command_region.tail_no_work.remaining_tiles.min;
  event_info.command_region.tail_no_work.remaining_tiles.max =
      region_snapshot.command_region.tail_no_work.remaining_tiles.max;
  memcpy(
      event_info.command_region.tail_no_work.remaining_tiles.bucket_counts,
      region_snapshot.command_region.tail_no_work.remaining_tiles.bucket_counts,
      sizeof(event_info.command_region.tail_no_work.remaining_tiles
                 .bucket_counts));
  event_info.command_region.tail_no_work.first_start_host_time_ns =
      region_snapshot.command_region.tail_no_work.first_start_host_time_ns;
  event_info.command_region.tail_no_work.last_end_host_time_ns =
      region_snapshot.command_region.tail_no_work.last_end_host_time_ns;
  event_info.command_region.tail_no_work.time_sums.start_offset_ns =
      region_snapshot.command_region.tail_no_work.time_sums.start_offset_ns;
  event_info.command_region.tail_no_work.time_sums.drain_duration_ns =
      region_snapshot.command_region.tail_no_work.time_sums.drain_duration_ns;
  event_info.command_region.first_useful_drain_start_host_time_ns =
      region_snapshot.command_region.first_useful_drain_start_host_time_ns;
  event_info.command_region.last_useful_drain_end_host_time_ns =
      region_snapshot.command_region.last_useful_drain_end_host_time_ns;
  event_info.command_region.start_host_time_ns =
      region_snapshot.command_region.start_host_time_ns != 0
          ? region_snapshot.command_region.start_host_time_ns
          : completed_region_end_host_time;
  event_info.command_region.end_host_time_ns = completed_region_end_host_time;
  event_info.next_command_region.index = next_region_index;
  event_info.next_command_region.tile_count = next_remaining_tiles;
  event_info.next_command_region.width_bucket =
      next_summary ? next_summary->width_bucket : 0;
  event_info.next_command_region.lookahead_width_bucket =
      next_summary ? next_summary->lookahead_width_bucket : 0;
  event_info.scheduler.worker_count = context->worker_count;
  event_info.scheduler.old_wake_budget = budget_update.old_budget;
  event_info.scheduler.new_wake_budget = budget_update.new_budget;
  event_info.scheduler.wake_delta = budget_update.wake_delta;
  event_info.retention.keep_active_count =
      region_snapshot.command_region.retention.keep_active_count;
  event_info.retention.publish_keep_active_count =
      region_snapshot.command_region.retention.publish_keep_active_count;
  event_info.retention.keep_warm_count =
      region_snapshot.command_region.retention.keep_warm_count;
  iree_hal_local_profile_recorder_append_command_region_event(
      context->profile.recorder, &event_info, /*out_event_id=*/NULL);
}

// Handles a completed region: advances to the next non-empty region or
// processes the block terminator (BRANCH/RETURN).
static void iree_hal_cmd_block_processor_handle_region_completion(
    iree_hal_cmd_block_processor_context_t* context,
    const iree_hal_cmd_block_header_t* block, iree_hal_cmd_block_state_t* state,
    const iree_hal_cmd_barrier_t* completed_barrier,
    int32_t completed_region_index,
    iree_hal_cmd_block_processor_drain_result_t* out_result) {
  void** binding_ptrs = iree_hal_cmd_block_state_binding_ptrs(
      state, context->max_region_dispatch_count);
  const bool profile_command_regions =
      iree_hal_cmd_block_processor_profile_records_regions(context);
  const iree_time_t completed_region_end_host_time =
      profile_command_regions ? iree_time_now() : 0;
  const uint32_t completed_block_sequence =
      profile_command_regions
          ? (uint32_t)iree_atomic_load(&context->block_sequence,
                                       iree_memory_order_relaxed)
          : 0;
  const uint32_t completed_region_epoch =
      profile_command_regions
          ? (uint32_t)iree_hal_cmd_block_region_state_epoch(iree_atomic_load(
                &state->region_state, iree_memory_order_relaxed))
          : 0;
  const iree_hal_cmd_block_processor_profile_region_snapshot_t
      completed_region_profile =
          iree_hal_cmd_block_processor_profile_snapshot_active_region(context);
  iree_hal_local_profile_host_execution_event_info_t* profile_events = NULL;
  iree_host_size_t profile_event_capacity = completed_barrier->dispatch_count;
  iree_host_size_t profile_event_count = 0;
  if (IREE_UNLIKELY(profile_event_capacity != 0 &&
                    context->profile.dispatches != NULL)) {
    profile_events =
        (iree_hal_local_profile_host_execution_event_info_t*)iree_alloca(
            profile_event_capacity * sizeof(*profile_events));
    profile_event_count = iree_hal_cmd_block_processor_profile_snapshot_region(
        context, completed_barrier, binding_ptrs, profile_events,
        profile_event_capacity);
  }

  // Find the next non-empty region using the summary table. Indirect dispatch
  // regions read their parameter buffers here, after the completed region has
  // produced any dynamic parameters.
  uint32_t next_remaining_tiles = 0;
  const int32_t next_region = iree_hal_cmd_block_processor_find_active_region(
      block, binding_ptrs, (uint16_t)(completed_region_index + 1),
      &next_remaining_tiles);

  if (next_region < (int32_t)block->region_count) {
    // Another region in this block. Assign a new global epoch and initialize
    // .data for the next region.
    iree_hal_cmd_block_processor_begin_retention_transition(context);
    int32_t region_epoch = context->next_epoch;
    context->next_epoch = region_epoch + 1;
    iree_hal_cmd_block_processor_init_region(context, state, region_epoch,
                                             next_region, next_remaining_tiles);
    iree_hal_cmd_block_processor_budget_update_t budget_update =
        iree_hal_cmd_block_processor_update_budget(context,
                                                   next_remaining_tiles);
    out_result->wake_delta += budget_update.wake_delta;
    iree_hal_cmd_block_processor_advance_retention_epoch(context);
    IREE_TRACE(iree_hal_cmd_block_processor_trace_region_transition());
    (void)budget_update;
    iree_hal_cmd_block_processor_profile_append_command_region_event(
        context, block, completed_block_sequence, completed_region_epoch,
        completed_region_index, completed_region_end_host_time,
        completed_region_profile, block, next_region, next_remaining_tiles,
        budget_update, IREE_HAL_PROFILE_COMMAND_REGION_EVENT_FLAG_NONE);
    iree_hal_cmd_block_processor_profile_append_host_execution_events(
        context, profile_events, profile_event_count);
    return;
  }

  // All regions in this block are done. Handle the terminator.
  const iree_hal_cmd_header_t* terminator =
      iree_hal_cmd_block_terminator(block);

  if (terminator->opcode == IREE_HAL_CMD_RETURN) {
    iree_hal_cmd_block_processor_budget_update_t budget_update = {0, 0, 0};
    IREE_TRACE(iree_hal_cmd_block_processor_trace_region_transition());
    (void)budget_update;
    iree_hal_cmd_block_processor_mark_completed(context);
    out_result->completed = true;
    iree_hal_cmd_block_processor_profile_append_command_region_event(
        context, block, completed_block_sequence, completed_region_epoch,
        completed_region_index, completed_region_end_host_time,
        completed_region_profile, NULL, -1, 0, budget_update,
        IREE_HAL_PROFILE_COMMAND_REGION_EVENT_FLAG_TERMINAL);
    iree_hal_cmd_block_processor_profile_append_host_execution_events(
        context, profile_events, profile_event_count);
    return;
  }

  if (terminator->opcode == IREE_HAL_CMD_BRANCH) {
    // Signal block transition in progress (odd → even). Workers seeing the
    // even value bail out of drain immediately. This is the writer-side of a
    // seqlock: even = transition in progress, odd = ready. The next odd
    // publication points workers at a fresh block-state slot.
    iree_atomic_fetch_add(&context->block_sequence, 1,
                          iree_memory_order_release);

    // Follow BRANCH to next block. Handle chains of empty blocks iteratively
    // to avoid stack overflow (pathological case: many blocks all with empty
    // predicated regions).
    const iree_hal_cmd_block_header_t* next_block =
        ((const iree_hal_cmd_branch_t*)terminator)->target;
    while (next_block) {
      iree_hal_cmd_block_state_t* next_state =
          iree_hal_cmd_block_processor_next_state(context);
      if (!next_state) {
        iree_hal_cmd_block_processor_report_error(
            context,
            iree_make_status(IREE_STATUS_INTERNAL,
                             "block processor state storage exhausted"));
        out_result->completed = true;
        iree_hal_cmd_block_processor_budget_update_t budget_update = {0, 0, 0};
        iree_hal_cmd_block_processor_profile_append_command_region_event(
            context, block, completed_block_sequence, completed_region_epoch,
            completed_region_index, completed_region_end_host_time,
            completed_region_profile, NULL, -1, 0, budget_update,
            IREE_HAL_PROFILE_COMMAND_REGION_EVENT_FLAG_BLOCK_TRANSITION |
                IREE_HAL_PROFILE_COMMAND_REGION_EVENT_FLAG_TERMINAL);
        iree_hal_cmd_block_processor_profile_append_host_execution_events(
            context, profile_events, profile_event_count);
        return;
      }
      iree_hal_cmd_block_processor_init_block(next_block, context, next_state);

      // Check if the block has any work (init_block sets region_state to the
      // first non-empty region, or region_count if all are empty).
      const int64_t next_region_state = iree_atomic_load(
          &next_state->region_state, iree_memory_order_relaxed);
      const int32_t first_active =
          iree_hal_cmd_block_region_state_index(next_region_state);
      if (first_active < (int32_t)next_block->region_count) {
        // Block has work. Update budget and signal ready (even → odd).
        const int64_t remaining_tiles = iree_atomic_load(
            &next_state->remaining_tiles, iree_memory_order_relaxed);
        iree_hal_cmd_block_processor_budget_update_t budget_update =
            iree_hal_cmd_block_processor_update_budget(
                context, (uint32_t)(remaining_tiles & 0xFFFFFFFFu));
        out_result->wake_delta += budget_update.wake_delta;
        IREE_TRACE(iree_hal_cmd_block_processor_trace_region_transition());
        (void)budget_update;
        iree_atomic_store(&context->current_block, (intptr_t)next_block,
                          iree_memory_order_relaxed);
        iree_hal_cmd_block_processor_publish_state(context, next_state);
        iree_atomic_fetch_add(&context->block_sequence, 1,
                              iree_memory_order_release);
        iree_hal_cmd_block_processor_advance_retention_epoch(context);
        iree_hal_cmd_block_processor_profile_append_command_region_event(
            context, block, completed_block_sequence, completed_region_epoch,
            completed_region_index, completed_region_end_host_time,
            completed_region_profile, next_block, first_active,
            (uint32_t)(remaining_tiles & 0xFFFFFFFFu), budget_update,
            IREE_HAL_PROFILE_COMMAND_REGION_EVENT_FLAG_BLOCK_TRANSITION);
        iree_hal_cmd_block_processor_profile_append_host_execution_events(
            context, profile_events, profile_event_count);
        return;
      }

      // Entirely empty block. Follow its terminator.
      const iree_hal_cmd_header_t* next_terminator =
          iree_hal_cmd_block_terminator(next_block);
      if (next_terminator->opcode == IREE_HAL_CMD_RETURN) {
        iree_hal_cmd_block_processor_mark_completed(context);
        out_result->completed = true;
        iree_hal_cmd_block_processor_budget_update_t budget_update = {0, 0, 0};
        iree_hal_cmd_block_processor_profile_append_command_region_event(
            context, block, completed_block_sequence, completed_region_epoch,
            completed_region_index, completed_region_end_host_time,
            completed_region_profile, NULL, -1, 0, budget_update,
            IREE_HAL_PROFILE_COMMAND_REGION_EVENT_FLAG_BLOCK_TRANSITION |
                IREE_HAL_PROFILE_COMMAND_REGION_EVENT_FLAG_TERMINAL);
        iree_hal_cmd_block_processor_profile_append_host_execution_events(
            context, profile_events, profile_event_count);
        return;
      }
      if (next_terminator->opcode == IREE_HAL_CMD_BRANCH) {
        next_block = ((const iree_hal_cmd_branch_t*)next_terminator)->target;
      } else {
        iree_hal_cmd_block_processor_report_error(
            context, iree_make_status(IREE_STATUS_INTERNAL,
                                      "unknown terminator opcode %d",
                                      next_terminator->opcode));
        out_result->completed = true;
        iree_hal_cmd_block_processor_budget_update_t budget_update = {0, 0, 0};
        iree_hal_cmd_block_processor_profile_append_command_region_event(
            context, block, completed_block_sequence, completed_region_epoch,
            completed_region_index, completed_region_end_host_time,
            completed_region_profile, NULL, -1, 0, budget_update,
            IREE_HAL_PROFILE_COMMAND_REGION_EVENT_FLAG_BLOCK_TRANSITION |
                IREE_HAL_PROFILE_COMMAND_REGION_EVENT_FLAG_TERMINAL);
        iree_hal_cmd_block_processor_profile_append_host_execution_events(
            context, profile_events, profile_event_count);
        return;
      }
    }
    // Fell off the block chain (next_block == NULL). This should not happen
    // in a well-formed recording — block chains always end with RETURN.
    iree_hal_cmd_block_processor_report_error(
        context,
        iree_make_status(IREE_STATUS_INTERNAL, "block chain ends with NULL"));
    out_result->completed = true;
    iree_hal_cmd_block_processor_budget_update_t budget_update = {0, 0, 0};
    iree_hal_cmd_block_processor_profile_append_command_region_event(
        context, block, completed_block_sequence, completed_region_epoch,
        completed_region_index, completed_region_end_host_time,
        completed_region_profile, NULL, -1, 0, budget_update,
        IREE_HAL_PROFILE_COMMAND_REGION_EVENT_FLAG_BLOCK_TRANSITION |
            IREE_HAL_PROFILE_COMMAND_REGION_EVENT_FLAG_TERMINAL);
    iree_hal_cmd_block_processor_profile_append_host_execution_events(
        context, profile_events, profile_event_count);
    return;
  }

  // Unknown terminator.
  iree_hal_cmd_block_processor_report_error(
      context,
      iree_make_status(IREE_STATUS_INTERNAL, "unknown terminator opcode %d",
                       terminator->opcode));
  out_result->completed = true;
  iree_hal_cmd_block_processor_budget_update_t budget_update = {0, 0, 0};
  iree_hal_cmd_block_processor_profile_append_command_region_event(
      context, block, completed_block_sequence, completed_region_epoch,
      completed_region_index, completed_region_end_host_time,
      completed_region_profile, NULL, -1, 0, budget_update,
      IREE_HAL_PROFILE_COMMAND_REGION_EVENT_FLAG_TERMINAL);
  iree_hal_cmd_block_processor_profile_append_host_execution_events(
      context, profile_events, profile_event_count);
}

// Multi-worker drain: one pass through the current active region.
static void iree_hal_cmd_block_processor_drain_multi_worker(
    iree_hal_cmd_block_processor_context_t* context,
    const iree_hal_cmd_block_processor_worker_context_t* worker_context,
    iree_hal_cmd_block_processor_worker_state_t* worker_state,
    iree_hal_cmd_block_processor_drain_result_t* out_result) {
  // Check for completion (error or RETURN reached).
  if (iree_atomic_load(&context->completed, iree_memory_order_acquire)) {
    out_result->completed = true;
    IREE_TRACE(out_result->reason =
                   IREE_HAL_CMD_BLOCK_PROCESSOR_DRAIN_REASON_COMPLETED);
    return;
  }

  // Seqlock reader: block_sequence uses odd/even parity to gate block
  // transitions. Odd = ready (workers may proceed), even = transition in
  // progress (workers must bail). The acquire pairs with the completer's
  // release, ensuring current_block/current_state point at initialized state
  // when the worker sees an odd value.
  int32_t block_sequence =
      iree_atomic_load(&context->block_sequence, iree_memory_order_acquire);
  out_result->block_sequence = block_sequence;
  if ((block_sequence & 1) == 0) {
    // Block transition in progress. Bail and retry on next drain().
    IREE_TRACE(out_result->reason =
                   IREE_HAL_CMD_BLOCK_PROCESSOR_DRAIN_REASON_BLOCK_TRANSITION);
    return;
  }
  if (block_sequence != worker_state->block_sequence) {
    worker_state->block_sequence = block_sequence;
  }

  const iree_hal_cmd_block_header_t* block =
      (const iree_hal_cmd_block_header_t*)iree_atomic_load(
          &context->current_block, iree_memory_order_relaxed);
  iree_hal_cmd_block_state_t* state =
      iree_hal_cmd_block_processor_current_state(context);

  // Recheck the block seqlock after sampling current_block/current_state. If a
  // block transition raced the sample, bail before interpreting either value.
  int32_t block_sequence_recheck =
      iree_atomic_load(&context->block_sequence, iree_memory_order_acquire);
  if (block_sequence_recheck != block_sequence) {
    IREE_TRACE(
        out_result->reason =
            IREE_HAL_CMD_BLOCK_PROCESSOR_DRAIN_REASON_STALE_BLOCK_SEQUENCE);
    return;
  }

  // Read the global epoch and active region index as one packed value.
  // region_state is the synchronization point for intra-block region
  // transitions: the completer stores it with release after writing
  // tile_indices, remaining_tiles, and the active region index. Acquiring
  // region_state guarantees all those writes are visible and prevents a worker
  // from combining the epoch from one region with the index from another.
  //
  // For block transitions, block_sequence (acquired above) provides
  // ordering; region_state is redundantly visible but harmless.
  int64_t region_state =
      iree_atomic_load(&state->region_state, iree_memory_order_acquire);
  int32_t region_epoch = iree_hal_cmd_block_region_state_epoch(region_state);
  out_result->region_epoch = region_epoch;
  int32_t active_region = iree_hal_cmd_block_region_state_index(region_state);
  IREE_TRACE({ out_result->active_region = active_region; });

  // All regions in this block are done. The completer is handling the
  // block terminator (or has already set completed).
  if (active_region >= (int32_t)block->region_count) {
    if (iree_atomic_load(&context->completed, iree_memory_order_acquire)) {
      out_result->completed = true;
    }
    IREE_TRACE(out_result->reason =
                   IREE_HAL_CMD_BLOCK_PROCESSOR_DRAIN_REASON_REGION_COMPLETE);
    return;
  }

  // Seqlock validation: re-read block_sequence to detect block transitions
  // that started between our initial read and here. If the value changed,
  // our reads of current_block, region_state, etc. may reflect
  // partially-initialized state from init_block. Bail and retry.
  block_sequence_recheck =
      iree_atomic_load(&context->block_sequence, iree_memory_order_acquire);
  if (block_sequence_recheck != block_sequence) {
    IREE_TRACE(
        out_result->reason =
            IREE_HAL_CMD_BLOCK_PROCESSOR_DRAIN_REASON_STALE_BLOCK_SEQUENCE);
    return;
  }

  // Check for errors before starting work.
  if (iree_hal_cmd_block_processor_has_error(context)) {
    out_result->completed = true;
    IREE_TRACE(out_result->reason =
                   IREE_HAL_CMD_BLOCK_PROCESSOR_DRAIN_REASON_ERROR);
    return;
  }

  // Resolve the current region's barrier from immutable block metadata after
  // acquiring region_state. This keeps stale workers from observing a
  // future-region barrier pointer before they observe that region's epoch.
  const iree_hal_cmd_barrier_t* barrier =
      iree_hal_cmd_block_region_barrier(block, (uint16_t)active_region);

  // Process the region's work commands cooperatively.
  const bool profile_command_region =
      iree_hal_cmd_block_processor_profile_records_regions(context);
  const iree_time_t command_region_drain_start_host_time_ns =
      IREE_UNLIKELY(profile_command_region) ? iree_time_now() : 0;
  uint32_t my_tiles = iree_hal_cmd_block_processor_process_region(
      barrier, region_epoch, worker_context, context, state);
  if (my_tiles == 0) {
    out_result->warm_retainer_limit =
        iree_hal_cmd_block_processor_region_warm_retainer_limit(context, block,
                                                                active_region);
    out_result->prefer_warm_spin =
        iree_hal_cmd_block_processor_region_prefers_warm_spin(block,
                                                              active_region);
  }
  if (IREE_UNLIKELY(profile_command_region)) {
    uint32_t remaining_tile_count = 0;
    bool record_drain = my_tiles != 0;
    if (my_tiles == 0) {
      const int64_t remaining_tiles =
          iree_atomic_load(&state->remaining_tiles, iree_memory_order_acquire);
      if ((remaining_tiles >> 32) == region_epoch) {
        remaining_tile_count = (uint32_t)(remaining_tiles & 0xFFFFFFFFu);
        record_drain = true;
      }
    }
    if (record_drain) {
      iree_hal_cmd_block_processor_profile_record_drain(
          context, my_tiles, remaining_tile_count,
          command_region_drain_start_host_time_ns, iree_time_now());
    }
  }
  out_result->tiles_executed += my_tiles;
  IREE_TRACE(out_result->reason =
                 my_tiles == 0
                     ? IREE_HAL_CMD_BLOCK_PROCESSOR_DRAIN_REASON_NO_TILES
                     : IREE_HAL_CMD_BLOCK_PROCESSOR_DRAIN_REASON_WORK);
  // Completer election via epoch-tagged remaining_tiles CAS.
  //
  // Workers with 0 tiles skip the election to avoid false positives.
  // Workers with tiles CAS-decrement the count, validating the epoch and
  // giving TSAN an explicit synchronization edge between one region's memory
  // effects and the completer's publication of the next region.
  // The worker whose CAS drives the count to zero becomes the completer.
  bool is_completer = false;
  if (my_tiles > 0) {
    const int64_t epoch_shifted = (int64_t)region_epoch << 32;
    int64_t current =
        iree_atomic_load(&state->remaining_tiles, iree_memory_order_acquire);
    while (true) {
      if ((current >> 32) != region_epoch) break;
      const uint32_t count = (uint32_t)(current & 0xFFFFFFFFu);
      IREE_ASSERT(count >= my_tiles);
      const uint32_t new_count = count - my_tiles;
      const int64_t desired = epoch_shifted | (int64_t)new_count;
      IREE_TRACE(out_result->remaining_tiles = new_count);
      if (iree_atomic_compare_exchange_weak(&state->remaining_tiles, &current,
                                            desired, iree_memory_order_acq_rel,
                                            iree_memory_order_acquire)) {
        is_completer = new_count == 0;
        break;
      }
    }
  }

  if (!is_completer) return;
  IREE_TRACE(out_result->reason =
                 IREE_HAL_CMD_BLOCK_PROCESSOR_DRAIN_REASON_COMPLETER);

  // === COMPLETER ===
  // No arrival wait is needed: epoch-tagged tile counters reject stale region
  // claims, and workers that claimed tiles keep remaining_tiles non-zero until
  // they report completion. Block transitions use a fresh state slot so old
  // binding arrays remain stable for stale readers.

  // Handle the completed region: advance to the next region or process
  // the block terminator.
  iree_hal_cmd_block_processor_handle_region_completion(
      context, block, state, barrier, active_region, out_result);
  iree_hal_cmd_block_processor_wake_additional_workers(context,
                                                       out_result->wake_delta);
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

// Sets up the context for multi-worker execution by finding the first block
// with actual work. Follows empty block chains (blocks where all regions
// have 0 tiles) iteratively until a block with work is found or RETURN is
// reached. If no work exists, sets completed=true so drain() returns
// immediately.
static void iree_hal_cmd_block_processor_setup_first_block(
    iree_hal_cmd_block_processor_context_t* context) {
  const iree_hal_cmd_block_header_t* block = context->recording->first_block;
  while (block) {
    iree_hal_cmd_block_state_t* state =
        iree_hal_cmd_block_processor_next_state(context);
    if (!state) {
      iree_hal_cmd_block_processor_report_error(
          context, iree_make_status(IREE_STATUS_INTERNAL,
                                    "block processor state storage exhausted"));
      return;
    }
    iree_hal_cmd_block_processor_init_block(block, context, state);

    // Check if the block has any work (init_block sets region_state to the
    // first non-empty region, or region_count if all regions are empty).
    const int64_t region_state =
        iree_atomic_load(&state->region_state, iree_memory_order_relaxed);
    const int32_t first_active =
        iree_hal_cmd_block_region_state_index(region_state);
    if (first_active < (int32_t)block->region_count) {
      // Block has work. Publish it and set sequence so workers can begin.
      iree_atomic_store(&context->current_block, (intptr_t)block,
                        iree_memory_order_relaxed);
      iree_hal_cmd_block_processor_publish_state(context, state);
      iree_atomic_store(&context->block_sequence, 1, iree_memory_order_release);
      return;
    }

    // Entirely empty block. Follow its terminator.
    const iree_hal_cmd_header_t* terminator =
        iree_hal_cmd_block_terminator(block);
    if (terminator->opcode == IREE_HAL_CMD_RETURN) {
      iree_atomic_store(&context->completed, 1, iree_memory_order_release);
      return;
    }
    if (terminator->opcode == IREE_HAL_CMD_BRANCH) {
      block = ((const iree_hal_cmd_branch_t*)terminator)->target;
    } else {
      iree_hal_cmd_block_processor_report_error(
          context,
          iree_make_status(IREE_STATUS_INTERNAL, "unknown terminator opcode %d",
                           terminator->opcode));
      return;
    }
  }
  // Fell off the block chain (NULL target). Should not happen in a
  // well-formed recording.
  iree_atomic_store(&context->completed, 1, iree_memory_order_release);
}

void iree_hal_cmd_block_processor_context_initialize(
    iree_hal_cmd_block_processor_context_t* out_context,
    const iree_hal_cmd_block_recording_t* recording,
    const iree_hal_cmd_binding_entry_t* binding_table,
    iree_host_size_t binding_table_length, iree_hal_cmd_block_state_t* state,
    iree_host_size_t state_size) {
  memset(out_context, 0, sizeof(*out_context));
  out_context->recording = recording;
  out_context->binding_table = binding_table;
  out_context->binding_table_length = binding_table_length;
  out_context->state_storage = state;
  out_context->state_size = state_size;
  out_context->state_stride = state_size;
  out_context->state_count = 1;
  out_context->worker_count = 1;
  iree_atomic_store(&out_context->current_wake_budget, 1,
                    iree_memory_order_relaxed);
  out_context->max_region_dispatch_count = recording->max_region_dispatch_count;
  out_context->max_total_binding_count = recording->max_total_binding_count;
  out_context->next_epoch = 1;
  if (recording->first_block) {
    iree_atomic_store(&out_context->current_block,
                      (intptr_t)recording->first_block,
                      iree_memory_order_relaxed);
    iree_hal_cmd_block_processor_publish_state(out_context, state);
  }
}

void iree_hal_cmd_block_processor_context_set_profile_recorder(
    iree_hal_cmd_block_processor_context_t* context,
    iree_hal_local_profile_recorder_t* recorder,
    iree_hal_local_profile_queue_scope_t scope, uint64_t submission_id,
    uint64_t command_buffer_id,
    iree_hal_cmd_block_processor_profile_dispatch_t* dispatches,
    iree_host_size_t dispatch_capacity) {
  if (!context) return;
  context->profile.recorder = recorder;
  context->profile.scope = scope;
  context->profile.submission_id = submission_id;
  context->profile.command_buffer_id = command_buffer_id;
  context->profile.dispatches = dispatches;
  context->profile.dispatch_capacity = dispatch_capacity;
  context->profile.command_region.events_enabled =
      iree_hal_local_profile_recorder_is_enabled(
          recorder, IREE_HAL_DEVICE_PROFILING_DATA_COMMAND_REGION_EVENTS);
  iree_hal_cmd_block_processor_profile_reset_dispatches(context);
  if (context->worker_count > 1 &&
      iree_hal_cmd_block_processor_profile_records_regions(context)) {
    iree_hal_cmd_block_state_t* state =
        iree_hal_cmd_block_processor_current_state(context);
    if (state) {
      int64_t remaining_tiles =
          iree_atomic_load(&state->remaining_tiles, iree_memory_order_relaxed);
      iree_hal_cmd_block_processor_profile_begin_region(
          context, (uint32_t)(remaining_tiles & 0xFFFFFFFFu));
    }
  }
}

int32_t iree_hal_cmd_block_processor_context_wake_budget(
    const iree_hal_cmd_block_processor_context_t* context) {
  return context ? iree_atomic_load(&context->current_wake_budget,
                                    iree_memory_order_relaxed)
                 : 1;
}

void iree_hal_cmd_block_processor_context_profile_record_retention(
    iree_hal_cmd_block_processor_context_t* context, bool keep_active,
    bool publish_keep_active, bool keep_warm) {
  if (!context ||
      !iree_hal_cmd_block_processor_profile_records_regions(context)) {
    return;
  }
  iree_atomic_fetch_add(&context->profile.command_region.no_work_drain_count, 1,
                        iree_memory_order_relaxed);
  if (keep_active) {
    iree_atomic_fetch_add(
        &context->profile.command_region.retention.keep_active_count, 1,
        iree_memory_order_relaxed);
  }
  if (publish_keep_active) {
    iree_atomic_fetch_add(
        &context->profile.command_region.retention.publish_keep_active_count, 1,
        iree_memory_order_relaxed);
  }
  if (keep_warm) {
    iree_atomic_fetch_add(
        &context->profile.command_region.retention.keep_warm_count, 1,
        iree_memory_order_relaxed);
  }
}

bool iree_hal_cmd_block_processor_context_did_advance(
    const iree_hal_cmd_block_processor_context_t* context,
    const iree_hal_cmd_block_processor_drain_result_t* drain_result) {
  if (!context || !drain_result) return true;
  if (iree_atomic_load(&context->completed, iree_memory_order_acquire)) {
    return true;
  }
  if (drain_result->block_sequence != 0) {
    int32_t block_sequence =
        iree_atomic_load(&context->block_sequence, iree_memory_order_acquire);
    if (block_sequence != drain_result->block_sequence) return true;
  }
  if (drain_result->region_epoch != 0) {
    iree_hal_cmd_block_state_t* state =
        iree_hal_cmd_block_processor_current_state(context);
    if (!state) return true;
    int64_t region_state =
        iree_atomic_load(&state->region_state, iree_memory_order_acquire);
    int32_t region_epoch = iree_hal_cmd_block_region_state_epoch(region_state);
    if (region_epoch != drain_result->region_epoch) return true;
  }
  return false;
}

iree_status_t iree_hal_cmd_block_processor_context_allocate(
    const iree_hal_cmd_block_recording_t* recording,
    const iree_hal_cmd_binding_entry_t* binding_table,
    iree_host_size_t binding_table_length, uint32_t worker_count,
    iree_allocator_t allocator,
    iree_hal_cmd_block_processor_context_t** out_context) {
  *out_context = NULL;
  if (!recording->first_block) return iree_ok_status();
  if (worker_count == 0) worker_count = 1;

  // Allocate .data sized to the highwater mark across all blocks. Multi-worker
  // execution gets one slot per block so block-local binding fixups are never
  // overwritten while stale workers may still be reading the previous block.
  const iree_host_size_t state_size = iree_hal_cmd_block_state_size(
      recording->max_region_dispatch_count, recording->max_total_binding_count);
  const iree_host_size_t state_stride =
      iree_host_align(state_size, iree_hardware_destructive_interference_size);
  const uint16_t state_count =
      worker_count > 1 ? (recording->block_count ? recording->block_count : 1)
                       : 1;

  // Allocate context + state in one allocation with cache line alignment
  // so the iree_alignas(64) atomic fields land at proper boundaries.
  const iree_host_size_t context_size =
      iree_host_align(sizeof(iree_hal_cmd_block_processor_context_t),
                      iree_hardware_destructive_interference_size);
  iree_host_size_t state_storage_size = 0;
  iree_host_size_t total_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(state_stride, state_count,
                                                &state_storage_size) ||
                    !iree_host_size_checked_add(
                        context_size, state_storage_size, &total_size))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "block processor state storage is too large: %zu bytes per block, %u "
        "blocks",
        (size_t)state_stride, state_count);
  }

  void* allocation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      allocator, total_size, iree_hardware_destructive_interference_size,
      /*offset=*/0, &allocation));

  iree_hal_cmd_block_processor_context_t* context =
      (iree_hal_cmd_block_processor_context_t*)allocation;
  iree_hal_cmd_block_state_t* state =
      (iree_hal_cmd_block_state_t*)((uint8_t*)allocation + context_size);
  iree_hal_cmd_block_processor_context_initialize(
      context, recording, binding_table, binding_table_length, state,
      state_size);
  context->worker_count = worker_count;
  context->state_stride = state_stride;
  context->state_count = state_count;

  if (worker_count > 1) {
    // Multi-worker: find the first block with work (following empty block
    // chains) and initialize .data. If no work exists, sets completed=true
    // so drain() returns immediately.
    iree_hal_cmd_block_processor_setup_first_block(context);
  }

  *out_context = context;
  return iree_ok_status();
}

void iree_hal_cmd_block_processor_drain(
    iree_hal_cmd_block_processor_context_t* context,
    const iree_hal_cmd_block_processor_worker_context_t* worker_context,
    iree_hal_cmd_block_processor_worker_state_t* worker_state,
    iree_hal_cmd_block_processor_drain_result_t* out_result) {
  out_result->tiles_executed = 0;
  out_result->completed = false;
  out_result->block_sequence = 0;
  out_result->region_epoch = 0;
  out_result->wake_delta = 0;
  out_result->warm_retainer_limit = 0;
  out_result->prefer_warm_spin = false;
  IREE_TRACE({
    out_result->reason = IREE_HAL_CMD_BLOCK_PROCESSOR_DRAIN_REASON_UNKNOWN;
    out_result->active_region = -1;
    out_result->remaining_tiles = 0;
  });

  if (!context) {
    out_result->completed = true;
    IREE_TRACE(out_result->reason =
                   IREE_HAL_CMD_BLOCK_PROCESSOR_DRAIN_REASON_NULL_CONTEXT);
    return;
  }
  IREE_ASSERT_ARGUMENT(worker_context);

  if (context->worker_count == 1) {
    // Single-worker fast path: no atomics, no synchronization.
    iree_status_t status = iree_hal_cmd_block_processor_execute_single_worker(
        context, worker_context, &out_result->tiles_executed);
    if (!iree_status_is_ok(status)) {
      iree_hal_cmd_block_processor_report_error(context, status);
    }
    out_result->completed = true;
    IREE_TRACE(out_result->reason =
                   IREE_HAL_CMD_BLOCK_PROCESSOR_DRAIN_REASON_SINGLE_WORKER);
    return;
  }

  iree_hal_cmd_block_processor_drain_multi_worker(context, worker_context,
                                                  worker_state, out_result);
}

iree_status_t iree_hal_cmd_block_processor_context_consume_result(
    iree_hal_cmd_block_processor_context_t* context) {
  if (!context) return iree_ok_status();
  intptr_t error = iree_atomic_exchange(&context->error_status, 0,
                                        iree_memory_order_acquire);
  return (iree_status_t)error;
}

void iree_hal_cmd_block_processor_context_free(
    iree_hal_cmd_block_processor_context_t* context,
    iree_allocator_t allocator) {
  if (!context) return;
  iree_allocator_free_aligned(allocator, context);
}
