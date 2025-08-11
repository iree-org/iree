// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/command_buffer.h"

#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/blit.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/util/block_pool.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Populates the device-side |out_ref| variant of a HAL buffer |ref|.
// This performs the HAL translation from an allocated buffer to the device
// buffer so that no access to HAL data structures is needed on device
// (iree_hal_buffer_t, etc).
static iree_status_t iree_hal_amdgpu_translate_device_buffer_ref(
    iree_hal_buffer_ref_t ref, iree_hal_amdgpu_device_buffer_ref_t* out_ref) {
  // Slot references are resolved when the command buffer is issued.
  // Record the range to translate from the buffer provided with the submission.
  if (!ref.buffer) {
    out_ref->offset = ref.offset;
    out_ref->type = IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_SLOT;
    out_ref->length = ref.length;
    out_ref->value.slot = ref.buffer_slot;
    return iree_ok_status();
  }

  // Resolve buffers to their device type and value bits (which is
  // type-specific). Note that we use the allocated buffer and not any wrapper
  // that may be around it (subspans, etc).
  iree_hal_amdgpu_device_buffer_type_t type = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_resolve_buffer(
      iree_hal_buffer_allocated_buffer(ref.buffer), &type,
      &out_ref->value.bits));
  out_ref->type = type;

  // Translate range from the allocated buffer base along with the reference
  // offset. This avoids the device needing to read the HAL buffer when issuing.
  out_ref->offset = iree_hal_buffer_byte_offset(ref.buffer) + ref.offset;
  if (ref.length == IREE_HAL_WHOLE_BUFFER) {
    out_ref->length = iree_hal_buffer_byte_length(ref.buffer);
  } else {
    out_ref->length = ref.length;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_query_id_scratch_t
//===----------------------------------------------------------------------===//

// Scratch space for acquiring query IDs that are associated with commands.
// Query IDs are dependent on HAL commands and not AQL packets so we can share
// them for all devices.
typedef struct iree_hal_amdgpu_query_id_scratch_t {
  // Next block-relative query ID when tracing commands.
  // Also represents the maximum ID required.
  iree_hal_amdgpu_device_command_query_id_t next;
  // Total number of commands query ID slots assigned.
  // Not all commands have query IDs and for those that don't the invalid value
  // IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID is assigned.
  uint32_t count;
  // Scratch space for recording query IDs.
  // Reused across blocks and copied out per-block as they each end recording.
  iree_hal_amdgpu_device_command_query_id_t values[/*command_capacity*/];
} iree_hal_amdgpu_query_id_scratch_t;

// Resets the scratch state to empty.
// Query ID values are not reset and should be assumed uninitialized.
static void iree_hal_amdgpu_query_id_scratch_reset(
    iree_hal_amdgpu_query_id_scratch_t* scratch) {
  if (!scratch) return;
  scratch->count = 0;
  scratch->next.control_id = 0;
  scratch->next.dispatch_id = 0;
}

// Assigns query IDs for the given command based on its type and increments the
// command count. If the command does not require query IDs an empty slot is
// assigned. Requires that the caller does bounds checking.
static void iree_hal_amdgpu_assign_cmd_query_ids(
    iree_hal_amdgpu_query_id_scratch_t* scratch,
    iree_hal_amdgpu_device_cmd_type_t type) {
  if (!scratch) return;
#if !(IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE)
  return;  // tracing disabled
#endif     // !IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
  enum {
    NONE = 0,
    CONTROL = 1 << 0,
    DISPATCH = 1 << 1,
    ALL = CONTROL | DISPATCH,
  };
  static const uint8_t required_ids[IREE_HAL_AMDGPU_DEVICE_CMD_MAX + 1] = {
      [IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_BEGIN] = ALL,
      [IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_END] = ALL,
      [IREE_HAL_AMDGPU_DEVICE_CMD_BARRIER] = NONE,
      [IREE_HAL_AMDGPU_DEVICE_CMD_SIGNAL_EVENT] = NONE,
      [IREE_HAL_AMDGPU_DEVICE_CMD_RESET_EVENT] = NONE,
      [IREE_HAL_AMDGPU_DEVICE_CMD_WAIT_EVENTS] = NONE,
      [IREE_HAL_AMDGPU_DEVICE_CMD_FILL_BUFFER] = DISPATCH,
      [IREE_HAL_AMDGPU_DEVICE_CMD_COPY_BUFFER] = DISPATCH,
      [IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH] = DISPATCH,
      [IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH_INDIRECT_DYNAMIC] = DISPATCH,
      [IREE_HAL_AMDGPU_DEVICE_CMD_BRANCH] = NONE,
      [IREE_HAL_AMDGPU_DEVICE_CMD_RETURN] = NONE,
  };
  iree_hal_amdgpu_device_command_query_id_t query_id = {
      .control_id = IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID,
      .dispatch_id = IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID,
  };
  if (required_ids[type] & CONTROL) {
    query_id.control_id = scratch->next.control_id++;
  }
  if (required_ids[type] & DISPATCH) {
    query_id.dispatch_id = scratch->next.dispatch_id++;
  }
  scratch->values[scratch->count++] = query_id;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_command_encoder_t
//===----------------------------------------------------------------------===//

// Per-device state managed by a command block encoder.
typedef struct iree_hal_amdgpu_device_encoder_state_t {
  // Arena of device-side memory used for metadata (blocks, query lists, etc)
  // reused across blocks.
  iree_hal_amdgpu_block_arena_t metadata_arena;
  // Arena of device-side memory used for embedded data (inline update buffers,
  // bindings, constants, etc) reused across blocks.
  iree_hal_amdgpu_block_arena_t storage_arena;

  // Command block storage.
  // Commands are fixed-size and we allocate one device memory block per CFG
  // block we encode. This wastes space and requires a compaction step during
  // finalization if we wanted to fix it up.
  struct {
    // Device block pool used for allocating command storage.
    iree_hal_amdgpu_block_pool_t* pool;
    // Head of the command block linked list.
    // This is the first encoded block.
    iree_hal_amdgpu_block_t* head;
    // Tail of the command block linked list.
    // This is the last (completed) encoded block.
    iree_hal_amdgpu_block_t* tail;
    // Current command block in device memory storing the encoded commands.
    // Reset each CFG block.
    iree_hal_amdgpu_block_t* current;
  } cmd_block;
} iree_hal_amdgpu_device_encoder_state_t;

// Manages command buffer block encoding for multiple devices.
// Information common to each device is shared (such as command counts and query
// IDs) while command and embedded data is stored per-device.
//
// Offsets and sizes here are limited both by the options provided during
// command buffer construction (usually derived from user options or device
// limits) and the maximum values allowed by the device-side scheduler. Most
// command packets use uint16_t offsets to fit under the fixed packet size limit
// but this happens to be in-line with what devices are practically limited to
// by design.
typedef struct iree_hal_amdgpu_command_encoder_t {
  // Whether a block is open for encoding (between a begin/end).
  uint64_t in_block : 1;
  // An execution barrier has been requested and the next command should have
  // its barrier bit set.
  uint64_t barrier_pending : 1;

  // Total number of devices being recorded. Used to size per-device state.
  iree_host_size_t device_count;

  // Total number of blocks recorded (including the current block, if any).
  iree_host_size_t block_count;

  // Scratch host memory used for accumulating data for finalization.
  // Only reset when recording completes.
  iree_arena_allocator_t* host_arena;

  // Query ID scratch space for per-command query ID allocation.
  // Shared across all devices and copied to per-device block metadata at the
  // end of encoding each block.
  iree_hal_amdgpu_query_id_scratch_t* query_ids;

  // Maximum number of commands that can be encoded into a block (including
  // terminator). Blocks are split automatically if they exceed the capacity.
  uint32_t command_capacity;
  // Current count of commands encoded into the block.
  // The encoder ensures that there is always room under the capacity for a
  // terminator command.
  uint16_t command_count;

  // Maximum number of AQL packets that can be used in a block.
  // This is checked against the target queue limits during command buffer
  // submission.
  uint32_t max_aql_packet_capacity;
  // Peak number of AQL packets required by any block encoded so far.
  // Depending on dynamic state when the command buffer is issued fewer packets
  // may be required and this is used to ensure HSA queue capacity is available
  // prior to issuing a command block.
  uint16_t peak_aql_packet_count;
  // Offset of the next AQL packet to be allocated.
  uint16_t aql_packet_offset;

  // Maximum kernarg capacity in bytes that can be used within a single block.
  uint32_t max_kernarg_capacity;
  // Peak kernarg size in bytes required by any block encoded so far.
  uint16_t peak_kernarg_size;
  // Offset of the next kernarg pointer to be allocated.
  // This may not be aligned (1 byte alignment) and needs to be aligned by the
  // allocator requesting space.
  uint16_t kernarg_offset;

  // Scratch space for the last command appended on each device. This is
  // returned to callers appending commands so they can do per-device updates.
  // Pointers are invalidated on block reset or when the next command is
  // appended.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_cmd_header_t**
      device_cmds /*[device_count]*/;

  // Active command buffer block encoder for each device if the command buffer
  // is recording. Encoders are acquired from the small block pool when
  // recording begins and discarded upon finalization.
  iree_hal_amdgpu_device_encoder_state_t device_state[/*device_count*/];
} iree_hal_amdgpu_command_encoder_t;

// Initializes a command encoder with limits based on the provided |options|
// that can encode into |device_count| devices. The encoder will be allocated
// from |host_arena| and iree_hal_amdgpu_command_encoder_deinitialize must be
// called prior to resetting the arena.
static iree_status_t iree_hal_amdgpu_command_encoder_initialize(
    const iree_hal_amdgpu_command_buffer_options_t* options,
    iree_host_size_t device_count, iree_arena_allocator_t* host_arena,
    iree_hal_amdgpu_command_encoder_t** out_encoder) {
  IREE_ASSERT_ARGUMENT(out_encoder);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_encoder = NULL;

  // Allocate the dynamically-sized encoder from the arena.
  iree_hal_amdgpu_command_encoder_t* encoder = NULL;
  const iree_host_size_t encoder_size =
      sizeof(*encoder) + device_count * sizeof(encoder->device_state[0]);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(host_arena, encoder_size, (void**)&encoder));

  // NOTE: arena allocations may be uninitialized.
  memset(encoder, 0, sizeof(*encoder));
  encoder->device_count = device_count;
  encoder->host_arena = host_arena;

  // Capacities are defined as options or are derived from available pools.
  // A larger pool will allow more commands to be encoded in a single block.
  // We require contiguous allocations for the commands (today) and if we want
  // to save space can perform a compaction during finalization (as commands
  // don't contain pointers to the command memory).
  encoder->command_capacity =
      iree_host_size_floor_div(options->device_block_pools[0]->large.block_size,
                               IREE_HAL_AMDGPU_DEVICE_CMD_SIZE);
  encoder->max_aql_packet_capacity = options->block_aql_packet_count;
  encoder->max_kernarg_capacity = UINT32_MAX;

// Allocate query IDs from the host arena.
// The query ID list is uninitialized but that's ok: we don't guarantee any
// queries beyond the current count are initialized.
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
  // TODO(benvanik): support options->recording_flags bit for tracing control
  // per-command-buffer. If not set we can disable it here and save some scratch
  // space and device uploads.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(host_arena,
                              sizeof(*encoder->query_ids) +
                                  encoder->command_capacity *
                                      sizeof(encoder->query_ids->values[0]),
                              (void**)&encoder->query_ids));
  encoder->query_ids->next.control_id = 0;
  encoder->query_ids->next.dispatch_id = 0;
  encoder->query_ids->count = 0;
#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

  // The device command scratch list is used for returning the commands
  // allocated on each device. This isn't great but since we have an area it's
  // cheap.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(host_arena,
                              device_count * sizeof(encoder->device_cmds[0]),
                              (void**)&encoder->device_cmds));

  // Initialize per-device encoder state.
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    iree_hal_amdgpu_device_encoder_state_t* device_state =
        &encoder->device_state[i];
    memset(device_state, 0, sizeof(*device_state));
    iree_hal_amdgpu_block_arena_initialize(
        &options->device_block_pools[i]->small, &device_state->metadata_arena);
    iree_hal_amdgpu_block_arena_initialize(
        &options->device_block_pools[i]->large, &device_state->storage_arena);
    // TODO(benvanik): choose the pool based on the expected command buffer
    // size. We don't know this today but could add a hint flag for "small" that
    // let us avoid taking a large block for 1 command.
    device_state->cmd_block.pool = &options->device_block_pools[i]->large;
  }

  *out_encoder = encoder;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Deinitializes an |encoder| and releases any resources regardless of whether a
// block is in progress (such as when cleaning up from a failure state).
static void iree_hal_amdgpu_command_encoder_deinitialize(
    iree_hal_amdgpu_command_encoder_t* encoder) {
  if (!encoder) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release all resources that the encoder may still hold on to.
  // In successful recordings these will have been taken by the parent command
  // buffer.
  for (iree_host_size_t i = 0; i < encoder->device_count; ++i) {
    iree_hal_amdgpu_device_encoder_state_t* device_state =
        &encoder->device_state[i];
    if (device_state->cmd_block.head) {
      iree_hal_amdgpu_block_pool_release_list(device_state->cmd_block.pool,
                                              device_state->cmd_block.head);
    }
    if (device_state->cmd_block.current) {
      iree_hal_amdgpu_block_pool_release(device_state->cmd_block.pool,
                                         device_state->cmd_block.current);
    }
    iree_hal_amdgpu_block_arena_deinitialize(&device_state->storage_arena);
    iree_hal_amdgpu_block_arena_deinitialize(&device_state->metadata_arena);
  }

  IREE_TRACE_ZONE_END(z0);
}

// Begins a new command buffer block and returns its block ordinal.
// The encoder must be in the default state with any prior block closed.
//
// The block ordinal will remain stable during recording and can be used when
// emitting branch operations as if it were a label. The final pointer of the
// block in device memory will not be available until recording has completed.
static iree_status_t iree_hal_amdgpu_command_encoder_begin_block(
    iree_hal_amdgpu_command_encoder_t* encoder, uint32_t* out_block_ordinal) {
  IREE_ASSERT_ARGUMENT(encoder);
  IREE_ASSERT(out_block_ordinal);
  *out_block_ordinal = 0;
  if (IREE_UNLIKELY(encoder->in_block)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "a block is already being recorded and must be "
                            "completed prior to recording another");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate initial empty command blocks on each device.
  for (iree_host_size_t i = 0; i < encoder->device_count; ++i) {
    iree_hal_amdgpu_device_encoder_state_t* device_state =
        &encoder->device_state[i];
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_amdgpu_block_pool_acquire(device_state->cmd_block.pool,
                                           &device_state->cmd_block.current));
  }

  // Begin block with implicit barrier at entry.
  encoder->in_block = 1;
  encoder->barrier_pending = 1;
  ++encoder->block_count;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Ends the current command buffer block and finalizes its device-side
// resources. Upon return the encoder will be in the default idle state.
// The caller must have inserted a terminator block prior to ending in order
// for the command buffer to be valid.
static iree_status_t iree_hal_amdgpu_command_encoder_end_block(
    iree_hal_amdgpu_command_encoder_t* encoder) {
  IREE_ASSERT_ARGUMENT(encoder);
  if (IREE_UNLIKELY(!encoder->in_block)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "no block is actively being recorded");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // Per-block metadata contains the query map sized based on the number of
  // commands recorded.
  const iree_hal_amdgpu_query_id_scratch_t* query_ids = encoder->query_ids;
  const iree_host_size_t total_metadata_size =
      sizeof(iree_hal_amdgpu_device_command_block_t) +
      (query_ids ? query_ids->count *
                       sizeof(iree_hal_amdgpu_device_command_query_id_t)
                 : 0);

  // TODO(benvanik): sort the commands by type? Would help us use larger
  // workgroup sizes (but may need padding); today if we used >1 workgroup size
  // and any command in the workgroup was different we'd end up in unhappy land.

  // Upload the block metadata to each device.
  for (iree_host_size_t i = 0; i < encoder->device_count; ++i) {
    iree_hal_amdgpu_device_encoder_state_t* device_state =
        &encoder->device_state[i];

    // Allocate block metadata from the block arena.
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_command_block_t*
        block_metadata = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_block_arena_allocate(
        &device_state->metadata_arena, total_metadata_size,
        (void**)&block_metadata));

    // Move the current block storage to the end of the block linked list.
    iree_hal_amdgpu_block_t* cmd_block = device_state->cmd_block.current;
    device_state->cmd_block.current = NULL;
    cmd_block->next = NULL;
    if (device_state->cmd_block.tail) {
      device_state->cmd_block.tail->next = cmd_block;
    } else {
      device_state->cmd_block.head = cmd_block;
    }
    device_state->cmd_block.tail = cmd_block;

    // Store the command buffer block metadata pointer in the device storage
    // block. This is free space that we can reuse without needing to keep
    // track of a growing list of blocks. Consider the list of block pool blocks
    // as an iovec and the metadata is attached to one of the entries.
    cmd_block->user_data[0] = (uint64_t)block_metadata;

    // The metadata references the command block.
    block_metadata->max_packet_count = encoder->aql_packet_offset;
    block_metadata->command_count = encoder->command_count;
    block_metadata->commands =
        (const iree_hal_amdgpu_device_cmd_t*)cmd_block->ptr;

    // Each device gets a copy of the query IDs in its local memory.
    if (query_ids) {
      block_metadata->query_map.max_control_query_count =
          query_ids->next.control_id;
      block_metadata->query_map.max_dispatch_query_count =
          query_ids->next.dispatch_id;
      iree_memcpy_stream_dst(block_metadata->query_map.query_ids,
                             query_ids->values,
                             query_ids->count * sizeof(query_ids->values[0]));
    }
  }

  // Reset encoder state for the next block.
  encoder->command_count = 0;
  encoder->peak_aql_packet_count = encoder->aql_packet_offset;
  encoder->aql_packet_offset = 0;
  encoder->peak_kernarg_size = encoder->kernarg_offset;
  encoder->kernarg_offset = 0;
  iree_hal_amdgpu_query_id_scratch_reset(encoder->query_ids);

  encoder->in_block = 0;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_command_encoder_append_cmd(
    iree_hal_amdgpu_command_encoder_t* encoder,
    iree_hal_amdgpu_device_cmd_type_t type,
    iree_hal_amdgpu_device_cmd_flags_t flags, uint16_t aql_packet_count,
    uint16_t kernarg_size, uint16_t kernarg_alignment,
    void* IREE_AMDGPU_DEVICE_PTR** out_device_cmds,
    uint32_t* out_kernarg_offset);

// Splits the current block in two by inserting a branch command.
// The block must have at least one command slot available so the branch can be
// inserted. All active state is carried over to the new block and since the
// split is internal no new block label is needed (the only incoming edge is
// from the current block that is being split).
static iree_status_t iree_hal_amdgpu_command_encoder_split_block(
    iree_hal_amdgpu_command_encoder_t* encoder) {
  IREE_ASSERT_ARGUMENT(encoder);
  if (IREE_UNLIKELY(!encoder->in_block)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "no block is actively being recorded");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // Insert a branch to the next block (that we will be beginning below).
  const uint32_t target_block = encoder->block_count + 1;
  iree_hal_amdgpu_device_cmd_branch_t** device_cmds = NULL;
  uint32_t kernarg_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_command_encoder_append_cmd(
              encoder, IREE_HAL_AMDGPU_DEVICE_CMD_BRANCH,
              IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_QUEUE_AWAIT_BARRIER,
              IREE_HAL_AMDGPU_DEVICE_CMD_BRANCH_AQL_PACKET_COUNT,
              IREE_HAL_AMDGPU_DEVICE_CMD_BRANCH_KERNARG_SIZE,
              IREE_HAL_AMDGPU_DEVICE_CMD_BRANCH_KERNARG_ALIGNMENT,
              (void***)&device_cmds, &kernarg_offset));
  for (iree_host_size_t i = 0; i < encoder->device_count; ++i) {
    device_cmds[i]->kernarg_offset = kernarg_offset;
    device_cmds[i]->target_block = target_block;
  }

  // End the current block now that we've terminated it with the branch.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_command_encoder_end_block(encoder));

  // Begin the new block. We don't care about the ordinal here as we already
  // calculated it above and no one will be able to branch to it.
  uint32_t block_ordinal = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_command_encoder_begin_block(encoder, &block_ordinal));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Appends a command to the current block.
// The command header will be initialized based on the current encoding state
// (such as whether a barrier is required) and the accounting of kernarg storage
// will be calculated. Callers must populate any additional command-specific
// information or flags per-device in the returned |out_device_cmds| list.
//
// If the current block has exceeded its capacity in the target device memory
// block it will be split and joined with a branch.
//
// NOTE: the command data returned should be treated as write-only as its memory
// lives on device and may be uncached. If we really wanted to optimize things
// we'd probably be producing commands in host memory and doing a non-temporal
// memcpy but that likely adds latency in the case of smaller command buffers.
// This way we are always writing a bulk of the data while we go and
// finalization per-device is just a quick fixup.
static iree_status_t iree_hal_amdgpu_command_encoder_append_cmd(
    iree_hal_amdgpu_command_encoder_t* encoder,
    iree_hal_amdgpu_device_cmd_type_t type,
    iree_hal_amdgpu_device_cmd_flags_t flags, uint16_t aql_packet_count,
    uint16_t kernarg_size, uint16_t kernarg_alignment,
    void* IREE_AMDGPU_DEVICE_PTR** out_device_cmds,
    uint32_t* out_kernarg_offset) {
  IREE_ASSERT_ARGUMENT(!kernarg_size || out_kernarg_offset);

  if (IREE_UNLIKELY(!encoder->in_block)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "no block is actively being recorded");
  }

  // Split the block when we have only one command remaining unless the command
  // being recorded is a terminator. We should always have one command available
  // for use (start with 1, check here that there's always 1, and split if
  // there's 0).
  //
  // Though rare it's possible the queue may not have enough space for the
  // requested AQL packets - usually command capacity is conservative enough to
  // be under any device limits but we do not want to fail once scheduled (as
  // that's a device loss).
  const bool is_terminator = type == IREE_HAL_AMDGPU_DEVICE_CMD_BRANCH ||
                             type == IREE_HAL_AMDGPU_DEVICE_CMD_RETURN;
  const bool exceeded_command_capacity =
      !is_terminator && encoder->command_count + 1 >= encoder->command_capacity;
  const bool exceeded_aql_packet_capacity =
      encoder->aql_packet_offset + aql_packet_count >
      encoder->max_aql_packet_capacity;
  const bool exceeded_kernarg_capacity =
      kernarg_size > 0 &&
      iree_host_align(encoder->kernarg_offset, kernarg_alignment) +
              kernarg_size >
          encoder->max_kernarg_capacity;
  if (exceeded_command_capacity || exceeded_aql_packet_capacity ||
      exceeded_kernarg_capacity) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_command_encoder_split_block(encoder));
  }

  // --- WARNING ------------------------------------------------------------ //
  // The split above may have reset encoder state and any state loaded from it
  // must be requeried for subsequent usage.
  // --- WARNING ------------------------------------------------------------ //

  // We update a scratch header and copy it to each device at the end.
  iree_hal_amdgpu_device_cmd_header_t cmd = {
      .type = type,
  };

  // Acquire the next command from the device memory block.
  uint16_t command_offset = encoder->command_count;
  ++encoder->command_count;

  // If a barrier is pending (a prior command requires exclusive execution) then
  // consume it. Since commands are processed in order all following commands
  // recorded will also wait.
  cmd.flags = flags;
  if (encoder->barrier_pending) {
    cmd.flags |= IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_QUEUE_AWAIT_BARRIER;
    encoder->barrier_pending = 0;
  }

  // Consume the requested packets from the AQL packet queue.
  // Note that these are relative values as actual queue offsets are calculated
  // when the commands are issued.
  cmd.packet_offset = encoder->aql_packet_offset;
  encoder->aql_packet_offset += aql_packet_count;

  // Kernarg storage is committed when the commands are executed; here we are
  // just tracking how much space is required and the offset of each requested
  // block in memory. Note that kernargs have an alignment that we need to
  // preserve.
  if (kernarg_size > 0) {
    iree_host_size_t kernarg_offset =
        iree_host_align(encoder->kernarg_offset, kernarg_alignment);
    encoder->kernarg_offset = encoder->kernarg_offset + kernarg_size;
    *out_kernarg_offset = (uint16_t)kernarg_offset;
  }

  // Assign tracing query IDs for the command (if needed).
  iree_hal_amdgpu_assign_cmd_query_ids(encoder->query_ids, cmd.type);

  // TODO(benvanik): find a way to avoid needing to do this here - if we could
  // instead pass back the command header and make callers responsible they
  // could do this themselves in the loop they are likely already performing.
  // For now this keeps things contained and simpler in cases where there is no
  // per-device command information.
  for (uint32_t i = 0; i < encoder->device_count; ++i) {
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_cmd_header_t* device_cmd =
        (iree_hal_amdgpu_device_cmd_header_t*)encoder->device_state[i]
            .cmd_block.current->ptr +
        command_offset;
    iree_memcpy_stream_dst(device_cmd, &cmd, sizeof(device_cmd));
    encoder->device_cmds[i] = device_cmd;
  }

  *out_device_cmds = (void**)encoder->device_cmds;
  return iree_ok_status();
}

// Emplaces a data buffer into the block storage on |device_index| and returns
// the pointer in device memory. The returned pointer is aligned to
// iree_hal_amdgpu_max_align_t bytes.
static iree_status_t iree_hal_amdgpu_command_encoder_emplace_data(
    iree_hal_amdgpu_command_encoder_t* encoder, iree_host_size_t device_index,
    iree_const_byte_span_t data, IREE_AMDGPU_DEVICE_PTR void** out_ptr) {
  iree_hal_amdgpu_device_encoder_state_t* device_state =
      &encoder->device_state[device_index];

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_block_arena_allocate(
      &device_state->storage_arena, data.data_length, (void**)&out_ptr));

  // Stream data to device memory (we will never read it on the host).
  iree_memcpy_stream_dst(*out_ptr, data.data, data.data_length);

  return iree_ok_status();
}

// Emplaces two data buffers into the block storage on |device_index| and
// returns the pointer to each in device memory. This is more efficient than
// emplacing both individually. Alignment is performed on each data span to
// iree_hal_amdgpu_max_align_t bytes.
static iree_status_t iree_hal_amdgpu_command_encoder_emplace_data_concat(
    iree_hal_amdgpu_command_encoder_t* encoder, iree_host_size_t device_index,
    iree_const_byte_span_t data0, iree_const_byte_span_t data1,
    IREE_AMDGPU_DEVICE_PTR void** out_ptr0,
    IREE_AMDGPU_DEVICE_PTR void** out_ptr1) {
  iree_hal_amdgpu_device_encoder_state_t* device_state =
      &encoder->device_state[device_index];

  // Allocate combined block with alignment on each pointer.
  // The allocation will start at the alignment and the total size will be
  // aligned so we just need to ensure our internal data1 pointer is aligned.
  const iree_host_size_t data1_offset =
      iree_host_align(data0.data_length, iree_hal_amdgpu_max_align_t);
  const iree_host_size_t total_size = data1_offset + data1.data_length;
  uint8_t* base_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_block_arena_allocate(
      &device_state->storage_arena, total_size, (void**)&base_ptr));
  *out_ptr0 = base_ptr;
  *out_ptr1 = base_ptr + data1_offset;

  // Stream data to device memory (we will never read it on the host).
  iree_memcpy_stream_dst(*out_ptr0, data0.data, data0.data_length);
  iree_memcpy_stream_dst(*out_ptr1, data1.data, data1.data_length);

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_command_buffer_options_t
//===----------------------------------------------------------------------===//

#define IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_HOST_SMALL_BLOCK_SIZE (4 * 1024 - 32)
#define IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_HOST_LARGE_BLOCK_SIZE (32 * 1024)
#define IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_METADATA_BLOCK_SIZE (8 * 1024 - 32)
#define IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_STORAGE_BLOCK_SIZE (64 * 1024)

void iree_hal_amdgpu_command_buffer_options_initialize(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_amdgpu_command_buffer_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));

  out_options->device_allocator = device_allocator;
  out_options->mode = mode;
  out_options->command_categories = command_categories;
  out_options->queue_affinity = queue_affinity;
  out_options->binding_capacity = binding_capacity;

  out_options->recording_flags =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORDING_FLAG_NONE;

  out_options->block_aql_packet_count =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_MAX_BLOCK_AQL_PACKET_COUNT;
}

iree_status_t iree_hal_amdgpu_command_buffer_options_verify(
    const iree_hal_amdgpu_command_buffer_options_t* options) {
  // Verify we can use the execution queues the command buffer is targeting.
  if (options->block_aql_packet_count <
          IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_BLOCK_AQL_PACKET_COUNT ||
      options->block_aql_packet_count >
          IREE_HAL_AMDGPU_COMMAND_BUFFER_MAX_BLOCK_AQL_PACKET_COUNT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "block_aql_packet_count must be between %d and %d "
        "but %" PRIhsz " was requested",
        IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_BLOCK_AQL_PACKET_COUNT,
        IREE_HAL_AMDGPU_COMMAND_BUFFER_MAX_BLOCK_AQL_PACKET_COUNT,
        options->block_aql_packet_count);
  }

  // Verify any devices are targeted. We could probably handle recording for no
  // devices and use that as a baseline performance metric but today zero
  // devices would definitely be unintended by users.
  int device_count =
      iree_hal_amdgpu_device_affinity_count(options->device_affinity);
  if (device_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one physical device must be specified");
  }

  // Host block pools use a small amount of the block space for internal
  // accounting and we must ensure that there's enough usable for our minimum
  // contiguous allocations. This may mean they have to be slightly oversized
  // if we end up needing a power-of-two allocation.
  if (options->host_block_pools->small.usable_block_size <
      IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_HOST_SMALL_BLOCK_SIZE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "host small block pool must have at least %d usable bytes per block "
        "but only has %" PRIhsz,
        IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_HOST_SMALL_BLOCK_SIZE,
        options->host_block_pools->small.usable_block_size);
  }
  if (options->host_block_pools->large.usable_block_size <
      IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_HOST_LARGE_BLOCK_SIZE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "host small block pool must have at least %d usable bytes per block "
        "but only has %" PRIhsz,
        IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_HOST_LARGE_BLOCK_SIZE,
        options->host_block_pools->large.usable_block_size);
  }

  // Verify that the device-side pools will fit our data structures that we
  // require to be contiguous.
  for (int i = 0; i < device_count; ++i) {
    if (options->device_block_pools[i]->small.block_size <
        IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_METADATA_BLOCK_SIZE) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "device[%d] small block pool must have at least "
          "%d bytes per block but only has %" PRIdsz,
          i, IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_METADATA_BLOCK_SIZE,
          options->device_block_pools[i]->small.block_size);
    }
    if (options->device_block_pools[i]->large.block_size <
        IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_STORAGE_BLOCK_SIZE) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "device[%d] large block pool must have at least "
          "%d bytes per block but only has %" PRIdsz,
          i, IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_STORAGE_BLOCK_SIZE,
          options->device_block_pools[i]->large.block_size);
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_command_buffer_t
//===----------------------------------------------------------------------===//

// A host-side HAL command buffer wrapping a device-side handle.
// Each command buffer can be recorded for multiple physical devices where all
// data will reside in device-local memory. Expensive validation, encoding
// logic, and resource tracking are amortized across all devices the command
// buffer is recorded for.
//
// Once finalized a command buffer owns per-device references to a list of
// memory blocks holding command buffer metadata such as the device-side
// iree_hal_amdgpu_device_command_buffer_t and its CFG blocks and list of
// memory blocks holding any embedded data used by the command buffer such as
// bindings/constants and inline buffer uploads.
//
// During recording iree_hal_amdgpu_command_encoder_t is used to track the
// current block, total counts required for allocating final device resources,
// and per-device arenas. Once recording completes the encoder is discarded.
//
// A goal was to have the cost of recording a command buffer for a single device
// not suffer from the support for multiple devices - in most cases no
// additional memory during recording and at rest will be allocated beyond what
// would have been needed if this only supported a single device and besides a
// predictable 1-trip loop has no meaningful additional per-command overhead.
//
// Attention is paid to recording performance even though we expect most large
// command buffers to be recorded at initialization time and reused. Though IREE
// compiled programs can easily reuse command buffers applications directly
// using the HAL may not be able to and we want one-shot command buffers to not
// be super inefficient. A one-shot command buffer will never achieve the
// latency of a CUDA stream-like API but 99% of application-level operations are
// copies that either could benefit from batching or could be done unbatched at
// the queue level. The HAL as a whole focuses more on the expensive parts of
// the program than the cheap/garbage-y ones: it's best to have deterministic
// excellent performance on critical workloads than to have low latency in
// naively authored applications. We do upload block metadata and embedded
// storage data to devices as we record the command buffer such that after
// recording completes we can launch the command buffer immediately but we
// cannot time-travel to begin executing while the command buffer is still
// recording.
//
// One thing we could support is using SDMA to upload the command buffer
// contents asynchronously with the recording. There are some notes sprinkled
// around about supporting relocatable command buffers by storing only relative
// offsets in all data. If command buffers _were_ relocatable we could record on
// the host and broadcast to all devices with SDMA, or record to a single device
// and use P2P SDMA to replicate it.
//
// Thread-compatible during recording and thread-safe once finalized. Multiple
// threads are allowed to submit the command buffer for execution concurrently.
typedef struct iree_hal_amdgpu_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;

  // Bitmap of physical devices the command buffer has been prepared for.
  // Not all devices may have a copy of the command buffer. Population count
  // denotes the number of devices.
  iree_hal_amdgpu_device_affinity_t device_affinity;

  // Maximum kernarg capacity required to execute any block in the command
  // buffer, in bytes.
  iree_host_size_t max_kernarg_capacity;

  // State used only during recording.
  struct {
    // Arena used during recording. Reset after recording completes.
    // Blocks are allocated from the host pool.
    iree_arena_allocator_t host_arena;

    // The last executable referenced by a dispatch. Since a large majority of
    // the time there is a single executable we fast-path this to avoid
    // thrashing the resource set LRU cache and growing the resource set
    // allocation/cleanup time.
    iree_hal_executable_t* last_executable;

    // Active command buffer block encoder for all devices if the command buffer
    // is recording. Encoders are acquired from the small block pool when
    // recording begins and discarded upon finalization.
    iree_hal_amdgpu_command_encoder_t* encoder;
  } recording_state;

  // Retains references to any HAL resources used by the command buffer.
  // Only one reference is needed regardless of the number of devices the
  // command buffer has been instantiated on.
  iree_hal_resource_set_t* resource_set;

  // Compacted list of device-side copies of the command buffer.
  // Only those device ordinals specified by the device_affinity bitmap are
  // present. A device affinity of 0b110 would lead to two device command
  // buffers in the list at [0] and [1].
  struct {
    // Device-side command buffer descriptor used to launch execution.
    // Only allocated after recording has ended. Lives within the metadata pool.
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_command_buffer_t* handle;
    // Pool used for the device-side command blocks.
    iree_hal_amdgpu_block_pool_t* cmd_block_pool;
    // List of blocks containing commands allocated from the device block pool.
    iree_hal_amdgpu_block_t* cmd_block_head;
    // Pool used for the device-side metadata like the command buffer and CFG
    // blocks that is usually smaller than the storage pool meant for bulk data.
    iree_hal_amdgpu_block_pool_t* metadata_pool;
    // List of blocks allocated from the device block pool storing the
    // device-side command buffer metadata.
    iree_hal_amdgpu_block_t* metadata_head;
    // Device block pool used for command buffer storage. Must be large enough
    // to contain any single dynamic allocation (64KB for updates) and
    // determines the maximum number of commands per command buffer block.
    iree_hal_amdgpu_block_pool_t* storage_pool;
    // List of blocks allocated from the device block pool storing the
    // device-side command buffer embedded data.
    iree_hal_amdgpu_block_t* storage_head;
  } device_state[/*popcnt(device_affinity)*/];
} iree_hal_amdgpu_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_amdgpu_command_buffer_vtable;

static iree_hal_amdgpu_command_buffer_t* iree_hal_amdgpu_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_command_buffer_vtable);
  return (iree_hal_amdgpu_command_buffer_t*)base_value;
}

iree_status_t iree_hal_amdgpu_command_buffer_create(
    const iree_hal_amdgpu_command_buffer_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_command_buffer = NULL;

  // Verify that the provided options are supported across all target devices.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_command_buffer_options_verify(options));

  // Allocate command buffer storage as:
  // { command buffer, device_state[], validation_state }
  iree_hal_amdgpu_command_buffer_t* command_buffer = NULL;
  const iree_host_size_t validation_state_size =
      iree_hal_command_buffer_validation_state_size(options->mode,
                                                    options->binding_capacity);
  const iree_host_size_t device_count =
      iree_hal_amdgpu_device_affinity_count(options->device_affinity);
  const iree_host_size_t total_size =
      sizeof(*command_buffer) +
      device_count * sizeof(command_buffer->device_state[0]) +
      validation_state_size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                (void**)&command_buffer));
  iree_hal_command_buffer_initialize(
      options->device_allocator, options->mode, options->command_categories,
      options->queue_affinity, options->binding_capacity,
      (uint8_t*)command_buffer + total_size - validation_state_size,
      &iree_hal_amdgpu_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->device_affinity = options->device_affinity;
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    command_buffer->device_state[i].cmd_block_pool =
        &options->device_block_pools[i]->large;
    command_buffer->device_state[i].metadata_pool =
        &options->device_block_pools[i]->small;
    command_buffer->device_state[i].storage_pool =
        &options->device_block_pools[i]->large;
  }

  // Setup initial recording state while we have all of the options available.
  // If we supported re-recording we'd need to save this information so that we
  // can reinitialize the state in _begin.
  iree_arena_initialize(&options->host_block_pools->large,
                        &command_buffer->recording_state.host_arena);
  iree_status_t status = iree_hal_amdgpu_command_encoder_initialize(
      options, device_count, &command_buffer->recording_state.host_arena,
      &command_buffer->recording_state.encoder);

  // Allocate resource set from a host block pool.
  // We expect to have a small number of resources but this may not be the case:
  // we may have 1 executable if every binding is indirect or 1000 buffers if
  // the user is silly and passed in 1000 buffers. We err on the side of not
  // wasting so much memory right now with the intent that everyone should be
  // running with reusable command buffers if they care. If we did go larger we
  // would want to repack/trim the resource set when freezing (if not one-shot
  // where it doesn't matter). The risk with large allocs is that a user with
  // 10000 reusable command buffers will eat all that memory forever.
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_allocate(&options->host_block_pools->small,
                                            &command_buffer->resource_set);
  }

  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_release(&command_buffer->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_command_buffer_t* command_buffer =
      iree_hal_amdgpu_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release all device resources (if we completed recording).
  const iree_host_size_t device_count =
      iree_hal_amdgpu_device_affinity_count(command_buffer->device_affinity);
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    iree_hal_amdgpu_block_pool_release_list(
        command_buffer->device_state[i].cmd_block_pool,
        command_buffer->device_state[i].cmd_block_head);
    iree_hal_amdgpu_block_pool_release_list(
        command_buffer->device_state[i].metadata_pool,
        command_buffer->device_state[i].metadata_head);
    iree_hal_amdgpu_block_pool_release_list(
        command_buffer->device_state[i].storage_pool,
        command_buffer->device_state[i].storage_head);
  }

  // Recording state should have been cleaned up but may not be if there was a
  // failure during recording.
  iree_hal_amdgpu_command_encoder_deinitialize(
      command_buffer->recording_state.encoder);
  iree_arena_deinitialize(&command_buffer->recording_state.host_arena);

  // Release all resources the command buffer is retaining (buffers,
  // executables, etc) and return the resource set memory back to the host pool.
  iree_hal_resource_set_free(command_buffer->resource_set);

  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_amdgpu_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_amdgpu_command_buffer_vtable);
}

iree_status_t iree_hal_amdgpu_command_buffer_query_execution_state(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t device_ordinal,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_command_buffer_t**
        out_device_command_buffer,
    iree_host_size_t* out_max_kernarg_capacity) {
  iree_hal_amdgpu_command_buffer_t* command_buffer =
      iree_hal_amdgpu_command_buffer_cast(base_command_buffer);

  const int device_count =
      iree_hal_amdgpu_device_affinity_count(command_buffer->device_affinity);
  if (IREE_UNLIKELY(device_ordinal >= device_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "device ordinal %" PRIhsz
        " out of range; command buffer was allocated for %d devices",
        device_ordinal, device_count);
  }

  iree_hal_amdgpu_device_command_buffer_t* device_command_buffer =
      command_buffer->device_state[device_ordinal].handle;
  if (IREE_UNLIKELY(device_command_buffer == NULL)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "command buffer was not recorded for device ordinal %" PRIhsz
        "; queue affinity provided during construction must include all "
        "devices the command buffer may be executed on");
  }

  *out_device_command_buffer = device_command_buffer;
  *out_max_kernarg_capacity = command_buffer->max_kernarg_capacity;

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_command_buffer_t* command_buffer =
      iree_hal_amdgpu_command_buffer_cast(base_command_buffer);

  // Today we only support a single recording. If we want to allow re-recording
  // (we don't) we'd need to preserve enough information to initialize the
  // encoder again.
  iree_hal_amdgpu_command_encoder_t* encoder =
      command_buffer->recording_state.encoder;
  if (IREE_UNLIKELY(!encoder)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffers can only be recorded once");
  }

  // Begin a new block.
  uint32_t block_ordinal = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_command_encoder_begin_block(encoder, &block_ordinal));
  IREE_ASSERT_EQ(block_ordinal, 0, "must start at block 0");

  return iree_ok_status();
}

// Finalizes command buffer recording by uploading metadata structures and
// possibly compacting command storage. Must be called after all blocks have
// completed recording.
static iree_status_t iree_hal_amdgpu_command_buffer_finalize(
    iree_hal_amdgpu_command_buffer_t* command_buffer) {
  IREE_ASSERT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_command_encoder_t* encoder =
      command_buffer->recording_state.encoder;
  if (IREE_UNLIKELY(encoder->in_block)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "last block was not ended prior to finalizing; "
                            "all recording must have completed");
  }

  // Capture host-side metadata used for submissions.
  command_buffer->max_kernarg_capacity = encoder->peak_kernarg_size;

  // Capture device-side metadata used for scheduling.
  for (iree_host_size_t i = 0; i < encoder->device_count; ++i) {
    iree_hal_amdgpu_device_encoder_state_t* device_state =
        &encoder->device_state[i];

    // Allocate the command buffer wrapper now that we know the final block
    // count.
    //
    // TODO(benvanik): this _may_ allocate a new block just for the metadata and
    // that's unfortunate: we should do something better to avoid 1% utilization
    // of a new block.
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_command_buffer_t* handle =
        NULL;
    const iree_host_size_t handle_size =
        sizeof(*handle) + encoder->block_count * sizeof(handle->blocks[0]);
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_block_arena_allocate(
        &device_state->metadata_arena, handle_size, (void**)&handle));
    handle->max_kernarg_capacity = encoder->peak_kernarg_size;
    handle->block_count = encoder->block_count;
    command_buffer->device_state[i].handle = handle;

    // Take ownership of the command block linked list and associated metadata.
    // After this point the command buffer will be responsible for cleaning up
    // the block storage when needed.
    iree_hal_amdgpu_block_t* cmd_head = device_state->cmd_block.head;
    device_state->cmd_block.head = NULL;
    device_state->cmd_block.tail = NULL;
    IREE_ASSERT(!device_state->cmd_block.current);
    command_buffer->device_state[i].cmd_block_head = cmd_head;
    iree_hal_amdgpu_block_t* cmd_block = cmd_head;
    for (iree_host_size_t j = 0; j < encoder->block_count;
         ++j, cmd_block = cmd_block->next) {
      handle->blocks[j] =
          (iree_hal_amdgpu_device_command_block_t*)cmd_block->user_data[0];
    }
    command_buffer->device_state[i].metadata_head =
        iree_hal_amdgpu_block_arena_release_blocks(
            &device_state->metadata_arena);

    // Take ownership of the embedded data storage linked list.
    command_buffer->device_state[i].storage_head =
        iree_hal_amdgpu_block_arena_release_blocks(
            &device_state->storage_arena);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_command_buffer_t* command_buffer =
      iree_hal_amdgpu_command_buffer_cast(base_command_buffer);

  iree_hal_amdgpu_command_encoder_t* encoder =
      command_buffer->recording_state.encoder;

  // Return from the command buffer.
  // A barrier is implicit and this will also take care of any pending barrier
  // bit if one is set. This _should_ always be able to be appended in the
  // current block without needing a split as we reserve one command slot for
  // terminators while recording.
  iree_hal_amdgpu_device_cmd_return_t** device_cmds = NULL;
  uint32_t kernarg_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_command_encoder_append_cmd(
      encoder, IREE_HAL_AMDGPU_DEVICE_CMD_RETURN,
      IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_QUEUE_AWAIT_BARRIER,
      IREE_HAL_AMDGPU_DEVICE_CMD_RETURN_AQL_PACKET_COUNT,
      IREE_HAL_AMDGPU_DEVICE_CMD_RETURN_KERNARG_SIZE,
      IREE_HAL_AMDGPU_DEVICE_CMD_RETURN_KERNARG_ALIGNMENT,
      (void***)&device_cmds, &kernarg_offset));
  for (iree_host_size_t i = 0; i < encoder->device_count; ++i) {
    device_cmds[i]->kernarg_offset = kernarg_offset;
  }

  // Flush the current block and reset the encoder.
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_command_encoder_end_block(encoder));

  // Finalize the command buffer and upload metadata to all devices.
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_command_buffer_finalize(command_buffer));

  // Deallocate the block encoders by resetting the arena they were allocated
  // from and cleaning up other host-only recording state.
  iree_hal_amdgpu_command_encoder_deinitialize(encoder);
  iree_arena_deinitialize(&command_buffer->recording_state.host_arena);
  memset(&command_buffer->recording_state, 0,
         sizeof(command_buffer->recording_state));

  // Freeze the resource set as all resources have been added and it is now
  // immutable. It can be nested within another resource set if it needs
  // extension.
  iree_hal_resource_set_freeze(command_buffer->resource_set);

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  // TODO(benvanik): if we route these to tooling/profilers we'll want to enable
  // them with some flag bits.
#if !(IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE)
  // Tracing disabled - this wouldn't go anywhere.
  return iree_ok_status();
#endif  // !IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

  iree_hal_amdgpu_command_buffer_t* command_buffer =
      iree_hal_amdgpu_command_buffer_cast(base_command_buffer);

  // TODO(benvanik): intern label and source location.
  // Tracy requires that the pointers remain valid until the termination of
  // the program _or_ they be dynamically allocated for single usage. The issue
  // is that with reusable command buffers we both don't know the lifetime of
  // the input we were given or how many times the command buffer will be used.
  //
  // One solution would be an big hash map. That adds overhead during recording
  // but is free from then on and for a majority of our command buffers that are
  // reusable will be in the noise. The set of unique source locations and
  // labels should be fairly small (we don't allow dynamic values) and these
  // debug groups aren't super common today.
  const char* label_literal = "todo_label_interning";
  static iree_hal_amdgpu_trace_src_loc_t dummy_src_loc = {
      .name = NULL,
      .function = NULL,
      .file = NULL,
      .line = 0,
      .color = 0,
  };
  iree_hal_amdgpu_trace_src_loc_ptr_t src_loc =
      (iree_hal_amdgpu_trace_src_loc_ptr_t)&dummy_src_loc;

  iree_hal_amdgpu_command_encoder_t* encoder =
      command_buffer->recording_state.encoder;

  // Begin scope with full barrier so that all prior work completes before any
  // nested commands begin execution. This needs us to set the flag as if an
  // execution barrier had been scheduled so that the next command recorded
  // waits.
  iree_hal_amdgpu_device_cmd_debug_group_begin_t** device_cmds = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_command_encoder_append_cmd(
      encoder, IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_BEGIN,
      IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_QUEUE_AWAIT_BARRIER,
      IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_BEGIN_AQL_PACKET_COUNT,
      /*kernarg_size=*/0, /*kernarg_alignment=*/0, (void***)&device_cmds,
      NULL));

  // Loop over each device this command buffer is being broadcast to apply
  // per-device information.
  for (iree_host_size_t i = 0; i < encoder->device_count; ++i) {
    iree_hal_amdgpu_device_cmd_debug_group_begin_t* cmd = device_cmds[i];
    cmd->src_loc = src_loc;
    cmd->label_literal = (uint64_t)label_literal;
    cmd->label_literal_length = label.size;
    cmd->color = label_color.value;
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
#if !(IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE)
  // Tracing disabled - this wouldn't go anywhere.
  return iree_ok_status();
#endif  // !IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
  iree_hal_amdgpu_command_buffer_t* command_buffer =
      iree_hal_amdgpu_command_buffer_cast(base_command_buffer);

  iree_hal_amdgpu_command_encoder_t* encoder =
      command_buffer->recording_state.encoder;

  // End scope with full barrier so that any nested commands complete before
  // capturing.
  iree_hal_amdgpu_device_cmd_debug_group_end_t** device_cmds = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_command_encoder_append_cmd(
      encoder, IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_END,
      IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_QUEUE_AWAIT_BARRIER,
      IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_END_AQL_PACKET_COUNT,
      /*kernarg_size=*/0, /*kernarg_alignment=*/0, (void***)&device_cmds,
      NULL));

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_amdgpu_command_buffer_t* command_buffer =
      iree_hal_amdgpu_command_buffer_cast(base_command_buffer);

  iree_hal_amdgpu_command_encoder_t* encoder =
      command_buffer->recording_state.encoder;

  // Force the next command to have a barrier flag on it and wait until all
  // prior commands have completed.
  encoder->barrier_pending = 1;

  // TODO(benvanik): emit a barrier command if needed.
  // I'm not sure it is today but we have it encoded for cases where we can't
  // carry across barrier bits (between blocks/etc).
  //
  // iree_hal_amdgpu_device_cmd_barrier_t** device_cmds = NULL;
  // IREE_RETURN_IF_ERROR(iree_hal_amdgpu_command_encoder_append_cmd(
  //     command_buffer->encoder, IREE_HAL_AMDGPU_DEVICE_CMD_BARRIER,
  //     IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_QUEUE_AWAIT_BARRIER,
  //     IREE_HAL_AMDGPU_DEVICE_CMD_BARRIER_AQL_PACKET_COUNT,
  //     /*kernarg_size=*/0, /*kernarg_alignment=*/0, (void***)&device_cmds,
  //     NULL));
  //
  // The case where this would be most helpful is on fork/join of multiple
  // commands. If we want to have an acquire, multiple dispatches, and release
  // we'd need the barrier packets to anchor those scope operations on.
  // The command buffer API doesn't let us predict subsequent commands on
  // barriers and without fixup we don't know when it's worth adding the extra
  // packet (which introduces additional issue latency). This is a place where
  // the command buffer API could be improved around fork/join subgraphs.

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_amdgpu_command_buffer_t* command_buffer =
      iree_hal_amdgpu_command_buffer_cast(base_command_buffer);

  // TODO(benvanik): WIP API and may change; signals the given event allowing
  // waiters to proceed.
  (void)command_buffer;
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "events not implemented");

  return status;
}

static iree_status_t iree_hal_amdgpu_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_amdgpu_command_buffer_t* command_buffer =
      iree_hal_amdgpu_command_buffer_cast(base_command_buffer);

  // TODO(benvanik): WIP API and may change; resets the given event to
  // unsignaled.
  (void)command_buffer;
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "events not implemented");

  return status;
}

static iree_status_t iree_hal_amdgpu_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_amdgpu_command_buffer_t* command_buffer =
      iree_hal_amdgpu_command_buffer_cast(base_command_buffer);

  // TODO(benvanik): WIP API and may change; waits on the list of events and
  // enacts the specified set of barriers. Implementations without fine-grained
  // tracking can treat this as an execution_barrier and ignore the
  // memory/buffer barriers provided.
  (void)command_buffer;
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "events not implemented");

  return status;
}

static iree_status_t iree_hal_amdgpu_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  iree_hal_amdgpu_command_buffer_t* command_buffer =
      iree_hal_amdgpu_command_buffer_cast(base_command_buffer);

  // TODO(benvanik): WIP API and may change; this is likely to become an
  // madvise-like command that can be used to control prefetching and other
  // cache behavior. The current discard behavior is a hint that the buffer
  // contents will never be used again and that if they are in a cache they need
  // not be written back to global memory.
  (void)command_buffer;

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_amdgpu_command_buffer_t* command_buffer =
      iree_hal_amdgpu_command_buffer_cast(base_command_buffer);

  // Translate the target buffer to its device-side representations. Translation
  // is currently device-independent.
  iree_hal_amdgpu_device_buffer_ref_t resolved_target_ref = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_translate_device_buffer_ref(
                           target_ref, &resolved_target_ref),
                       "resolving target_ref");

  // Retain buffers (if specified) for the lifetime of the command buffer.
  if (target_ref.buffer) {
    iree_hal_resource_t* resources[] = {
        (iree_hal_resource_t*)target_ref.buffer,
    };
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
        command_buffer->resource_set, IREE_ARRAYSIZE(resources), resources));
  }

  iree_hal_amdgpu_command_encoder_t* encoder =
      command_buffer->recording_state.encoder;

  // Append command to all devices.
  iree_hal_amdgpu_device_cmd_fill_buffer_t** device_cmds = NULL;
  uint32_t kernarg_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_command_encoder_append_cmd(
      encoder, IREE_HAL_AMDGPU_DEVICE_CMD_FILL_BUFFER,
      IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_NONE,
      IREE_HAL_AMDGPU_DEVICE_CMD_FILL_BUFFER_AQL_PACKET_COUNT,
      IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_SIZE,
      IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_ALIGNMENT,
      (void***)&device_cmds, &kernarg_offset));

  // Loop over each device this command buffer is being broadcast to apply
  // per-device information.
  for (iree_host_size_t i = 0; i < encoder->device_count; ++i) {
    iree_hal_amdgpu_device_cmd_fill_buffer_t* cmd = device_cmds[i];
    cmd->kernarg_offset = kernarg_offset;
    cmd->target_ref = resolved_target_ref;
    memcpy(&cmd->pattern, pattern, pattern_length);
    cmd->pattern_length = pattern_length;
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_amdgpu_command_buffer_t* command_buffer =
      iree_hal_amdgpu_command_buffer_cast(base_command_buffer);

  // Translate the target buffer to its device-side representations. Translation
  // is currently device-independent.
  iree_hal_amdgpu_device_buffer_ref_t resolved_target_ref = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_translate_device_buffer_ref(
                           target_ref, &resolved_target_ref),
                       "resolving target_ref");

  // If the size exceeds the data block capacity we cannot allocate it - we
  // could support oversized allocations but command buffers are designed to
  // only contain small allocations and the device block pools should have a
  // large enough size (we verify on construction).
  const iree_device_size_t aligned_size =
      iree_device_align(target_ref.length, iree_hal_amdgpu_max_align_t);
  if (IREE_UNLIKELY(aligned_size >
                    IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_STORAGE_BLOCK_SIZE)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "aligned size of embedded command buffer data %" PRIdsz
        " exceeds minimum block pool storage capacity of %d per block",
        aligned_size, IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_STORAGE_BLOCK_SIZE);
  }

  // Retain buffers (if specified) for the lifetime of the command buffer.
  if (target_ref.buffer) {
    iree_hal_resource_t* resources[] = {
        (iree_hal_resource_t*)target_ref.buffer,
    };
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
        command_buffer->resource_set, IREE_ARRAYSIZE(resources), resources));
  }

  iree_hal_amdgpu_command_encoder_t* encoder =
      command_buffer->recording_state.encoder;

  // Append command to all devices.
  iree_hal_amdgpu_device_cmd_copy_buffer_t** device_cmds = NULL;
  uint32_t kernarg_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_command_encoder_append_cmd(
      encoder, IREE_HAL_AMDGPU_DEVICE_CMD_COPY_BUFFER,
      IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_NONE,
      IREE_HAL_AMDGPU_DEVICE_CMD_COPY_BUFFER_AQL_PACKET_COUNT,
      IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE,
      IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_ALIGNMENT,
      (void***)&device_cmds, &kernarg_offset));

  // Loop over each device this command buffer is being broadcast to apply
  // per-device information.
  for (iree_host_size_t i = 0; i < encoder->device_count; ++i) {
    iree_hal_amdgpu_device_cmd_copy_buffer_t* cmd = device_cmds[i];
    cmd->kernarg_offset = kernarg_offset;
    cmd->target_ref = resolved_target_ref;

    // Capture and embed the source data per-device.
    //
    // TODO(benvanik): hoist embedding to lead device if requested by
    // IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORDING_FLAG_DATA_ON_LEAD_PHYSICAL_DEVICE.
    // That would reduce total memory consumption in multi-device cases at the
    // cost of additional cross-device traffic when the command buffer is issued
    // on other devices. I don't think we want that, but may be worthwhile in
    // cases where most workloads are on the lead device, the lead device is in
    // an independent power island, or the user _really_ cares about memory
    // consumption of a lot of command buffers.
    cmd->source_ref = (iree_hal_amdgpu_device_buffer_ref_t){
        .type = IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_PTR,
        .offset = 0,
        .length = resolved_target_ref.length,
        .value.bits = 0,
    };
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_command_encoder_emplace_data(
            encoder, i, iree_make_const_byte_span(source_buffer, source_offset),
            &cmd->source_ref.value.ptr),
        "embedding update data of %" PRIu64 "B",
        (uint64_t)resolved_target_ref.length);
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_amdgpu_command_buffer_t* command_buffer =
      iree_hal_amdgpu_command_buffer_cast(base_command_buffer);

  // Translate the buffers to their device-side representations. Translation is
  // currently device-independent.
  iree_hal_amdgpu_device_buffer_ref_t resolved_source_ref = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_translate_device_buffer_ref(
                           source_ref, &resolved_source_ref),
                       "resolving source_ref");
  iree_hal_amdgpu_device_buffer_ref_t resolved_target_ref = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_translate_device_buffer_ref(
                           target_ref, &resolved_target_ref),
                       "resolving target_ref");

  // Retain buffers (if specified) for the lifetime of the command buffer.
  if (source_ref.buffer || target_ref.buffer) {
    iree_hal_resource_t* resources[] = {
        (iree_hal_resource_t*)source_ref.buffer,
        (iree_hal_resource_t*)target_ref.buffer,
    };
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
        command_buffer->resource_set, IREE_ARRAYSIZE(resources), resources));
  }

  iree_hal_amdgpu_command_encoder_t* encoder =
      command_buffer->recording_state.encoder;

  // Append command to all devices.
  iree_hal_amdgpu_device_cmd_copy_buffer_t** device_cmds = NULL;
  uint32_t kernarg_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_command_encoder_append_cmd(
      encoder, IREE_HAL_AMDGPU_DEVICE_CMD_COPY_BUFFER,
      IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_NONE,
      IREE_HAL_AMDGPU_DEVICE_CMD_COPY_BUFFER_AQL_PACKET_COUNT,
      IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE,
      IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_ALIGNMENT,
      (void***)&device_cmds, &kernarg_offset));

  // Loop over each device this command buffer is being broadcast to apply
  // per-device information.
  for (iree_host_size_t i = 0; i < encoder->device_count; ++i) {
    iree_hal_amdgpu_device_cmd_copy_buffer_t* cmd = device_cmds[i];
    cmd->kernarg_offset = kernarg_offset;
    cmd->source_ref = resolved_source_ref;
    cmd->target_ref = resolved_target_ref;
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  iree_hal_amdgpu_command_buffer_t* command_buffer =
      iree_hal_amdgpu_command_buffer_cast(base_command_buffer);

  // TODO(benvanik): perform the collective operation defined by op. See the
  // headers for more information. The channel is fixed for a particular
  // recording but note that either buffer may be a reference to a binding table
  // slot in which case it will be provided during submission to a queue.
  (void)command_buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "collectives not implemented");

  return status;
}

// Records a direct or indirect dispatch on each physical device required.
// If |is_indirect| is true the |workgroup_count| must contain a valid workgroup
// count buffer reference and otherwise the static workgroup count dimensions
// will be used.
static iree_status_t iree_hal_amdgpu_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_amdgpu_command_buffer_t* command_buffer =
      iree_hal_amdgpu_command_buffer_cast(base_command_buffer);

  if (iree_hal_dispatch_uses_custom_arguments(flags)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "direct/indirect arguments are not supported in AMDGPU yet");
  }

  // Configure dispatch flags controlling how the scheduler interprets the
  // command when issuing. We use one code path for static dispatches where
  // the workgroup count is available either directly in the recorded command or
  // via an indirect workgroup count buffer that is known static at the time the
  // command buffer is issued. Dynamic indirect workgroup counts that may be
  // populated by a prior dispatch/transfer in the same command buffer require
  // some additional work.
  iree_hal_amdgpu_device_dispatch_flags_t cmd_flags =
      IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_NONE;
  if (iree_any_bit_set(flags,
                       IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS)) {
    cmd_flags |= IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_STATIC;
  } else if (iree_any_bit_set(
                 flags,
                 IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_DYNAMIC)) {
    cmd_flags |= IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_DYNAMIC;
  }

  // Resolve the workgroup count buffer to the thin buffer ref used by the
  // device-side command buffer.
  iree_hal_amdgpu_device_workgroup_count_t workgroup_count;
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    iree_hal_amdgpu_device_buffer_ref_t resolved_workgroup_count_ref = {0};
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_translate_device_buffer_ref(
            config.workgroup_count_ref, &resolved_workgroup_count_ref),
        "resolving indirect workgroups_ref");
    if (IREE_UNLIKELY(resolved_workgroup_count_ref.length <
                      sizeof(uint32_t[3]))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "indirect workgroup count buffer reference must be at least "
          "uint32_t[3] (12B) but it resolved to %" PRIu64 "B",
          (uint64_t)resolved_workgroup_count_ref.length);
    }
    workgroup_count = (iree_hal_amdgpu_device_workgroup_count_t){
        .ref =
            {
                .type = resolved_workgroup_count_ref.type,
                .offset = resolved_workgroup_count_ref.offset,
                .value.bits = resolved_workgroup_count_ref.value.bits,
            },
    };
  } else {
    workgroup_count = (iree_hal_amdgpu_device_workgroup_count_t){
        .dims =
            {
                config.workgroup_count[0],
                config.workgroup_count[1],
                config.workgroup_count[2],
            },
    };
  }

  // Lookup the kernel metadata in host memory so we don't risk crossing a bus.
  // Validation is amortized across all target devices as all devices will have
  // the same export (just different kernel object pointers).
  const iree_hal_amdgpu_device_kernel_args_t* host_kernel_args = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_lookup_kernel_args_for_host(
      executable, entry_point, &host_kernel_args));
  if (IREE_UNLIKELY(constants.data_length !=
                    host_kernel_args->constant_count * sizeof(uint32_t))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch requires %" PRIhsz
                            "B of constants but %" PRIhsz "B was provided",
                            host_kernel_args->constant_count * sizeof(uint32_t),
                            constants.data_length);
  } else if (IREE_UNLIKELY(bindings.count != host_kernel_args->binding_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch requires %u bindings but %" PRIhsz
                            " were provided",
                            host_kernel_args->binding_count, bindings.count);
  }

  // Constant and binding capacity should have been checked when the executable
  // was loaded. These limits ensure we don't blow the host stack while
  // recording.
  IREE_ASSERT_LE(
      constants.data_length,
      IREE_HAL_AMDGPU_MAX_DISPATCH_CONSTANT_COUNT * sizeof(uint32_t));
  IREE_ASSERT_LE(bindings.count, IREE_HAL_AMDGPU_MAX_DISPATCH_BINDING_COUNT);

  // If the kernarg size declared by the kernel is larger than we expect it
  // (likely) means there are implicit kernargs expected following the explicit
  // kernargs we provide for bindings and constants. These are aligned to 8
  // bytes and follow any padding after the explicit args. Technically the
  // compiler will truncate the size to only those fields required but doing
  // that logic during command issue is more expensive than just eating the
  // extra few dozen bytes of kernarg space that's mostly filled with zeros so
  // we go to the max size we allow. This handling isn't great but does allow us
  // to run HIP kernels that follow our ABI even if they use device library code
  // that often pulls in these implicit args.
  //
  // See iree_amdgpu_kernel_implicit_args_t for more information on implicit
  // args. Since most are constant we don't actually produce them here and leave
  // that to the issue phase. Note that the implicit args contain the workgroup
  // count and when indirect we need to update the implicit args along with the
  // dispatch packet because they store redundant (to us/HIP) information. Yuck.
  const uint16_t explicit_kernarg_size =
      bindings.count * sizeof(uint64_t) +
      iree_host_align(constants.data_length, 8);
  const uint16_t implicit_kernarg_size =
      host_kernel_args->kernarg_size - explicit_kernarg_size;
  cmd_flags |= implicit_kernarg_size > 0
                   ? IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_IMPLICIT_ARGS
                   : 0;
  const uint16_t kernarg_size =
      implicit_kernarg_size > 0
          ? explicit_kernarg_size +
                iree_max(implicit_kernarg_size,
                         IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE)
          : host_kernel_args->kernarg_size;

  // Resolve all bindings ahead of the per-device work.
  // Today we do not have per-device resolution but could in the future if we
  // wanted to support replication of buffers across devices such that they had
  // unique resolved addresses.
  //
  // TODO(benvanik): add support for lead-physical-device mode:
  // IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORDING_FLAG_DATA_ON_LEAD_PHYSICAL_DEVICE.
  // We'd produce this directly into the lead device memory (+ constants)
  // outside of the loop and reference it instead of copying it to each device.
  // A bulk of the command buffer memory is in constants/bindings so if there
  // was ever a need for lead-device placement it'd be for this information. I'm
  // not sure there's a need, though, so it's kept simple/time-efficient today.
  iree_hal_amdgpu_device_buffer_ref_t* resolved_bindings =
      (iree_hal_amdgpu_device_buffer_ref_t*)iree_alloca(
          bindings.count * sizeof(iree_hal_amdgpu_device_buffer_ref_t));
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_translate_device_buffer_ref(
            bindings.values[i],
            (iree_hal_amdgpu_device_buffer_ref_t*)&resolved_bindings[i]),
        "resolving binding[%" PRIhsz "]", i);
  }

  // Insert indirect workgroup count buffers into the resource set.
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
        command_buffer->resource_set, 1, &config.workgroup_count_ref.buffer));
  }

  // Insert all bindings into the resource set.
  // We do this once regardless of how many devices we broadcast to.
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert_strided(
      command_buffer->resource_set, bindings.count, bindings.values,
      offsetof(iree_hal_buffer_ref_t, buffer), sizeof(iree_hal_buffer_ref_t)));

  // Insert executable into the resource set.
  // This will keep all device copies alive (including those for devices we
  // aren't recording for - that's ok).
  //
  // NOTE: this is 99.9% of the time the same executable and to avoid thrashing
  // the resource set LRU cache we special case checking for this exact case.
  // This is a micro-optimization that mostly shows up in either smaller
  // resource sets or faster cleanup time (fewer redundant references to the
  // same executable).
  if (executable != command_buffer->recording_state.last_executable) {
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
        command_buffer->resource_set, 1, &executable));
    command_buffer->recording_state.last_executable = executable;
  }

  iree_hal_amdgpu_command_encoder_t* encoder =
      command_buffer->recording_state.encoder;

  // Append command based on what type of dispatch it is.
  iree_hal_amdgpu_device_cmd_dispatch_t** device_cmds = NULL;
  uint32_t kernarg_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_command_encoder_append_cmd(
      encoder,
      iree_all_bits_set(cmd_flags,
                        IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_DYNAMIC)
          ? IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH_INDIRECT_DYNAMIC
          : IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH,
      IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_NONE,
      IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH_AQL_PACKET_COUNT(cmd_flags),
      kernarg_size, host_kernel_args->kernarg_alignment, (void***)&device_cmds,
      &kernarg_offset));

  // Loop over each device this command buffer is being broadcast to apply
  // per-device information.
  IREE_HAL_AMDGPU_FOR_PHYSICAL_DEVICE(command_buffer->device_affinity) {
    iree_hal_amdgpu_device_cmd_dispatch_t* cmd = device_cmds[device_index];
    cmd->kernarg_offset = kernarg_offset;

    // Lookup the per-device kernel information. Each device has its own
    // loaded kernel object pointer.
    const iree_hal_amdgpu_device_kernel_args_t* device_kernel_args = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_lookup_kernel_args_for_device(
            executable, entry_point, device_ordinal, &device_kernel_args));

    cmd->config.flags = cmd_flags;
    cmd->config.kernel_args = device_kernel_args;
    memcpy(&cmd->config.workgroup_count, &workgroup_count,
           sizeof(cmd->config.workgroup_count));

    // Copy resolved bindings and constants into the embedded data block.
    // Since this will be going across the PCI bus we do it as a memcpy in hopes
    // of getting decent transaction overhead.
    //
    // The dispatch command does not require that the bindings and constants be
    // adjacent in memory but we do so here to reduce overheads by only having
    // one block growth check and range of memory being touched instead of two.
    // In the future if we want to do things like share embedded data across
    // devices to reduce memory consumption we could still shard the commands
    // (as we have to for unique kernel ptrs) but could share these.
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_command_encoder_emplace_data_concat(
        encoder, device_index,
        iree_make_const_byte_span(
            resolved_bindings, bindings.count * sizeof(resolved_bindings[0])),
        constants, (void**)&cmd->bindings, (void**)&cmd->constants));
  }

  return iree_ok_status();
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_amdgpu_command_buffer_vtable = {
        .destroy = iree_hal_amdgpu_command_buffer_destroy,
        .begin = iree_hal_amdgpu_command_buffer_begin,
        .end = iree_hal_amdgpu_command_buffer_end,
        .begin_debug_group = iree_hal_amdgpu_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_amdgpu_command_buffer_end_debug_group,
        .execution_barrier = iree_hal_amdgpu_command_buffer_execution_barrier,
        .signal_event = iree_hal_amdgpu_command_buffer_signal_event,
        .reset_event = iree_hal_amdgpu_command_buffer_reset_event,
        .wait_events = iree_hal_amdgpu_command_buffer_wait_events,
        .advise_buffer = iree_hal_amdgpu_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_amdgpu_command_buffer_fill_buffer,
        .update_buffer = iree_hal_amdgpu_command_buffer_update_buffer,
        .copy_buffer = iree_hal_amdgpu_command_buffer_copy_buffer,
        .collective = iree_hal_amdgpu_command_buffer_collective,
        .dispatch = iree_hal_amdgpu_command_buffer_dispatch,
};
