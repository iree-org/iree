// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared command building functions for the block ISA.
//
// These functions bridge HAL API parameters to block ISA builder calls,
// handling validation, ISA encoding, and fixup entry setup. Both
// block_command_buffer.c (for command buffer recording) and task_queue.c
// (for direct queue operations) call these same functions.
//
// Each build function:
//   - Captures binding_data_base from the builder's current state
//   - Appends the ISA command via the builder (may trigger block splits)
//   - Fills all command-specific fields
//   - Pre-fills fixup data_index values (binding_data_base + i)
//   - Returns the fixup array for the caller to resolve bindings
//   - Returns a rollback token for error recovery
//
// The caller is responsible for:
//   - Resolving each fixup entry: set host_ptr + flags for direct bindings,
//     or host_ptr=NULL + slot + offset for indirect bindings
//   - On failure after build: calling iree_hal_cmd_build_rollback()

#ifndef IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_COMMAND_OPS_H_
#define IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_COMMAND_OPS_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/block_builder.h"
#include "iree/hal/drivers/local_task/block_isa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Rollback support
//===----------------------------------------------------------------------===//

// Token returned by build functions for rollback on post-build failures.
// Captures the parameters needed by iree_hal_cmd_block_builder_pop_cmd().
typedef struct iree_hal_cmd_build_token_t {
  iree_host_size_t cmd_bytes;
  uint16_t fixup_count;
  uint16_t binding_count;
  uint32_t tile_count;
} iree_hal_cmd_build_token_t;

// Rolls back the most recently built command using the token from the build
// function. Use this when a post-build operation (e.g., binding resolution)
// fails and the command must be removed from the builder.
//
// Only valid for the command buffer path where the builder persists across
// commands. For the queue path (stack-local builder), just deinitialize the
// builder on error — rollback is unnecessary.
void iree_hal_cmd_build_rollback(iree_hal_cmd_block_builder_t* builder,
                                 iree_hal_cmd_build_token_t token);

//===----------------------------------------------------------------------===//
// FILL
//===----------------------------------------------------------------------===//

// Builds a FILL command into the builder.
//
// Appends a fill command that writes |pattern| (1/2/4 bytes) repeated across
// |length| bytes of the target buffer. Large fills decompose into transfer
// tiles so multiple workers can participate.
//
// Returns 1 fixup entry via |out_fixups|:
//   fixups[0]: target buffer (data_index pre-filled, caller resolves binding)
iree_status_t iree_hal_cmd_build_fill(iree_hal_cmd_block_builder_t* builder,
                                      iree_device_size_t length,
                                      const void* pattern,
                                      iree_host_size_t pattern_length,
                                      iree_hal_cmd_fixup_t** out_fixups,
                                      iree_hal_cmd_build_token_t* out_token);

//===----------------------------------------------------------------------===//
// COPY
//===----------------------------------------------------------------------===//

// Builds a COPY command into the builder.
//
// Appends a copy command that copies |length| bytes from the source buffer to
// the target buffer. Large copies decompose into transfer tiles so multiple
// workers can participate.
//
// Returns 2 fixup entries via |out_fixups|:
//   fixups[0]: source buffer (data_index pre-filled, caller resolves binding)
//   fixups[1]: target buffer (data_index pre-filled, caller resolves binding)
iree_status_t iree_hal_cmd_build_copy(iree_hal_cmd_block_builder_t* builder,
                                      iree_device_size_t length,
                                      iree_hal_cmd_fixup_t** out_fixups,
                                      iree_hal_cmd_build_token_t* out_token);

//===----------------------------------------------------------------------===//
// UPDATE
//===----------------------------------------------------------------------===//

// Builds an UPDATE command into the builder.
//
// Copies |length| bytes from |source_buffer| + |source_offset| inline into
// the ISA command stream (.text). At execution time, the processor memcpy's
// the inline data to the target buffer.
//
// Returns 1 fixup entry via |out_fixups|:
//   fixups[0]: target buffer (data_index pre-filled, caller resolves binding)
iree_status_t iree_hal_cmd_build_update(iree_hal_cmd_block_builder_t* builder,
                                        const void* source_buffer,
                                        iree_host_size_t source_offset,
                                        iree_device_size_t length,
                                        iree_hal_cmd_fixup_t** out_fixups,
                                        iree_hal_cmd_build_token_t* out_token);

//===----------------------------------------------------------------------===//
// DISPATCH
//===----------------------------------------------------------------------===//

// Builds a DISPATCH command into the builder.
//
// Validates the executable's dispatch attributes against the provided
// constants and binding count. Resolves the function pointer and environment
// at build time (baked into .text for zero-indirection execution).
//
// Returns fixup entries via |out_fixups|:
//   fixups[0..binding_count-1]: binding buffers (data_index pre-filled,
//                                caller resolves binding for each)
//   fixups[binding_count]: indirect workgroup-count buffer when
//                          _INDIRECT_PARAMETERS is used
iree_status_t iree_hal_cmd_build_dispatch(
    iree_hal_cmd_block_builder_t* builder, iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_host_size_t binding_count, iree_hal_dispatch_flags_t flags,
    iree_hal_cmd_fixup_t** out_fixups, iree_hal_cmd_build_token_t* out_token);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_COMMAND_OPS_H_
