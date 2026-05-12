// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_AQL_PROGRAM_VALIDATION_H_
#define IREE_HAL_DRIVERS_AMDGPU_AQL_PROGRAM_VALIDATION_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/aql_program_builder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Validates that |block| has a replayable branch or return terminator.
iree_status_t iree_hal_amdgpu_aql_program_validate_block_terminator(
    const iree_hal_amdgpu_command_buffer_block_header_t* block);

// Resolves the next block for currently supported linear branch replay.
iree_status_t iree_hal_amdgpu_aql_program_next_linear_block(
    const iree_hal_amdgpu_aql_program_t* program,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    uint32_t target_block_ordinal,
    const iree_hal_amdgpu_command_buffer_block_header_t** out_next_block);

// Validates that |program| can be replayed without invoking AQL block
// processors.
iree_status_t iree_hal_amdgpu_aql_program_validate_metadata_only(
    const iree_hal_amdgpu_aql_program_t* program);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_AQL_PROGRAM_VALIDATION_H_
