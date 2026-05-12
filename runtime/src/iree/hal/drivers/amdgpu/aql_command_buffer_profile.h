// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_AQL_COMMAND_BUFFER_PROFILE_H_
#define IREE_HAL_DRIVERS_AMDGPU_AQL_COMMAND_BUFFER_PROFILE_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/aql_program_builder.h"
#include "iree/hal/drivers/amdgpu/profile_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Synthesizes command-operation profile metadata for every command in a
// finalized AQL command-buffer program and registers it with
// |profile_metadata|.
//
// This is a cold command-buffer finalization path. It performs one temporary
// host allocation sized to |program->command_count| and does not run during
// queue submission or replay.
iree_status_t iree_hal_amdgpu_aql_command_buffer_register_profile_operations(
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    uint64_t command_buffer_id, const iree_hal_amdgpu_aql_program_t* program,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_AQL_COMMAND_BUFFER_PROFILE_H_
