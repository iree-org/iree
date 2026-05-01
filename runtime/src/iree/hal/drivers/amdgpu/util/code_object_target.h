// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_CODE_OBJECT_TARGET_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_CODE_OBJECT_TARGET_H_

#include "iree/hal/drivers/amdgpu/util/target_id.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// AMDGPU Code Object Targets
//===----------------------------------------------------------------------===//

// Recovers the AMDGPU target ID encoded in an HSA code-object ELF header.
//
// The returned processor string is borrowed from static target tables and
// remains valid for the lifetime of the process.
iree_status_t iree_hal_amdgpu_code_object_target_id_from_elf(
    iree_const_byte_span_t elf_data,
    iree_hal_amdgpu_target_id_t* out_target_id);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_CODE_OBJECT_TARGET_H_
