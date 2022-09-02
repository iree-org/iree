// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_EXECUTABLE_ENVIRONMENT_H_
#define IREE_HAL_LOCAL_EXECUTABLE_ENVIRONMENT_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/cpu.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_processor_*_t
//===----------------------------------------------------------------------===//

// Queries the current processor information and writes it to |out_processor|.
// |temp_allocator| may be used for temporary allocations required while
// querying. If the processor cannot be queried then |out_processor| will be
// zeroed.
void iree_hal_processor_query(iree_allocator_t temp_allocator,
                              iree_hal_processor_v0_t* out_processor);

// Looks up a field of the processor information by canonicalized string key.
iree_status_t iree_hal_processor_lookup_by_key(
    const iree_hal_processor_v0_t* processor, iree_string_view_t key,
    int64_t* IREE_RESTRICT out_value);

//===----------------------------------------------------------------------===//
// iree_hal_executable_environment_*_t
//===----------------------------------------------------------------------===//

// Initializes |out_environment| to the default empty environment.
// No imports will be available unless overridden during loading.
// |temp_allocator| may be used for temporary allocations during initialization.
void iree_hal_executable_environment_initialize(
    iree_allocator_t temp_allocator,
    iree_hal_executable_environment_v0_t* out_environment);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_EXECUTABLE_ENVIRONMENT_H_
