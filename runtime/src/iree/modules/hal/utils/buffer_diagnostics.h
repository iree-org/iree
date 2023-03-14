// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_HAL_UTILS_BUFFER_DIAGNOSTICS_H_
#define IREE_MODULES_HAL_UTILS_BUFFER_DIAGNOSTICS_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/vm/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_buffer_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_modules_buffer_assert(
    iree_vm_ref_t buffer_ref, iree_vm_ref_t message_ref,
    iree_device_size_t minimum_length,
    iree_hal_memory_type_t required_memory_types,
    iree_hal_buffer_usage_t required_buffer_usage);

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_modules_buffer_view_assert(
    iree_vm_ref_t buffer_view_ref, iree_vm_ref_t message_ref,
    iree_hal_element_type_t expected_element_type,
    iree_hal_encoding_type_t expected_encoding_type,
    iree_host_size_t expected_shape_rank,
    const iree_hal_dim_t* expected_shape_dims);

iree_status_t iree_hal_modules_buffer_view_trace(
    iree_vm_ref_t key_ref, iree_vm_size_t buffer_view_count,
    iree_vm_abi_r_t* buffer_view_refs, iree_allocator_t host_allocator);

#endif  // IREE_MODULES_HAL_UTILS_BUFFER_DIAGNOSTICS_H_
