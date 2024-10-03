// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/util/affinity.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_allocator_t
//===----------------------------------------------------------------------===//

// Creates a AMDGPU buffer allocator used for persistent allocations.
iree_status_t iree_hal_amdgpu_allocator_create(
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

// DO NOT SUBMIT
// allocate ringbuffer svm
// iree_hal_amdgpu_allocator_allocate_ringbuffer(capacity)
// power of two only

#endif  // IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_H_
