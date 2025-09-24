// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_CHANNEL_H_
#define IREE_HAL_DRIVERS_AMDGPU_CHANNEL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_channel_t
//===----------------------------------------------------------------------===//

// Creates an AMDGPU HAL collective channel using the given |params|.
iree_status_t iree_hal_amdgpu_channel_create(iree_hal_channel_params_t params,
                                             iree_allocator_t host_allocator,
                                             iree_hal_channel_t** out_channel);

#endif  // IREE_HAL_DRIVERS_AMDGPU_CHANNEL_H_
