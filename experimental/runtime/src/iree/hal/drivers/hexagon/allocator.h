// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HEXAGON_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_HEXAGON_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_allocator_t
//===----------------------------------------------------------------------===//

// Creates a Hexagon buffer allocator.
//
// Buffer allocation is not implemented yet: the allocator has no memory heaps
// and all allocation, import, and export requests fail with
// IREE_STATUS_UNIMPLEMENTED.
iree_status_t iree_hal_hexagon_allocator_create(
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

#endif  // IREE_HAL_DRIVERS_HEXAGON_ALLOCATOR_H_
