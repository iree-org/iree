// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_NULL_EVENT_H_
#define IREE_HAL_DRIVERS_NULL_EVENT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_null_event_t
//===----------------------------------------------------------------------===//

// WIP API and may change. Mostly ignored for now.
iree_status_t iree_hal_null_event_create(
    iree_hal_queue_affinity_t queue_affinity, iree_hal_event_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_event_t** out_event);

#endif  // IREE_HAL_DRIVERS_NULL_EVENT_H_
