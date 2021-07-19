// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_NOP_EVENT_H_
#define IREE_HAL_DRIVERS_WEBGPU_NOP_EVENT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_status_t iree_hal_webgpu_nop_event_create(iree_allocator_t host_allocator,
                                               iree_hal_event_t** out_event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_NOP_EVENT_H_
