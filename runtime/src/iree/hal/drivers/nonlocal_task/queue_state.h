// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_LOCAL_TASK_TASK_QUEUE_STATE_H_
#define IREE_HAL_DRIVERS_LOCAL_TASK_TASK_QUEUE_STATE_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/api.h"
#include "iree/task/scope.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// State tracking for an individual queue.
//
// Thread-compatible: only intended to be used by a queue with the submission
// lock held.
typedef struct iree_hal_task_queue_state_t {
  // TODO(#4518): track event state.
  int reserved;
} iree_hal_task_queue_state_t;

// Initializes queue state with the given |identifier| used to annotate tasks
// submitted to the queue.
void iree_hal_task_queue_state_initialize(
    iree_hal_task_queue_state_t* out_queue_state);

// Deinitializes queue state and cleans up any tracking intermediates.
void iree_hal_task_queue_state_deinitialize(
    iree_hal_task_queue_state_t* queue_state);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_TASK_TASK_QUEUE_STATE_H_
