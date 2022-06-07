// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/task_queue_state.h"

#include <string.h>

void iree_hal_task_queue_state_initialize(
    iree_hal_task_queue_state_t* out_queue_state) {
  memset(out_queue_state, 0, sizeof(*out_queue_state));
}

void iree_hal_task_queue_state_deinitialize(
    iree_hal_task_queue_state_t* queue_state) {}
