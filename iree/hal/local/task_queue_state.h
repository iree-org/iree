// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_HAL_LOCAL_TASK_QUEUE_STATE_H_
#define IREE_HAL_LOCAL_TASK_QUEUE_STATE_H_

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
typedef struct {
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

#endif  // IREE_HAL_LOCAL_TASK_QUEUE_STATE_H_
