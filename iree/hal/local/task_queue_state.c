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

#include "iree/hal/local/task_queue_state.h"

#include "iree/base/tracing.h"

void iree_hal_task_queue_state_initialize(
    iree_hal_task_queue_state_t* out_queue_state) {
  memset(out_queue_state, 0, sizeof(*out_queue_state));
}

void iree_hal_task_queue_state_deinitialize(
    iree_hal_task_queue_state_t* queue_state) {}
