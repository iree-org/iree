// Copyright 2020 Google LLC
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

#ifndef IREE_BASE_INTERNAL_FPU_STATE_H_
#define IREE_BASE_INTERNAL_FPU_STATE_H_

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/target_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// iree_fpu_state_*
//==============================================================================

// Flags controlling FPU features.
enum iree_fpu_state_flags_e {
  // Platform default.
  IREE_FPU_STATE_DEFAULT = 0,

  // Denormals can cause some serious slowdowns in certain ISAs where they may
  // be implemented in microcode. Flushing them to zero instead of letting them
  // propagate ensures that the slow paths aren't hit. This is a fast-math style
  // optimization (and is often part of all compiler's fast-math set of flags).
  //
  // https://en.wikipedia.org/wiki/Denormal_number
  // https://carlh.net/plugins/denormals.php
  // https://www.xspdf.com/resolution/50507310.html
  IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO = 1 << 0,
};
typedef uint32_t iree_fpu_state_flags_t;

// Opaque FPU state vector manipulated with iree_fpu_* functions.
typedef struct {
  uint64_t previous_value;
  uint64_t current_value;
} iree_fpu_state_t;

// Pushes a new floating-point unit (FPU) state for the current thread.
// May lead to a pipeline flush; avoid if possible.
iree_fpu_state_t iree_fpu_state_push(iree_fpu_state_flags_t flags);

// Restores the FPU state of the thread to its original value.
// May lead to a pipeline flush; avoid if possible.
void iree_fpu_state_pop(iree_fpu_state_t state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_FPU_STATE_H_
