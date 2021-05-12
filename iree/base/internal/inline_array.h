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

#ifndef IREE_BASE_INTERNAL_INLINE_ARRAY_H_
#define IREE_BASE_INTERNAL_INLINE_ARRAY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// iree_inline_array_t
//==============================================================================

// Maximum number of bytes that can be allocated from the stack.
// Arrays exceeding this size will incur a heap allocation.
#define IREE_INLINE_ARRAY_MAX_STACK_ALLOCATION 512

#define iree_inline_array(type, variable, initial_size, allocator)       \
  const iree_allocator_t variable##_allocator = (allocator);             \
  struct {                                                               \
    iree_host_size_t size;                                               \
    type* data;                                                          \
  } variable = {                                                         \
      (initial_size),                                                    \
      NULL,                                                              \
  };                                                                     \
  if (IREE_UNLIKELY(sizeof(type) * (initial_size) >                      \
                    IREE_INLINE_ARRAY_MAX_STACK_ALLOCATION)) {           \
    IREE_CHECK_OK(iree_allocator_malloc(variable##_allocator,            \
                                        sizeof(type) * (initial_size),   \
                                        (void**)&(variable).data));      \
  } else {                                                               \
    (variable).data = (type*)iree_alloca(sizeof(type) * (initial_size)); \
  }

#define iree_inline_array_deinitialize(variable)                 \
  if (IREE_UNLIKELY(sizeof(*(variable).data) * (variable).size > \
                    IREE_INLINE_ARRAY_MAX_STACK_ALLOCATION)) {   \
    iree_allocator_free(variable##_allocator, (variable).data);  \
  }

#define iree_inline_array_size(variable) (variable).size

#define iree_inline_array_capacity(variable) (variable).capacity
#define iree_inline_array_data(variable) (variable).data

#define iree_inline_array_at(variable, index) &(variable).data[(index)]

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_INLINE_ARRAY_H_
