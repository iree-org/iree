// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_UTIL_QUEUE_H_
#define IREE_HAL_DRIVERS_HIP_UTIL_QUEUE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A circular array where we can push to the back and pop from the front.
// The helper functions allow you to index into the array. Furthermore an
// initial allocation may be provided inline as an optimization.
typedef struct iree_hal_hip_util_queue_t {
  iree_allocator_t allocator;
  uint8_t* elements;
  iree_host_size_t element_size;
  iree_host_size_t element_count;
  iree_host_size_t capacity;
  iree_host_size_t head;
  uint8_t initial_allocation[];
} iree_hal_hip_util_queue_t;

// Initializes the queue with elements of the given |element_size|.
//
// Optionally |inline_count| can be provided to notify the queue
// that an initial allocation is present for the given number of elements.
void iree_hal_hip_util_queue_initialize(iree_allocator_t allocator,
                                        iree_host_size_t element_size,
                                        iree_host_size_t inline_count,
                                        iree_hal_hip_util_queue_t* out_queue);

// Deinitializes the list, it does not have to be empty.
void iree_hal_hip_util_queue_deinitialize(iree_hal_hip_util_queue_t* queue);

// Copies the given element into the back of the array. This may cause a
// re-allocation of data.
iree_status_t iree_hal_hip_util_queue_push_back(
    iree_hal_hip_util_queue_t* queue, void* element);

// Pops the element from the front of the array and moves the head.
void iree_hal_hip_util_queue_pop_front(iree_hal_hip_util_queue_t* queue,
                                       iree_host_size_t count);

// Returns a pointer to the element at index i
void* iree_hal_hip_util_queue_at(const iree_hal_hip_util_queue_t* queue,
                                 iree_host_size_t i);

#define IREE_HAL_HIP_UTIL_TYPED_QUEUE_WRAPPER(name, type,                      \
                                              default_element_count)           \
  typedef struct name##_t {                                                    \
    iree_allocator_t allocator;                                                \
    void* elements;                                                            \
    iree_host_size_t element_size;                                             \
    iree_host_size_t element_count;                                            \
    iree_host_size_t capacity;                                                 \
    iree_host_size_t head;                                                     \
    iree_alignas(iree_max_align_t) uint8_t                                     \
        initial_allocation[default_element_count * sizeof(type)];              \
  } name##_t;                                                                  \
  static inline void name##_initialize(iree_allocator_t allocator,             \
                                       name##_t* out_queue) {                  \
    iree_hal_hip_util_queue_initialize(allocator, sizeof(type),                \
                                       default_element_count,                  \
                                       (iree_hal_hip_util_queue_t*)out_queue); \
  }                                                                            \
  static inline void name##_deinitialize(name##_t* out_queue) {                \
    iree_hal_hip_util_queue_deinitialize(                                      \
        (iree_hal_hip_util_queue_t*)out_queue);                                \
  }                                                                            \
  iree_status_t name##_push_back(name##_t* queue, type element) {              \
    return iree_hal_hip_util_queue_push_back(                                  \
        (iree_hal_hip_util_queue_t*)queue, &element);                          \
  }                                                                            \
  void name##_pop_front(name##_t* queue, iree_host_size_t count) {             \
    iree_hal_hip_util_queue_pop_front((iree_hal_hip_util_queue_t*)queue,       \
                                      count);                                  \
  }                                                                            \
  type name##_at(name##_t* queue, iree_host_size_t i) {                        \
    type t;                                                                    \
    memcpy(&t,                                                                 \
           iree_hal_hip_util_queue_at((iree_hal_hip_util_queue_t*)queue, i),   \
           sizeof(type));                                                      \
    return t;                                                                  \
  }                                                                            \
  bool name##_empty(name##_t* queue) { return queue->element_count == 0; }     \
  iree_host_size_t name##_count(name##_t* queue) {                             \
    return queue->element_count;                                               \
  }

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  //  IREE_HAL_DRIVERS_HIP_UTIL_QUEUE_H_
