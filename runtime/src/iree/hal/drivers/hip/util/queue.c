// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/util/queue.h"

void iree_hal_hip_util_queue_initialize(iree_allocator_t allocator,
                                        iree_host_size_t element_size,
                                        iree_host_size_t inline_count,
                                        iree_hal_hip_util_queue_t* out_queue) {
  out_queue->allocator = allocator;
  out_queue->elements = &out_queue->initial_allocation[0];
  out_queue->element_size = element_size;
  out_queue->element_count = 0;
  out_queue->capacity = inline_count;
  out_queue->head = 0;
}

void iree_hal_hip_util_queue_deinitialize(iree_hal_hip_util_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  if (queue->elements != &queue->initial_allocation[0]) {
    iree_allocator_free(queue->allocator, queue->elements);
  }
}

iree_status_t iree_hal_hip_util_queue_push_back(
    iree_hal_hip_util_queue_t* queue, void* element) {
  // Expand the queue if necessary.
  if (queue->capacity == queue->element_count) {
    uint8_t* new_mem = NULL;
    queue->capacity = iree_max(16, queue->capacity * 2);
    if (queue->elements == &queue->initial_allocation[0]) {
      IREE_RETURN_IF_ERROR(iree_allocator_malloc(
          queue->allocator, queue->element_size * queue->capacity,
          (void**)&new_mem));
      memcpy(new_mem, queue->elements + (queue->head * queue->element_size),
             (queue->element_count - queue->head) * queue->element_size);
      memcpy(new_mem +
                 ((queue->element_count - queue->head) * queue->element_size),
             queue->elements, queue->head * queue->element_size);
      queue->head = 0;
    } else {
      new_mem = queue->elements;
      IREE_RETURN_IF_ERROR(iree_allocator_realloc(
          queue->allocator, queue->element_size * queue->capacity,
          (void**)&new_mem));
      const iree_host_size_t num_head_elements =
          queue->element_count - queue->head;
      const iree_host_size_t num_wrapped_elements =
          queue->element_count - num_head_elements;

      // If we have wrapped elements, then we move them to the end after the
      // head, since we have at least doubled the size of out array, there is
      // enough room.
      if (num_wrapped_elements) {
        memcpy(
            new_mem + (queue->head + num_head_elements) * queue->element_size,
            new_mem, num_wrapped_elements * queue->element_size);
      }
    }
    queue->elements = new_mem;
  }
  memcpy(queue->elements +
             (((queue->head + queue->element_count) % queue->capacity) *
              queue->element_size),
         element, queue->element_size);
  ++queue->element_count;
  return iree_ok_status();
}

void iree_hal_hip_util_queue_pop_front(iree_hal_hip_util_queue_t* queue,
                                       iree_host_size_t count) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_LE(count, queue->element_count, "Popping too many elements");
  queue->head += count;
  queue->head = queue->head % queue->capacity;
  queue->element_count -= count;
}

void* iree_hal_hip_util_queue_at(const iree_hal_hip_util_queue_t* queue,
                                 iree_host_size_t i) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_LT(i, queue->element_count, "Index out of range");
  return queue->elements +
         ((queue->head + i) % queue->capacity) * queue->element_size;
}
