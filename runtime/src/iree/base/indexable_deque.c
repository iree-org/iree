// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/indexable_deque.h"

#include "iree/base/assert.h"

void iree_indexable_queue_initialize(iree_indexable_queue_t* queue,
                                     iree_allocator_t allocator,
                                     iree_host_size_t element_size,
                                     iree_host_size_t inline_count) {
  queue->allocator = allocator;
  queue->elements = &queue->initial_allocation[0];
  queue->element_size = element_size;
  queue->element_count = 0;
  queue->capacity = inline_count;
  queue->head = 0;
}

void iree_indexable_queue_deinitialize(iree_indexable_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  if (queue->elements != &queue->initial_allocation[0]) {
    iree_allocator_free(queue->allocator, queue->elements);
  }
}

iree_status_t iree_indexable_queue_push_back(iree_indexable_queue_t* queue,
                                             void* element) {
  // Expand
  if (queue->capacity == queue->element_count) {
    uint8_t* new_mem;
    queue->capacity = iree_max(16, queue->capacity << 1);
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        queue->allocator, queue->element_size * queue->capacity,
        (void**)&new_mem));
    memcpy(new_mem, queue->elements + (queue->head * queue->element_size),
           (queue->element_count - queue->head) * queue->element_size);
    memcpy(
        new_mem + ((queue->element_count - queue->head) * queue->element_size),
        queue->elements, queue->head * queue->element_size);
    if (queue->elements != &queue->initial_allocation[0]) {
      iree_allocator_free(queue->allocator, queue->elements);
    }
    queue->head = 0;
    queue->elements = new_mem;
  }
  memcpy(queue->elements +
             (((queue->head + queue->element_count) % queue->capacity) *
              queue->element_size),
         element, queue->element_size);
  queue->element_count++;
  return iree_ok_status();
}

void iree_indexable_queue_pop_front(iree_indexable_queue_t* queue,
                                    iree_host_size_t count) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT(count <= queue->element_count, "Popping too many elements");
  queue->head += count;
  queue->head = queue->head % queue->capacity;
  queue->element_count -= count;
}

void* iree_indexable_queue_at(iree_indexable_queue_t* queue,
                              iree_host_size_t i) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT(i < queue->element_count, "Index out of range");
  return queue->elements +
         ((queue->head + i) % queue->capacity) * queue->element_size;
}
