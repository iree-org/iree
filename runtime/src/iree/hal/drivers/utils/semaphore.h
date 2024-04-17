// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_UTILS_SEMAPHORE_INTERNAL_H_
#define IREE_HAL_DRIVERS_UTILS_SEMAPHORE_INTERNAL_H_

#include "iree/base/api.h"
#include "iree/hal/semaphore.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

static inline void iree_hal_semaphore_list_swap_elements(
    iree_hal_semaphore_list_t* semaphore_list, iree_host_size_t i,
    iree_host_size_t j) {
  IREE_ASSERT(i >= 0 && i < semaphore_list->count);
  IREE_ASSERT(j >= 0 && j < semaphore_list->count);

  if (IREE_UNLIKELY(i == j)) {
    return;
  }

  iree_hal_semaphore_t* tmp_semaphore = semaphore_list->semaphores[i];
  uint64_t tmp_payload_value = semaphore_list->payload_values[i];

  semaphore_list->semaphores[i] = semaphore_list->semaphores[j];
  semaphore_list->payload_values[i] = semaphore_list->payload_values[j];

  semaphore_list->semaphores[j] = tmp_semaphore;
  semaphore_list->payload_values[j] = tmp_payload_value;
}

// Swap i-th element with the last and then remove the last.
static inline void iree_hal_semaphore_list_remove_element(
    iree_hal_semaphore_list_t* semaphore_list, iree_host_size_t i) {
  iree_hal_semaphore_list_swap_elements(semaphore_list, i,
                                        semaphore_list->count - 1);
  --semaphore_list->count;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_UTILS_SEMAPHORE_INTERNAL_H_
