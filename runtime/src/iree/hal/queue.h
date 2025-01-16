// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_QUEUE_H_
#define IREE_HAL_QUEUE_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// A bitmap indicating logical device queue affinity.
// Used to direct submissions to specific device queues or locate memory nearby
// where it will be used. The meaning of the bits in the bitmap is
// implementation-specific: a bit may represent a logical queue in an underlying
// API such as a VkQueue or a physical queue such as a discrete virtual device.
//
// Bitwise operations can be performed on affinities; for example AND'ing two
// affinities will produce the intersection and OR'ing will produce the union.
// This enables just-in-time selection as a command buffer could be made
// available to some set of queues when recorded and then AND'ed with an actual
// set of queues to execute on during submission.
typedef uint64_t iree_hal_queue_affinity_t;

// Specifies that any queue may be selected.
#define IREE_HAL_QUEUE_AFFINITY_ANY ((iree_hal_queue_affinity_t)(-1))
#define IREE_HAL_MAX_QUEUES (sizeof(iree_hal_queue_affinity_t) / 8)

// Returns true if the |queue_affinity| is empty (none specified).
#define iree_hal_queue_affinity_is_empty(queue_affinity) ((queue_affinity) == 0)

// Returns true if the |queue_affinity| is indicating any/all queues.
#define iree_hal_queue_affinity_is_any(queue_affinity) \
  ((queue_affinity) == IREE_HAL_QUEUE_AFFINITY_ANY)

// Returns the total number of queues specified in the |queue_affinity| mask.
#define iree_hal_queue_affinity_count(queue_affinity) \
  iree_math_count_ones_u64(queue_affinity)

// Returns the index of the first set bit in |queue_affinity|.
// Requires that at least one bit be set.
#define iree_hal_queue_affinity_find_first_set(queue_affinity) \
  iree_math_count_trailing_zeros_u64(queue_affinity)

// Logically shifts the queue affinity to the right by the given amount.
#define iree_hal_queue_affinity_shr(queue_affinity, amount) \
  iree_shr((queue_affinity), (amount))

// Updates |inout_affinity| to only include those bits set in |mask_affinity|.
#define iree_hal_queue_affinity_and_into(inout_affinity, mask_affinity) \
  (inout_affinity) = ((inout_affinity) & (mask_affinity))

// Updates |inout_affinity| to include bits set in |mask_affinity|.
#define iree_hal_queue_affinity_or_into(inout_affinity, mask_affinity) \
  (inout_affinity) = ((inout_affinity) | (mask_affinity))

// Loops over each queue in the given |queue_affinity| bitmap.
//
// The following variables are available within the loop:
//     queue_count: total number of queues used
//     queue_index: loop index (0 to queue_count)
//   queue_ordinal: queue ordinal (0 to the total number of queues)
//
// Example:
//  IREE_HAL_FOR_QUEUE_AFFINITY(my_queue_affinity) {
//    compact_queue_list[queue_index];     // 0 to my_queue_affinity count
//    full_queue_list[queue_ordinal];      // 0 to available queues
//  }
#define IREE_HAL_FOR_QUEUE_AFFINITY(queue_affinity)                           \
  iree_hal_queue_affinity_t _queue_bits = (queue_affinity);                   \
  for (int queue_index = 0, _queue_ordinal_base = 0,                          \
           queue_count = iree_hal_queue_affinity_count(_queue_bits),          \
           _bit_offset = 0,                                                   \
           queue_ordinal =                                                    \
               iree_hal_queue_affinity_find_first_set(_queue_bits);           \
       queue_index < queue_count;                                             \
       ++queue_index, _queue_ordinal_base += _bit_offset + 1,                 \
           _queue_bits =                                                      \
               iree_hal_queue_affinity_shr(_queue_bits, _bit_offset + 1),     \
           _bit_offset = iree_hal_queue_affinity_find_first_set(_queue_bits), \
           queue_ordinal = _queue_ordinal_base + _bit_offset)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_QUEUE_H_
