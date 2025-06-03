// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_BITMAP_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_BITMAP_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_bitmap_t
//===----------------------------------------------------------------------===//

// Bits per word of bitmap data.
#define IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD (sizeof(uint64_t) * 8)

// Reference to a bitmap stored in 64-bit words.
// This is a fat pointer to the storage and does not store anything itself.
// Intended for small counts (~dozens to hundreds).
//
// Bits outside of the bit count range may be modified by operations and their
// contents should be treated as undefined.
//
// We could reduce this or specialize to a single pointer by packing the bit
// count in the upper byte of the pointer given that most usage is <= 64 bits.
// Today this type generally only lives on the stack or in registers so we don't
// bother as shifting around would be more expensive.
typedef struct iree_hal_amdgpu_bitmap_t {
  iree_host_size_t bit_count;
  uint64_t* words;
} iree_hal_amdgpu_bitmap_t;

// Calculates the total number of words required to store |bit_count| bits.
#define iree_hal_amdgpu_bitmap_calculate_words(bit_count) \
  iree_host_size_ceil_div(bit_count, IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD)

// Returns true if no bits are set in the bitmap.
bool iree_hal_amdgpu_bitmap_empty(iree_hal_amdgpu_bitmap_t bitmap);

// TODO(benvanik): iree_hal_amdgpu_bitmap_count (popcnt loop with masking).

// Returns true if the bit at |bit_index| is 1.
// Expects |bit_index| to be in bounds.
bool iree_hal_amdgpu_bitmap_test(iree_hal_amdgpu_bitmap_t bitmap,
                                 iree_host_size_t bit_index);

// Sets the bit at |bit_index| to 1.
// Expects |bit_index| to be in bounds.
void iree_hal_amdgpu_bitmap_set(iree_hal_amdgpu_bitmap_t bitmap,
                                iree_host_size_t bit_index);

// Sets all bits between |bit_index| and |bit_index| + |bitmap_length| to 1.
// Expects the entire range to be in bounds.
void iree_hal_amdgpu_bitmap_set_span(iree_hal_amdgpu_bitmap_t bitmap,
                                     iree_host_size_t bit_index,
                                     iree_host_size_t bit_length);

// Sets all bits in the bitmap to 1.
void iree_hal_amdgpu_bitmap_set_all(iree_hal_amdgpu_bitmap_t bitmap);

// Resets the bit at |bit_index| to 0.
// Expects |bit_index| to be in bounds.
void iree_hal_amdgpu_bitmap_reset(iree_hal_amdgpu_bitmap_t bitmap,
                                  iree_host_size_t bit_index);

// Resets all bits between |bit_index| and |bit_index| + |bitmap_length| to 0.
void iree_hal_amdgpu_bitmap_reset_span(iree_hal_amdgpu_bitmap_t bitmap,
                                       iree_host_size_t bit_index,
                                       iree_host_size_t bit_length);

// Resets all bits in the bitmap to 0.
void iree_hal_amdgpu_bitmap_reset_all(iree_hal_amdgpu_bitmap_t bitmap);

// Finds the first set bit (value 1) starting from |bit_offset|.
// Returns the bitmap size if no set bit is found.
// Expects |bit_offset| to be in bounds or 0.
iree_host_size_t iree_hal_amdgpu_bitmap_find_first_set(
    iree_hal_amdgpu_bitmap_t bitmap, iree_host_size_t bit_offset);

// Finds the first unset bit (value 0) starting from |bit_offset|.
// Returns the bitmap size if no unset bit is found.
// Expects |bit_offset| to be in bounds or 0.
iree_host_size_t iree_hal_amdgpu_bitmap_find_first_unset(
    iree_hal_amdgpu_bitmap_t bitmap, iree_host_size_t bit_offset);

// Finds the first contiguous |bit_length| span of unset bits (value 0) starting
// from |bit_offset|.
// Returns the bitmap size if no span of unset bits is found.
// Expects the entire range to be in bounds or |bit_offset|/|bit_length| as 0.
iree_host_size_t iree_hal_amdgpu_bitmap_find_first_unset_span(
    iree_hal_amdgpu_bitmap_t bitmap, iree_host_size_t bit_offset,
    iree_host_size_t bit_length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_BITMAP_H_
