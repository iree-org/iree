// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/bitmap.h"

#include "iree/base/internal/math.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_bitmap_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): move to iree/base/internal/math.h? Also used in device-side
// library (which we can't use runtime headers in, so we need copies somewhere).
//
// https://en.wikipedia.org/wiki/Find_first_set
#define IREE_HAL_AMDGPU_FFS_U64(v) \
  ((v) == 0 ? -1 : iree_math_count_trailing_zeros_u64(v))

// Returns a word with the bit at |bit_offset| set.
//
// Examples:
//   _BIT_OFFSET_TO_WORD_MASK(0)   = 0b00..001
//   _BIT_OFFSET_TO_WORD_MASK(1)   = 0b00..010
//   _BIT_OFFSET_TO_WORD_MASK(2)   = 0b00..100
#define _BIT_OFFSET_TO_WORD_MASK(bit_offset) \
  (1ull << ((bit_offset) % IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD))

// Returns a word index in the bitmap containing the bit at |bit_offset|.
//
// Examples:
//   _BIT_OFFSET_TO_WORD_INDEX(0)   = 0
//   _BIT_OFFSET_TO_WORD_INDEX(64)  = 1
//   _BIT_OFFSET_TO_WORD_INDEX(127) = 1
//   _BIT_OFFSET_TO_WORD_INDEX(128) = 2
#define _BIT_OFFSET_TO_WORD_INDEX(bit_offset) \
  ((bit_offset) / IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD)

// Returns a word mask that includes all valid bits after |bit_offset|.
//
// Examples:
//   _BIT_PREFIX_WORD_MASK(0) = 0b11..111
//   _BIT_PREFIX_WORD_MASK(1) = 0b11..110
//   _BIT_PREFIX_WORD_MASK(2) = 0b11..100
//   _BIT_PREFIX_WORD_MASK(3) = 0b11..000
#define _BIT_PREFIX_WORD_MASK(bit_offset) \
  (~0ull << ((bit_offset) & (IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD - 1)))

// Returns a word mask that includes all valid bits up to |bit_offset|.
//
// Examples:
//   _BIT_SUFFIX_WORD_MASK(0) = 0b00..000
//   _BIT_SUFFIX_WORD_MASK(1) = 0b00..001
//   _BIT_SUFFIX_WORD_MASK(2) = 0b00..011
//   _BIT_SUFFIX_WORD_MASK(3) = 0b00..111
#define _BIT_SUFFIX_WORD_MASK(bit_offset) \
  (~0ull >> (-(bit_offset) & (IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD - 1)))

// Scan full words first and handle any remaining bits after.
bool iree_hal_amdgpu_bitmap_empty(iree_hal_amdgpu_bitmap_t bitmap) {
  iree_host_size_t i = 0;
  for (i = 0; i < bitmap.bit_count / IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD;
       ++i) {
    if (bitmap.words[i]) return false;
  }
  if (bitmap.bit_count % IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD) {
    if (bitmap.words[i] & _BIT_SUFFIX_WORD_MASK(bitmap.bit_count)) {
      return false;
    }
  }
  return true;
}

bool iree_hal_amdgpu_bitmap_test(iree_hal_amdgpu_bitmap_t bitmap,
                                 iree_host_size_t bit_index) {
  return 1ull & (bitmap.words[_BIT_OFFSET_TO_WORD_INDEX(bit_index)] >>
                 (bit_index & (IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD - 1)));
}

void iree_hal_amdgpu_bitmap_set(iree_hal_amdgpu_bitmap_t bitmap,
                                iree_host_size_t bit_index) {
  const uint64_t word_mask = _BIT_OFFSET_TO_WORD_MASK(bit_index);
  uint64_t* word_ptr = bitmap.words + _BIT_OFFSET_TO_WORD_INDEX(bit_index);
  *word_ptr |= word_mask;
}

void iree_hal_amdgpu_bitmap_set_span(iree_hal_amdgpu_bitmap_t bitmap,
                                     iree_host_size_t bit_index,
                                     iree_host_size_t bit_length) {
  const iree_host_size_t bit_end = bit_index + bit_length;

  // Set from the start of the span to the last full word.
  int64_t bit_chunk = IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD -
                      (bit_index % IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD);
  uint64_t* word_ptr = bitmap.words + _BIT_OFFSET_TO_WORD_INDEX(bit_index);
  uint64_t word_mask = _BIT_PREFIX_WORD_MASK(bit_index);
  while ((int64_t)bit_length - bit_chunk >= 0) {
    *word_ptr |= word_mask;
    word_mask = ~0ull;
    bit_length -= bit_chunk;
    bit_chunk = IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD;
    ++word_ptr;
  }

  // Set the suffix bits in the last word (if any).
  if (bit_length > 0) {
    word_mask &= _BIT_SUFFIX_WORD_MASK(bit_end);
    *word_ptr |= word_mask;
  }
}

void iree_hal_amdgpu_bitmap_set_all(iree_hal_amdgpu_bitmap_t bitmap) {
  const iree_host_size_t word_count =
      iree_hal_amdgpu_bitmap_calculate_words(bitmap.bit_count);
  memset(bitmap.words, 0xFF, word_count * sizeof(uint64_t));
}

void iree_hal_amdgpu_bitmap_reset(iree_hal_amdgpu_bitmap_t bitmap,
                                  iree_host_size_t bit_index) {
  const uint64_t word_mask = _BIT_OFFSET_TO_WORD_MASK(bit_index);
  uint64_t* word_ptr = bitmap.words + _BIT_OFFSET_TO_WORD_INDEX(bit_index);
  *word_ptr &= ~word_mask;
}

void iree_hal_amdgpu_bitmap_reset_span(iree_hal_amdgpu_bitmap_t bitmap,
                                       iree_host_size_t bit_index,
                                       iree_host_size_t bit_length) {
  const iree_host_size_t bit_end = bit_index + bit_length;

  // Reset from the start of the span to the last full word.
  int64_t bit_chunk = IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD -
                      (bit_index % IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD);
  uint64_t* word_ptr = bitmap.words + _BIT_OFFSET_TO_WORD_INDEX(bit_index);
  uint64_t word_mask = _BIT_PREFIX_WORD_MASK(bit_index);
  while ((int64_t)bit_length - bit_chunk >= 0) {
    *word_ptr &= ~word_mask;
    word_mask = ~0ull;
    bit_length -= bit_chunk;
    bit_chunk = IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD;
    ++word_ptr;
  }

  // Reset the suffix bits in the last word (if any).
  if (bit_length > 0) {
    word_mask &= _BIT_SUFFIX_WORD_MASK(bit_end);
    *word_ptr &= ~word_mask;
  }
}

void iree_hal_amdgpu_bitmap_reset_all(iree_hal_amdgpu_bitmap_t bitmap) {
  const iree_host_size_t word_count =
      iree_hal_amdgpu_bitmap_calculate_words(bitmap.bit_count);
  memset(bitmap.words, 0x00, word_count * sizeof(uint64_t));
}

static iree_host_size_t iree_hal_amdgpu_bitmap_find_next_set_bit(
    const uint64_t* words, iree_host_size_t bit_count,
    iree_host_size_t bit_offset) {
  if (IREE_UNLIKELY(bit_offset >= bit_count)) return bit_count;
  const uint64_t word_mask = _BIT_PREFIX_WORD_MASK(bit_offset);
  iree_host_size_t word_index = _BIT_OFFSET_TO_WORD_INDEX(bit_offset);
  uint64_t word = 0;
  for (word = words[word_index] & word_mask; !word; word = words[word_index]) {
    if ((word_index + 1) * IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD >= bit_count) {
      return bit_count;  // hit end without finding anything
    }
    ++word_index;
  }
  return iree_min(word_index * IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD +
                      IREE_HAL_AMDGPU_FFS_U64(word),
                  bit_count);
}

iree_host_size_t iree_hal_amdgpu_bitmap_find_first_set(
    iree_hal_amdgpu_bitmap_t bitmap, iree_host_size_t bit_offset) {
  return iree_hal_amdgpu_bitmap_find_next_set_bit(bitmap.words,
                                                  bitmap.bit_count, bit_offset);
}

static iree_host_size_t iree_hal_amdgpu_bitmap_find_next_unset_bit(
    const uint64_t* words, iree_host_size_t bit_count,
    iree_host_size_t bit_offset) {
  if (IREE_UNLIKELY(bit_offset >= bit_count)) return bit_count;
  const uint64_t word_mask = _BIT_PREFIX_WORD_MASK(bit_offset);
  iree_host_size_t word_index = _BIT_OFFSET_TO_WORD_INDEX(bit_offset);
  uint64_t word = 0;
  for (word = ~words[word_index] & word_mask; !word;
       word = ~words[word_index]) {
    if ((word_index + 1) * IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD >= bit_count) {
      return bit_count;  // hit end without finding anything
    }
    ++word_index;
  }
  return iree_min(word_index * IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD +
                      IREE_HAL_AMDGPU_FFS_U64(word),
                  bit_count);
}

iree_host_size_t iree_hal_amdgpu_bitmap_find_first_unset(
    iree_hal_amdgpu_bitmap_t bitmap, iree_host_size_t bit_offset) {
  return iree_hal_amdgpu_bitmap_find_next_unset_bit(
      bitmap.words, bitmap.bit_count, bit_offset);
}

iree_host_size_t iree_hal_amdgpu_bitmap_find_first_unset_span(
    iree_hal_amdgpu_bitmap_t bitmap, iree_host_size_t bit_offset,
    iree_host_size_t bit_length) {
  iree_host_size_t bit_index = 0;
  do {
    bit_index = iree_hal_amdgpu_bitmap_find_next_unset_bit(
        bitmap.words, bitmap.bit_count, bit_offset);
    const iree_host_size_t bit_end = bit_index + bit_length;
    if (bit_end > bitmap.bit_count) return bitmap.bit_count;
    const iree_host_size_t next_index =
        iree_hal_amdgpu_bitmap_find_next_set_bit(bitmap.words, bit_end,
                                                 bit_index);
    if (next_index < bit_end) {
      bit_offset = next_index + 1;
      continue;  // resume from next set bit (as we know there's nothing before)
    } else {
      break;  // found
    }
  } while (true);
  return iree_min(bit_index, bitmap.bit_count);
}
