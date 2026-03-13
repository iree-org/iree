// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_CPU_SET_H_
#define IREE_BASE_INTERNAL_CPU_SET_H_

#include "iree/base/api.h"
#include "iree/base/internal/math.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Rationale for using uint64 instead of uintptr:
// * On 64-bit archs, this makes no difference.
// * On 32-bit archs, this ensures that we can use the top bit for tagging.
//   Otherwise, as 32-bit archs can use up all their pointer bits, we would have
//   to perform more complicated low-bit tagging.
// * This uniformity is useful as the direct storage access leaks the word size.
// * This also ensures that we always have inline storage up to the same bit
//   count.
typedef uint64_t iree_cpu_set_word_t;

#define IREE_CPU_SET_BITS_PER_WORD (8 * sizeof(iree_cpu_set_word_t))
#define IREE_CPU_SET_INLINE_VALUE_IS_INLINE_SHIFT \
  (IREE_CPU_SET_BITS_PER_WORD - 1)
#define IREE_CPU_SET_INLINE_VALUE_BIT_SIZE_SHIFT \
  (IREE_CPU_SET_BITS_PER_WORD - 8)
#define IREE_CPU_SET_INLINE_VALUE_IS_INLINE_MASK \
  (((iree_cpu_set_word_t)1) << IREE_CPU_SET_INLINE_VALUE_IS_INLINE_SHIFT)
#define IREE_CPU_SET_INLINE_VALUE_BIT_SIZE_MASK \
  (IREE_CPU_SET_INLINE_VALUE_IS_INLINE_MASK -   \
   (((iree_cpu_set_word_t)1) << IREE_CPU_SET_INLINE_VALUE_BIT_SIZE_SHIFT))
#define IREE_CPU_SET_INLINE_VALUE_BITS_MASK     \
  (~(IREE_CPU_SET_INLINE_VALUE_IS_INLINE_MASK | \
     IREE_CPU_SET_INLINE_VALUE_BIT_SIZE_MASK))

// iree_cpu_set_t: a better cpu_set_t.
//
// Standard cpu_set_t is hampered by compatibility requirements:
// * It grew in an era where inline storage was all one'd ever need. Now it is
//   stranded with both a large static size (1024 bit in glibc) AND a dynamic
//   mode.
// * Its bit layout is opaque, forcing inefficient bitwise access.
//
// iree_cpu_set adresses these issues:
// * Explicit layout by 64-bit words similar to iree_bitmap_t (but owning
//   its data, unlike iree_bitmap_t which is non-owning).
// * Direct word addressing.
// * Static size just a single word. Storage is inline up to 56 bits.
//   * Inline-storage case: Indicated by bit 63 set to 1. Bit storage in bits
//     0-55. bit size encoded in bits 56-62.
//   * Dynamic-storage case: Indicated by bit 63 set to 0. Then the whole
//     64-bit value is directly a pointer (not even a tagged pointer) to
//     out-of-line storage. The out-of-line storage starts with one uint64
//     holding the bit size, followed by the bits storage itself.
// * The only assumption on pointers is that bit 63 is 0. No tagged pointer is
//   stored in memory and no additional pointer bits are used.
//
typedef struct iree_cpu_set_t {
  iree_cpu_set_word_t raw;
} iree_cpu_set_t;

// Returns true if storage is inline. Equivalent to: bit_size <= 56.
static inline bool iree_cpu_set_is_inline(iree_cpu_set_t cpu_set) {
  return cpu_set.raw & IREE_CPU_SET_INLINE_VALUE_IS_INLINE_MASK;
}

// Returns the bit size. This is set once during allocation, immutable after.
// Warning: this returns the storage size, NOT the number of bits that are one.
// See iree_cpu_set_population_count.
static inline iree_host_size_t iree_cpu_set_get_bit_size(
    iree_cpu_set_t cpu_set) {
  if (iree_cpu_set_is_inline(cpu_set)) {
    return (cpu_set.raw & IREE_CPU_SET_INLINE_VALUE_BIT_SIZE_MASK) >>
           IREE_CPU_SET_INLINE_VALUE_BIT_SIZE_SHIFT;
  }
  return ((const iree_cpu_set_word_t*)cpu_set.raw)[0];
}

// Returns the number of iree_cpu_set_word_t words needed to store the given
// number of bits.
// This may return 1 even if storage is out-of-line (for bit sizes 57..64).
// This does not include the header storing bit size in out-of-line storage,
// so this isn't the allocation size.
static inline iree_host_size_t iree_cpu_set_word_count(
    iree_host_size_t bit_size) {
  return (bit_size + IREE_CPU_SET_BITS_PER_WORD - 1) /
         IREE_CPU_SET_BITS_PER_WORD;
}

// Returns a pointer to the words storing the bits. Bit 0 in this cpu set is
// bit 0 in the returned words[0].
static inline iree_cpu_set_word_t* iree_cpu_set_get_mutable_words(
    iree_cpu_set_t* cpu_set) {
  if (iree_cpu_set_is_inline(*cpu_set)) {
    return &cpu_set->raw;
  }
  return ((iree_cpu_set_word_t*)cpu_set->raw) + 1;
}

// Returns a pointer to the words storing the bits. Bit 0 in this cpu set is
// bit 0 in the returned words[0].
static inline const iree_cpu_set_word_t* iree_cpu_set_get_const_words(
    const iree_cpu_set_t* cpu_set) {
  if (iree_cpu_set_is_inline(*cpu_set)) {
    return &cpu_set->raw;
  }
  return ((const iree_cpu_set_word_t*)cpu_set->raw) + 1;
}

// Sets the specified bit to 1. Like CPU_SET.
static inline void iree_cpu_set_set_bit(iree_cpu_set_t* cpu_set,
                                        iree_host_size_t bit_pos) {
  iree_cpu_set_word_t* words = iree_cpu_set_get_mutable_words(cpu_set);
  iree_host_size_t word_index = bit_pos / IREE_CPU_SET_BITS_PER_WORD;
  iree_host_size_t bit_index = bit_pos % IREE_CPU_SET_BITS_PER_WORD;
  words[word_index] |= (((iree_cpu_set_word_t)1) << bit_index);
}

// Sets the specified bit to 0. Like CPU_CLR.
static inline void iree_cpu_set_clear_bit(iree_cpu_set_t* cpu_set,
                                          iree_host_size_t bit_pos) {
  iree_cpu_set_word_t* words = iree_cpu_set_get_mutable_words(cpu_set);
  iree_host_size_t word_index = bit_pos / IREE_CPU_SET_BITS_PER_WORD;
  iree_host_size_t bit_index = bit_pos % IREE_CPU_SET_BITS_PER_WORD;
  words[word_index] &= ~(((iree_cpu_set_word_t)1) << bit_index);
}

// Returns the specified bit. Like CPU_ISSET.
static inline bool iree_cpu_set_get_bit(iree_cpu_set_t cpu_set,
                                        iree_host_size_t bit_pos) {
  const iree_cpu_set_word_t* words = iree_cpu_set_get_const_words(&cpu_set);
  iree_host_size_t word_index = bit_pos / IREE_CPU_SET_BITS_PER_WORD;
  iree_host_size_t bit_index = bit_pos % IREE_CPU_SET_BITS_PER_WORD;
  return 0 != (words[word_index] & (((iree_cpu_set_word_t)1) << bit_index));
}

// Resets all bits to 0. Like CPU_ZERO.
static inline void iree_cpu_set_clear(iree_cpu_set_t* cpu_set) {
  if (iree_cpu_set_is_inline(*cpu_set)) {
    cpu_set->raw &= ~IREE_CPU_SET_INLINE_VALUE_BITS_MASK;
    return;
  }
  iree_host_size_t bit_size = iree_cpu_set_get_bit_size(*cpu_set);
  iree_host_size_t word_count = iree_cpu_set_word_count(bit_size);
  iree_cpu_set_word_t* words = iree_cpu_set_get_mutable_words(cpu_set);
  memset(words, 0, word_count * sizeof(iree_cpu_set_word_t));
}

// Returns the number of bits set to 1. Like CPU_COUNT.
static inline iree_host_size_t iree_cpu_set_population_count(
    iree_cpu_set_t cpu_set) {
  if (iree_cpu_set_is_inline(cpu_set)) {
    return iree_math_count_ones_u64(cpu_set.raw &
                                    IREE_CPU_SET_INLINE_VALUE_BITS_MASK);
  }
  iree_host_size_t bit_size = iree_cpu_set_get_bit_size(cpu_set);
  iree_host_size_t word_count = iree_cpu_set_word_count(bit_size);
  const iree_cpu_set_word_t* words = iree_cpu_set_get_const_words(&cpu_set);
  iree_host_size_t popcount = 0;
  for (iree_host_size_t i = 0; i < word_count; ++i) {
    popcount += iree_math_count_ones_u64(words[i]);
  }
  return popcount;
}

// Returns true if the sets are equal. Like CPU_EQUAL.
static inline bool iree_cpu_set_equal(iree_cpu_set_t cpu_set_1,
                                      iree_cpu_set_t cpu_set_2) {
  // Early return when both sets are inline. Important as that's the most common
  // case, and it's far cheaper as it's a single word comparison.
  if (iree_cpu_set_is_inline(cpu_set_1) && iree_cpu_set_is_inline(cpu_set_2)) {
    return cpu_set_1.raw == cpu_set_2.raw;
  }

  iree_host_size_t bit_size = iree_cpu_set_get_bit_size(cpu_set_1);
  iree_host_size_t bit_size_2 = iree_cpu_set_get_bit_size(cpu_set_2);
  if (bit_size_2 != bit_size) {
    return false;
  }
  const iree_cpu_set_word_t* words_1 = iree_cpu_set_get_const_words(&cpu_set_1);
  const iree_cpu_set_word_t* words_2 = iree_cpu_set_get_const_words(&cpu_set_2);
  iree_host_size_t word_count = iree_cpu_set_word_count(bit_size);
  return 0 ==
         memcmp(words_1, words_2, word_count * sizeof(iree_cpu_set_word_t));
}

// Allocates a new set with the given bit size. Like CPU_ALLOC, but also
// initializing all bits to 0 in the allocated words, even past the bit size.
static inline iree_status_t iree_cpu_set_allocate(iree_allocator_t allocator,
                                                  iree_host_size_t bit_size,
                                                  iree_cpu_set_t* out_cpu_set) {
  if (bit_size <= IREE_CPU_SET_INLINE_VALUE_BIT_SIZE_SHIFT) {
    // Initialize all stored bits to 0, even past bit_size.
    out_cpu_set->raw = IREE_CPU_SET_INLINE_VALUE_IS_INLINE_MASK |
                       (((iree_cpu_set_word_t)bit_size)
                        << IREE_CPU_SET_INLINE_VALUE_BIT_SIZE_SHIFT);
    return iree_ok_status();
  }
  iree_host_size_t words_to_allocate =
      /*header: bit size*/ 1 + iree_cpu_set_word_count(bit_size);
  iree_cpu_set_word_t* words = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, words_to_allocate * sizeof(iree_cpu_set_word_t),
      (void**)&words));
  // Initialize all stored bits to 0, even past bit_size.
  memset(words, 0, words_to_allocate * sizeof(iree_cpu_set_word_t));
  words[0] = bit_size;  // header: bit size
  out_cpu_set->raw = (iree_cpu_set_word_t)(uintptr_t)words;
  assert(!iree_cpu_set_is_inline(*out_cpu_set) &&
         "Malloc'd pointer has high bit set!");
  return iree_ok_status();
}

// Deallocates a set. Like CPU_FREE.
static inline void iree_cpu_set_free(iree_allocator_t allocator,
                                     iree_cpu_set_t* cpu_set) {
  if (!iree_cpu_set_is_inline(*cpu_set)) {
    iree_allocator_free(allocator, (void*)(uintptr_t)(cpu_set->raw));
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_INTERNAL_CPU_SET_H_
