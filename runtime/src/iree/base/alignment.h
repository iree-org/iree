// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implementation of the primitives from stdalign.h used for cross-target
// value alignment specification and queries.

#ifndef IREE_BASE_ALIGNMENT_H_
#define IREE_BASE_ALIGNMENT_H_

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/attributes.h"
#include "iree/base/config.h"
#include "iree/base/target_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// IREE_PTR_SIZE_*
//==============================================================================

// Verify that the pointer size of the machine matches the expectation that
// uintptr_t can round-trip the value. This isn't a common issue unless the
// toolchain is doing weird things.
// See https://stackoverflow.com/q/51616057.
static_assert(sizeof(void*) == sizeof(uintptr_t),
              "can't determine pointer size");

#if UINTPTR_MAX == 0xFFFFFFFF
#define IREE_PTR_SIZE_32
#define IREE_PTR_SIZE 4
#elif UINTPTR_MAX == 0xFFFFFFFFFFFFFFFFu
#define IREE_PTR_SIZE_64
#define IREE_PTR_SIZE 8
#else
#error "can't determine pointer size"
#endif

//===----------------------------------------------------------------------===//
// Alignment utilities
//===----------------------------------------------------------------------===//

// Returns the number of elements in an array as a compile-time constant, which
// can be used in defining new arrays. Fails at compile-time if |arr| is not a
// static array (such as if used on a pointer type). Similar to `countof()`.
//
// Example:
//  uint8_t kConstantArray[512];
//  assert(IREE_ARRAYSIZE(kConstantArray) == 512);
#define IREE_ARRAYSIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#define iree_min(lhs, rhs) ((lhs) <= (rhs) ? (lhs) : (rhs))
#define iree_max(lhs, rhs) ((lhs) <= (rhs) ? (rhs) : (lhs))

// https://en.cppreference.com/w/c/types/max_align_t
#if defined(IREE_PLATFORM_WINDOWS)
// NOTE: 16 is a specified Microsoft API requirement for some functions.
#define iree_max_align_t 16
#else
#define iree_max_align_t sizeof(long double)
#endif  // IREE_PLATFORM_*

// https://en.cppreference.com/w/c/language/_Alignas
// https://en.cppreference.com/w/c/language/_Alignof
#if defined(IREE_COMPILER_MSVC)
#define iree_alignas(x) __declspec(align(x))
#define iree_alignof(x) __alignof(x)
#else
#define iree_alignas(x) __attribute__((__aligned__(x)))
#define iree_alignof(x) __alignof__(x)
#endif  // IREE_COMPILER_*

// Aligns |value| up to the given power-of-two |alignment| if required.
// https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
static inline iree_host_size_t iree_host_align(iree_host_size_t value,
                                               iree_host_size_t alignment) {
  return (value + (alignment - 1)) & ~(alignment - 1);
}

// Returns true if |value| is a power-of-two.
static inline bool iree_host_size_is_power_of_two(iree_host_size_t value) {
  return (value != 0) && ((value & (value - 1)) == 0);
}

// Returns true if |value| matches the given minimum |alignment|.
static inline bool iree_host_size_has_alignment(iree_host_size_t value,
                                                iree_host_size_t alignment) {
  return iree_host_align(value, alignment) == value;
}

// Aligns |value| up to the given power-of-two |alignment| if required.
// https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
static inline iree_device_size_t iree_device_align(
    iree_device_size_t value, iree_device_size_t alignment) {
  return (value + (alignment - 1)) & ~(alignment - 1);
}

// Returns true if |value| is a power-of-two.
static inline bool iree_device_size_is_power_of_two(iree_device_size_t value) {
  return (value != 0) && ((value & (value - 1)) == 0);
}

// Returns true if |value| matches the given minimum |alignment|.
static inline bool iree_device_size_has_alignment(
    iree_device_size_t value, iree_device_size_t alignment) {
  return iree_device_align(value, alignment) == value;
}

// Returns true if |value| is a power-of-two.
static inline bool iree_is_power_of_two_uint64(uint64_t value) {
  return (value != 0) && ((value & (value - 1)) == 0);
}

// TODO(benvanik): when C23 is fully adopted we can make a single generic
// version of the alignment functions that uses typeof to cast back to the
// expected result. For now we explicitly spell out the variants we want.

// Aligns |value| up to the given power-of-two |alignment| if required.
// https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
static inline uint64_t iree_align_uint64(uint64_t value, uint64_t alignment) {
  return (value + (alignment - 1)) & ~(alignment - 1);
}

// Returns the size of a struct padded out to iree_max_align_t.
// This must be used when performing manual trailing allocation packing to
// ensure the alignment requirements of the trailing data are satisfied.
//
// NOTE: do not use this if using VLAs (`struct { int trailing[]; }`) - those
// must precisely follow the normal sizeof(t) as the compiler does the padding
// for you.
//
// Example:
//  some_buffer_ptr_t* p = NULL;
//  iree_host_size_t total_size = iree_sizeof_struct(*buffer) + extra_data_size;
//  IREE_CHECK_OK(iree_allocator_malloc(allocator, total_size, (void**)&p));
#define iree_sizeof_struct(t) iree_host_align(sizeof(t), iree_max_align_t)

// Returns the ceil-divide of |lhs| by non-zero |rhs|.
static inline iree_host_size_t iree_host_size_ceil_div(iree_host_size_t lhs,
                                                       iree_host_size_t rhs) {
  return ((lhs != 0) && (lhs > 0) == (rhs > 0))
             ? ((lhs + ((rhs > 0) ? -1 : 1)) / rhs) + 1
             : -(-lhs / rhs);
}

// Returns the floor-divide of |lhs| by non-zero |rhs|.
static inline iree_host_size_t iree_host_size_floor_div(iree_host_size_t lhs,
                                                        iree_host_size_t rhs) {
  return ((lhs != 0) && ((lhs < 0) != (rhs < 0)))
             ? -((-lhs + ((rhs < 0) ? 1 : -1)) / rhs) - 1
             : lhs / rhs;
}

// Returns the ceil-divide of |lhs| by non-zero |rhs|.
static inline iree_device_size_t iree_device_size_ceil_div(
    iree_device_size_t lhs, iree_device_size_t rhs) {
  return ((lhs != 0) && (lhs > 0) == (rhs > 0))
             ? ((lhs + ((rhs > 0) ? -1 : 1)) / rhs) + 1
             : -(-lhs / rhs);
}

// Returns the floor-divide of |lhs| by non-zero |rhs|.
static inline iree_device_size_t iree_device_size_floor_div(
    iree_device_size_t lhs, iree_device_size_t rhs) {
  return ((lhs != 0) && ((lhs < 0) != (rhs < 0)))
             ? -((-lhs + ((rhs < 0) ? 1 : -1)) / rhs) - 1
             : lhs / rhs;
}

// Returns the greatest common divisor between two values.
//
// See: https://en.wikipedia.org/wiki/Greatest_common_divisor
//
// Examples:
//  gcd(8, 16) = 8
//  gcd(3, 5) = 1
static inline iree_device_size_t iree_device_size_gcd(iree_device_size_t a,
                                                      iree_device_size_t b) {
  if (b == 0) return a;
  return iree_device_size_gcd(b, a % b);
}

// Returns the least common multiple between two values, often used for
// finding a common alignment.
//
// See: https://en.wikipedia.org/wiki/Least_common_multiple
//
// Examples:
//  lcm(8, 16) = 16
//  lcm(3, 5) = 15 (15 % 3 == 0, 15 % 5 == 0)
static inline iree_device_size_t iree_device_size_lcm(iree_device_size_t a,
                                                      iree_device_size_t b) {
  return a * (b / iree_device_size_gcd(a, b));
}

//===----------------------------------------------------------------------===//
// Byte and page range manipulation
//===----------------------------------------------------------------------===//

// Defines a range of bytes with any arbitrary alignment.
// Most operations will adjust this range by the allocation granularity, meaning
// that a range that straddles a page boundary will be specifying multiple pages
// (such as offset=1, length=4096 with a page size of 4096 indicating 2 pages).
typedef struct iree_byte_range_t {
  iree_host_size_t offset;
  iree_host_size_t length;
} iree_byte_range_t;

// Defines a range of bytes with page-appropriate alignment.
// Any operation taking page ranges requires that the offset and length respect
// the size and granularity requirements of the page mode the memory was defined
// with. For example, if an allocation is using large pages then both offset and
// length must be multiples of the iree_memory_info_t::large_page_granularity.
typedef struct iree_page_range_t {
  iree_host_size_t offset;
  iree_host_size_t length;
} iree_page_range_t;

// Aligns |addr| up to |page_alignment|.
static inline uintptr_t iree_page_align_start(uintptr_t addr,
                                              iree_host_size_t page_alignment) {
  return addr & (~(page_alignment - 1));
}

// Aligns |addr| down to |page_alignment|.
static inline uintptr_t iree_page_align_end(uintptr_t addr,
                                            iree_host_size_t page_alignment) {
  return iree_page_align_start(addr + (page_alignment - 1), page_alignment);
}

// Unions two page ranges to create the min/max extents of both.
static inline iree_page_range_t iree_page_range_union(
    const iree_page_range_t a, const iree_page_range_t b) {
  iree_host_size_t start = iree_min(a.offset, b.offset);
  iree_host_size_t end = iree_max(a.offset + a.length, b.offset + b.length);
  iree_page_range_t page_range = {0};
  page_range.offset = start;
  page_range.length = end - start;
  return page_range;
}

// Aligns a byte range to page boundaries defined by |page_alignment|.
static inline iree_page_range_t iree_align_byte_range_to_pages(
    const iree_byte_range_t byte_range, iree_host_size_t page_alignment) {
  iree_page_range_t page_range = {0};
  page_range.offset = iree_host_align(byte_range.offset, page_alignment);
  page_range.length = iree_host_align(byte_range.length, page_alignment);
  return page_range;
}

// Computes a page-aligned range base and total length from a range.
// This will produce a starting address <= the range offset and a length >=
// the range length.
static inline void iree_page_align_range(void* base_address,
                                         iree_byte_range_t range,
                                         iree_host_size_t page_alignment,
                                         void** out_start_address,
                                         iree_host_size_t* out_aligned_length) {
  void* range_start = (void*)iree_page_align_start(
      (uintptr_t)base_address + range.offset, page_alignment);
  void* range_end = (void*)iree_page_align_end(
      (uintptr_t)base_address + range.offset + range.length, page_alignment);
  *out_start_address = range_start;
  *out_aligned_length =
      (iree_host_size_t)range_end - (iree_host_size_t)range_start;
}

//===----------------------------------------------------------------------===//
// Alignment intrinsics
//===----------------------------------------------------------------------===//

#if IREE_HAVE_BUILTIN(__builtin_unreachable) || defined(__GNUC__)
#define IREE_BUILTIN_UNREACHABLE() __builtin_unreachable()
#elif defined(IREE_COMPILER_MSVC)
#define IREE_BUILTIN_UNREACHABLE() __assume(false)
#else
#define IREE_BUILTIN_UNREACHABLE() ((void)0)
#endif  // IREE_HAVE_BUILTIN(__builtin_unreachable) || defined(__GNUC__)

#if !defined(__cplusplus)
#define IREE_DECLTYPE(v) __typeof__(v)
#else
#define IREE_DECLTYPE(v) decltype(v)
#endif  // __cplusplus

#if IREE_HAVE_BUILTIN(__builtin_assume_aligned) || defined(__GNUC__)
// NOTE: gcc only assumes on the result so we have to reset ptr.
#define IREE_BUILTIN_ASSUME_ALIGNED(ptr, size) \
  (ptr = (IREE_DECLTYPE(ptr))(__builtin_assume_aligned((void*)(ptr), (size))))
#elif 0  // defined(IREE_COMPILER_MSVC)
#define IREE_BUILTIN_ASSUME_ALIGNED(ptr, size) \
  (__assume((((uintptr_t)(ptr)) & ((1 << (size))) - 1)) == 0)
#else
#define IREE_BUILTIN_ASSUME_ALIGNED(ptr, size) \
  ((((uintptr_t)(ptr) % (size)) == 0) ? (ptr)  \
                                      : (IREE_BUILTIN_UNREACHABLE(), (ptr)))
#endif  // IREE_HAVE_BUILTIN(__builtin_assume_aligned) || defined(__GNUC__)

//===----------------------------------------------------------------------===//
// Alignment-safe memory accesses
//===----------------------------------------------------------------------===//

// Map little-endian byte indices in memory to the host memory order indices.
#if defined(IREE_ENDIANNESS_LITTLE)
#define IREE_LE_IDX_1(i) (i)
#define IREE_LE_IDX_2(i) (i)
#define IREE_LE_IDX_4(i) (i)
#define IREE_LE_IDX_8(i) (i)
#else
#define IREE_LE_IDX_1(i) (i)
#define IREE_LE_IDX_2(i) (1 - (i))
#define IREE_LE_IDX_4(i) (3 - (i))
#define IREE_LE_IDX_8(i) (7 - (i))
#endif  // IREE_ENDIANNESS_*

#if IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED_8

static inline uint8_t iree_unaligned_load_le_u8(const uint8_t* ptr) {
  return *ptr;
}

static inline void iree_unaligned_store_le_u8(uint8_t* ptr, uint8_t value) {
  *ptr = value;
}

#else

#if defined(IREE_ENDIANNESS_LITTLE)

#define iree_unaligned_load_le_u8(ptr) *(ptr)

#define iree_unaligned_store_le_u8(ptr, value) *(ptr) = (value)

#else

#error "TODO(benvanik): little-endian load/store for big-endian archs"

#endif  // IREE_ENDIANNESS_*

#endif  // IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED_8

#if IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED_16

static inline uint16_t iree_unaligned_load_le_u16(const uint16_t* ptr) {
  const uint8_t* p = (const uint8_t*)ptr;
  return ((uint16_t)p[IREE_LE_IDX_2(0)]) | ((uint16_t)p[IREE_LE_IDX_2(1)] << 8);
}

static inline void iree_unaligned_store_le_u16(uint16_t* ptr, uint16_t value) {
  uint8_t* p = (uint8_t*)ptr;
  p[IREE_LE_IDX_2(0)] = value;
  p[IREE_LE_IDX_2(1)] = value >> 8;
}

#else

#if defined(IREE_ENDIANNESS_LITTLE)

#define iree_unaligned_load_le_u16(ptr) *(ptr)

#define iree_unaligned_store_le_u16(ptr, value) *(ptr) = (value)

#else

#error "TODO(benvanik): little-endian load/store for big-endian archs"

#endif  // IREE_ENDIANNESS_*

#endif  // IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED_16

#if IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED_32

static inline uint32_t iree_unaligned_load_le_u32(const uint32_t* ptr) {
  const uint8_t* p = (const uint8_t*)ptr;
  return ((uint32_t)p[IREE_LE_IDX_4(0)]) |
         ((uint32_t)p[IREE_LE_IDX_4(1)] << 8) |
         ((uint32_t)p[IREE_LE_IDX_4(2)] << 16) |
         ((uint32_t)p[IREE_LE_IDX_4(3)] << 24);
}
static inline float iree_unaligned_load_le_f32(const float* ptr) {
  uint32_t uint_value = iree_unaligned_load_le_u32((const uint32_t*)ptr);
  float value;
  memcpy(&value, &uint_value, sizeof(value));
  return value;
}

static inline void iree_unaligned_store_le_u32(uint32_t* ptr, uint32_t value) {
  uint8_t* p = (uint8_t*)ptr;
  p[IREE_LE_IDX_4(0)] = value;
  p[IREE_LE_IDX_4(1)] = value >> 8;
  p[IREE_LE_IDX_4(2)] = value >> 16;
  p[IREE_LE_IDX_4(3)] = value >> 24;
}
static inline void iree_unaligned_store_le_f32(float* ptr, float value) {
  uint32_t uint_value;
  memcpy(&uint_value, &value, sizeof(value));
  iree_unaligned_store_le_u32((uint32_t*)ptr, uint_value);
}

#else

#if defined(IREE_ENDIANNESS_LITTLE)

#define iree_unaligned_load_le_u32(ptr) *(ptr)
#define iree_unaligned_load_le_f32(ptr) *(ptr)

#define iree_unaligned_store_le_u32(ptr, value) *(ptr) = (value)
#define iree_unaligned_store_le_f32(ptr, value) *(ptr) = (value)

#else

#error "TODO(benvanik): little-endian load/store for big-endian archs"

#endif  // IREE_ENDIANNESS_*

#endif  // IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED_32

#if IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED_64

static inline uint64_t iree_unaligned_load_le_u64(const uint64_t* ptr) {
  const uint8_t* p = (const uint8_t*)ptr;
  return ((uint64_t)p[IREE_LE_IDX_8(0)]) |
         ((uint64_t)p[IREE_LE_IDX_8(1)] << 8) |
         ((uint64_t)p[IREE_LE_IDX_8(2)] << 16) |
         ((uint64_t)p[IREE_LE_IDX_8(3)] << 24) |
         ((uint64_t)p[IREE_LE_IDX_8(4)] << 32) |
         ((uint64_t)p[IREE_LE_IDX_8(5)] << 40) |
         ((uint64_t)p[IREE_LE_IDX_8(6)] << 48) |
         ((uint64_t)p[IREE_LE_IDX_8(7)] << 56);
}
static inline double iree_unaligned_load_le_f64(const double* ptr) {
  uint64_t uint_value = iree_unaligned_load_le_u64((const uint64_t*)ptr);
  double value;
  memcpy(&value, &uint_value, sizeof(value));
  return value;
}

static inline void iree_unaligned_store_le_u64(uint64_t* ptr, uint64_t value) {
  uint8_t* p = (uint8_t*)ptr;
  p[IREE_LE_IDX_8(0)] = value;
  p[IREE_LE_IDX_8(1)] = value >> 8;
  p[IREE_LE_IDX_8(2)] = value >> 16;
  p[IREE_LE_IDX_8(3)] = value >> 24;
  p[IREE_LE_IDX_8(4)] = value >> 32;
  p[IREE_LE_IDX_8(5)] = value >> 40;
  p[IREE_LE_IDX_8(6)] = value >> 48;
  p[IREE_LE_IDX_8(7)] = value >> 56;
}
static inline void iree_unaligned_store_le_f64(double* ptr, double value) {
  uint64_t uint_value;
  memcpy(&uint_value, &value, sizeof(value));
  iree_unaligned_store_le_u64((uint64_t*)ptr, uint_value);
}

#else

#if defined(IREE_ENDIANNESS_LITTLE)

#define iree_unaligned_load_le_u64(ptr) *(ptr)
#define iree_unaligned_load_le_f64(ptr) *(ptr)

#define iree_unaligned_store_le_u64(ptr, value) *(ptr) = (value)
#define iree_unaligned_store_le_f64(ptr, value) *(ptr) = (value)

#else

#error "TODO(benvanik): little-endian load/store for big-endian archs"

#endif  // IREE_ENDIANNESS_*

#endif  // IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED_64

// clang-format off

// Dereferences |ptr| and returns the value.
// Automatically handles unaligned accesses on architectures that may not
// support them natively (or efficiently). Memory is treated as little-endian.
#define iree_unaligned_load_le(ptr)                                               \
  _Generic((ptr),                                                              \
        int8_t*: iree_unaligned_load_le_u8((const uint8_t*)(ptr)),             \
       uint8_t*: iree_unaligned_load_le_u8((const uint8_t*)(ptr)),             \
       int16_t*: iree_unaligned_load_le_u16((const uint16_t*)(ptr)),           \
      uint16_t*: iree_unaligned_load_le_u16((const uint16_t*)(ptr)),           \
       int32_t*: iree_unaligned_load_le_u32((const uint32_t*)(ptr)),           \
      uint32_t*: iree_unaligned_load_le_u32((const uint32_t*)(ptr)),           \
       int64_t*: iree_unaligned_load_le_u64((const uint64_t*)(ptr)),           \
      uint64_t*: iree_unaligned_load_le_u64((const uint64_t*)(ptr)),           \
         float*: iree_unaligned_load_le_f32((const float*)(ptr)),              \
        double*: iree_unaligned_load_le_f64((const double*)(ptr))              \
  )

// Dereferences |ptr| and writes the given |value|.
// Automatically handles unaligned accesses on architectures that may not
// support them natively (or efficiently). Memory is treated as little-endian.
#define iree_unaligned_store(ptr, value)                                       \
  _Generic((ptr),                                                              \
        int8_t*: iree_unaligned_store_le_u8((uint8_t*)(ptr), value),           \
       uint8_t*: iree_unaligned_store_le_u8((uint8_t*)(ptr), value),           \
       int16_t*: iree_unaligned_store_le_u16((uint16_t*)(ptr), value),         \
      uint16_t*: iree_unaligned_store_le_u16((uint16_t*)(ptr), value),         \
       int32_t*: iree_unaligned_store_le_u32((uint32_t*)(ptr), value),         \
      uint32_t*: iree_unaligned_store_le_u32((uint32_t*)(ptr), value),         \
       int64_t*: iree_unaligned_store_le_u64((uint64_t*)(ptr), value),         \
      uint64_t*: iree_unaligned_store_le_u64((uint64_t*)(ptr), value),         \
         float*: iree_unaligned_store_le_f32((float*)(ptr), value),            \
        double*: iree_unaligned_store_le_f64((double*)(ptr), value)            \
  )

// clang-format on

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_ALIGNMENT_H_
