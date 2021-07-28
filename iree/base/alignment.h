// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implementation of the primitives from stdalign.h used for cross-target
// value alignment specification and queries.

#ifndef IREE_BASE_ALIGNMENT_H_
#define IREE_BASE_ALIGNMENT_H_

#include <stddef.h>

#include "iree/base/config.h"
#include "iree/base/target_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Alignment utilities
//===----------------------------------------------------------------------===//

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

// Aligns |value| up to the given power-of-two |alignment| if required.
// https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
static inline iree_device_size_t iree_device_align(
    iree_device_size_t value, iree_device_size_t alignment) {
  return (value + (alignment - 1)) & ~(alignment - 1);
}

// Returns the size of a struct padded out to iree_max_align_t.
// This must be used when performing manual trailing allocation packing to
// ensure the alignment requirements of the trailing data are satisified.
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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_ALIGNMENT_H_
