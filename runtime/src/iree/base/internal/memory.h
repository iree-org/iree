// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_MEMORY_H_
#define IREE_BASE_INTERNAL_MEMORY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Memory subsystem information and control
//===----------------------------------------------------------------------===//

enum iree_memory_feature_bits_e {
  IREE_MEMORY_FEATURE_NONE = 0u,

  // Indicates whether executable pages may be allocated within the process.
  // Some platforms or release environments have restrictions on whether
  // executable pages may be allocated from user code (such as iOS).
  IREE_MEMORY_FEATURE_ALLOCATABLE_EXECUTABLE_PAGES = 1u << 0,
};
typedef uint32_t iree_memory_features_t;

// System platform/environment information defining memory parameters.
// These can be used to control application behavior (such as whether to enable
// a JIT if executable pages can be allocated) and allow callers to compute
// memory ranges based on the variable page size of the platform.
typedef struct iree_memory_info_t {
  // The page size and the granularity of page protection and commitment. This
  // is the page size used by the iree_memory_view_t functions.
  iree_host_size_t normal_page_size;

  // The granularity for the starting address at which virtual memory can be
  // allocated.
  iree_host_size_t normal_page_granularity;

  // The minimum page size and granularity for large pages or 0 if unavailable.
  // To use large pages the size and alignment must be a multiple of this value
  // and the IREE_MEMORY_VIEW_FLAG_LARGE_PAGES must be set.
  iree_host_size_t large_page_granularity;

  // Bitfield of features supported by the platform.
  iree_memory_features_t supported_features;
} iree_memory_info_t;

// Queries the system platform/environment memory information.
// Callers should cache the results to avoid repeated queries, such as storing
// the used fields in an allocator upon initialization to reuse during
// allocations made via the allocator.
iree_memory_info_t iree_memory_query_info(void);

// Enter a W^X region where pages will be changed RW->RX or RX->RW and write
// protection should be suspended. Only effects the calling thread and must be
// paired with iree_memory_jit_context_end.
void iree_memory_jit_context_begin(void);

// Exits a W^X region previously entered with iree_memory_jit_context_begin.
void iree_memory_jit_context_end(void);

// Flushes the CPU instruction cache for a given range of bytes.
// May be a no-op depending on architecture, but must be called prior to
// executing code from any pages that have been written during load.
void iree_memory_flush_icache(void* base_address, iree_host_size_t length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_INTERNAL_MEMORY_H_
