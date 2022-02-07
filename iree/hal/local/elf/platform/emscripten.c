// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/local/elf/platform.h"

#if defined(IREE_PLATFORM_EMSCRIPTEN)

#include <malloc.h>
#include <stdlib.h>

//==============================================================================
// Memory subsystem information and control
//==============================================================================

// WebAssembly.Memory pages are 64KB (2^16 bytes).
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WebAssembly/Memory
#define IREE_MEMORY_PAGE_SIZE_NORMAL 65536
#define IREE_MEMORY_PAGE_SIZE_LARGE 65536

void iree_memory_query_info(iree_memory_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));

  out_info->normal_page_size = IREE_MEMORY_PAGE_SIZE_NORMAL;
  out_info->normal_page_granularity = IREE_MEMORY_PAGE_SIZE_NORMAL;
  out_info->large_page_granularity = IREE_MEMORY_PAGE_SIZE_LARGE;

  out_info->can_allocate_executable_pages = false;
}

void iree_memory_jit_context_begin(void) {}

void iree_memory_jit_context_end(void) {}

//==============================================================================
// Virtual address space manipulation
//==============================================================================

iree_status_t iree_memory_view_reserve(iree_memory_view_flags_t flags,
                                       iree_host_size_t total_length,
                                       iree_allocator_t allocator,
                                       void** out_base_address) {
  *out_base_address = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      iree_allocator_malloc(allocator, total_length, out_base_address);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_memory_view_release(void* base_address, iree_host_size_t total_length,
                              iree_allocator_t allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(allocator, base_address);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_memory_view_commit_ranges(
    void* base_address, iree_host_size_t range_count,
    const iree_byte_range_t* ranges, iree_memory_access_t initial_access) {
  // No-op.
  return iree_ok_status();
}

iree_status_t iree_memory_view_protect_ranges(void* base_address,
                                              iree_host_size_t range_count,
                                              const iree_byte_range_t* ranges,
                                              iree_memory_access_t new_access) {
  // No-op.
  return iree_ok_status();
}

void iree_memory_view_flush_icache(void* base_address,
                                   iree_host_size_t length) {
  // WebAssembly does not support the llvm.clear_cache intrinsic, see
  // https://reviews.llvm.org/D64322.
  // __builtin___clear_cache(base_address, base_address + length);
}

#endif  // IREE_PLATFORM_GENERIC
