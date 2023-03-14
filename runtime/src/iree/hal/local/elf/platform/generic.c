// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/local/elf/platform.h"

#if defined(IREE_PLATFORM_GENERIC)

#include <malloc.h>
#include <stdlib.h>

//==============================================================================
// Memory subsystem information and control
//==============================================================================

// TODO(benvanik): control with a config.h.
#define IREE_MEMORY_PAGE_SIZE_NORMAL 4096
#define IREE_MEMORY_PAGE_SIZE_LARGE 4096

void iree_memory_query_info(iree_memory_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));

  out_info->normal_page_size = IREE_MEMORY_PAGE_SIZE_NORMAL;
  out_info->normal_page_granularity = IREE_MEMORY_PAGE_SIZE_NORMAL;
  out_info->large_page_granularity = IREE_MEMORY_PAGE_SIZE_LARGE;

  out_info->can_allocate_executable_pages = true;
}

void iree_memory_jit_context_begin(void) {}

void iree_memory_jit_context_end(void) {}

//==============================================================================
// Virtual address space manipulation
//==============================================================================

iree_status_t iree_memory_view_reserve(iree_memory_view_flags_t flags,
                                       iree_host_size_t total_length,
                                       iree_allocator_t host_allocator,
                                       void** out_base_address) {
  *out_base_address = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_length, out_base_address);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_memory_view_release(void* base_address, iree_host_size_t total_length,
                              iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(host_allocator, base_address);
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

// IREE_ELF_CLEAR_CACHE can be defined externally to override this default
// behavior.
#if !defined(IREE_ELF_CLEAR_CACHE)
// __has_builtin was added in GCC 10, so just hard-code the availability
// for < 10, special cased here so it can be dropped once no longer needed.
#if defined __GNUC__ && __GNUC__ < 10
#define IREE_ELF_CLEAR_CACHE(start, end) __builtin___clear_cache(start, end)
#elif defined __has_builtin
#if __has_builtin(__builtin___clear_cache)
#define IREE_ELF_CLEAR_CACHE(start, end) __builtin___clear_cache(start, end)
#endif  // __builtin___clear_cache
#endif  // __has_builtin
#endif  // !defined(IREE_ELF_CLEAR_CACHE)

#if !defined(IREE_ELF_CLEAR_CACHE)
#error "no instruction cache clear implementation"
#endif  // !defined(IREE_ELF_CLEAR_CACHE)

void iree_memory_view_flush_icache(void* base_address,
                                   iree_host_size_t length) {
  IREE_ELF_CLEAR_CACHE(base_address, ((char*)base_address) + length);
}

#endif  // IREE_PLATFORM_GENERIC
