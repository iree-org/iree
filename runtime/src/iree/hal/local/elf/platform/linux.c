// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/local/elf/platform.h"

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)

#include <errno.h>
#include <sys/mman.h>
#include <unistd.h>

//==============================================================================
// Memory subsystem information and control
//==============================================================================

void iree_memory_query_info(iree_memory_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));

  int page_size = sysconf(_SC_PAGESIZE);
  out_info->normal_page_size = page_size;
  out_info->normal_page_granularity = page_size;

  // Large pages arent't currently used so we aren't introducing the build goo
  // to detect and use them yet.
  // https://linux.die.net/man/3/gethugepagesizes
  // http://manpages.ubuntu.com/manpages/bionic/man3/gethugepagesize.3.html
  // Would be:
  //   #include <hugetlbfs.h>
  //   out_info->large_page_granularity = gethugepagesize();
  out_info->large_page_granularity = page_size;

  out_info->can_allocate_executable_pages = true;
}

void iree_memory_jit_context_begin(void) {}

void iree_memory_jit_context_end(void) {}

//==============================================================================
// Virtual address space manipulation
//==============================================================================

static int iree_memory_access_to_prot(iree_memory_access_t access) {
  int prot = 0;
  if (access & IREE_MEMORY_ACCESS_READ) prot |= PROT_READ;
  if (access & IREE_MEMORY_ACCESS_WRITE) prot |= PROT_WRITE;
  if (access & IREE_MEMORY_ACCESS_EXECUTE) prot |= PROT_EXEC;
  return prot;
}

iree_status_t iree_memory_view_reserve(iree_memory_view_flags_t flags,
                                       iree_host_size_t total_length,
                                       iree_allocator_t host_allocator,
                                       void** out_base_address) {
  *out_base_address = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  int mmap_prot = PROT_NONE;
  int mmap_flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;

  iree_status_t status = iree_ok_status();
  void* base_address = mmap(NULL, total_length, mmap_prot, mmap_flags, -1, 0);
  if (base_address == MAP_FAILED) {
    status = iree_make_status(iree_status_code_from_errno(errno),
                              "mmap reservation failed");
  }

  *out_base_address = base_address;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_memory_view_release(void* base_address, iree_host_size_t total_length,
                              iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // NOTE: return value ignored as this is a shutdown path.
  munmap(base_address, total_length);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_memory_view_commit_ranges(
    void* base_address, iree_host_size_t range_count,
    const iree_byte_range_t* ranges, iree_memory_access_t initial_access) {
  IREE_TRACE_ZONE_BEGIN(z0);

  int mmap_prot = iree_memory_access_to_prot(initial_access);
  int mmap_flags = MAP_PRIVATE | MAP_ANON | MAP_FIXED;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < range_count; ++i) {
    void* range_start = NULL;
    iree_host_size_t aligned_length = 0;
    iree_page_align_range(base_address, ranges[i], getpagesize(), &range_start,
                          &aligned_length);
    void* result =
        mmap(range_start, aligned_length, mmap_prot, mmap_flags, -1, 0);
    if (result == MAP_FAILED) {
      status = iree_make_status(iree_status_code_from_errno(errno),
                                "mmap commit failed");
      break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_memory_view_protect_ranges(void* base_address,
                                              iree_host_size_t range_count,
                                              const iree_byte_range_t* ranges,
                                              iree_memory_access_t new_access) {
  IREE_TRACE_ZONE_BEGIN(z0);

  int mmap_prot = iree_memory_access_to_prot(new_access);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < range_count; ++i) {
    void* range_start = NULL;
    iree_host_size_t aligned_length = 0;
    iree_page_align_range(base_address, ranges[i], getpagesize(), &range_start,
                          &aligned_length);
    int ret = mprotect(range_start, aligned_length, mmap_prot);
    if (ret != 0) {
      status = iree_make_status(iree_status_code_from_errno(errno),
                                "mprotect failed");
      break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
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

#endif  // IREE_PLATFORM_*
