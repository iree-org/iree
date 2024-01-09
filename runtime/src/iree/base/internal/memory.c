// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/memory.h"

//===----------------------------------------------------------------------===//
// Memory subsystem information and control
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_APPLE)

#include <unistd.h>

iree_memory_info_t iree_memory_query_info(void) {
  const int page_size = sysconf(_SC_PAGESIZE);
  return (iree_memory_info_t){
      .normal_page_size = page_size,
      .normal_page_granularity = page_size,
      .large_page_granularity = 2 * 1024 * 1024,  // What V8 uses.
      .supported_features = IREE_MEMORY_FEATURE_ALLOCATABLE_EXECUTABLE_PAGES,
  };
}

#elif defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)

#include <unistd.h>

iree_memory_info_t iree_memory_query_info(void) {
  const int page_size = sysconf(_SC_PAGESIZE);
  return (iree_memory_info_t){
      .normal_page_size = page_size,
      .normal_page_granularity = page_size,
      // Large pages arent't currently used so we aren't introducing the build
      // goo to detect and use them yet.
      // https://linux.die.net/man/3/gethugepagesizes
      // http://manpages.ubuntu.com/manpages/bionic/man3/gethugepagesize.3.html
      // Would be:
      //   #include <hugetlbfs.h>
      //   out_info->large_page_granularity = gethugepagesize();
      .large_page_granularity = page_size,
      .supported_features = IREE_MEMORY_FEATURE_ALLOCATABLE_EXECUTABLE_PAGES,
  };
}

#elif defined(IREE_PLATFORM_WINDOWS)

iree_memory_info_t iree_memory_query_info(void) {
  // When running in non-desktop mode the application can define the
  // `codeGeneration` property to enable use of PAGE_EXECUTE but cannot use
  // PAGE_EXECUTE_READWRITE - it's still possible to make that work but it
  // requires aliasing views (one with READWRITE and one with EXECUTE) and I'm
  // not sure if anyone will ever care.
  iree_memory_features_t supported_features = 0;
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
  supported_features |= IREE_MEMORY_FEATURE_ALLOCATABLE_EXECUTABLE_PAGES;
#endif  // WINAPI_PARTITION_DESKTOP

  SYSTEM_INFO system_info;
  GetSystemInfo(&system_info);
  return (iree_memory_info_t){
      .normal_page_size = system_info.dwPageSize,
      .normal_page_granularity = system_info.dwAllocationGranularity,
      .large_page_granularity = GetLargePageMinimum(),
      .supported_features = supported_features,
  };
}

#else

// Users can override these with compiler defines if they want.
#if !defined(IREE_MEMORY_PAGE_SIZE_NORMAL)
#define IREE_MEMORY_PAGE_SIZE_NORMAL 4096
#endif  // !IREE_MEMORY_PAGE_SIZE_NORMAL
#if !defined(IREE_MEMORY_PAGE_SIZE_LARGE)
#define IREE_MEMORY_PAGE_SIZE_LARGE 4096
#endif  // !IREE_MEMORY_PAGE_SIZE_LARGE

iree_memory_info_t iree_memory_query_info(void) {
  return (iree_memory_info_t){
      .normal_page_size = IREE_MEMORY_PAGE_SIZE_NORMAL,
      .normal_page_granularity = IREE_MEMORY_PAGE_SIZE_NORMAL,
      .large_page_granularity = IREE_MEMORY_PAGE_SIZE_LARGE,
      .supported_features = IREE_MEMORY_FEATURE_ALLOCATABLE_EXECUTABLE_PAGES,
  };
}

#endif  // IREE_PLATFORM_*

#if defined(IREE_PLATFORM_APPLE) && defined(MAC_OS_VERSION_11_0) && \
    MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_11_0

#include <pthread.h>

void iree_memory_jit_context_begin(void) {
  if (__builtin_available(macOS 11.0, *) &&
      pthread_jit_write_protect_supported_np()) {
    pthread_jit_write_protect_np(0);
  }
}

void iree_memory_jit_context_end(void) {
  if (__builtin_available(macOS 11.0, *) &&
      pthread_jit_write_protect_supported_np()) {
    pthread_jit_write_protect_np(1);
  }
}

#else

void iree_memory_jit_context_begin(void) {}
void iree_memory_jit_context_end(void) {}

#endif  // IREE_PLATFORM_APPLE

#if defined(IREE_PLATFORM_APPLE)

void sys_icache_invalidate(void* start, size_t len);
void iree_memory_flush_icache(void* base_address, iree_host_size_t length) {
  sys_icache_invalidate(base_address, length);
}

#elif defined(IREE_PLATFORM_EMSCRIPTEN)

void iree_memory_flush_icache(void* base_address, iree_host_size_t length) {
  // No-op.
}

#elif defined(IREE_PLATFORM_WINDOWS)

void iree_memory_flush_icache(void* base_address, iree_host_size_t length) {
  FlushInstructionCache(GetCurrentProcess(), base_address, length);
}

#else

// IREE_MEMORY_FLUSH_ICACHE can be defined externally to override this default
// behavior.
#if !defined(IREE_MEMORY_FLUSH_ICACHE)
// __has_builtin was added in GCC 10, so just hard-code the availability
// for < 10, special cased here so it can be dropped once no longer needed.
#if defined __GNUC__ && __GNUC__ < 10
#define IREE_MEMORY_FLUSH_ICACHE(start, end) __builtin___clear_cache(start, end)
#elif defined __has_builtin
#if __has_builtin(__builtin___clear_cache)
#define IREE_MEMORY_FLUSH_ICACHE(start, end) __builtin___clear_cache(start, end)
#endif  // __builtin___clear_cache
#endif  // __has_builtin
#endif  // !defined(IREE_MEMORY_FLUSH_ICACHE)

#if !defined(IREE_MEMORY_FLUSH_ICACHE)
#error "no instruction cache clear implementation"
#endif  // !defined(IREE_MEMORY_FLUSH_ICACHE)

void iree_memory_flush_icache(void* base_address, iree_host_size_t length) {
  IREE_MEMORY_FLUSH_ICACHE(base_address, ((char*)base_address) + length);
}

#endif  // IREE_PLATFORM_*
