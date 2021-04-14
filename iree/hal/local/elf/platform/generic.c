// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

void iree_memory_jit_context_begin() {}

void iree_memory_jit_context_end() {}

//==============================================================================
// Virtual address space manipulation
//==============================================================================

iree_status_t iree_memory_view_reserve(iree_memory_view_flags_t flags,
                                       iree_host_size_t total_length,
                                       void** out_base_address) {
  *out_base_address = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  void* base_address =
      aligned_alloc(IREE_MEMORY_PAGE_SIZE_NORMAL, total_length);
  if (base_address == NULL) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "malloc failed on reservation");
  }

  *out_base_address = base_address;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_memory_view_release(void* base_address,
                              iree_host_size_t total_length) {
  IREE_TRACE_ZONE_BEGIN(z0);

  free(base_address);

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
#if defined __has_builtin
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
  IREE_ELF_CLEAR_CACHE(base_address, base_address + length);
}

#endif  // IREE_PLATFORM_GENERIC
