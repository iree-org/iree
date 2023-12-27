// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/elf/platform.h"

#if defined(IREE_PLATFORM_WINDOWS)

//==============================================================================
// Virtual address space manipulation
//==============================================================================

// https://docs.microsoft.com/en-us/windows/win32/memory/memory-protection-constants
static DWORD iree_memory_access_to_win32_page_flags(
    iree_memory_access_t access) {
  DWORD protect = 0;
  if (access & IREE_MEMORY_ACCESS_EXECUTE) {
    if (access & IREE_MEMORY_ACCESS_WRITE) {
      protect |= PAGE_EXECUTE_READWRITE;
    } else if (access & IREE_MEMORY_ACCESS_READ) {
      protect |= PAGE_EXECUTE_READ;
    } else {
      protect |= PAGE_EXECUTE;
    }
  } else if (access & IREE_MEMORY_ACCESS_WRITE) {
    protect |= PAGE_READWRITE;
  } else if (access & IREE_MEMORY_ACCESS_READ) {
    protect |= PAGE_READONLY;
  } else {
    protect |= PAGE_NOACCESS;
  }
  return protect;
}

iree_status_t iree_memory_view_reserve(iree_memory_view_flags_t flags,
                                       iree_host_size_t total_length,
                                       iree_allocator_t host_allocator,
                                       void** out_base_address) {
  *out_base_address = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  void* base_address =
      VirtualAlloc(NULL, total_length, MEM_RESERVE, PAGE_NOACCESS);
  if (base_address == NULL) {
    status = iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                              "VirtualAlloc failed to reserve");
  }

  *out_base_address = base_address;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_memory_view_release(void* base_address, iree_host_size_t total_length,
                              iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // NOTE: return value ignored as this is a shutdown path.
  VirtualFree(base_address, 0, MEM_RELEASE);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_memory_view_commit_ranges(
    void* base_address, iree_host_size_t range_count,
    const iree_byte_range_t* ranges, iree_memory_access_t initial_access) {
  IREE_TRACE_ZONE_BEGIN(z0);

  DWORD initial_protect =
      iree_memory_access_to_win32_page_flags(initial_access);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < range_count; ++i) {
    if (!VirtualAlloc((uint8_t*)base_address + ranges[i].offset,
                      ranges[i].length, MEM_COMMIT, initial_protect)) {
      status =
          iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                           "VirtualAlloc failed to commit");
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

  DWORD new_protect = iree_memory_access_to_win32_page_flags(new_access);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < range_count; ++i) {
    uint8_t* range_address = (uint8_t*)base_address + ranges[i].offset;
    DWORD old_protect = 0;
    BOOL ret = VirtualProtect(range_address, ranges[i].length, new_protect,
                              &old_protect);
    if (!ret) {
      status =
          iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                           "VirtualProtect failed");
      break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

#endif  // IREE_PLATFORM_WINDOWS
