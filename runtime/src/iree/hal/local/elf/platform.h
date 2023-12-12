// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_ELF_PLATFORM_H_
#define IREE_HAL_LOCAL_ELF_PLATFORM_H_

#include "iree/base/api.h"

// TODO(benvanik): move the rest of this to iree/base/internal/. A lot of this
// code comes from an old partial implementation of memory objects that should
// be finished. When done it will replace the need for all of these platform
// files. Portions have already been moved to memory.h:
#include "iree/base/internal/memory.h"

//===----------------------------------------------------------------------===//
// Virtual address space manipulation
//===----------------------------------------------------------------------===//

// Defines which access operations are allowed on a view of memory.
// Attempts to perform an access not originally allowed when the view was
// defined may result in process termination/exceptions/sadness on platforms
// with real MMUs and are generally not detectable: treat limited access as a
// fail-safe mechanism only.
enum iree_memory_access_bits_t {
  // Pages in the view may be read by the process.
  // Some platforms may not respect this value being unset meaning that reads
  // will still succeed.
  IREE_MEMORY_ACCESS_READ = 1u << 0,
  // Pages in the view may be written by the process.
  // If unset then writes will result in process termination.
  IREE_MEMORY_ACCESS_WRITE = 1u << 1,
  // Pages in the view can be executed as native machine code.
  // Callers must ensure iree_memory_info_t::can_allocate_executable_pages is
  // true prior to requesting executable memory as certain platforms or release
  // environments may not support allocating/using executable pages.
  IREE_MEMORY_ACCESS_EXECUTE = 1u << 2,
};
typedef uint32_t iree_memory_access_t;

// Flags used to control the behavior of allocated memory views.
enum iree_memory_view_flag_bits_t {
  // TODO(benvanik): pull from memory_object.h.
  IREE_MEMORY_VIEW_FLAG_NONE = 0u,

  // Indicates that the memory may be used to execute code.
  // May be used to ask for special privileges (like MAP_JIT on MacOS).
  IREE_MEMORY_VIEW_FLAG_MAY_EXECUTE = 1u << 10,
};
typedef uint32_t iree_memory_view_flags_t;

// Reserves a range of virtual address space in the host process.
// The base alignment will be that of the page granularity as specified
// (normal or large) in |flags| and |total_length| will be adjusted to match.
//
// The resulting range at |out_base_address| will be uncommitted and
// inaccessible on systems with memory protection. Pages within the range must
// first be committed with iree_memory_view_commit_ranges and then may have
// their access permissions changed with iree_memory_view_protect_ranges.
//
// Implemented by VirtualAlloc+MEM_RESERVE/mmap+PROT_NONE.
iree_status_t iree_memory_view_reserve(iree_memory_view_flags_t flags,
                                       iree_host_size_t total_length,
                                       iree_allocator_t host_allocator,
                                       void** out_base_address);

// Releases a range of virtual address
void iree_memory_view_release(void* base_address, iree_host_size_t total_length,
                              iree_allocator_t host_allocator);

// Commits pages overlapping the byte ranges defined by |byte_ranges|.
// Ranges will be adjusted to the page granularity of the view.
//
// Implemented by VirtualAlloc+MEM_COMMIT/mmap+!PROT_NONE.
iree_status_t iree_memory_view_commit_ranges(
    void* base_address, iree_host_size_t range_count,
    const iree_byte_range_t* ranges, iree_memory_access_t initial_access);

// Changes the access protection of view byte ranges defined by |byte_ranges|.
// Ranges will be adjusted to the page granularity of the view.
//
// Implemented by VirtualProtect/mprotect:
//  https://docs.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualprotect
//  https://man7.org/linux/man-pages/man2/mprotect.2.html
iree_status_t iree_memory_view_protect_ranges(void* base_address,
                                              iree_host_size_t range_count,
                                              const iree_byte_range_t* ranges,
                                              iree_memory_access_t new_access);

#endif  // IREE_HAL_LOCAL_ELF_PLATFORM_H_
