// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/shm.h"

#if defined(IREE_PLATFORM_WINDOWS)

// clang-format off
#include <windows.h>
// clang-format on

#include <wchar.h>

#include "iree/base/internal/memory.h"

// Named file mappings use the "Local\" namespace prefix for session-local
// visibility. The prefix is 6 wide chars ("Local\").
#define IREE_SHM_WIN32_PREFIX L"Local\\"
#define IREE_SHM_WIN32_PREFIX_LENGTH 6

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Converts a Windows HANDLE to an iree_shm_handle_t.
static inline iree_shm_handle_t iree_shm_handle_from_win32(HANDLE win_handle) {
  iree_shm_handle_t handle;
  handle.value = (uint64_t)(uintptr_t)win_handle;
  return handle;
}

// Converts an iree_shm_handle_t back to a Windows HANDLE.
static inline HANDLE iree_shm_handle_to_win32(iree_shm_handle_t handle) {
  return (HANDLE)(uintptr_t)handle.value;
}

// Maps a file mapping handle into the process address space.
// |access_flags| is passed to MapViewOfFile (FILE_MAP_ALL_ACCESS, optionally
// with FILE_MAP_LARGE_PAGES). |numa_node| is the preferred NUMA node for
// MapViewOfFileExNuma, or IREE_NUMA_NODE_ANY for default placement.
static iree_status_t iree_shm_map_win32(HANDLE mapping_handle,
                                        iree_host_size_t size,
                                        DWORD access_flags,
                                        iree_numa_node_id_t numa_node,
                                        void** out_base) {
  void* base = NULL;
  if (numa_node != IREE_NUMA_NODE_ANY) {
    base = MapViewOfFileExNuma(mapping_handle, access_flags,
                               /*dwFileOffsetHigh=*/0, /*dwFileOffsetLow=*/0,
                               size, /*lpBaseAddress=*/NULL, (DWORD)numa_node);
  } else {
    base = MapViewOfFile(mapping_handle, access_flags,
                         /*dwFileOffsetHigh=*/0, /*dwFileOffsetLow=*/0, size);
  }
  if (IREE_UNLIKELY(!base)) {
    return iree_make_status(
        iree_status_code_from_win32_error(GetLastError()),
        "MapViewOfFile failed for shared memory region of %" PRIhsz " bytes",
        size);
  }
  *out_base = base;
  return iree_ok_status();
}

// Populates an output mapping from a successfully created/opened handle.
// On failure, closes the handle and returns the error.
static iree_status_t iree_shm_finalize_mapping_win32(
    HANDLE mapping_handle, iree_host_size_t size, DWORD access_flags,
    iree_numa_node_id_t numa_node, iree_shm_mapping_t* out_mapping) {
  void* base = NULL;
  iree_status_t status =
      iree_shm_map_win32(mapping_handle, size, access_flags, numa_node, &base);
  if (!iree_status_is_ok(status)) {
    CloseHandle(mapping_handle);
    return status;
  }
  out_mapping->base = base;
  out_mapping->size = size;
  out_mapping->handle = iree_shm_handle_from_win32(mapping_handle);
  return iree_ok_status();
}

// Convenience wrapper for openers that don't know the creator's flags.
// Tries FILE_MAP_LARGE_PAGES first (required when the section was created with
// SEC_LARGE_PAGES), falling back to plain FILE_MAP_ALL_ACCESS for sections
// created without large pages.
static iree_status_t iree_shm_finalize_mapping_win32_default(
    HANDLE mapping_handle, iree_host_size_t size,
    iree_shm_mapping_t* out_mapping) {
  void* base = NULL;
  iree_status_t status = iree_shm_map_win32(
      mapping_handle, size, FILE_MAP_ALL_ACCESS | FILE_MAP_LARGE_PAGES,
      IREE_NUMA_NODE_ANY, &base);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    status = iree_shm_map_win32(mapping_handle, size, FILE_MAP_ALL_ACCESS,
                                IREE_NUMA_NODE_ANY, &base);
  }
  if (!iree_status_is_ok(status)) {
    CloseHandle(mapping_handle);
    return status;
  }
  out_mapping->base = base;
  out_mapping->size = size;
  out_mapping->handle = iree_shm_handle_from_win32(mapping_handle);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_shm_*
//===----------------------------------------------------------------------===//

iree_host_size_t iree_shm_required_size(iree_host_size_t requested_size) {
  iree_host_size_t page_size = iree_memory_query_info().normal_page_size;
  if (requested_size == 0) return page_size;
  return (requested_size + page_size - 1) & ~(page_size - 1);
}

iree_status_t iree_shm_create(const iree_numa_alloc_options_t* options,
                              iree_host_size_t minimum_size,
                              iree_shm_mapping_t* out_mapping) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_mapping, 0, sizeof(*out_mapping));
  out_mapping->handle = IREE_SHM_HANDLE_INVALID;

  if (IREE_UNLIKELY(minimum_size == 0)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shared memory size must be > 0");
  }

  iree_host_size_t size = iree_shm_required_size(minimum_size);

  // Determine creation flags based on placement options.
  DWORD protect_flags = PAGE_READWRITE;
  DWORD access_flags = FILE_MAP_ALL_ACCESS;
  iree_numa_node_id_t numa_node = IREE_NUMA_NODE_ANY;
  bool use_large_pages = false;

  if (options) {
    numa_node = options->node_id;

    // Try large pages if requested. SEC_LARGE_PAGES requires
    // SeLockMemoryPrivilege and SEC_COMMIT. The size must be aligned to
    // GetLargePageMinimum().
    if (iree_any_bit_set(options->flags,
                         IREE_MEMORY_PLACEMENT_FLAG_EXPLICIT_HUGE_PAGES)) {
      SIZE_T large_page_min = GetLargePageMinimum();
      if (large_page_min > 0) {
        iree_host_size_t aligned_size = 0;
        if (iree_host_size_checked_align(size, (iree_host_size_t)large_page_min,
                                         &aligned_size)) {
          size = aligned_size;
          protect_flags = PAGE_READWRITE | SEC_LARGE_PAGES | SEC_COMMIT;
          access_flags |= FILE_MAP_LARGE_PAGES;
          use_large_pages = true;
        }
      }
    }
  }

  // CreateFileMappingNumaW is used when a NUMA node preference is set.
  // INVALID_HANDLE_VALUE for hFile creates a page-file backed mapping
  // (anonymous shared memory). NULL name makes it unnamed.
  HANDLE mapping_handle = NULL;
  if (numa_node != IREE_NUMA_NODE_ANY) {
    mapping_handle = CreateFileMappingNumaW(
        INVALID_HANDLE_VALUE, /*lpFileMappingAttributes=*/NULL, protect_flags,
        (DWORD)(size >> 32), (DWORD)(size & 0xFFFFFFFF), /*lpName=*/NULL,
        (DWORD)numa_node);
  } else {
    mapping_handle = CreateFileMappingW(
        INVALID_HANDLE_VALUE, /*lpFileMappingAttributes=*/NULL, protect_flags,
        (DWORD)(size >> 32), (DWORD)(size & 0xFFFFFFFF), /*lpName=*/NULL);
  }

  // If large pages failed (no privilege, insufficient large pages), retry
  // without SEC_LARGE_PAGES.
  if (!mapping_handle && use_large_pages) {
    protect_flags = PAGE_READWRITE;
    access_flags = FILE_MAP_ALL_ACCESS;
    size = iree_shm_required_size(minimum_size);  // Un-align from large pages.
    if (numa_node != IREE_NUMA_NODE_ANY) {
      mapping_handle = CreateFileMappingNumaW(
          INVALID_HANDLE_VALUE, /*lpFileMappingAttributes=*/NULL, protect_flags,
          (DWORD)(size >> 32), (DWORD)(size & 0xFFFFFFFF), /*lpName=*/NULL,
          (DWORD)numa_node);
    } else {
      mapping_handle = CreateFileMappingW(
          INVALID_HANDLE_VALUE, /*lpFileMappingAttributes=*/NULL, protect_flags,
          (DWORD)(size >> 32), (DWORD)(size & 0xFFFFFFFF), /*lpName=*/NULL);
    }
  }

  // NUMA fallback: if NUMA-specific creation failed (invalid node, container
  // restrictions, NUMA disabled in VM), retry without NUMA preference. This
  // makes NUMA a best-effort hint, matching the Linux mbind semantics where
  // binding failures are silently ignored.
  if (!mapping_handle && numa_node != IREE_NUMA_NODE_ANY) {
    numa_node = IREE_NUMA_NODE_ANY;
    mapping_handle = CreateFileMappingW(
        INVALID_HANDLE_VALUE, /*lpFileMappingAttributes=*/NULL, protect_flags,
        (DWORD)(size >> 32), (DWORD)(size & 0xFFFFFFFF), /*lpName=*/NULL);
  }

  if (IREE_UNLIKELY(!mapping_handle)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "CreateFileMappingW failed for %" PRIhsz " bytes",
                            size);
  }

  iree_status_t status = iree_shm_finalize_mapping_win32(
      mapping_handle, size, access_flags, numa_node, out_mapping);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_shm_create_named(iree_string_view_t name,
                                    const iree_numa_alloc_options_t* options,
                                    iree_host_size_t minimum_size,
                                    iree_shm_mapping_t* out_mapping) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_mapping, 0, sizeof(*out_mapping));
  out_mapping->handle = IREE_SHM_HANDLE_INVALID;

  if (IREE_UNLIKELY(minimum_size == 0)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shared memory size must be > 0");
  }

  // Build the wide-character name in the Local namespace.
  // Format: "Local\<name>" — the prefix ensures session-local visibility.
  wchar_t
      wide_name[IREE_SHM_WIN32_PREFIX_LENGTH + IREE_SHM_MAX_NAME_LENGTH + 1];
  if (IREE_UNLIKELY(name.size == 0 || name.size > IREE_SHM_MAX_NAME_LENGTH)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shared memory name must be 1-%d characters, "
                            "got %" PRIhsz,
                            IREE_SHM_MAX_NAME_LENGTH, name.size);
  }
  wmemcpy(wide_name, IREE_SHM_WIN32_PREFIX, IREE_SHM_WIN32_PREFIX_LENGTH);
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    wide_name[IREE_SHM_WIN32_PREFIX_LENGTH + i] =
        (wchar_t)(unsigned char)name.data[i];
  }
  wide_name[IREE_SHM_WIN32_PREFIX_LENGTH + name.size] = L'\0';

  iree_host_size_t size = iree_shm_required_size(minimum_size);

  // Determine creation flags based on placement options (same logic as
  // anonymous create — named mappings support both large pages and NUMA).
  DWORD protect_flags = PAGE_READWRITE;
  DWORD access_flags = FILE_MAP_ALL_ACCESS;
  iree_numa_node_id_t numa_node = IREE_NUMA_NODE_ANY;
  bool use_large_pages = false;

  if (options) {
    numa_node = options->node_id;
    if (iree_any_bit_set(options->flags,
                         IREE_MEMORY_PLACEMENT_FLAG_EXPLICIT_HUGE_PAGES)) {
      SIZE_T large_page_min = GetLargePageMinimum();
      if (large_page_min > 0) {
        iree_host_size_t aligned_size = 0;
        if (iree_host_size_checked_align(size, (iree_host_size_t)large_page_min,
                                         &aligned_size)) {
          size = aligned_size;
          protect_flags = PAGE_READWRITE | SEC_LARGE_PAGES | SEC_COMMIT;
          access_flags |= FILE_MAP_LARGE_PAGES;
          use_large_pages = true;
        }
      }
    }
  }

  HANDLE mapping_handle = NULL;
  if (numa_node != IREE_NUMA_NODE_ANY) {
    mapping_handle = CreateFileMappingNumaW(
        INVALID_HANDLE_VALUE, /*lpFileMappingAttributes=*/NULL, protect_flags,
        (DWORD)(size >> 32), (DWORD)(size & 0xFFFFFFFF), wide_name,
        (DWORD)numa_node);
  } else {
    mapping_handle = CreateFileMappingW(
        INVALID_HANDLE_VALUE, /*lpFileMappingAttributes=*/NULL, protect_flags,
        (DWORD)(size >> 32), (DWORD)(size & 0xFFFFFFFF), wide_name);
  }

  // Large page fallback.
  if (!mapping_handle && use_large_pages) {
    protect_flags = PAGE_READWRITE;
    access_flags = FILE_MAP_ALL_ACCESS;
    size = iree_shm_required_size(minimum_size);
    if (numa_node != IREE_NUMA_NODE_ANY) {
      mapping_handle = CreateFileMappingNumaW(
          INVALID_HANDLE_VALUE, /*lpFileMappingAttributes=*/NULL, protect_flags,
          (DWORD)(size >> 32), (DWORD)(size & 0xFFFFFFFF), wide_name,
          (DWORD)numa_node);
    } else {
      mapping_handle = CreateFileMappingW(
          INVALID_HANDLE_VALUE, /*lpFileMappingAttributes=*/NULL, protect_flags,
          (DWORD)(size >> 32), (DWORD)(size & 0xFFFFFFFF), wide_name);
    }
  }

  // NUMA fallback (see iree_shm_create for rationale).
  if (!mapping_handle && numa_node != IREE_NUMA_NODE_ANY) {
    numa_node = IREE_NUMA_NODE_ANY;
    mapping_handle = CreateFileMappingW(
        INVALID_HANDLE_VALUE, /*lpFileMappingAttributes=*/NULL, protect_flags,
        (DWORD)(size >> 32), (DWORD)(size & 0xFFFFFFFF), wide_name);
  }

  if (IREE_UNLIKELY(!mapping_handle)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "CreateFileMappingW(named) failed");
  }

  // Check if the mapping already existed. CreateFileMappingW succeeds with
  // ERROR_ALREADY_EXISTS when the name is already taken; we want exclusive
  // creation semantics matching POSIX O_EXCL.
  if (GetLastError() == ERROR_ALREADY_EXISTS) {
    CloseHandle(mapping_handle);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "shared memory region with this name already "
                            "exists");
  }

  iree_status_t status = iree_shm_finalize_mapping_win32(
      mapping_handle, size, access_flags, numa_node, out_mapping);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_shm_open_handle(iree_shm_handle_t handle,
                                   iree_host_size_t size,
                                   iree_shm_mapping_t* out_mapping) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_mapping, 0, sizeof(*out_mapping));
  out_mapping->handle = IREE_SHM_HANDLE_INVALID;

  if (IREE_UNLIKELY(!iree_shm_handle_is_valid(handle))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid shared memory handle");
  }
  if (IREE_UNLIKELY(size == 0)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shared memory size must be > 0");
  }

  // Duplicate the handle so the mapping owns its own copy.
  HANDLE source_handle = iree_shm_handle_to_win32(handle);
  HANDLE mapping_handle = NULL;
  if (IREE_UNLIKELY(!DuplicateHandle(GetCurrentProcess(), source_handle,
                                     GetCurrentProcess(), &mapping_handle, 0,
                                     FALSE, DUPLICATE_SAME_ACCESS))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "DuplicateHandle failed for shared memory handle");
  }

  iree_status_t status = iree_shm_finalize_mapping_win32_default(
      mapping_handle, size, out_mapping);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_shm_open_named(iree_string_view_t name,
                                  iree_host_size_t size,
                                  iree_shm_mapping_t* out_mapping) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_mapping, 0, sizeof(*out_mapping));
  out_mapping->handle = IREE_SHM_HANDLE_INVALID;

  if (IREE_UNLIKELY(size == 0)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shared memory size must be > 0");
  }

  // Build the wide-character name with "Local\" prefix.
  wchar_t
      wide_name[IREE_SHM_WIN32_PREFIX_LENGTH + IREE_SHM_MAX_NAME_LENGTH + 1];
  if (IREE_UNLIKELY(name.size == 0 || name.size > IREE_SHM_MAX_NAME_LENGTH)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shared memory name must be 1-%d characters, "
                            "got %" PRIhsz,
                            IREE_SHM_MAX_NAME_LENGTH, name.size);
  }
  wmemcpy(wide_name, IREE_SHM_WIN32_PREFIX, IREE_SHM_WIN32_PREFIX_LENGTH);
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    wide_name[IREE_SHM_WIN32_PREFIX_LENGTH + i] =
        (wchar_t)(unsigned char)name.data[i];
  }
  wide_name[IREE_SHM_WIN32_PREFIX_LENGTH + name.size] = L'\0';

  // Try with FILE_MAP_LARGE_PAGES first — required when the section was
  // created with SEC_LARGE_PAGES. Fall back to plain FILE_MAP_ALL_ACCESS
  // for sections created without large pages.
  HANDLE mapping_handle = OpenFileMappingW(
      FILE_MAP_ALL_ACCESS | FILE_MAP_LARGE_PAGES, FALSE, wide_name);
  if (!mapping_handle) {
    mapping_handle = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, wide_name);
  }
  if (IREE_UNLIKELY(!mapping_handle)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "OpenFileMappingW failed");
  }

  iree_status_t status = iree_shm_finalize_mapping_win32_default(
      mapping_handle, size, out_mapping);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_shm_close(iree_shm_mapping_t* mapping) {
  if (!mapping || !mapping->base) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  UnmapViewOfFile(mapping->base);
  if (iree_shm_handle_is_valid(mapping->handle)) {
    CloseHandle(iree_shm_handle_to_win32(mapping->handle));
  }
  memset(mapping, 0, sizeof(*mapping));
  mapping->handle = IREE_SHM_HANDLE_INVALID;
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_shm_handle_dup(iree_shm_handle_t source,
                                  iree_shm_handle_t* out_handle) {
  *out_handle = IREE_SHM_HANDLE_INVALID;
  if (IREE_UNLIKELY(!iree_shm_handle_is_valid(source))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cannot duplicate an invalid handle");
  }
  HANDLE new_handle = NULL;
  if (IREE_UNLIKELY(!DuplicateHandle(
          GetCurrentProcess(), iree_shm_handle_to_win32(source),
          GetCurrentProcess(), &new_handle, 0, FALSE, DUPLICATE_SAME_ACCESS))) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "DuplicateHandle failed");
  }
  *out_handle = iree_shm_handle_from_win32(new_handle);
  return iree_ok_status();
}

void iree_shm_handle_close(iree_shm_handle_t* handle) {
  if (!handle || !iree_shm_handle_is_valid(*handle)) return;
  CloseHandle(iree_shm_handle_to_win32(*handle));
  *handle = IREE_SHM_HANDLE_INVALID;
}

iree_status_t iree_shm_seal(iree_shm_mapping_t* mapping,
                            iree_shm_seal_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (IREE_UNLIKELY(!mapping || !mapping->base)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cannot seal a NULL or unmapped region");
  }
  if (flags == IREE_SHM_SEAL_NONE) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // IREE_SHM_SEAL_WRITE: change the view protection to PAGE_READONLY.
  // IREE_SHM_SEAL_SHRINK/GROW: Windows file mappings are inherently fixed-size.
  // IREE_SHM_SEAL_SEAL: no mechanism to undo VirtualProtect through the IREE
  //   API, so this is a no-op.
  if (flags & IREE_SHM_SEAL_WRITE) {
    DWORD old_protect = 0;
    if (IREE_UNLIKELY(!VirtualProtect(mapping->base, mapping->size,
                                      PAGE_READONLY, &old_protect))) {
      DWORD error = GetLastError();
      IREE_TRACE_ZONE_END(z0);
      // Sections created with SEC_LARGE_PAGES do not support page protection
      // changes. Report as UNAVAILABLE using the same contract as macOS (which
      // doesn't support sealing at all) so callers implementing defense-in-
      // depth can check and proceed.
      if (error == ERROR_INVALID_PARAMETER) {
        return iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "write sealing not supported (likely large-page-backed section)");
      }
      return iree_make_status(iree_status_code_from_win32_error(error),
                              "VirtualProtect(PAGE_READONLY) failed");
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_shm_seal_flags_t iree_shm_query_seals(const iree_shm_mapping_t* mapping) {
  if (!mapping || !mapping->base) return IREE_SHM_SEAL_NONE;
  MEMORY_BASIC_INFORMATION memory_info;
  if (VirtualQuery(mapping->base, &memory_info, sizeof(memory_info)) == 0) {
    return IREE_SHM_SEAL_NONE;
  }
  iree_shm_seal_flags_t flags = IREE_SHM_SEAL_NONE;
  if (memory_info.Protect == PAGE_READONLY) {
    flags |= IREE_SHM_SEAL_WRITE;
  }
  return flags;
}

#endif  // IREE_PLATFORM_WINDOWS
