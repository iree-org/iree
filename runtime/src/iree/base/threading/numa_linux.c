// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/numa_impl.h"

// Linux NUMA implementation using raw syscalls (no libnuma dependency).
// Supports mmap + mbind for NUMA-placed allocation, explicit huge pages
// (MAP_HUGETLB), and transparent huge pages (MADV_HUGEPAGE).

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)

#include <dirent.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

//===----------------------------------------------------------------------===//
// Syscall definitions (no libnuma dependency)
//===----------------------------------------------------------------------===//

// mbind memory policy constants.
#define IREE_MPOL_DEFAULT 0
#define IREE_MPOL_PREFERRED 1
#define IREE_MPOL_BIND 2

// mbind flags.
#define IREE_MPOL_MF_STRICT (1 << 0)
#define IREE_MPOL_MF_MOVE (1 << 1)

// MAP_HUGETLB and MAP_HUGE_* constants (may not be defined on older kernels).
#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif
#define IREE_MAP_HUGE_SHIFT 26
#define IREE_MAP_HUGE_2MB (21 << IREE_MAP_HUGE_SHIFT)
#define IREE_MAP_HUGE_1GB (30 << IREE_MAP_HUGE_SHIFT)

// Syscall numbers vary by architecture. We define them if not already provided
// by the system headers (which is rare — usually only <asm/unistd.h> has them
// and we want to avoid that dependency).
#if defined(__x86_64__)
#ifndef __NR_mbind
#define __NR_mbind 237
#endif
#ifndef __NR_get_mempolicy
#define __NR_get_mempolicy 239
#endif
#ifndef __NR_getcpu
#define __NR_getcpu 309
#endif
#elif defined(__aarch64__)
#ifndef __NR_mbind
#define __NR_mbind 235
#endif
#ifndef __NR_get_mempolicy
#define __NR_get_mempolicy 237
#endif
#ifndef __NR_getcpu
#define __NR_getcpu 168
#endif
#elif defined(__riscv)
#ifndef __NR_mbind
#define __NR_mbind 235
#endif
#ifndef __NR_get_mempolicy
#define __NR_get_mempolicy 237
#endif
#ifndef __NR_getcpu
#define __NR_getcpu 168
#endif
#endif

//===----------------------------------------------------------------------===//
// NUMA topology queries
//===----------------------------------------------------------------------===//

// Shared initialization state.
iree_once_flag iree_numa_init_flag = IREE_ONCE_FLAG_INIT;
uint64_t iree_numa_online_nodes_storage[IREE_NUMA_NODE_MASK_WORDS] = {0};
iree_host_size_t iree_numa_online_node_count = 0;

void iree_numa_initialize(void) {
  iree_bitmap_t bitmap = iree_numa_online_nodes_bitmap();

  // Scan /sys/devices/system/node/node[0-9]+ directories.
  DIR* dir = opendir("/sys/devices/system/node");
  if (!dir) {
    // Fallback: assume single node 0.
    iree_bitmap_set(bitmap, 0);
    iree_numa_online_node_count = 1;
    return;
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != NULL) {
    // Match "node" prefix followed by a digit.
    if (strncmp(entry->d_name, "node", 4) == 0 && entry->d_name[4] >= '0' &&
        entry->d_name[4] <= '9') {
      // Parse node number from "nodeN" name.
      iree_host_size_t node_id =
          (iree_host_size_t)strtoul(entry->d_name + 4, NULL, 10);
      if (node_id < IREE_NUMA_MAX_NODES) {
        iree_bitmap_set(bitmap, node_id);
      }
    }
  }
  closedir(dir);

  iree_numa_online_node_count = iree_bitmap_count(bitmap);
  if (iree_numa_online_node_count == 0) {
    // Safety fallback: ensure at least node 0 is present.
    iree_bitmap_set(bitmap, 0);
    iree_numa_online_node_count = 1;
  }
}

IREE_API_EXPORT iree_host_size_t iree_numa_node_count(void) {
  iree_call_once(&iree_numa_init_flag, iree_numa_initialize);
  return iree_numa_online_node_count;
}

IREE_API_EXPORT iree_bitmap_t iree_numa_online_nodes(void) {
  iree_call_once(&iree_numa_init_flag, iree_numa_initialize);
  return iree_numa_online_nodes_bitmap();
}

IREE_API_EXPORT iree_numa_node_id_t iree_numa_node_for_current_thread(void) {
#if defined(__NR_getcpu)
  unsigned cpu = 0;
  unsigned node = 0;
  long result = syscall(__NR_getcpu, &cpu, &node, NULL);
  if (result == 0) {
    return (iree_numa_node_id_t)node;
  }
#endif
  return 0;
}

//===----------------------------------------------------------------------===//
// Huge page helpers
//===----------------------------------------------------------------------===//

// Returns the MAP_HUGE_* flag for the given huge page size.
// Returns 0 if the size doesn't match a known huge page size.
static int iree_numa_huge_page_flag(iree_host_size_t page_size) {
  iree_host_size_t resolved = iree_numa_resolve_huge_page_size(page_size);
  if (resolved == IREE_NUMA_HUGE_PAGE_SIZE_2MB) {
    return IREE_MAP_HUGE_2MB;
  } else if (resolved == IREE_NUMA_HUGE_PAGE_SIZE_1GB) {
    return IREE_MAP_HUGE_1GB;
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// NUMA memory binding
//===----------------------------------------------------------------------===//

// Builds a nodemask bitmask for mbind() with a single node set.
// The kernel expects an array of unsigned long with the appropriate bit set.
// Returns the mask size in bytes.
static iree_host_size_t iree_numa_build_nodemask(
    iree_numa_node_id_t node_id, unsigned long* mask,
    iree_host_size_t mask_capacity_bytes) {
  memset(mask, 0, mask_capacity_bytes);
  iree_host_size_t word_index = node_id / (sizeof(unsigned long) * 8);
  iree_host_size_t bit_index = node_id % (sizeof(unsigned long) * 8);
  iree_host_size_t required_bytes = (word_index + 1) * sizeof(unsigned long);
  if (required_bytes <= mask_capacity_bytes) {
    mask[word_index] = 1UL << bit_index;
  }
  return required_bytes;
}

// Calls mbind() to bind memory to a NUMA node.
// Returns 0 on success, -1 on failure (with errno set).
static int iree_numa_mbind(void* ptr, iree_host_size_t size,
                           iree_numa_node_id_t node_id, int policy,
                           unsigned flags) {
#if defined(__NR_mbind)
  // Maximum 1024 NUMA nodes (128 bytes of mask). Sufficient for any
  // real system — even large HPC machines have fewer than 256 nodes.
  unsigned long nodemask[16];
  iree_host_size_t mask_bytes =
      iree_numa_build_nodemask(node_id, nodemask, sizeof(nodemask));
  if (mask_bytes > sizeof(nodemask)) {
    // Node ID too large for our static mask. This would require a truly
    // enormous machine with >1024 NUMA nodes.
    errno = EINVAL;
    return -1;
  }
  // mbind maxnode is in bits, not bytes.
  unsigned long maxnode = mask_bytes * 8;
  return (int)syscall(__NR_mbind, ptr, size, policy, nodemask, maxnode, flags);
#else
  errno = ENOSYS;
  return -1;
#endif
}

IREE_API_EXPORT iree_status_t iree_numa_bind_memory(
    void* ptr, iree_host_size_t size, iree_numa_node_id_t node_id) {
  if (!ptr || size == 0 || node_id == IREE_NUMA_NODE_ANY) {
    return iree_ok_status();
  }
  // Use MPOL_BIND for strict placement with MPOL_MF_MOVE to migrate any
  // already-faulted pages.
  int result =
      iree_numa_mbind(ptr, size, node_id, IREE_MPOL_BIND, IREE_MPOL_MF_MOVE);
  if (result != 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "mbind() failed for %" PRIhsz
                            " bytes to NUMA node %" PRIu32,
                            size, node_id);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// NUMA-aware allocation
//===----------------------------------------------------------------------===//

// Attempts allocation via mmap with explicit huge pages (MAP_HUGETLB).
// Returns true if successful and populates |out_ptr| and |out_info|.
static bool iree_numa_try_alloc_explicit_huge(
    iree_host_size_t size, const iree_numa_alloc_options_t* options,
    void** out_ptr, iree_numa_alloc_info_t* out_info) {
  iree_host_size_t huge_page_size =
      iree_numa_resolve_huge_page_size(options->huge_page_size);
  int huge_flag = iree_numa_huge_page_flag(huge_page_size);
  if (huge_flag == 0) return false;

  // Round up to huge page alignment (with overflow check).
  iree_host_size_t aligned_size = 0;
  if (!iree_host_size_checked_align(size, huge_page_size, &aligned_size)) {
    return false;
  }

  void* ptr =
      mmap(NULL, aligned_size, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | huge_flag, -1, 0);
  if (ptr == MAP_FAILED) return false;

  *out_ptr = ptr;
  out_info->allocated_size = aligned_size;
  out_info->huge_page_size = huge_page_size;
  out_info->method = IREE_NUMA_ALLOC_METHOD_EXPLICIT_HUGE_PAGES;
  return true;
}

// Attempts allocation via mmap with MADV_HUGEPAGE hint (THP).
// Returns true if successful and populates |out_ptr| and |out_info|.
static bool iree_numa_try_alloc_transparent_huge(
    iree_host_size_t size, void** out_ptr, iree_numa_alloc_info_t* out_info) {
  void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) return false;

#ifdef MADV_HUGEPAGE
  // Best-effort hint; ignore errors.
  madvise(ptr, size, MADV_HUGEPAGE);
#endif

  *out_ptr = ptr;
  out_info->allocated_size = size;
  out_info->huge_page_size = 0;
  out_info->method = IREE_NUMA_ALLOC_METHOD_TRANSPARENT_HUGE_PAGES;
  return true;
}

// Attempts allocation via plain mmap (no huge pages).
// Returns true if successful and populates |out_ptr| and |out_info|.
static bool iree_numa_try_alloc_mmap(iree_host_size_t size, void** out_ptr,
                                     iree_numa_alloc_info_t* out_info) {
  void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) return false;

  *out_ptr = ptr;
  out_info->allocated_size = size;
  out_info->huge_page_size = 0;
  out_info->method = IREE_NUMA_ALLOC_METHOD_MMAP;
  return true;
}

IREE_API_EXPORT iree_status_t
iree_numa_alloc(iree_host_size_t size, const iree_numa_alloc_options_t* options,
                void** out_ptr, iree_numa_alloc_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_ptr);
  IREE_ASSERT_ARGUMENT(out_info);
  *out_ptr = NULL;
  memset(out_info, 0, sizeof(*out_info));

  bool allocated = false;

  // Try explicit huge pages if requested.
  if (!allocated && options->use_explicit_huge_pages) {
    allocated =
        iree_numa_try_alloc_explicit_huge(size, options, out_ptr, out_info);
  }

  // Try transparent huge pages if requested (or as fallback from explicit).
  if (!allocated && (options->hint_transparent_huge_pages ||
                     options->use_explicit_huge_pages)) {
    allocated = iree_numa_try_alloc_transparent_huge(size, out_ptr, out_info);
  }

  // Try plain mmap (for NUMA binding support without huge pages).
  if (!allocated) {
    allocated = iree_numa_try_alloc_mmap(size, out_ptr, out_info);
  }

  // Final fallback: standard allocator (no NUMA binding possible).
  // This is only reached if all mmap attempts fail, which is unlikely on Linux
  // (mmap typically only fails under extreme OOM).
  if (!allocated) {
    iree_host_size_t alignment = options->alignment;
    if (alignment == 0) {
      alignment = iree_memory_query_info().normal_page_size;
    }
    iree_host_size_t aligned_size = 0;
    if (!iree_host_size_checked_align(size, alignment, &aligned_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "size %" PRIhsz " alignment to %" PRIhsz
                              " overflows",
                              size, alignment);
    }
    void* ptr = NULL;
    iree_status_t status = iree_aligned_alloc(alignment, aligned_size, &ptr);
    if (!iree_status_is_ok(status)) return status;
    *out_ptr = ptr;
    out_info->allocated_size = aligned_size;
    out_info->huge_page_size = 0;
    out_info->method = IREE_NUMA_ALLOC_METHOD_STANDARD;
    allocated = true;
  }

  // Apply NUMA binding if requested and we used mmap (standard allocator
  // memory cannot be reliably bound).
  if (options->node_id != IREE_NUMA_NODE_ANY &&
      out_info->method != IREE_NUMA_ALLOC_METHOD_STANDARD) {
    // Use MPOL_PREFERRED: the kernel places pages on the requested node when
    // possible but falls back to other nodes under memory pressure. This is
    // more robust than MPOL_BIND for allocation-time binding where we haven't
    // pre-reserved capacity.
    int result = iree_numa_mbind(*out_ptr, out_info->allocated_size,
                                 options->node_id, IREE_MPOL_PREFERRED, 0);
    if (result != 0) {
      // mbind failure is non-fatal for allocation: the memory is still valid,
      // just not NUMA-placed. Log and continue.
      // On systems without NUMA or with restricted capabilities (containers),
      // mbind may return EPERM or ENOSYS.
      (void)result;
    }
  }

  return iree_ok_status();
}

IREE_API_EXPORT void iree_numa_free(void* ptr,
                                    const iree_numa_alloc_info_t* info) {
  if (!ptr) return;
  IREE_ASSERT_ARGUMENT(info);

  switch (info->method) {
    case IREE_NUMA_ALLOC_METHOD_EXPLICIT_HUGE_PAGES:
    case IREE_NUMA_ALLOC_METHOD_TRANSPARENT_HUGE_PAGES:
    case IREE_NUMA_ALLOC_METHOD_MMAP:
      munmap(ptr, info->allocated_size);
      return;
    case IREE_NUMA_ALLOC_METHOD_STANDARD:
      iree_aligned_free(ptr);
      return;
  }
  // Unreachable if info was populated by iree_numa_alloc().
  IREE_ASSERT(false, "unknown NUMA allocation method %d", (int)info->method);
}

#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID
