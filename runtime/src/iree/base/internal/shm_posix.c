// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/shm.h"

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_APPLE)

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(IREE_PLATFORM_LINUX)
#include <sys/syscall.h>
#endif  // IREE_PLATFORM_LINUX

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/memory.h"

// Linux fcntl seal constants from linux/fcntl.h. Not always available from
// userspace libc headers, so we define them directly.
#if defined(IREE_PLATFORM_LINUX)
#define IREE_F_ADD_SEALS 1033
#define IREE_F_GET_SEALS 1034
#define IREE_F_SEAL_SEAL 0x0001
#define IREE_F_SEAL_SHRINK 0x0002
#define IREE_F_SEAL_GROW 0x0004
#define IREE_F_SEAL_WRITE 0x0008
#endif  // IREE_PLATFORM_LINUX

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Converts a POSIX fd to an iree_shm_handle_t.
static inline iree_shm_handle_t iree_shm_handle_from_fd(int fd) {
  iree_shm_handle_t handle;
  handle.value = (uint64_t)(uintptr_t)fd;
  return handle;
}

// Converts an iree_shm_handle_t back to a POSIX fd.
static inline int iree_shm_handle_to_fd(iree_shm_handle_t handle) {
  return (int)(intptr_t)handle.value;
}

// Maps an fd into the process address space. On Linux, if the fd has
// F_SEAL_WRITE set, the mapping is created read-only (the kernel rejects
// writable shared mappings on write-sealed memfds).
static iree_status_t iree_shm_map_fd(int fd, iree_host_size_t size,
                                     void** out_base) {
  int prot = PROT_READ | PROT_WRITE;
#if defined(IREE_PLATFORM_LINUX)
  int seals = fcntl(fd, IREE_F_GET_SEALS);
  if (seals != -1 && (seals & IREE_F_SEAL_WRITE)) {
    prot = PROT_READ;
  }
#endif  // IREE_PLATFORM_LINUX
  void* base = mmap(NULL, size, prot, MAP_SHARED, fd, 0);
  if (IREE_UNLIKELY(base == MAP_FAILED)) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "mmap failed for shared memory region of %" PRIhsz
                            " bytes (%d)",
                            size, errno);
  }
  *out_base = base;
  return iree_ok_status();
}

// Verifies the fd's backing store is at least |size| bytes. Without this,
// mmap silently succeeds but accessing memory beyond the file end causes
// SIGBUS. Windows MapViewOfFile fails with an error in the same scenario,
// so this check gives POSIX the same fail-early behavior.
static iree_status_t iree_shm_validate_size(int fd, iree_host_size_t size) {
  struct stat st;
  if (IREE_UNLIKELY(fstat(fd, &st) == -1)) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "fstat failed on shared memory fd (%d)", errno);
  }
  if (IREE_UNLIKELY((iree_host_size_t)st.st_size < size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "shared memory region is %" PRIhsz
                            " bytes but %" PRIhsz " bytes were requested",
                            (iree_host_size_t)st.st_size, size);
  }
  return iree_ok_status();
}

// Populates an output mapping from a successfully created/opened fd.
// On failure, closes the fd and returns the error.
static iree_status_t iree_shm_finalize_mapping(
    int fd, iree_host_size_t size, iree_shm_mapping_t* out_mapping) {
  iree_status_t status = iree_shm_validate_size(fd, size);
  if (!iree_status_is_ok(status)) {
    close(fd);
    return status;
  }
  void* base = NULL;
  status = iree_shm_map_fd(fd, size, &base);
  if (!iree_status_is_ok(status)) {
    close(fd);
    return status;
  }
  out_mapping->base = base;
  out_mapping->size = size;
  out_mapping->handle = iree_shm_handle_from_fd(fd);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_shm_*
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_LINUX)

// Creates anonymous shared memory using memfd_create (Linux 3.17+).
// No filesystem footprint; the fd is the only reference.
static iree_status_t iree_shm_create_anonymous_fd(iree_host_size_t size,
                                                  int* out_fd) {
  // memfd_create is not wrapped by all libc versions, so use syscall directly.
  // MFD_ALLOW_SEALING enables the F_SEAL_* API so callers can later seal the
  // region against writes via iree_shm_seal().
  int fd = (int)syscall(SYS_memfd_create, "iree_shm",
                        /*MFD_CLOEXEC=*/0x0001U |
                            /*MFD_ALLOW_SEALING=*/0x0002U);
  if (IREE_UNLIKELY(fd == -1)) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "memfd_create failed (%d)", errno);
  }
  if (IREE_UNLIKELY(ftruncate(fd, (off_t)size) == -1)) {
    iree_status_t status = iree_make_status(
        iree_status_code_from_errno(errno),
        "ftruncate to %" PRIhsz " bytes failed (%d)", size, errno);
    close(fd);
    return status;
  }

  // Seal the memfd against resizing to prevent a peer from ftruncating it
  // smaller, which would cause SIGBUS when the mapping accesses memory beyond
  // the new size. We intentionally omit F_SEAL_SEAL so that callers can later
  // add F_SEAL_WRITE via iree_shm_seal().
  if (IREE_UNLIKELY(fcntl(fd, IREE_F_ADD_SEALS,
                          IREE_F_SEAL_SHRINK | IREE_F_SEAL_GROW) == -1)) {
    iree_status_t status =
        iree_make_status(iree_status_code_from_errno(errno),
                         "fcntl(F_ADD_SEALS, SHRINK|GROW) failed (%d)", errno);
    close(fd);
    return status;
  }

  *out_fd = fd;
  return iree_ok_status();
}

#elif defined(IREE_PLATFORM_APPLE)

// Creates anonymous shared memory using shm_open with a unique name, then
// immediately unlinks the name. The fd and any mappings remain valid after
// unlinking; only the name is removed from the namespace.
static iree_status_t iree_shm_create_anonymous_fd(iree_host_size_t size,
                                                  int* out_fd) {
  // Generate a unique name. shm_open names must start with '/'.
  // We use the pid and a counter to avoid collisions. If a previous process
  // with the same PID crashed between shm_open and shm_unlink, the name may
  // still exist in the kernel namespace, so we retry with a new counter value.
  static iree_atomic_int32_t counter = IREE_ATOMIC_VAR_INIT(0);
  char name[64];
  int fd = -1;
  for (int attempt = 0; attempt < 8; ++attempt) {
    int32_t sequence =
        iree_atomic_fetch_add(&counter, 1, iree_memory_order_relaxed);
    iree_snprintf(name, sizeof(name), "/iree_shm_%d_%d", (int)getpid(),
                  sequence);
    fd = shm_open(name, O_CREAT | O_RDWR | O_EXCL, 0600);
    if (fd != -1) break;
    if (errno != EEXIST) {
      return iree_make_status(iree_status_code_from_errno(errno),
                              "shm_open(%s) failed (%d)", name, errno);
    }
    // Name collision from a leaked entry — try the next counter value.
  }
  if (IREE_UNLIKELY(fd == -1)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "shm_open failed after 8 attempts due to "
                            "leaked names from crashed processes");
  }

  // Unlink immediately so the name doesn't persist in the namespace.
  // The fd and mappings remain valid.
  shm_unlink(name);

  if (IREE_UNLIKELY(ftruncate(fd, (off_t)size) == -1)) {
    iree_status_t status = iree_make_status(
        iree_status_code_from_errno(errno),
        "ftruncate to %" PRIhsz " bytes failed (%d)", size, errno);
    close(fd);
    return status;
  }
  *out_fd = fd;
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_LINUX / IREE_PLATFORM_APPLE

iree_host_size_t iree_shm_required_size(iree_host_size_t requested_size) {
  iree_host_size_t page_size = iree_memory_query_info().normal_page_size;
  if (requested_size == 0) return page_size;
  return (requested_size + page_size - 1) & ~(page_size - 1);
}

iree_status_t iree_shm_create(iree_shm_options_t options,
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
  int fd = -1;
  iree_status_t status = iree_shm_create_anonymous_fd(size, &fd);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  status = iree_shm_finalize_mapping(fd, size, out_mapping);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_shm_create_named(iree_string_view_t name,
                                    iree_shm_options_t options,
                                    iree_host_size_t minimum_size,
                                    iree_shm_mapping_t* out_mapping) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_mapping, 0, sizeof(*out_mapping));
  out_mapping->handle = IREE_SHM_HANDLE_INVALID;

#if defined(IREE_PLATFORM_ANDROID)
  // Android's bionic libc does not provide shm_open/shm_unlink. Named shared
  // memory is not supported; use anonymous shared memory (iree_shm_create)
  // with handle passing instead.
  (void)name;
  (void)options;
  (void)minimum_size;
  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "named shared memory is not supported on Android (no shm_open)");
#else

  if (IREE_UNLIKELY(minimum_size == 0)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shared memory size must be > 0");
  }

  // shm_open requires a NUL-terminated string. The name must start with '/'.
  char name_buffer[IREE_SHM_MAX_NAME_LENGTH + 1];
  if (IREE_UNLIKELY(name.size == 0 || name.size > IREE_SHM_MAX_NAME_LENGTH)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "shared memory name must be 1-%d characters, got %" PRIhsz,
        IREE_SHM_MAX_NAME_LENGTH, name.size);
  }
  memcpy(name_buffer, name.data, name.size);
  name_buffer[name.size] = '\0';

  iree_host_size_t size = iree_shm_required_size(minimum_size);

  // O_EXCL ensures we create a new region; fails if the name already exists.
  int fd = shm_open(name_buffer, O_CREAT | O_RDWR | O_EXCL, 0600);
  if (IREE_UNLIKELY(fd == -1)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "shm_open(%s) failed (%d)", name_buffer, errno);
  }

  if (IREE_UNLIKELY(ftruncate(fd, (off_t)size) == -1)) {
    iree_status_t status = iree_make_status(
        iree_status_code_from_errno(errno),
        "ftruncate to %" PRIhsz " bytes failed (%d)", size, errno);
    close(fd);
    shm_unlink(name_buffer);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_status_t status = iree_shm_finalize_mapping(fd, size, out_mapping);
  if (!iree_status_is_ok(status)) {
    shm_unlink(name_buffer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;

#endif  // IREE_PLATFORM_ANDROID
}

iree_status_t iree_shm_open_handle(iree_shm_handle_t handle,
                                   iree_shm_options_t options,
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

  int fd = iree_shm_handle_to_fd(handle);

  // Duplicate the fd so the mapping has its own independent handle.
  int mapping_fd = dup(fd);
  if (IREE_UNLIKELY(mapping_fd == -1)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "dup failed for shared memory handle (%d)", errno);
  }

  iree_status_t status =
      iree_shm_finalize_mapping(mapping_fd, size, out_mapping);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_shm_open_named(iree_string_view_t name,
                                  iree_shm_options_t options,
                                  iree_host_size_t size,
                                  iree_shm_mapping_t* out_mapping) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_mapping, 0, sizeof(*out_mapping));
  out_mapping->handle = IREE_SHM_HANDLE_INVALID;

#if defined(IREE_PLATFORM_ANDROID)
  (void)name;
  (void)options;
  (void)size;
  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "named shared memory is not supported on Android (no shm_open)");
#else

  if (IREE_UNLIKELY(size == 0)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shared memory size must be > 0");
  }

  char name_buffer[IREE_SHM_MAX_NAME_LENGTH + 1];
  if (IREE_UNLIKELY(name.size == 0 || name.size > IREE_SHM_MAX_NAME_LENGTH)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "shared memory name must be 1-%d characters, got %" PRIhsz,
        IREE_SHM_MAX_NAME_LENGTH, name.size);
  }
  memcpy(name_buffer, name.data, name.size);
  name_buffer[name.size] = '\0';

  // O_RDWR without O_CREAT: open existing only.
  int fd = shm_open(name_buffer, O_RDWR, 0);
  if (IREE_UNLIKELY(fd == -1)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "shm_open(%s) failed (%d)", name_buffer, errno);
  }

  iree_status_t status = iree_shm_finalize_mapping(fd, size, out_mapping);
  IREE_TRACE_ZONE_END(z0);
  return status;

#endif  // IREE_PLATFORM_ANDROID
}

void iree_shm_close(iree_shm_mapping_t* mapping) {
  if (!mapping || !mapping->base) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  munmap(mapping->base, mapping->size);
  if (iree_shm_handle_is_valid(mapping->handle)) {
    close(iree_shm_handle_to_fd(mapping->handle));
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
  int new_fd = dup(iree_shm_handle_to_fd(source));
  if (IREE_UNLIKELY(new_fd == -1)) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "dup failed (%d)", errno);
  }
  *out_handle = iree_shm_handle_from_fd(new_fd);
  return iree_ok_status();
}

void iree_shm_handle_close(iree_shm_handle_t* handle) {
  if (!handle || !iree_shm_handle_is_valid(*handle)) return;
  close(iree_shm_handle_to_fd(*handle));
  *handle = IREE_SHM_HANDLE_INVALID;
}

#if defined(IREE_PLATFORM_LINUX)

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

  int fd = iree_shm_handle_to_fd(mapping->handle);

  // Verify the fd supports sealing. Only memfd-backed regions have the seal
  // interface; shm_open regions return -1/EINVAL.
  int current_seals = fcntl(fd, IREE_F_GET_SEALS);
  if (IREE_UNLIKELY(current_seals == -1)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "sealing not supported for this shared memory region "
        "(only memfd-backed anonymous regions support seals)");
  }

  // Convert our flags to kernel seal flags.
  int kernel_seals = 0;
  if (flags & IREE_SHM_SEAL_WRITE) kernel_seals |= IREE_F_SEAL_WRITE;
  if (flags & IREE_SHM_SEAL_SHRINK) kernel_seals |= IREE_F_SEAL_SHRINK;
  if (flags & IREE_SHM_SEAL_GROW) kernel_seals |= IREE_F_SEAL_GROW;
  if (flags & IREE_SHM_SEAL_SEAL) kernel_seals |= IREE_F_SEAL_SEAL;

  // F_SEAL_WRITE requires that no shared writable VMAs exist for the file.
  // munmap our mapping first so the kernel accepts the seal, then remap as
  // read-only. We cannot use mprotect here because sanitizer interceptors
  // (ASAN, MSAN) may keep the underlying VMA writable, causing F_ADD_SEALS
  // to fail with EBUSY.
  void* old_base = mapping->base;
  iree_host_size_t old_size = mapping->size;
  if (flags & IREE_SHM_SEAL_WRITE) {
    munmap(old_base, old_size);
    mapping->base = NULL;
  }

  if (IREE_UNLIKELY(fcntl(fd, IREE_F_ADD_SEALS, kernel_seals) == -1)) {
    int saved_errno = errno;
    // Restore the writable mapping. If this fails too, the mapping is lost —
    // tear it down completely so the caller gets a clean error and a zeroed
    // struct rather than a half-valid mapping they can't use.
    if (flags & IREE_SHM_SEAL_WRITE) {
      void* base =
          mmap(NULL, old_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      if (IREE_LIKELY(base != MAP_FAILED)) {
        mapping->base = base;
      } else {
        close(fd);
        memset(mapping, 0, sizeof(*mapping));
        mapping->handle = IREE_SHM_HANDLE_INVALID;
      }
    }
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(saved_errno),
                            "fcntl(F_ADD_SEALS) failed (%d)", saved_errno);
  }

  // Remap as read-only now that the seal is applied. If this fails, the seal
  // is permanent but we have no mapping — tear down cleanly so the caller
  // can reopen the region via the handle if needed.
  if (flags & IREE_SHM_SEAL_WRITE) {
    void* base = mmap(NULL, old_size, PROT_READ, MAP_SHARED, fd, 0);
    if (IREE_UNLIKELY(base == MAP_FAILED)) {
      int saved_errno = errno;
      close(fd);
      memset(mapping, 0, sizeof(*mapping));
      mapping->handle = IREE_SHM_HANDLE_INVALID;
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(iree_status_code_from_errno(saved_errno),
                              "mmap(PROT_READ) after sealing failed (%d)",
                              saved_errno);
    }
    mapping->base = base;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_shm_seal_flags_t iree_shm_query_seals(const iree_shm_mapping_t* mapping) {
  if (!mapping || !mapping->base) return IREE_SHM_SEAL_NONE;
  int fd = iree_shm_handle_to_fd(mapping->handle);
  int kernel_seals = fcntl(fd, IREE_F_GET_SEALS);
  if (kernel_seals == -1) return IREE_SHM_SEAL_NONE;
  iree_shm_seal_flags_t flags = IREE_SHM_SEAL_NONE;
  if (kernel_seals & IREE_F_SEAL_WRITE) flags |= IREE_SHM_SEAL_WRITE;
  if (kernel_seals & IREE_F_SEAL_SHRINK) flags |= IREE_SHM_SEAL_SHRINK;
  if (kernel_seals & IREE_F_SEAL_GROW) flags |= IREE_SHM_SEAL_GROW;
  if (kernel_seals & IREE_F_SEAL_SEAL) flags |= IREE_SHM_SEAL_SEAL;
  return flags;
}

#elif defined(IREE_PLATFORM_APPLE)

iree_status_t iree_shm_seal(iree_shm_mapping_t* mapping,
                            iree_shm_seal_flags_t flags) {
  if (IREE_UNLIKELY(!mapping || !mapping->base)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cannot seal a NULL or unmapped region");
  }
  if (flags == IREE_SHM_SEAL_NONE) return iree_ok_status();
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "memory sealing is not supported on macOS");
}

iree_shm_seal_flags_t iree_shm_query_seals(const iree_shm_mapping_t* mapping) {
  return IREE_SHM_SEAL_NONE;
}

#endif  // IREE_PLATFORM_LINUX / IREE_PLATFORM_APPLE

#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_APPLE
