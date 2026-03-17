// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_NET_CARRIER_RDMA_LIBVERBS_H_
#define IREE_NET_CARRIER_RDMA_LIBVERBS_H_

#include <errno.h>
#include <stdint.h>

#include "iree/base/api.h"

// libibverbs types and constants from vendored rdma-core headers.
#include "third_party/rdma-core-headers/include/infiniband/verbs.h"  // IWYU pragma: export

// Undef macros that conflict with our function pointer names.
// verbs.h defines these as macros that expand to inline wrappers for
// compile-time optimization, but we need the actual symbol names for dynamic
// loading.
#undef ibv_reg_mr
#undef ibv_reg_mr_iova
#undef ibv_query_port

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_dynamic_library_t iree_dynamic_library_t;

//===----------------------------------------------------------------------===//
// Compile-time Configuration
//===----------------------------------------------------------------------===//

// By default we dynamically link against libibverbs. This allows us to produce
// binaries that run on systems without RDMA support available.
//
// Set `-DIREE_NET_LIBVERBS_STATIC=1` to link against the system libibverbs
// package directly.
#if !defined(IREE_NET_LIBVERBS_STATIC)
#define IREE_NET_LIBVERBS_STATIC 0
#endif  // IREE_NET_LIBVERBS_STATIC

//===----------------------------------------------------------------------===//
// iree_net_libverbs_t
//===----------------------------------------------------------------------===//

// Dynamically loaded libibverbs.so (or equivalent).
// Contains function pointers to resolved libibverbs API symbols.
//
// Thread-safe; immutable after initialization.
typedef struct iree_net_libverbs_t {
#if !IREE_NET_LIBVERBS_STATIC
  // Loaded libibverbs dynamic library.
  iree_dynamic_library_t* library;

  // Function pointers resolved from the library.
#define IREE_NET_LIBVERBS_PFN(result_type, symbol, decl, args) \
  result_type (*symbol)(decl);
#define DECL(...) __VA_ARGS__
#define ARGS(...)
#include "iree/net/carrier/rdma/libverbs_tables.h"  // IWYU pragma: export

#endif  // !IREE_NET_LIBVERBS_STATIC

  // Optional: ibv_reg_dmabuf_mr (kernel 5.12+, libibverbs 34+).
  // Check with iree_net_libverbs_has_dmabuf_mr().
  struct ibv_mr* (*ibv_reg_dmabuf_mr)(struct ibv_pd* pd, uint64_t offset,
                                      size_t length, uint64_t iova, int fd,
                                      int access);
} iree_net_libverbs_t;

// Initializes |out_libverbs| in-place with dynamically loaded libibverbs
// symbols. iree_net_libverbs_deinitialize must be used to release library
// resources.
//
// |search_paths| will override the default library search paths and look for
// the canonical library file under each before falling back to the defaults.
// The `IREE_NET_LIBVERBS_PATH` environment variable can also be set and will
// be checked after the explicitly provided search paths.
IREE_API_EXPORT iree_status_t iree_net_libverbs_initialize(
    iree_string_view_list_t search_paths, iree_allocator_t host_allocator,
    iree_net_libverbs_t* out_libverbs);

// Deinitializes |libverbs| by unloading the backing library.
IREE_API_EXPORT void iree_net_libverbs_deinitialize(
    iree_net_libverbs_t* libverbs);

// Returns true if ibv_reg_dmabuf_mr is available (kernel 5.12+).
static inline bool iree_net_libverbs_has_dmabuf_mr(
    const iree_net_libverbs_t* libverbs) {
  return libverbs->ibv_reg_dmabuf_mr != NULL;
}

// Appends the absolute path of the shared library providing the dynamic
// symbols.
IREE_API_EXPORT iree_status_t iree_net_libverbs_append_path_to_builder(
    const iree_net_libverbs_t* libverbs, iree_string_builder_t* builder);

//===----------------------------------------------------------------------===//
// errno to iree_status_t mapping
//===----------------------------------------------------------------------===//

// Maps an errno value to an IREE status code.
static inline iree_status_code_t iree_net_rdma_errno_to_status_code(int err) {
  switch (err) {
    case 0:
      return IREE_STATUS_OK;
    case EINVAL:
      return IREE_STATUS_INVALID_ARGUMENT;
    case ENOMEM:
      return IREE_STATUS_RESOURCE_EXHAUSTED;
    case ENODEV:
    case ENOENT:
      return IREE_STATUS_NOT_FOUND;
    case EACCES:
    case EPERM:
      return IREE_STATUS_PERMISSION_DENIED;
    case EBUSY:
      return IREE_STATUS_UNAVAILABLE;
    case ENOSYS:
    case EOPNOTSUPP:
      return IREE_STATUS_UNIMPLEMENTED;
    case EAGAIN:
      return IREE_STATUS_UNAVAILABLE;
    case ETIMEDOUT:
      return IREE_STATUS_DEADLINE_EXCEEDED;
    case ECONNREFUSED:
    case ECONNRESET:
    case ENOTCONN:
      return IREE_STATUS_UNAVAILABLE;
    case EFAULT:
      return IREE_STATUS_INVALID_ARGUMENT;
    default:
      return IREE_STATUS_UNKNOWN;
  }
}

// Returns an IREE status with the errno message formatted.
static inline iree_status_t iree_status_from_errno(const char* file,
                                                   uint32_t line, int err,
                                                   const char* symbol) {
  if (err == 0) return iree_ok_status();
  return iree_make_status_with_location(
      file, line, iree_net_rdma_errno_to_status_code(err), "[%s] errno %d: %s",
      symbol, err, strerror(err));
}

// Macro to create a status from errno with file/line info.
#define IREE_NET_RETURN_ERRNO_IF_ERROR(err, symbol)                     \
  do {                                                                  \
    int _err = (err);                                                   \
    if (_err != 0) {                                                    \
      return iree_status_from_errno(__FILE__, __LINE__, _err, #symbol); \
    }                                                                   \
  } while (0)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_LIBVERBS_H_
