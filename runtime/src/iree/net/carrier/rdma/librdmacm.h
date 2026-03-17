// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_NET_CARRIER_RDMA_LIBRDMACM_H_
#define IREE_NET_CARRIER_RDMA_LIBRDMACM_H_

#include <stdint.h>

#include "iree/base/api.h"

// librdmacm types and constants from vendored rdma-core headers.
#include "third_party/rdma-core-headers/include/rdma/rdma_cma.h"  // IWYU pragma: export

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_dynamic_library_t iree_dynamic_library_t;

//===----------------------------------------------------------------------===//
// Compile-time Configuration
//===----------------------------------------------------------------------===//

// By default we dynamically link against librdmacm. This allows us to produce
// binaries that run on systems without RDMA support available.
//
// Set `-DIREE_NET_LIBRDMACM_STATIC=1` to link against the system librdmacm
// package directly.
#if !defined(IREE_NET_LIBRDMACM_STATIC)
#define IREE_NET_LIBRDMACM_STATIC 0
#endif  // IREE_NET_LIBRDMACM_STATIC

//===----------------------------------------------------------------------===//
// iree_net_librdmacm_t
//===----------------------------------------------------------------------===//

// Dynamically loaded librdmacm.so (or equivalent).
// Contains function pointers to resolved librdmacm API symbols.
//
// Thread-safe; immutable after initialization.
typedef struct iree_net_librdmacm_t {
#if !IREE_NET_LIBRDMACM_STATIC
  // Loaded librdmacm dynamic library.
  iree_dynamic_library_t* library;

  // Function pointers resolved from the library.
#define IREE_NET_LIBRDMACM_PFN(result_type, symbol, decl, args) \
  result_type (*symbol)(decl);
#define DECL(...) __VA_ARGS__
#define ARGS(...)
#include "iree/net/carrier/rdma/librdmacm_tables.h"  // IWYU pragma: export

#endif  // !IREE_NET_LIBRDMACM_STATIC
} iree_net_librdmacm_t;

// Initializes |out_librdmacm| in-place with dynamically loaded librdmacm
// symbols. iree_net_librdmacm_deinitialize must be used to release library
// resources.
//
// |search_paths| will override the default library search paths and look for
// the canonical library file under each before falling back to the defaults.
// The `IREE_NET_LIBRDMACM_PATH` environment variable can also be set and will
// be checked after the explicitly provided search paths.
IREE_API_EXPORT iree_status_t iree_net_librdmacm_initialize(
    iree_string_view_list_t search_paths, iree_allocator_t host_allocator,
    iree_net_librdmacm_t* out_librdmacm);

// Deinitializes |librdmacm| by unloading the backing library.
IREE_API_EXPORT void iree_net_librdmacm_deinitialize(
    iree_net_librdmacm_t* librdmacm);

// Appends the absolute path of the shared library providing the dynamic
// symbols.
IREE_API_EXPORT iree_status_t iree_net_librdmacm_append_path_to_builder(
    const iree_net_librdmacm_t* librdmacm, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_LIBRDMACM_H_
