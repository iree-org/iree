// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PARAMETER_UTIL_H_
#define IREE_TOOLING_PARAMETER_UTIL_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_io_parameter_index_t iree_io_parameter_index_t;
typedef struct iree_io_scope_map_t iree_io_scope_map_t;

// Populates |scope_map| with parameter indices as specified by flags.
iree_status_t iree_tooling_build_parameter_indices_from_flags(
    iree_io_scope_map_t* scope_map);

typedef struct iree_io_parameter_provider_t iree_io_parameter_provider_t;

// Builds an I/O parameters module based on the runtime flags provided.
// If |additional_provider_count| is non-zero then |additional_providers|
// contains providers that will be added to the module in addition to those
// parsed from --parameters= flags. Additional providers are retained by the
// module and can be released by the caller after this call returns.
iree_status_t iree_tooling_create_parameters_module_from_flags(
    iree_vm_instance_t* instance, iree_host_size_t additional_provider_count,
    iree_io_parameter_provider_t** additional_providers,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PARAMETER_UTIL_H_
