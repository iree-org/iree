// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PARAMETER_UTIL_H_
#define IREE_TOOLING_PARAMETER_UTIL_H_

#include "iree/base/api.h"
#include "iree/io/parameter_provider.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a parameter provider from the file at |path|, if supported.
iree_status_t iree_tooling_create_parameter_provider_from_file(
    iree_string_view_t scope, iree_string_view_t path,
    iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider);

// Builds an I/O parameters module based on the runtime flags provided.
iree_status_t iree_tooling_create_parameters_module_from_flags(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PARAMETER_UTIL_H_
