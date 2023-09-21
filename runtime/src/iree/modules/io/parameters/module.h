// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_IO_PARAMETERS_MODULE_H_
#define IREE_MODULES_IO_PARAMETERS_MODULE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_provider.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a module for accessing parameters via a set of |providers|.
// The providers are retained for the lifetime of the module.
IREE_API_EXPORT iree_status_t iree_io_parameters_module_create(
    iree_vm_instance_t* instance, iree_host_size_t provider_count,
    iree_io_parameter_provider_t* const* providers,
    iree_allocator_t host_allocator,
    iree_vm_module_t** IREE_RESTRICT out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_IO_PARAMETERS_MODULE_H_
