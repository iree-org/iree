// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_DYNAMIC_API_H_
#define IREE_VM_DYNAMIC_API_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"

//===----------------------------------------------------------------------===//
//                                                                            //
//    ██╗░░░██╗███╗░░██╗░██████╗████████╗░█████╗░██████╗░██╗░░░░░███████╗     //
//    ██║░░░██║████╗░██║██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██║░░░░░██╔════╝     //
//    ██║░░░██║██╔██╗██║╚█████╗░░░░██║░░░███████║██████╦╝██║░░░░░█████╗░░     //
//    ██║░░░██║██║╚████║░╚═══██╗░░░██║░░░██╔══██║██╔══██╗██║░░░░░██╔══╝░░     //
//    ╚██████╔╝██║░╚███║██████╔╝░░░██║░░░██║░░██║██████╦╝███████╗███████╗     //
//    ░╚═════╝░╚═╝░░╚══╝╚═════╝░░░░╚═╝░░░╚═╝░░╚═╝╚═════╝░╚══════╝╚══════╝     //
//                                                                            //
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Version code indicating the minimum required runtime structures.
// Runtimes cannot load dynamic modules with different versions.
//
// NOTE: until we hit v1 the versioning scheme here is not set in stone.
// We may want to make this major release number, date codes (0x20220307),
// or some semantic versioning we track in whatever spec we end up having.
typedef uint32_t iree_vm_dynamic_module_version_t;

#define IREE_VM_DYNAMIC_MODULE_VERSION_0_1 0x00000001u

// The latest version of the dynamic module API.
#define IREE_VM_DYNAMIC_MODULE_VERSION_LATEST IREE_VM_DYNAMIC_MODULE_VERSION_0_1

// Exported function from dynamic libraries for creating dynamic modules.
// This should be implemented as pure as possible and may be called many times
// while in process. |allocator| must be used for all allocations.
//
// The provided |max_version| is the maximum version the caller supports;
// callees must return NULL for |out_module| if their lowest available version
// is greater than the max version supported by the caller.
typedef iree_status_t(IREE_API_PTR* iree_vm_dynamic_module_create_fn_t)(
    iree_vm_dynamic_module_version_t max_version, iree_vm_instance_t* instance,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t allocator, iree_vm_module_t** out_module);

// Function name exported from dynamic libraries (pass to dlsym) by default.
#define IREE_VM_DYNAMIC_MODULE_EXPORT_NAME "iree_vm_dynamic_module_create"

// Decorate exported functions with this.
#if defined(_WIN32) || defined(__CYGWIN__)
#define IREE_VM_DYNAMIC_MODULE_EXPORT __declspec(dllexport)
#else
#define IREE_VM_DYNAMIC_MODULE_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_DYNAMIC_API_H_
