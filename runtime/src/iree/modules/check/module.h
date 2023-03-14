// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_CHECK_MODULE_H_
#define IREE_MODULES_CHECK_MODULE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a native custom module.
iree_status_t iree_check_module_create(iree_vm_instance_t* instance,
                                       iree_allocator_t allocator,
                                       iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_CHECK_MODULE_H_
