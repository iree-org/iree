// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/builtin_types.h"

iree_status_t iree_vm_buffer_register_types(iree_vm_instance_t* instance);
iree_status_t iree_vm_list_register_types(iree_vm_instance_t* instance);

IREE_API_EXPORT iree_status_t
iree_vm_register_builtin_types(iree_vm_instance_t* instance) {
  IREE_RETURN_IF_ERROR(iree_vm_buffer_register_types(instance));
  IREE_RETURN_IF_ERROR(iree_vm_list_register_types(instance));
  return iree_ok_status();
}
