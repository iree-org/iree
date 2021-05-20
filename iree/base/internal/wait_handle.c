// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/wait_handle.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// iree_wait_handle_t
//===----------------------------------------------------------------------===//

iree_status_t iree_wait_handle_wrap_primitive(
    iree_wait_primitive_type_t primitive_type,
    iree_wait_primitive_value_t primitive_value,
    iree_wait_handle_t* out_handle) {
  memset(out_handle, 0, sizeof(*out_handle));
  out_handle->type = primitive_type;
  out_handle->value = primitive_value;
  return iree_ok_status();
}

void iree_wait_handle_deinitialize(iree_wait_handle_t* handle) {
  memset(handle, 0, sizeof(*handle));
}
