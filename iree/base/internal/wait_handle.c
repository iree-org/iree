// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/base/internal/wait_handle.h"

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
