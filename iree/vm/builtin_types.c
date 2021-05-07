// Copyright 2019 Google LLC
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

#include "iree/vm/builtin_types.h"

iree_status_t iree_vm_buffer_register_types();
iree_status_t iree_vm_list_register_types();

IREE_API_EXPORT iree_status_t iree_vm_register_builtin_types() {
  IREE_RETURN_IF_ERROR(iree_vm_buffer_register_types());
  IREE_RETURN_IF_ERROR(iree_vm_list_register_types());
  return iree_ok_status();
}
