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

#include "iree/vm/api.h"

// This would be generated together with the functions in the header
#include "iree/samples/emitc_modules/add_module.module"

static iree_status_t add_module_create(iree_allocator_t allocator,
                                       iree_vm_module_t** out_module) {
  // NOTE: this module has neither shared or per-context module state.
  iree_vm_module_t interface;
  IREE_RETURN_IF_ERROR(iree_vm_module_initialize(&interface, NULL));
  return iree_vm_native_module_create(&interface, &add_module_descriptor_,
                                      allocator, out_module);
}
