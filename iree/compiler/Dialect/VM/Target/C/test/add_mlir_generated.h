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

// This is the (slightly modified and formatted) result of running
// iree-translate
// --iree-vm-ir-to-c-module on the file add.mlir

// #include "vm_c_funcs.h"
#include "iree/compiler/Dialect/VM/Target/C/vm_c_funcs.h"

iree_status_t test_function(int32_t v1, int32_t v2, int32_t *out0,
                            int32_t *out1) {
  int32_t v3 = vm_add_i32(v1, v2);
  int32_t v4 = vm_add_i32(v3, v3);
  *out0 = v3;
  *out1 = v4;
  return iree_ok_status();
}