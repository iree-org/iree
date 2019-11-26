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

// RUN: iree-translate -split-input-file -iree-vm-ir-to-bytecode-module -iree-vm-bytecode-module-output-format=flatbuffer-text %s | FileCheck %s --enable-var-scope --dump-input=fail

// CHECK: name: "simple_module"
vm.module @simple_module {
  // CHECK: exported_functions:
  // CHECK: local_name: "func"
  vm.export @func

  // CHECK: internal_functions:
  // CHECK: local_name: "func"
  vm.func @func(%arg0 : i32) -> i32 {
    vm.return %arg0 : i32
  }

  // CHECK: function_descriptors:
  // CHECK-NEXT: bytecode_offset: 0
  // CHECK-NEXT: bytecode_length: 3
  // CHECK-NEXT: i32_register_count: 1
  // CHECK-NEXT: ref_register_count: 0
  // CHECK: bytecode_data: [ 67, 1, 0 ]
}
