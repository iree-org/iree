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

// Tests printing and parsing of structural ops.

// RUN: iree-opt -split-input-file %s | FileCheck %s --enable-var-scope --dump-input=fail

// CHECK-LABEL: @module_empty
vm.module @module_empty {}

// -----

// CHECK-LABEL: @module_attributed attributes {a}
vm.module @module_attributed attributes {a} {
  // CHECK: vm.func @fn()
  vm.func @fn()
}

// -----

// CHECK-LABEL: @module_structure
vm.module @module_structure {
  // CHECK-NEXT: vm.global.i32 @g0 : i32
  vm.global.i32 @g0 : i32
  // CHECK-NEXT: vm.export @fn
  vm.export @fn
  // CHECK-NEXT: vm.func @fn
  vm.func @fn(%arg0 : i32) -> i32 {
    vm.return %arg0 : i32
  }

  // CHECK-LABEL: vm.func @fn_attributed(%arg0: i32) -> i32
  // CHECK: attributes {a}
  vm.func @fn_attributed(%arg0 : i32) -> i32
      attributes {a} {
    vm.return %arg0 : i32
  }
}

// -----

// CHECK-LABEL: @export_funcs
vm.module @export_funcs {
  // CHECK-NEXT: vm.export @fn
  vm.export @fn
  // CHECK-NEXT: vm.export @fn as("fn_alias")
  vm.export @fn as("fn_alias")
  // CHECK-NEXT: vm.func @fn()
  vm.func @fn() {
    vm.return
  }

  // CHECK-LABEL: vm.export @fn as("fn_attributed") attributes {a}
  vm.export @fn as("fn_attributed") attributes {a}
}

// -----

// CHECK-LABEL: @import_funcs
vm.module @import_funcs {
  // CHECK-NEXT: vm.import @my.fn_empty()
  vm.import @my.fn_empty()

  // CHECK-NEXT: vm.import @my.fn(%foo : i32, %bar : i32) -> i32
  vm.import @my.fn(%foo : i32, %bar : i32) -> i32

  // CHECK-NEXT: vm.import @my.fn_attrs(%foo : i32, %bar : i32) -> i32 attributes {a}
  vm.import @my.fn_attrs(%foo : i32, %bar : i32) -> i32 attributes {a}

  // CHECK-NEXT: vm.import @my.fn_varargs(%foo : i32, %bar : tuple<i32, i32>...) -> i32
  vm.import @my.fn_varargs(%foo : i32, %bar : tuple<i32, i32>...) -> i32
}
