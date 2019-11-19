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

// RUN: iree-opt -split-input-file -pass-pipeline='iree-convert-std-to-vm' %s | FileCheck %s --dump-input=fail

// -----
// Checks literal specifics of structural transforms (more verbose checks
// than usual since the conversion code is manual).
// CHECK-LABEL: @t001_module_all_options
module @t001_module_all_options {

// CHECK: module @my_module {
module @my_module {
  // CHECK: vm.func @my_fn([[ARG0:%[a-zA-Z0-9]+]]: i32) -> i32
  func @my_fn(%arg0: i32) -> (i32) {
    // CHECK: vm.return [[ARG0]] : i32
    return %arg0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t002_no_args_results
module @t002_no_args_results {

module @my_module {
  // CHECK: vm.func @my_fn() {
  func @my_fn() -> () {
    // CHECK: vm.return
    return
  }
}

}

// -----
// CHECK-LABEL: @t003_unnamed_module
module @t003_unnamed_module {

// CHECK: module @module {
module {
}

}
