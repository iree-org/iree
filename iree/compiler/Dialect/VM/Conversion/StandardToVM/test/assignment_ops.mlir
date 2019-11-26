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

// RUN: iree-opt -split-input-file -pass-pipeline='iree-convert-std-to-vm' %s | FileCheck %s --enable-var-scope --dump-input=fail

// -----
// CHECK-LABEL: @t001_cmp_select
module @t001_cmp_select {

module @my_module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0 : i32, %arg1 : i32) -> (i32) {
    // Note that in std, cmp returns an i1 and this relies on the dialect
    // conversion framework promoting that to i32.
    // CHECK: [[CMP:%[a-zA-Z0-9]+]] = vm.cmp.eq.i32
    %1 = cmpi "eq", %arg0, %arg1 : i32
    // CHECK: vm.select.i32 [[CMP]], [[ARG0]], [[ARG1]] : i32
    %2 = select %1, %arg0, %arg1 : i32
    return %2 : i32
  }
}

}
