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
// CHECK-LABEL: @t001_const.i32.nonzero
module @t001_const.i32.nonzero {

module {
  func @non_zero() -> (i32) {
    // CHECK: vm.const.i32 1 : i32
    %1 = constant 1 : i32
    return %1 : i32
  }
}

}

// -----
// CHECK-LABEL: @t001_const.i32.zero
module @t001_const.i32.zero {

module {
  func @zero() -> (i32) {
    // CHECK: vm.const.i32.zero : i32
    %1 = constant 0 : i32
    return %1 : i32
  }
}

}
