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

// Tests printing and parsing of assignment ops.

// RUN: iree-opt -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: @select_i32
vm.module @my_module {
  vm.func @select_i32(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: %0 = vm.select.i32 %arg0, %arg1, %arg2 : i32
    %0 = vm.select.i32 %arg0, %arg1, %arg2 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @select_ref
vm.module @my_module {
  vm.func @select_ref(%arg0 : i32,
                      %arg1 : !ireex.opaque_ref,
                      %arg2 : !ireex.opaque_ref) -> !ireex.opaque_ref {
    // CHECK: %ref = vm.select.ref %arg0, %arg1, %arg2 : !ireex.opaque_ref
    %ref = vm.select.ref %arg0, %arg1, %arg2 : !ireex.opaque_ref
    vm.return %ref : !ireex.opaque_ref
  }
}
