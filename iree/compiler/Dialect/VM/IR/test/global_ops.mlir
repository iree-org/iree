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

// Tests printing and parsing of global ops.

// RUN: iree-opt -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: @global_load_i32
vm.module @my_module {
  vm.global.i32 @g0 : i32
  vm.func @global_load_i32() -> i32 {
    // CHECK: %g0 = vm.global.load.i32 @g0 : i32
    %g0 = vm.global.load.i32 @g0 : i32
    vm.return %g0 : i32
  }
}

// -----

// CHECK-LABEL: @global_store_i32
vm.module @my_module {
  vm.global.i32 @g0 mutable : i32
  vm.func @global_store_i32(%arg0 : i32) {
    // CHECK: vm.global.store.i32 @g0, %arg0 : i32
    vm.global.store.i32 @g0, %arg0 : i32
    vm.return
  }
}

// -----

// CHECK-LABEL: @global_load_ref
vm.module @my_module {
  vm.global.ref @g0 : !ireex.opaque_ref
  vm.func @global_load_ref() -> !ireex.opaque_ref {
    // CHECK: %g0 = vm.global.load.ref @g0 : !ireex.opaque_ref
    %g0 = vm.global.load.ref @g0 : !ireex.opaque_ref
    vm.return %g0 : !ireex.opaque_ref
  }
}

// -----

// CHECK-LABEL: @global_store_ref
vm.module @my_module {
  vm.global.ref @g0 mutable : !ireex.opaque_ref
  vm.func @global_store_ref(%arg0 : !ireex.opaque_ref) {
    // CHECK: vm.global.store.ref @g0, %arg0 : !ireex.opaque_ref
    vm.global.store.ref @g0, %arg0 : !ireex.opaque_ref
    vm.return
  }
}

// -----

// CHECK-LABEL: @global_reset_ref
vm.module @my_module {
  vm.global.ref @g0 mutable : !ireex.opaque_ref
  vm.func @global_reset_ref() {
    // CHECK: vm.global.reset.ref @g0
    vm.global.reset.ref @g0
    vm.return
  }
}
