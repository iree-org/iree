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

// Tests folding and canonicalization of constant ops.

// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(cse),vm.module(canonicalize)' %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: @const_i32_folds
vm.module @const_i32_folds {
  // CHECK-LABEL: @cse
  vm.func @cse() -> (i32, i32) {
    // CHECK-NEXT: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1, %c1 : i32, i32
    %0 = vm.const.i32 1 : i32
    %1 = vm.const.i32 1 : i32
    vm.return %0, %1 : i32, i32
  }

  // CHECK-LABEL: @cse_zero
  vm.func @cse_zero() -> (i32, i32) {
    // CHECK-NEXT: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero, %zero : i32, i32
    %0 = vm.const.i32 0 : i32
    %1 = vm.const.i32 0 : i32
    vm.return %0, %1 : i32, i32
  }
}

// -----

// CHECK-LABEL: @const_ref_folds
vm.module @const_ref_folds {
  // CHECK-LABEL: @cse_null
  vm.func @cse_null() -> (!ireex.opaque_ref, !ireex.opaque_ref) {
    // CHECK-NEXT: %null = vm.const.ref.zero : !ireex.opaque_ref
    // CHECK-NEXT: vm.return %null, %null : !ireex.opaque_ref, !ireex.opaque_ref
    %0 = vm.const.ref.zero : !ireex.opaque_ref
    %1 = vm.const.ref.zero : !ireex.opaque_ref
    vm.return %0, %1 : !ireex.opaque_ref, !ireex.opaque_ref
  }
}

// -----

// CHECK-LABEL: @const_rodata_folds
vm.module @const_rodata_folds {
  // CHECK-NEXT: vm.rodata @r2
  vm.rodata @r2 dense<[9, 9, 9]> : tensor<3xi32>
  // CHECK-NEXT: @cse_rodata_loads
  vm.func @cse_rodata_loads() -> (!ireex.byte_buffer_ref, !ireex.byte_buffer_ref) {
    // CHECK-NEXT: %r2 = vm.const.ref.rodata @r2 : !ireex.byte_buffer_ref
    // CHECK-NEXT: vm.return %r2, %r2 : !ireex.byte_buffer_ref, !ireex.byte_buffer_ref
    %0 = vm.const.ref.rodata @r2 : !ireex.byte_buffer_ref
    %1 = vm.const.ref.rodata @r2 : !ireex.byte_buffer_ref
    vm.return %0, %1 : !ireex.byte_buffer_ref, !ireex.byte_buffer_ref
  }
}
