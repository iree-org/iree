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

// Tests folding and canonicalization of global ops.

// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(canonicalize)' %s | FileCheck %s --enable-var-scope --dump-input=fail

// CHECK-LABEL: @global_i32_folds
vm.module @global_i32_folds {
  // CHECK: vm.global.i32 @g0 mutable 123 : i32
  vm.global.i32 @g0 mutable init(@g0init) : i32
  vm.func @g0init() -> i32 {
    %c123 = vm.const.i32 123 : i32
    vm.return %c123 : i32
  }
}

// -----

// CHECK-LABEL: @global_ref_folds
vm.module @global_ref_folds {
  // CHECK: vm.global.ref @g0 mutable : !ireex.opaque_ref
  vm.global.ref @g0 mutable init(@g0init) : !ireex.opaque_ref
  vm.func @g0init() -> !ireex.opaque_ref {
    %null = vm.const.ref.zero : !ireex.opaque_ref
    vm.return %null : !ireex.opaque_ref
  }
}

// -----

// CHECK-LABEL: @global_load_i32_folds
vm.module @global_load_i32_folds {
  vm.global.i32 @g0 123 : i32
  // CHECK-LABEL: @inline_const_value
  vm.func @inline_const_value() -> i32 {
    // CHECK-NEXT: %c123 = vm.const.i32 123 : i32
    // CHECK-NEXT: vm.return %c123 : i32
    %g0 = vm.global.load.i32 @g0 : i32
    vm.return %g0 : i32
  }

  vm.global.i32 @g1 mutable 123 : i32
  // CHECK-LABEL: @ignore_nonconst_value
  vm.func @ignore_nonconst_value() -> i32 {
    // NOTE: ensure we don't inline non-constant values.
    // CHECK-NEXT: %g1 = vm.global.load.i32 @g1 : i32
    // CHECK-NEXT: vm.return %g1 : i32
    %g1 = vm.global.load.i32 @g1 : i32
    vm.return %g1 : i32
  }
}

// -----

// CHECK-LABEL: @global_load_ref_folds
vm.module @global_load_ref_folds {
  vm.global.ref @g0 : !ireex.opaque_ref
  // CHECK-LABEL: @inline_const_null
  vm.func @inline_const_null() -> !ireex.opaque_ref {
    // CHECK-NEXT: %null = vm.const.ref.zero : !ireex.opaque_ref
    // CHECK-NEXT: vm.return %null : !ireex.opaque_ref
    %g0 = vm.global.load.ref @g0 : !ireex.opaque_ref
    vm.return %g0 : !ireex.opaque_ref
  }

  vm.global.ref @g1 mutable : !ireex.opaque_ref
  // CHECK-LABEL: @ignore_nonconst_value
  vm.func @ignore_nonconst_value() -> !ireex.opaque_ref {
    // NOTE: ensure we don't inline non-constant values.
    // CHECK-NEXT: %g1 = vm.global.load.ref @g1 : !ireex.opaque_ref
    // CHECK-NEXT: vm.return %g1 : !ireex.opaque_ref
    %g1 = vm.global.load.ref @g1 : !ireex.opaque_ref
    vm.return %g1 : !ireex.opaque_ref
  }
}
