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

// Tests folding and canonicalization of control flow ops.

// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(canonicalize)' %s | IreeFileCheck %s

// CHECK-LABEL: @cond_br_folds
vm.module @cond_br_folds {
  // CHECK-LABEL: @const_cond_br_true
  vm.func @const_cond_br_true(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK-NEXT: vm.br ^bb1(%arg0 : i32)
    %c1 = vm.const.i32 1 : i32
    vm.cond_br %c1, ^bb1(%arg0 : i32), ^bb2(%arg1 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  ^bb2(%1 : i32):
    vm.return %1 : i32
  }

  // CHECK-LABEL: @const_cond_br_false
  vm.func @const_cond_br_false(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK-NEXT: vm.br ^bb1(%arg1 : i32)
    %zero = vm.const.i32.zero : i32
    vm.cond_br %zero, ^bb1(%arg0 : i32), ^bb2(%arg1 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  ^bb2(%1 : i32):
    vm.return %1 : i32
  }

  // CHECK-LABEL: @same_target_same_args_cond_br
  vm.func @same_target_same_args_cond_br(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK-NEXT: vm.br ^bb1(%arg1 : i32)
    vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb1(%arg1 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  }

  // CHECK-LABEL: @same_target_diff_args_cond_br
  vm.func @same_target_diff_args_cond_br(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // NOTE: args differ, so cannot fold.
    // CHECK-NEXT: vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb1(%arg2 : i32)
    vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb1(%arg2 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  }

  // CHECK-LABEL: @swap_inverted_cond_br
  vm.func @swap_inverted_cond_br(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK-NEXT: vm.cond_br %arg0, ^bb2(%arg1 : i32), ^bb1(%arg0 : i32)
    %inv = vm.not.i32 %arg0 : i32
    vm.cond_br %inv, ^bb1(%arg0 : i32), ^bb2(%arg1 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  ^bb2(%1 : i32):
    vm.return %1 : i32
  }

  // CHECK-LABEL: @erase_unused_pure_call
  vm.func @erase_unused_pure_call(%arg0 : i32) {
    %0 = vm.call @nonvariadic_pure_func(%arg0) : (i32) -> i32
    %1 = vm.call.variadic @variadic_pure_func([%arg0]) : (i32 ...) -> i32
    // CHECK-NEXT: vm.return
    vm.return
  }
  vm.import @nonvariadic_pure_func(%arg0 : i32) -> i32 attributes {nosideeffects}
  vm.import @variadic_pure_func(%arg0 : i32 ...) -> i32 attributes {nosideeffects}

  // CHECK-LABEL: @convert_nonvariadic_to_call
  vm.func @convert_nonvariadic_to_call(%arg0 : i32) -> (i32, i32) {
    // CHECK-NEXT: vm.call @nonvariadic_func(%arg0) : (i32) -> i32
    %0 = vm.call.variadic @nonvariadic_func(%arg0) : (i32) -> i32
    // CHECK-NEXT: vm.call.variadic @variadic_func
    %1 = vm.call.variadic @variadic_func(%arg0, []) : (i32, i32 ...) -> i32
    // CHECK-NEXT: vm.return
    vm.return %0, %1 : i32, i32
  }
  vm.import @nonvariadic_func(%arg0 : i32) -> i32
  vm.import @variadic_func(%arg0 : i32, %arg1 : i32 ...) -> i32
}
