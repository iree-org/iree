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

// Tests folding and canonicalization of debug ops.

// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(canonicalize)' %s | IreeFileCheck %s

// CHECK-LABEL: @cond_break_folds
vm.module @cond_break_folds {
  // CHECK-LABEL: @const_cond_break_true
  vm.func @const_cond_break_true(%arg0 : i32) -> i32 {
    // CHECK-NEXT: vm.break ^bb1(%arg0 : i32)
    %c1 = vm.const.i32 1 : i32
    vm.cond_break %c1, ^bb1(%arg0 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  }

  // CHECK-LABEL: @const_cond_break_false
  vm.func @const_cond_break_false(%arg0 : i32) -> i32 {
    // CHECK-NEXT: vm.br ^bb1(%arg0 : i32)
    %zero = vm.const.i32.zero : i32
    vm.cond_break %zero, ^bb1(%arg0 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  }
}
