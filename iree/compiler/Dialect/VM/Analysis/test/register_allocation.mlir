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

// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(test-iree-vm-register-allocation)' %s | FileCheck %s --enable-var-scope --dump-input=fail

// CHECK-LABEL: @module
vm.module @module {
  // CHECK-LABEL: @single_block
  vm.func @single_block(%arg0 : i32) -> i32 {
    // CHECK: vm.add.i32
    // CHECK-SAME: block_registers = [0 : i32]
    // CHECK-SAME: result_registers = [1 : i32]
    %0 = vm.add.i32 %arg0, %arg0 : i32
    // CHECK: vm.sub.i32
    // CHECK-SAME: result_registers = [0 : i32]
    %1 = vm.sub.i32 %arg0, %0 : i32
    vm.return %1 : i32
  }

  // CHECK-LABEL: @unused_arg
  vm.func @unused_arg(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: vm.const.i32.zero
    // CHECK-SAME: block_registers = [0 : i32, 0 : i32]
    // CHECK-SAME: result_registers = [0 : i32]
    %zero = vm.const.i32.zero : i32
    vm.return %zero : i32
  }

  // CHECK-LABEL: @dominating_values
  vm.func @dominating_values(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: vm.const.i32.zero
    // CHECK-SAME: block_registers = [0 : i32, 1 : i32]
    // CHECK-SAME: result_registers = [2 : i32]
    %c4 = vm.const.i32.zero : i32
    vm.br ^bb1(%arg0 : i32)
  ^bb1(%0 : i32):
    // CHECK: vm.add.i32
    // CHECK-SAME: block_registers = [0 : i32]
    // CHECK-SAME: result_registers = [0 : i32]
    %1 = vm.add.i32 %0, %arg1 : i32
    // CHECK: vm.mul.i32
    // CHECK-SAME: result_registers = [0 : i32]
    %2 = vm.mul.i32 %1, %c4 : i32
    vm.return %2 : i32
  }

  // CHECK-LABEL: @branch_args
  vm.func @branch_args(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: vm.br
    // CHECK-SAME: block_registers = [0 : i32, 1 : i32]
    vm.br ^bb1(%arg0, %arg1 : i32, i32)
  ^bb1(%0 : i32, %1 : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_registers = [0 : i32, 1 : i32]
    vm.return %0 : i32
  }

  // CHECK-LABEL: @cond_branch_args
  vm.func @cond_branch_args(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: vm.cond_br
    // CHECK-SAME: block_registers = [0 : i32, 1 : i32, 2 : i32]
    vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)
  ^bb1(%0 : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_registers = [0 : i32]
    vm.return %0 : i32
  ^bb2(%1 : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_registers = [0 : i32]
    vm.return %1 : i32
  }

  // CHECK-LABEL: @loop
  vm.func @loop() -> i32 {
    // CHECK: vm.const.i32
    // CHECK-SAME: block_registers = []
    // CHECK-SAME: result_registers = [0 : i32]
    %c1 = vm.const.i32 1 : i32
    // CHECK: vm.const.i32
    // CHECK-SAME: result_registers = [1 : i32]
    %c5 = vm.const.i32 5 : i32
    // CHECK: vm.const.i32.zero
    // CHECK-SAME: result_registers = [2 : i32]
    %i0 = vm.const.i32.zero : i32
    vm.br ^loop(%i0 : i32)
  ^loop(%i : i32):
    // CHECK: vm.add.i32
    // CHECK-SAME: block_registers = [2 : i32]
    // CHECK-SAME: result_registers = [2 : i32]
    %in = vm.add.i32 %i, %c1 : i32
    // CHECK: vm.cmp.lt.i32.s
    // CHECK-SAME: result_registers = [3 : i32]
    %cmp = vm.cmp.lt.i32.s %in, %c5 : i32
    vm.cond_br %cmp, ^loop(%in : i32), ^loop_exit(%in : i32)
  ^loop_exit(%ie : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_registers = [2 : i32]
    vm.return %ie : i32
  }
}
