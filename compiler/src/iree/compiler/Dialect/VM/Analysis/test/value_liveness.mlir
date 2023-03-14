// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(vm.func(test-iree-vm-value-liveness)))" %s | FileCheck %s

// CHECK-LABEL: @module
vm.module @module {
  // CHECK-LABEL: @single_block
  vm.func @single_block(%arg0 : i32) -> i32 {
    // CHECK: vm.add.i32
    // CHECK-SAME: block_defined = ["%0", "%1", "%arg0"]
    // CHECK-SAME: block_live = ["%0", "%1", "%arg0"]
    // CHECK-SAME: block_live_in = []
    // CHECK-SAME: block_live_out = []
    // CHECK-SAME: live = ["%0", "%arg0"]
    %0 = vm.add.i32 %arg0, %arg0 : i32
    // CHECK: vm.sub.i32
    // CHECK-SAME: live = ["%0", "%1", "%arg0"]
    %1 = vm.sub.i32 %arg0, %0 : i32
    // CHECK: vm.return
    // CHECK-SAME: live = ["%1"]
    vm.return %1 : i32
  }

  // CHECK-LABEL: @unused_arg
  vm.func @unused_arg(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: vm.const.i32.zero
    // CHECK-SAME: block_defined = ["%arg0", "%arg1", "%zero"]
    // CHECK-SAME: block_live = ["%zero"]
    // CHECK-SAME: block_live_in = []
    // CHECK-SAME: block_live_out = []
    // CHECK-SAME: live = ["%arg0", "%arg1", "%zero"]
    %zero = vm.const.i32.zero
    // CHECK: vm.return
    // CHECK-SAME: live = ["%zero"]
    vm.return %zero : i32
  }

  // CHECK-LABEL: @dominating_values
  vm.func @dominating_values(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: vm.const.i32.zero
    // CHECK-SAME: block_defined = ["%arg0", "%arg1", "%zero"]
    // CHECK-SAME: block_live = ["%arg0"]
    // CHECK-SAME: block_live_in = []
    // CHECK-SAME: block_live_out = ["%arg1", "%zero"]
    // CHECK-SAME: live = ["%arg0", "%arg1", "%zero"]
    %c4 = vm.const.i32.zero
    // CHECK: vm.br
    // CHECK-SAME: live = ["%arg0", "%arg1", "%zero"]
    vm.br ^bb1(%arg0 : i32)
  ^bb1(%0 : i32):
    // CHECK: vm.add.i32
    // CHECK-SAME: block_defined = ["%1", "%2", "%bb1_arg0"]
    // CHECK-SAME: block_live = ["%1", "%2", "%arg1", "%bb1_arg0", "%zero"]
    // CHECK-SAME: block_live_in = ["%arg1", "%zero"]
    // CHECK-SAME: block_live_out = []
    // CHECK-SAME: live = ["%1", "%arg1", "%bb1_arg0", "%zero"]
    %1 = vm.add.i32 %0, %arg1 : i32
    // CHECK: vm.mul.i32
    // CHECK-SAME: live = ["%1", "%2", "%zero"]
    %2 = vm.mul.i32 %1, %c4 : i32
    // CHECK: vm.return
    // CHECK-SAME: live = ["%2"]
    vm.return %2 : i32
  }

  // CHECK-LABEL: @branch_args
  vm.func @branch_args(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: vm.br
    // CHECK-SAME: block_defined = ["%arg0", "%arg1"]
    // CHECK-SAME: block_live = ["%arg0", "%arg1"]
    // CHECK-SAME: block_live_in = []
    // CHECK-SAME: block_live_out = []
    // CHECK-SAME: live = ["%arg0", "%arg1"]
    vm.br ^bb1(%arg0, %arg1 : i32, i32)
  ^bb1(%0 : i32, %1 : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_defined = ["%bb1_arg0", "%bb1_arg1"]
    // CHECK-SAME: block_live = ["%bb1_arg0"]
    // CHECK-SAME: block_live_in = []
    // CHECK-SAME: block_live_out = []
    // CHECK-SAME: live = ["%bb1_arg0", "%bb1_arg1"]
    vm.return %0 : i32
  }

  // CHECK-LABEL: @cond_branch_args
  vm.func @cond_branch_args(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: vm.cond_br
    // CHECK-SAME: block_defined = ["%arg0", "%arg1", "%arg2"]
    // CHECK-SAME: block_live = ["%arg0", "%arg1", "%arg2"]
    // CHECK-SAME: block_live_in = []
    // CHECK-SAME: block_live_out = []
    // CHECK-SAME: live = ["%arg0", "%arg1", "%arg2"]
    vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)
  ^bb1(%0 : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_defined = ["%bb1_arg0"]
    // CHECK-SAME: block_live = ["%bb1_arg0"]
    // CHECK-SAME: block_live_in = []
    // CHECK-SAME: block_live_out = []
    // CHECK-SAME: live = ["%bb1_arg0"]
    vm.return %0 : i32
  ^bb2(%1 : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_defined = ["%bb2_arg0"]
    // CHECK-SAME: block_live = ["%bb2_arg0"]
    // CHECK-SAME: block_live_in = []
    // CHECK-SAME: block_live_out = []
    // CHECK-SAME: live = ["%bb2_arg0"]
    vm.return %1 : i32
  }

  // CHECK-LABEL: @loop
  vm.func @loop() -> i32 {
    // CHECK: vm.const.i32
    // CHECK-SAME: block_defined = ["%c1", "%c5", "%zero"]
    // CHECK-SAME: block_live = ["%zero"]
    // CHECK-SAME: block_live_in = []
    // CHECK-SAME: block_live_out = ["%c1", "%c5"]
    // CHECK-SAME: live = ["%c1"]
    %c1 = vm.const.i32 1
    // CHECK: vm.const.i32
    // CHECK-SAME: live = ["%c1", "%c5"]
    %c5 = vm.const.i32 5
    // CHECK: vm.const.i32.zero
    // CHECK-SAME: live = ["%c1", "%c5", "%zero"]
    %i0 = vm.const.i32.zero
    // CHECK: vm.br
    // CHECK-SAME: live = ["%c1", "%c5", "%zero"]
    vm.br ^loop(%i0 : i32)
  ^loop(%i : i32):
    // CHECK: vm.add.i32
    // CHECK-SAME: block_defined = ["%1", "%bb1_arg0", "%slt"]
    // CHECK-SAME: block_live = ["%1", "%bb1_arg0", "%c1", "%c5", "%slt"]
    // CHECK-SAME: block_live_in = ["%c1", "%c5"]
    // CHECK-SAME: block_live_out = ["%c1", "%c5"]
    // CHECK-SAME: live = ["%1", "%bb1_arg0", "%c1", "%c5"]
    %in = vm.add.i32 %i, %c1 : i32
    // CHECK: vm.cmp.lt.i32.s
    // CHECK-SAME: live = ["%1", "%c1", "%c5", "%slt"]
    %cmp = vm.cmp.lt.i32.s %in, %c5 : i32
    // CHECK: vm.cond_br
    // CHECK-SAME: live = ["%1", "%c1", "%c5", "%slt"]
    vm.cond_br %cmp, ^loop(%in : i32), ^loop_exit(%in : i32)
  ^loop_exit(%ie : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_defined = ["%bb2_arg0"]
    // CHECK-SAME: block_live = ["%bb2_arg0"]
    // CHECK-SAME: block_live_in = []
    // CHECK-SAME: block_live_out = []
    // CHECK-SAME: live = ["%bb2_arg0"]
    vm.return %ie : i32
  }
}
