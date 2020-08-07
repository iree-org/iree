// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(test-iree-vm-register-allocation)' %s | IreeFileCheck %s

// CHECK-LABEL: @module
vm.module @module {
  // CHECK-LABEL: @single_block
  vm.func @single_block(%arg0 : i32) -> i32 {
    // CHECK: vm.add.i32
    // CHECK-SAME: block_registers = ["i0"]
    // CHECK-SAME: result_registers = ["i1"]
    %0 = vm.add.i32 %arg0, %arg0 : i32
    // CHECK: vm.sub.i32
    // CHECK-SAME: result_registers = ["i0"]
    %1 = vm.sub.i32 %arg0, %0 : i32
    vm.return %1 : i32
  }

  // CHECK-LABEL: @unused_arg
  vm.func @unused_arg(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: vm.const.i32.zero
    // CHECK-SAME: block_registers = ["i0", "i1"]
    // CHECK-SAME: result_registers = ["i0"]
    %zero = vm.const.i32.zero : i32
    vm.return %zero : i32
  }

  // CHECK-LABEL: @dominating_values
  vm.func @dominating_values(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
    // CHECK: vm.const.i32 5
    // CHECK-SAME: block_registers = ["i0", "i1"]
    // CHECK-SAME: result_registers = ["i2"]
    %c5 = vm.const.i32 5 : i32
    // CHECK: vm.cond_br
    // CHECK-SAME: remap_registers = [
    // CHECK-SAME:   [], ["i1->i3"]
    // CHECK-SAME: ]
    vm.cond_br %arg0, ^bb1(%arg0 : i32), ^bb2(%arg1 : i32)
  ^bb1(%0 : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0"]
    vm.return %0, %arg1 : i32, i32
  ^bb2(%1 : i32):
    // CHECK: vm.add.i32
    // CHECK-SAME: block_registers = ["i3"]
    // CHECK-SAME: result_registers = ["i0"]
    %2 = vm.add.i32 %1, %arg0 : i32
    // CHECK: vm.add.i32
    // CHECK-SAME: result_registers = ["i1"]
    %3 = vm.add.i32 %2, %arg1 : i32
    // CHECK: vm.mul.i32
    // CHECK-SAME: result_registers = ["i0"]
    %4 = vm.mul.i32 %2, %1 : i32
    // CHECK: vm.mul.i32
    // CHECK-SAME: result_registers = ["i0"]
    %5 = vm.mul.i32 %4, %c5 : i32
    vm.return %5, %3 : i32, i32
  }

  // CHECK-LABEL: @branch_args
  vm.func @branch_args(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: vm.br
    // CHECK-SAME: block_registers = ["i0", "i1"]
    // CHECK-SAME: remap_registers = [
    // CHECK-SAME:   []
    // CHECK-SAME: ]
    vm.br ^bb1(%arg0, %arg1 : i32, i32)
  ^bb1(%0 : i32, %1 : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0", "i1"]
    vm.return %0 : i32
  }

  // CHECK-LABEL: @branch_args_cycle
  vm.func @branch_args_cycle(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: vm.br
    // CHECK-SAME: block_registers = ["i0", "i1"]
    // CHECK-SAME: remap_registers = [
    // CHECK-SAME:   ["i0->i2", "i1->i0", "i2->i1"]
    // CHECK-SAME: ]
    vm.br ^bb1(%arg1, %arg0 : i32, i32)
  ^bb1(%0 : i32, %1 : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0", "i1"]
    vm.return %0 : i32
  }

  // CHECK-LABEL: @branch_args_cycle_64
  vm.func @branch_args_cycle_64(%arg0 : i64, %arg1 : i64) -> i64 {
    // CHECK: vm.br
    // CHECK-SAME: block_registers = ["i0+1", "i2+3"]
    // CHECK-SAME: remap_registers = [
    // CHECK-SAME:   ["i0+1->i4+5", "i2+3->i0+1", "i4+5->i2+3"]
    // CHECK-SAME: ]
    vm.br ^bb1(%arg1, %arg0 : i64, i64)
  ^bb1(%0 : i64, %1 : i64):
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0+1", "i2+3"]
    vm.return %0 : i64
  }

  // CHECK-LABEL: @branch_args_swizzled
  vm.func @branch_args_swizzled(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: vm.br
    // CHECK-SAME: block_registers = ["i0", "i1", "i2"]
    // CHECK-SAME: remap_registers = [
    // CHECK-SAME:   ["i2->i3", "i0->i2", "i1->i0", "i3->i1"]
    // CHECK-SAME: ]
    vm.br ^bb1(%arg1, %arg2, %arg0 : i32, i32, i32)
  ^bb1(%0 : i32, %1 : i32, %2 : i32):
    // CHECK: vm.br
    // CHECK-SAME: block_registers = ["i0", "i1", "i2"]
    // CHECK-SAME: remap_registers = [
    // CHECK-SAME:   ["i0->i3", "i2->i0", "i3->i2"]
    // CHECK-SAME: ]
    vm.br ^bb2(%2, %1, %0 : i32, i32, i32)
  ^bb2(%3 : i32, %4 : i32, %5 : i32):
    // CHECK: vm.br
    // CHECK-SAME: block_registers = ["i0", "i1", "i2"]
    // CHECK-SAME: remap_registers = [
    // CHECK-SAME:   ["i0->i2", "i1->i0"]
    // CHECK-SAME: ]
    vm.br ^bb3(%4, %4, %3 : i32, i32, i32)
  ^bb3(%6 : i32, %7 : i32, %8 : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0", "i1", "i2"]
    vm.return %6 : i32
  }

  // CHECK-LABEL: @cond_branch_args
  vm.func @cond_branch_args(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: vm.cond_br
    // CHECK-SAME: block_registers = ["i0", "i1", "i2"]
    // CHECK-SAME: remap_registers = [
    // CHECK-SAME:   ["i1->i0"],
    // CHECK-SAME:   ["i2->i0"]
    // CHECK-SAME: ]
    vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)
  ^bb1(%0 : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0"]
    vm.return %0 : i32
  ^bb2(%1 : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0"]
    vm.return %1 : i32
  }

  // CHECK-LABEL: @cond_branch_args_swizzled
  vm.func @cond_branch_args_swizzled(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: vm.cond_br
    // CHECK-SAME: block_registers = ["i0", "i1", "i2"]
    // CHECK-SAME: remap_registers = [
    // CHECK-SAME:   ["i1->i0", "i2->i1"],
    // CHECK-SAME:   ["i0->i3", "i1->i0", "i3->i1"]
    // CHECK-SAME: ]
    vm.cond_br %arg0, ^bb1(%arg1, %arg2 : i32, i32), ^bb2(%arg1, %arg0 : i32, i32)
  ^bb1(%0 : i32, %1 : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0", "i1"]
    vm.return %0 : i32
  ^bb2(%2 : i32, %3 : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0", "i1"]
    vm.return %3 : i32
  }

  // CHECK-LABEL: @cond_branch_args_swizzled_64
  vm.func @cond_branch_args_swizzled_64(%arg0 : i32, %arg1 : i64, %arg2 : i64) -> i64 {
    // CHECK: vm.cond_br
    // CHECK-SAME: block_registers = ["i0", "i2+3", "i4+5"]
    // CHECK-SAME: remap_registers = [
    // CHECK-SAME:   ["i2+3->i0+1", "i4+5->i2+3"],
    // CHECK-SAME:   ["i2+3->i0+1"]
    // CHECK-SAME: ]
    vm.cond_br %arg0, ^bb1(%arg1, %arg2 : i64, i64), ^bb2(%arg1, %arg1 : i64, i64)
  ^bb1(%0 : i64, %1 : i64):
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0+1", "i2+3"]
    vm.return %0 : i64
  ^bb2(%2 : i64, %3 : i64):
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0+1", "i2+3"]
    vm.return %3 : i64
  }

  // CHECK-LABEL: @loop
  vm.func @loop() -> i32 {
    // CHECK: vm.const.i32
    // CHECK-SAME: block_registers = []
    // CHECK-SAME: result_registers = ["i0"]
    %c1 = vm.const.i32 1 : i32
    // CHECK: vm.const.i32
    // CHECK-SAME: result_registers = ["i1"]
    %c5 = vm.const.i32 5 : i32
    // CHECK: vm.const.i32.zero
    // CHECK-SAME: result_registers = ["i2"]
    %i0 = vm.const.i32.zero : i32
    // CHECK: vm.br
    // CHECK-SAME: remap_registers = [
    // CHECK-SAME:   []
    // CHECK-SAME: ]
    vm.br ^loop(%i0 : i32)
  ^loop(%i : i32):
    // CHECK: vm.add.i32
    // CHECK-SAME: block_registers = ["i2"]
    // CHECK-SAME: result_registers = ["i2"]
    %in = vm.add.i32 %i, %c1 : i32
    // CHECK: vm.cmp.lt.i32.s
    // CHECK-SAME: result_registers = ["i3"]
    %cmp = vm.cmp.lt.i32.s %in, %c5 : i32
    // CHECK: vm.cond_br
    // CHECK-SAME: remap_registers = [
    // CHECK-SAME:   [],
    // CHECK-SAME:   ["i2->i0"]
    // CHECK-SAME: ]
    vm.cond_br %cmp, ^loop(%in : i32), ^loop_exit(%in : i32)
  ^loop_exit(%ie : i32):
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0"]
    vm.return %ie : i32
  }
}
