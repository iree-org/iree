// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(vm.func(test-iree-vm-live-intervals)))" %s | FileCheck %s

// Tests for live interval construction.
// Each value gets an interval [start, end] representing its lifetime.
// - start: instruction index where value is defined
// - end: instruction index of last use (inclusive)

//===----------------------------------------------------------------------===//
// Single block - basic cases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_single_block
vm.module @module_single_block {

  // CHECK-LABEL: @simple_chain
  // Simple chain of operations - each value used once.
  vm.func @simple_chain(%arg0: i32) -> i32 {
    // CHECK: vm.add.i32
    // CHECK-SAME: op_index = 0
    // CHECK-SAME: result_intervals = ["[0, 1]"]
    %0 = vm.add.i32 %arg0, %arg0 : i32
    // CHECK: vm.mul.i32
    // CHECK-SAME: op_index = 1
    // CHECK-SAME: result_intervals = ["[1, 2]"]
    %1 = vm.mul.i32 %0, %arg0 : i32
    // CHECK: vm.return
    // CHECK-SAME: op_index = 2
    vm.return %1 : i32
  }

  // CHECK-LABEL: @multiple_uses
  // Value used multiple times - interval extends to last use.
  vm.func @multiple_uses(%arg0: i32) -> i32 {
    // CHECK: vm.add.i32
    // CHECK-SAME: result_intervals = ["[0, 2]"]
    %0 = vm.add.i32 %arg0, %arg0 : i32
    // %0 used here...
    // CHECK: vm.mul.i32
    // CHECK-SAME: result_intervals = ["[1, 2]"]
    %1 = vm.mul.i32 %0, %arg0 : i32
    // ...and here again
    // CHECK: vm.add.i32
    // CHECK-SAME: result_intervals = ["[2, 3]"]
    %2 = vm.add.i32 %0, %1 : i32
    vm.return %2 : i32
  }

  // CHECK-LABEL: @unused_value
  // Unused value - interval has start == end (just the definition).
  vm.func @unused_value() -> i32 {
    // Unused - interval is just the definition point.
    // CHECK: vm.const.i32.zero
    // CHECK-SAME: result_intervals = ["[0, 0]"]
    %unused = vm.const.i32.zero
    // CHECK: vm.const.i32
    // CHECK-SAME: result_intervals = ["[1, 2]"]
    %used = vm.const.i32 42
    vm.return %used : i32
  }

  // CHECK-LABEL: @block_arg_interval
  // Block arguments have intervals starting at the block's first op.
  vm.func @block_arg_interval(%arg0: i32, %arg1: i32) -> i32 {
    // Block args should have intervals covering their uses.
    // CHECK: vm.add.i32
    // CHECK-SAME: block_arg_intervals = ["[0, 0]", "[0, 1]"]
    %0 = vm.add.i32 %arg0, %arg1 : i32
    // %arg1 used again here.
    %1 = vm.mul.i32 %0, %arg1 : i32
    vm.return %1 : i32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Ref types
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_refs
vm.module @module_refs {

  vm.import private @use_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @ref_interval
  // Ref types should be marked with "ref" in the interval.
  vm.func @ref_interval(%buf: !vm.buffer) {
    // CHECK: vm.call @use_buffer
    // CHECK-SAME: block_arg_intervals = ["[0, 0] ref"]
    vm.call @use_buffer(%buf) : (!vm.buffer) -> ()
    vm.return
  }

  // CHECK-LABEL: @ref_multiple_uses
  // Ref used multiple times - interval extends to last use.
  vm.func @ref_multiple_uses(%buf: !vm.buffer) {
    // CHECK: vm.call @use_buffer
    // First op - buf interval should extend to last use (op 1).
    // CHECK-SAME: block_arg_intervals = ["[0, 1] ref"]
    vm.call @use_buffer(%buf) : (!vm.buffer) -> ()
    vm.call @use_buffer(%buf) : (!vm.buffer) -> ()
    vm.return
  }

  // CHECK-LABEL: @mixed_types
  // Mix of i32 and ref types.
  vm.func @mixed_types(%val: i32, %buf: !vm.buffer) {
    // CHECK: vm.call @use_buffer
    // Should have both types correctly identified.
    // %val used at return (op 1), %buf used at call (op 0).
    // CHECK-SAME: block_arg_intervals = ["[0, 1]", "[0, 0] ref"]
    vm.call @use_buffer(%buf) : (!vm.buffer) -> ()
    vm.return %val : i32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Control flow - branches
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_control_flow
vm.module @module_control_flow {

  // CHECK-LABEL: @diamond_cfg
  // Diamond CFG - value used in both branches.
  vm.func @diamond_cfg(%cond: i32, %val: i32) -> i32 {
    // %val is used in both branches, so its interval extends across the diamond.
    vm.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    // CHECK: vm.add.i32
    %0 = vm.add.i32 %val, %val : i32
    vm.br ^bb3(%0 : i32)
  ^bb2:
    // CHECK: vm.sub.i32
    %1 = vm.sub.i32 %val, %val : i32
    vm.br ^bb3(%1 : i32)
  ^bb3(%result: i32):
    vm.return %result : i32
  }

  // CHECK-LABEL: @sequential_blocks
  // Sequential blocks - value passed through block args.
  vm.func @sequential_blocks(%arg0: i32) -> i32 {
    // Value flows through multiple blocks via block arguments.
    vm.br ^bb1(%arg0 : i32)
  ^bb1(%val: i32):
    // CHECK: vm.add.i32
    %0 = vm.add.i32 %val, %val : i32
    vm.br ^bb2(%0 : i32)
  ^bb2(%result: i32):
    vm.return %result : i32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Loops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_loops
vm.module @module_loops {

  // CHECK-LABEL: @simple_loop
  // Simple loop - values live across back-edge.
  vm.func @simple_loop(%count: i32) -> i32 {
    %c0 = vm.const.i32.zero
    %c1 = vm.const.i32 1
    vm.br ^loop(%c0, %count : i32, i32)
  ^loop(%acc: i32, %remaining: i32):
    %done = vm.cmp.eq.i32 %remaining, %c0 : i32
    vm.cond_br %done, ^exit(%acc : i32), ^body
  ^body:
    %new_acc = vm.add.i32 %acc, %c1 : i32
    %new_remaining = vm.sub.i32 %remaining, %c1 : i32
    vm.br ^loop(%new_acc, %new_remaining : i32, i32)
  ^exit(%result: i32):
    vm.return %result : i32
  }
}

// -----

//===----------------------------------------------------------------------===//
// 64-bit values
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_64bit
vm.module @module_64bit {

  // CHECK-LABEL: @i64_values
  // 64-bit values should also get intervals.
  vm.func @i64_values(%arg0: i64) -> i64 {
    // CHECK: vm.add.i64
    // CHECK-SAME: result_intervals = ["[0, 1]"]
    %0 = vm.add.i64 %arg0, %arg0 : i64
    vm.return %0 : i64
  }
}
