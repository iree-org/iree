// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(vm.func(test-iree-vm-register-allocation)))" %s | FileCheck %s

// Tests for integer register allocation patterns.
// These tests verify the linear scan allocator for i32/i64 values including:
// - Entry block argument allocation (ABI requirements)
// - Cross-block register reuse
// - Branch argument remapping
// - Loop patterns
// - Coalescing hints

//===----------------------------------------------------------------------===//
// Entry Block Arguments (ABI Requirement)
//===----------------------------------------------------------------------===//

// Arguments to entry blocks (function parameters) must be allocated
// monotonically - even if an argument is unused, it occupies its register
// to maintain ABI stability.

// CHECK-LABEL: @module_entry_args
vm.module @module_entry_args {

  // CHECK-LABEL: @entry_args_i32_only
  // Basic i32 entry args - allocated i0, i1, i2.
  vm.func @entry_args_i32_only(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0", "i1", "i2"]
    vm.return %arg0 : i32
  }

  // CHECK-LABEL: @entry_args_with_i64
  // i64 args must be aligned to even register boundary.
  // %arg0: i0, %arg1: i2+3 (skips i1 for alignment), %arg2: i4.
  vm.func @entry_args_with_i64(%arg0: i32, %arg1: i64, %arg2: i32) -> i32 {
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0", "i2+3", "i4"]
    vm.return %arg0 : i32
  }

  // CHECK-LABEL: @entry_args_unused
  // Unused args still occupy their registers.
  vm.func @entry_args_unused(%arg0: i32, %unused: i32, %arg2: i32) -> i32 {
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0", "i1", "i2"]
    vm.return %arg2 : i32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Single Block Sequential Allocation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_single_block
vm.module @module_single_block {

  // CHECK-LABEL: @sequential_allocation
  // %0 is allocated i1, then %1 reuses i1 since %0 dies at the sub.
  vm.func @sequential_allocation(%arg0: i32) -> i32 {
    // CHECK: vm.add.i32
    // CHECK-SAME: block_registers = ["i0"]
    // CHECK-SAME: result_registers = ["i1"]
    %0 = vm.add.i32 %arg0, %arg0 : i32
    // CHECK: vm.sub.i32
    // %0 dies here, so result reuses i1.
    // CHECK-SAME: result_registers = ["i1"]
    %1 = vm.sub.i32 %arg0, %0 : i32
    vm.return %1 : i32
  }

  // CHECK-LABEL: @const_allocation
  // Constants allocate result registers.
  vm.func @const_allocation() -> i32 {
    // CHECK: vm.const.i32 42
    // CHECK-SAME: result_registers = ["i0"]
    %c = vm.const.i32 42
    vm.return %c : i32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Cross-Block Register Reuse
//===----------------------------------------------------------------------===//

// When a value dies in one path, its register can be reused in another.

// CHECK-LABEL: @module_cross_block_reuse
vm.module @module_cross_block_reuse {

  // CHECK-LABEL: @dead_value_register_reused
  // %x is only used in ^bb1, so its register can be reused in ^bb2.
  vm.func @dead_value_register_reused(%arg0: i32, %cond: i32) -> i32 {
    // CHECK: vm.add.i32
    // CHECK-SAME: block_registers = ["i0", "i1"]
    %x = vm.add.i32 %arg0, %arg0 : i32
    // CHECK: vm.cond_br
    vm.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    // %x is used here, then dead.
    vm.return %x : i32
  ^bb2:
    // %x is NOT used here - register can be reused for %y.
    %y = vm.mul.i32 %arg0, %arg0 : i32
    vm.return %y : i32
  }

  // CHECK-LABEL: @nested_control_flow_reuse
  // Value liveness across multiple nested paths.
  vm.func @nested_control_flow_reuse(%arg0: i32, %cond1: i32, %cond2: i32) -> i32 {
    %x = vm.add.i32 %arg0, %arg0 : i32
    // CHECK: vm.cond_br
    vm.cond_br %cond1, ^outer_true, ^outer_false
  ^outer_true:
    // %x is live here.
    vm.cond_br %cond2, ^inner_true, ^inner_false
  ^inner_true:
    vm.return %x : i32
  ^inner_false:
    %y = vm.mul.i32 %x, %arg0 : i32
    vm.return %y : i32
  ^outer_false:
    // %x is NOT used in this path, register can be reused.
    %z = vm.sub.i32 %arg0, %arg0 : i32
    vm.return %z : i32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Branch Argument Remapping
//===----------------------------------------------------------------------===//

// When branch operands don't match block arg registers, remapping is needed.

// CHECK-LABEL: @module_branch_remap
vm.module @module_branch_remap {

  // CHECK-LABEL: @simple_branch_remap
  // Branch operands need to be remapped to block arg registers.
  vm.func @simple_branch_remap(%arg0: i32, %arg1: i32) -> i32 {
    // CHECK: vm.br
    // CHECK-SAME: block_registers = ["i0", "i1"]
    // CHECK-SAME: remap_registers = [
    vm.br ^bb1(%arg0, %arg1 : i32, i32)
  ^bb1(%0: i32, %1: i32):
    // CHECK: vm.return
    // CHECK-SAME: block_registers =
    vm.return %0 : i32
  }

  // CHECK-LABEL: @branch_args_swap
  // Swapping args requires cycle detection in remap.
  vm.func @branch_args_swap(%arg0: i32, %arg1: i32) -> i32 {
    // CHECK: vm.br
    // CHECK-SAME: block_registers = ["i0", "i1"]
    // CHECK-SAME: remap_registers = [
    vm.br ^bb1(%arg1, %arg0 : i32, i32)
  ^bb1(%0: i32, %1: i32):
    vm.return %0 : i32
  }

  // CHECK-LABEL: @branch_args_swap_i64
  // i64 swap requires handling aligned register pairs.
  vm.func @branch_args_swap_i64(%arg0: i64, %arg1: i64) -> i64 {
    // CHECK: vm.br
    // CHECK-SAME: block_registers = ["i0+1", "i2+3"]
    // CHECK-SAME: remap_registers = [
    vm.br ^bb1(%arg1, %arg0 : i64, i64)
  ^bb1(%0: i64, %1: i64):
    vm.return %0 : i64
  }

  // CHECK-LABEL: @branch_args_triple_shuffle
  // Three-way shuffle: [a,b,c] -> [b,c,a].
  vm.func @branch_args_triple_shuffle(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    // CHECK: vm.br
    // CHECK-SAME: block_registers = ["i0", "i1", "i2"]
    // CHECK-SAME: remap_registers = [
    vm.br ^bb1(%arg1, %arg2, %arg0 : i32, i32, i32)
  ^bb1(%0: i32, %1: i32, %2: i32):
    vm.return %0 : i32
  }

  // CHECK-LABEL: @cond_branch_different_remaps
  // Different remapping needed for each successor.
  vm.func @cond_branch_different_remaps(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    // CHECK: vm.cond_br
    // CHECK-SAME: block_registers = ["i0", "i1", "i2"]
    // CHECK-SAME: remap_registers = [
    vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)
  ^bb1(%0: i32):
    vm.return %0 : i32
  ^bb2(%1: i32):
    vm.return %1 : i32
  }

  // CHECK-LABEL: @cond_branch_swapped_args
  // Conditional branch with different arg orders to same-shaped successors.
  vm.func @cond_branch_swapped_args(%cond: i32, %arg1: i32, %arg2: i32) -> i32 {
    // CHECK: vm.cond_br
    // CHECK-SAME: block_registers = ["i0", "i1", "i2"]
    // CHECK-SAME: remap_registers = [
    vm.cond_br %cond, ^bb1(%arg1, %arg2 : i32, i32), ^bb2(%arg2, %arg1 : i32, i32)
  ^bb1(%0: i32, %1: i32):
    vm.return %0 : i32
  ^bb2(%2: i32, %3: i32):
    vm.return %3 : i32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Loop Patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_loops
vm.module @module_loops {

  // CHECK-LABEL: @simple_loop
  // Basic loop with induction variable.
  vm.func @simple_loop() -> i32 {
    // CHECK: vm.const.i32
    // CHECK-SAME: result_registers = ["i0"]
    %c1 = vm.const.i32 1
    // CHECK: vm.const.i32
    // CHECK-SAME: result_registers = ["i1"]
    %c5 = vm.const.i32 5
    // CHECK: vm.const.i32.zero
    // CHECK-SAME: result_registers = ["i2"]
    %i0 = vm.const.i32.zero
    // CHECK: vm.br
    vm.br ^loop(%i0 : i32)
  ^loop(%i: i32):
    %in = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %in, %c5 : i32
    // CHECK: vm.cond_br
    vm.cond_br %cmp, ^loop(%in : i32), ^exit(%in : i32)
  ^exit(%ie: i32):
    vm.return %ie : i32
  }

  // CHECK-LABEL: @loop_with_multiple_carried_values
  // Loop carrying multiple values (accumulator pattern).
  vm.func @loop_with_multiple_carried_values(%init: i32, %limit: i32) -> i32 {
    %c1 = vm.const.i32 1
    vm.br ^header(%init, %init : i32, i32)
  ^header(%i: i32, %acc: i32):
    %cmp = vm.cmp.lt.i32.s %i, %limit : i32
    vm.cond_br %cmp, ^body, ^exit
  ^body:
    %next_i = vm.add.i32 %i, %c1 : i32
    %next_acc = vm.add.i32 %acc, %i : i32
    vm.br ^header(%next_i, %next_acc : i32, i32)
  ^exit:
    vm.return %acc : i32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Coalescing (Register Hints)
//===----------------------------------------------------------------------===//

// When a branch operand interval ends exactly where a block arg starts,
// they can share the same register via coalescing.

// CHECK-LABEL: @module_coalescing
vm.module @module_coalescing {

  // CHECK-LABEL: @coalescing_simple
  // Branch operand should hint block arg to use same register.
  vm.func @coalescing_simple(%arg0: i32, %cond: i32) -> i32 {
    // CHECK: vm.const.i32
    // CHECK-SAME: block_registers = ["i0", "i1"]
    %c1 = vm.const.i32 1
    // When branching with %c1, block arg could get same register.
    // CHECK: vm.cond_br
    vm.cond_br %cond, ^bb1(%c1 : i32), ^bb2
  ^bb1(%v: i32):
    vm.return %v : i32
  ^bb2:
    vm.return %arg0 : i32
  }

  // CHECK-LABEL: @diamond_coalescing
  // Diamond pattern - values from both paths merge.
  vm.func @diamond_coalescing(%arg0: i32, %cond: i32) -> i32 {
    // CHECK: vm.cond_br
    vm.cond_br %cond, ^left, ^right
  ^left:
    %x = vm.add.i32 %arg0, %arg0 : i32
    vm.br ^merge(%x : i32)
  ^right:
    %y = vm.sub.i32 %arg0, %arg0 : i32
    vm.br ^merge(%y : i32)
  ^merge(%result: i32):
    // %result comes from either %x or %y.
    vm.return %result : i32
  }
}

// -----

//===----------------------------------------------------------------------===//
// i64 Alignment
//===----------------------------------------------------------------------===//

// i64 values must be aligned to even register boundaries.

// CHECK-LABEL: @module_i64_alignment
vm.module @module_i64_alignment {

  // CHECK-LABEL: @i64_non_entry_block
  // Non-entry block args with i64 must also be aligned.
  vm.func @i64_non_entry_block(%arg0: i64, %cond: i32) -> i64 {
    // Entry i64 should be on even boundary.
    // CHECK: vm.cond_br
    // CHECK-SAME: block_registers = ["i0+1", "i2"]
    vm.cond_br %cond, ^bb1(%arg0 : i64), ^bb2
  ^bb1(%v: i64):
    // Non-entry i64 block arg should also be properly aligned.
    vm.return %v : i64
  ^bb2:
    %zero = vm.const.i64.zero
    vm.return %zero : i64
  }

  // CHECK-LABEL: @i64_cond_branch_swap
  // i64 swap in conditional branch.
  vm.func @i64_cond_branch_swap(%cond: i32, %arg1: i64, %arg2: i64) -> i64 {
    // CHECK: vm.cond_br
    // CHECK-SAME: block_registers = ["i0", "i2+3", "i4+5"]
    // CHECK-SAME: remap_registers = [
    vm.cond_br %cond, ^bb1(%arg1, %arg2 : i64, i64), ^bb2(%arg2, %arg1 : i64, i64)
  ^bb1(%0: i64, %1: i64):
    vm.return %0 : i64
  ^bb2(%2: i64, %3: i64):
    vm.return %3 : i64
  }
}
