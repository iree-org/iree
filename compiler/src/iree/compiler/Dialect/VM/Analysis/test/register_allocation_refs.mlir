// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(vm.func(test-iree-vm-register-allocation)))" %s | FileCheck %s

// Tests for ref register allocation with MOVE bit computation.
// MOVE bit is indicated by uppercase 'R' (move=true) vs lowercase 'r' (move=false).
// Example: R0 = ref register 0 with move, r0 = ref register 0 without move.
//
// IMPORTANT: These tests include explicit vm.discard.refs operations as they
// would appear after MaterializeRefDiscards runs. This is critical because:
// 1. The register allocator must account for discards in live interval calculation
// 2. Discards are NOT "real" uses - they don't get MOVE semantics
// 3. MOVE elision: when a preceding use has MOVE, the discard is elided at runtime

//===----------------------------------------------------------------------===//
// Entry Block Arguments (ABI Requirement)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_entry_args
vm.module @module_entry_args {

  vm.import private @consume3(%a : !vm.buffer, %b : !vm.buffer, %c : !vm.buffer)
  vm.import private @consume_mixed(%i0 : i32, %r0 : !vm.buffer, %i1 : i32, %r1 : !vm.buffer)

  // CHECK-LABEL: @entry_ref_args_monotonic
  // Entry block ref arguments must be allocated monotonically for ABI stability.
  vm.func @entry_ref_args_monotonic(%r0: !vm.buffer, %r1: !vm.buffer, %r2: !vm.buffer) {
    // CHECK: vm.call @consume3
    // CHECK-SAME: block_registers = ["r0", "r1", "r2"]
    // CHECK-SAME: operand_registers = ["R0", "R1", "R2"]
    vm.call @consume3(%r0, %r1, %r2) : (!vm.buffer, !vm.buffer, !vm.buffer) -> ()
    // Discards follow call - elided because call had MOVE
    vm.discard.refs %r0, %r1, %r2 : !vm.buffer, !vm.buffer, !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @entry_mixed_types_monotonic
  // Mixed i32 and ref args should each be allocated monotonically in their banks.
  vm.func @entry_mixed_types_monotonic(%i0: i32, %r0: !vm.buffer, %i1: i32, %r1: !vm.buffer) {
    // CHECK: vm.call @consume_mixed
    // CHECK-SAME: block_registers = ["i0", "r0", "i1", "r1"]
    // CHECK-SAME: operand_registers = ["i0", "R0", "i1", "R1"]
    vm.call @consume_mixed(%i0, %r0, %i1, %r1) : (i32, !vm.buffer, i32, !vm.buffer) -> ()
    vm.discard.refs %r0, %r1 : !vm.buffer, !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Basic Linear Flow - MOVE Bit Semantics
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_basic
vm.module @module_basic {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @ref_simple_use
  // Single use of ref - MOVE on call, discard is elided at runtime.
  vm.func @ref_simple_use(%buf : !vm.buffer) {
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    // Discard follows - elided because call had MOVE
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @ref_multiple_uses
  // Multiple uses - only last real use gets MOVE. Discard is elided.
  vm.func @ref_multiple_uses(%buf : !vm.buffer) {
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["r0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["r0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    // CHECK: vm.call @consume_buffer
    // Last real use - gets MOVE.
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // Discard never gets MOVE.
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @ref_used_in_return
  // Ref used in return - IS MOVE since return transfers ownership to caller.
  // No discard needed for return values.
  vm.func @ref_used_in_return(%buf : !vm.buffer) -> !vm.buffer {
    // CHECK: vm.return
    // CHECK-SAME: operand_registers = ["R0"]
    vm.return %buf : !vm.buffer
  }
}

// -----

//===----------------------------------------------------------------------===//
// Same Value Used Multiple Times in Same Instruction
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_same_value
vm.module @module_same_value {

  vm.import private @use_two_buffers(%a : !vm.buffer, %b : !vm.buffer)
  vm.import private @use_three_buffers(%a : !vm.buffer, %b : !vm.buffer, %c : !vm.buffer)

  // CHECK-LABEL: @same_ref_twice
  // Same ref used twice in one call - only LAST operand gets MOVE.
  vm.func @same_ref_twice(%buf : !vm.buffer) {
    // CHECK: vm.call @use_two_buffers
    // CHECK-SAME: operand_registers = ["r0", "R0"]
    vm.call @use_two_buffers(%buf, %buf) : (!vm.buffer, !vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @same_ref_three_times
  // Same ref used three times - only LAST operand gets MOVE.
  vm.func @same_ref_three_times(%buf : !vm.buffer) {
    // CHECK: vm.call @use_three_buffers
    // CHECK-SAME: operand_registers = ["r0", "r0", "R0"]
    vm.call @use_three_buffers(%buf, %buf, %buf) : (!vm.buffer, !vm.buffer, !vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @same_ref_used_after
  // Same ref twice, but also used after - NO MOVE on any in first call.
  vm.func @same_ref_used_after(%buf : !vm.buffer) {
    // CHECK: vm.call @use_two_buffers
    // First call - not last use, no MOVE.
    // CHECK-SAME: operand_registers = ["r0", "r0"]
    vm.call @use_two_buffers(%buf, %buf) : (!vm.buffer, !vm.buffer) -> ()
    // CHECK: vm.call @use_two_buffers
    // Second call - this IS last use, MOVE on last operand.
    // CHECK-SAME: operand_registers = ["r0", "R0"]
    vm.call @use_two_buffers(%buf, %buf) : (!vm.buffer, !vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Diamond Control Flow
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_diamond
vm.module @module_diamond {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @diamond_both_paths_use
  // Diamond CFG - same ref passed to both branches.
  // Since same value appears twice in operand list, last occurrence gets MOVE.
  // Block args coalesce with branch operands, so they share the same register.
  vm.func @diamond_both_paths_use(%cond : i32, %buf : !vm.buffer) {
    // CHECK: vm.cond_br
    // CHECK-SAME: operand_registers = ["i0", "r0", "R0"]
    vm.cond_br %cond, ^bb1(%buf : !vm.buffer), ^bb2(%buf : !vm.buffer)
  ^bb1(%b1 : !vm.buffer):
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%b1) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %b1 : !vm.buffer
    vm.return
  ^bb2(%b2 : !vm.buffer):
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%b2) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %b2 : !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @diamond_one_path_uses
  // Diamond - ref only passed to one branch.
  // The other branch gets a discard for the original value.
  vm.func @diamond_one_path_uses(%cond : i32, %buf : !vm.buffer) {
    // CHECK: vm.cond_br
    // Only one branch gets the ref, so it IS MOVE.
    // Block arg gets remapped to r1.
    // CHECK-SAME: operand_registers = ["i0", "R0"]
    // CHECK-SAME: remap_registers = {{\[}}{{\[}}"R0->r1"{{\]}}, {{\[}}{{\]}}{{\]}}
    vm.cond_br %cond, ^bb1(%buf : !vm.buffer), ^bb2
  ^bb1(%b1 : !vm.buffer):
    // CHECK: vm.call @consume_buffer
    // Block arg is r1 (remapped).
    // CHECK-SAME: operand_registers = ["R1"]
    vm.call @consume_buffer(%b1) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r1"]
    vm.discard.refs %b1 : !vm.buffer
    vm.return
  ^bb2:
    // Ref dies on this edge - discard for original %buf (r0).
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @diamond_asymmetric_call
  // Diamond where one path calls with ref, other doesn't use it.
  vm.func @diamond_asymmetric_call(%cond : i32, %buf : !vm.buffer) {
    // CHECK: vm.cond_br
    // Same ref to both branches.
    // CHECK-SAME: operand_registers = ["i0", "r0", "R0"]
    vm.cond_br %cond, ^use_path(%buf : !vm.buffer), ^nouse_path(%buf : !vm.buffer)
  ^use_path(%b1 : !vm.buffer):
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%b1) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %b1 : !vm.buffer
    vm.return
  ^nouse_path(%b2 : !vm.buffer):
    // No real use of %b2 here - just discard.
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %b2 : !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Loop Patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_loops
vm.module @module_loops {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @loop_with_ref
  // Loop - ref live across back-edge.
  // Call inside loop is NOT MOVE (ref escapes via back-edge).
  // Back-edge IS MOVE (ownership transfers to next iteration).
  // Exit edge gets a discard.
  vm.func @loop_with_ref(%count : i32, %buf : !vm.buffer) {
    %c0 = vm.const.i32.zero
    %c1 = vm.const.i32 1
    // CHECK: vm.br ^bb1
    // Initial branch - MOVE since not used again in entry.
    // CHECK-SAME: operand_registers = ["i1", "R0"]
    vm.br ^loop(%c0, %buf : i32, !vm.buffer)
  ^loop(%i : i32, %b : !vm.buffer):
    // Use the buffer.
    // CHECK: vm.call @consume_buffer
    // Ref escapes via back-edge, not last use.
    // CHECK-SAME: operand_registers = ["r0"]
    vm.call @consume_buffer(%b) : (!vm.buffer) -> ()
    %i_next = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %i_next, %count : i32
    // CHECK: vm.cond_br
    // Back-edge: ref gets MOVE to transfer to next iteration.
    // Exit edge: no ref operand - discard happens in exit block.
    // CHECK-SAME: operand_registers = ["i3", "i1", "R0"]
    vm.cond_br %cmp, ^loop(%i_next, %b : i32, !vm.buffer), ^exit
  ^exit:
    // Discard the loop-carried ref that died on exit edge.
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %b : !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @loop_ref_used_after_exit
  // Ref from entry block used inside and after loop.
  // Not a loop-carried value - different liveness pattern.
  vm.func @loop_ref_used_after_exit(%count : i32, %buf : !vm.buffer) {
    %c0 = vm.const.i32.zero
    %c1 = vm.const.i32 1
    // CHECK: vm.br ^bb1
    vm.br ^loop(%c0 : i32)
  ^loop(%i : i32):
    // CHECK: vm.call @consume_buffer
    // Not MOVE - ref is used after loop exits.
    // CHECK-SAME: operand_registers = ["r0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    %i_next = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %i_next, %count : i32
    // CHECK: vm.cond_br
    vm.cond_br %cmp, ^loop(%i_next : i32), ^exit
  ^exit:
    // CHECK: vm.call @consume_buffer
    // Final use - MOVE.
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Nested Loops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_nested_loops
vm.module @module_nested_loops {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @nested_loop_outer_ref
  // Outer loop carries a ref, inner loop just iterates.
  // The ref should NOT get MOVE inside either loop body.
  vm.func @nested_loop_outer_ref(%outer_n : i32, %inner_n : i32, %buf : !vm.buffer) {
    %c0 = vm.const.i32.zero
    %c1 = vm.const.i32 1
    // CHECK: vm.br ^bb1
    // Initial branch to outer loop - MOVE since not used again in entry.
    // CHECK-SAME: operand_registers = ["i2", "R0"]
    vm.br ^outer(%c0, %buf : i32, !vm.buffer)
  ^outer(%outer_i : i32, %outer_buf : !vm.buffer):
    // Use the buffer in outer loop.
    // CHECK: vm.call @consume_buffer
    // Not last use - used again at outer back-edge.
    // CHECK-SAME: operand_registers = ["r0"]
    vm.call @consume_buffer(%outer_buf) : (!vm.buffer) -> ()
    vm.br ^inner(%c0 : i32)
  ^inner(%inner_i : i32):
    // Inner loop doesn't touch the ref.
    %inner_next = vm.add.i32 %inner_i, %c1 : i32
    %inner_cmp = vm.cmp.lt.i32.s %inner_next, %inner_n : i32
    vm.cond_br %inner_cmp, ^inner(%inner_next : i32), ^outer_check
  ^outer_check:
    %outer_next = vm.add.i32 %outer_i, %c1 : i32
    %outer_cmp = vm.cmp.lt.i32.s %outer_next, %outer_n : i32
    // CHECK: vm.cond_br {{.*}} ^bb1
    // Outer back-edge - ref does NOT get MOVE because it's also live to exit.
    // (Value used by discard in exit block.)
    // CHECK-SAME: operand_registers = ["i5", "i4", "r0"]
    vm.cond_br %outer_cmp, ^outer(%outer_next, %outer_buf : i32, !vm.buffer), ^exit
  ^exit:
    // Discard after loop exits.
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %outer_buf : !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Ref Swap (Ping-Pong) Pattern
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_ping_pong
vm.module @module_ping_pong {

  vm.import private @use_two_buffers(%a : !vm.buffer, %b : !vm.buffer)

  // CHECK-LABEL: @ping_pong_swap
  // Loop that swaps two refs on each iteration.
  // Both refs should get MOVE on the back-edge.
  // Requires scratch register for cyclic permutation.
  vm.func @ping_pong_swap(%n : i32, %a : !vm.buffer, %b : !vm.buffer) {
    %c0 = vm.const.i32.zero
    %c1 = vm.const.i32 1
    // CHECK: vm.br ^bb1
    // Initial branch - both refs MOVE to loop header.
    // CHECK-SAME: operand_registers = ["i1", "R0", "R1"]
    vm.br ^loop(%c0, %a, %b : i32, !vm.buffer, !vm.buffer)
  ^loop(%i : i32, %x : !vm.buffer, %y : !vm.buffer):
    // Use both buffers.
    // CHECK: vm.call @use_two_buffers
    // Neither is last use - both used at back-edge.
    // CHECK-SAME: operand_registers = ["r0", "r1"]
    vm.call @use_two_buffers(%x, %y) : (!vm.buffer, !vm.buffer) -> ()
    %i_next = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %i_next, %n : i32
    // CHECK: vm.cond_br
    // Back-edge SWAPS positions: x->y_arg, y->x_arg.
    // Both should get MOVE since they're last uses in this instruction.
    // CHECK-SAME: operand_registers = ["i3", "i1", "R1", "R0"]
    vm.cond_br %cmp, ^loop(%i_next, %y, %x : i32, !vm.buffer, !vm.buffer), ^exit
  ^exit:
    // Discard both after loop exits.
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0", "r1"]
    vm.discard.refs %x, %y : !vm.buffer, !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Select Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_select
vm.module @module_select {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @select_ref_both_used
  // Select - both operands are last use.
  vm.func @select_ref_both_used(%cond : i32, %a : !vm.buffer, %b : !vm.buffer) {
    // CHECK: vm.select.ref
    // CHECK-SAME: operand_registers = ["i0", "R0", "R1"]
    // CHECK-SAME: result_registers = ["r2"]
    %result = vm.select.ref %cond, %a, %b : !vm.buffer
    // Discard non-selected operand - both are last use above
    // but only one survives in %result.
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0", "r1"]
    vm.discard.refs %a, %b : !vm.buffer, !vm.buffer
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R2"]
    vm.call @consume_buffer(%result) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r2"]
    vm.discard.refs %result : !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @select_ref_one_reused
  // Select - one operand reused after.
  vm.func @select_ref_one_reused(%cond : i32, %a : !vm.buffer, %b : !vm.buffer) {
    // CHECK: vm.select.ref
    // %a is reused, %b is not.
    // CHECK-SAME: operand_registers = ["i0", "r0", "R1"]
    // CHECK-SAME: result_registers = ["r2"]
    %result = vm.select.ref %cond, %a, %b : !vm.buffer
    // Discard %b - no longer accessible via %result
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r1"]
    vm.discard.refs %b : !vm.buffer
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R2"]
    vm.call @consume_buffer(%result) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r2"]
    vm.discard.refs %result : !vm.buffer
    // CHECK: vm.call @consume_buffer
    // %a is last use here.
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%a) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %a : !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Global Refs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_globals
vm.module @module_globals {

  vm.global.ref private mutable @global_buf : !vm.buffer

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @store_to_global
  // Storing to global - ref still in scope after store.
  vm.func @store_to_global(%buf : !vm.buffer) {
    // CHECK: vm.global.store.ref
    // CHECK-SAME: operand_registers = ["r0"]
    vm.global.store.ref %buf, @global_buf : !vm.buffer
    // CHECK: vm.call @consume_buffer
    // After store, this IS the last use.
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @load_and_use_global
  // Loading from global - creates new ref.
  vm.func @load_and_use_global() {
    %buf = vm.global.load.ref @global_buf : !vm.buffer
    // CHECK: vm.call @consume_buffer
    // Loaded ref, this is last use.
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// List Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_lists
vm.module @module_lists {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @list_set_ref
  // Setting ref into list - list operand does NOT get MOVE (isRefOperandMovable
  // returns false for it). Only value operand gets MOVE.
  vm.func @list_set_ref(%list : !vm.list<!vm.buffer>, %idx : i32, %buf : !vm.buffer) {
    // CHECK: vm.list.set.ref
    // CHECK-SAME: operand_registers = ["r0", "i0", "R1"]
    vm.list.set.ref %list, %idx, %buf : (!vm.list<!vm.buffer>, i32, !vm.buffer)
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0", "r1"]
    vm.discard.refs %list, %buf : !vm.list<!vm.buffer>, !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @list_get_ref
  // Getting ref from list - creates new ref.
  vm.func @list_get_ref(%list : !vm.list<!vm.buffer>, %idx : i32) {
    // CHECK: vm.list.get.ref
    // CHECK-SAME: operand_registers = ["r0", "i0"]
    %buf = vm.list.get.ref %list, %idx : (!vm.list<!vm.buffer>, i32) -> !vm.buffer
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R1"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0", "r1"]
    vm.discard.refs %list, %buf : !vm.list<!vm.buffer>, !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_comparisons
vm.module @module_comparisons {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @cmp_eq_ref
  // Ref comparison - both refs survive (comparison doesn't consume).
  vm.func @cmp_eq_ref(%a : !vm.buffer, %b : !vm.buffer) -> i32 {
    // CHECK: vm.cmp.eq.ref
    // CHECK-SAME: operand_registers = ["r0", "r1"]
    %eq = vm.cmp.eq.ref %a, %b : !vm.buffer
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0", "r1"]
    vm.discard.refs %a, %b : !vm.buffer, !vm.buffer
    vm.return %eq : i32
  }

  // CHECK-LABEL: @cmp_eq_ref_reused
  // One ref reused after comparison.
  vm.func @cmp_eq_ref_reused(%a : !vm.buffer, %b : !vm.buffer) -> i32 {
    // CHECK: vm.cmp.eq.ref
    // CHECK-SAME: operand_registers = ["r0", "r1"]
    %eq = vm.cmp.eq.ref %a, %b : !vm.buffer
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r1"]
    vm.discard.refs %b : !vm.buffer
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%a) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %a : !vm.buffer
    vm.return %eq : i32
  }

  // CHECK-LABEL: @cmp_nz_ref
  // Null check - ref survives.
  vm.func @cmp_nz_ref(%buf : !vm.buffer) -> i32 {
    // CHECK: vm.cmp.nz.ref
    // CHECK-SAME: operand_registers = ["r0"]
    %nz = vm.cmp.nz.ref %buf : !vm.buffer
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return %nz : i32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Const Ref Zero
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_const_ref
vm.module @module_const_ref {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @const_ref_zero
  // const.ref.zero creates a null ref.
  vm.func @const_ref_zero() {
    %null = vm.const.ref.zero : !vm.buffer
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%null) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %null : !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Mixed Refs and Primitives
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_mixed
vm.module @module_mixed {

  vm.import private @mixed_args(%i : i32, %buf : !vm.buffer, %j : i32)

  // CHECK-LABEL: @mixed_operands
  // Refs mixed with primitives.
  vm.func @mixed_operands(%i : i32, %buf : !vm.buffer, %j : i32) {
    // CHECK: vm.call @mixed_args
    // Only ref should have MOVE consideration.
    // CHECK-SAME: operand_registers = ["i0", "R0", "i1"]
    vm.call @mixed_args(%i, %buf, %j) : (i32, !vm.buffer, i32) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @mixed_with_ref_reuse
  // Ref reused after mixed call.
  vm.func @mixed_with_ref_reuse(%i : i32, %buf : !vm.buffer, %j : i32) {
    // CHECK: vm.call @mixed_args
    // Ref is NOT last use.
    // CHECK-SAME: operand_registers = ["i0", "r0", "i1"]
    vm.call @mixed_args(%i, %buf, %j) : (i32, !vm.buffer, i32) -> ()
    // CHECK: vm.call @mixed_args
    // Now ref IS last use.
    // CHECK-SAME: operand_registers = ["i0", "R0", "i1"]
    vm.call @mixed_args(%i, %buf, %j) : (i32, !vm.buffer, i32) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Same Ref to Same Block via Both Edges
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_same_target
vm.module @module_same_target {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @same_ref_both_edges_same_target
  // cond_br where both edges go to same block with same ref.
  // This is like passing the same ref twice - only last operand gets MOVE.
  vm.func @same_ref_both_edges_same_target(%cond : i32, %buf : !vm.buffer) {
    // CHECK: vm.cond_br
    // Same ref to same target via both edges.
    // First occurrence (true branch operand) - NOT MOVE.
    // Second occurrence (false branch operand) - IS MOVE.
    // CHECK-SAME: operand_registers = ["i0", "r0", "R0"]
    vm.cond_br %cond, ^merge(%buf : !vm.buffer), ^merge(%buf : !vm.buffer)
  ^merge(%b : !vm.buffer):
    // CHECK: vm.call @consume_buffer
    vm.call @consume_buffer(%b) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs
    vm.discard.refs %b : !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Irreducible CFG with Refs
//===----------------------------------------------------------------------===//

// An irreducible CFG has multiple entry points to a loop-like structure.
// This tests that register allocation handles refs correctly when the CFG
// cannot be reduced to a simple loop structure.

// CHECK-LABEL: @module_irreducible_cfg
vm.module @module_irreducible_cfg {

  // Irreducible CFG: ^A and ^B can each reach the other, with external entry
  // to both. The ref must be tracked correctly through all paths.
  //
  //     entry
  //     /   \
  //    v     v
  //   ^A <-> ^B
  //    \     /
  //     v   v
  //     exit (gets ref from both paths)
  //
  // CHECK-LABEL: @irreducible_ref_flow
  vm.func @irreducible_ref_flow(%cond1: i32, %cond2: i32, %ref: !vm.buffer) {
    // Entry branches to either A or B, forwarding ref to both.
    // The last edge (to B) gets MOVE since ref is not used after.
    // CHECK: vm.cond_br
    // CHECK-SAME: operand_registers = ["i0", "r0", "R0"]
    vm.cond_br %cond1, ^A(%ref : !vm.buffer), ^B(%ref : !vm.buffer)

  ^A(%ref_a: !vm.buffer):
    %nz_a = vm.cmp.nz.ref %ref_a : !vm.buffer
    // A can go to B (forwarding ref) or exit (forwarding ref).
    // CHECK: vm.cond_br
    // CHECK-SAME: operand_registers = ["i1", "r0", "R0"]
    vm.cond_br %cond2, ^B(%ref_a : !vm.buffer), ^exit(%ref_a : !vm.buffer)

  ^B(%ref_b: !vm.buffer):
    %nz_b = vm.cmp.nz.ref %ref_b : !vm.buffer
    // B can go to A (forwarding ref) or exit (forwarding ref).
    // CHECK: vm.cond_br
    // CHECK-SAME: operand_registers = ["i1", "r0", "R0"]
    vm.cond_br %cond2, ^A(%ref_b : !vm.buffer), ^exit(%ref_b : !vm.buffer)

  ^exit(%ref_exit : !vm.buffer):
    // Discard the ref that reached here.
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %ref_exit : !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// vm.br_table with Refs
//===----------------------------------------------------------------------===//

// Branch table with refs tests the handling of multiple successors where the
// same ref may be forwarded to different targets with different block args.

// CHECK-LABEL: @module_br_table_refs
vm.module @module_br_table_refs {

  // Simple br_table: ref forwarded to all cases.
  // The same ref goes to all three targets.
  // CHECK-LABEL: @br_table_ref_all_cases
  vm.func @br_table_ref_all_cases(%idx: i32, %ref: !vm.buffer) {
    // CHECK: vm.br_table
    // All cases receive the same ref. The last case (case 1) gets MOVE.
    // CHECK: operand_registers = ["i0", "r0", "r0", "R0"]
    vm.br_table %idx {
      default: ^bb_default(%ref : !vm.buffer),
      0: ^bb0(%ref : !vm.buffer),
      1: ^bb1(%ref : !vm.buffer)
    }
  ^bb_default(%arg_default: !vm.buffer):
    // CHECK: vm.discard.refs
    vm.discard.refs %arg_default : !vm.buffer
    vm.return
  ^bb0(%arg0: !vm.buffer):
    // CHECK: vm.discard.refs
    vm.discard.refs %arg0 : !vm.buffer
    vm.return
  ^bb1(%arg1: !vm.buffer):
    // CHECK: vm.discard.refs
    vm.discard.refs %arg1 : !vm.buffer
    vm.return
  }

  // br_table: ref forwarded to some cases, not others.
  // Cases without the ref need discards (for original ref).
  // CHECK-LABEL: @br_table_ref_some_cases
  vm.func @br_table_ref_some_cases(%idx: i32, %ref: !vm.buffer) {
    // CHECK: vm.br_table
    // Ref forwarded to default and case 0, but NOT case 1.
    // Case 0 gets MOVE since case 1 doesn't use the ref.
    // CHECK: operand_registers = ["i0", "r0", "R0"]
    vm.br_table %idx {
      default: ^bb_default(%ref : !vm.buffer),
      0: ^bb0(%ref : !vm.buffer),
      1: ^bb1
    }
  ^bb_default(%arg_default: !vm.buffer):
    // CHECK: vm.discard.refs
    vm.discard.refs %arg_default : !vm.buffer
    vm.return
  ^bb0(%arg0: !vm.buffer):
    // CHECK: vm.discard.refs
    vm.discard.refs %arg0 : !vm.buffer
    vm.return
  ^bb1:
    // No ref here - discard for original %ref that died on this edge.
    // CHECK: vm.discard.refs
    vm.discard.refs %ref : !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Regression Test: Discard-Register Collision with Block Arguments
//===----------------------------------------------------------------------===//

// This test verifies that when a discard and a block argument are at the same
// block (same instruction position), they get different registers.
//
// The bug occurs when:
// 1. A value (%outer) ends in a predecessor block
// 2. The successor block has a block arg (%inner) AND a discard for %outer
// 3. Both %outer's discard and %inner's definition are at the same position
// 4. Register allocation reuses %outer's register for %inner
// 5. The discard kills %inner instead of %outer!
//
// FIX: Extend %outer's interval to include the discard, preventing reuse.

// CHECK-LABEL: @discard_block_arg_collision
vm.module @discard_block_arg_collision {
  vm.func @discard_block_arg_collision(%cond: i32) {
    // CHECK: %[[OUTER:.+]] = vm.const.ref.rodata @data
    // CHECK-SAME: result_registers = ["r0"]
    %outer = vm.const.ref.rodata @data : !vm.buffer
    // CHECK: vm.call @use_outer(%[[OUTER]])
    // CHECK-SAME: operand_registers = ["r0"]
    vm.call @use_outer(%outer) : (!vm.buffer) -> ()
    // CHECK: %[[INNER_SRC:.+]] = vm.call @create
    // CHECK-SAME: result_registers = ["r1"]
    %inner_src = vm.call @create() : () -> !vm.buffer
    // CHECK: vm.br ^{{.+}}(%[[INNER_SRC]] : !vm.buffer)
    // CHECK-SAME: operand_registers = ["R1"]
    vm.br ^bb1(%inner_src : !vm.buffer)

  // CHECK: ^{{.+}}(%[[INNER:.+]]: !vm.buffer):
  ^bb1(%inner: !vm.buffer):
    // Discard targets %outer (r0), NOT %inner (r1)
    // block_registers shows block arg is r1 (different from discard's r0)
    // CHECK: vm.discard.refs %[[OUTER]]
    // CHECK-SAME: block_registers = ["r1"]
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %outer : !vm.buffer
    // %inner is still alive in r1
    // CHECK: vm.call @use_inner(%[[INNER]])
    // CHECK-SAME: operand_registers = ["R1"]
    vm.call @use_inner(%inner) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r1"]
    vm.discard.refs %inner : !vm.buffer
    vm.return
  }

  vm.rodata private @data dense<[1, 2, 3]> : tensor<3xi8>
  vm.import private @use_outer(%buf: !vm.buffer)
  vm.import private @use_inner(%buf: !vm.buffer)
  vm.import private @create() -> !vm.buffer
}

// -----

//===----------------------------------------------------------------------===//
// Scratch Register in Ref Swap Pattern (Ping-Pong)
//===----------------------------------------------------------------------===//

// When swapping refs across a loop back-edge (ping-pong pattern), register
// allocation uses a scratch register for the cyclic permutation.
// The scratch register must have MOVE semantics to release its ref
// after the copy, preventing leaks when the branch takes the exit path.

// CHECK-LABEL: @module_scratch_register_swap
vm.module @module_scratch_register_swap {

  // CHECK-LABEL: @ping_pong_swap
  vm.func @ping_pong_swap(%ref_a: !vm.buffer, %ref_b: !vm.buffer, %n: i32) {
    %c0 = vm.const.i32.zero
    %c1 = vm.const.i32 1
    // CHECK: vm.br ^bb1
    // CHECK-SAME: operand_registers = ["i1", "R0", "R1"]
    vm.br ^loop(%c0, %ref_a, %ref_b : i32, !vm.buffer, !vm.buffer)
  ^loop(%i: i32, %x: !vm.buffer, %y: !vm.buffer):
    // CHECK: block_registers = ["i1", "r0", "r1"]
    %cmp_x = vm.cmp.nz.ref %x : !vm.buffer
    %cmp_y = vm.cmp.nz.ref %y : !vm.buffer
    %i_next = vm.add.i32 %i, %c1 : i32
    %continue = vm.cmp.lt.i32.s %i_next, %n : i32
    // CHECK: vm.cond_br
    // The swap creates a cyclic permutation requiring a scratch register.
    // The scratch register source must have MOVE to release the ref.
    // CHECK-SAME: operand_registers = ["i3", "i1", "R1", "R0"]
    vm.cond_br %continue, ^loop(%i_next, %y, %x : i32, !vm.buffer, !vm.buffer), ^exit
  ^exit:
    // CHECK: vm.discard.refs
    // CHECK-SAME: block_registers = []
    // CHECK-SAME: operand_registers = ["r0", "r1"]
    vm.discard.refs %x, %y : !vm.buffer, !vm.buffer
    vm.return
  }
}
