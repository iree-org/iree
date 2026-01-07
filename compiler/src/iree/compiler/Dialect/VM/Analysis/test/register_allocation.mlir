// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(vm.func(test-iree-vm-register-allocation)))" %s | FileCheck %s

// Representative mixed int+ref tests for register allocation.
// These are complex, realistic examples that exercise multiple allocation features.
// For exhaustive int-only tests, see register_allocation_ints.mlir.
// For exhaustive ref-only tests, see register_allocation_refs.mlir.
//
// MOVE bit semantics:
// - Uppercase 'R' = move (ownership transfer, last use)
// - Lowercase 'r' = retain (not last use)

//===----------------------------------------------------------------------===//
// Mixed Entry Arguments
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_mixed_entry
vm.module @module_mixed_entry {

  vm.import private @use_all(%i0 : i32, %r0 : !vm.buffer, %i1 : i64, %r1 : !vm.buffer, %i2 : i32)

  // CHECK-LABEL: @entry_mixed_types
  // Entry args: int bank monotonic (i0, i2+3, i4), ref bank monotonic (r0, r1).
  vm.func @entry_mixed_types(%i0: i32, %r0: !vm.buffer, %i1: i64, %r1: !vm.buffer, %i2: i32) {
    // CHECK: vm.call @use_all
    // CHECK-SAME: block_registers = ["i0", "r0", "i2+3", "r1", "i4"]
    // CHECK-SAME: operand_registers = ["i0", "R0", "i2+3", "R1", "i4"]
    vm.call @use_all(%i0, %r0, %i1, %r1, %i2) : (i32, !vm.buffer, i64, !vm.buffer, i32) -> ()
    vm.discard.refs %r0, %r1 : !vm.buffer, !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Complex Control Flow with Mixed Types
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_complex_cfg
vm.module @module_complex_cfg {

  vm.import private @use_int(%i : i32)
  vm.import private @use_ref(%r : !vm.buffer)
  vm.import private @produce_ref() -> !vm.buffer

  // CHECK-LABEL: @diamond_mixed
  // Diamond with both int and ref - tests independent bank allocation.
  vm.func @diamond_mixed(%cond: i32, %init: i32, %buf: !vm.buffer) -> i32 {
    // CHECK: vm.cond_br
    // CHECK-SAME: operand_registers = ["i0", "r0", "R0"]
    vm.cond_br %cond, ^left(%buf : !vm.buffer), ^right(%buf : !vm.buffer)
  ^left(%b1: !vm.buffer):
    // CHECK: vm.call @use_ref
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @use_ref(%b1) : (!vm.buffer) -> ()
    vm.discard.refs %b1 : !vm.buffer
    %x = vm.add.i32 %init, %init : i32
    vm.br ^merge(%x : i32)
  ^right(%b2: !vm.buffer):
    // CHECK: vm.call @use_ref
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @use_ref(%b2) : (!vm.buffer) -> ()
    vm.discard.refs %b2 : !vm.buffer
    %y = vm.sub.i32 %init, %init : i32
    vm.br ^merge(%y : i32)
  ^merge(%result: i32):
    // CHECK: vm.return
    vm.return %result : i32
  }

  // CHECK-LABEL: @loop_with_mixed_carried
  // Loop carrying both int and ref values.
  vm.func @loop_with_mixed_carried(%n: i32, %buf: !vm.buffer) -> i32 {
    %c0 = vm.const.i32.zero
    %c1 = vm.const.i32 1
    // CHECK: vm.br ^bb1
    // Same value (%c0) used twice in operands - allocator uses i1 for both.
    // CHECK-SAME: operand_registers = ["i1", "i1", "R0"]
    vm.br ^loop(%c0, %c0, %buf : i32, i32, !vm.buffer)
  ^loop(%i: i32, %acc: i32, %b: !vm.buffer):
    // CHECK: vm.call @use_ref
    // Not MOVE - ref escapes via back-edge.
    // CHECK-SAME: operand_registers = ["r0"]
    vm.call @use_ref(%b) : (!vm.buffer) -> ()
    %i_next = vm.add.i32 %i, %c1 : i32
    %acc_next = vm.add.i32 %acc, %i : i32
    %cmp = vm.cmp.lt.i32.s %i_next, %n : i32
    // CHECK: vm.cond_br
    vm.cond_br %cmp, ^loop(%i_next, %acc_next, %b : i32, i32, !vm.buffer), ^exit(%acc_next : i32)
  ^exit(%final: i32):
    // Ref dies on exit edge.
    // CHECK: vm.discard.refs
    vm.discard.refs %b : !vm.buffer
    vm.return %final : i32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Branch Argument Remapping
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_remap
vm.module @module_remap {

  vm.import private @use_ref(%r : !vm.buffer)

  // CHECK-LABEL: @int_swap_via_branch
  // Swapping int values requires cycle detection in remapper.
  vm.func @int_swap_via_branch(%a: i32, %b: i32) -> i32 {
    // CHECK: vm.br
    // CHECK-SAME: block_registers = ["i0", "i1"]
    // CHECK-SAME: remap_registers = [
    vm.br ^bb1(%b, %a : i32, i32)
  ^bb1(%x: i32, %y: i32):
    vm.return %x : i32
  }

  // CHECK-LABEL: @ref_swap_via_branch
  // Swapping ref values requires scratch register for cyclic permutation.
  vm.func @ref_swap_via_branch(%a: !vm.buffer, %b: !vm.buffer) {
    // CHECK: vm.br
    // CHECK-SAME: block_registers = ["r0", "r1"]
    // CHECK-SAME: operand_registers = ["R1", "R0"]
    // CHECK-SAME: remap_registers = [
    vm.br ^bb1(%b, %a : !vm.buffer, !vm.buffer)
  ^bb1(%x: !vm.buffer, %y: !vm.buffer):
    vm.call @use_ref(%x) : (!vm.buffer) -> ()
    vm.call @use_ref(%y) : (!vm.buffer) -> ()
    vm.discard.refs %x, %y : !vm.buffer, !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @mixed_swap_via_branch
  // Swapping mixed types - each bank independent.
  vm.func @mixed_swap_via_branch(%i0: i32, %r0: !vm.buffer, %i1: i32, %r1: !vm.buffer) {
    // CHECK: vm.br
    // Swaps: i0<->i1, r0<->r1
    // CHECK-SAME: operand_registers = ["i1", "R1", "i0", "R0"]
    // CHECK-SAME: remap_registers = [
    vm.br ^bb1(%i1, %r1, %i0, %r0 : i32, !vm.buffer, i32, !vm.buffer)
  ^bb1(%a: i32, %b: !vm.buffer, %c: i32, %d: !vm.buffer):
    vm.call @use_ref(%b) : (!vm.buffer) -> ()
    vm.call @use_ref(%d) : (!vm.buffer) -> ()
    vm.discard.refs %b, %d : !vm.buffer, !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Coalescing
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_coalesce
vm.module @module_coalesce {

  vm.import private @use_ref(%r : !vm.buffer)
  vm.import private @produce_ref() -> !vm.buffer

  // CHECK-LABEL: @coalesce_branch_operand_to_block_arg
  // Branch operand should coalesce with block arg when intervals meet exactly.
  vm.func @coalesce_branch_operand_to_block_arg(%cond: i32) {
    %c1 = vm.const.i32 1
    %buf = vm.call @produce_ref() : () -> !vm.buffer
    // CHECK: vm.cond_br
    // %buf should coalesce with block arg.
    vm.cond_br %cond, ^use(%buf : !vm.buffer), ^discard
  ^use(%b: !vm.buffer):
    // CHECK: vm.call @use_ref
    // Block arg may get remapped to different register.
    // CHECK-SAME: operand_registers = ["R{{[0-9]+}}"]
    vm.call @use_ref(%b) : (!vm.buffer) -> ()
    vm.discard.refs %b : !vm.buffer
    vm.return
  ^discard:
    // CHECK: vm.discard.refs
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Select with Mixed Liveness
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_select_mixed
vm.module @module_select_mixed {

  vm.import private @use_int(%i : i32)
  vm.import private @use_ref(%r : !vm.buffer)

  // CHECK-LABEL: @select_ref_with_int_cond
  // Select ref based on int condition, with mixed liveness.
  vm.func @select_ref_with_int_cond(%cond: i32, %count: i32, %a: !vm.buffer, %b: !vm.buffer) {
    // Use count to keep it live across select.
    vm.call @use_int(%count) : (i32) -> ()
    // CHECK: vm.select.ref
    // CHECK-SAME: operand_registers = ["i0", "R0", "R1"]
    %result = vm.select.ref %cond, %a, %b : !vm.buffer
    vm.discard.refs %a, %b : !vm.buffer, !vm.buffer
    // CHECK: vm.call @use_ref
    vm.call @use_ref(%result) : (!vm.buffer) -> ()
    vm.discard.refs %result : !vm.buffer
    // Use count again.
    vm.call @use_int(%count) : (i32) -> ()
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Realistic Computation Pattern
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_realistic
vm.module @module_realistic {

  vm.import private @load_buffer(%idx: i32) -> !vm.buffer
  vm.import private @process_buffer(%buf: !vm.buffer, %param: i32) -> i32
  vm.import private @store_result(%buf: !vm.buffer, %result: i32)

  // CHECK-LABEL: @process_loop
  // Realistic pattern: load buffers, process in loop, store results.
  vm.func @process_loop(%n: i32) -> i32 {
    %c0 = vm.const.i32.zero
    %c1 = vm.const.i32 1
    vm.br ^loop(%c0, %c0 : i32, i32)
  ^loop(%i: i32, %total: i32):
    // Load a buffer.
    %buf = vm.call @load_buffer(%i) : (i32) -> !vm.buffer
    // Process it.
    %result = vm.call @process_buffer(%buf, %i) : (!vm.buffer, i32) -> i32
    // Store result (buf still needed).
    // CHECK: vm.call @store_result
    // CHECK-SAME: operand_registers = ["R0", "i{{[0-9]+}}"]
    vm.call @store_result(%buf, %result) : (!vm.buffer, i32) -> ()
    vm.discard.refs %buf : !vm.buffer
    // Accumulate.
    %new_total = vm.add.i32 %total, %result : i32
    %i_next = vm.add.i32 %i, %c1 : i32
    %continue = vm.cmp.lt.i32.s %i_next, %n : i32
    vm.cond_br %continue, ^loop(%i_next, %new_total : i32, i32), ^exit(%new_total : i32)
  ^exit(%final: i32):
    vm.return %final : i32
  }
}
