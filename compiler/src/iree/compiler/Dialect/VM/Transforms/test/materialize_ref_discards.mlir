// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(vm.module(iree-vm-materialize-ref-discards))" \
// RUN:   %s | FileCheck %s

// Single ref, single use - discard after use.
// CHECK-LABEL: @single_ref_single_use
// CHECK-SAME: (%[[BUF:.*]]: !vm.buffer)
vm.module @my_module {
  vm.func @single_ref_single_use(%buf: !vm.buffer) {
    // CHECK: vm.call @consume(%[[BUF]])
    // CHECK-NEXT: vm.discard.refs %[[BUF]] : !vm.buffer
    vm.call @consume(%buf) : (!vm.buffer) -> ()
    vm.return
  }
  vm.import private @consume(%buf: !vm.buffer)
}

// -----

// Multiple uses - discard after LAST use only.
// CHECK-LABEL: @multiple_uses
// CHECK-SAME: (%[[BUF:.*]]: !vm.buffer)
vm.module @my_module {
  vm.func @multiple_uses(%buf: !vm.buffer) {
    // CHECK: vm.call @consume(%[[BUF]])
    // CHECK-NOT: vm.discard.refs
    vm.call @consume(%buf) : (!vm.buffer) -> ()
    // CHECK: vm.call @consume(%[[BUF]])
    // CHECK-NEXT: vm.discard.refs %[[BUF]] : !vm.buffer
    vm.call @consume(%buf) : (!vm.buffer) -> ()
    vm.return
  }
  vm.import private @consume(%buf: !vm.buffer)
}

// -----

// Ref escapes via return - no discard.
// CHECK-LABEL: @ref_returned
// CHECK-SAME: (%[[BUF:.*]]: !vm.buffer)
vm.module @my_module {
  vm.func @ref_returned(%buf: !vm.buffer) -> !vm.buffer {
    // CHECK-NOT: vm.discard.refs
    // CHECK: vm.return %[[BUF]]
    vm.return %buf : !vm.buffer
  }
}

// -----

// Ref escapes via branch - no discard in source block.
// CHECK-LABEL: @ref_passed_to_successor
// CHECK-SAME: (%[[BUF:.*]]: !vm.buffer, %[[COND:.*]]: i32)
vm.module @my_module {
  vm.func @ref_passed_to_successor(%buf: !vm.buffer, %cond: i32) {
    // CHECK: vm.cond_br %[[COND]], ^[[BB1:.*]](%[[BUF]] : !vm.buffer), ^[[BB2:.*]]
    // CHECK-NOT: vm.discard.refs
    vm.cond_br %cond, ^bb1(%buf : !vm.buffer), ^bb2
  // CHECK: ^[[BB1]](%[[ARG:.*]]: !vm.buffer):
  ^bb1(%arg: !vm.buffer):
    // CHECK: vm.call @consume(%[[ARG]])
    // CHECK-NEXT: vm.discard.refs %[[ARG]] : !vm.buffer
    vm.call @consume(%arg) : (!vm.buffer) -> ()
    vm.br ^exit
  ^bb2:
    vm.br ^exit
  ^exit:
    vm.return
  }
  vm.import private @consume(%buf: !vm.buffer)
}

// -----

// Multiple refs dying at same point - batched into single discard.
// CHECK-LABEL: @multiple_refs_same_death_point
// CHECK-SAME: (%[[A:.*]]: !vm.buffer, %[[B:.*]]: !vm.buffer)
vm.module @my_module {
  vm.func @multiple_refs_same_death_point(%a: !vm.buffer, %b: !vm.buffer) {
    // CHECK: vm.call @consume2(%[[A]], %[[B]])
    // CHECK-NEXT: vm.discard.refs %[[A]], %[[B]] : !vm.buffer, !vm.buffer
    vm.call @consume2(%a, %b) : (!vm.buffer, !vm.buffer) -> ()
    vm.return
  }
  vm.import private @consume2(%a: !vm.buffer, %b: !vm.buffer)
}

// -----

// Idempotency - existing discard should not get duplicate.
// CHECK-LABEL: @already_discarded
// CHECK-SAME: (%[[BUF:.*]]: !vm.buffer)
vm.module @my_module {
  vm.func @already_discarded(%buf: !vm.buffer) {
    // CHECK: vm.call @consume(%[[BUF]])
    vm.call @consume(%buf) : (!vm.buffer) -> ()
    // CHECK-NEXT: vm.discard.refs %[[BUF]] : !vm.buffer
    // CHECK-NOT: vm.discard.refs
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }
  vm.import private @consume(%buf: !vm.buffer)
}

// -----

// Unused ref result - discard immediately after defining op.
// This was a leak case: ref-producing op with no uses never got discarded.
// CHECK-LABEL: @unused_ref_result
vm.module @my_module {
  vm.func @unused_ref_result() {
    // CHECK: %[[UNUSED:.*]] = vm.call @create_buffer()
    // CHECK-NEXT: vm.discard.refs %[[UNUSED]] : !vm.buffer
    %unused = vm.call @create_buffer() : () -> !vm.buffer
    vm.return
  }
  vm.import private @create_buffer() -> !vm.buffer
}

// -----

// Diamond CFG - ref used only on one path.
// This was a leak case: ref in liveOut but not used on all paths.
// The ref should be discarded at the start of the path where it's not used.
// CHECK-LABEL: @diamond_ref_one_path
// CHECK-SAME: (%[[BUF:.*]]: !vm.buffer, %[[COND:.*]]: i32)
vm.module @my_module {
  vm.func @diamond_ref_one_path(%buf: !vm.buffer, %cond: i32) {
    // CHECK: vm.cond_br %[[COND]], ^[[THEN:.*]], ^[[ELSE:.*]]
    vm.cond_br %cond, ^then, ^else
  // CHECK: ^[[THEN]]:
  ^then:
    // CHECK: vm.call @consume(%[[BUF]])
    // CHECK-NEXT: vm.discard.refs %[[BUF]] : !vm.buffer
    vm.call @consume(%buf) : (!vm.buffer) -> ()
    vm.br ^exit
  // CHECK: ^[[ELSE]]:
  ^else:
    // CHECK: vm.discard.refs %[[BUF]] : !vm.buffer
    // Ref dies here - not used on this path
    vm.br ^exit
  ^exit:
    vm.return
  }
  vm.import private @consume(%buf: !vm.buffer)
}

// -----

// Diamond CFG - ref used on both paths.
// Each path uses the ref, so no edge discards needed - just discards after use.
// CHECK-LABEL: @diamond_ref_both_paths
// CHECK-SAME: (%[[BUF:.*]]: !vm.buffer, %[[COND:.*]]: i32)
vm.module @my_module {
  vm.func @diamond_ref_both_paths(%buf: !vm.buffer, %cond: i32) {
    // CHECK: vm.cond_br %[[COND]], ^[[THEN:.*]], ^[[ELSE:.*]]
    vm.cond_br %cond, ^then, ^else
  // CHECK: ^[[THEN]]:
  ^then:
    // CHECK: vm.call @consume(%[[BUF]])
    // CHECK-NEXT: vm.discard.refs %[[BUF]] : !vm.buffer
    vm.call @consume(%buf) : (!vm.buffer) -> ()
    vm.br ^exit
  // CHECK: ^[[ELSE]]:
  ^else:
    // CHECK: vm.call @consume(%[[BUF]])
    // CHECK-NEXT: vm.discard.refs %[[BUF]] : !vm.buffer
    vm.call @consume(%buf) : (!vm.buffer) -> ()
    vm.br ^exit
  ^exit:
    vm.return
  }
  vm.import private @consume(%buf: !vm.buffer)
}

// -----

// Complex: unused ref result combined with diamond pattern.
// CHECK-LABEL: @unused_and_diamond
// CHECK-SAME: (%[[COND:.*]]: i32)
vm.module @my_module {
  vm.func @unused_and_diamond(%cond: i32) {
    // CHECK: %[[UNUSED:.*]] = vm.call @create_buffer()
    // CHECK-NEXT: vm.discard.refs %[[UNUSED]] : !vm.buffer
    %unused = vm.call @create_buffer() : () -> !vm.buffer
    // CHECK: %[[USED:.*]] = vm.call @create_buffer()
    %used = vm.call @create_buffer() : () -> !vm.buffer
    // CHECK: vm.cond_br %[[COND]], ^[[THEN:.*]], ^[[ELSE:.*]]
    vm.cond_br %cond, ^then, ^else
  // CHECK: ^[[THEN]]:
  ^then:
    // CHECK: vm.call @consume(%[[USED]])
    // CHECK-NEXT: vm.discard.refs %[[USED]] : !vm.buffer
    vm.call @consume(%used) : (!vm.buffer) -> ()
    vm.br ^exit
  // CHECK: ^[[ELSE]]:
  ^else:
    // Edge discard for %used which isn't used on this path.
    // CHECK: vm.discard.refs %[[USED]] : !vm.buffer
    vm.br ^exit
  ^exit:
    vm.return
  }
  vm.import private @create_buffer() -> !vm.buffer
  vm.import private @consume(%buf: !vm.buffer)
}

// -----

// Unused block argument - ref forwarded to block arg that is never used.
// This was a leak case: block args are not operations, so main loop missed them.
// CHECK-LABEL: @unused_block_arg
// CHECK-SAME: (%[[BUF:.*]]: !vm.buffer)
vm.module @my_module {
  vm.func @unused_block_arg(%buf: !vm.buffer) {
    // CHECK: vm.br ^[[BB1:.*]](%[[BUF]] : !vm.buffer)
    vm.br ^bb1(%buf : !vm.buffer)
  // CHECK: ^[[BB1]](%[[UNUSED_ARG:.*]]: !vm.buffer):
  // CHECK-NEXT: vm.discard.refs %[[UNUSED_ARG]] : !vm.buffer
  ^bb1(%unused_arg: !vm.buffer):
    // Block arg is never used - should be discarded at block entry.
    vm.return
  }
}

// -----

// Multiple predecessors - ref used on both paths before merge.
// This tests the last-use discard (not edge discard).
// CHECK-LABEL: @multi_pred_merge
// CHECK-SAME: (%[[BUF:.*]]: !vm.buffer, %[[COND:.*]]: i32)
vm.module @my_module {
  vm.func @multi_pred_merge(%buf: !vm.buffer, %cond: i32) {
    // CHECK: vm.cond_br %[[COND]], ^[[LEFT:.*]], ^[[RIGHT:.*]]
    vm.cond_br %cond, ^left, ^right
  // CHECK: ^[[LEFT]]:
  ^left:
    // CHECK: vm.call @consume(%[[BUF]])
    // CHECK-NEXT: vm.discard.refs %[[BUF]] : !vm.buffer
    vm.call @consume(%buf) : (!vm.buffer) -> ()
    vm.br ^merge
  // CHECK: ^[[RIGHT]]:
  ^right:
    // CHECK: vm.call @consume(%[[BUF]])
    // CHECK-NEXT: vm.discard.refs %[[BUF]] : !vm.buffer
    vm.call @consume(%buf) : (!vm.buffer) -> ()
    vm.br ^merge
  ^merge:
    vm.return
  }
  vm.import private @consume(%buf: !vm.buffer)
}

// -----

// Multiple predecessors - ref dies on edge from both.
// The pull model should insert ONE discard at merge, not two.
// Since ref is not used anywhere, it's discarded in entry block.
// CHECK-LABEL: @multi_pred_edge_dead
// CHECK-SAME: (%[[BUF:.*]]: !vm.buffer, %[[COND:.*]]: i32)
vm.module @my_module {
  vm.func @multi_pred_edge_dead(%buf: !vm.buffer, %cond: i32) {
    // The ref is never used, so it's discarded right at entry.
    // CHECK: vm.discard.refs %[[BUF]] : !vm.buffer
    // CHECK: vm.cond_br %[[COND]]
    vm.cond_br %cond, ^left, ^right
  ^left:
    // Ref not used here, flows through.
    vm.br ^merge
  ^right:
    // Ref not used here either.
    vm.br ^merge
  ^merge:
    vm.return
  }
}

// -----

// Loop with ref - ref used in loop body, dies after loop exit.
// CHECK-LABEL: @loop_ref_dies_after
// CHECK-SAME: (%[[BUF:.*]]: !vm.buffer, %[[N:.*]]: i32)
vm.module @my_module {
  vm.func @loop_ref_dies_after(%buf: !vm.buffer, %n: i32) {
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    vm.br ^loop(%c0 : i32)
  ^loop(%i: i32):
    vm.call @consume(%buf) : (!vm.buffer) -> ()
    %next = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %next, %n : i32
    vm.cond_br %cmp, ^loop(%next : i32), ^exit
  // Ref dies at exit - should be discarded here.
  // CHECK: ^bb2:
  // CHECK-NEXT: vm.discard.refs %[[BUF]] : !vm.buffer
  // CHECK-NEXT: vm.return
  ^exit:
    vm.return
  }
  vm.import private @consume(%buf: !vm.buffer)
}

// -----

// Each ref value has independent lifetime - casts produce new refs to discard.
// CHECK-LABEL: @cast_refs_independent_lifetimes
vm.module @my_module {
  vm.func @cast_refs_independent_lifetimes() {
    %c128 = vm.const.i64 128
    %c16 = vm.const.i32 16
    // CHECK: %[[BUFFER:.*]] = vm.buffer.alloc
    %buffer = vm.buffer.alloc %c128, %c16 : !vm.buffer
    // CHECK: %[[REF:.*]] = vm.cast.ref.any %[[BUFFER]]
    %ref = vm.cast.ref.any %buffer : !vm.buffer -> !vm.ref<?>
    // CHECK: %[[CAST:.*]] = vm.cast.any.ref %[[REF]]
    %cast = vm.cast.any.ref %ref : !vm.ref<?> -> !vm.buffer
    // %ref is discarded after its last use (vm.cast.any.ref)
    // CHECK: vm.discard.refs %[[REF]]
    // CHECK: vm.cmp.eq.ref %[[BUFFER]], %[[CAST]]
    %eq = vm.cmp.eq.ref %buffer, %cast : !vm.buffer
    // Both %buffer and %cast die at same point - batched discard
    // CHECK: vm.discard.refs %[[BUFFER]], %[[CAST]] : !vm.buffer, !vm.buffer
    vm.return
  }
}

// -----

// Each cast produces a new ref with independent lifetime.
// Refs are discarded after their last use.
// CHECK-LABEL: @chained_casts_independent
vm.module @my_module {
  vm.func @chained_casts_independent() {
    %c128 = vm.const.i64 128
    %c16 = vm.const.i32 16
    // CHECK: %[[BUF:.*]] = vm.buffer.alloc
    %buf = vm.buffer.alloc %c128, %c16 : !vm.buffer
    // CHECK: %[[REF1:.*]] = vm.cast.ref.any %[[BUF]]
    %ref1 = vm.cast.ref.any %buf : !vm.buffer -> !vm.ref<?>
    // CHECK: %[[BUF2:.*]] = vm.cast.any.ref %[[REF1]]
    %buf2 = vm.cast.any.ref %ref1 : !vm.ref<?> -> !vm.buffer
    // ref1's last use is vm.cast.any.ref, discard it now
    // CHECK: vm.discard.refs %[[REF1]]
    // CHECK: %[[REF2:.*]] = vm.cast.ref.any %[[BUF2]]
    %ref2 = vm.cast.ref.any %buf2 : !vm.buffer -> !vm.ref<?>
    // buf2's last use is vm.cast.ref.any, discard it now
    // CHECK: vm.discard.refs %[[BUF2]]
    // CHECK: vm.call @use_buffer(%[[BUF]])
    vm.call @use_buffer(%buf) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs %[[BUF]]
    // CHECK: vm.call @use_ref(%[[REF2]])
    vm.call @use_ref(%ref2) : (!vm.ref<?>) -> ()
    // CHECK: vm.discard.refs %[[REF2]]
    vm.return
  }
  vm.import private @use_buffer(%buf: !vm.buffer)
  vm.import private @use_ref(%ref: !vm.ref<?>)
}

// -----

// Each ref discarded after its last use, no aliasing.
// CHECK-LABEL: @ref_used_then_original_used
vm.module @my_module {
  vm.func @ref_used_then_original_used() {
    %c128 = vm.const.i64 128
    %c16 = vm.const.i32 16
    // CHECK: %[[BUF:.*]] = vm.buffer.alloc
    %buf = vm.buffer.alloc %c128, %c16 : !vm.buffer
    // CHECK: %[[REF:.*]] = vm.cast.ref.any %[[BUF]]
    %ref = vm.cast.ref.any %buf : !vm.buffer -> !vm.ref<?>
    // CHECK: vm.call @use_ref(%[[REF]])
    vm.call @use_ref(%ref) : (!vm.ref<?>) -> ()
    // ref's last use is use_ref, discard it
    // CHECK: vm.discard.refs %[[REF]]
    // CHECK: vm.call @use_buffer(%[[BUF]])
    vm.call @use_buffer(%buf) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs %[[BUF]]
    vm.return
  }
  vm.import private @use_buffer(%buf: !vm.buffer)
  vm.import private @use_ref(%ref: !vm.ref<?>)
}

// -----

// Ref used in branch, original used after merge - independent lifetimes.
// CHECK-LABEL: @ref_in_branch_original_after_merge
vm.module @my_module {
  vm.func @ref_in_branch_original_after_merge(%cond: i32) {
    %c128 = vm.const.i64 128
    %c16 = vm.const.i32 16
    // CHECK: %[[BUF:.*]] = vm.buffer.alloc
    %buf = vm.buffer.alloc %c128, %c16 : !vm.buffer
    // CHECK: %[[REF:.*]] = vm.cast.ref.any %[[BUF]]
    %ref = vm.cast.ref.any %buf : !vm.buffer -> !vm.ref<?>
    // CHECK: vm.cond_br
    vm.cond_br %cond, ^left, ^right
  ^left:
    // CHECK: vm.call @use_ref(%[[REF]])
    vm.call @use_ref(%ref) : (!vm.ref<?>) -> ()
    // ref's last use on this path
    // CHECK: vm.discard.refs %[[REF]]
    vm.br ^merge
  ^right:
    // ref not used on this path - edge discard
    // CHECK: vm.discard.refs %[[REF]]
    vm.br ^merge
  ^merge:
    // CHECK: vm.call @use_buffer(%[[BUF]])
    vm.call @use_buffer(%buf) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs %[[BUF]]
    vm.return
  }
  vm.import private @use_buffer(%buf: !vm.buffer)
  vm.import private @use_ref(%ref: !vm.ref<?>)
}

// -----

// Ref used in loop, original used after loop exit - independent lifetimes.
// CHECK-LABEL: @ref_in_loop_original_after
vm.module @my_module {
  vm.func @ref_in_loop_original_after(%n: i32) {
    %c128 = vm.const.i64 128
    %c16 = vm.const.i32 16
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    // CHECK: %[[BUF:.*]] = vm.buffer.alloc
    %buf = vm.buffer.alloc %c128, %c16 : !vm.buffer
    // CHECK: %[[REF:.*]] = vm.cast.ref.any %[[BUF]]
    %ref = vm.cast.ref.any %buf : !vm.buffer -> !vm.ref<?>
    vm.br ^loop(%c0 : i32)
  ^loop(%i: i32):
    // CHECK: vm.call @use_ref(%[[REF]])
    vm.call @use_ref(%ref) : (!vm.ref<?>) -> ()
    %next = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %next, %n : i32
    vm.cond_br %cmp, ^loop(%next : i32), ^exit
  ^exit:
    // ref is live throughout loop, dies at exit
    // CHECK: vm.discard.refs %[[REF]]
    // CHECK: vm.call @use_buffer(%[[BUF]])
    vm.call @use_buffer(%buf) : (!vm.buffer) -> ()
    // CHECK: vm.discard.refs %[[BUF]]
    vm.return
  }
  vm.import private @use_buffer(%buf: !vm.buffer)
  vm.import private @use_ref(%ref: !vm.ref<?>)
}

// -----

//===----------------------------------------------------------------------===//
// Cross-block SSA use tests (refs used in successors via direct dominance,
// not passed as block arguments). These test the liveness analysis's ability
// to track values that are live across block boundaries without being
// explicitly forwarded.
//===----------------------------------------------------------------------===//

// Cross-block SSA use - ref used in successor via direct dominance (no block arg).
// This is the key bug case: ref must NOT be discarded before cond_br.
// CHECK-LABEL: @cross_block_ssa_use_cond_br
vm.module @my_module {
  vm.rodata private @data dense<[1, 2, 3]> : tensor<3xi32>
  vm.func @cross_block_ssa_use_cond_br(%cond: i32) {
    // CHECK: %[[REF:.*]] = vm.const.ref.rodata @data
    %ref = vm.const.ref.rodata @data : !vm.buffer
    // CHECK-NOT: vm.discard.refs
    // CHECK: vm.cond_br %{{.*}}, ^[[BB1:.*]], ^[[BB2:.*]]
    vm.cond_br %cond, ^bb1, ^bb2
  // CHECK: ^[[BB1]]:
  ^bb1:
    // CHECK: vm.cmp.nz.ref %[[REF]]
    // CHECK-NEXT: vm.discard.refs %[[REF]]
    %nz = vm.cmp.nz.ref %ref : !vm.buffer
    vm.return
  // CHECK: ^[[BB2]]:
  ^bb2:
    // Ref not used here - discard at block entry
    // CHECK: vm.discard.refs %[[REF]]
    vm.return
  }
}

// -----

// Cross-block SSA use with unconditional branch.
// CHECK-LABEL: @cross_block_ssa_use_br
vm.module @my_module {
  vm.rodata private @data dense<[1, 2, 3]> : tensor<3xi32>
  vm.func @cross_block_ssa_use_br() {
    // CHECK: %[[REF:.*]] = vm.const.ref.rodata @data
    %ref = vm.const.ref.rodata @data : !vm.buffer
    // CHECK-NOT: vm.discard.refs
    // CHECK: vm.br ^[[BB1:.*]]
    vm.br ^bb1
  // CHECK: ^[[BB1]]:
  ^bb1:
    // CHECK: vm.cmp.nz.ref %[[REF]]
    // CHECK-NEXT: vm.discard.refs %[[REF]]
    %nz = vm.cmp.nz.ref %ref : !vm.buffer
    vm.return
  }
}

// -----

// Ref used in loop body without being loop-carried (via back-edge block arg).
// The ref dominates the loop and is used each iteration.
// CHECK-LABEL: @ref_used_in_loop_not_carried
vm.module @my_module {
  vm.rodata private @data dense<[1, 2, 3]> : tensor<3xi32>
  vm.func @ref_used_in_loop_not_carried() {
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    %c3 = vm.const.i32 3
    // CHECK: %[[REF:.*]] = vm.const.ref.rodata @data
    %ref = vm.const.ref.rodata @data : !vm.buffer
    // CHECK-NOT: vm.discard.refs
    // CHECK: vm.br ^[[LOOP:.*]](
    vm.br ^loop(%c0 : i32)
  // CHECK: ^[[LOOP]](
  ^loop(%i: i32):
    // CHECK: vm.cmp.nz.ref %[[REF]]
    %nz = vm.cmp.nz.ref %ref : !vm.buffer
    %next = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %next, %c3 : i32
    // CHECK-NOT: vm.discard.refs
    // CHECK: vm.cond_br
    vm.cond_br %cmp, ^loop(%next : i32), ^exit
  // CHECK: ^[[EXIT:.*]]:
  ^exit:
    // Ref dies here after loop exits
    // CHECK: vm.discard.refs %[[REF]]
    // CHECK-NEXT: vm.return
    vm.return
  }
}

// -----

// Multiple refs with different cross-block lifetimes.
// ref_a used only in bb1, ref_b used only in bb2.
// CHECK-LABEL: @multiple_refs_different_cross_block_paths
vm.module @my_module {
  vm.rodata private @data_a dense<[1, 2, 3]> : tensor<3xi32>
  vm.rodata private @data_b dense<[4, 5, 6]> : tensor<3xi32>
  vm.func @multiple_refs_different_cross_block_paths(%cond: i32) {
    // CHECK-DAG: %[[REF_A:.*]] = vm.const.ref.rodata @data_a
    // CHECK-DAG: %[[REF_B:.*]] = vm.const.ref.rodata @data_b
    %ref_a = vm.const.ref.rodata @data_a : !vm.buffer
    %ref_b = vm.const.ref.rodata @data_b : !vm.buffer
    // CHECK-NOT: vm.discard.refs
    // CHECK: vm.cond_br
    vm.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    // ref_b not used here - discard it
    // ref_a used here - discard after use
    // CHECK: vm.discard.refs %[[REF_B]]
    // CHECK: vm.cmp.nz.ref %[[REF_A]]
    // CHECK-NEXT: vm.discard.refs %[[REF_A]]
    %nz_a = vm.cmp.nz.ref %ref_a : !vm.buffer
    vm.return
  ^bb2:
    // ref_a not used here - discard it
    // ref_b used here - discard after use
    // CHECK: vm.discard.refs %[[REF_A]]
    // CHECK: vm.cmp.nz.ref %[[REF_B]]
    // CHECK-NEXT: vm.discard.refs %[[REF_B]]
    %nz_b = vm.cmp.nz.ref %ref_b : !vm.buffer
    vm.return
  }
}

// -----

// Ref used in multiple successor blocks (both paths of diamond).
// CHECK-LABEL: @cross_block_ssa_both_paths
vm.module @my_module {
  vm.rodata private @data dense<[1, 2, 3]> : tensor<3xi32>
  vm.func @cross_block_ssa_both_paths(%cond: i32) {
    // CHECK: %[[REF:.*]] = vm.const.ref.rodata @data
    %ref = vm.const.ref.rodata @data : !vm.buffer
    // CHECK-NOT: vm.discard.refs
    // CHECK: vm.cond_br
    vm.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    // CHECK: vm.cmp.nz.ref %[[REF]]
    // CHECK-NEXT: vm.discard.refs %[[REF]]
    %nz1 = vm.cmp.nz.ref %ref : !vm.buffer
    vm.br ^exit
  ^bb2:
    // CHECK: vm.cmp.nz.ref %[[REF]]
    // CHECK-NEXT: vm.discard.refs %[[REF]]
    %nz2 = vm.cmp.nz.ref %ref : !vm.buffer
    vm.br ^exit
  ^exit:
    vm.return
  }
}

// -----

// Nested control flow - ref used in deeply nested block.
// CHECK-LABEL: @cross_block_ssa_nested
vm.module @my_module {
  vm.rodata private @data dense<[1, 2, 3]> : tensor<3xi32>
  vm.func @cross_block_ssa_nested(%cond1: i32, %cond2: i32) {
    // CHECK: %[[REF:.*]] = vm.const.ref.rodata @data
    %ref = vm.const.ref.rodata @data : !vm.buffer
    // CHECK-NOT: vm.discard.refs
    // CHECK: vm.cond_br
    vm.cond_br %cond1, ^outer_then, ^outer_else
  ^outer_then:
    // CHECK-NOT: vm.discard.refs %[[REF]]
    // CHECK: vm.cond_br
    vm.cond_br %cond2, ^inner_then, ^inner_else
  ^inner_then:
    // CHECK: vm.cmp.nz.ref %[[REF]]
    // CHECK-NEXT: vm.discard.refs %[[REF]]
    %nz = vm.cmp.nz.ref %ref : !vm.buffer
    vm.return
  ^inner_else:
    // CHECK: vm.discard.refs %[[REF]]
    vm.return
  ^outer_else:
    // CHECK: vm.discard.refs %[[REF]]
    vm.return
  }
}

// -----

// Ref defined in predecessor, used across multiple hops.
// entry -> bb1 -> bb2, ref defined in entry, used in bb2.
// CHECK-LABEL: @cross_block_ssa_multi_hop
vm.module @my_module {
  vm.rodata private @data dense<[1, 2, 3]> : tensor<3xi32>
  vm.func @cross_block_ssa_multi_hop() {
    // CHECK: %[[REF:.*]] = vm.const.ref.rodata @data
    %ref = vm.const.ref.rodata @data : !vm.buffer
    // CHECK-NOT: vm.discard.refs
    // CHECK: vm.br ^[[BB1:.*]]
    vm.br ^bb1
  // CHECK: ^[[BB1]]:
  ^bb1:
    // ref not used here, flows through
    // CHECK-NOT: vm.discard.refs
    // CHECK: vm.br ^[[BB2:.*]]
    vm.br ^bb2
  // CHECK: ^[[BB2]]:
  ^bb2:
    // CHECK: vm.cmp.nz.ref %[[REF]]
    // CHECK-NEXT: vm.discard.refs %[[REF]]
    %nz = vm.cmp.nz.ref %ref : !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Partial-edge death tests: ref forwarded on SOME edges but not ALL.
// These test the case where a ref is passed as a branch argument on some
// successor edges but not others. The ref must be discarded on the edges
// where it's NOT forwarded.
//===----------------------------------------------------------------------===//

// Loop-carried ref dies on exit edge.
// The ref is forwarded on the back-edge to ^loop, but NOT to ^exit.
// A discard must be inserted on the exit edge.
// CHECK-LABEL: @loop_carried_ref_dies_on_exit
vm.module @my_module {
  vm.rodata private @data dense<[1, 2, 3]> : tensor<3xi32>
  vm.func @loop_carried_ref_dies_on_exit(%n: i32) {
    %ref = vm.const.ref.rodata @data : !vm.buffer
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    // CHECK: vm.br ^[[LOOP:.*]](%{{.*}}, %[[REF:.*]] : i32, !vm.buffer)
    vm.br ^loop(%c0, %ref : i32, !vm.buffer)
  // CHECK: ^[[LOOP]](%[[I:.*]]: i32, %[[LOOP_REF:.*]]: !vm.buffer):
  ^loop(%i: i32, %loop_ref: !vm.buffer):
    // Verify the ref is valid each iteration.
    // CHECK: vm.cmp.nz.ref %[[LOOP_REF]]
    %nz = vm.cmp.nz.ref %loop_ref : !vm.buffer
    %next = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %next, %n : i32
    // Ref forwarded on back-edge, NOT forwarded on exit edge.
    // CHECK: vm.cond_br %{{.*}}, ^[[LOOP]](%{{.*}}, %[[LOOP_REF]] : i32, !vm.buffer), ^[[EXIT:.*]]
    vm.cond_br %cmp, ^loop(%next, %loop_ref : i32, !vm.buffer), ^exit
  // CHECK: ^[[EXIT]]:
  ^exit:
    // Ref dies here - must be discarded.
    // CHECK-NEXT: vm.discard.refs %[[LOOP_REF]] : !vm.buffer
    // CHECK-NEXT: vm.return
    vm.return
  }
}

// -----

// Multiple loop-carried refs die on exit edge - batched discard.
// CHECK-LABEL: @multiple_loop_carried_refs_die_on_exit
vm.module @my_module {
  vm.rodata private @data_a dense<[1, 2, 3]> : tensor<3xi32>
  vm.rodata private @data_b dense<[4, 5, 6]> : tensor<3xi32>
  vm.func @multiple_loop_carried_refs_die_on_exit(%n: i32) {
    %ref_a = vm.const.ref.rodata @data_a : !vm.buffer
    %ref_b = vm.const.ref.rodata @data_b : !vm.buffer
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    // CHECK: vm.br ^[[LOOP:.*]](%{{.*}}, %[[REF_A:.*]], %[[REF_B:.*]] : i32, !vm.buffer, !vm.buffer)
    vm.br ^loop(%c0, %ref_a, %ref_b : i32, !vm.buffer, !vm.buffer)
  // CHECK: ^[[LOOP]](%[[I:.*]]: i32, %[[LOOP_A:.*]]: !vm.buffer, %[[LOOP_B:.*]]: !vm.buffer):
  ^loop(%i: i32, %loop_a: !vm.buffer, %loop_b: !vm.buffer):
    // CHECK: vm.cmp.nz.ref %[[LOOP_A]]
    %nz_a = vm.cmp.nz.ref %loop_a : !vm.buffer
    // CHECK: vm.cmp.nz.ref %[[LOOP_B]]
    %nz_b = vm.cmp.nz.ref %loop_b : !vm.buffer
    %next = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %next, %n : i32
    // CHECK: vm.cond_br %{{.*}}, ^[[LOOP]](%{{.*}}, %[[LOOP_A]], %[[LOOP_B]] : i32, !vm.buffer, !vm.buffer), ^[[EXIT:.*]]
    vm.cond_br %cmp, ^loop(%next, %loop_a, %loop_b : i32, !vm.buffer, !vm.buffer), ^exit
  // CHECK: ^[[EXIT]]:
  ^exit:
    // Both refs die here - batched discard (order may vary).
    // CHECK-NEXT: vm.discard.refs %{{.*}}, %{{.*}} : !vm.buffer, !vm.buffer
    // CHECK-NEXT: vm.return
    vm.return
  }
}

// -----

// Ref forwarded on one cond_br path, not the other (not a loop).
// CHECK-LABEL: @ref_forwarded_one_path_only
vm.module @my_module {
  vm.rodata private @data dense<[1, 2, 3]> : tensor<3xi32>
  vm.func @ref_forwarded_one_path_only(%cond: i32) {
    // CHECK: %[[REF:.*]] = vm.const.ref.rodata @data
    %ref = vm.const.ref.rodata @data : !vm.buffer
    // Ref forwarded to ^use_ref, NOT forwarded to ^no_ref.
    // CHECK: vm.cond_br %{{.*}}, ^[[USE:.*]](%[[REF]] : !vm.buffer), ^[[NO_USE:.*]]
    vm.cond_br %cond, ^use_ref(%ref : !vm.buffer), ^no_ref
  // CHECK: ^[[USE]](%[[ARG:.*]]: !vm.buffer):
  ^use_ref(%arg: !vm.buffer):
    // CHECK: vm.cmp.nz.ref %[[ARG]]
    // CHECK-NEXT: vm.discard.refs %[[ARG]] : !vm.buffer
    %nz = vm.cmp.nz.ref %arg : !vm.buffer
    vm.return
  // CHECK: ^[[NO_USE]]:
  ^no_ref:
    // Ref dies here because it wasn't forwarded.
    // CHECK-NEXT: vm.discard.refs %[[REF]] : !vm.buffer
    // CHECK-NEXT: vm.return
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// vm.br_table with refs
//===----------------------------------------------------------------------===//

// br_table: ref forwarded to some cases, not others.
// The ref needs a discard on branches that don't receive it.
// CHECK-LABEL: @br_table_ref_partial_forward
vm.module @my_module {
  vm.rodata private @data dense<[1, 2, 3]> : tensor<3xi32>
  vm.func @br_table_ref_partial_forward(%idx: i32) {
    // CHECK: %[[REF:.*]] = vm.const.ref.rodata @data
    %ref = vm.const.ref.rodata @data : !vm.buffer
    // CHECK: vm.br_table
    vm.br_table %idx {
      default: ^bb_default(%ref : !vm.buffer),
      0: ^bb0(%ref : !vm.buffer),
      1: ^bb1
    }
  // CHECK: ^bb1(%[[ARG0:.*]]: !vm.buffer):
  ^bb_default(%arg0: !vm.buffer):
    // CHECK: vm.cmp.nz.ref %[[ARG0]]
    // CHECK-NEXT: vm.discard.refs %[[ARG0]]
    %nz0 = vm.cmp.nz.ref %arg0 : !vm.buffer
    vm.return
  // CHECK: ^bb2(%[[ARG1:.*]]: !vm.buffer):
  ^bb0(%arg1: !vm.buffer):
    // CHECK: vm.cmp.nz.ref %[[ARG1]]
    // CHECK-NEXT: vm.discard.refs %[[ARG1]]
    %nz1 = vm.cmp.nz.ref %arg1 : !vm.buffer
    vm.return
  // CHECK: ^bb3:
  ^bb1:
    // Ref not forwarded here - needs discard.
    // CHECK-NEXT: vm.discard.refs %[[REF]]
    // CHECK-NEXT: vm.return
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// vm.call.yieldable with refs - terminators that use refs as call operands
//===----------------------------------------------------------------------===//

// Yieldable call with ref operands.
// This was the actual bug: refs passed to vm.call.yieldable were discarded
// BEFORE the call because mid-block discards were inserting before terminators.
// The ref must NOT be discarded before the call - it's used by the call.
// CHECK-LABEL: @yieldable_call_ref_operands
vm.module @my_module {
  vm.import private @consume(%buf: !vm.buffer) -> !vm.buffer attributes {vm.yield}
  vm.func @yieldable_call_ref_operands(%buf: !vm.buffer) -> !vm.buffer {
    // CHECK-NOT: vm.discard.refs
    // CHECK: vm.call.yieldable @consume({{.*}}) : (!vm.buffer) -> ^bb1 (!vm.buffer)
    vm.call.yieldable @consume(%buf) : (!vm.buffer) -> ^done(!vm.buffer)
  // CHECK: ^bb1(%[[RESULT:.*]]: !vm.buffer):
  ^done(%result: !vm.buffer):
    // CHECK: vm.return %[[RESULT]]
    vm.return %result : !vm.buffer
  }
}

// -----

// Ref used before yieldable call, then passed to call.
// The ref must NOT be discarded after the first use since it's still needed.
// CHECK-LABEL: @ref_used_then_passed_to_yieldable
vm.module @my_module {
  vm.import private @consume(%buf: !vm.buffer) -> !vm.buffer attributes {vm.yield}
  vm.func @ref_used_then_passed_to_yieldable(%buf: !vm.buffer) -> !vm.buffer {
    // CHECK: vm.cmp.nz.ref
    %nz = vm.cmp.nz.ref %buf : !vm.buffer
    // Ref NOT discarded after cmp.nz.ref - it's used by the call below.
    // CHECK-NOT: vm.discard.refs
    // CHECK: vm.call.yieldable @consume({{.*}}) : (!vm.buffer) -> ^bb1 (!vm.buffer)
    vm.call.yieldable @consume(%buf) : (!vm.buffer) -> ^done(!vm.buffer)
  ^done(%result: !vm.buffer):
    vm.return %result : !vm.buffer
  }
}

// -----

//===----------------------------------------------------------------------===//
// Irreducible CFG with refs
//===----------------------------------------------------------------------===//

// Irreducible CFG: ^A and ^B can each reach the other, with external entry
// to both. Refs die on exit edges (critical edges split into new blocks).
// CHECK-LABEL: @irreducible_ref_discards
vm.module @my_module {
  vm.rodata private @data dense<[1, 2, 3]> : tensor<3xi32>
  vm.func @irreducible_ref_discards(%cond1: i32, %cond2: i32) {
    // CHECK: %[[REF:.*]] = vm.const.ref.rodata @data
    %ref = vm.const.ref.rodata @data : !vm.buffer
    // CHECK: vm.cond_br %{{.*}}, ^[[A:.*]](%[[REF]] : !vm.buffer), ^[[B:.*]](%[[REF]] : !vm.buffer)
    vm.cond_br %cond1, ^A(%ref : !vm.buffer), ^B(%ref : !vm.buffer)
  // CHECK: ^[[A]](%[[REF_A:.*]]: !vm.buffer):
  ^A(%ref_a: !vm.buffer):
    // CHECK: vm.cmp.nz.ref %[[REF_A]]
    %nz_a = vm.cmp.nz.ref %ref_a : !vm.buffer
    // A -> B forwards ref, A -> exit needs discard (via critical edge split).
    // CHECK: vm.cond_br %{{.*}}, ^[[B]](%[[REF_A]] : !vm.buffer), ^[[SPLIT_A:.*]]
    vm.cond_br %cond2, ^B(%ref_a : !vm.buffer), ^exit
  // CHECK: ^[[B]](%[[REF_B:.*]]: !vm.buffer):
  ^B(%ref_b: !vm.buffer):
    // CHECK: vm.cmp.nz.ref %[[REF_B]]
    %nz_b = vm.cmp.nz.ref %ref_b : !vm.buffer
    // B -> A forwards ref, B -> exit needs discard (via critical edge split).
    // CHECK: vm.cond_br %{{.*}}, ^[[A]](%[[REF_B]] : !vm.buffer), ^[[SPLIT_B:.*]]
    vm.cond_br %cond2, ^A(%ref_b : !vm.buffer), ^exit
  // Critical edge splits: discard then branch to exit.
  // CHECK: ^[[SPLIT_A]]:
  // CHECK-NEXT: vm.discard.refs %[[REF_A]] : !vm.buffer
  // CHECK-NEXT: vm.br ^[[EXIT:.*]]
  // CHECK: ^[[SPLIT_B]]:
  // CHECK-NEXT: vm.discard.refs %[[REF_B]] : !vm.buffer
  // CHECK-NEXT: vm.br ^[[EXIT]]
  // CHECK: ^[[EXIT]]:
  // CHECK-NEXT: vm.return
  ^exit:
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// MOVE semantics for yieldable calls - comprehensive tests
//===----------------------------------------------------------------------===//

// Multiple ref operands to yieldable call - all have MOVE semantics.
// None should be discarded; callee takes ownership of all.
// CHECK-LABEL: @yieldable_multiple_ref_operands
vm.module @my_module {
  vm.import private @multi(%a: !vm.buffer, %b: !vm.buffer) -> !vm.buffer attributes {vm.yield}
  vm.func @yieldable_multiple_ref_operands(%buf1: !vm.buffer, %buf2: !vm.buffer) -> !vm.buffer {
    // Use both refs before the call to ensure they're live.
    // CHECK: vm.cmp.nz.ref
    %nz1 = vm.cmp.nz.ref %buf1 : !vm.buffer
    // CHECK: vm.cmp.nz.ref
    %nz2 = vm.cmp.nz.ref %buf2 : !vm.buffer
    // Neither ref should be discarded - both are passed to yieldable call.
    // CHECK-NOT: vm.discard.refs
    // CHECK: vm.call.yieldable @multi(%{{.*}}, %{{.*}}) : (!vm.buffer, !vm.buffer) -> ^bb1 (!vm.buffer)
    vm.call.yieldable @multi(%buf1, %buf2) : (!vm.buffer, !vm.buffer) -> ^done(!vm.buffer)
  ^done(%result: !vm.buffer):
    vm.return %result : !vm.buffer
  }
}

// -----

// Variadic yieldable call with ref operands - MOVE semantics applies.
// CHECK-LABEL: @variadic_yieldable_ref_operands
vm.module @my_module {
  vm.import private @variadic(%a: !vm.buffer, %b: !vm.buffer) -> !vm.buffer attributes {vm.yield}
  vm.func @variadic_yieldable_ref_operands(%buf1: !vm.buffer, %buf2: !vm.buffer) -> !vm.buffer {
    // CHECK: vm.cmp.nz.ref
    %nz1 = vm.cmp.nz.ref %buf1 : !vm.buffer
    // CHECK: vm.cmp.nz.ref
    %nz2 = vm.cmp.nz.ref %buf2 : !vm.buffer
    // Neither ref should be discarded - both are passed to variadic yieldable call.
    // CHECK-NOT: vm.discard.refs
    // CHECK: vm.call.variadic.yieldable @variadic(%{{.*}}, %{{.*}})
    vm.call.variadic.yieldable @variadic(%buf1, %buf2) {segment_sizes = dense<[1, 1]> : vector<2xi16>, segment_types = [!vm.buffer, !vm.buffer]} : (!vm.buffer, !vm.buffer) -> ^done(!vm.buffer)
  ^done(%result: !vm.buffer):
    vm.return %result : !vm.buffer
  }
}

// -----

// Ref NOT passed to call but live across it - needs discard AFTER call completes.
// This tests the case where a ref is live-out past a yieldable call.
// CHECK-LABEL: @ref_live_across_yieldable
vm.module @my_module {
  vm.import private @compute(%x: i32) -> i32 attributes {vm.yield}
  vm.func @ref_live_across_yieldable(%x: i32) {
    // CHECK: %[[BUF:.*]] = vm.const.ref.rodata
    %buf = vm.const.ref.rodata @data : !vm.buffer
    // Ref is not passed to call, lives past it.
    // CHECK: vm.call.yieldable @compute(%{{.*}}) : (i32) -> ^bb1 (i32)
    vm.call.yieldable @compute(%x) : (i32) -> ^done(i32)
  // CHECK: ^bb1(%{{.*}}: i32):
  ^done(%result: i32):
    // Ref is used after call completes.
    // CHECK: vm.cmp.nz.ref %[[BUF]]
    %nz = vm.cmp.nz.ref %buf : !vm.buffer
    // Discard after last use.
    // CHECK-NEXT: vm.discard.refs %[[BUF]]
    // CHECK-NEXT: vm.return
    vm.return
  }
  vm.rodata private @data dense<[1, 2, 3]> : tensor<3xi32>
}

// -----

// Unused ref result from yieldable call - needs discard.
// CHECK-LABEL: @yieldable_unused_ref_result
vm.module @my_module {
  vm.import private @produce(%x: i32) -> !vm.buffer attributes {vm.yield}
  vm.func @yieldable_unused_ref_result(%x: i32) {
    // CHECK: vm.call.yieldable @produce(%{{.*}}) : (i32) -> ^bb1 (!vm.buffer)
    vm.call.yieldable @produce(%x) : (i32) -> ^done(!vm.buffer)
  // CHECK: ^bb1(%[[RESULT:.*]]: !vm.buffer):
  ^done(%result: !vm.buffer):
    // Unused ref result needs discard.
    // CHECK-NEXT: vm.discard.refs %[[RESULT]]
    // CHECK-NEXT: vm.return
    vm.return
  }
}
