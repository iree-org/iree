// RUN: iree-opt --split-input-file --iree-util-ipo %s | FileCheck %s

// Tests that unused args get dropped.

// CHECK-LABEL: util.func private @unused_arg_callee
// CHECK-SAME: (%[[ARG1:.+]]: index) -> index
util.func private @unused_arg_callee(%arg0: index, %arg1: index) -> index {
  // CHECK: %[[ADD:.+]] = arith.addi %[[ARG1]], %[[ARG1]]
  %add = arith.addi %arg1, %arg1 : index
  // CHECK: util.return %[[ADD]]
  util.return %add : index
}

// CHECK: util.func public @unused_arg_caller_a(%[[A_ARG0:.+]]: index, %[[A_ARG1:.+]]: index)
util.func public @unused_arg_caller_a(%arg0: index, %arg1: index) -> (index, index) {
  // CHECK: %[[A_RET0:.+]] = util.call @unused_arg_callee(%[[A_ARG0]]) : (index) -> index
  %ret0 = util.call @unused_arg_callee(%arg0, %arg0) : (index, index) -> index
  // CHECK: %[[A_RET1:.+]] = util.call @unused_arg_callee(%[[A_ARG1]]) : (index) -> index
  %ret1 = util.call @unused_arg_callee(%arg0, %arg1) : (index, index) -> index
  // CHECK: util.return %[[A_RET0]], %[[A_RET1]]
  util.return %ret0, %ret1 : index, index
}
// CHECK: util.func public @unused_arg_caller_b(%[[B_ARG0:.+]]: index)
util.func public @unused_arg_caller_b(%arg0: index) -> index {
  // CHECK: %[[B_RET0:.+]] = util.call @unused_arg_callee(%[[B_ARG0]]) : (index) -> index
  %ret0 = util.call @unused_arg_callee(%arg0, %arg0) : (index, index) -> index
  // CHECK: util.return %[[B_RET0]]
  util.return %ret0 : index
}

// -----

// Tests that uniformly unused results get dropped.

// CHECK-LABEL: util.func private @unused_result_callee
// CHECK-SAME: (%[[ARG0:.+]]: index, %[[ARG1:.+]]: index) -> index
util.func private @unused_result_callee(%arg0: index, %arg1: index) -> (index, index) {
  // CHECK: %[[ADD0:.+]] = arith.addi %[[ARG0]], %[[ARG1]]
  %add0 = arith.addi %arg0, %arg1 : index
  // CHECK: %[[ADD1:.+]] = arith.addi %[[ADD0]], %[[ARG0]]
  %add1 = arith.addi %add0, %arg0 : index
  // CHECK: util.return %[[ADD1]]
  util.return %add0, %add1 : index, index
}

// CHECK: util.func public @unused_result_caller_a(%[[A_ARG0:.+]]: index, %[[A_ARG1:.+]]: index)
util.func public @unused_result_caller_a(%arg0: index, %arg1: index) -> index {
  // CHECK: %[[A_RET1:.+]] = util.call @unused_result_callee(%[[A_ARG0]], %[[A_ARG1]]) : (index, index) -> index
  %ret:2 = util.call @unused_result_callee(%arg0, %arg1) : (index, index) -> (index, index)
  // CHECK: util.return %[[A_RET1]]
  util.return %ret#1 : index
}
// CHECK: util.func public @unused_result_caller_b(%[[B_ARG0:.+]]: index, %[[B_ARG1:.+]]: index)
util.func public @unused_result_caller_b(%arg0: index, %arg1: index) -> index {
  // CHECK: %[[B_RET1:.+]] = util.call @unused_result_callee(%[[B_ARG0]], %[[B_ARG1]]) : (index, index) -> index
  %ret:2 = util.call @unused_result_callee(%arg0, %arg1) : (index, index) -> (index, index)
  // CHECK: util.return %[[B_RET1]]
  util.return %ret#1 : index
}
// CHECK: util.func public @unused_result_caller_c(%[[C_ARG0:.+]]: index, %[[C_ARG1:.+]]: index)
util.func public @unused_result_caller_c(%arg0: index, %arg1: index) {
  // CHECK: %[[C_RET1:.+]] = util.call @unused_result_callee(%[[C_ARG0]], %[[C_ARG1]]) : (index, index) -> index
  %ret:2 = util.call @unused_result_callee(%arg0, %arg1) : (index, index) -> (index, index)
  // CHECK: util.return
  util.return
}

// -----

// Tests that uniformly duplicate args get combined.

// CHECK-LABEL: util.func private @dupe_arg_callee
// CHECK-SAME: (%[[ARG0:.+]]: index, %[[ARG1:.+]]: index) -> index
util.func private @dupe_arg_callee(%arg0: index, %arg1: index, %arg0_dupe: index) -> index {
  // CHECK: %[[ADD0:.+]] = arith.addi %[[ARG0]], %[[ARG1]]
  %add0 = arith.addi %arg0, %arg1 : index
  // CHECK: %[[ADD1:.+]] = arith.addi %[[ADD0]], %[[ARG0]]
  %add1 = arith.addi %add0, %arg0_dupe : index
  // CHECK: util.return %[[ADD1]]
  util.return %add1 : index
}

// CHECK: util.func public @dupe_arg_caller_a(%[[A_ARG0:.+]]: index, %[[A_ARG1:.+]]: index)
util.func public @dupe_arg_caller_a(%arg0: index, %arg1: index) -> (index, index) {
  // CHECK: %[[A_RET0:.+]] = util.call @dupe_arg_callee(%[[A_ARG0]], %[[A_ARG0]]) : (index, index) -> index
  %ret0 = util.call @dupe_arg_callee(%arg0, %arg0, %arg0) : (index, index, index) -> index
  // CHECK: %[[A_RET1:.+]] = util.call @dupe_arg_callee(%[[A_ARG0]], %[[A_ARG1]]) : (index, index) -> index
  %ret1 = util.call @dupe_arg_callee(%arg0, %arg1, %arg0) : (index, index, index) -> index
  // CHECK: util.return %[[A_RET0]], %[[A_RET1]]
  util.return %ret0, %ret1 : index, index
}
// CHECK: util.func public @dupe_arg_caller_b(%[[B_ARG0:.+]]: index)
util.func public @dupe_arg_caller_b(%arg0: index) -> index {
  // CHECK: %[[B_RET0:.+]] = util.call @dupe_arg_callee(%[[B_ARG0]], %[[B_ARG0]]) : (index, index) -> index
  %ret0 = util.call @dupe_arg_callee(%arg0, %arg0, %arg0) : (index, index, index) -> index
  // CHECK: util.return %[[B_RET0]]
  util.return %ret0 : index
}

// -----

// Tests that duplicate arguments that point at a base unused argument ensure
// that base argument stays live. Note that %arg0 is not used in the callee
// but a duplicate of it is.

// CHECK-LABEL: util.func private @dupe_unused_arg_callee
// CHECK-SAME: (%[[CALLEE_ARG0:.+]]: index) -> index
util.func private @dupe_unused_arg_callee(%arg0: index, %arg0_dupe: index) -> (index, index) {
  // CHECK: %[[CALLEE_RET0:.+]] = arith.addi %[[CALLEE_ARG0]], %[[CALLEE_ARG0]]
  %ret0 = arith.addi %arg0_dupe, %arg0_dupe : index
  // CHECK: util.return %[[CALLEE_RET0]]
  util.return %ret0, %arg0 : index, index
}

// CHECK: util.func public @dupe_unused_arg_caller(%[[CALLER_ARG0:.+]]: index)
util.func public @dupe_unused_arg_caller(%arg0: index) -> (index, index) {
  // CHECK: %[[CALLER_RET0:.+]] = util.call @dupe_unused_arg_callee(%[[CALLER_ARG0]]) : (index) -> index
  %ret:2 = util.call @dupe_unused_arg_callee(%arg0, %arg0) : (index, index) -> (index, index)
  // CHECK: util.return %[[CALLER_RET0]], %[[CALLER_ARG0]]
  util.return %ret#0, %ret#1 : index, index
}

// -----

// Tests that uniformly duplicate results get combined.

// CHECK-LABEL: util.func private @dupe_result_callee
// CHECK-SAME: (%[[ARG0:.+]]: i1, %[[ARG1:.+]]: index) -> (index, index)
util.func private @dupe_result_callee(%arg0: i1, %arg1: index) -> (index, index, index) {
  // CHECK: %[[ADD0:.+]] = arith.addi %[[ARG1]], %[[ARG1]]
  %add0 = arith.addi %arg1, %arg1 : index
  // CHECK: %[[ADD1:.+]] = arith.addi %[[ADD0]], %[[ARG1]]
  %add1 = arith.addi %add0, %arg1 : index
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // CHECK: util.return %[[ADD0]], %[[ADD0]]
  util.return %add0, %add0, %add0 : index, index, index
^bb2:
  // CHECK: util.return %[[ADD0]], %[[ADD1]]
  util.return %add0, %add1, %add0 : index, index, index
}

// CHECK: util.func public @dupe_result_caller(%[[ARG0:.+]]: i1, %[[ARG1:.+]]: index)
util.func public @dupe_result_caller(%arg0: i1, %arg1: index) -> (index, index, index) {
  // CHECK: %[[RET:.+]]:2 = util.call @dupe_result_callee(%[[ARG0]], %[[ARG1]]) : (i1, index) -> (index, index)
  %ret:3 = util.call @dupe_result_callee(%arg0, %arg1) : (i1, index) -> (index, index, index)
  // CHECK: util.return %[[RET]]#0, %[[RET]]#1, %[[RET]]#0
  util.return %ret#0, %ret#1, %ret#2 : index, index, index
}

// -----

// Tests that uniformly constant args get inlined into callees.

// CHECK-LABEL: util.func private @uniform_arg_callee
// CHECK-SAME: () -> index
util.func private @uniform_arg_callee(%arg0: index) -> index {
  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[ADD:.+]] = arith.addi %[[C1]], %[[C1]]
  %add = arith.addi %arg0, %arg0 : index
  // CHECK: util.return %[[ADD]]
  util.return %add : index
}

// CHECK: util.func public @uniform_arg_caller_a
util.func public @uniform_arg_caller_a() -> (index, index) {
  %c1 = arith.constant 1 : index
  // CHECK: %[[A_RET0:.+]] = util.call @uniform_arg_callee() : () -> index
  %ret0 = util.call @uniform_arg_callee(%c1) : (index) -> index
  // CHECK: %[[A_RET1:.+]] = util.call @uniform_arg_callee() : () -> index
  %ret1 = util.call @uniform_arg_callee(%c1) : (index) -> index
  // CHECK: util.return %[[A_RET0]], %[[A_RET1]]
  util.return %ret0, %ret1 : index, index
}
// CHECK: util.func public @uniform_arg_caller_b
util.func public @uniform_arg_caller_b() -> index {
  %c1 = arith.constant 1 : index
  // CHECK: %[[B_RET0:.+]] = util.call @uniform_arg_callee() : () -> index
  %ret0 = util.call @uniform_arg_callee(%c1) : (index) -> index
  // CHECK: util.return %[[B_RET0]]
  util.return %ret0 : index
}

// -----

// Tests that uniformly constant results get inlined into callers.

// CHECK-LABEL: util.func private @uniform_result_callee
// CHECK-SAME: (%[[ARG0:.+]]: i1)
util.func private @uniform_result_callee(%arg0: i1) -> index {
  %c0 = arith.constant 0 : index
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // CHECK: util.return
  util.return %c0 : index
^bb2:
  // CHECK: util.return
  util.return %c0 : index
}

// CHECK: util.func public @uniform_result_caller(%[[ARG0:.+]]: i1)
util.func public @uniform_result_caller(%arg0: i1) -> index {
  // CHECK: call @uniform_result_callee(%[[ARG0]]) : (i1) -> ()
  %ret0 = util.call @uniform_result_callee(%arg0) : (i1) -> index
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: util.return %[[C0]]
  util.return %ret0 : index
}

// -----

// Tests that uniformly duplicate constant results get combined/inlined.

// CHECK-LABEL: util.func private @dupe_constant_result_callee
// CHECK-SAME: (%[[ARG0:.+]]: i1) -> index
util.func private @dupe_constant_result_callee(%arg0: i1) -> (index, index, index) {
  // CHECK: %[[C0:.+]] = arith.constant 0
  %c0 = arith.constant 0 : index
  // CHECK: %[[C1:.+]] = arith.constant 1
  %c1 = arith.constant 1 : index
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // CHECK: util.return %[[C0]]
  util.return %c0, %c0, %c0 : index, index, index
^bb2:
  // CHECK: util.return %[[C1]]
  util.return %c0, %c1, %c0 : index, index, index
}

// CHECK: util.func public @dupe_constant_result_caller(%[[ARG0:.+]]: i1)
util.func public @dupe_constant_result_caller(%arg0: i1) -> (index, index, index) {
  // CHECK: %[[RET:.+]] = util.call @dupe_constant_result_callee(%[[ARG0]]) : (i1) -> index
  %ret:3 = util.call @dupe_constant_result_callee(%arg0) : (i1) -> (index, index, index)
  // CHECK: %[[C0_INLINE:.+]] = arith.constant 0
  // CHECK-NEXT: %[[C0_INLINE_DUPE:.+]] = arith.constant 0
  // CHECK: util.return %[[C0_INLINE]], %[[RET]], %[[C0_INLINE_DUPE]]
  util.return %ret#0, %ret#1, %ret#2 : index, index, index
}

// -----

// Tests that public functions are unmodified (the unused arg is not dropped).

// CHECK-LABEL: util.func public @public_unused_arg
// CHECK-SAME: (%[[ARG0:.+]]: index)
util.func public @public_unused_arg(%arg0: index) {
  util.return
}

// -----

// Tests that non-uniform call args don't get optimized.

// CHECK-LABEL: util.func private @nonuniform_arg_callee
// CHECK-SAME: (%[[ARG0:.+]]: index) -> index
util.func private @nonuniform_arg_callee(%arg0: index) -> index {
  // CHECK: %[[ADD:.+]] = arith.addi %[[ARG0]], %[[ARG0]]
  %add = arith.addi %arg0, %arg0 : index
  // CHECK: util.return %[[ADD]]
  util.return %add : index
}

// CHECK: util.func public @nonuniform_arg_caller_a(%[[A_ARG0:.+]]: index)
util.func public @nonuniform_arg_caller_a(%arg0: index) -> (index, index) {
  // CHECK: %[[A_RET0:.+]] = util.call @nonuniform_arg_callee(%[[A_ARG0]]) : (index) -> index
  %ret0 = util.call @nonuniform_arg_callee(%arg0) : (index) -> index
  // CHECK: %[[A_RET1:.+]] = util.call @nonuniform_arg_callee(%[[A_ARG0]]) : (index) -> index
  %ret1 = util.call @nonuniform_arg_callee(%arg0) : (index) -> index
  // CHECK: util.return %[[A_RET0]], %[[A_RET1]]
  util.return %ret0, %ret1 : index, index
}
// CHECK: util.func public @nonuniform_arg_caller_b(%[[B_ARG0:.+]]: index)
util.func public @nonuniform_arg_caller_b(%arg0: index) -> index {
  // CHECK: %[[B_RET0:.+]] = util.call @nonuniform_arg_callee(%[[B_ARG0]]) : (index) -> index
  %ret0 = util.call @nonuniform_arg_callee(%arg0) : (index) -> index
  // CHECK: util.return %[[B_RET0]]
  util.return %ret0 : index
}

// -----

// Tests that non-uniform call args w/ constants don't get optimized.

// CHECK-LABEL: util.func private @nonuniform_constant_arg_callee
// CHECK-SAME: (%[[ARG0:.+]]: index) -> index
util.func private @nonuniform_constant_arg_callee(%arg0: index) -> index {
  // CHECK: %[[ADD:.+]] = arith.addi %[[ARG0]], %[[ARG0]]
  %add = arith.addi %arg0, %arg0 : index
  // CHECK: util.return %[[ADD]]
  util.return %add : index
}

// CHECK: util.func public @nonuniform_arg_caller(%[[CALLER_ARG0:.+]]: index)
util.func public @nonuniform_arg_caller(%arg0: index) -> (index, index) {
  // CHECK-DAG: %[[C10:.+]] = arith.constant 10
  %c10 = arith.constant 10 : index
  // CHECK: %[[RET0:.+]] = util.call @nonuniform_constant_arg_callee(%[[CALLER_ARG0]]) : (index) -> index
  %ret0 = util.call @nonuniform_constant_arg_callee(%arg0) : (index) -> index
  // CHECK: %[[RET1:.+]] = util.call @nonuniform_constant_arg_callee(%[[C10]]) : (index) -> index
  %ret1 = util.call @nonuniform_constant_arg_callee(%c10) : (index) -> index
  // CHECK: util.return %[[RET0]], %[[RET1]]
  util.return %ret0, %ret1 : index, index
}

// -----

// Tests that non-uniform call args w/ constants don't get optimized (order
// flipped from above).

// CHECK-LABEL: util.func private @nonuniform_constant_arg_callee_flipped
// CHECK-SAME: (%[[ARG0:.+]]: index) -> index
util.func private @nonuniform_constant_arg_callee_flipped(%arg0: index) -> index {
  // CHECK: %[[ADD:.+]] = arith.addi %[[ARG0]], %[[ARG0]]
  %add = arith.addi %arg0, %arg0 : index
  // CHECK: util.return %[[ADD]]
  util.return %add : index
}

// CHECK: util.func public @nonuniform_arg_caller_flipped(%[[CALLER_ARG0:.+]]: index)
util.func public @nonuniform_arg_caller_flipped(%arg0: index) -> (index, index) {
  // CHECK-DAG: %[[C10:.+]] = arith.constant 10
  %c10 = arith.constant 10 : index
  // CHECK: %[[RET0:.+]] = util.call @nonuniform_constant_arg_callee_flipped(%[[C10]]) : (index) -> index
  %ret0 = util.call @nonuniform_constant_arg_callee_flipped(%c10) : (index) -> index
  // CHECK: %[[RET1:.+]] = util.call @nonuniform_constant_arg_callee_flipped(%[[CALLER_ARG0]]) : (index) -> index
  %ret1 = util.call @nonuniform_constant_arg_callee_flipped(%arg0) : (index) -> index
  // CHECK: util.return %[[RET0]], %[[RET1]]
  util.return %ret0, %ret1 : index, index
}

// -----

// Tests that non-uniform call args w/ constants don't get optimized.

// CHECK-LABEL: util.func private @nonuniform_constant_arg_callee
// CHECK-SAME: (%[[ARG0:.+]]: index) -> index
util.func private @nonuniform_constant_arg_callee(%arg0: index) -> index {
  // CHECK: %[[ADD:.+]] = arith.addi %[[ARG0]], %[[ARG0]]
  %add = arith.addi %arg0, %arg0 : index
  // CHECK: util.return %[[ADD]]
  util.return %add : index
}

// CHECK: util.func public @nonuniform_arg_caller(%[[CALLER_ARG0:.+]]: index)
util.func public @nonuniform_arg_caller(%arg0: index) -> (index, index) {
  // CHECK-DAG: %[[C10:.+]] = arith.constant 10
  %c10 = arith.constant 10 : index
  // CHECK: %[[RET0:.+]] = util.call @nonuniform_constant_arg_callee(%[[CALLER_ARG0]]) : (index) -> index
  %ret0 = util.call @nonuniform_constant_arg_callee(%arg0) : (index) -> index
  // CHECK: %[[RET1:.+]] = util.call @nonuniform_constant_arg_callee(%[[C10]]) : (index) -> index
  %ret1 = util.call @nonuniform_constant_arg_callee(%c10) : (index) -> index
  // CHECK: util.return %[[RET0]], %[[RET1]]
  util.return %ret0, %ret1 : index, index
}

// -----

// Tests that non-uniform call args w/ constants don't get optimized (order
// flipped from above).

// CHECK-LABEL: util.func private @nonuniform_constant_arg_callee_flipped
// CHECK-SAME: (%[[ARG0:.+]]: index) -> index
util.func private @nonuniform_constant_arg_callee_flipped(%arg0: index) -> index {
  // CHECK: %[[ADD:.+]] = arith.addi %[[ARG0]], %[[ARG0]]
  %add = arith.addi %arg0, %arg0 : index
  // CHECK: util.return %[[ADD]]
  util.return %add : index
}

// CHECK: util.func public @nonuniform_arg_caller_flipped(%[[CALLER_ARG0:.+]]: index)
util.func public @nonuniform_arg_caller_flipped(%arg0: index) -> (index, index) {
  // CHECK-DAG: %[[C10:.+]] = arith.constant 10
  %c10 = arith.constant 10 : index
  // CHECK: %[[RET0:.+]] = util.call @nonuniform_constant_arg_callee_flipped(%[[C10]]) : (index) -> index
  %ret0 = util.call @nonuniform_constant_arg_callee_flipped(%c10) : (index) -> index
  // CHECK: %[[RET1:.+]] = util.call @nonuniform_constant_arg_callee_flipped(%[[CALLER_ARG0]]) : (index) -> index
  %ret1 = util.call @nonuniform_constant_arg_callee_flipped(%arg0) : (index) -> index
  // CHECK: util.return %[[RET0]], %[[RET1]]
  util.return %ret0, %ret1 : index, index
}

// -----

// Tests that non-uniform call results don't get optimized.

// CHECK-LABEL: util.func private @nonuniform_result_callee
// CHECK-SAME: (%[[ARG0:.+]]: i1) -> index
util.func private @nonuniform_result_callee(%arg0: i1) -> index {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // CHECK: %[[C0:.+]] = arith.constant 0
  %c0 = arith.constant 0 : index
  // CHECK: util.return %[[C0]]
  util.return %c0 : index
^bb2:
  // CHECK: %[[C1:.+]] = arith.constant 1
  %c1 = arith.constant 1 : index
  // CHECK: util.return %[[C1]]
  util.return %c1 : index
}

// CHECK: util.func public @nonuniform_result_caller(%[[ARG0:.+]]: i1)
util.func public @nonuniform_result_caller(%arg0: i1) -> index {
  // CHECK: %[[RET0:.+]] = util.call @nonuniform_result_callee(%[[ARG0]]) : (i1) -> index
  %ret0 = util.call @nonuniform_result_callee(%arg0) : (i1) -> index
  // CHECK: util.return %[[RET0]]
  util.return %ret0 : index
}

// -----

// Tests that args that directly pass-through to results get hoisted out into
// the caller.

// CHECK-LABEL: util.func private @passthrough_callee() {
util.func private @passthrough_callee(%arg0: index) -> index {
  // CHECK: util.return
  util.return %arg0 : index
}

// CHECK: util.func public @passthrough_caller(%[[ARG0:.+]]: index)
util.func public @passthrough_caller(%arg0: index) -> index {
  // CHECK: call @passthrough_callee() : () -> ()
  %ret0 = util.call @passthrough_callee(%arg0) : (index) -> index
  // CHECK: util.return %[[ARG0]]
  util.return %ret0 : index
}

// -----

// Tests that args that directly pass-through to results get hoisted out into
// the caller but they are preserved as args if they are used for other things.

// CHECK-LABEL: util.func private @passthrough_preserve_arg_callee
// CHECK-SAME: (%[[ARG0:.+]]: index) -> index {
util.func private @passthrough_preserve_arg_callee(%arg0: index) -> (index, index) {
  // CHECK: %[[ADD:.+]] = arith.addi %[[ARG0]], %[[ARG0]]
  %add = arith.addi %arg0, %arg0 : index
  // CHECK: util.return %[[ADD]]
  util.return %arg0, %add : index, index
}

// CHECK: util.func public @passthrough_preserve_arg_caller(%[[ARG0:.+]]: index)
util.func public @passthrough_preserve_arg_caller(%arg0: index) -> (index, index) {
  // CHECK: %[[RET1:.+]] = util.call @passthrough_preserve_arg_callee(%[[ARG0]]) : (index) -> index
  %ret:2 = util.call @passthrough_preserve_arg_callee(%arg0) : (index) -> (index, index)
  // CHECK: util.return %[[ARG0]], %[[RET1]]
  util.return %ret#0, %ret#1 : index, index
}
