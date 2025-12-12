// RUN: iree-opt --split-input-file %s | FileCheck %s

// Tests that raw SSA identifiers in CHECK lines trigger errors.

// CHECK-LABEL: @raw_ssa_bad
util.func @raw_ssa_bad(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %offset = arith.constant 0 : index
  // CHECK: %0 = arith.addf
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  // CHECK: %offset = arith.constant
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that IR without CHECK lines triggers error.

util.func @no_checks(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that bare TODO/FIXME/NOTE comments trigger errors.

// CHECK-LABEL: @bare_todo
util.func @bare_todo(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // TODO
  %c0 = arith.constant 0 : index
  // FIXME
  // NOTE
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that weak TODO explanations (1-3 words) trigger errors.

// CHECK-LABEL: @weak_todo
util.func @weak_todo(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // TODO: fix this
  %c0 = arith.constant 0 : index
  // FIXME: broken
  // NOTE: important
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that detailed TODO explanations pass.

// CHECK-LABEL: @good_todo
util.func @good_todo(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // TODO(#1234): Add support for dynamic shapes after upstream MLIR lands the DynamicShapeInterface.
  %c0 = arith.constant 0 : index
  // FIXME: The dominance check fails for block arguments in loop headers because SSA def precedes use.
  // NOTE: This await cannot be eliminated because public functions can't return timepoints per the ABI.
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that CHECK-NOT without any anchors triggers error.

// CHECK-LABEL: @unanchored_check_not
util.func @unanchored_check_not(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: stream.timepoint.await
  %c0 = arith.constant 0 : index
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that CHECK-NOT without before anchor triggers error.

// CHECK-LABEL: @missing_before_anchor
util.func @missing_before_anchor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: stream.timepoint.await
  // CHECK: util.return
  %c0 = arith.constant 0 : index
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that CHECK-NOT without after anchor triggers error.

// CHECK-LABEL: @missing_after_anchor
util.func @missing_after_anchor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: arith.constant
  // CHECK-NOT: stream.timepoint.await
  %c0 = arith.constant 0 : index
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that properly anchored CHECK-NOT passes.

// CHECK-LABEL: @properly_anchored
util.func @properly_anchored(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: arith.constant
  // CHECK-NOT: stream.timepoint.await
  // CHECK: util.return
  %c0 = arith.constant 0 : index
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that excessive wildcards (>2) trigger warnings.

// CHECK-LABEL: @excessive_wildcards
util.func @excessive_wildcards(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex::@dispatch[{{.+}}]({{.+}}, {{.+}}, {{.+}}) : ({{.+}})
  %0 = stream.async.dispatch @ex::@dispatch[%c0](%arg0, %arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that %[[C0]], %[[ARG0]] style names trigger warnings.

// CHECK-LABEL: @bad_names
util.func @bad_names(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[C0:.+]] = arith.constant 0
  %c0 = arith.constant 0 : index
  // CHECK: %[[C123_I32:.+]] = arith.constant 123
  %c123 = arith.constant 123 : i32
  // CHECK: %[[RESULT:.+]] = arith.addf %[[ARG0:.+]], %[[ARG1:.+]]
  %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that IR/CHECK name mismatches trigger warnings.

// CHECK-LABEL: @mismatched_names
util.func @mismatched_names(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[SIZE:.+]] = arith.constant
  %transient_size = arith.constant 123 : index
  // CHECK: %[[READY:.+]] = stream.timepoint.await
  %slice_ready = stream.timepoint.immediate => !stream.timepoint
  // CHECK: %[[RESULT:.+]] = arith.addf
  %double_value = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.return %double_value : tensor<4xf32>
}

// -----

// Tests that matched IR/CHECK names don't trigger warnings.

// CHECK-LABEL: @matched_names
util.func @matched_names(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[TRANSIENT_SIZE:.+]] = arith.constant
  %transient_size = arith.constant 123 : index
  // CHECK: %[[SLICE_READY:.+]] = stream.timepoint.await
  %slice_ready = stream.timepoint.immediate => !stream.timepoint
  // CHECK: %[[DOUBLE_VALUE:.+]] = arith.addf
  %double_value = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.return %double_value : tensor<4xf32>
}

// -----

// Tests that non-semantic IR names don't trigger mismatch warnings.

// CHECK-LABEL: @no_semantic_names
util.func @no_semantic_names(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[C0:.+]] = arith.constant
  %c0 = arith.constant 0 : index
  // CHECK: %[[RESULT:.+]] = arith.addf
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that wildcards in terminators trigger warnings.

// CHECK-LABEL: @wildcard_terminators
util.func @wildcard_terminators(%arg0: tensor<4xf32>, %arg1: i1) -> tensor<4xf32> {
  // CHECK: scf.yield {{.+}}, {{.+}}
  %result = scf.if %arg1 -> (tensor<4xf32>) {
    %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
    scf.yield %0 : tensor<4xf32>
  } else {
    scf.yield %arg0 : tensor<4xf32>
  }
  // CHECK: util.return {{.+}}
  util.return %result : tensor<4xf32>
}

// -----

// Tests that explicit terminator operands don't trigger warnings.

// CHECK-LABEL: @explicit_terminators
util.func @explicit_terminators(%arg0: tensor<4xf32>, %arg1: i1) -> tensor<4xf32> {
  // CHECK: %[[RESULT:.+]] = scf.if
  %result = scf.if %arg1 -> (tensor<4xf32>) {
    // CHECK: %[[SUM:.+]] = arith.addf
    %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
    // CHECK: scf.yield %[[SUM]]
    scf.yield %0 : tensor<4xf32>
  } else {
    // CHECK: scf.yield %arg0
    scf.yield %arg0 : tensor<4xf32>
  }
  // CHECK: util.return %[[RESULT]]
  util.return %result : tensor<4xf32>
}

// -----

// Tests that func.return and cf.br with wildcards trigger warnings.

// CHECK-LABEL: @other_terminators
func.func @other_terminators(%arg0: i32, %arg1: i1) -> i32 {
  // CHECK: cf.cond_br {{.+}}, ^bb1({{.+}} : i32), ^bb2
  cf.cond_br %arg1, ^bb1(%arg0 : i32), ^bb2
^bb1(%val: i32):
  // CHECK: func.return {{.+}}
  func.return %val : i32
^bb2:
  %c0 = arith.constant 0 : i32
  // CHECK: cf.br ^bb1({{.+}} : i32)
  cf.br ^bb1(%c0 : i32)
}

// -----

// Tests that CHECK before LABEL triggers warning.

// CHECK: %[[FOO:.+]] = arith.constant
// CHECK-LABEL: @check_before_label
util.func @check_before_label(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that CHECK after LABEL doesn't trigger warning.

// CHECK-LABEL: @proper_label_context
util.func @proper_label_context(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[C0:.+]] = arith.constant
  %c0 = arith.constant 0 : index
  // CHECK: %[[RESULT:.+]] = arith.addf
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  // CHECK: util.return %[[RESULT]]
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that CHECK after first LABEL is ok (multi-function file).

// CHECK-LABEL: @first_function
util.func @first_function(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: arith.constant
  %c0 = arith.constant 0 : index
  util.return %arg0 : tensor<4xf32>
}

// CHECK-LABEL: @second_function
util.func @second_function(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: arith.addf
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that unused captures trigger warnings.

// CHECK-LABEL: @unused_captures
util.func @unused_captures(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[UNUSED:.+]] = arith.constant
  %c0 = arith.constant 0 : index
  // CHECK: %[[ANOTHER_UNUSED:.+]] = arith.addf
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  // CHECK: util.return
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that used captures don't trigger warnings.

// CHECK-LABEL: @used_captures
util.func @used_captures(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[C0:.+]] = arith.constant
  %c0 = arith.constant 0 : index
  // CHECK: %[[RESULT:.+]] = arith.addf
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  // CHECK: some.op %[[C0]]
  // CHECK: util.return %[[RESULT]]
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that captures used in CHECK-SAME don't trigger warnings.

// CHECK-LABEL: @check_same_usage
util.func @check_same_usage(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[TP:.+]] = stream.timepoint.immediate
  %tp = stream.timepoint.immediate => !stream.timepoint
  // CHECK: stream.async.execute
  // CHECK-SAME: await(%[[TP]])
  %result = stream.async.execute await(%tp) => with() {
    %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
    stream.yield %0 : tensor<4xf32>
  } => !stream.timepoint
  util.return %arg0 : tensor<4xf32>
}

// -----

// Tests that captures used in same line don't trigger warnings.

// CHECK-LABEL: @inline_usage
util.func @inline_usage(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: util.return %[[RESULT:.+]]
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that tuple destructuring extracts ALL variable names correctly.

// CHECK-LABEL: @tuple_destructuring
util.func @tuple_destructuring(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // CHECK: %[[INIT:.+]], %[[INIT_TP:.+]] = stream.test.timeline_op
  %init, %init_tp = stream.test.timeline_op with(%arg0) : (tensor<4xf32>) -> (tensor<4xf32>, !stream.timepoint)
  util.return %init, %init : tensor<4xf32>, tensor<4xf32>
}

// -----

// Tests that FileCheck tuple syntax (:N suffix) is recognized correctly.

// CHECK-LABEL: @filecheck_tuple_syntax
util.func @filecheck_tuple_syntax(%arg0: tensor<4xf32>, %cond: i1) -> (tensor<4xf32>, !stream.timepoint) {
  // CHECK: %[[BRANCH:.+]]:2 = scf.if
  %branch_resource, %branch_tp = scf.if %cond -> (tensor<4xf32>, !stream.timepoint) {
    %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
    %tp = stream.timepoint.immediate => !stream.timepoint
    scf.yield %0, %tp : tensor<4xf32>, !stream.timepoint
  } else {
    %tp = stream.timepoint.immediate => !stream.timepoint
    scf.yield %arg0, %tp : tensor<4xf32>, !stream.timepoint
  }
  util.return %branch_resource, %branch_tp : tensor<4xf32>, !stream.timepoint
}

// -----

// Tests that intentional naming variance doesn't trigger false positives.

// CHECK-LABEL: @intentional_variance
util.func @intentional_variance(%arg0: tensor<4xf32>, %cond: i1) -> tensor<4xf32> {
  // CHECK: %[[TP_IMMEDIATE:.+]] = stream.timepoint.immediate
  %tp = stream.timepoint.immediate => !stream.timepoint
  // CHECK: %[[RESULT:.+]] = scf.if
  %result = scf.if %cond -> (tensor<4xf32>) {
    // CHECK: %[[TP_THEN:.+]] = stream.timepoint.immediate
    %tp_then = stream.timepoint.immediate => !stream.timepoint
    %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
    scf.yield %0 : tensor<4xf32>
  } else {
    // CHECK: %[[TP_ELSE:.+]] = stream.timepoint.immediate
    %tp_else = stream.timepoint.immediate => !stream.timepoint
    scf.yield %arg0 : tensor<4xf32>
  }
  util.return %result : tensor<4xf32>
}

// -----

// Tests that scope-aware matching prevents false positives from distant variables.

// CHECK-LABEL: @scope_aware_matching
util.func @scope_aware_matching(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[TP0:.+]] = stream.timepoint.immediate
  %tp0 = stream.timepoint.immediate => !stream.timepoint
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  // More than 5 lines away from %tp0 - should not trigger false positive.
  // CHECK: %[[TP1:.+]] = stream.timepoint.immediate
  %tp1 = stream.timepoint.immediate => !stream.timepoint
  util.return %arg0 : tensor<4xf32>
}

// -----

// Regression test for Bug #1: Tuple positional pairing.
// Verifies that CHECK captures are paired positionally with IR variables in tuples.
// Without positional pairing, R_DEFAULT would incorrectly match against tp_default.

// CHECK-LABEL: @tuple_positional_pairing
util.func @tuple_positional_pairing(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[R_DEFAULT:.+]], %[[TP_DEFAULT:.+]] = stream.test.timeline_op
  %r_default, %tp_default = stream.test.timeline_op with() : () -> (!stream.resource<external>, !stream.timepoint)
  util.return %arg0 : tensor<4xf32>
}

// -----

// Regression test for Bug #2: Target line detection for scope.
// Verifies that CHECK only compares against the next IR line, not distant variables.
// Without target line detection, INNER would incorrectly match against %iter from line 8.

// CHECK-LABEL: @target_line_scope
util.func @target_line_scope(%arg0: tensor<4xf32>, %cond: i1) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %iter = arith.constant 0 : index
  %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %arg0) -> (tensor<4xf32>) {
    // CHECK: %[[INNER:.+]]:2 = scf.if
    %inner, %inner_tp = scf.if %cond -> (tensor<4xf32>, !stream.timepoint) {
      %tp = stream.timepoint.immediate => !stream.timepoint
      scf.yield %arg, %tp : tensor<4xf32>, !stream.timepoint
    } else {
      %tp = stream.timepoint.immediate => !stream.timepoint
      scf.yield %arg, %tp : tensor<4xf32>, !stream.timepoint
    }
    scf.yield %inner : tensor<4xf32>
  }
  util.return %result : tensor<4xf32>
}

// -----

// Tests that non-semantic capture warnings only fire on definitions, not usages.
// Definitions have : pattern like %[[C0:.+]], usages are bare like %[[C0]].

// CHECK-LABEL: @definition_vs_usage
util.func @definition_vs_usage(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // These definitions should warn (non-semantic names).
  // CHECK: %[[C0:.+]] = arith.constant
  %c0 = arith.constant 0 : index
  // CHECK: %[[ARG1:.*]] = some.op
  %value = some.op %arg0 : tensor<4xf32>

  // These usages should NOT warn (no : pattern).
  // CHECK: scf.yield %[[C0]], %[[ARG1]]
  // CHECK-SAME: uses(%[[C0]])
  // CHECK: util.return %[[ARG1]]

  util.return %value : tensor<4xf32>
}

// -----

// Tests that hardened patterns handle various FileCheck regex syntaxes.

// CHECK-LABEL: @hardened_patterns
util.func @hardened_patterns(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // Different pattern types - all should warn on definition.
  // CHECK: %[[C0:.+]] = arith.constant
  %c0 = arith.constant 0 : index
  // CHECK: %[[C1:.*]] = arith.constant
  %c1 = arith.constant 1 : index
  // CHECK: %[[ARG1:[a-zA-Z0-9]+]] = some.op
  %value1 = some.op %arg0 : tensor<4xf32>
  // CHECK: %[[C2_I32:\d+]] = arith.constant
  %c2 = arith.constant 2 : i32

  // Usages with same patterns should NOT warn.
  // CHECK: another.op %[[C0]], %[[C1]], %[[ARG1]], %[[C2_I32]]

  util.return %value1 : tensor<4xf32>
}

// -----

// Tests that unambiguous bare labels don't trigger warnings.

// CHECK-LABEL: @unambiguous_label
util.func @unambiguous_label(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: arith.constant
  %c0 = arith.constant 0 : index
  util.return %arg0 : tensor<4xf32>
}

// -----

// Tests that ambiguous bare labels trigger warnings with all matches shown.

// CHECK-LABEL: @dispatch
util.func @dispatch(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // This should warn because @dispatch appears in multiple places.
  // CHECK: stream.async.dispatch @dispatch
  %result = stream.async.dispatch @dispatch[%c0](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  util.return %result : tensor<4xf32>
}

stream.executable private @dispatch {
  stream.executable.export public @entry
  builtin.module {
    func.func @entry(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      return %arg0 : tensor<4xf32>
    }
  }
}

// -----

// Tests that labels not found in IR trigger warnings.

// CHECK-LABEL: @typo_in_label
util.func @typo_in_function_name(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // Label doesn't match actual function name - should warn.
  %c0 = arith.constant 0 : index
  util.return %arg0 : tensor<4xf32>
}

// -----

// Tests that labels with operation prefix don't trigger checks.

// CHECK-LABEL: util.func @proper_prefix
util.func @proper_prefix(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // Has operation prefix, no check needed even if ambiguous.
  %c0 = arith.constant 0 : index
  util.return %arg0 : tensor<4xf32>
}

// -----

// Tests that globals with ambiguous names trigger warnings.

// CHECK-LABEL: @common_global
util.global private @common_global : tensor<4xf32>
util.global private mutable @common_global_2 : tensor<4xf32>

util.func @use_common_global(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // Reference to @common_global - creates ambiguity.
  %0 = util.global.load @common_global : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that SSA wildcards using {{.*}} trigger warnings.

// CHECK-LABEL: @invalid_ssa_wildcard
util.func @invalid_ssa_wildcard(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %{{.*}} = arith.constant
  %c0 = arith.constant 0 : index
  // CHECK: %{{.+}} = arith.constant
  %c1 = arith.constant 1 : index
  util.return %arg0 : tensor<4xf32>
}

// -----

// Tests that symbol wildcards using {{.*}} trigger warnings.

// CHECK-LABEL: @invalid_symbol_wildcard
util.func @invalid_symbol_wildcard(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: util.global.load @{{.*}}
  %0 = util.global.load @some_global : tensor<4xf32>
  // CHECK: util.global.load @{{.+}}
  %1 = util.global.load @another_global : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

util.global private @some_global : tensor<4xf32>
util.global private @another_global : tensor<4xf32>

// -----

// Tests that %arg0 in operands triggers error.

// CHECK-LABEL: @raw_arg_in_operand
util.func @raw_arg_in_operand(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: arith.addf %arg0, %arg1
  %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that semantic names like %buffer trigger error.

// CHECK-LABEL: @semantic_names
util.func @semantic_names(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: stream.async.dispatch @dispatch(%buffer, %offset)
  "stream.async.dispatch"(%arg0, %arg0) {callee = @dispatch} : (tensor<4xf32>, tensor<4xf32>) -> ()
  util.return %arg0 : tensor<4xf32>
}

// -----

// Tests that %arg0 in function signatures triggers error.

// CHECK: util.func public @sig_test(%arg0: tensor<4xf32>)
util.func public @sig_test(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  util.return %arg0 : tensor<4xf32>
}

// -----

// Tests that multiple raw SSA on same line all trigger errors.

// CHECK-LABEL: @multiple_raw_ssa
util.func @multiple_raw_ssa(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: arith.addf %a, %b, %c, %d
  %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that NOLINT on preceding line suppresses errors.

// CHECK-LABEL: @nolint_test
util.func @nolint_test(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // NOLINT: raw_ssa_identifier - crash reproducer test
  // CHECK: %0 = arith.constant
  // CHECK: return %0
  %c0 = arith.constant 0 : index
  util.return %arg0 : tensor<4xf32>
}

// -----

// Tests that captures are NOT flagged.

// CHECK-LABEL: @captures_ok
util.func @captures_ok(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: arith.addf %[[A:.+]], %[[B:.+]]
  %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
  // CHECK: util.return %[[RES:.+]]
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that MLIR constants (%c0, %c123, %cst) are exempted (anti-pattern but widespread).

// CHECK-LABEL: @constants_exempted
util.func @constants_exempted(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %c0 = arith.constant 0
  %c0 = arith.constant 0 : index
  // CHECK: %c123_i32 = arith.constant 123
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %cst = arith.constant 3.14
  %cst = arith.constant 3.14 : f32
  // CHECK: %cst_0 = arith.constant 2.71
  %cst_0 = arith.constant 2.71 : f32
  util.return %arg0 : tensor<4xf32>
}
