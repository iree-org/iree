// RUN: iree-opt -split-input-file -iree-util-hoist-into-globals -allow-unregistered-dialect %s | IreeFileCheck %s

// CHECK-LABEL: @hoist_simple_const_expr
module @hoist_simple_const_expr {
  // CHECK: util.global private @[[HOISTED_SYM:.*]] : i32
  // CHECK: func @main
  builtin.func @main() -> (i32) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    // CHECK-NOT: arith.constant
    // CHECK-NOT: iree_unregistered.const_expr
    // CHECK: %[[VAL:.*]] = util.global.load @[[HOISTED_SYM]] : i32
    // CHECK: return %[[VAL]]
    %2 = "iree_unregistered.const_expr"(%0, %1) : (i32, i32) -> i32
    return %2 : i32
  }
  // CHECK: util.initializer {
  // CHECK:   %[[C0:.*]] = arith.constant 0 : i32
  // CHECK:   %[[C1:.*]] = arith.constant 1 : i32
  // CHECK:   %[[CE0:.*]] = "iree_unregistered.const_expr"(%[[C0]], %[[C1]])
  // CHECK:   util.global.store %[[CE0]], @[[HOISTED_SYM]] : i32
  // CHECK:   util.initializer.return
  // CHECK: }
}

// -----
// We do a bit more exhaustive checking on this one but in subsequent do simple
// checks.
// CHECK-LABEL: @do_not_hoist_variable_op
// CHECK-NOT: util.global
// CHECK: func @main
// CHECK: %[[VAL:.*]] = "iree_unregistered.var_expr"
// CHECK: return %[[VAL]]
// CHECK-NOT: util.initializer
module @do_not_hoist_variable_op {
  builtin.func @main() -> (i32) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    %2 = "iree_unregistered.var_expr"(%0, %1) : (i32, i32) -> i32
    return %2 : i32
  }
}

// -----
// CHECK-LABEL: @do_not_hoist_variable_operands
// CHECK-NOT: util.global
// CHECK-NOT: util.initializer
module @do_not_hoist_variable_operands {
  builtin.func @main(%arg0 : i32) -> (i32) {
    %0 = arith.constant 0 : i32
    %2 = "iree_unregistered.const_expr"(%0, %arg0) : (i32, i32) -> i32
    return %2 : i32
  }
}

// -----
// Tests a const-expr tree with multiple uses at different levels.
// CHECK-LABEL: @hoist_tree_const_expr
module @hoist_tree_const_expr {
  // CHECK: util.global private @[[HOISTED_1:.*]] : i32
  // CHECK: util.global private @[[HOISTED_0:.*]] : i32
  // CHECK: util.global private @latent_global : i32
  util.global private @latent_global : i32

  // CHECK: func @main
  builtin.func @main() -> (i32, i32, i32) {
    // CHECK: %[[LOAD_HOISTED_1:.*]] = util.global.load @[[HOISTED_1]] : i32
    // CHECK: %[[RESULT:.*]] = "iree_unregistered.var_expr"(%[[LOAD_HOISTED_1]])
    // CHECK-DAG: %[[LOAD_HOISTED_0:.*]] = util.global.load @[[HOISTED_0]] : i32
    // CHECK-DAG: %[[LOAD_HOISTED_1:.*]] = util.global.load @[[HOISTED_1]] : i32
    // CHECK: return %[[LOAD_HOISTED_0]], %[[LOAD_HOISTED_1]], %[[RESULT]]
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    %2 = "iree_unregistered.const_expr"(%0, %1) : (i32, i32) -> i32
    %3 = util.global.load @latent_global : i32
    %4 = "iree_unregistered.const_expr"(%2, %3) : (i32, i32) -> i32
    %5 = "iree_unregistered.var_expr"(%4) : (i32) -> i32
    return %2, %4, %5 : i32, i32, i32
  }
  // CHECK: util.initializer {
  // CHECK:   %[[C0:.*]] = arith.constant 0 : i32
  // CHECK:   %[[C1:.*]] = arith.constant 1 : i32
  // CHECK:   %[[CE0:.*]] = "iree_unregistered.const_expr"(%[[C0]], %[[C1]])
  // CHECK:   util.global.store %[[CE0]], @[[HOISTED_0]] : i32
  // CHECK:   util.initializer.return
  // CHECK: }
  // CHECK: util.initializer {
  // CHECK:   %[[LOAD_HOISTED_0:.*]] = util.global.load @[[HOISTED_0]] : i32
  // CHECK:   %[[LOAD_LATENT_GLOBAL:.*]] = util.global.load @latent_global : i32
  // CHECK:   %[[CE1:.*]] = "iree_unregistered.const_expr"(%[[LOAD_HOISTED_0]], %[[LOAD_LATENT_GLOBAL]])
  // CHECK:   util.global.store %[[CE1]], @[[HOISTED_1]] : i32
  // CHECK:   util.initializer.return
  // CHECK: }
}

// -----
// Ensures that non-leaf const-exprs (i.e. think broadcasts and other ops
// that should be considered const-expr but that you never want to hoist as
// a leaf) are hoisted internal to a const-expr but are left as-is at the leaf.
// CHECK-LABEL: @hoist_non_leaf_const_expr
module @hoist_non_leaf_const_expr {
  // CHECK: util.global private @[[HOISTED:.*]] : i32
  // CHECK: func @main
  builtin.func @main() -> (i32) {
    // CHECK: %[[LOAD_HOISTED:.*]] = util.global.load @[[HOISTED]] : i32
    // CHECK: %[[RESULT:.*]] = "iree_unregistered.non_leaf_const_expr"(%hoisted)
    // CHECK: return %[[RESULT]]
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    %2 = "iree_unregistered.non_leaf_const_expr"(%0, %1) : (i32, i32) -> i32
    %3 = "iree_unregistered.const_expr"(%2) : (i32) -> i32
    %4 = "iree_unregistered.non_leaf_const_expr"(%3) : (i32) -> i32
    return %4 : i32
  }
  // CHECK: util.initializer {
  // CHECK:   %[[C0:.*]] = arith.constant 0 : i32
  // CHECK:   %[[C1:.*]] = arith.constant 1 : i32
  // CHECK:   %[[CE0:.*]] = "iree_unregistered.non_leaf_const_expr"(%[[C0]], %[[C1]])
  // CHECK:   %[[CE1:.*]] = "iree_unregistered.const_expr"(%[[CE0]])
  // CHECK:   util.global.store %[[CE1]], @[[HOISTED]] : i32
  // CHECK:   util.initializer.return
  // CHECK: }
}
