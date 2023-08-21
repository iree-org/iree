// RUN: iree-opt --split-input-file --iree-util-hoist-into-globals --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @hoist_simple_const_expr
module @hoist_simple_const_expr {
  // CHECK: util.global private @[[HOISTED_SYM:.*]] : i32
  // CHECK: func.func @main
  func.func @main() -> (i32) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    // CHECK-NOT: arith.constant
    // CHECK-NOT: iree_unregistered.const_expr
    // CHECK: %[[VAL:.*]] = util.global.load @[[HOISTED_SYM]] : i32
    // CHECK: return %[[VAL]]
    %2 = "iree_unregistered.const_expr"(%0, %1) : (i32, i32) -> i32
    return %2 : i32
  }
  // CHECK: util.initializer attributes {iree.compiler.consteval} {
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
// CHECK: func.func @main
// CHECK: %[[VAL:.*]] = "iree_unregistered.var_expr"
// CHECK: return %[[VAL]]
// CHECK-NOT: util.initializer
module @do_not_hoist_variable_op {
  func.func @main() -> (i32) {
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
  func.func @main(%arg0 : i32) -> (i32) {
    %0 = arith.constant 0 : i32
    %2 = "iree_unregistered.const_expr"(%0, %arg0) : (i32, i32) -> i32
    return %2 : i32
  }
}

// -----
// CHECK-LABEL: @do_not_hoist_sub_byte_aligned_scalar_leaf
// CHECK-NOT: util.global
// CHECK-NOT: util.initializer
module @do_not_hoist_sub_byte_aligned_scalar_leaf {
  func.func @main() -> (i32) {
    %0 = arith.constant 1 : i1
    %2 = "iree_unregistered.var_expr"(%0) : (i1) -> i32
    return %2 : i32
  }
}

// -----
// CHECK-LABEL: @do_not_hoist_sub_byte_aligned_tensor_leaf
// CHECK-NOT: util.global
// CHECK-NOT: util.initializer
module @do_not_hoist_sub_byte_aligned_tensor_leaf {
  func.func @main() -> (i32) {
    %0 = arith.constant dense<true> : tensor<i1>
    %2 = "iree_unregistered.var_expr"(%0) : (tensor<i1>) -> i32
    return %2 : i32
  }
}

// -----
// CHECK-LABEL: @hoist_sub_byte_aligned_scalar_transitive
// CHECK: util.global private {{.*}} : i32
// Can hoist a const-expr tree that transitively includes sub-byte aligned
// values.
module @hoist_sub_byte_aligned_scalar_transitive {
  func.func @main() -> (i32) {
    %0 = arith.constant 1 : i1
    %2 = "iree_unregistered.const_expr"(%0) : (i1) -> i32
    return %2 : i32
  }
}

// -----
// CHECK-LABEL: @do_not_hoist_sub_byte_tensor_transitive
// CHECK-NOT: util.global
// We do not yet support constexpr of sub-byte types that are 
// Can hoist a const-expr tree that transitively includes sub-byte aligned
// values.
module @do_not_hoist_sub_byte_tensor_transitive {
  func.func @main() -> (i32) {
    %0 = arith.constant dense<3> : tensor<i4>
    %2 = "iree_unregistered.const_expr"(%0) : (tensor<i4>) -> i32
    return %2 : i32
  }
}

// -----
// CHECK-LABEL: @hoist_i1_tensor_transitive
// CHECK: util.global private {{.*}} : i32
// We presently expand i1 -> i8 for legacy reasons. As such, we support
// it, even though we don't generally support sub-byte constexprs.
module @hoist_i1_tensor_transitive {
  func.func @main() -> (i32) {
    %0 = arith.constant dense<true> : tensor<i1>
    %2 = "iree_unregistered.const_expr"(%0) : (tensor<i1>) -> i32
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

  // CHECK: func.func @main
  func.func @main() -> (i32, i32, i32) {
    // CHECK-DAG: %[[LOAD_HOISTED_0:.*]] = util.global.load @[[HOISTED_0]] : i32
    // CHECK-DAG: %[[LOAD_HOISTED_1:.*]] = util.global.load @[[HOISTED_1]] : i32
    // CHECK-DAG: %[[RESULT:.*]] = "iree_unregistered.var_expr"(%[[LOAD_HOISTED_1]])
    // CHECK: return %[[LOAD_HOISTED_0]], %[[LOAD_HOISTED_1]], %[[RESULT]]
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    %2 = "iree_unregistered.const_expr"(%0, %1) : (i32, i32) -> i32
    %3 = util.global.load @latent_global : i32
    %4 = "iree_unregistered.const_expr"(%2, %3) : (i32, i32) -> i32
    %5 = "iree_unregistered.var_expr"(%4) : (i32) -> i32
    return %2, %4, %5 : i32, i32, i32
  }
  // CHECK: util.initializer attributes {iree.compiler.consteval} {
  // CHECK:   %[[C0:.*]] = arith.constant 0 : i32
  // CHECK:   %[[C1:.*]] = arith.constant 1 : i32
  // CHECK:   %[[CE0:.*]] = "iree_unregistered.const_expr"(%[[C0]], %[[C1]])
  // CHECK:   util.global.store %[[CE0]], @[[HOISTED_0]] : i32
  // CHECK:   util.initializer.return
  // CHECK: }
  // CHECK: util.initializer attributes {iree.compiler.consteval} {
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
  // CHECK: func.func @main
  func.func @main() -> (i32) {
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
  // CHECK: util.initializer attributes {iree.compiler.consteval} {
  // CHECK:   %[[C0:.*]] = arith.constant 0 : i32
  // CHECK:   %[[C1:.*]] = arith.constant 1 : i32
  // CHECK:   %[[CE0:.*]] = "iree_unregistered.non_leaf_const_expr"(%[[C0]], %[[C1]])
  // CHECK:   %[[CE1:.*]] = "iree_unregistered.const_expr"(%[[CE0]])
  // CHECK:   util.global.store %[[CE1]], @[[HOISTED]] : i32
  // CHECK:   util.initializer.return
  // CHECK: }
}

// -----
// CHECK-LABEL: @hoist_implicit_capture
module @hoist_implicit_capture {
  // CHECK: util.global private @[[HOISTED_SYM:.*]] : i32
  // CHECK: func.func @main
  func.func @main() -> (i32) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    // CHECK-NOT: arith.constant
    // CHECK-NOT: iree_unregistered.const_expr
    // CHECK: %[[VAL:.*]] = util.global.load @[[HOISTED_SYM]] : i32
    // CHECK: return %[[VAL]]
    %2 = "iree_unregistered.const_expr"(%0) ({
    ^bb0(%inner0 : i32):
      %3 = arith.addi %inner0, %1 : i32
      "iree_unregistered.yield"(%3) : (i32) -> i32
    }) : (i32) -> i32
    return %2 : i32
  }
  // Key checks: arith.constant 1 gets pulled in to the initializer
  // and the reference is updated correctly in the custom op region.
  // CHECK: util.initializer attributes {iree.compiler.consteval} {
  // CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : i32
  // CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : i32
  // CHECK:       %[[CE0:.*]] = "iree_unregistered.const_expr"(%[[C0]])
  // CHECK:         ^bb0(%[[B0:.*]]: i32):
  // CHECK:         arith.addi %[[B0]], %[[C1]]
  // CHECK:       util.global.store %[[CE0]], @[[HOISTED_SYM]] : i32
  // CHECK:       util.initializer.return
  // CHECK: }
}

// -----
// CHECK-LABEL: @do_not_hoist_non_value_type_results
// CHECK-NOT: util.global
// CHECK-NOT: util.initializer
module @do_not_hoist_non_value_type_results {
  func.func @main() -> (!iree_unregistered.unknown_type) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    %2 = "iree_unregistered.const_expr"(%0, %1) : (i32, i32) -> !iree_unregistered.unknown_type
    return %2 : !iree_unregistered.unknown_type
  }
}

// -----

module @do_not_hoist_uses_within_dispatches {
  func.func @main() -> (tensor<i32>) {
    %cst = arith.constant dense<[2, 3]>: tensor<2xi32>
    %result = flow.dispatch.region -> (tensor<i32>) {
      %slice = tensor.extract_slice %cst[0] [1] [1] : tensor<2xi32> to tensor<i32>
      flow.return %slice : tensor<i32>
    }
    return %result : tensor<i32>
  }
}
// CHECK-LABEL: @do_not_hoist_uses_within_dispatches
//       CHECK:   %[[CST:.+]] = arith.constant
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.region
//       CHECK:     %[[SLICE:.+]] = tensor.extract_slice %[[CST]]
//       CHECK:     flow.return %[[SLICE]]
//       CHECK:   return %[[RESULT]]
