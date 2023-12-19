// RUN: iree-opt --split-input-file --iree-util-hoist-into-globals="max-size-increase-threshold=64" --allow-unregistered-dialect %s | FileCheck %s

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
// Tests a const-expr with disabled-hoist consumers due to ineligible.
// CHECK-LABEL: @hoist_const_expr_with_ineligible_consumer
module @hoist_const_expr_with_ineligible_consumer {
  // CHECK: util.global private @[[HOISTED_0:.*]] : i32
  // CHECK: func.func @main
  func.func @main() -> i32 {
    // CHECK-DAG: %[[LOAD_HOISTED_0:.*]] = util.global.load @[[HOISTED_0]] : i32
    // CHECK-DAG: %[[RESULT:.*]] = "iree_unregistered.var_expr"(%[[LOAD_HOISTED_0]])
    // CHECK: return %[[RESULT]]
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    %2 = "iree_unregistered.const_expr"(%0, %1) : (i32, i32) -> i32
    %3 = "iree_unregistered.var_expr"(%2) : (i32) -> i32
    return %3 : i32
  }
  // CHECK: util.initializer attributes {iree.compiler.consteval} {
  // CHECK:   %[[C0:.*]] = arith.constant 0 : i32
  // CHECK:   %[[C1:.*]] = arith.constant 1 : i32
  // CHECK:   %[[CE0:.*]] = "iree_unregistered.const_expr"(%[[C0]], %[[C1]])
  // CHECK:   util.global.store %[[CE0]], @[[HOISTED_0]] : i32
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
module @hoist_uses_within_control_flows {
  // CHECK: util.global private @[[HOISTED:.*]] : i32
  // CHECK: func.func @main
  func.func @main(%bound : i32) -> i32 {
    // CHECK: arith.constant 0
    // CHECK-NOT: arith.constant 2
    // CHECK-NOT: arith.constant 5
    %cst0 = arith.constant 0 : i32
    %cst2 = arith.constant 2 : i32
    %cst5 = arith.constant 5 : i32
    // CHECK: scf.while
    %res = scf.while (%iter = %cst0) : (i32) -> i32 {
      // CHECK: arith.cmpi
      %cond = arith.cmpi slt, %iter, %bound : i32
      scf.condition(%cond) %iter : i32
    // CHECK: } do {
    } do {
    // CHECK: ^bb0(%[[ARG1:.*]]: i32)
    ^bb0(%arg1: i32):
      // CHECK: %[[STEP:.*]] = util.global.load @[[HOISTED]] : i32
      // CHECK-NOT: arith.subi
      // CHECK: %[[NEXT:.*]] = arith.addi %[[STEP]], %[[ARG1]]
      // CHECK: scf.yield %[[NEXT]]
      %step = arith.subi %cst5, %cst2 : i32
      %next = arith.addi %step, %arg1 : i32
      scf.yield %next : i32
    }
    return %res : i32
  }
  // CHECK: util.initializer attributes {iree.compiler.consteval} {
  // CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : i32
  // CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : i32
  // CHECK-DAG:   %[[SUB:.*]] = arith.subi %[[C5]], %[[C2]] : i32
  // CHECK:       util.global.store %[[SUB]], @[[HOISTED]] : i32
  // CHECK:       util.initializer.return
  // CHECK: }
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

// -----
#map = affine_map<(d0, d1) -> (d0, d1)>
module @do_not_hoist_uses_within_dispatches {
  func.func @main() -> tensor<2x2xi32> {
    %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
    %1 = arith.constant dense<[[6, 7], [8,9]]> : tensor<2x2xi32>
    %expanded = tensor.expand_shape %0[[0, 1]] : tensor<4xi32> into tensor<2x2xi32>
    %2 = tensor.empty() : tensor<2x2xi32>
    %3 = flow.dispatch.region -> (tensor<2x2xi32>) {
      %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded, %1 : tensor<2x2xi32>, tensor<2x2xi32>) outs(%2 : tensor<2x2xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %13 = arith.addi %in, %in_0 : i32
        linalg.yield %13 : i32
      } -> tensor<2x2xi32>
      flow.return %4 : tensor<2x2xi32>
    }
    return %3 : tensor<2x2xi32>
  }
}
// CHECK-LABEL: @do_not_hoist_uses_within_dispatches
//       CHECK:   %[[CST:.+]] = arith.constant
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[CST]]
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.region
//       CHECK:     %[[ADD:.+]] = linalg.generic
//  CHECK-SAME:     %[[EXPANDED]]
//       CHECK:     flow.return %[[ADD]]
//       CHECK:   return %[[RESULT]]

// -----

// The --iree-util-const-expr-max-size-increase-threshold flag controls the
// maximum size increase (vs sum of size of it's roots) allowed for hoisting a
// constant expression. The threshold is set to 64 bytes in this test suite.
// In this test, the size increase is exactly 64 bytes, so the constant
// expression is hoisted.
// CHECK-LABEL: @hoist_no_significant_size_increase_const_expr
// CHECK: util.global
// CHECK: util.initializer
module @hoist_no_significant_size_increase_const_expr {
  func.func @main() -> (tensor<128xi8>) {
    %0 = arith.constant dense<0> : tensor<32xi8>
    %1 = arith.constant dense<0> : tensor<32xi8>
    %2 = "iree_unregistered.const_expr"(%0, %1)
    : (tensor<32xi8>, tensor<32xi8>) -> tensor<128xi8>
    return %2 : tensor<128xi8>
  }
}

// -----

// In this test, the size increase is 65 bytes, so the constant expression is
// not hoisted.
// CHECK-LABEL: @do_not_hoist_significant_size_increase_const_expr
// CHECK-NOT: util.global
// CHECK-NOT: util.initializer
module @do_not_hoist_significant_size_increase_const_expr {
  func.func @main() -> (tensor<129xi8>) {
    %0 = arith.constant dense<0> : tensor<32xi8>
    %1 = arith.constant dense<0> : tensor<32xi8>
    %2 = "iree_unregistered.const_expr"(%0, %1)
    : (tensor<32xi8>, tensor<32xi8>) -> tensor<129xi8>
    return %2 : tensor<129xi8>
  }
}

// -----

// The hoisting in this case is nested on the outer module, and the inner
// module is a different logical program, so we shouldn't hoist to the outer
// module.
// CHECK-LABEL: @nested_program_const_expr
// CHECK-NOT: util.global
// CHECK-NOT: util.initializer
module @nested_program_const_expr {
  module {
    func.func @main() -> (i32) {
      %0 = arith.constant 0 : i32
      %1 = arith.constant 1 : i32
      %2 = "iree_unregistered.const_expr"(%0, %1) : (i32, i32) -> i32
      return %2 : i32
    }
  }
}
