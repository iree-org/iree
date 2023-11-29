// RUN: iree-opt --split-input-file --iree-global-optimization-hoist-constant-expressions --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @hoist_sub_byte_tensor_store
module @hoist_sub_byte_tensor_store {
  func.func @main() -> (tensor<64xi4>) {
    %0 = arith.constant dense<3> : tensor<64xi32>
    %2 = "iree_unregistered.const_expr"(%0) : (tensor<64xi32>) -> tensor<64xi4>
    return %2 : tensor<64xi4>
  }
}

// CHECK: util.global private @{{.*}} : tensor<32xi8>
// CHECK: func.func @main() -> tensor<64xi4>
// CHECK:   %[[GLOBAL_LD:.+]] = util.global.load @{{.*}} : tensor<32xi8>
// CHECK:   %[[ORIG_VAL:.+]] = flow.tensor.bitcast %[[GLOBAL_LD]] : tensor<32xi8> -> tensor<64xi4>
// CHECK:   return %[[ORIG_VAL]]

// CHECK: util.initializer attributes {iree.compiler.consteval}
// CHECK:   %[[CEXPR:.+]] = "iree_unregistered.const_expr"
// CHECK:   %[[CASTED_GLOBAL:.+]] = flow.tensor.bitcast %[[CEXPR]] : tensor<64xi4> -> tensor<32xi8>
// CHECK:   util.global.store %[[CASTED_GLOBAL]]
// CHECK:   util.initializer.return

// -----

// CHECK-LABEL: @hoist_tree_const_expr_i4
module @hoist_tree_const_expr_i4 {
  // CHECK: util.global private @[[HOISTED_1:.*]] : tensor<4xi8>
  // CHECK: util.global private @[[HOISTED_0:.*]] : tensor<4xi8>
  // CHECK: util.global private @latent_global : tensor<8xi4>
  util.global private @latent_global : tensor<8xi4>

  // CHECK: func.func @main
  func.func @main() -> (tensor<8xi4>, tensor<8xi4>, tensor<8xi4>) {
    // CHECK-DAG: %[[LOAD_HOISTED_0:.*]] = util.global.load @[[HOISTED_0]] : tensor<4xi8>
    // CHECK-DAG: %[[BITCAST_0:.*]] = flow.tensor.bitcast %[[LOAD_HOISTED_0]] : tensor<4xi8> -> tensor<8xi4>
    // CHECK-DAG: %[[LOAD_HOISTED_1:.*]] = util.global.load @[[HOISTED_1]] : tensor<4xi8>
    // CHECK-DAG: %[[BITCAST_1:.*]] = flow.tensor.bitcast %[[LOAD_HOISTED_1]] : tensor<4xi8> -> tensor<8xi4>
    // CHECK-DAG: %[[RESULT:.*]] = "iree_unregistered.var_expr"(%[[BITCAST_1]])
    // CHECK: return %[[BITCAST_0]], %[[BITCAST_1]], %[[RESULT]]
    %0 = arith.constant dense<0> : tensor<8xi4>
    %1 = arith.constant dense<1> : tensor<8xi4>
    %2 = "iree_unregistered.const_expr"(%0, %1) : (tensor<8xi4>, tensor<8xi4>) -> tensor<8xi4>
    %3 = util.global.load @latent_global : tensor<8xi4>
    %4 = "iree_unregistered.const_expr"(%2, %3) : (tensor<8xi4>, tensor<8xi4>) -> tensor<8xi4>
    %5 = "iree_unregistered.var_expr"(%4) : (tensor<8xi4>) -> tensor<8xi4>
    return %2, %4, %5 : tensor<8xi4>, tensor<8xi4>, tensor<8xi4>
  }
  // CHECK: util.initializer attributes {iree.compiler.consteval} {
  // CHECK:   %[[C0:.*]] = arith.constant dense<0> : tensor<8xi4>
  // CHECK:   %[[C1:.*]] = arith.constant dense<1> : tensor<8xi4>
  // CHECK:   %[[CE0:.*]] = "iree_unregistered.const_expr"(%[[C0]], %[[C1]])
  // CHECK:   %[[BITCAST_2:.*]] = flow.tensor.bitcast %[[CE0]] : tensor<8xi4> -> tensor<4xi8>
  // CHECK:   util.global.store %[[BITCAST_2]], @[[HOISTED_0]] : tensor<4xi8>
  // CHECK:   util.initializer.return
  // CHECK: }
  // CHECK: util.initializer attributes {iree.compiler.consteval} {
  // CHECK:   %[[LOAD_HOISTED_0:.*]] = util.global.load @[[HOISTED_0]] : tensor<4xi8>
  // CHECK:   %[[BITCAST_3:.*]] = flow.tensor.bitcast %[[LOAD_HOISTED_0]] : tensor<4xi8> -> tensor<8xi4>
  // CHECK:   %[[LOAD_LATENT_GLOBAL:.*]] = util.global.load @latent_global : tensor<8xi4>
  // CHECK:   %[[CE1:.*]] = "iree_unregistered.const_expr"(%[[BITCAST_3]], %[[LOAD_LATENT_GLOBAL]])
  // CHECK:   %[[BITCAST_4:.*]] = flow.tensor.bitcast %[[CE1]] : tensor<8xi4> -> tensor<4xi8>
  // CHECK:   util.global.store %[[BITCAST_4]], @[[HOISTED_1]] : tensor<4xi8>
  // CHECK:   util.initializer.return
  // CHECK: }
}

// -----

// CHECK-LABEL: @hoist_sub_byte_tensor_transitive
// CHECK: util.global
module @hoist_sub_byte_tensor_transitive {
  func.func @main() -> (i32) {
    %0 = arith.constant dense<3> : tensor<i4>
    %2 = "iree_unregistered.const_expr"(%0) : (tensor<i4>) -> i32
    return %2 : i32
  }
}
// We do not need to cast for transitive sub-byte values.
// CHECK-NOT: flow.tensor.bitcast
