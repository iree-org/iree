// RUN: iree-opt --split-input-file --iree-global-optimization-hoist-constant-expressions --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @hoist_sub_byte_tensor_store
module @hoist_sub_byte_tensor_store {
  util.func public @main() -> (tensor<64xi4>) {
    %0 = arith.constant dense<3> : tensor<64xi32>
    %2 = "iree_unregistered.const_expr"(%0) : (tensor<64xi32>) -> tensor<64xi4>
    util.return %2 : tensor<64xi4>
  }
}

// CHECK: util.global private @{{.*}} : tensor<32xi8>
// CHECK: util.func public @main() -> tensor<64xi4>
// CHECK:   %[[GLOBAL_LD:.+]] = util.global.load @{{.*}} : tensor<32xi8>
// CHECK:   %[[ORIG_VAL:.+]] = flow.tensor.bitcast %[[GLOBAL_LD]] : tensor<32xi8> -> tensor<64xi4>
// CHECK:   util.return %[[ORIG_VAL]]

// CHECK: util.initializer attributes {iree.compiler.consteval}
// CHECK:   %[[CEXPR:.+]] = "iree_unregistered.const_expr"
// CHECK:   %[[CASTED_GLOBAL:.+]] = flow.tensor.bitcast %[[CEXPR]] : tensor<64xi4> -> tensor<32xi8>
// CHECK:   util.global.store %[[CASTED_GLOBAL]]
// CHECK:   util.return

// -----

// CHECK-LABEL: @hoist_tree_const_expr_i4
module @hoist_tree_const_expr_i4 {
  // CHECK: util.global private @[[HOISTED_1:.*]] : tensor<4xi8>
  // CHECK: util.global private @[[HOISTED_0:.*]] : tensor<4xi8>
  // CHECK: util.global private @latent_global : tensor<8xi4>
  util.global private @latent_global : tensor<8xi4>

  // CHECK: util.func public @main
  util.func public @main() -> (tensor<8xi4>, tensor<8xi4>, tensor<8xi4>) {
    // CHECK-DAG: %[[LOAD_HOISTED_0:.*]] = util.global.load @[[HOISTED_0]] : tensor<4xi8>
    // CHECK-DAG: %[[BITCAST_0:.*]] = flow.tensor.bitcast %[[LOAD_HOISTED_0]] : tensor<4xi8> -> tensor<8xi4>
    // CHECK-DAG: %[[LOAD_HOISTED_1:.*]] = util.global.load @[[HOISTED_1]] : tensor<4xi8>
    // CHECK-DAG: %[[BITCAST_1:.*]] = flow.tensor.bitcast %[[LOAD_HOISTED_1]] : tensor<4xi8> -> tensor<8xi4>
    // CHECK-DAG: %[[RESULT:.*]] = "iree_unregistered.var_expr"(%[[BITCAST_1]])
    // CHECK: util.return %[[BITCAST_0]], %[[BITCAST_1]], %[[RESULT]]
    %0 = arith.constant dense<0> : tensor<8xi4>
    %1 = arith.constant dense<1> : tensor<8xi4>
    %2 = "iree_unregistered.const_expr"(%0, %1) : (tensor<8xi4>, tensor<8xi4>) -> tensor<8xi4>
    %3 = util.global.load @latent_global : tensor<8xi4>
    %4 = "iree_unregistered.const_expr"(%2, %3) : (tensor<8xi4>, tensor<8xi4>) -> tensor<8xi4>
    %5 = "iree_unregistered.var_expr"(%4) : (tensor<8xi4>) -> tensor<8xi4>
    util.return %2, %4, %5 : tensor<8xi4>, tensor<8xi4>, tensor<8xi4>
  }
  // CHECK: util.initializer attributes {iree.compiler.consteval} {
  // CHECK:   %[[C0:.*]] = arith.constant dense<0> : tensor<8xi4>
  // CHECK:   %[[C1:.*]] = arith.constant dense<1> : tensor<8xi4>
  // CHECK:   %[[CE0:.*]] = "iree_unregistered.const_expr"(%[[C0]], %[[C1]])
  // CHECK:   %[[BITCAST_2:.*]] = flow.tensor.bitcast %[[CE0]] : tensor<8xi4> -> tensor<4xi8>
  // CHECK:   util.global.store %[[BITCAST_2]], @[[HOISTED_0]] : tensor<4xi8>
  // CHECK:   util.return
  // CHECK: }
  // CHECK: util.initializer attributes {iree.compiler.consteval} {
  // CHECK:   %[[LOAD_HOISTED_0:.*]] = util.global.load @[[HOISTED_0]] : tensor<4xi8>
  // CHECK:   %[[BITCAST_3:.*]] = flow.tensor.bitcast %[[LOAD_HOISTED_0]] : tensor<4xi8> -> tensor<8xi4>
  // CHECK:   %[[LOAD_LATENT_GLOBAL:.*]] = util.global.load @latent_global : tensor<8xi4>
  // CHECK:   %[[CE1:.*]] = "iree_unregistered.const_expr"(%[[BITCAST_3]], %[[LOAD_LATENT_GLOBAL]])
  // CHECK:   %[[BITCAST_4:.*]] = flow.tensor.bitcast %[[CE1]] : tensor<8xi4> -> tensor<4xi8>
  // CHECK:   util.global.store %[[BITCAST_4]], @[[HOISTED_1]] : tensor<4xi8>
  // CHECK:   util.return
  // CHECK: }
}

// -----

// CHECK-LABEL: @hoist_sub_byte_tensor_transitive
// CHECK: util.global
module @hoist_sub_byte_tensor_transitive {
  util.func public @main() -> (i32) {
    %0 = arith.constant dense<3> : tensor<i4>
    %2 = "iree_unregistered.const_expr"(%0) : (tensor<i4>) -> i32
    util.return %2 : i32
  }
}
// We do not need to cast for transitive sub-byte values.
// CHECK-NOT: flow.tensor.bitcast

// -----

// CHECK-LABEL: @hoist_sub_byte_aligned_scalar_transitive
// CHECK-NOT: util.global
module @hoist_sub_byte_aligned_scalar_transitive {
 func.func @main() -> i4 {
    %c1_i4 = arith.constant 1 : i4
    %0 = "iree_unregistered.const_expr"(%c1_i4) : (i4) -> i4
    return %0 : i4
  }
}

// -----

// CHECK-LABEL: @hoist_constant_pack_computation
// CHECK: util.global
module @hoist_constant_pack_computation {
  func.func @main() -> tensor<4x1x16x2xi4> {
  %pad = arith.constant 5 : i4
  %val1 = stablehlo.constant dense<3> : tensor<7x15xi4>
  %val2 = tensor.empty() : tensor<4x1x16x2xi4>
  %ret = tensor.pack %val1 padding_value(%pad : i4) inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %val2 : tensor<7x15xi4> -> tensor<4x1x16x2xi4>
  return %ret : tensor<4x1x16x2xi4>
 }
}

// -----

// We should not hoist metadata ops alone.
// CHECK-LABEL: @do_not_hoist_metadata_leaf
// CHECK-NOT: util.global
module @do_not_hoist_metadata_leaf {
  util.func public @main() -> (tensor<1xi32>) {
    %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi8>
    %1 = flow.tensor.bitcast %0 : tensor<4xi8> -> tensor<1xi32>
    util.return %1 : tensor<1xi32>
  }
}

