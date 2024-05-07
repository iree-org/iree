// RUN: iree-opt --split-input-file --iree-global-optimization-hoist-constant-expressions --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @hoist_sub_byte_tensor_store
module @hoist_sub_byte_tensor_store {
  // CHECK: util.global private @{{.*}} : tensor<32xi8>
  // CHECK: util.initializer
  // CHECK:   %[[CEXPR:.+]] = "iree_unregistered.const_expr"
  // CHECK:   %[[CASTED_GLOBAL:.+]] = flow.tensor.bitcast %[[CEXPR]] : tensor<64xi4> -> tensor<32xi8>
  // CHECK:   util.global.store %[[CASTED_GLOBAL]]
  // CHECK:   util.return

  // CHECK: util.func public @main() -> tensor<64xi4>
  // CHECK:   %[[GLOBAL_LD:.+]] = util.global.load immutable @{{.*}} : tensor<32xi8>
  // CHECK:   %[[ORIG_VAL:.+]] = flow.tensor.bitcast %[[GLOBAL_LD]] : tensor<32xi8> -> tensor<64xi4>
  // CHECK:   util.return %[[ORIG_VAL]]
  util.func public @main() -> (tensor<64xi4>) {
    %0 = arith.constant dense<3> : tensor<64xi32>
    %2 = "iree_unregistered.const_expr"(%0) : (tensor<64xi32>) -> tensor<64xi4>
    util.return %2 : tensor<64xi4>
  }
}

// -----

// CHECK-LABEL: @hoist_tree_const_expr_i4
module @hoist_tree_const_expr_i4 {
  // CHECK: util.global private @latent_global : tensor<8xi4>
  util.global private @latent_global : tensor<8xi4>

  // CHECK: util.global private @[[HOISTED_0:.*]] : tensor<4xi8>
  // CHECK: util.initializer
  // CHECK:   %[[C0:.*]] = arith.constant dense<0> : tensor<8xi4>
  // CHECK:   %[[C1:.*]] = arith.constant dense<1> : tensor<8xi4>
  // CHECK:   %[[CE0:.*]] = "iree_unregistered.const_expr"(%[[C0]], %[[C1]])
  // CHECK:   %[[BITCAST_2:.*]] = flow.tensor.bitcast %[[CE0]] : tensor<8xi4> -> tensor<4xi8>
  // CHECK:   util.global.store %[[BITCAST_2]], @[[HOISTED_0]] : tensor<4xi8>
  // CHECK:   util.return

  // CHECK: util.global private @[[HOISTED_1:.*]] : tensor<4xi8>
  // CHECK: util.initializer
  // CHECK:   %[[LOAD_HOISTED_0:.*]] = util.global.load @[[HOISTED_0]] : tensor<4xi8>
  // CHECK:   %[[BITCAST_3:.*]] = flow.tensor.bitcast %[[LOAD_HOISTED_0]] : tensor<4xi8> -> tensor<8xi4>
  // CHECK:   %[[LOAD_LATENT_GLOBAL:.*]] = util.global.load @latent_global : tensor<8xi4>
  // CHECK:   %[[CE1:.*]] = "iree_unregistered.const_expr"(%[[BITCAST_3]], %[[LOAD_LATENT_GLOBAL]])
  // CHECK:   %[[BITCAST_4:.*]] = flow.tensor.bitcast %[[CE1]] : tensor<8xi4> -> tensor<4xi8>
  // CHECK:   util.global.store %[[BITCAST_4]], @[[HOISTED_1]] : tensor<4xi8>
  // CHECK:   util.return

  // CHECK: util.func public @main
  util.func public @main() -> (tensor<8xi4>, tensor<8xi4>, tensor<8xi4>) {
    // CHECK-DAG: %[[LOAD_HOISTED_0:.*]] = util.global.load immutable @[[HOISTED_0]] : tensor<4xi8>
    // CHECK-DAG: %[[BITCAST_0:.*]] = flow.tensor.bitcast %[[LOAD_HOISTED_0]] : tensor<4xi8> -> tensor<8xi4>
    // CHECK-DAG: %[[LOAD_HOISTED_1:.*]] = util.global.load immutable @[[HOISTED_1]] : tensor<4xi8>
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
}

// -----

// CHECK-LABEL: @hoist_sub_byte_tensor_transitive
module @hoist_sub_byte_tensor_transitive {
  // CHECK: util.global
  // CHECK: util.initializer
  // We do not need to cast for transitive sub-byte values.
  // CHECK-NOT: flow.tensor.bitcast
  util.func public @main() -> (i32) {
    %0 = arith.constant dense<3> : tensor<i4>
    %2 = "iree_unregistered.const_expr"(%0) : (tensor<i4>) -> i32
    util.return %2 : i32
  }
}

// -----

// CHECK-LABEL: @hoist_sub_byte_aligned_scalar_transitive
module @hoist_sub_byte_aligned_scalar_transitive {
  // CHECK-NOT: util.global
 util.func @main() -> i4 {
    %c1_i4 = arith.constant 1 : i4
    %0 = "iree_unregistered.const_expr"(%c1_i4) : (i4) -> i4
    util.return %0 : i4
  }
}

// -----

// CHECK-LABEL: @hoist_constant_pack_computation
module @hoist_constant_pack_computation {
  // CHECK: util.global
  util.func @main() -> tensor<4x1x16x2xi4> {
    %pad = arith.constant 5 : i4
    %val1 = stablehlo.constant dense<3> : tensor<7x15xi4>
    %val2 = tensor.empty() : tensor<4x1x16x2xi4>
    %ret = tensor.pack %val1 padding_value(%pad : i4) inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %val2 : tensor<7x15xi4> -> tensor<4x1x16x2xi4>
    util.return %ret : tensor<4x1x16x2xi4>
  }
}

// -----

// We should not hoist metadata ops alone.

// CHECK-LABEL: @do_not_hoist_metadata_leaf
module @do_not_hoist_metadata_leaf {
  // CHECK-NOT: util.global
  util.func public @main() -> tensor<1xi32> {
    %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi8>
    %1 = flow.tensor.bitcast %0 : tensor<4xi8> -> tensor<1xi32>
    util.return %1 : tensor<1xi32>
  }
}

// -----

// CHECK-LABEL: @hoist_inline_parameters
module @hoist_inline_parameters {
  //      CHECK: util.global private @[[HOISTED:.+]] : tensor<i32>
  //      CHECK: util.initializer {
  // CHECK-NEXT:   flow.tensor.constant #flow.parameter.named<"compile"::"constant_hoisted_0">
  // CHECK-NEXT:   "iree_unregistered.const_expr"
  util.func public @main() -> tensor<i32> {
    // CHECK: util.global.load immutable @[[HOISTED]]
    %parameter = flow.tensor.constant #flow.parameter.named<"compile"::"constant_hoisted_0"> : tensor<i32>
    %0 = "iree_unregistered.const_expr"(%parameter) : (tensor<i32>) -> tensor<i32>
    util.return %0 : tensor<i32>
  }
}

// -----

// Tests that hoistable attributes like the device affinity get cloned onto the
// global and initializer.

// CHECK-LABEL: @hoist_dialect_attrs
module @hoist_dialect_attrs {
  //      CHECK: util.global private @[[HOISTED:[a-z0-9_]+]]
  // CHECK-SAME:   hal.affinity = #hal.affinity.queue<[0, 1]>
  //      CHECK: util.initializer
  // CHECK-SAME:   hal.affinity = #hal.affinity.queue<[0, 1]>
  util.func public @main() -> tensor<i32> attributes {
    hal.affinity = #hal.affinity.queue<[0, 1]>
  } {
    %0 = arith.constant dense<3> : tensor<i32>
    %1 = "iree_unregistered.const_expr"(%0) : (tensor<i32>) -> tensor<i32>
    util.return %1 : tensor<i32>
  }
}
