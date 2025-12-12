// RUN: iree-opt --split-input-file --iree-global-optimization-hoist-constant-expressions --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @hoist_sub_byte_tensor_store
module @hoist_sub_byte_tensor_store {
  // CHECK: util.global private @{{.*}} : tensor<32xi8>
  // CHECK: util.initializer
  // CHECK:   %[[CEXPR:.+]] = "iree_unregistered.const_expr"
  // CHECK:   %[[CASTED_GLOBAL:.+]] = iree_tensor_ext.bitcast %[[CEXPR]] : tensor<64xi4> -> tensor<32xi8>
  // CHECK:   util.global.store %[[CASTED_GLOBAL]]
  // CHECK:   util.return

  // CHECK: util.func public @main() -> tensor<64xi4>
  // CHECK:   %[[GLOBAL_LD:.+]] = util.global.load immutable @{{.*}} : tensor<32xi8>
  // CHECK:   %[[ORIG_VAL:.+]] = iree_tensor_ext.bitcast %[[GLOBAL_LD]] : tensor<32xi8> -> tensor<64xi4>
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
  // CHECK:   %[[BITCAST_2:.*]] = iree_tensor_ext.bitcast %[[CE0]] : tensor<8xi4> -> tensor<4xi8>
  // CHECK:   util.global.store %[[BITCAST_2]], @[[HOISTED_0]] : tensor<4xi8>
  // CHECK:   util.return

  // CHECK: util.global private @[[HOISTED_1:.*]] : tensor<4xi8>
  // CHECK: util.initializer
  // CHECK:   %[[LOAD_HOISTED_0:.*]] = util.global.load @[[HOISTED_0]] : tensor<4xi8>
  // CHECK:   %[[BITCAST_3:.*]] = iree_tensor_ext.bitcast %[[LOAD_HOISTED_0]] : tensor<4xi8> -> tensor<8xi4>
  // CHECK:   %[[LOAD_LATENT_GLOBAL:.*]] = util.global.load @latent_global : tensor<8xi4>
  // CHECK:   %[[CE1:.*]] = "iree_unregistered.const_expr"(%[[BITCAST_3]], %[[LOAD_LATENT_GLOBAL]])
  // CHECK:   %[[BITCAST_4:.*]] = iree_tensor_ext.bitcast %[[CE1]] : tensor<8xi4> -> tensor<4xi8>
  // CHECK:   util.global.store %[[BITCAST_4]], @[[HOISTED_1]] : tensor<4xi8>
  // CHECK:   util.return

  // CHECK: util.func public @main
  util.func public @main() -> (tensor<8xi4>, tensor<8xi4>, tensor<8xi4>) {
    // CHECK-DAG: %[[LOAD_HOISTED_0:.*]] = util.global.load immutable @[[HOISTED_0]] : tensor<4xi8>
    // CHECK-DAG: %[[BITCAST_0:.*]] = iree_tensor_ext.bitcast %[[LOAD_HOISTED_0]] : tensor<4xi8> -> tensor<8xi4>
    // CHECK-DAG: %[[LOAD_HOISTED_1:.*]] = util.global.load immutable @[[HOISTED_1]] : tensor<4xi8>
    // CHECK-DAG: %[[BITCAST_1:.*]] = iree_tensor_ext.bitcast %[[LOAD_HOISTED_1]] : tensor<4xi8> -> tensor<8xi4>
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
  // CHECK-NOT: iree_tensor_ext.bitcast
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
    %ret = linalg.pack %val1 padding_value(%pad : i4) inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %val2 : tensor<7x15xi4> -> tensor<4x1x16x2xi4>
    util.return %ret : tensor<4x1x16x2xi4>
  }
}

// -----

// We should not hoist metadata ops alone.

// CHECK-LABEL: @do_not_hoist_metadata_leaf
module @do_not_hoist_metadata_leaf {
  // CHECK-NOT: util.global
  util.func public @flow_main() -> tensor<1xi32> {
    %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi8>
    %1 = flow.tensor.bitcast %0 : tensor<4xi8> -> tensor<1xi32>
    util.return %1 : tensor<1xi32>
  }
  util.func public @tensor_ext_main() -> tensor<1xi32> {
    %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi8>
    %1 = iree_tensor_ext.bitcast %0 : tensor<4xi8> -> tensor<1xi32>
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
  //      CHECK: util.global private @device
  util.global private @device : !hal.device
  //      CHECK: util.global private @[[HOISTED:[a-z0-9_]+]]
  // CHECK-SAME:   stream.affinity = #hal.device.affinity<@device>
  //      CHECK: util.initializer
  // CHECK-SAME:   stream.affinity = #hal.device.affinity<@device>
  util.func public @main() -> tensor<i32> attributes {
    stream.affinity = #hal.device.affinity<@device>
  } {
    %0 = arith.constant dense<3> : tensor<i32>
    %1 = "iree_unregistered.const_expr"(%0) : (tensor<i32>) -> tensor<i32>
    util.return %1 : tensor<i32>
  }
}

// -----

// CHECK-LABEL: @hoist_index
module @hoist_index {
  // CHECK: util.global private @[[HOISTED:.*]] : i64
  // CHECK: util.initializer
  // CHECK:   %[[C0:.*]] = arith.constant 0 : index
  // CHECK:   %[[CEXPR:.*]] = "iree_unregistered.const_expr"(%[[C0]])
  // CHECK:   %[[CAST:.*]] = arith.index_cast %[[CEXPR]] : index to i64
  // CHECK:   util.global.store %[[CAST]], @[[HOISTED]] : i64
  // CHECK:   util.return

  // CHECK: util.func public @main() -> index
  // CHECK:   %[[GLOBAL_LD:.*]] = util.global.load immutable @[[HOISTED]] : i64
  // CHECK:   %[[ORIG_VAL:.*]] = arith.index_cast %[[GLOBAL_LD]] : i64 to index
  // CHECK:   util.return %[[ORIG_VAL]]
  util.func public @main() -> (index) {
    %0 = arith.constant 0 : index
    %1 = "iree_unregistered.const_expr"(%0) : (index) -> index
    util.return %1 : index
  }
}

// -----

// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing<>
// CHECK-DAG:   #[[$ENCODING_WITH_TYPE:.+]] = #iree_encoding.testing<original_element_type = i4>
// CHECK-LABEL: @hoist_subbyte_with_encoding
#encoding = #iree_encoding.testing<>
module @hoist_subbyte_with_encoding {
  // The global stores i8 with encoding that tracks the original i4 type.
  // CHECK: util.global private @[[HOISTED:.*]] : tensor<32xi8, #[[$ENCODING_WITH_TYPE]]>
  // CHECK: util.initializer
  // CHECK:   %[[CST:.*]] = arith.constant dense<3> : tensor<64xi4>
  // CHECK:   %[[ENCODE:.*]] = flow.tensor.encode %[[CST]] : tensor<64xi4> -> tensor<64xi4, #[[$ENCODING]]>
  // CHECK:   %[[CEXPR:.*]] = "iree_unregistered.const_expr"(%[[ENCODE]])
  // CHECK:   %[[CAST:.*]] = iree_tensor_ext.bitcast %[[CEXPR]] : tensor<64xi4, #[[$ENCODING]]> -> tensor<32xi8, #[[$ENCODING_WITH_TYPE]]>
  // CHECK:   util.global.store %[[CAST]], @[[HOISTED]] : tensor<32xi8, #[[$ENCODING_WITH_TYPE]]>
  // CHECK:   util.return

  // CHECK: util.func public @main() -> tensor<64xi4, #[[$ENCODING]]>
  // CHECK:   %[[GLOBAL_LD:.*]] = util.global.load immutable @[[HOISTED]] : tensor<32xi8, #[[$ENCODING_WITH_TYPE]]>
  // CHECK:   %[[ORIG_VAL:.*]] = iree_tensor_ext.bitcast %[[GLOBAL_LD]] : tensor<32xi8, #[[$ENCODING_WITH_TYPE]]> -> tensor<64xi4, #[[$ENCODING]]>
  // CHECK:   util.return %[[ORIG_VAL]]
  util.func public @main() -> tensor<64xi4, #encoding> {
    %0 = arith.constant dense<3> : tensor<64xi4>
    %1 = flow.tensor.encode %0 : tensor<64xi4> -> tensor<64xi4, #encoding>
    %2 = "iree_unregistered.const_expr"(%1) : (tensor<64xi4, #encoding>) -> tensor<64xi4, #encoding>
    util.return %2 : tensor<64xi4, #encoding>
  }
}

// -----

// Negative test: tensor<63xi4> has 63*4=252 bits, not divisible by 8.
// The bitcast conversion to i8 fails, so the tensor is hoisted without
// byte-packing (stays as i4, no original_element_type added).
// CHECK-DAG: #[[$ENCODING_UNALIGNED:.+]] = #iree_encoding.testing<>
// CHECK-LABEL: @hoist_subbyte_unaligned_with_encoding
#encoding_unaligned = #iree_encoding.testing<>
module @hoist_subbyte_unaligned_with_encoding {
  // Since 63*4=252 bits is not byte-aligned, conversion fails.
  // The tensor is still hoisted but WITHOUT converting to i8 storage.
  // Note: no original_element_type field because no bitcast occurred.
  // CHECK: util.global private @[[HOISTED:.*]] : tensor<63xi4, #[[$ENCODING_UNALIGNED]]>
  // CHECK: util.initializer
  // CHECK:   %[[CST:.*]] = arith.constant dense<3> : tensor<63xi4>
  // CHECK:   %[[ENCODE:.*]] = flow.tensor.encode %[[CST]] : tensor<63xi4> -> tensor<63xi4, #[[$ENCODING_UNALIGNED]]>
  // CHECK:   %[[CEXPR:.*]] = "iree_unregistered.const_expr"(%[[ENCODE]])
  // CHECK:   util.global.store %[[CEXPR]], @[[HOISTED]] : tensor<63xi4, #[[$ENCODING_UNALIGNED]]>
  // CHECK:   util.return
  //
  // CHECK: util.func public @main() -> tensor<63xi4, #[[$ENCODING_UNALIGNED]]>
  // CHECK:   %[[GLOBAL_LD:.*]] = util.global.load immutable @[[HOISTED]] : tensor<63xi4, #[[$ENCODING_UNALIGNED]]>
  // CHECK:   util.return %[[GLOBAL_LD]]
  util.func public @main() -> tensor<63xi4, #encoding_unaligned> {
    %0 = arith.constant dense<3> : tensor<63xi4>
    %1 = flow.tensor.encode %0 : tensor<63xi4> -> tensor<63xi4, #encoding_unaligned>
    %2 = "iree_unregistered.const_expr"(%1) : (tensor<63xi4, #encoding_unaligned>) -> tensor<63xi4, #encoding_unaligned>
    util.return %2 : tensor<63xi4, #encoding_unaligned>
  }
}
