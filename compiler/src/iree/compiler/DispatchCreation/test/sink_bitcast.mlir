// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-dispatch-creation-sink-bitcast,canonicalize))" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @propagateBitCastThroughExpandShape
// CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: tensor<?x32xf4E2M1FN>
// CHECK-SAME: %[[D0:[A-Za-z0-9]+]]: index
// CHECK-SAME: %[[D1:[A-Za-z0-9]+]]: index
func.func @propagateBitCastThroughExpandShape(%arg0: tensor<?x32xf4E2M1FN>, %d0: index, %d1: index) -> tensor<4x?x512x16xi8> {
  // CHECK: %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1, 2], [3]]
  // CHECK-SAME: output_shape [4, %[[D1]], 512, 32]
  // CHECK-SAME: tensor<?x32xf4E2M1FN> into tensor<4x?x512x32xf4E2M1FN>
  %0 = iree_tensor_ext.bitcast %arg0 : tensor<?x32xf4E2M1FN>{%d0} -> tensor<?x16xi8>{%d0}
  %1 = tensor.expand_shape %0 [[0, 1, 2], [3]] output_shape [4, %d1, 512, 16]
      : tensor<?x16xi8> into tensor<4x?x512x16xi8>
  // CHECK: %[[BITCAST:.+]] = iree_tensor_ext.bitcast %[[EXPAND]]
  // CHECK-SAME: tensor<4x?x512x32xf4E2M1FN>
  // CHECK-SAME: tensor<4x?x512x16xi8>
  // CHECK: return %[[BITCAST]]
  return %1 : tensor<4x?x512x16xi8>
}

// -----

// CHECK-LABEL: @propagate1dBitCastThroughExpandShape
// CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: tensor<?xf8E8M0FNU>
// CHECK-SAME: %[[D0:[A-Za-z0-9]+]]: index
// CHECK-SAME: %[[D1:[A-Za-z0-9]+]]: index
func.func @propagate1dBitCastThroughExpandShape(%arg0: tensor<?xf8E8M0FNU>, %d0: index, %d1: index) -> tensor<4x?x512xi8> {
  // The bitcast should propagate through expand_shape.
  // CHECK: %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1, 2]]
  // CHECK-SAME: output_shape [4, %[[D1]], 512]
  // CHECK-SAME: tensor<?xf8E8M0FNU> into tensor<4x?x512xf8E8M0FNU>
  // CHECK: %[[BITCAST:.+]] = iree_tensor_ext.bitcast %[[EXPAND]]
  // CHECK-SAME: tensor<4x?x512xf8E8M0FNU>
  // CHECK-SAME: tensor<4x?x512xi8>
  // CHECK: return %[[BITCAST]]
  %0 = iree_tensor_ext.bitcast %arg0 : tensor<?xf8E8M0FNU>{%d0} -> tensor<?xi8>{%d0}
  %1 = tensor.expand_shape %0 [[0, 1, 2]] output_shape [4, %d1, 512]
      : tensor<?xi8> into tensor<4x?x512xi8>
  return %1 : tensor<4x?x512xi8>
}

// -----

// CHECK-LABEL: @propagateBitCastThroughCollapseShape
// CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: tensor<4x?x512x32xf4E2M1FN>
// CHECK-SAME: %[[D0:[A-Za-z0-9]+]]: index
func.func @propagateBitCastThroughCollapseShape(%arg0: tensor<4x?x512x32xf4E2M1FN>, %d0: index) -> tensor<4x?x8192xi8> {
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0], [1], [2, 3]]
  // CHECK-SAME: tensor<4x?x512x32xf4E2M1FN> into tensor<4x?x16384xf4E2M1FN>
  %0 = iree_tensor_ext.bitcast %arg0 : tensor<4x?x512x32xf4E2M1FN>{%d0} -> tensor<4x?x512x16xi8>{%d0}
  %1 = tensor.collapse_shape %0 [[0], [1], [2, 3]]
      : tensor<4x?x512x16xi8> into tensor<4x?x8192xi8>
  // CHECK: %[[BITCAST:.+]] = iree_tensor_ext.bitcast %[[COLLAPSE]]
  // CHECK-SAME: tensor<4x?x16384xf4E2M1FN>
  // CHECK-SAME: tensor<4x?x8192xi8>
  // CHECK: return %[[BITCAST]]
  return %1 : tensor<4x?x8192xi8>
}

// -----

// Test that consecutive bitcasts are folded after propagation through reshapes.
// CHECK-LABEL: @bitcastExpandCollapseChain
// CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: tensor<?x32xf4E2M1FN>
// CHECK-SAME: %[[D1:[A-Za-z0-9]+]]: index
func.func @bitcastExpandCollapseChain(%arg0: tensor<?x32xf4E2M1FN>, %d1: index) -> tensor<4x?x16384xf4E2M1FN> {
  // d0 = d1 * 2048 (the first dim of the input tensor)
  %c2048 = arith.constant 2048 : index
  %d0 = arith.muli %d1, %c2048 : index
  // The two bitcasts should be propagated through and cancel out, leaving just
  // the expand_shape and collapse_shape operating on the original f4E2M1FN type.
  // CHECK-DAG: %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1, 2], [3]]
  // CHECK-SAME: output_shape [4, %[[D1]], 512, 32]
  // CHECK-SAME: tensor<?x32xf4E2M1FN> into tensor<4x?x512x32xf4E2M1FN>
  // CHECK-DAG: %[[COLLAPSE:.+]] = tensor.collapse_shape %[[EXPAND]] {{\[\[}}0], [1], [2, 3]]
  // CHECK-SAME: tensor<4x?x512x32xf4E2M1FN> into tensor<4x?x16384xf4E2M1FN>
  // CHECK: return %[[COLLAPSE]]
  %0 = iree_tensor_ext.bitcast %arg0 : tensor<?x32xf4E2M1FN>{%d0} -> tensor<?x16xi8>{%d0}
  %1 = tensor.expand_shape %0 [[0, 1, 2], [3]] output_shape [4, %d1, 512, 16]
      : tensor<?x16xi8> into tensor<4x?x512x16xi8>
  %2 = tensor.collapse_shape %1 [[0], [1], [2, 3]]
      : tensor<4x?x512x16xi8> into tensor<4x?x8192xi8>
  %3 = iree_tensor_ext.bitcast %2 : tensor<4x?x8192xi8>{%d1} -> tensor<4x?x16384xf4E2M1FN>{%d1}
  return %3 : tensor<4x?x16384xf4E2M1FN>
}
