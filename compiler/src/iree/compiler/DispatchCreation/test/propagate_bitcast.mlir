// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-dispatch-creation-bubble-up-expand-shapes,canonicalize))" --split-input-file %s | FileCheck %s

// Test bubbling expand_shape through bitcast: expand_shape(bitcast(x)) -> bitcast(expand_shape(x)).
// CHECK-LABEL: @bubbleExpandThroughBitCast
// CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: tensor<?x32xf4E2M1FN>
// CHECK-SAME: %[[D0:[A-Za-z0-9]+]]: index
// CHECK-SAME: %[[D1:[A-Za-z0-9]+]]: index
func.func @bubbleExpandThroughBitCast(%arg0: tensor<?x32xf4E2M1FN>, %d0: index, %d1: index) -> tensor<4x?x512x16xi8> {
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

// Test bubbling expand_shape through bitcast for rank-1 tensors.
// CHECK-LABEL: @bubbleExpand1dThroughBitCast
// CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: tensor<?xf8E8M0FNU>
// CHECK-SAME: %[[D0:[A-Za-z0-9]+]]: index
// CHECK-SAME: %[[D1:[A-Za-z0-9]+]]: index
func.func @bubbleExpand1dThroughBitCast(%arg0: tensor<?xf8E8M0FNU>, %d0: index, %d1: index) -> tensor<4x?x512xi8> {
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

// Test sinking collapse_shape through bitcast: bitcast(collapse_shape(x)) ->
// collapse_shape(bitcast(x)).
// CHECK-LABEL: @sinkCollapseThroughBitCast
// CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: tensor<4x?x512x32xf4E2M1FN>
// CHECK-SAME: %[[D0:[A-Za-z0-9]+]]: index
func.func @sinkCollapseThroughBitCast(%arg0: tensor<4x?x512x32xf4E2M1FN>, %d0: index) -> tensor<4x?x8192xi8> {
  // CHECK: %[[BITCAST:.+]] = iree_tensor_ext.bitcast %[[ARG0]]
  // CHECK-SAME: tensor<4x?x512x32xf4E2M1FN>
  // CHECK-SAME: tensor<4x?x512x16xi8>
  %0 = tensor.collapse_shape %arg0 [[0], [1], [2, 3]]
      : tensor<4x?x512x32xf4E2M1FN> into tensor<4x?x16384xf4E2M1FN>
  %1 = iree_tensor_ext.bitcast %0 : tensor<4x?x16384xf4E2M1FN>{%d0} -> tensor<4x?x8192xi8>{%d0}
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %[[BITCAST]] {{\[\[}}0], [1], [2, 3]]
  // CHECK-SAME: tensor<4x?x512x16xi8> into tensor<4x?x8192xi8>
  // CHECK: return %[[COLLAPSE]]
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

// -----

// Negative test: When src_bits < dst_bits (f4 -> i8) and the last reassociation
// group has multiple dimensions, the pattern should NOT apply.
// CHECK-LABEL: @multiDimLastGroup
// CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: tensor<16xf4E2M1FN>
func.func @multiDimLastGroup(%arg0: tensor<16xf4E2M1FN>) -> tensor<2x4xi8> {
  // CHECK: %[[BITCAST:.+]] = iree_tensor_ext.bitcast %[[ARG0]]
  // CHECK-SAME: tensor<16xf4E2M1FN> -> tensor<8xi8>
  // CHECK: %[[EXPAND:.+]] = tensor.expand_shape %[[BITCAST]]
  // CHECK: return %[[EXPAND]]
  %0 = iree_tensor_ext.bitcast %arg0 : tensor<16xf4E2M1FN> -> tensor<8xi8>
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [2, 4]
      : tensor<8xi8> into tensor<2x4xi8>
  return %1 : tensor<2x4xi8>
}

// -----

// Negative test: When the collapse source has dynamic last dim, the pattern
// should NOT apply.
// CHECK-LABEL: @dynamicLastDim
// CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: tensor<4x?xf4E2M1FN>
// CHECK-SAME: %[[D0:[A-Za-z0-9]+]]: index
func.func @dynamicLastDim(%arg0: tensor<4x?xf4E2M1FN>, %d0: index) -> tensor<?xi8> {
  // The collapse source has dynamic last dim, so pattern should NOT fire.
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG0]]
  // CHECK: %[[BITCAST:.+]] = iree_tensor_ext.bitcast %[[COLLAPSE]]
  // CHECK: return %[[BITCAST]]
  %0 = tensor.collapse_shape %arg0 [[0, 1]]
      : tensor<4x?xf4E2M1FN> into tensor<?xf4E2M1FN>
  %c4 = arith.constant 4 : index
  %d1 = arith.muli %d0, %c4 : index
  %c2 = arith.constant 2 : index
  %d2 = arith.divui %d1, %c2 : index
  %1 = iree_tensor_ext.bitcast %0 : tensor<?xf4E2M1FN>{%d1} -> tensor<?xi8>{%d2}
  return %1 : tensor<?xi8>
}

// -----

// Negative test: When a dynamic dimension is merged with other dimensions in
// collapse_shape, the pattern should NOT apply.
// CHECK-LABEL: @mergedDynamicDim
// CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: tensor<4x?x8xf4E2M1FN>
// CHECK-SAME: %[[D0:[A-Za-z0-9]+]]: index
func.func @mergedDynamicDim(%arg0: tensor<4x?x8xf4E2M1FN>, %d0: index) -> tensor<4x?xi8> {
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG0]]
  // CHECK: %[[BITCAST:.+]] = iree_tensor_ext.bitcast %[[COLLAPSE]]
  // CHECK: return %[[BITCAST]]
  %0 = tensor.collapse_shape %arg0 [[0], [1, 2]]
      : tensor<4x?x8xf4E2M1FN> into tensor<4x?xf4E2M1FN>
  %c8 = arith.constant 8 : index
  %d1 = arith.muli %d0, %c8 : index
  %c2 = arith.constant 2 : index
  %d2 = arith.divui %d1, %c2 : index
  %1 = iree_tensor_ext.bitcast %0 : tensor<4x?xf4E2M1FN>{%d1} -> tensor<4x?xi8>{%d2}
  return %1 : tensor<4x?xi8>
}
