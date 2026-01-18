// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @tensorBitCastEmptyProducer
util.func public @tensorBitCastEmptyProducer(%arg0: index, %arg1: index, %arg2:index) -> tensor<?x?xi16> {
  // CHECK-NEXT: %[[EMPTY:.+]] = tensor.empty(%arg1, %arg2) : tensor<?x?xi16>
  %0 = tensor.empty(%arg0) : tensor<?xi32>
  %1 = iree_tensor_ext.bitcast %0 : tensor<?xi32>{%arg0} -> tensor<?x?xi16>{%arg1, %arg2}
  // CHECK-NEXT: return %[[EMPTY]]
  util.return %1 : tensor<?x?xi16>
}

// -----

// CHECK-LABEL: @tensorBitCastCastProducer
util.func public @tensorBitCastCastProducer(%arg0: tensor<10x3x6xi8>, %arg1: index, %arg2: index) -> tensor<?x?x3xi16> {
  // CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: tensor<10x3x6xi8>
  %0 = tensor.cast %arg0 : tensor<10x3x6xi8> to tensor<?x?x6xi8>
  // CHECK-NEXT: %[[BITCAST:.+]] = iree_tensor_ext.bitcast %[[ARG0]] : tensor<10x3x6xi8> -> tensor<10x3x3xi16>
  %1 = iree_tensor_ext.bitcast %0 : tensor<?x?x6xi8>{%arg1, %arg2} -> tensor<?x?x3xi16>{%arg1, %arg2}
  // CHECK-NEXT: %[[CAST:.+]] = tensor.cast %[[BITCAST]] : tensor<10x3x3xi16> to tensor<?x?x3xi16>
  util.return %1 : tensor<?x?x3xi16>
  // CHECK-NEXT: util.return %[[CAST]]
}

// -----

// CHECK-LABEL: @bitcastOfCastWithConstAndDynDims
util.func public @bitcastOfCastWithConstAndDynDims(%arg0: tensor<4x?x4096xi8>, %d0: index) -> tensor<?x?x?xf4E2M1FN> {
  // CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: tensor<4x?x4096xi8>
  // CHECK-SAME: %[[D0:[A-Za-z0-9]+]]: index
  %c4 = arith.constant 4 : index
  %c4096 = arith.constant 4096 : index
  %c8192 = arith.constant 8192 : index
  %0 = tensor.cast %arg0 : tensor<4x?x4096xi8> to tensor<?x?x?xi8>
  // CHECK: %[[BITCAST:.+]] = iree_tensor_ext.bitcast %[[ARG0]] : tensor<4x?x4096xi8>{%[[D0]]} -> tensor<4x?x8192xf4E2M1FN>{%[[D0]]}
  %1 = iree_tensor_ext.bitcast %0 : tensor<?x?x?xi8>{%c4, %d0, %c4096} -> tensor<?x?x?xf4E2M1FN>{%c4, %d0, %c8192}
  // CHECK: %[[CAST:.+]] = tensor.cast %[[BITCAST]] : tensor<4x?x8192xf4E2M1FN> to tensor<?x?x?xf4E2M1FN>
  util.return %1 : tensor<?x?x?xf4E2M1FN>
  // CHECK: util.return %[[CAST]]
}

// -----

// CHECK-LABEL: @bitcastOfCastAllDynWithConstants
util.func public @bitcastOfCastAllDynWithConstants(%arg0: tensor<4x128x4096xi8>) -> tensor<?x?x?xi4> {
  // CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: tensor<4x128x4096xi8>
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c8192 = arith.constant 8192 : index
  %0 = tensor.cast %arg0 : tensor<4x128x4096xi8> to tensor<4x?x4096xi8>
  // CHECK: %[[BITCAST:.+]] = iree_tensor_ext.bitcast %[[ARG0]] : tensor<4x128x4096xi8> -> tensor<4x128x8192xi4>
  %1 = iree_tensor_ext.bitcast %0 : tensor<4x?x4096xi8>{%c128} -> tensor<?x?x?xi4>{%c4, %c128, %c8192}
  // CHECK: %[[CAST:.+]] = tensor.cast %[[BITCAST]] : tensor<4x128x8192xi4> to tensor<?x?x?xi4>
  util.return %1 : tensor<?x?x?xi4>
  // CHECK: util.return %[[CAST]]
}

// -----

// CHECK-LABEL: @bitcastOfCastRankMismatch
util.func public @bitcastOfCastRankMismatch(%arg0: tensor<128x?xf32>, %d0: index, %d1: index) -> tensor<?xi32> {
  %c128 = arith.constant 128 : index
  %0 = tensor.cast %arg0 : tensor<128x?xf32> to tensor<?x?xf32>
  // CHECK: tensor.cast
  // CHECK: iree_tensor_ext.bitcast
  %1 = iree_tensor_ext.bitcast %0 : tensor<?x?xf32>{%c128, %d0} -> tensor<?xi32>{%d1}
  util.return %1 : tensor<?xi32>
  // CHECK: util.return
}

// -----

// CHECK-LABEL: @barrierStartFoldDuplicate
util.func public @barrierStartFoldDuplicate(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]
  %0 = iree_tensor_ext.compute_barrier.start %arg0 : tensor<4x8xf32> -> tensor<4x8xf32>
  // CHECK-NEXT: %[[BARRIER:.+]] = iree_tensor_ext.compute_barrier.start %[[ARG0]] : tensor<4x8xf32> -> tensor<4x8xf32>
  %1 = iree_tensor_ext.compute_barrier.start %0 : tensor<4x8xf32> -> tensor<4x8xf32>
  // CHECK-NEXT: util.return %[[BARRIER]]
  util.return %1 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @barrierEndFoldDuplicate
util.func public @barrierEndFoldDuplicate(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]
  %0 = iree_tensor_ext.compute_barrier.end %arg0 : tensor<4x8xf32> -> tensor<4x8xf32>
  // CHECK-NEXT: %[[BARRIER:.+]] = iree_tensor_ext.compute_barrier.end %[[ARG0]] : tensor<4x8xf32> -> tensor<4x8xf32>
  %1 = iree_tensor_ext.compute_barrier.end %0 : tensor<4x8xf32> -> tensor<4x8xf32>
  // CHECK-NEXT: util.return %[[BARRIER]]
  util.return %1 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @propagateBitCastThroughExpandShape
// CHECK-SAME:  %[[ARG0:[A-Za-z0-9]+]]: tensor<?x32xf4E2M1FN>
// CHECK-SAME:  %[[D0:[A-Za-z0-9]+]]: index
// CHECK-SAME:  %[[D1:[A-Za-z0-9]+]]: index
util.func public @propagateBitCastThroughExpandShape(%arg0: tensor<?x32xf4E2M1FN>, %d0: index, %d1: index) -> tensor<4x?x512x16xi8> {
  // CHECK: %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1, 2], [3]]
  // CHECK-SAME: output_shape [4, %[[D1]], 512, 32]
  // CHECK-SAME: tensor<?x32xf4E2M1FN> into tensor<4x?x512x32xf4E2M1FN>
  %0 = iree_tensor_ext.bitcast %arg0 : tensor<?x32xf4E2M1FN>{%d0} -> tensor<?x16xi8>{%d0}
  %1 = tensor.expand_shape %0 [[0, 1, 2], [3]] output_shape [4, %d1, 512, 16]
      : tensor<?x16xi8> into tensor<4x?x512x16xi8>
  // CHECK: %[[BITCAST:.+]] = iree_tensor_ext.bitcast %[[EXPAND]]
  // CHECK-SAME: tensor<4x?x512x32xf4E2M1FN>
  // CHECK-SAME: tensor<4x?x512x16xi8>
  // CHECK: util.return %[[BITCAST]]
  util.return %1 : tensor<4x?x512x16xi8>
}

// -----

// CHECK-LABEL: @propagate1dBitCastThroughExpandShape
// CHECK-SAME:  %[[ARG0:[A-Za-z0-9]+]]: tensor<?xf8E8M0FNU>
// CHECK-SAME:  %[[D0:[A-Za-z0-9]+]]: index
// CHECK-SAME:  %[[D1:[A-Za-z0-9]+]]: index
util.func public @propagate1dBitCastThroughExpandShape(%arg0: tensor<?xf8E8M0FNU>, %d0: index, %d1: index) -> tensor<4x?x512xi8> {
  // The bitcast should propagate through expand_shape.
  // CHECK: %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1, 2]]
  // CHECK-SAME: output_shape [4, %[[D1]], 512]
  // CHECK-SAME: tensor<?xf8E8M0FNU> into tensor<4x?x512xf8E8M0FNU>
  // CHECK: %[[BITCAST:.+]] = iree_tensor_ext.bitcast %[[EXPAND]]
  // CHECK-SAME: tensor<4x?x512xf8E8M0FNU>
  // CHECK-SAME: tensor<4x?x512xi8>
  // CHECK: util.return %[[BITCAST]]
  %0 = iree_tensor_ext.bitcast %arg0 : tensor<?xf8E8M0FNU>{%d0} -> tensor<?xi8>{%d0}
  %1 = tensor.expand_shape %0 [[0, 1, 2]] output_shape [4, %d1, 512]
      : tensor<?xi8> into tensor<4x?x512xi8>
  util.return %1 : tensor<4x?x512xi8>
}

// -----

// CHECK-LABEL: @propagateBitCastThroughCollapseShape
// CHECK-SAME:  %[[ARG0:[A-Za-z0-9]+]]: tensor<4x?x512x32xf4E2M1FN>
// CHECK-SAME:  %[[D0:[A-Za-z0-9]+]]: index
util.func public @propagateBitCastThroughCollapseShape(%arg0: tensor<4x?x512x32xf4E2M1FN>, %d0: index) -> tensor<4x?x8192xi8> {
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0], [1], [2, 3]]
  // CHECK-SAME: tensor<4x?x512x32xf4E2M1FN> into tensor<4x?x16384xf4E2M1FN>
  %0 = iree_tensor_ext.bitcast %arg0 : tensor<4x?x512x32xf4E2M1FN>{%d0} -> tensor<4x?x512x16xi8>{%d0}
  %1 = tensor.collapse_shape %0 [[0], [1], [2, 3]]
      : tensor<4x?x512x16xi8> into tensor<4x?x8192xi8>
  // CHECK: %[[BITCAST:.+]] = iree_tensor_ext.bitcast %[[COLLAPSE]]
  // CHECK-SAME: tensor<4x?x16384xf4E2M1FN>
  // CHECK-SAME: tensor<4x?x8192xi8>
  // CHECK: util.return %[[BITCAST]]
  util.return %1 : tensor<4x?x8192xi8>
}

// -----

// CHECK-LABEL: @foldConsecutiveBitCasts
// CHECK-SAME:  %[[ARG0:[A-Za-z0-9]+]]: tensor<4x?x16384xf4E2M1FN>
// CHECK-SAME:  %[[D0:[A-Za-z0-9]+]]: index
util.func public @foldConsecutiveBitCasts(%arg0: tensor<4x?x16384xf4E2M1FN>, %d0: index) -> tensor<4x?x16384xf4E2M1FN> {
  // CHECK-NOT: iree_tensor_ext.bitcast
  // CHECK: util.return %[[ARG0]]
  %0 = iree_tensor_ext.bitcast %arg0 : tensor<4x?x16384xf4E2M1FN>{%d0} -> tensor<4x?x8192xi8>{%d0}
  %1 = iree_tensor_ext.bitcast %0 : tensor<4x?x8192xi8>{%d0} -> tensor<4x?x16384xf4E2M1FN>{%d0}
  util.return %1 : tensor<4x?x16384xf4E2M1FN>
}

// -----

// CHECK-LABEL: @bitcastExpandCollapseChain
// CHECK-SAME:  %[[ARG0:[A-Za-z0-9]+]]: tensor<?x32xf4E2M1FN>
// CHECK-SAME:  %[[D1:[A-Za-z0-9]+]]: index
util.func public @bitcastExpandCollapseChain(%arg0: tensor<?x32xf4E2M1FN>, %d1: index) -> tensor<4x?x16384xf4E2M1FN> {
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
  // CHECK: util.return %[[COLLAPSE]]
  %0 = iree_tensor_ext.bitcast %arg0 : tensor<?x32xf4E2M1FN>{%d0} -> tensor<?x16xi8>{%d0}
  %1 = tensor.expand_shape %0 [[0, 1, 2], [3]] output_shape [4, %d1, 512, 16]
      : tensor<?x16xi8> into tensor<4x?x512x16xi8>
  %2 = tensor.collapse_shape %1 [[0], [1], [2, 3]]
      : tensor<4x?x512x16xi8> into tensor<4x?x8192xi8>
  %3 = iree_tensor_ext.bitcast %2 : tensor<4x?x8192xi8>{%d1} -> tensor<4x?x16384xf4E2M1FN>{%d1}
  util.return %3 : tensor<4x?x16384xf4E2M1FN>
}
