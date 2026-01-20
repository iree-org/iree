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
