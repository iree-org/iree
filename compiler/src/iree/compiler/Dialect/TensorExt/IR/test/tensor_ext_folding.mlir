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

// CHECK-LABEL: @barrierFoldDuplicate
util.func public @barrierFoldDuplicate(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]
  %0 = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %arg0 : tensor<4x8xf32> -> tensor<4x8xf32>
  // CHECK-NEXT: %[[BARRIER:.+]] = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %[[ARG0]] : tensor<4x8xf32> -> tensor<4x8xf32>
  %1 = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %0 : tensor<4x8xf32> -> tensor<4x8xf32>
  // CHECK-NEXT: util.return %[[BARRIER]]
  util.return %1 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @barrierNoFoldDifferentDirection
util.func public @barrierNoFoldDifferentDirection(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]
  // CHECK-NEXT: %[[BARRIER0:.+]] = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %[[ARG0]] : tensor<4x8xf32> -> tensor<4x8xf32>
  %0 = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %arg0 : tensor<4x8xf32> -> tensor<4x8xf32>
  // CHECK-NEXT: %[[BARRIER1:.+]] = iree_tensor_ext.compute_barrier<down, "AllowExpand|AllowCollapse"> %[[BARRIER0]] : tensor<4x8xf32> -> tensor<4x8xf32>
  %1 = iree_tensor_ext.compute_barrier<down, "AllowExpand|AllowCollapse"> %0 : tensor<4x8xf32> -> tensor<4x8xf32>
  // CHECK-NEXT: util.return %[[BARRIER1]]
  util.return %1 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @barrierNoFoldDifferentFlags
util.func public @barrierNoFoldDifferentFlags(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]
  // CHECK-NEXT: %[[BARRIER0:.+]] = iree_tensor_ext.compute_barrier<up, AllowExpand> %[[ARG0]] : tensor<4x8xf32> -> tensor<4x8xf32>
  %0 = iree_tensor_ext.compute_barrier<up, "AllowExpand"> %arg0 : tensor<4x8xf32> -> tensor<4x8xf32>
  // CHECK-NEXT: %[[BARRIER1:.+]] = iree_tensor_ext.compute_barrier<up, AllowCollapse> %[[BARRIER0]] : tensor<4x8xf32> -> tensor<4x8xf32>
  %1 = iree_tensor_ext.compute_barrier<up, "AllowCollapse"> %0 : tensor<4x8xf32> -> tensor<4x8xf32>
  // CHECK-NEXT: util.return %[[BARRIER1]]
  util.return %1 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @barrierNoFoldDifferentFlagCombination
util.func public @barrierNoFoldDifferentFlagCombination(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]
  // CHECK-NEXT: %[[BARRIER0:.+]] = iree_tensor_ext.compute_barrier<up, AllowExpand> %[[ARG0]] : tensor<4x8xf32> -> tensor<4x8xf32>
  %0 = iree_tensor_ext.compute_barrier<up, "AllowExpand"> %arg0 : tensor<4x8xf32> -> tensor<4x8xf32>
  // CHECK-NEXT: %[[BARRIER1:.+]] = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %[[BARRIER0]] : tensor<4x8xf32> -> tensor<4x8xf32>
  %1 = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %0 : tensor<4x8xf32> -> tensor<4x8xf32>
  // CHECK-NEXT: util.return %[[BARRIER1]]
  util.return %1 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @barrierFoldSameDirectionAndFlagsDown
util.func public @barrierFoldSameDirectionAndFlagsDown(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]
  %0 = iree_tensor_ext.compute_barrier<down, "AllowExpand"> %arg0 : tensor<4x8xf32> -> tensor<4x8xf32>
  // CHECK-NEXT: %[[BARRIER:.+]] = iree_tensor_ext.compute_barrier<down, AllowExpand> %[[ARG0]] : tensor<4x8xf32> -> tensor<4x8xf32>
  %1 = iree_tensor_ext.compute_barrier<down, "AllowExpand"> %0 : tensor<4x8xf32> -> tensor<4x8xf32>
  // CHECK-NEXT: util.return %[[BARRIER]]
  util.return %1 : tensor<4x8xf32>
}
