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
util.func public @tensorBitCastCastProducer(%arg0: tensor<3x6xi8>, %arg1: index) -> tensor<?x3xi16> {
  // CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: tensor<3x6xi8>
  %0 = tensor.cast %arg0 : tensor<3x6xi8> to tensor<?x6xi8>
  // CHECK-NEXT: %[[BITCAST:.+]] = iree_tensor_ext.bitcast %[[ARG0]] : tensor<3x6xi8> -> tensor<3x3xi16>
  %1 = iree_tensor_ext.bitcast %0 : tensor<?x6xi8>{%arg1} -> tensor<?x3xi16>{%arg1}
  // CHECK-NEXT: %[[CAST:.+]] = tensor.cast %[[BITCAST]] : tensor<3x3xi16> to tensor<?x3xi16>
  util.return %1 : tensor<?x3xi16>
  // CHECK-NEXT: util.return %[[CAST]]
}
