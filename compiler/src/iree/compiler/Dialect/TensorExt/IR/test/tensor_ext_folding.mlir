// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @tensorBitCastEmptyProducer
util.func public @tensorBitCastEmptyProducer(%arg0: index, %arg1: index, %arg2:index) -> tensor<?x?xi16> {
  // CHECK-NEXT: %[[EMPTY:.+]] = tensor.empty(%arg1, %arg2) : tensor<?x?xi16>
  %0 = tensor.empty(%arg0) : tensor<?xi32>
  %1 = iree_tensor_ext.bitcast %0 : tensor<?xi32>{%arg0} -> tensor<?x?xi16>{%arg1, %arg2}
  // CHECK-NEXT: return %[[EMPTY]]
  util.return %1 : tensor<?x?xi16>
}
