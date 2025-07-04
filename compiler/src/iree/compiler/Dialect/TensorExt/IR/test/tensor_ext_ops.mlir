// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @tensorBitCast
util.func public @tensorBitCast(%arg0: tensor<16xi32>) -> tensor<4x8xi16> {
  // CHECK-NEXT: %0 = iree_tensor_ext.bitcast %arg0 : tensor<16xi32> -> tensor<4x8xi16>
  %0 = iree_tensor_ext.bitcast %arg0 : tensor<16xi32> -> tensor<4x8xi16>
  util.return %0 : tensor<4x8xi16>
}

// -----

// CHECK-LABEL: @tensorBitCastDynamic
util.func public @tensorBitCastDynamic(%arg0: tensor<?x16xi32>, %arg1: index, %arg2: index, %arg3:index) -> tensor<?x?x4x8xi16> {
  // CHECK-NEXT: %0 = iree_tensor_ext.bitcast %arg0 : tensor<?x16xi32>{%arg1} -> tensor<?x?x4x8xi16>{%arg2, %arg3}
  %0 = iree_tensor_ext.bitcast %arg0 : tensor<?x16xi32>{%arg1} -> tensor<?x?x4x8xi16>{%arg2, %arg3}
  util.return %0 : tensor<?x?x4x8xi16>
}
