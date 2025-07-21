// RUN: iree-opt --inline --split-input-file %s | FileCheck %s

util.func private @tensor_ext_impl(%arg0: tensor<16xi32>) -> tensor<4x8xi16> {
  %0 = iree_tensor_ext.bitcast %arg0 : tensor<16xi32> -> tensor<4x8xi16>
  util.return %0 : tensor<4x8xi16>
}

util.func public @tensor_ext(%arg0: tensor<16xi32>) -> tensor<4x8xi16> {
  %0 = util.call @tensor_ext_impl(%arg0) : (tensor<16xi32>) -> (tensor<4x8xi16>)
  util.return %0 : tensor<4x8xi16>
}

// CHECK-LABEL: util.func public @tensor_ext(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<16xi32>
//  CHECK-NEXT:   iree_tensor_ext.bitcast %[[ARG0]]
