// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-stablehlo-legalize-custom-calls),cse)" \
// RUN:   %s | FileCheck %s

// CHECK-LABEL: @householder
func.func public @householder(%arg0: tensor<4x3xf32>, %arg1: tensor<2xf32>) -> (tensor<4x3xf32>) {
  // CHECK: linalg.generic
  // CHECK: scf.for
  // CHECK:   linalg.generic
  // CHECK:   stablehlo.dot_general
  %0 = stablehlo.custom_call @ProductOfElementaryHouseholderReflectors(%arg0, %arg1) : (tensor<4x3xf32>, tensor<2xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// CHECK-LABEL: @attention
func.func public @attention(%query: tensor<1x3x4xf32>, %key: tensor<1x3x4xf32>, %value: tensor<1x3x4xf32>, %scale: tensor<f32>) -> (tensor<1x3x4xf32>) {
  // CHECK: linalg_ext.attention
  %0 = stablehlo.custom_call @iree_attention(%query, %key, %value, %scale) {api_version = 2 : i32, transpose_v = false} : (tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<f32>) -> tensor<1x3x4xf32>
  return %0 : tensor<1x3x4xf32>
}
