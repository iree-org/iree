// RUN: iree-opt --split-input-file --iree-stablehlo-legalize-custom-calls \
// RUN:   --cse %s | FileCheck %s

// CHECK-LABEL: @householder
func.func public @householder(%arg0: tensor<4x3xf32>, %arg1: tensor<2xf32>) -> (tensor<4x3xf32>) {
  // CHECK: linalg.generic
  // CHECK: scf.for
  // CHECK:   linalg.generic
  // CHECK:   stablehlo.dot_general
  %0 = stablehlo.custom_call @ProductOfElementaryHouseholderReflectors(%arg0, %arg1) : (tensor<4x3xf32>, tensor<2xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}
