// RUN: iree-opt --split-input-file --iree-auto-input-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @simple_add
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x2xi32>, %[[ARG1:.*]]: tensor<2x2xi32>) -> tensor<2x2xi32>
func.func @simple_add(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: return %[[RESULT]] : tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}
