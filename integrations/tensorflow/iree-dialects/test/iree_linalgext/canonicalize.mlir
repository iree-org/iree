// RUN: iree-dialects-opt -canonicalize -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @tensor.cast(
func @tensor.cast(%arg0: tensor<3x5xi32>) -> tensor<3x5xi32> {
  %init = linalg.init_tensor [3, 5] : tensor<3x5xi32>

  %casted_arg0 = tensor.cast %arg0 : tensor<3x5xi32> to tensor<?x?xi32>
  %casted_init = tensor.cast %init : tensor<3x5xi32> to tensor<?x?xi32>

// CHECK:      iree_linalg_ext.reverse
// CHECK-SAME:   ins(%{{[a-zA-Z0-9]*}} : tensor<3x5xi32>)
// CHECK-SAME:  outs(%{{[a-zA-Z0-9]*}} : tensor<3x5xi32>)
  %0 = iree_linalg_ext.reverse
         dimensions(dense<0> : tensor<1xi64>)
         ins(%casted_arg0 : tensor<?x?xi32>)
         outs(%casted_init : tensor<?x?xi32>) : tensor<?x?xi32>

  %1 = tensor.cast %0 : tensor<?x?xi32> to tensor<3x5xi32>

  return %1: tensor<3x5xi32>
}
