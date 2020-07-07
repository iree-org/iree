// RUN: iree-opt -split-input-file -verify-diagnostics -iree-shape-convert-hlo %s | IreeFileCheck %s

// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>, %arg1: tensor<2xindex>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[SHAPE:.+]] = "shapex.from_extent_tensor"(%arg1) : (tensor<2xindex>) -> !shapex.ranked_shape<[?,?]>
  // CHECK-DAG: %[[BROADCASTED:.+]] = "shapex.ranked_broadcast_in_dim"(%arg0, %0) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: return %[[BROADCASTED]]
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {broadcast_dimensions = dense<[1]> : tensor<1xi64>}: (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
