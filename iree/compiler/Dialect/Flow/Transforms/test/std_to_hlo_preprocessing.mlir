// RUN: iree-opt -split-input-file -iree-flow-std-to-hlo-preprocessing %s | IreeFileCheck %s

func @select_scalar(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
  %0 = tensor.extract %arg0[] : tensor<i1>
  %1 = select %0, %arg1, %arg2 : tensor<i32>
  return %1 : tensor<i32>
}
// CHECK-LABEL: func @select_scalar
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK:         %[[RES:.*]] = "mhlo.select"(%[[ARG0]], %[[ARG1]], %[[ARG2]])
// CHECK:         return %[[RES]]

// -----

func @select_1d(%arg0: tensor<i1>, %arg1: tensor<4xi32>, %arg2: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tensor.extract %arg0[] : tensor<i1>
  %1 = select %0, %arg1, %arg2 : tensor<4xi32>
  return %1 : tensor<4xi32>
}
// CHECK-LABEL: func @select_1d
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK:         %[[BROADCAST:.*]] = "mhlo.broadcast"(%[[ARG0]]) {
// CHECK-SAME:      broadcast_sizes = dense<4> : tensor<1xi64>
// CHECK-SAME:    } : (tensor<i1>) -> tensor<4xi1>
// CHECK:         %[[RES:.*]] = "mhlo.select"(%[[BROADCAST]], %[[ARG1]], %[[ARG2]])
// CHECK:         return %[[RES]]
