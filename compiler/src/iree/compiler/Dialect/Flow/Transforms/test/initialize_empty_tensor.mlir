// RUN: iree-opt --iree-flow-initialize-empty-tensors %s | FileCheck %s

func.func @return_zero_init(%arg0 : index, %arg1 : index) -> (tensor<?x?x42xi32>, tensor<?x42x?xf32>) {
  %0 = linalg.init_tensor [%arg0, %arg1, 42] : tensor<?x?x42xi32>
  %1 = linalg.init_tensor [%arg1, 42, %arg0] : tensor<?x42x?xf32>
  return %0, %1 : tensor<?x?x42xi32>, tensor<?x42x?xf32>
}
//      CHECK: func.func @return_zero_init(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-DAG:   %[[ZERO_INT:.+]] = arith.constant 0 : i32
//  CHECK-DAG:   %[[ZERO_FLOAT:.+]] = arith.constant 0.000000e+00 : f32
//  CHECK-DAG:   %[[SPLAT_INT:.+]] = flow.tensor.splat %[[ZERO_INT]] : tensor<?x?x42xi32>{%[[ARG0]], %[[ARG1]]}
//  CHECK-DAG:   %[[SPLAT_FLOAT:.+]] = flow.tensor.splat %[[ZERO_FLOAT]] : tensor<?x42x?xf32>{%[[ARG1]], %[[ARG0]]}
//      CHECK:   return %[[SPLAT_INT]], %[[SPLAT_FLOAT]]
