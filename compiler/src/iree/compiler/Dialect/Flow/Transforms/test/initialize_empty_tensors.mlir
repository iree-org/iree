// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-flow-initialize-empty-tensors{zero-fill=true}))' --split-input-file %s | FileCheck %s --check-prefix=ZERO-CHECK
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-flow-initialize-empty-tensors{zero-fill=false}))' --split-input-file %s | FileCheck %s --check-prefix=EMPTY-CHECK

func.func @return_zero_init(%arg0 : index, %arg1 : index) -> (tensor<?x?x42xi32>, tensor<?x42x?xf32>) {
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?x42xi32>
  %1 = tensor.empty(%arg1, %arg0) : tensor<?x42x?xf32>
  return %0, %1 : tensor<?x?x42xi32>, tensor<?x42x?xf32>
}

//      ZERO-CHECK: func.func @return_zero_init(
// ZERO-CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
// ZERO-CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  ZERO-CHECK-DAG:   %[[ZERO_INT:.+]] = arith.constant 0 : i32
//  ZERO-CHECK-DAG:   %[[ZERO_FLOAT:.+]] = arith.constant 0.000000e+00 : f32
//  ZERO-CHECK-DAG:   %[[SPLAT_INT:.+]] = flow.tensor.splat %[[ZERO_INT]] : tensor<?x?x42xi32>{%[[ARG0]], %[[ARG1]]}
//  ZERO-CHECK-DAG:   %[[SPLAT_FLOAT:.+]] = flow.tensor.splat %[[ZERO_FLOAT]] : tensor<?x42x?xf32>{%[[ARG1]], %[[ARG0]]}
//      ZERO-CHECK:   return %[[SPLAT_INT]], %[[SPLAT_FLOAT]]

//      EMPTY-CHECK: func.func @return_zero_init(
// EMPTY-CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
// EMPTY-CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  EMPTY-CHECK-DAG:   %[[EMPTY_INT:.+]] = flow.tensor.empty : tensor<?x?x42xi32>{%[[ARG0]], %[[ARG1]]}
//  EMPTY-CHECK-DAG:   %[[EMPTY_FLOAT:.+]] = flow.tensor.empty : tensor<?x42x?xf32>{%[[ARG1]], %[[ARG0]]}
//      EMPTY-CHECK:   return %[[EMPTY_INT]], %[[EMPTY_FLOAT]]
