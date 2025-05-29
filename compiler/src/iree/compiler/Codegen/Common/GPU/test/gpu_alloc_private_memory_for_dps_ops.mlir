// RUN: iree-opt %s --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-alloc-private-memory-for-dps-ops))" | FileCheck %s

// CHECK-LABEL: func.func @unused_result_copied
// CHECK-DAG: %[[SRC:.*]] = bufferization.alloc_tensor() copy(%{{.*}}) {memory_space = #gpu.address_space<private>} : tensor<1x10xf32>
// CHECK-DAG: iree_linalg_ext.sort{{.*}} dimension(1) outs(%[[SRC]], %{{.*}} : tensor<1x10xf32>, tensor<1x10xi64>)
func.func @unused_result_copied(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x10xf32>>, %arg1: tensor<1x10xi64>) -> tensor<1x10xi64> {
  %2 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 10], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x10xf32>> -> tensor<1x10xf32>
  %3:2 = iree_linalg_ext.sort dimension(1) outs(%2, %arg1 :tensor<1x10xf32>, tensor<1x10xi64>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: i64, %arg7: i64):
    %16 = arith.cmpf oge, %arg4, %arg5 : f32
    iree_linalg_ext.yield %16 : i1
  } -> tensor<1x10xf32>, tensor<1x10xi64>
  return %3#1 : tensor<1x10xi64>
}

// -----

// CHECK-LABEL: func.func @big_result_not_copied
// CHECK-NOT: bufferization.alloc_tensor()
func.func @big_result_not_copied(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x33xf32>>, %arg1: tensor<1x33xi64>) -> tensor<1x33xi64> {
  %cst = arith.constant dense<1.000000e+00> : tensor<1x33xf32>
  %2 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 33], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x33xf32>> -> tensor<1x33xf32>
  %3:2 = iree_linalg_ext.sort dimension(1) outs(%2, %arg1 :tensor<1x33xf32>, tensor<1x33xi64>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: i64, %arg7: i64):
    %16 = arith.cmpf oge, %arg4, %arg5 : f32
    iree_linalg_ext.yield %16 : i1
  } -> tensor<1x33xf32>, tensor<1x33xi64>
  return %3#1 : tensor<1x33xi64>
}

// -----

// CHECK-LABEL: func.func @used_result_not_copied
// CHECK-NOT: bufferization.alloc_tensor()
func.func @used_result_not_copied(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x10xf32>>, %arg1: tensor<1x10xi64>) -> (tensor<1x10xf32>, tensor<1x10xi64>) {
  %2 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 10], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x10xf32>> -> tensor<1x10xf32>
  %3:2 = iree_linalg_ext.sort dimension(1) outs(%2, %arg1 :tensor<1x10xf32>, tensor<1x10xi64>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: i64, %arg7: i64):
    %16 = arith.cmpf oge, %arg4, %arg5 : f32
    iree_linalg_ext.yield %16 : i1
  } -> tensor<1x10xf32>, tensor<1x10xi64>
  return %3#0, %3#1 : tensor<1x10xf32>, tensor<1x10xi64>
}

// -----

// CHECK-LABEL: func @memref_semantics(
//  CHECK-SAME:   %[[DEST:.+]]: memref<?x?xf32>
//       CHECK:   linalg.fill {{.*}} outs(%[[DEST]]
func.func @memref_semantics(%dest: memref<?x?xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill ins(%cst : f32) outs(%dest : memref<?x?xf32>)
  return
}
