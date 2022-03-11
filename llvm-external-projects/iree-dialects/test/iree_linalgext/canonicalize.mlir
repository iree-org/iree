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

// CHECK-LABEL: func @canonicalize_insert_slice_indices(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>,
//  CHECK-SAME:     %[[idx:.*]]: index
func @canonicalize_insert_slice_indices(
    %arg0 : tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %idx : index) -> tensor<?x?xf32>
{
  %cst = arith.constant 4.200000e+01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %2 = iree_linalg_ext.in_parallel %idx  -> (tensor<?x?xf32>) {
    ^bb0(%arg3: index):  // no predecessors
      iree_linalg_ext.perform_concurrently {
        // CHECK: iree_linalg_ext.parallel_insert_slice %[[arg0]] into %arg1[%[[idx]], 0] [1, 5] [1, 1]
        iree_linalg_ext.parallel_insert_slice %arg0 into %arg1[%idx, %c0] [%c1, 5] [%c1, %c1] : tensor<?x?xf32> into tensor<?x?xf32>
      }
  }
  return %2 : tensor<?x?xf32>
}
