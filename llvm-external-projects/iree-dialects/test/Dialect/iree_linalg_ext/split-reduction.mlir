// RUN: iree-dialects-opt --test-topk-split-reduction %s | FileCheck %s

func.func @topk_split_reduction(%input_values: tensor<30xf32>, %input_indices: tensor<30xi32>, %out_values: tensor<3xf32>, %out_indices: tensor<3xi32>) -> (tensor<3xf32>, tensor<3xi32>) {
  %0:2 = iree_linalg_ext.topk
        dimension(0)
        ins(%input_values, %input_indices : tensor<30xf32> , tensor<30xi32>)
        outs(%out_values, %out_indices : tensor<3xf32>, tensor<3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<3xf32>, tensor<3xi32>
  return %0#0, %0#1 : tensor<3xf32>, tensor<3xi32>
}

// CHECK-LABEL: func.func @topk_split_reduction
